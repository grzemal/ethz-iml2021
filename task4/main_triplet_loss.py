import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

img_width = 224
img_height = 224


def preprocess_image_train(filename):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (img_width, img_height))
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image


def preprocess_image_test(filename):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (img_width, img_height))
    return image


def preprocess_data_train(x):
    anchor, positive, negative = x
    return tf.stack([preprocess_image_train(anchor),
                     preprocess_image_train(positive),
                     preprocess_image_train(negative)], axis=0), 1


def preprocess_data_test(x):
    anchor, positive, negative = x
    return tf.stack([preprocess_image_test(anchor),
                     preprocess_image_test(positive),
                     preprocess_image_test(negative)], axis=0)


def create_model():
    inputs = tf.keras.Input(shape=(3, img_height, img_width, 3))
    encoder = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(img_height, img_width, 3))
    encoder.trainable = False
    decoder = tf.keras.Sequential([tf.keras.layers.GlobalAveragePooling2D(),
                                   tf.keras.layers.Dense(128),
                                   tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))
                                   ])
    anchor, positive, negative = inputs[:, 0, ...], inputs[:, 1, ...], inputs[:, 2, ...]
    anchor = decoder(encoder(anchor))
    positive = decoder(encoder(positive))
    negative = decoder(encoder(negative))
    embedding = tf.stack([anchor, positive, negative], axis=-1)
    siamese_network = tf.keras.Model(inputs=inputs, outputs=embedding)
    siamese_network.summary()
    return siamese_network


def create_inference_model(model):
    distance_positive, distance_negative = compute_distances(model.output)
    predictions = tf.cast(tf.greater_equal(distance_negative, distance_positive), tf.int8)
    return tf.keras.Model(inputs=model.inputs, outputs=predictions)


def compute_distances(embeddings):
    anchor, positive, negative = embeddings[..., 0], embeddings[..., 1], embeddings[..., 2]
    distance_positive = tf.reduce_sum(tf.square(anchor - positive), 1)
    distance_negative = tf.reduce_sum(tf.square(anchor - negative), 1)
    return distance_positive, distance_negative


def make_train_val():
    triplets = np.loadtxt('train_triplets.txt', dtype=str)
    train_samples, val_samples = train_test_split(triplets, test_size=0.2)
    np.savetxt('val_samples.txt', val_samples, fmt='%s %s %s')
    np.savetxt('train_samples.txt', train_samples, fmt='%s %s %s')
    return len(train_samples)


def create_dataset(dataset_filename, training=True):
    data = np.loadtxt(dataset_filename, dtype=str)
    anchor_images = ['food/' + image[0] + '.jpg' for image in data]
    positive_images = ['food/' + image[1] + '.jpg' for image in data]
    negative_images = ['food/' + image[2] + '.jpg' for image in data]
    x = tf.data.Dataset.from_tensor_slices((anchor_images, positive_images, negative_images))
    dataset = tf.data.Dataset.zip((x,))
    if training:
        dataset = dataset.map(preprocess_data_train)
    else:
        dataset = dataset.map(preprocess_data_test)
    return dataset


def triplet_loss(_, embeddings):
    distance_positive, distance_negative = compute_distances(embeddings)
    return tf.reduce_mean(tf.math.softplus(distance_positive - distance_negative))


def accuracy(_, embeddings):
    distance_positive, distance_negative = compute_distances(embeddings)
    return tf.reduce_mean(tf.cast(tf.greater_equal(distance_negative, distance_positive), tf.float32))


epochs = 1
train_batch_size = 32
inference_batch_size = 256
num_train_samples = make_train_val()

train_dataset = create_dataset('train_samples.txt')
val_dataset = create_dataset('val_samples.txt')

train_dataset = train_dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=True)
val_dataset = val_dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=True)

train_dataset = train_dataset.batch(train_batch_size)
val_dataset = val_dataset.batch(train_batch_size)

model = create_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=triplet_loss, metrics=[accuracy])
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1)
model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, validation_steps=20)

test_dataset = create_dataset('test_triplets.txt', training=False)
test_dataset = test_dataset.batch(inference_batch_size).prefetch(2)

inference_model = create_inference_model(model)
predictions = inference_model.predict(test_dataset, verbose=1)
np.savetxt('predictions.txt', predictions, fmt='%i')
