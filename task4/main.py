import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import optimizers
from tensorflow.keras import applications
import keras.backend as K

### IMAGE PARAMETERS ###
min_width = 64
min_height = 64


def preprocess_image(filename):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (min_width, min_height))
    return image


def preprocess_data(x, labels):
    anchor, pos_neg, neg_pos = x
    return tf.stack([preprocess_image(anchor),
                     preprocess_image(pos_neg),
                     preprocess_image(neg_pos)], axis=0), labels


def preprocess_data_test(x_test):
    anchor, pos_neg, neg_pos = x_test
    return tf.stack([preprocess_image(anchor),
                     preprocess_image(pos_neg),
                     preprocess_image(neg_pos)], axis=0)


# from zipfile import ZipFile
# zip_ref = ZipFile('food.zip', 'r')
# extracting all the files
# print('Extracting all the files now...')
# zip_ref.extractall()
# zip_ref.close()
# print('Done!')

train_data = np.loadtxt('train_triplets.txt', dtype=str)
anchor_images = ['food/' + image[0] + '.jpg' for image in train_data]
positive_images = ['food/' + image[1] + '.jpg' for image in train_data]
negative_images = ['food/' + image[2] + '.jpg' for image in train_data]

data_length = len(anchor_images)
labels1 = [1.0] * data_length
labels2 = [0.0] * data_length
labels = labels1 + labels2

x = tf.data.Dataset.from_tensor_slices((anchor_images + anchor_images, positive_images + negative_images,
                                        negative_images + positive_images))
labels = tf.data.Dataset.from_tensor_slices(labels)

dataset = tf.data.Dataset.zip((x, labels))
dataset = dataset.shuffle(buffer_size=len(dataset), reshuffle_each_iteration=True)
dataset = dataset.map(preprocess_data)

split = 0.3
train_dataset = dataset.take(int(split * len(dataset)))
number_of_samples = len(train_dataset)
val_dataset = dataset.skip(int(split * len(dataset)))

epochs = 5
batch_size = 32
train_dataset = train_dataset.batch(batch_size)
val_dataset = val_dataset.batch(batch_size)

# import matplotlib.pyplot as plt
# for images, labels in train_dataset.take(1):# only take first element of dataset
#     numpy_images = images.numpy()
#     numpy_labels = labels.numpy()
#     # print(numpy_labels)
#
# fig = plt.figure()
# ax = fig.add_subplot(3, 3, 1)
# imgplot = plt.imshow(numpy_images[0, 0, ...])
# ax.set_title(numpy_labels[0])
# ax = fig.add_subplot(3, 3, 2)
# imgplot = plt.imshow(numpy_images[0, 1, ...])
# ax.set_title('Image A')
# ax = fig.add_subplot(3, 3, 3)
# imgplot = plt.imshow(numpy_images[0, 2, ...])
# ax.set_title('Image B')
#
# ax = fig.add_subplot(3, 3, 4)
# imgplot = plt.imshow(numpy_images[1, 0, ...])
# ax.set_title(numpy_labels[1])
# ax = fig.add_subplot(3, 3, 5)
# imgplot = plt.imshow(numpy_images[1, 1, ...])
# ax.set_title('Image A')
# ax = fig.add_subplot(3, 3, 6)
# imgplot = plt.imshow(numpy_images[1, 2, ...])
# ax.set_title('Image B')
#
# ax = fig.add_subplot(3, 3, 7)
# imgplot = plt.imshow(numpy_images[2, 0, ...])
# ax.set_title(numpy_labels[2])
# ax = fig.add_subplot(3, 3, 8)
# imgplot = plt.imshow(numpy_images[2, 1, ...])
# ax.set_title('Image A')
# ax = fig.add_subplot(3, 3, 9)
# imgplot = plt.imshow(numpy_images[2, 2, ...])
# ax.set_title('Image B')
# plt.show()

## MODEL
base_model = applications.mobilenet_v2.MobileNetV2(input_shape=(min_width, min_height, 3),
                                                   include_top=False, weights='imagenet')
for layer in base_model.layers:
    layer.trainable = False

layer = layers.GlobalAveragePooling2D()(base_model.output)
embedding_network = Model(base_model.input, layer)

input = layers.Input((3, min_width, min_height, 3))
input_1, input_2, input_3 = input[:, 0, ...], input[:, 1, ...], input[:, 2, ...]

tower_1 = embedding_network(input_1)
tower_2 = embedding_network(input_2)
tower_3 = embedding_network(input_3)

merge_layer = layers.Concatenate()([tower_1, tower_2, tower_3])
x = layers.Dense(32, activation='relu')(merge_layer)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(16, activation='relu')(x)
x = layers.BatchNormalization()(x)
output_layer = layers.Dense(1, activation="sigmoid")(x)
siamese = Model(inputs=input, outputs=output_layer)

siamese.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')
siamese.summary()
# es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
siamese.fit(train_dataset, epochs=epochs, validation_data=val_dataset, validation_steps=10)

#######################################################
##################### PREDICTION  #####################
#######################################################
test_batch_size = 64
test_data = np.loadtxt('test_triplets.txt', dtype=str)
anchor_images_test = ['food/' + image[0] + '.jpg' for image in test_data]
first_image_test = ['food/' + image[1] + '.jpg' for image in test_data]
second_image_test = ['food/' + image[2] + '.jpg' for image in test_data]

x_test = tf.data.Dataset.from_tensor_slices((anchor_images_test, first_image_test, second_image_test))
dataset_test = tf.data.Dataset.zip((x_test,))
dataset_test = dataset_test.map(preprocess_data_test)

dataset_test = dataset_test.batch(test_batch_size)

predictions = siamese.predict(dataset_test, verbose=1)
predictions = np.round(predictions)
np.savetxt('predictions.txt', predictions, fmt='%i')


