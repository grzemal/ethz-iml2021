import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet
import efficientnet.keras as efn

target_shape = (32, 32)


def preprocess_image(filename):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """

    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    return image


def preprocess_triplets(anchor, positive, negative):
    """
    Given the filenames corresponding to the three images, load and
    preprocess them.
    """

    return (
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative),
    )


def visualize(anchor, positive, negative):
    """Visualize a few triplets from the supplied batches."""

    def show(ax, image):
        ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig = plt.figure(figsize=(9, 9))

    axs = fig.subplots(3, 3)
    for i in range(3):
        show(axs[i, 0], anchor[i])
        show(axs[i, 1], positive[i])
        show(axs[i, 2], negative[i])
    plt.show()


# get all paths out of .txt file

train_data = np.loadtxt('train_triplets.txt', dtype=str)
anchor_images = ['food/' + image[0] + '.jpg' for image in train_data]
positive_images = ['food/' + image[1] + '.jpg' for image in train_data]
negative_images = ['food/' + image[2] + '.jpg' for image in train_data]

image_count = len(anchor_images)

anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
positive_dataset = tf.data.Dataset.from_tensor_slices(positive_images)
negative_dataset = tf.data.Dataset.from_tensor_slices(negative_images)

dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
dataset = dataset.shuffle(buffer_size=1024)
dataset = dataset.map(preprocess_triplets)

# Let's now split our dataset in train and validation.
split = 0.8
train_dataset = dataset.take(round(image_count * split))
val_dataset = dataset.skip(round(image_count * split))

train_dataset = train_dataset.batch(32, drop_remainder=False)
train_dataset = train_dataset.prefetch(8)

val_dataset = val_dataset.batch(32, drop_remainder=False)
val_dataset = val_dataset.prefetch(8)

visualize(*list(train_dataset.take(1).as_numpy_iterator())[0])

### EMBEDDING GENERATOR MODEL ###

base_cnn = efn.EfficientNetB0(
    weights="imagenet", input_shape=target_shape + (3,), include_top=False
)

flatten = layers.Flatten()(base_cnn.output)
dense1 = layers.Dense(512, activation="relu")(flatten)
dense1 = layers.BatchNormalization()(dense1)
dense2 = layers.Dense(256, activation="relu")(dense1)
dense2 = layers.BatchNormalization()(dense2)
output = layers.Dense(256)(dense2)

embedding = Model(base_cnn.input, output, name="Embedding")

trainable = False
for layer in base_cnn.layers:
    if layer.name == "conv5_block1_out":
        trainable = True
    layer.trainable = trainable

### Setting up Siamese Network Model ###

class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
positive_input = layers.Input(name="positive", shape=target_shape + (3,))
negative_input = layers.Input(name="negative", shape=target_shape + (3,))

distances = DistanceLayer()(
    embedding(efn.preprocess_input(anchor_input)),
    embedding(efn.preprocess_input(positive_input)),
    embedding(efn.preprocess_input(negative_input)),
)

siamese_network = Model(
    inputs=[anchor_input, positive_input, negative_input], outputs=distances
)

class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]


siamese_model = SiameseModel(siamese_network)
siamese_model.compile(optimizer=optimizers.Adam(0.0001))
siamese_model.fit(train_dataset, epochs=1, validation_data=None, use_multiprocessing=True, workers=4)


#######################################################
##################### PREDICTION  #####################
#######################################################
test_batch_size = 64
test_data = np.loadtxt('test_triplets.txt', dtype=str)

anchor_images_test = ['food/' + image[0] + '.jpg' for image in test_data]
left_images = ['food/' + image[1] + '.jpg' for image in test_data]
right_images = ['food/' + image[2] + '.jpg' for image in test_data]

image_count = len(anchor_images_test)

anchor_dataset_test = tf.data.Dataset.from_tensor_slices(anchor_images_test)
left_dataset = tf.data.Dataset.from_tensor_slices(left_images)
right_dataset = tf.data.Dataset.from_tensor_slices(right_images)

dataset_test = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
dataset_test = dataset_test.shuffle(buffer_size=1024)
dataset_test = dataset_test.map(preprocess_triplets)

dataset_test = dataset_test.batch(test_batch_size)

cosine_similarity = metrics.CosineSimilarity()
predictions = []

#predictions = siamese_model.predict(train_dataset, verbose=1)

print(predictions)

for i in range(image_count):

    sample = next(iter(dataset_test))
    #visualize(*sample)

    anchor, left, right = sample
    anchor_embedding, left_embedding, right_embedding = (
        embedding(efn.preprocess_input(anchor)),
        embedding(efn.preprocess_input(left)),
        embedding(efn.preprocess_input(right)),
    )
    left_similarity = cosine_similarity(anchor_embedding, left_embedding)
    right_similarity = cosine_similarity(anchor_embedding, right_embedding)

    if left_similarity > right_similarity:
        predictions.append(1)
    else:
        predictions.append(0)

np.savetxt('predictions.txt', predictions, fmt='%i')