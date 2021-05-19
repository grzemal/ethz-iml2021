import numpy as np
import matplotlib.pyplot as plt
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

### IMAGE PARAMETERS ###
min_width = 354
min_height = 242

mean = [0.607784796814423, 0.5161963794783105, 0.41263983204695315]
std = [0.2677660693068199, 0.27784572521180145, 0.2986767732471361]

### /IMAGE PARAMETERS ###

from image_utils import cal_dir_stat, get_min_size

base_dir = 'food'
train_dir = 'train_dir'
val_dir = 'val_dir'


# try:
#    os.mkdir(train_dir)
#    os.mkdir(val_dir)
# except OSError as error:
#    print(error)


@tf.function
def preprocess_image(filename):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """
    print(filename)
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (min_width, min_height))
    return image


def preprocess_data(x, labels):
    anchor, pos_neg, neg_pos = x
    return (
        tf.concat((preprocess_image(anchor), preprocess_image(pos_neg), preprocess_image(neg_pos)),
                  axis=0),
        labels)

def preprocess_data_test(x_test):
    anchor, pos_neg, neg_pos = x_test
    return(
        tf.concat((preprocess_image(anchor), preprocess_image(pos_neg), preprocess_image(neg_pos)),
                  axis=0))

# def get_concatenated_images(filenames):
#     concatenated_images = []
#     for i,triplets in enumerate(filenames):
#         list_of_images = []
#         for path in triplets:
#             image_string = tf.io.read_file(path)
#             image = tf.image.decode_jpeg(image_string, channels=3)
#             image = tf.image.convert_image_dtype(image, tf.float32)
#             image = tf.image.resize(image, target_shape)
#             list_of_images.append(image)
#         concatenated_images.append(tf.concat([list_of_images[0], list_of_images[1], list_of_images[2]], 0))
#         print(i)
#     return tf.convert_to_tensor(concatenated_images)


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
labels1 = [1] * data_length
labels2 = [0] * data_length
labels = labels1 + labels2

x = tf.data.Dataset.from_tensor_slices((anchor_images + anchor_images, positive_images + negative_images,
                                        negative_images + positive_images))
labels = tf.data.Dataset.from_tensor_slices(labels)

dataset = tf.data.Dataset.zip((x, labels))
dataset = dataset.shuffle(buffer_size=1024)
dataset = dataset.map(preprocess_data)

split = 0.8
train_dataset = dataset.take(round(data_length * split))
val_dataset = dataset.skip(round(data_length * split))

train_dataset = train_dataset.batch(32, drop_remainder=False)
train_dataset = train_dataset.prefetch(8)

val_dataset = val_dataset.batch(32, drop_remainder=False)
val_dataset = val_dataset.prefetch(8)

## MODEL
from tensorflow.keras.applications.inception_v3 import InceptionV3

base_model = InceptionV3(input_shape=(min_width * 3, min_height, 3), include_top=False, weights='imagenet')
for layer in base_model.layers:
    layer.trainable = False

flatten = layers.Flatten()(base_model.output)
dense1 = layers.Dense(512, activation="relu")(flatten)
dense1 = layers.BatchNormalization()(dense1)
dropout1 = layers.Dropout(0.3)(dense1)
dense2 = layers.Dense(256, activation="relu")(dropout1)
dense2 = layers.BatchNormalization()(dense2)
output = layers.Dense(1, activation="sigmoid")(dense2)

model = Model(base_model.input, output)
model.summary()
loss_fn = tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss=loss_fn)
model.fit(train_dataset, epochs=2, steps_per_epoch=4)

## Prediction
test_data = np.loadtxt('test_triplets.txt', dtype=str)
anchor_images_test = ['food/' + image[0] + '.jpg' for image in test_data]
first_image_test = ['food/' + image[1] + '.jpg' for image in test_data]
second_image_test = ['food/' + image[2] + '.jpg' for image in test_data]

x_test = tf.data.Dataset.from_tensor_slices((anchor_images_test, first_image_test, second_image_test))
dataset_test = tf.data.Dataset.zip((x_test,))
dataset_test = dataset_test.map(preprocess_data_test)

dataset_test = dataset_test.batch(32, drop_remainder=False)
dataset_test = dataset_test.prefetch(8)

predictions = model.predict(dataset_test, verbose=1, steps=2)

np.savetxt('predictions.txt', predictions, fmt='%i')

# dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
# dataset = dataset.shuffle(buffer_size=1024)
# dataset = dataset.map(preprocess_triplets)
#
# # Let's now split our dataset in train and validation.
# image_count = len(anchor_images)
# train_dataset = dataset.take(round(image_count * 0.8))
# val_dataset = dataset.skip(round(image_count * 0.8))
#
# train_dataset = train_dataset.batch(32, drop_remainder=False)
# train_dataset = train_dataset.prefetch(8)
#
# val_dataset = val_dataset.batch(32, drop_remainder=False)
# val_dataset = val_dataset.prefetch(8)
#
#
# #PRETRAINED MODEL
# base_cnn = resnet.ResNet50(
#     weights="imagenet", input_shape=target_shape + (3,), include_top=False
# )
#
# flatten = layers.Flatten()(base_cnn.output)
# dense1 = layers.Dense(512, activation="relu")(flatten)
# dense1 = layers.BatchNormalization()(dense1)
# dense2 = layers.Dense(256, activation="relu")(dense1)
# dense2 = layers.BatchNormalization()(dense2)
# output = layers.Dense(256)(dense2)
#
# embedding = Model(base_cnn.input, output, name="Embedding")
#
# trainable = False
# for layer in base_cnn.layers:
#     if layer.name == "conv5_block1_out":
#         trainable = True
#     layer.trainable = trainable
#
#
# ## Setting up the Siamese Network model
# class DistanceLayer(layers.Layer):
#     """
#     This layer is responsible for computing the distance between the anchor
#     embedding and the positive embedding, and the anchor embedding and the
#     negative embedding.
#     """
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#     def call(self, anchor, positive, negative):
#         ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
#         an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
#         return (ap_distance, an_distance)
#
#
# anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
# positive_input = layers.Input(name="positive", shape=target_shape + (3,))
# negative_input = layers.Input(name="negative", shape=target_shape + (3,))
#
# distances = DistanceLayer()(
#     embedding(resnet.preprocess_input(anchor_input)),
#     embedding(resnet.preprocess_input(positive_input)),
#     embedding(resnet.preprocess_input(negative_input)),
# )
#
# siamese_network = Model(
#     inputs=[anchor_input, positive_input, negative_input], outputs=distances
# )
#
#
# ## Putting everything together
# class SiameseModel(Model):
#     """The Siamese Network model with a custom training and testing loops.
#     Computes the triplet loss using the three embeddings produced by the
#     Siamese Network.
#     The triplet loss is defined as:
#        L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
#     """
#
#     def __init__(self, siamese_network, margin=0.5):
#         super(SiameseModel, self).__init__()
#         self.siamese_network = siamese_network
#         self.margin = margin
#         self.loss_tracker = metrics.Mean(name="loss")
#
#     def call(self, inputs):
#         return self.siamese_network(inputs)
#
#     def train_step(self, data):
#         # GradientTape is a context manager that records every operation that
#         # you do inside. We are using it here to compute the loss so we can get
#         # the gradients and apply them using the optimizer specified in
#         # `compile()`.
#         with tf.GradientTape() as tape:
#             loss = self._compute_loss(data)
#
#         # Storing the gradients of the loss function with respect to the
#         # weights/parameters.
#         gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
#
#         # Applying the gradients on the model using the specified optimizer
#         self.optimizer.apply_gradients(
#             zip(gradients, self.siamese_network.trainable_weights)
#         )
#
#         # Let's update and return the training loss metric.
#         self.loss_tracker.update_state(loss)
#         return {"loss": self.loss_tracker.result()}
#
#     def test_step(self, data):
#         loss = self._compute_loss(data)
#
#         # Let's update and return the loss metric.
#         self.loss_tracker.update_state(loss)
#         return {"loss": self.loss_tracker.result()}
#
#     def _compute_loss(self, data):
#         # The output of the network is a tuple containing the distances
#         # between the anchor and the positive example, and the anchor and
#         # the negative example.
#         ap_distance, an_distance = self.siamese_network(data)
#
#         # Computing the Triplet Loss by subtracting both distances and
#         # making sure we don't get a negative value.
#         loss = ap_distance - an_distance
#         loss = tf.maximum(loss + self.margin, 0.0)
#         return loss
#
#     @property
#     def metrics(self):
#         # We need to list our metrics here so the `reset_states()` can be
#         # called automatically.
#         return [self.loss_tracker]
#
#
# ## Training
# siamese_model = SiameseModel(siamese_network)
# siamese_model.compile(optimizer=optimizers.Adam(0.0001))
# siamese_model.fit(train_dataset, epochs=10, validation_data=val_dataset)
#
# sample = next(iter(train_dataset))
# anchor, positive, negative = sample
# anchor_embedding, positive_embedding, negative_embedding = (
#     embedding(resnet.preprocess_input(anchor)),
#     embedding(resnet.preprocess_input(positive)),
#     embedding(resnet.preprocess_input(negative)),
# )
# cosine_similarity = metrics.CosineSimilarity()
#
# positive_similarity = cosine_similarity(anchor_embedding, positive_embedding)
# print("Positive similarity:", positive_similarity.numpy())
#
# negative_similarity = cosine_similarity(anchor_embedding, negative_embedding)
# print("Negative similarity", negative_similarity.numpy())
