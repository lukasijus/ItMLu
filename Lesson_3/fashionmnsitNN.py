from __future__ import absolute_import, division, print_function

# Import Tensorflow and Tensorflow datasets
import tensorflow as tf
import tensorflow_datasets as tfds

# Helper libraries
import math
import numpy as np
import matplotlib.pyplot as plt
import nrml
import plotImages
print(tf.__version__)

# Build dataset (use only once)
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

class_names = [
'T-shirt/top',
'Trouser    ',
'Pullover   ',
'Dress      ',
'Coat       ',
'Sandal     ',
'Shirt      ',
'Sneaker    ',
'Bag        ',
'Ankle boot '
]

train_numbers = metadata.splits['train'].num_examples
test_numbers = metadata.splits['test'].num_examples
print('Training number examples {}'.format(train_numbers))
print('Test number examples {}'.format(test_numbers))

train_dataset = train_dataset.map(nrml.normalize)
test_dataset = test_dataset.map(nrml.normalize)

# Iteration parameters
BATCH_SIZE = 32
train_dataset = train_dataset.repeat().shuffle(train_numbers).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# Create a model
INPUT_NEURONS = 256
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(INPUT_NEURONS, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
# Compile a model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

# V Train the model V
model.fit(
    train_dataset,
    epochs=3,
    steps_per_epoch=math.ceil(train_numbers/BATCH_SIZE)
)

# Evaluation
test_loss, test_accuracy = model.evaluate(
    test_dataset,
    steps=math.ceil(test_numbers/BATCH_SIZE)
)
np.set_printoptions(precision=2)
print('Accuracy of the model:', test_accuracy)

# Make predictions
batchCount = 1
i = 0
for test_images, test_labels in test_dataset.take(batchCount):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions = model.predict(test_images)
    i += 1
    plotImages.plotbatch(test_labels,class_names,predictions,test_images)

# Review
print('Shape of the prediction array: ', predictions.shape)
print('The predicted class names for the n batch: ', predictions[0])
np.set_printoptions(precision=2)
print('Sort by Highest rated: ', np.sort(predictions[0])[::-1])
print('Labels: ', test_labels[0])
print('Labels shape: ', test_labels.shape)
print('Shape of predicted batch labels: ', predictions.shape)
print('Shape of predicted batch image: ', test_images.shape)
print('Shape of predicted labels: ', predictions[0].shape)
print('Shape of predicted image: ', test_images[0].shape)





