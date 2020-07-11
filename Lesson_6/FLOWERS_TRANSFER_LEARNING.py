import os
import sys
sys.path.append(os.path.abspath('..'))

from PIL import Image

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfd

from Lesson_3.plotImages import dim

def format_image(image, label):
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))/255.0
  return image, label

# Check the minimum dimensions
# dim('C:\\Users\\lukas.rimkus\\.keras\\datasets\\flower_photos\\validation')
# Download Flower dataset
# Already in C:\Users\lukas.rimkus\.keras\datasets
# Actualy following the example couase doesn't really work like that
# This is tensorflow_dataset that contains not IMAGES but labels.txt and tf_flowers-train.tfrecord-00000-of-00002 files
(training_set, validation_set), info = tfd.load(
    'tf_flowers',
    split=['train[:80%]', 'train[70%:]'],
    with_info=True,
    as_supervised=True,
)
print('training_set: ', training_set)
print('validation_set: ', validation_set)
print('dataset_info: ', info)

# Prepare the data
IMG_SIZE = 224
BATCH_SIZE = 32

num_classes = info.features['label'].num_classes

num_training_examples = 0
num_validation_examples = 0

for example in training_set:
  num_training_examples += 1

for example in validation_set:
  num_validation_examples += 1

print('Total Number of Classes: {}'.format(num_classes))
print('Total Number of Training Images: {}'.format(num_training_examples))
print('Total Number of Validation Images: {} \n'.format(num_validation_examples))


# CREATE BATCHES
train_batches = training_set.shuffle(num_training_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)

validation_batches = validation_set.map(format_image).batch(BATCH_SIZE).prefetch(1)

# CREATE FEATURE EXTRACTOR
URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(URL,
                                   input_shape=(IMG_SIZE, IMG_SIZE, 3))

feature_extractor.trainable = False

model = tf.keras.Sequential([
  feature_extractor,
  tf.keras.layers.Dense(num_classes)
])

model.summary()

model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

EPOCHS = 6

history = model.fit(train_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches)


# Plot Training and Validation Graphs

import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Check Predictions
import numpy as np

class_names = np.array(info.features['label'].names)

print(class_names)

image_batch, label_batch = next(iter(train_batches))


image_batch = image_batch.numpy()
label_batch = label_batch.numpy()

predicted_batch = model.predict(image_batch)
predicted_batch = tf.squeeze(predicted_batch).numpy()

predicted_ids = np.argmax(predicted_batch, axis=-1)
predicted_class_names = class_names[predicted_ids]

print(predicted_class_names)

# Perform Transfer Learning with the Inception Model

IMAGE_RES = 299

(training_set, validation_set), dataset_info = tfd.load(
    'tf_flowers',
    with_info=True,
    as_supervised=True,
    split=['train[:70%]', 'train[70%:]'],
)
train_batches = training_set.shuffle(num_training_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = validation_set.map(format_image).batch(BATCH_SIZE).prefetch(1)

URL = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"
feature_extractor = hub.KerasLayer(URL,
  input_shape=(IMAGE_RES, IMAGE_RES, 3),
  trainable=False)

model_inception = tf.keras.Sequential([
  feature_extractor,
  tf.keras.layers.Dense(num_classes)
])

model_inception.summary()

model_inception.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

EPOCHS = 6

history = model_inception.fit(train_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches)



