import os
import sys
sys.path.append(os.path.abspath('..'))

from PIL import Image

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfd

from Lesson_3.plotImages import dim

# Check the minimum dimensions
# dim('C:\\Users\\lukas.rimkus\\.keras\\datasets\\flower_photos\\validation')
# Download Flower dataset
# Already in C:\Users\lukas.rimkus\.keras\datasets
# Actualy following the example couase doesn't really work like that
# This is tensorflow_dataset that contains not IMAGES but labels.txt and tf_flowers-train.tfrecord-00000-of-00002 files
(training_set, validation_set), dataset_info = tfd.load(
    'tf_flowers',
    split=['train[:70%]', 'train[70%:]'],
    with_info=True,
    as_supervised=True,
)
print('training_set: ', training_set)
print('validation_set: ', validation_set)
print('dataset_info: ', dataset_info)

# Load the data
IMG_SIZE = 125
BATCH_SIZE = 32


def format_image(image, label):
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))/255.0
  return image, label


prepare_data = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=25, zoom_range=0.3, horizontal_flip=True, rescale=1./255)
train_data = prepare_data.flow_from_directory(
    directory='C:\\Users\\lukas.rimkus\\.keras\\datasets\\flower_photos\\train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True
)
validation_data = prepare_data.flow_from_directory(
    directory='C:\\Users\\lukas.rimkus\\.keras\\datasets\\flower_photos\\validation',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)
print('train_data: ', train_data)
print('validation_data: ', validation_data)

# load the model (MOBILE NET MODEL FEATURE EXTRACTOR)
URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
feature_extractor = hub.KerasLayer(URL,
                                   input_shape=(IMG_SIZE, IMG_SIZE, 3))


