# Going further with CNN
import sys
import os
sys.path.append(os.path.abspath('..'))

from Lesson_4.plotImages import plotimages,plotbatch, plotImagesGrid
from Lesson_4.nrml import normalize

import tensorflow as tf
import tensorflow_datasets
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import math
import matplotlib.pyplot as plt

# Download the dataset
# _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
# zip_dir = tf.keras.utils.get_file('cats_and_dogs_filtered.zip', origin=_URL, extract=True)

# zip_dir_base = os.path.dirname(zip_dir)
cats_n_dogs = 'cats_and_dogs_filtered'
flowers = 'flower_photos'
base_dir = os.path.join('C:\\Users\lukas.rimkus\.keras\datasets', cats_n_dogs)
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))
num_cats_vl = len(os.listdir(validation_cats_dir))
num_dogs_vl = len(os.listdir(validation_dogs_dir))

print('number of training cat images: ' , num_cats_tr)
print('number of training dog images: ' , num_dogs_tr)
print('number of validation cat images: ' , num_cats_vl)
print('number of validation dog images: ' , num_dogs_vl)

# Model parameters
BATCH_SIZE = 100
IMAGE_SHAPE = 150
class_names = ['cat', 'dog']
training_image_generator = ImageDataGenerator(rescale=1./255)
validation_image_generator = ImageDataGenerator(rescale=1./255)

# Put all together
image_gen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_data_gen = image_gen_train.flow_from_directory( batch_size=BATCH_SIZE,
                                                directory=train_dir,
                                                shuffle=True,
                                                target_size=(IMAGE_SHAPE, IMAGE_SHAPE),
                                                class_mode='binary')

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImagesGrid(augmented_images)

validation_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=validation_dir,
                                                              shuffle=False,
                                                              target_size=(IMAGE_SHAPE, IMAGE_SHAPE),
                                                              class_mode='binary')




# plotImagesGrid(sample_training_images[:5])


# Build a model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape= (IMAGE_SHAPE, IMAGE_SHAPE, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])
#
# Compile a model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

# # Iteration parameters
LAYERS = len(model.layers)
EPOCH = 1

# V Train the model V
history = model.fit(
    train_data_gen,
    epochs=EPOCH,
    steps_per_epoch=math.ceil((num_dogs_tr + num_cats_tr)/BATCH_SIZE),
    validation_data=validation_data_gen,
    validation_steps=math.ceil((num_cats_vl+num_dogs_vl)/BATCH_SIZE)
)
#
# Visualizing
# Retrieve a list of accuracy results on training and validation data
# sets for each training epoch
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Retrieve a list of list results on training and validation data
# sets for each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')

plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation `loss')
plt.show()