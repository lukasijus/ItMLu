import tensorflow as tf
import tensorflow_datasets as tds
import tensorflow_hub as hub

import numpy as np
import matplotlib.pyplot as plt
import math
import time

# PART 1. import examples using tensorflow datasets
(train_examples, validation_examples) , dataset_info = tds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[:80%]'],
    with_info=True,
    as_supervised=True
)

print('dataset_info: ', dataset_info)
print('train_examples: ', train_examples)
print('validation_examples: ', validation_examples)

# Format and prepare batches

def format_image(image, label):
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
    return image, label

num_examples = dataset_info.splits['train'].num_examples

print('num_examples: ', num_examples)

IMAGE_RES = 224
BATCH_SIZE = 32

train_batches       = train_examples.cache().shuffle(num_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)
vaildation_batches  = validation_examples.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)

print('train_batches: ', train_batches)
print('validation_batches: ', vaildation_batches)

# PART 2. TRANSFER lEARNING
URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(URL,
                                   input_shape=(IMAGE_RES, IMAGE_RES, 3))

feature_extractor.trainable = False

# Attach classification head
model = tf.keras.Sequential([
    feature_extractor,
    tf.keras.layers.Dense(2)
])

model.summary()

# Train the model
model.compile(
    optimizer='adam',
    loss=tf.losses.sparse_categorical_crossentropy,
    metrics=['accuracy']
)

EPOCHS = 3
history = model.fit(train_batches,epochs=EPOCHS,validation_data=vaildation_batches)

# Check the predictions
class_names = np.array(dataset_info.features['label'].names)
# Run an image batch through the model and see whats happening
image_batch, label_batch = next(iter(train_batches.take(1)))
print('image_batch: ', image_batch, 'label_batch: ', label_batch)
image_batch = image_batch.numpy()
label_batch = label_batch.numpy()
print('image_batch: ', image_batch, 'label_batch: ', label_batch)

predicted_batch = model.predict(image_batch)
print('predicted_batch: ', predicted_batch)
predicted_batch = tf.squeeze(predicted_batch).numpy()
print('predicted_batch: ', predicted_batch)
predicted_ids = np.argmax(predicted_batch, axis=-1)
predicted_class_names = class_names[predicted_ids]
print('predicted_class_names: ', predicted_class_names)

plt.figure(figsize=(10,9))
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  color = "blue" if predicted_ids[n] == label_batch[n] else "red"
  plt.title(predicted_class_names[n].title(), color=color)
  plt.axis('off')
_ = plt.suptitle("Model predictions (blue: correct, red: incorrect)")

# PART 3. SAVE AS KERAS .h5 MODEL
t = time.time()

export_path_keras = "./{}.h5".format(int(t))
print(export_path_keras)

model.save(export_path_keras)