import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow_hub as hub

(train_examples, validation_examples), info = tfds.load(
    'cats_vs_dogs',
    with_info=True,
    as_supervised=True,
    split=['train[:80%]', 'train[80%:]'],
)

num_examples = info.splits['train'].num_examples
num_classes = info.features['label'].num_classes

for i, example_image in enumerate(train_examples.take(3)):
  print("Image {} shape: {}".format(i+1, example_image[0].shape))

IMAGE_RES = 224

def format_image(image, label):
  image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
  return image, label

BATCH_SIZE = 32

train_batches      = train_examples.shuffle(num_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = validation_examples.map(format_image).batch(BATCH_SIZE).prefetch(1)

print('train batches: ', train_batches)
print('validation batches: ', validation_batches)

# CLASSIFIER_URL ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
#
# model = tf.keras.Sequential([
#     hub.KerasLayer(CLASSIFIER_URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))
# ])
#
# image_batch, label_batch = next(iter(train_batches.take(1)))
# image_batch = image_batch.numpy()
# label_batch = label_batch.numpy()
# print('image batch: ',  image_batch)
#
# result_batch = model.predict(image_batch)
#
# import numpy as np
#
# labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
# imagenet_labels = np.array(open(labels_path).read().splitlines())
#
# predicted_class_names = imagenet_labels[np.argmax(result_batch, axis=-1)]
# print(predicted_class_names)
#
# plt.figure(figsize=(10,9))
# for n in range(30):
#     plt.subplot(6,5,n+1)
#     plt.subplots_adjust(hspace = 0.3)
#     plt.imshow(image_batch[n])
#     plt.title(predicted_class_names[n])
#     plt.axis('off')
#     _ = plt.suptitle("ImageNet predictions")
#
#
# URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
# feature_extractor = hub.KerasLayer(URL,
#                                    input_shape=(IMAGE_RES, IMAGE_RES,3))
#
# feature_batch = feature_extractor(image_batch)
# print(feature_batch.shape)
#
# feature_extractor.trainable = False
#
# model = tf.keras.Sequential([
#     feature_extractor,
#     tf.keras.layers.Dense(2)
# ])
#
# model.summary()
#
# model.compile(
#   optimizer='adam',
#   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#   metrics=['accuracy'])
#
# EPOCHS = 1
# history = model.fit(train_batches,
#                     epochs=EPOCHS,
#                     validation_data=validation_batches)
#
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
#
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs_range = range(EPOCHS)
#
# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
#
# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()
#
# class_names = np.array(info.features['label'].names)
# print(class_names)
#
# predicted_batch = model.predict(image_batch)
# predicted_batch = tf.squeeze(predicted_batch).numpy()
# predicted_ids = np.argmax(predicted_batch, axis=-1)
# predicted_class_names = class_names[predicted_ids]
# print(predicted_class_names)
#
# print("Labels: ", label_batch)
# print("Predicted labels: ", predicted_ids)
#
# plt.figure(figsize=(10,9))
# for n in range(30):
#   plt.subplot(6,5,n+1)
#   plt.subplots_adjust(hspace = 0.3)
#   plt.imshow(image_batch[n])
#   color = "blue" if predicted_ids[n] == label_batch[n] else "red"
#   plt.title(predicted_class_names[n].title(), color=color)
#   plt.axis('off')
# _ = plt.suptitle("Model predictions (blue: correct, red: incorrect)")
# plt.show()