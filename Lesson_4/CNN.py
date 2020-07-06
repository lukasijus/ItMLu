import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

# Improve progress bar
import tqdm
import tqdm.auto
tqdm.tqdm = tqdm.auto.tqdm
import nrml
import math
import plotImages

dataset, metadata = tfds.load('fashion_mnist', as_supervised=True,with_info=True,)
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



# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu, input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2,2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])


# Compile a model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

# Iteration parameters
LAYERS = len(model.layers)
BATCH_SIZE = 32
EPOCH = 10
train_dataset = train_dataset.repeat().shuffle(train_numbers).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# V Train the model V
model.fit(
    train_dataset,
    epochs=EPOCH,
    steps_per_epoch=math.ceil(train_numbers/BATCH_SIZE)
)

# Evaluation
test_loss, test_accuracy = model.evaluate(
    test_dataset,
    steps=math.ceil(test_numbers/BATCH_SIZE)
)


# Make predictions
batchCount = 1
i = 0
name = '_LAYERS_' + str(LAYERS) + '_INPUT_NEURON_=_' + 'CNN=32-64' + '_BATCH_SIZE_=_' + str(BATCH_SIZE) + '_Epoch_=_' + str(EPOCH)
for test_images, test_labels in test_dataset.take(batchCount):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions = model.predict(test_images)
    i += 1
    plotImages.plotbatch(name,test_loss,test_accuracy,BATCH_SIZE,test_labels,class_names,predictions,test_images, save = True)
