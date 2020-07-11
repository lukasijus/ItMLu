import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tds

(train_examples, validation_examples) , dataset_info = tds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[:80%]'],
    with_info=True,
    as_supervised=True
)


# Format and prepare batches


def format_image(image, label):
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
    return image, label

num_examples = dataset_info.splits['train'].num_examples

IMAGE_RES = 224
BATCH_SIZE = 32

train_batches       = train_examples.cache().shuffle(num_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches  = validation_examples.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)

image_batch, label_batch = next(iter(train_batches.take(1)))

reloaded = tf.keras.models.load_model(
    'C:\\Users\\lukas.rimkus\\Software\\Introduction_to_Machine_Learning_Udacity\\Lesson_7\\1594470041.h5',
    custom_objects={'KerasLayer': hub.KerasLayer }
)

reloaded.summary()

EPOCHS = 3
history = reloaded.fit(train_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches)

import time

# SAVE AS KERAS .h5 MODEL
t = time.time()

export_path_keras = "./{}.h5".format(int(t))
print(export_path_keras)

reloaded.save(export_path_keras)

# SAVE AS SAVE_MODEL

export_path_sm = "./{}".format(int(t))
print(export_path_sm)
tf.saved_model.save(reloaded, export_path_sm)