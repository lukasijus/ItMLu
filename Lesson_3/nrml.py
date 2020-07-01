import tensorflow as tf
# Normalize images from (0, 255) to (0, 1)
def normalize(images, labels):
    images = tf.cast(images,tf.float32)
    images /= 255
    return images,labels

