# Initialize all the imports for your Neural network
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import math
import numpy as np

import Lesson_4.plotImages as plt2
import glob

# Download dataset
def download(url, file):
    return tf.keras.utils.get_file(file, origin=url, extract=True)
