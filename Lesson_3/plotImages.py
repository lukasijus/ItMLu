# This function plots images
import matplotlib.pyplot as plt
import math
def plotimages(class_names, dataset, imageCount):
    i = 0
    for image, label in dataset.take(imageCount):
        image = image.numpy().reshape((28, 28))
        x_axis_length = int(round(math.sqrt(imageCount)))
        y_axis_length = int(round(math.sqrt(imageCount)))
        plt.subplot(x_axis_length, y_axis_length, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image, cmap=plt.cm.binary)
        plt.grid(False)
        plt.xlabel(class_names[label])
        i = i + 1
    plt.show()

