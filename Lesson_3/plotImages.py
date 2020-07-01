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

def plotbatch(test_labels,class_names,predictions,test_images):
    index = 0
    for image in test_images:
        plt.subplot(4,8, index + 1)
        plt.subplots_adjust(
                    left=0.03,
                    bottom=0.13,
                    right=0.98,
                    top=0.98,
                    wspace=0.95,
                    hspace=0.98)
        plt.grid(False)
        plt.imshow(test_images[index].reshape((28,28)), cmap=plt.cm.binary)
        class_name_index = 0
        highest_prediction = 0
        highest_prediction_index = 0
        for value in predictions[index]:
            if value > highest_prediction:
                highest_prediction = value
                highest_prediction_index = class_name_index
            class_name_index += 1
        plt.xlabel('P: ' + str(highest_prediction) +
                   '\n' + 'PC: ' + class_names[highest_prediction_index] +
                   '\n' + 'GTC: ' + class_names[test_labels[index]])
        index += 1
    plt.show()
