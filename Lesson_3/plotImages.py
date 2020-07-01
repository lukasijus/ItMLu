# This function plots images
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

from colour import Color

red = Color("red")

def plotimages(class_names, dataset, imageCount):
    i = 0
    for image, label in dataset.take(imageCount):
        image = image.numpy().reshape((28, 28))
        x_axis_length = int(round(math.sqrt(imageCount)))
        y_axis_length = int(round(math.sqrt(imageCount)))
        fig, ax = plt.subplot(x_axis_length, y_axis_length, i + 1)
        ax.imshow(image, cmap=plt.cm.binary)
        rect = patches.Rectangle((1, 1), 26, 26, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.grid(False)
        plt.xlabel(class_names[label])
        i = i + 1
    plt.show()

def plotbatch(name, loss, accuracy, batch_size, test_labels,class_names,predictions,test_images):
    index = 0
    loss = float("{:.2f}".format(loss))
    accuracy = float("{:.2f}".format(accuracy))
    fig, axes = plt.subplots(nrows=int(round(math.sqrt(batch_size))), ncols=int(round(math.ceil(math.sqrt(batch_size)))))
    fig.suptitle(name + '\n' + 'Loss: ' + str(loss) + ' ' + 'Accuracy: ' + str(accuracy))
    axes = axes.flatten()
    plt.subplots_adjust(
        left=0.03,
        bottom=0.13,
        right=0.98,
        top=0.93,
        wspace=0.95,
        hspace=0.98)
    plt.grid(False)
    plt.axis('off')
    for image in test_images:
        rect = patches.Rectangle((0, 0), 27, 27, linewidth=5, edgecolor='g', facecolor='none')
        class_name_index = 0
        highest_prediction = 0
        highest_prediction_index = 0
        for value in predictions[index]:
            if value > highest_prediction:
                highest_prediction = value
                highest_prediction_index = class_name_index
            class_name_index += 1
        axes[index].set_xlabel('P: ' + str(float("{:.2f}".format(highest_prediction))) +
                   '\n' + 'PC: ' + class_names[highest_prediction_index] +
                   '\n' + 'GTC: ' + class_names[test_labels[index]])
        if highest_prediction_index != test_labels[index]:
            rect = patches.Rectangle((0, 0), 27, 27, linewidth=5, edgecolor='r', facecolor='none')
        axes[index].add_patch(rect)
        axes[index].xaxis.set_label_coords(0.05, -0.025)
        axes[index].imshow(test_images[index].reshape((28,28)), cmap=plt.cm.binary)
        axes[index].set_yticklabels([])
        axes[index].set_xticklabels([])
        index += 1
    plt.show()
