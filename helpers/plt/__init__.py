import numpy as np
from matplotlib import pyplot as plt

def show_example(example, label, dim=(28, 28)):
    """
    Displays a single example and label.
    If example is already unrolled from (28, 28) to (784,), it will be reshaped first.
    """
    example = np.squeeze(example)
    label = np.squeeze(label)
    if example.shape != dim:
        example = example.reshape(dim)
    plt.imshow(example, cmap='binary')
    plt.xlabel('Label: ' + str(int(label)))
    plt.xticks([])
    plt.yticks([])
    plt.show()


def plot_train_digits(images,labels):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i].squeeze(2), cmap='Greys')
        # The CIFAR labels happen to be arrays, 
        # which is why you need the extra index
        plt.xlabel(labels[i])
    plt.show()

    
def show_ten_examples(examples, labels, preds):
    """
    Display 10 examples along with their labels.
    If the labels are same as predictions, they are displayed in green.
    If the labels and predictions are different, they are displayed in red.
    """
    plt.figure(figsize=(8, 4))
    for i in range(0, 10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(np.reshape(examples[i], (28, 28)), cmap='binary')
        plt.xticks([])
        plt.yticks([])
        y = int(np.squeeze(labels[i]))
        p = int(np.squeeze(preds[i]))
        plt.xlabel(str(p), color='green' if y == p else 'red')
    plt.show()

def plot_metrics(model):
    """
    Plot validation accuracy and validation loss during model training.
    """
    plt.figure(figsize=(12, 4))

    accuracies = model.accuracies
    losses = model.losses
    iterations = len(losses)

    plt.subplot(1, 2, 1)
    plt.plot(range(iterations), accuracies, 'r-')
    plt.ylim([0., 1.])
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations')
    plt.title('Acc: {:.3f}'.format(accuracies[-1]))

    plt.subplot(1, 2, 2)
    plt.plot(range(iterations), losses, 'b-')
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.title('Loss: {:.3f}'.format(losses[-1]))

    plt.show()

def plot_accuracies(history):
    """
    plot graph for accuracy and loss
    """
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    plt.show()

