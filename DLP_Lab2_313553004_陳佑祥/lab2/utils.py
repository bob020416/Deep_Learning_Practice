# script for drawing figures, and more if needed
import matplotlib.pyplot as plt
import numpy as np
import itertools
import torch
from sklearn.metrics import confusion_matrix

def plot_loss(train_losses, val_losses, title='Loss over epochs'):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy(train_accuracies, val_accuracies, title='Accuracy over epochs'):
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(cm, class_names):
    """
    Plots the confusion matrix using matplotlib.

    Args:
        cm: Confusion matrix from sklearn.metrics.confusion_matrix
        class_names: List of class names
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def calculate_confusion_matrix(model, data_loader, class_names):
    """
    Calculates and plots the confusion matrix for the given model and data loader.

    Args:
        model: Trained model
        data_loader: DataLoader for the dataset
        class_names: List of class names
    """
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names)
