import torch
from utils import *
from torchvision import datasets, transforms
import numpy as np

class LabeledUnlabeledMNIST:
    def __init__(self, labeled_percent):
        self.labeled_percent = labeled_percent
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        self.mnist_train, self.mnist_test = download(transform)
        self.labeled_data, self.unlabeled_data = self.split_data()

    def split_data(self):
        # Determine number of labeled and unlabeled samples
        num_labeled_samples = int(len(self.mnist_train) * self.labeled_percent)
        num_unlabeled_samples = len(self.mnist_train) - num_labeled_samples

        # Create shuffled indices
        indices = np.arange(len(self.mnist_train))
        np.random.shuffle(indices)

        # Split indices into labeled and unlabeled indices
        labeled_indices = indices[:num_labeled_samples]
        unlabeled_indices = indices[num_labeled_samples:]

        # Create labeled and unlabeled datasets
        labeled_images = self.mnist_train.data[labeled_indices].float() / 255.0
        labeled_labels = self.mnist_train.targets[labeled_indices].numpy()
        labeled_data = list(zip(labeled_images, labeled_labels))

        unlabeled_images = self.mnist_train.data[unlabeled_indices].float() / 255.0
        unlabeled_labels = np.full(num_unlabeled_samples, -1)
        unlabeled_data = list(zip(unlabeled_images, unlabeled_labels))

        return labeled_data, unlabeled_data

# # Example usage
# labeled_percent = 0.5
# mnist_loader = LabeledUnlabeledMNIST(labeled_percent)
# print(f"Number of labeled samples: {len(mnist_loader.labeled_data)}")
# print(f"Number of unlabeled samples: {len(mnist_loader.unlabeled_data)}")
