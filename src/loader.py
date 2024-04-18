from typing import Callable, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.utils import *


class CustomMNIST(Dataset):
    def __init__(
            self,
            data,
            targets,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
        ):
        assert(len(data) == len(targets))
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class LabeledUnlabeledMNIST:
    def __init__(self, labeled_percent: float = 1.0):
        self.labeled_percent = labeled_percent
        transform = get_transform()
        
        self.mnist_train, self.mnist_test = download(transform)
        self.labeled_data, self.unlabeled_data = self.split_data()
        self.labeled_dataset = CustomMNIST(*self.labeled_data, transform=transform)
        self.unlabeled_dataset = CustomMNIST(*self.unlabeled_data, transform=transform)

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
        labeled_images = self.mnist_train.data[labeled_indices]
        labeled_labels = self.mnist_train.targets[labeled_indices]

        unlabeled_images = self.mnist_train.data[unlabeled_indices]
        unlabeled_labels = torch.from_numpy(np.full(num_unlabeled_samples, -1))

        return (labeled_images, labeled_labels), (unlabeled_images, unlabeled_labels)

# # Example usage
# labeled_percent = 0.5
# mnist_loader = LabeledUnlabeledMNIST(labeled_percent)
# print(f"Number of labeled samples: {len(mnist_loader.labeled_dataset)}")
# print(f"Number of unlabeled samples: {len(mnist_loader.unlabeled_dataset)}")
