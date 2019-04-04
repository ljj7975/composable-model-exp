from torchvision import transforms
from .mnist_dataset import MNIST
from torchvision import datasets
from base import BaseDataLoader
import torch
import os

class MnistDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, size_per_class=None, training=True, target_class=None, unknown=True, seed=0):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])

        self.data_dir = data_dir
        self.target_class = target_class
        self.unknown = unknown

        self.dataset = MNIST(
                    self.data_dir,
                    train=training,
                    download=True,
                    target_class=target_class,
                    size_per_class=size_per_class,
                    transform=trsfm,
                    unknown=unknown,
                    seed=seed)

        super(MnistDataLoader, self).__init__(
                    self.dataset,
                    batch_size, shuffle,
                    validation_split,
                    num_workers,
                    seed)
