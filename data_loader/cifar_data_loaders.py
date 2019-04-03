from torchvision import transforms
from .cifar_dataset import CIFAR10,CIFAR100
from torchvision import datasets
from base import BaseDataLoader
import torch
import os

class Cifar10DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, size_per_class=None, training=True, target_class=[0,1,2,3,4,5,6,7,8,9], unknown=True, seed=0):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        if not os.path.isdir(data_dir):
            data_dir = '/data/cifar10'
        self.data_dir = data_dir
        self.target_class = target_class
        self.unknown = unknown

        # self.dataset = CIFAR10(
        #             self.data_dir,
        #             train=training,
        #             download=True,
        #             target_class=target_class,
        #             size_per_class=size_per_class,
        #             transform=trsfm,
        #             unknown=unknown,
        #             seed=seed)
        self.dataset = CIFAR10(
                    self.data_dir,
                    train=training,
                    download=True,
                    transform=trsfm)

        super(Cifar10Loader, self).__init__(
                    self.dataset,
                    batch_size, shuffle,
                    validation_split,
                    num_workers,
                    seed)

class Cifar100DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, size_per_class=None, training=True, target_class=[0,1,2,3,4,5,6,7,8,9], unknown=True, seed=0):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])

        if not os.path.isdir(data_dir):
            data_dir = '/data/cifar100'
        self.data_dir = data_dir
        self.target_class = target_class
        self.unknown = unknown

        # self.dataset = CIFAR100(
        #             self.data_dir,
        #             train=training,
        #             download=True,
        #             target_class=target_class,
        #             size_per_class=size_per_class,
        #             transform=trsfm,
        #             unknown=unknown,
        #             seed=seed)

        self.dataset = CIFAR100(
                    self.data_dir,
                    train=training,
                    download=True,
                    transform=trsfm)

        super(Cifar100Loader, self).__init__(
                    self.dataset,
                    batch_size, shuffle,
                    validation_split,
                    num_workers,
                    seed)
