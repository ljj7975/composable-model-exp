from torchvision import transforms
from .mnist_dataset import MNIST
from torchvision import datasets
from base import BaseDataLoader
import torch

class MnistDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, size_per_class=None, training=True, target_class=[0,1,2,3,4,5,6,7,8,9], unknown=True):
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
                    unknown=unknown)

        super(MnistDataLoader, self).__init__(
                    self.dataset,
                    batch_size, shuffle,
                    validation_split,
                    num_workers)