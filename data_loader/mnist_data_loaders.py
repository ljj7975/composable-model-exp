from torchvision import transforms
from .mnist_dataset import MNIST
from torchvision import datasets
from base import BaseDataLoader
import torch

class MnistDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True, target_class=None, keep_unknown=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        self.data_dir = data_dir
        self.target_class = target_class
        self.keep_unknown = keep_unknown

        if self.target_class:
            if self.keep_unknown:
                self.dataset = datasets.MNIST(
                    self.data_dir,
                    train=training,
                    download=True,
                    transform=trsfm)

                super(MnistDataLoader, self).__init__(
                    self.dataset,
                    batch_size, shuffle,
                    validation_split,
                    num_workers,
                    collate_fn = self.__collate_fn__)
            else:
                self.dataset = MNIST(
                    self.data_dir,
                    target_class=target_class,
                    train=training,
                    download=True,
                    transform=trsfm)

                super(MnistDataLoader, self).__init__(
                    self.dataset,
                    batch_size, shuffle,
                    validation_split,
                    num_workers,
                    collate_fn = self.__collate_fn__)
        else:
            self.dataset = datasets.MNIST(
                self.data_dir,
                train=training,
                download=True,
                transform=trsfm)

            super(MnistDataLoader, self).__init__(
                self.dataset,
                batch_size, shuffle,
                validation_split,
                num_workers)

    def __collate_fn__(self, batch):
        data, label = zip(*batch)

        # Merge images (from tuple of 3D tensor to 4D tensor).
        processed_data = torch.stack(data, 0)
        processed_label = torch.zeros(len(label))
        negative_index = len(self.target_class)

        for index, val in enumerate(label):
            try:
                processed_label[index] = self.target_class.index(val)
            except ValueError:
                processed_label[index] = negative_index

        return processed_data, processed_label.long()
