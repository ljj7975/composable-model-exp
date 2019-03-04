from torchvision import transforms
from .mnist_dataset import MNIST
from base import BaseDataLoader
import torch

class MnistDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True, target_class=None):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        self.data_dir = data_dir
        self.target_class = target_class
        self.dataset = MNIST(
            self.data_dir,
            target_class=target_class,
            train=training,
            download=True,
            transform=trsfm)

        if self.target_class:
            super(MnistDataLoader, self).__init__(
                self.dataset,
                batch_size, shuffle,
                validation_split,
                num_workers,
                collate_fn = self.__collate_fn__)
        else:
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

        for index, val in enumerate(label):
            processed_label[index] = self.target_class.index(val)

        return processed_data, processed_label.long()
