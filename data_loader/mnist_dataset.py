import warnings
import torch.utils.data as data
from PIL import Image
import os
import gzip
import numpy as np
import torch
import codecs
from utils.util import download_url, makedir_exist_ok

class MNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, target_class=None, size_per_class=None, train=True,
                 transform=None, target_transform=None, download=False,
                 unknown=False, seed=0):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        np.random.seed(seed)

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

        # resize the data

        # self.size_per_class = size_per_class

        # if self.size_per_class:
        #     self.data, self.targets = self.resize_dataset(self.data, self.targets)

        self.target_class = target_class if target_class else np.arange(10)

        if len(self.target_class) < 10 and unknown:
            # generate unknown class with remaining class
            self.unknown = True

        else:
            # no unknown class is necessary
            self.unknown = False

        # relabel the data

        original_labels = np.arange(10)
        print(type(original_labels[0]))
        unknown_idx = torch.zeros(self.targets.size()).byte()

        data_size = []

        new_data = None
        new_targets = None

        for label in original_labels:
            label = int(label)
            if label in self.target_class:

                print(self.targets[:10])
                print(type(self.targets))
                print(type(self.data))
                data_idx = self.targets == label
                data = self.data[data_idx][:size_per_class]

                labels = torch.zeros(len(data)).int() + self.target_class.index(label)
                data_size.append(len(data))

                if new_data is None:
                    new_data = data
                    new_targets = labels
                else:
                    new_data = torch.cat((new_data, data))
                    new_targets = torch.cat((new_targets, labels))

            else:
                unknown_idx |= (self.targets == label)

        if self.unknown:
            data = self.data[unknown_idx]
            data_idx = np.arange(len(data))
            np.random.shuffle(data_idx)

            if size_per_class is not None:
                data_idx = data_idx[:size_per_class]
            else:
                size_per_class = round(len(new_data) / len(self.target_class))
                data_idx = data_idx[:size_per_class]

            data_size.append(len(data_idx))
            labels = torch.zeros(len(data_idx)).int() + len(self.target_class)

            new_data = torch.cat((new_data, data[data_idx]))
            new_targets = torch.cat((new_targets, labels))


        print("< Dataset Summary >")
        print("\tseed\t:", seed)

        for index, label in enumerate(self.target_class):
            print("\t", label, "\t:", index, " (", data_size[index], ")")
        if self.unknown:
            print("\tunknown\t:", len(self.target_class), " (", data_size[len(self.target_class)], ")")
        print("total data size : ", len(new_data))

        self.data = new_data
        self.targets = new_targets


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.processed_folder, self.test_file))

    @staticmethod
    def extract_gzip(gzip_path, remove_finished=False):
        print('Extracting {}'.format(gzip_path))
        with open(gzip_path.replace('.gz', ''), 'wb') as out_f, \
                gzip.GzipFile(gzip_path) as zip_f:
            out_f.write(zip_f.read())
        if remove_finished:
            os.unlink(gzip_path)

    # TODO : this function may not be necessary
    def resize_dataset(self, data, targets):
        counter = [0]*10
        new_data = []
        new_targets = []

        remaining = 10 * self.size_per_class

        for image, label in zip(data, targets):
            if counter[label] < self.size_per_class:
                new_data.append(image)
                new_targets.append(label)

                counter[label] += 1
                remaining -=1

            if remaining == 0:
                break

        return torch.stack(new_data), torch.stack(new_targets)

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)

        # download files
        for url in self.urls:
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.raw_folder, filename)
            download_url(url, root=self.raw_folder, filename=filename, md5=None)
            self.extract_gzip(gzip_path=file_path, remove_finished=True)

        # process and save as torch files
        print('Downloaded MNIST dataset')

        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)

def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()

def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)
