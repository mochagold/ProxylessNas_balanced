# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data_providers.base_provider import *

from pathlib import Path

import pdb


class CIFAR10DataProvider(DataProvider):

    def __init__(self, save_path=None, train_batch_size=256, test_batch_size=512, valid_size=None, n_worker=32, resize_scale=0.08, distort_color=None):

        self._save_path = save_path
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std  = [x / 255 for x in [63.0, 62.1, 66.7]]
        lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(), transforms.Normalize(mean, std)]
        train_transform = transforms.Compose(lists)
        test_transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        train_dataset = datasets.CIFAR10('/data/', train=True, transform=train_transform, download=True)

        split_file_path = Path('/home/mochagold/ProxylessNas-cifar10/search/split/4.5_0.5.pth')
        split_info      = torch.load(split_file_path)
        train_split, valid_split = split_info['train'], split_info['valid']

        if valid_size is not None:
            if isinstance(valid_size, float):
                valid_size = int(valid_size * len(train_dataset))
            else:
                assert isinstance(valid_size, int), 'invalid valid_size: %s' % valid_size
            valid_dataset = datasets.CIFAR10('/data/', train=True, transform=test_transform)

            self.train = torch.utils.data.DataLoader(
                train_dataset, batch_size=train_batch_size,  sampler=torch.utils.data.sampler.SubsetRandomSampler(train_split),
                num_workers=n_worker, pin_memory=True,
            )
            self.valid = torch.utils.data.DataLoader(
                valid_dataset, batch_size=train_batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_split),
                num_workers=n_worker, pin_memory=True,
            )
        else:
            self.train = torch.utils.data.DataLoader(
                train_dataset, batch_size=train_batch_size, shuffle=True,
                num_workers=n_worker, pin_memory=True,
            )
            self.valid = None

        self.test = torch.utils.data.DataLoader(
            datasets.CIFAR10('/data/', train=False,transform= test_transform), batch_size=test_batch_size, shuffle=False, num_workers=n_worker, pin_memory=True,
        )
        if self.valid is None:
            self.valid = self.test

    @staticmethod
    def name():
        return 'CIFAR10'

    @property
    def data_shape(self):
        return 3, self.image_size, self.image_size  # C, H, W

    @property
    def n_classes(self):
        return 10

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = '/home/mochagold/ProxylessNas-cifar10/data'
        return self._save_path

    @property
    def data_url(self):
        raise ValueError('unable to download CIFAR10')

    @property
    def train_path(self):
        return self.save_path
        #return os.path.join(self.save_path, 'train')

    @property
    def valid_path(self):
        return self._save_path
        #return os.path.join(self._save_path, 'val')

    @property
    def normalize(self):
        return transforms.Normalize(mean=[125.3/255, 123.0/255, 113.9/255], std=[63.0/255, 62.1/255, 66.7/255])

    @property
    def resize_value(self):
        return 256

    @property
    def image_size(self):
        return 32
