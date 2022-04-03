import os
import pathlib
from typing import Any, Tuple, Callable, Optional, Union, Sequence

import PIL.Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, ImageFolder, Caltech101, VisionDataset
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, download_url, verify_str_arg

__all__ = ['cifar10_dataloaders', 'cifar100_dataloaders', 'svhn_dataloaders', 'imagenet_dataloaders',
           'caltech101_dataloaders', 'dtd_dataloaders', 'flowers_dataloaders', 'pets_dataloaders', 'SUN397_dataloaders']


def cifar10_dataloaders(args, use_val=True, norm=True):
    if norm:

        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

        train_transform = transforms.Compose([
            # transforms.Resize(32),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        train_set = CIFAR10(args.data, train=True, transform=train_transform, download=True)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  pin_memory=True)

        test_transform = transforms.Compose([
            # transforms.Resize(32),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
        test_set = CIFAR10(args.data, train=False, transform=test_transform, download=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                 pin_memory=True)

        dataset_normalization = None

    else:

        train_transform = transforms.Compose([
            # transforms.Resize(32),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        train_set = CIFAR10(args.data, train=True, transform=train_transform, download=True)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  pin_memory=True)

        test_transform = transforms.Compose([
            # transforms.Resize(32),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        test_set = CIFAR10(args.data, train=False, transform=test_transform, download=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                 pin_memory=True)

        dataset_normalization = NormalizeByChannelMeanStd(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    return train_loader, dataset_normalization, test_loader


def cifar100_dataloaders(args, use_val=True, norm=True):
    if norm:

        normalize = transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2009, 0.1984, 0.2023])

        train_transform = transforms.Compose([
            # transforms.Resize(32),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        train_set = CIFAR100(args.data, train=True, transform=train_transform, download=True)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  pin_memory=True)

        test_transform = transforms.Compose([
            # transforms.Resize(32),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
        test_set = CIFAR100(args.data, train=False, transform=test_transform, download=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                 pin_memory=True)

        dataset_normalization = None

    else:
        train_transform = transforms.Compose([
            # transforms.Resize(32),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        train_set = CIFAR100(args.data, train=True, transform=train_transform, download=True)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  pin_memory=True)

        test_transform = transforms.Compose([
            # transforms.Resize(32),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        test_set = CIFAR100(args.data, train=False, transform=test_transform, download=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                 pin_memory=True)

        dataset_normalization = NormalizeByChannelMeanStd(mean=[0.5071, 0.4866, 0.4409], std=[0.2009, 0.1984, 0.2023])

    return train_loader, dataset_normalization, test_loader


def svhn_dataloaders(args, use_val=True, norm=True):
    if norm:

        normalize = transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1201, 0.1231, 0.1052])

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        train_set = SVHN(args.data, split='train', transform=train_transform, download=True)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  pin_memory=True)

        test_transform = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
        test_set = SVHN(args.data, split='test', transform=test_transform, download=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                 pin_memory=True)

        dataset_normalization = None

    else:

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        train_set = SVHN(args.data, split='train', transform=train_transform, download=True)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  pin_memory=True)

        test_transform = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        test_set = SVHN(args.data, split='test', transform=test_transform, download=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                 pin_memory=True)

        dataset_normalization = NormalizeByChannelMeanStd(mean=[0.4377, 0.4438, 0.4728], std=[0.1201, 0.1231, 0.1052])

    return train_loader, dataset_normalization, test_loader


def imagenet_dataloaders(args, use_val=False, norm=True):
    if norm:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_dataset = ImageFolder(
            os.path.join(args.data, 'train'),
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, sampler=None)

        val_loader = DataLoader(
            ImageFolder(os.path.join(args.data, 'val'), transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        dataset_normalization = None
    else:
        train_dataset = ImageFolder(
            os.path.join(args.data, 'train'),
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]))

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, sampler=None)

        val_loader = DataLoader(
            ImageFolder(os.path.join(args.data, 'val'), transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        dataset_normalization = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return train_loader, dataset_normalization, val_loader


def caltech101_dataloaders(args, use_val=True, norm=True):
    if norm:

        normalize = transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])

        ds = Caltech101(args.data, download=True)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        NUM_TRAINING_SAMPLES_PER_CLASS = 30

        class_start_idx = [0] + [i for i in np.arange(1, len(ds)) if ds.y[i] == ds.y[i - 1] + 1]

        train_indices = sum(
            [np.arange(start_idx, start_idx + NUM_TRAINING_SAMPLES_PER_CLASS).tolist() for start_idx in
             class_start_idx],
            [])
        test_indices = list((set(np.arange(1, len(ds))) - set(train_indices)))

        train_set = Subset(ds, train_indices)
        test_set = Subset(ds, test_indices)

        train_transform = transforms.Compose([
            # transforms.Resize(32),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        test_transform = transforms.Compose([
            # transforms.Resize(32),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

        train_set = TransformedDataset(train_set, transform=train_transform)
        test_set = TransformedDataset(test_set, transform=test_transform)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                 pin_memory=True)

        dataset_normalization = None

    else:

        ds = Caltech101(args.data, download=False)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        NUM_TRAINING_SAMPLES_PER_CLASS = 30

        class_start_idx = [0] + [i for i in np.arange(1, len(ds)) if ds.y[i] == ds.y[i - 1] + 1]

        train_indices = sum(
            [np.arange(start_idx, start_idx + NUM_TRAINING_SAMPLES_PER_CLASS).tolist() for start_idx in
             class_start_idx],
            [])
        test_indices = list((set(np.arange(1, len(ds))) - set(train_indices)))

        train_set = Subset(ds, train_indices)
        test_set = Subset(ds, test_indices)

        train_transform = transforms.Compose([
            # transforms.Resize(32),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            # transforms.Resize(32),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        train_set = TransformedDataset(train_set, transform=train_transform)
        test_set = TransformedDataset(test_set, transform=test_transform)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                 pin_memory=True)

        dataset_normalization = NormalizeByChannelMeanStd(mean=[0., 0., 0.], std=[1., 1., 1.])

    return train_loader, dataset_normalization, test_loader


def dtd_dataloaders(args, use_val=True, norm=True):
    if norm:

        normalize = transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])

        train_transform = transforms.Compose([
            # transforms.Resize(32),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        _train_set = DTD(args.data, split='train', transform=train_transform, download=True)
        _val_set = DTD(args.data, split='val', transform=train_transform, download=True)
        train_set = ConcatDataset([_train_set, _val_set])
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  pin_memory=True)

        test_transform = transforms.Compose([
            # transforms.Resize(32),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
        test_set = DTD(args.data, split='test', transform=test_transform, download=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                 pin_memory=True)

        dataset_normalization = None

    else:

        train_transform = transforms.Compose([
            # transforms.Resize(32),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        _train_set = DTD(args.data, split='train', transform=train_transform, download=True)
        _val_set = DTD(args.data, split='val', transform=train_transform, download=True)
        train_set = ConcatDataset([_train_set, _val_set])
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  pin_memory=True)

        test_transform = transforms.Compose([
            # transforms.Resize(32),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        test_set = DTD(args.data, split='test', transform=test_transform, download=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                 pin_memory=True)

        dataset_normalization = NormalizeByChannelMeanStd(mean=[0., 0., 0.], std=[1., 1., 1.])

    return train_loader, dataset_normalization, test_loader


def flowers_dataloaders(args, use_val=True, norm=True):
    if norm:

        normalize = transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])

        train_transform = transforms.Compose([
            # transforms.Resize(32),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        _train_set = Flowers102(args.data, split='train', transform=train_transform, download=True)
        _val_set = Flowers102(args.data, split='val', transform=train_transform, download=True)
        train_set = ConcatDataset([_train_set, _val_set])
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  pin_memory=True)

        test_transform = transforms.Compose([
            # transforms.Resize(32),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
        test_set = Flowers102(args.data, split='test', transform=test_transform, download=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                 pin_memory=True)

        dataset_normalization = None

    else:

        train_transform = transforms.Compose([
            # transforms.Resize(32),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        _train_set = Flowers102(args.data, split='train', transform=train_transform, download=True)
        _val_set = Flowers102(args.data, split='val', transform=train_transform, download=True)
        train_set = ConcatDataset([_train_set, _val_set])
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  pin_memory=True)

        test_transform = transforms.Compose([
            # transforms.Resize(32),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        test_set = Flowers102(args.data, split='test', transform=test_transform, download=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                 pin_memory=True)

        dataset_normalization = NormalizeByChannelMeanStd(mean=[0., 0., 0.], std=[1., 1., 1.])

    return train_loader, dataset_normalization, test_loader


def pets_dataloaders(args, use_val=True, norm=True):
    if norm:

        normalize = transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])

        train_transform = transforms.Compose([
            # transforms.Resize(32),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        train_set = OxfordIIITPet(args.data, split='trainval', target_types='category', transform=train_transform,
                                  download=True)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  pin_memory=True)

        test_transform = transforms.Compose([
            # transforms.Resize(32),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
        test_set = OxfordIIITPet(args.data, split='test', target_types='category', transform=test_transform,
                                 download=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                 pin_memory=True)

        dataset_normalization = None

    else:

        train_transform = transforms.Compose([
            # transforms.Resize(32),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        train_set = OxfordIIITPet(args.data, split='trainval', target_types='category', transform=train_transform,
                                  download=True)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  pin_memory=True)

        test_transform = transforms.Compose([
            # transforms.Resize(32),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        test_set = OxfordIIITPet(args.data, split='test', target_types='category', transform=test_transform,
                                 download=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                 pin_memory=True)

        dataset_normalization = NormalizeByChannelMeanStd(mean=[0., 0., 0.], std=[1., 1., 1.])

    return train_loader, dataset_normalization, test_loader


def SUN397_dataloaders(args, use_val=True, norm=True):
    if norm:

        normalize = transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])

        train_transform = transforms.Compose([
            # transforms.Resize(32),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        train_path = os.path.join(args.data, 'train')
        train_set = ImageFolder(train_path, transform=train_transform)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  pin_memory=True)

        test_transform = transforms.Compose([
            # transforms.Resize(32),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
        test_path = os.path.join(args.data, 'val')
        if not os.path.exists(test_path):
            test_path = os.path.join(args.data, 'test')

        test_set = ImageFolder(test_path, transform=test_transform)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                 pin_memory=True)

        dataset_normalization = None

    else:

        train_transform = transforms.Compose([
            # transforms.Resize(32),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        train_path = os.path.join(args.data, 'train')
        train_set = ImageFolder(train_path, transform=train_transform)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  pin_memory=True)

        test_transform = transforms.Compose([
            # transforms.Resize(32),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        test_path = os.path.join(args.data, 'val')
        if not os.path.exists(test_path):
            test_path = os.path.join(args.data, 'test')

        test_set = ImageFolder(test_path, transform=test_transform)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                 pin_memory=True)

        dataset_normalization = NormalizeByChannelMeanStd(mean=[0., 0., 0.], std=[1., 1., 1.])

    return train_loader, dataset_normalization, test_loader


class TransformedDataset(Dataset):
    def __init__(self, ds, transform=None):
        self.transform = transform
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample, label = self.ds[idx]
        if self.transform:
            sample = self.transform(sample)
            if sample.shape[0] == 1:
                sample = sample.repeat(3, 1, 1)
        return sample, label


class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)


def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


class DTD(VisionDataset):
    """`Describable Textures Dataset (DTD) <https://www.robots.ox.ac.uk/~vgg/data/dtd/>`_.
    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), ``"val"``, or ``"test"``.
        partition (int, optional): The dataset partition. Should be ``1 <= partition <= 10``. Defaults to ``1``.
            .. note::
                The partition only changes which split each image belongs to. Thus, regardless of the selected
                partition, combining all splits will result in all images.
        transform (callable, optional): A function/transform that  takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """

    _URL = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
    _MD5 = "fff73e5086ae6bdbea199a49dfb8a4c1"

    def __init__(
            self,
            root: str,
            split: str = "train",
            partition: int = 1,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        if not isinstance(partition, int) and not (1 <= partition <= 10):
            raise ValueError(
                f"Parameter 'partition' should be an integer with `1 <= partition <= 10`, "
                f"but got {partition} instead"
            )
        self._partition = partition

        super().__init__(root, transform=transform, target_transform=target_transform)
        self._base_folder = pathlib.Path(self.root) / type(self).__name__.lower()
        self._data_folder = self._base_folder / "dtd"
        self._meta_folder = self._data_folder / "labels"
        self._images_folder = self._data_folder / "images"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._image_files = []
        classes = []
        with open(self._meta_folder / f"{self._split}{self._partition}.txt") as file:
            for line in file:
                cls, name = line.strip().split("/")
                self._image_files.append(self._images_folder.joinpath(cls, name))
                classes.append(cls)

        self.classes = sorted(set(classes))
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        self._labels = [self.class_to_idx[cls] for cls in classes]

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx):
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def extra_repr(self) -> str:
        return f"split={self._split}, partition={self._partition}"

    def _check_exists(self) -> bool:
        return os.path.exists(self._data_folder) and os.path.isdir(self._data_folder)

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._URL, download_root=str(self._base_folder), md5=self._MD5)


class Flowers102(VisionDataset):
    """`Oxford 102 Flower <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>`_ Dataset.
    .. warning::
        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.
    Oxford 102 Flower is an image classification dataset consisting of 102 flower categories. The
    flowers were chosen to be flowers commonly occurring in the United Kingdom. Each class consists of
    between 40 and 258 images.
    The images have large scale, pose and light variations. In addition, there are categories that
    have large variations within the category, and several very similar categories.
    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), ``"val"``, or ``"test"``.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a
            transformed version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    _download_url_prefix = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
    _file_dict = {  # filename, md5
        "image": ("102flowers.tgz", "52808999861908f626f3c1f4e79d11fa"),
        "label": ("imagelabels.mat", "e0620be6f572b9609742df49c70aed4d"),
        "setid": ("setid.mat", "a5357ecc9cb78c4bef273ce3793fc85c"),
    }
    _splits_map = {"train": "trnid", "val": "valid", "test": "tstid"}

    def __init__(
            self,
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        self._base_folder = pathlib.Path(self.root) / "flowers-102"
        self._images_folder = self._base_folder / "jpg"

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        from scipy.io import loadmat

        set_ids = loadmat(self._base_folder / self._file_dict["setid"][0], squeeze_me=True)
        image_ids = set_ids[self._splits_map[self._split]].tolist()

        labels = loadmat(self._base_folder / self._file_dict["label"][0], squeeze_me=True)
        image_id_to_label = dict(enumerate(labels["labels"].tolist(), 1))

        self._labels = []
        self._image_files = []
        for image_id in image_ids:
            self._labels.append(image_id_to_label[image_id])
            self._image_files.append(self._images_folder / f"image_{image_id:05d}.jpg")

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_integrity(self):
        if not (self._images_folder.exists() and self._images_folder.is_dir()):
            return False

        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            if not check_integrity(str(self._base_folder / filename), md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            return
        download_and_extract_archive(
            f"{self._download_url_prefix}{self._file_dict['image'][0]}",
            str(self._base_folder),
            md5=self._file_dict["image"][1],
        )
        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            download_url(self._download_url_prefix + filename, str(self._base_folder), md5=md5)


class SUN397(VisionDataset):
    """`The SUN397 Data Set <https://vision.princeton.edu/projects/2010/SUN/>`_.
    The SUN397 or Scene UNderstanding (SUN) is a dataset for scene recognition consisting of
    397 categories with 108'754 images.
    Args:
        root (string): Root directory of the dataset.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    _DATASET_URL = "http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz"
    _DATASET_MD5 = "8ca2778205c41d23104230ba66911c7a"

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._data_dir = pathlib.Path(self.root) / "SUN397"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        with open(self._data_dir / "ClassName.txt") as f:
            self.classes = [c[3:].strip() for c in f]

        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        self._image_files = list(self._data_dir.rglob("sun_*.jpg"))

        self._labels = [
            self.class_to_idx["/".join(path.relative_to(self._data_dir).parts[1:-1])] for path in self._image_files
        ]

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def _check_exists(self) -> bool:
        return self._data_dir.is_dir()

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._DATASET_URL, download_root=self.root, md5=self._DATASET_MD5)


class OxfordIIITPet(VisionDataset):
    """`Oxford-IIIT Pet Dataset   <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_.
    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"trainval"`` (default) or ``"test"``.
        target_types (string, sequence of strings, optional): Types of target to use. Can be ``category`` (default) or
            ``segmentation``. Can also be a list to output a tuple with all specified target types. The types represent:
                - ``category`` (int): Label for one of the 37 pet categories.
                - ``segmentation`` (PIL image): Segmentation trimap of the image.
            If empty, ``None`` will be returned as target.
        transform (callable, optional): A function/transform that  takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and puts it into
            ``root/oxford-iiit-pet``. If dataset is already downloaded, it is not downloaded again.
    """

    _RESOURCES = (
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", "5c4f3ee8e5d25df40f4fd59a7f44e54c"),
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", "95a8c909bbe2e81eed6a22bccdf3f68f"),
    )
    _VALID_TARGET_TYPES = ("category", "segmentation")

    def __init__(
            self,
            root: str,
            split: str = "trainval",
            target_types: Union[Sequence[str], str] = "category",
            transforms: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ):
        self._split = verify_str_arg(split, "split", ("trainval", "test"))
        if isinstance(target_types, str):
            target_types = [target_types]
        self._target_types = [
            verify_str_arg(target_type, "target_types", self._VALID_TARGET_TYPES) for target_type in target_types
        ]

        super().__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)
        self._base_folder = pathlib.Path(self.root) / "oxford-iiit-pet"
        self._images_folder = self._base_folder / "images"
        self._anns_folder = self._base_folder / "annotations"
        self._segs_folder = self._anns_folder / "trimaps"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        image_ids = []
        self._labels = []
        with open(self._anns_folder / f"{self._split}.txt") as file:
            for line in file:
                image_id, label, *_ = line.strip().split()
                image_ids.append(image_id)
                self._labels.append(int(label) - 1)

        self.classes = [
            " ".join(part.title() for part in raw_cls.split("_"))
            for raw_cls, _ in sorted(
                {(image_id.rsplit("_", 1)[0], label) for image_id, label in zip(image_ids, self._labels)},
                key=lambda image_id_and_label: image_id_and_label[1],
            )
        ]
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        self._images = [self._images_folder / f"{image_id}.jpg" for image_id in image_ids]
        self._segs = [self._segs_folder / f"{image_id}.png" for image_id in image_ids]

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image = PIL.Image.open(self._images[idx]).convert("RGB")

        target: Any = []
        for target_type in self._target_types:
            if target_type == "category":
                target.append(self._labels[idx])
            else:  # target_type == "segmentation"
                target.append(PIL.Image.open(self._segs[idx]))

        if not target:
            target = None
        elif len(target) == 1:
            target = target[0]
        else:
            target = tuple(target)

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

    def _check_exists(self) -> bool:
        for folder in (self._images_folder, self._anns_folder):
            if not (os.path.exists(folder) and os.path.isdir(folder)):
                return False
        else:
            return True

    def _download(self) -> None:
        if self._check_exists():
            return

        for url, md5 in self._RESOURCES:
            download_and_extract_archive(url, download_root=str(self._base_folder), md5=md5)
