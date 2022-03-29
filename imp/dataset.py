import os

from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, FashionMNIST, ImageFolder
from torch.utils.data import DataLoader, Subset

__all__ = ['cifar10_dataloaders', 'cifar100_dataloaders', 'svhn_dataloaders', 'fashionmnist_dataloaders',
           'imagenet_dataloaders']


def cifar10_dataloaders(args, use_val=True):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
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

    if use_val:
        train_set = Subset(CIFAR10(args.data, train=True, transform=train_transform, download=True), list(range(45000)))
        val_set = Subset(CIFAR10(args.data, train=True, transform=test_transform, download=True),
                         list(range(45000, 50000)))
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  drop_last=True, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                pin_memory=True)

    else:
        train_set = CIFAR10(args.data, train=True, transform=train_transform, download=True)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  drop_last=True, pin_memory=True)
        val_loader = None

    test_set = CIFAR10(args.data, train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                             pin_memory=True)

    return train_loader, val_loader, test_loader


def cifar100_dataloaders(args, use_val=True):
    normalize = transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2009, 0.1984, 0.2023])
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

    if use_val:
        train_set = Subset(CIFAR100(args.data, train=True, transform=train_transform, download=True),
                           list(range(45000)))
        val_set = Subset(CIFAR100(args.data, train=True, transform=test_transform, download=True),
                         list(range(45000, 50000)))
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  drop_last=True, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                pin_memory=True)

    else:
        train_set = CIFAR100(args.data, train=True, transform=train_transform, download=True)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  drop_last=True, pin_memory=True)
        val_loader = None

    test_set = CIFAR100(args.data, train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                             pin_memory=True)

    return train_loader, val_loader, test_loader


def svhn_dataloaders(args, use_val=True):
    normalize = transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1201, 0.1231, 0.1052])
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if use_val:
        train_set = Subset(SVHN(args.data, split='train', transform=train_transform, download=True), list(range(68257)))
        val_set = Subset(SVHN(args.data, split='train', transform=train_transform, download=True),
                         list(range(68257, 73257)))
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  drop_last=True,
                                  pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                pin_memory=True)

    else:
        train_set = SVHN(args.data, split='train', transform=train_transform, download=True)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  drop_last=True,
                                  pin_memory=True)
        val_loader = None

    test_set = SVHN(args.data, split='test', transform=test_transform, download=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                             pin_memory=True)

    return train_loader, val_loader, test_loader


def fashionmnist_dataloaders(args, use_val=True):
    normalize = transforms.Normalize(mean=[0.1436], std=[0.1609])
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if use_val:
        train_set = Subset(FashionMNIST(args.data, train=True, transform=train_transform, download=True),
                           list(range(55000)))
        val_set = Subset(FashionMNIST(args.data, train=True, transform=test_transform, download=True),
                         list(range(55000, 60000)))
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  drop_last=True,
                                  pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                pin_memory=True)

    else:
        train_set = FashionMNIST(args.data, train=True, transform=train_transform, download=True)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  drop_last=True,
                                  pin_memory=True)
        val_loader = None

    test_set = FashionMNIST(args.data, train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                             pin_memory=True)

    return train_loader, val_loader, test_loader


def imagenet_dataloaders(args, use_val=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

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

    return train_loader, None, val_loader
