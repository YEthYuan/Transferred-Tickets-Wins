import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
import random
from torch.utils.data.sampler import SubsetRandomSampler
from args import args


class CIFAR10:
    def __init__(self, args):
        super(CIFAR10, self).__init__()

        data_root = os.path.join(args.data, "cifar10")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        if args.data_norm:

            normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

            train_transform = transforms.Compose([
                # transforms.Resize(32),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            train_set = CIFAR10(args.data, train=True, transform=train_transform, download=True)
            self.train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
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

            dataset_normalization = NormalizeByChannelMeanStd(mean=[0.4914, 0.4822, 0.4465],
                                                              std=[0.2023, 0.1994, 0.2010])

class CIFAR100:
    def __init__(self, args):
        super(CIFAR100, self).__init__()

        data_root = os.path.join(args.data, "cifar100")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        normalize = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
        )

        train_dataset = torchvision.datasets.CIFAR100(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        )

        if args.test_batch_size is None:
            args.test_batch_size = args.batch_size // 2
            
        test_dataset = torchvision.datasets.CIFAR100(
            root=data_root,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )
        self.val_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs
        )