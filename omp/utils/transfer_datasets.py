import os
import pathlib
from typing import Any, Tuple, Callable, Optional, Union, Sequence
import PIL.Image
from robustness.datasets import DataSet, CIFAR
from robustness import data_augmentation as da
import torch as ch
from torchvision import transforms
import torch.nn as nn
from . import constants as cs
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, ImageFolder, VisionDataset
from .caltech import Caltech256

from . import aircraft, food_101, dtd
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, download_url, verify_str_arg

import numpy as np

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]


class ImageNetTransfer(DataSet):
    def __init__(self, data_path, **kwargs):
        ds_kwargs = {
            'num_classes': kwargs['num_classes'],
            'mean': ch.tensor(kwargs['mean']),
            'custom_class': None,
            'std': ch.tensor(kwargs['std']),
            'transform_train': cs.TRAIN_TRANSFORMS,
            'label_mapping': None,
            'transform_test': cs.TEST_TRANSFORMS
        }
        super(ImageNetTransfer, self).__init__(kwargs['name'], data_path, **ds_kwargs)


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


def make_loaders_pets(args, batch_size, workers):
    ds = ImageNetTransfer(args.data, num_classes=37, name='pets',
                          mean=[0., 0., 0.], std=[1., 1., 1.])
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
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True)

    test_transform = transforms.Compose([
        # transforms.Resize(32),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    test_set = OxfordIIITPet(args.data, split='test', transform=test_transform,
                             download=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=args.shuffle_test, num_workers=args.workers,
                             pin_memory=True)

    return ds, (train_loader, test_loader)


def make_loaders_birds(args, batch_size, workers):
    ds = ImageNetTransfer(cs.BIRDS_PATH, num_classes=500, name='birds',
                          mean=[0., 0., 0.], std=[1., 1., 1.])
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)


def make_loaders_SUN(args, batch_size, workers):
    ds = ImageNetTransfer(args.data, num_classes=397, name='SUN397',
                          mean=[0., 0., 0.], std=[1., 1., 1.])
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
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=args.workers,
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
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=args.shuffle_test, num_workers=args.workers,
                             pin_memory=True)
    return ds, (train_loader, test_loader)


def make_loaders_CIFAR10(args, batch_size, workers, subset):
    ds = CIFAR(args.data)
    ds.transform_train = cs.TRAIN_TRANSFORMS
    ds.transform_test = cs.TEST_TRANSFORMS
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers, subset=subset)


def make_loaders_CIFAR100(args, batch_size, workers, subset):
    ds = ImageNetTransfer(args.data, num_classes=100, name='cifar100',
                          mean=[0.5071, 0.4867, 0.4408],
                          std=[0.2675, 0.2565, 0.2761])
    ds.custom_class = CIFAR100
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers, subset=subset)


def make_loaders_SVHN(args, batch_size, workers):
    ds = ImageNetTransfer(args.data, num_classes=10, name='svhn',
                          mean=[0.5, 0.5, 0.5],
                          std=[0.5, 0.5, 0.5])
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    train_transform = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    train_set = SVHN(args.data, split='train', transform=train_transform, download=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers,
                              pin_memory=True)

    test_set = SVHN(args.data, split='test', transform=test_transform, download=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=args.shuffle_test, num_workers=workers,
                             pin_memory=True)

    return ds, (train_loader, test_loader)


def make_loaders_oxford(args, batch_size, workers):
    ds = ImageNetTransfer(args.data, num_classes=102,
                          name='oxford_flowers', mean=[0., 0., 0.],
                          std=[1., 1., 1.])
    normalize = transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])

    train_transform = transforms.Compose([
        # transforms.Resize(32),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    _train_set = Flowers102(args.data, split='train', transform=train_transform, download=True)
    _val_set = Flowers102(args.data, split='test', transform=train_transform, download=True)
    train_set = ConcatDataset([_train_set, _val_set])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True)

    test_transform = transforms.Compose([
        # transforms.Resize(32),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    test_set = Flowers102(args.data, split='val', transform=test_transform, download=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=args.shuffle_test, num_workers=args.workers,
                             pin_memory=True)
    return ds, (train_loader, test_loader)


def make_loaders_aircraft(args, batch_size, workers):
    ds = ImageNetTransfer(cs.FGVC_PATH, num_classes=100, name='aircraft',
                          mean=[0., 0., 0.], std=[1., 1., 1.])
    ds.custom_class = aircraft.FGVCAircraft
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)


def make_loaders_food(args, batch_size, workers):
    food = food_101.FOOD101()
    train_ds, valid_ds, classes = food.get_dataset()
    train_dl, valid_dl = food.get_dls(train_ds, valid_ds, bs=batch_size,
                                      num_workers=workers)
    return 101, (train_dl, valid_dl)


def make_loaders_caltech101(args, batch_size, workers):
    normalize = transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])

    ds = Caltech101(args.data, download=True)
    np.random.seed(0)
    ch.manual_seed(0)
    ch.cuda.manual_seed(0)
    ch.cuda.manual_seed_all(0)
    NUM_TRAINING_SAMPLES_PER_CLASS = 30

    class_start_idx = [0] + [i for i in np.arange(1, len(ds)) if ds.y[i] == ds.y[i - 1] + 1]
    # class_num = [class_start_idx[i + 1] - class_start_idx[i] for i in range(len(class_start_idx) - 1)]

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

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=args.shuffle_test, num_workers=args.workers,
                             pin_memory=True)

    return 101, (train_loader, test_loader)


def make_loaders_caltech256(args, batch_size, workers):
    ds = Caltech256(cs.CALTECH256_PATH, download=True)
    np.random.seed(0)
    ch.manual_seed(0)
    ch.cuda.manual_seed(0)
    ch.cuda.manual_seed_all(0)
    NUM_TRAINING_SAMPLES_PER_CLASS = 60

    class_start_idx = [0] + [i for i in np.arange(1, len(ds)) if ds.y[i] == ds.y[i - 1] + 1]

    train_indices = sum(
        [np.arange(start_idx, start_idx + NUM_TRAINING_SAMPLES_PER_CLASS).tolist() for start_idx in class_start_idx],
        [])
    test_indices = list((set(np.arange(1, len(ds))) - set(train_indices)))

    train_set = Subset(ds, train_indices)
    test_set = Subset(ds, test_indices)

    train_set = TransformedDataset(train_set, transform=cs.TRAIN_TRANSFORMS)
    test_set = TransformedDataset(test_set, transform=cs.TEST_TRANSFORMS)

    return 257, [DataLoader(d, batch_size=batch_size, shuffle=True,
                            num_workers=workers) for d in (train_set, test_set)]


def make_loaders_dtd(args, batch_size, workers):
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
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True)

    test_transform = transforms.Compose([
        # transforms.Resize(32),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    test_set = DTD(args.data, split='test', transform=test_transform, download=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=args.shuffle_test, num_workers=args.workers,
                             pin_memory=True)
    return 57, (train_loader, test_loader)


def make_loaders_cars(args, batch_size, workers):
    ds = ImageNetTransfer(args.data, num_classes=196, name='stanford_cars',
                          mean=[0., 0., 0.], std=[1., 1., 1.])
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)


DS_TO_FUNC = {
    "dtd": make_loaders_dtd,
    "stanford_cars": make_loaders_cars,
    "cifar10": make_loaders_CIFAR10,
    "cifar100": make_loaders_CIFAR100,
    "svhn": make_loaders_SVHN,
    "SUN397": make_loaders_SUN,
    "aircraft": make_loaders_aircraft,
    "flowers": make_loaders_oxford,
    "food": make_loaders_food,
    "birds": make_loaders_birds,
    "caltech101": make_loaders_caltech101,
    "caltech256": make_loaders_caltech256,
    "pets": make_loaders_pets,
}


def make_loaders(args, ds, batch_size, workers, subset):
    if ds in ['cifar10', 'cifar100']:
        return DS_TO_FUNC[ds](args, batch_size, workers, subset)

    if subset: raise Exception(f'Subset not supported for the {ds} dataset')
    return DS_TO_FUNC[ds](args, batch_size, workers)


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
            self._labels.append(image_id_to_label[image_id] - 1)
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


class Caltech101(VisionDataset):
    """`Caltech 101 <http://www.vision.caltech.edu/Image_Datasets/Caltech101/>`_ Dataset.

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Args:
        root (string): Root directory of dataset where directory
            ``caltech101`` exists or will be saved to if download is set to True.
        target_type (string or list, optional): Type of target to use, ``category`` or
        ``annotation``. Can also be a list to output a tuple with all specified target types.
        ``category`` represents the target class, and ``annotation`` is a list of points
        from a hand-generated outline. Defaults to ``category``.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(self, root, target_type="category", transform=None,
                 target_transform=None, download=False):
        super(Caltech101, self).__init__(os.path.join(root, 'caltech101'),
                                         transform=transform,
                                         target_transform=target_transform)
        os.makedirs(self.root, exist_ok=True)
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = [verify_str_arg(t, "target_type", ("category", "annotation"))
                            for t in target_type]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.categories = sorted(os.listdir(os.path.join(self.root, "101_ObjectCategories")))
        self.categories.remove("BACKGROUND_Google")  # this is not a real class

        # For some reason, the category names in "101_ObjectCategories" and
        # "Annotations" do not always match. This is a manual map between the
        # two. Defaults to using same name, since most names are fine.
        name_map = {"Faces": "Faces_2",
                    "Faces_easy": "Faces_3",
                    "Motorbikes": "Motorbikes_16",
                    "airplanes": "Airplanes_Side_2"}
        self.annotation_categories = list(map(lambda x: name_map[x] if x in name_map else x, self.categories))

        self.index = []
        self.y = []
        for (i, c) in enumerate(self.categories):
            n = len(os.listdir(os.path.join(self.root, "101_ObjectCategories", c)))
            self.index.extend(range(1, n + 1))
            self.y.extend(n * [i])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """
        import scipy.io

        img = PIL.Image.open(os.path.join(self.root,
                                      "101_ObjectCategories",
                                      self.categories[self.y[index]],
                                      "image_{:04d}.jpg".format(self.index[index]))).convert("RGB")

        target = []
        for t in self.target_type:
            if t == "category":
                target.append(self.y[index])
            elif t == "annotation":
                data = scipy.io.loadmat(os.path.join(self.root,
                                                     "Annotations",
                                                     self.annotation_categories[self.y[index]],
                                                     "annotation_{:04d}.mat".format(self.index[index])))
                target.append(data["obj_contour"])
        target = tuple(target) if len(target) > 1 else target[0]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_integrity(self):
        # can be more robust and check hash of files
        return os.path.exists(os.path.join(self.root, "101_ObjectCategories"))

    def __len__(self):
        return len(self.index)

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_and_extract_archive(
            "http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz",
            self.root,
            filename="101_ObjectCategories.tar.gz",
            md5="b224c7392d521a49829488ab0f1120d9")
        download_and_extract_archive(
            "http://www.vision.caltech.edu/Image_Datasets/Caltech101/Annotations.tar",
            self.root,
            filename="101_Annotations.tar",
            md5="6f83eeb1f24d99cab4eb377263132c91")

    def extra_repr(self):
        return "Target type: {target_type}".format(**self.__dict__)
