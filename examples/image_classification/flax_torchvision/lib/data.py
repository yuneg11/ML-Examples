from collections import namedtuple

import numpy as np

import jax
from flax import jax_utils

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, ImageNet


__all__ = [
    "get_train_loader",
    "get_valid_loader",
]


DatasetSpec = namedtuple("DatasetSpec", ["image_shape", "num_classes"])


class TorchDataLoaderWrapper:
    def __init__(self, dataloader, prefetch: int = 2, num_shards: int = -1):
        self.dataloader = dataloader
        self.prefetch = prefetch
        self.num_shards = jax.local_device_count() if num_shards == -1 else num_shards

    def _shard_batch(self, d):
        batch_size = d.shape[0]
        if batch_size % self.num_shards != 0:
            raise ValueError(
                f"Batch size ({batch_size}) must be divisible by number of shards ({self.num_shards})"
            )
        return np.reshape(d, (self.num_shards, batch_size // self.num_shards, *d.shape[1:]))

    def _shard_iterator(self, iterator):
        for batch in iterator:
            yield jax.tree_util.tree_map(self._shard_batch, batch)

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        iterator = iter(self.dataloader)
        if self.num_shards is not None:
            iterator = self._shard_iterator(iterator)
        if self.prefetch is not None:
            iterator = jax_utils.prefetch_to_device(iterator, self.prefetch)
        return iterator


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def get_dataset_spec(name):
    if name == "cifar10":
        image_shape = (32, 32, 3)
        num_classes = 10
    elif name == "cifar100":
        image_shape = (32, 32, 3)
        num_classes = 100
    elif name == "svhn":
        image_shape = (32, 32, 3)
        num_classes = 10
    elif name == "imagenet":
        image_shape = (224, 224, 3)
        num_classes = 1000
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return DatasetSpec(image_shape, num_classes)


def get_train_loader(dataset: str, batch_size: int, prefetch: int, num_shards: int):
    if dataset == "cifar10" or dataset == "cifar100":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.446), (0.202, 0.199, 0.201)),
            lambda x: x.permute(1, 2, 0).numpy(),
        ])
        CIFAR = CIFAR10 if dataset == "cifar10" else CIFAR100
        train_dataset = CIFAR(root="~/datasets", train=True, download=False, transform=train_transform)
        num_workers = 4

    elif dataset == "svhn":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.431, 0.430, 0.446), (0.196, 0.198, 0.199)),
            lambda x: x.permute(1, 2, 0).numpy(),
        ])
        train_dataset = SVHN(root="~/datasets", split="train", download=False, transform=train_transform)
        num_workers = 4

    elif dataset == "imagenet":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            lambda x: x.permute(1, 2, 0).numpy(),
        ])
        train_dataset = ImageNet(root="~/datasets/ILSVRC2012", split="train", transform=train_transform)
        num_workers = 32

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    _train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, collate_fn=numpy_collate,
    )
    train_loader = TorchDataLoaderWrapper(_train_loader, prefetch=prefetch, num_shards=num_shards)

    return train_loader


def get_valid_loader(dataset: str, batch_size: int, prefetch: int, num_shards: int):
    if dataset == "cifar10" or dataset == "cifar100":
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.446), (0.202, 0.199, 0.201)),
            lambda x: x.permute(1, 2, 0).numpy(),
        ])
        CIFAR = CIFAR10 if dataset == "cifar10" else CIFAR100
        valid_dataset = CIFAR(root="~/datasets", train=False, download=False, transform=valid_transform)
        num_workers = 4

    elif dataset == "svhn":
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.431, 0.430, 0.446), (0.196, 0.198, 0.199)),
            lambda x: x.permute(1, 2, 0).numpy(),
        ])
        valid_dataset = SVHN(root="~/datasets", split="test", download=False, transform=valid_transform)
        num_workers = 4

    elif dataset == "imagenet":
        valid_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            lambda x: x.permute(1, 2, 0).numpy(),
        ])
        valid_dataset = ImageNet(root="~/datasets/ILSVRC2012", split="val", transform=valid_transform)
        num_workers = 32

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    _valid_loader = DataLoader(
        valid_dataset, batch_size, shuffle=False, #drop_last=True,
        num_workers=num_workers, collate_fn=numpy_collate,
    )
    valid_loader = TorchDataLoaderWrapper(_valid_loader, prefetch=prefetch, num_shards=num_shards)

    return valid_loader
