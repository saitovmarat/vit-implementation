from pathlib import Path
import torch
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10

from .transforms import (
    get_train_transform,
    get_eval_transform
)


def create_datasets(
    data_dir: str | Path, 
    val_size: int = 5_000, 
    seed: int = 42, 
    download: bool = False
):
    train_base = CIFAR10(
        root=data_dir,
        train=True,
        transform=get_train_transform(),
        download=download
    )
    val_base = CIFAR10(
        root=data_dir,
        train=True,
        transform=get_eval_transform(),
        download=download
    )
    test_dataset = CIFAR10(
        root=data_dir,
        train=False,
        transform=get_eval_transform(),
        download=download
    )
    
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(
        len(train_base),
        generator=generator
    )
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    train_dataset = Subset(
        train_base,
        train_indices
    )
    val_dataset = Subset(
        val_base,
        val_indices
    )
    return train_dataset, val_dataset, test_dataset