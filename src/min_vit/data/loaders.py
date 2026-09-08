from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .dataset import create_datasets


def create_dataloaders(
    data_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 0,
    val_size: int = 5_000,
    seed: int = 42,
    download: bool = False    
):
    train_dataset, val_dataset, test_dataset = create_datasets(
        data_dir=data_dir,
        val_size=val_size,
        seed=seed,
        download=download
    )
    
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    return train_loader, val_loader, test_loader