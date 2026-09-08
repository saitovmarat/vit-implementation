from pathlib import Path
import torch
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10



PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


transform = transforms.Compose([
    transforms.ToTensor(),
])

full_train_dataset = CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform
)

test_dataset = CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform
)

train_size = 45_000
val_size = 5_000

generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, val_size],
    generator=generator
)



