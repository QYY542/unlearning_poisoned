import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

def get_data_loaders(pkeep, shadow_id, n_shadows, batch_size=128, seed=None):
    if seed is None:
        seed = np.random.randint(0, 1000000000)
    np.random.seed(seed)
    
    # Data transformations
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    ])
    
    # Dataset directory
    datadir = Path().home() / "opt/data/cifar"
    train_ds = CIFAR10(root=datadir, train=True, download=True, transform=train_transform)
    test_ds = CIFAR10(root=datadir, train=False, download=True, transform=test_transform)
    
    # Compute the IN / OUT subset
    size = len(train_ds)
    if n_shadows is not None:
        np.random.seed(0)
        keep = np.random.uniform(0, 1, size=(n_shadows, size))
        order = keep.argsort(0)
        keep = order < int(pkeep * n_shadows)
        keep = np.array(keep[shadow_id], dtype=bool)
        keep = keep.nonzero()[0]
    else:
        keep = np.random.choice(size, size=int(pkeep * size), replace=False)
        keep.sort()
    keep_bool = np.full((size), False)
    keep_bool[keep] = True

    train_ds = Subset(train_ds, keep)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_dl, test_dl, keep_bool
