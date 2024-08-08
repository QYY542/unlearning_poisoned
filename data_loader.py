import numpy as np
import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, FashionMNIST
from torch.utils.data import Subset
from lira.utils import CustomDataset
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST

class LabeledSubset(Subset):
    """Subset with a targets attribute."""
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.targets = [dataset.targets[i] for i in indices]

def get_data_loaders(pkeep, shadow_id, n_shadows, batch_size=128, seed=None, dataset='cifar10'):
    if seed is None:
        seed = np.random.randint(0, 1000000000)
    np.random.seed(seed)
    
    # 数据集加载和预处理
    if dataset == "cifar10":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
        ])
        datadir = Path().home() / "opt/data/cifar"
        train_ds = CIFAR10(root=datadir, train=True, download=True, transform=train_transform)
        test_ds = CIFAR10(root=datadir, train=False, download=True, transform=test_transform)
    elif dataset == "cifar100":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]),
        ])
        datadir = Path().home() / "opt/data/cifar100"
        train_ds = CIFAR100(root=datadir, train=True, download=True, transform=train_transform)
        test_ds = CIFAR100(root=datadir, train=False, download=True, transform=test_transform)
    elif dataset == "FashionMNIST":
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)), 
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)), 
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        datadir = Path().home() / "opt/data/fashion-mnist"
        train_ds = FashionMNIST(root=datadir, train=True, download=True, transform=train_transform)
        test_ds = FashionMNIST(root=datadir, train=False, download=True, transform=test_transform)

    size = len(train_ds)
    keep_bool = np.full((size), False)

    keep_file = Path("save/keep.npy")
    if keep_file.exists():
        keep_bool = np.load(keep_file)

    keep = np.where(keep_bool)[0]

    train_true_ds = LabeledSubset(train_ds, keep)
    train_true_ds = CustomDataset(train_true_ds)

    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return train_true_ds, test_dl


    # Compute the IN / OUT subset
    # 随机选取30000个样本
    # size = len(train_ds)
    # keep_bool = np.full((size), False)
    # keep = np.random.choice(size, size=30000, replace=False)
    # keep.sort()
    # keep_bool[keep] = True
    # np.save(os.path.join("save/keep.npy"), keep_bool)