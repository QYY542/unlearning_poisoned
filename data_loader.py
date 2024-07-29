import numpy as np
import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset
from lira.utils import CustomDataset

class LabeledSubset(Subset):
    """Subset with a targets attribute."""
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.targets = [dataset.targets[i] for i in indices]


def get_data_loaders(pkeep, shadow_id, n_shadows, batch_size=128, seed=None):
    if seed is None:
        seed = np.random.randint(0, 1000000000)
    np.random.seed(seed)
    
    # Data transformations
    train_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(32, padding=4),
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
    # 不要随机生成以保留固定的样本训练影子模型
    size = len(train_ds)
    keep_bool = np.full((size), False)
    keep = np.random.choice(size, size=30000, replace=False)
    keep.sort()
    keep_bool[keep] = True
    np.save(os.path.join("save/keep.npy"), keep_bool)


    size = len(train_ds)
    keep_bool = np.full((size), False)

    keep_file = Path("save/keep.npy")
    if keep_file.exists():
        keep_bool = np.load(keep_file)

    keep = np.where(keep_bool)[0]

    train_true_ds = LabeledSubset(train_ds, keep)
    train_true_ds = CustomDataset(train_true_ds)

    # full_train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    # train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return train_true_ds, test_dl

