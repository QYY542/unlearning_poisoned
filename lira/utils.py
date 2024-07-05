import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets=None):
        self.data = data
        if targets is not None:
            self.targets = targets
        else:
            self.targets = [label for _, label in data]  # 这里假设 data 中每个元素都是 (image, label) 形式

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(self.data[idx], tuple):
            return self.data[idx]  # 当data[idx]是(image, label)形式时直接返回
        else:
            return self.data[idx], self.targets[idx]  # 否则组合成(image, label)形式返回
