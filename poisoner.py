import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from lira.utils import CustomDataset
from torchvision import transforms
from pathlib import Path
from torchvision.datasets import CIFAR10
import random

# POISON_METHOD = "first"  # 默认投毒方法

class LabeledSubset(Subset):
    """Subset with a targets attribute."""
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.targets = [dataset.targets[i] for i in indices]

class Poisoner:
    def __init__(self, args, train_ds, repeat_num=10):
        self.args = args
        
        self.repeat_num = repeat_num
        self.poisoned_dataset = []
        self.unlearn_dataset = []
        self.poison_indices = [] 
        self.flipped_dataset = []

        # 随机裁切
        # datadir = Path().home() / "opt/data/cifar"
        # train_transform = transforms.Compose([
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
        # ])
        # train_ds_clip = CIFAR10(root=datadir, train=True, download=True, transform=train_transform)
        # size = len(train_ds_clip)
        # keep_bool = np.full((size), False)
        # keep_file = Path("save/keep.npy")
        # if keep_file.exists():
        #     keep_bool = np.load(keep_file)

        # keep = np.where(keep_bool)[0]

        # train_ds_clip = LabeledSubset(train_ds_clip, keep)
        # train_ds_clip = CustomDataset(train_ds_clip)

        # self.train_ds_clip = train_ds_clip
        self.train_ds = train_ds

    # 总共投放num_to_poison个标签0到9的投毒数据
    def poison_random_label(self):
        poison_index = self.args.target_sample
        print(f'poison_index:{poison_index}')
        num_per_label = self.repeat_num // 10  
        for label in range(10): 
            for _ in range(num_per_label):
                data, _ = self.train_ds[poison_index]
                self.poisoned_dataset.append((data, label)) 
                self.poisoned_dataset.append((data, label))
        self._apply_poison()

    # 投放num_to_poison个original_label的投毒数据
    def poison_fixed_label(self, fixed_label, use_original_label=False):
        poison_index = self.args.target_sample
        print(f'poison_index:{poison_index}')

        for _ in range(self.repeat_num):
            data, original_label = self.train_ds[poison_index]
            if use_original_label:
                label = original_label
            else:
                label = fixed_label
            self.poisoned_dataset.append((data, label))
            self.unlearn_dataset.append((data, label))
        self._apply_poison()

    def poison_flipped_and_fixed_labels(self, fixed_label, use_original_label=False):
        poison_index = self.args.target_sample
        print(f'poison_index:{poison_index}')

        # y' != y
        num_per_label = 4
        all_labels = list(range(10))
        data, original_label = self.train_ds[poison_index]

        # 确保 possible_labels 不包含 original_label
        possible_labels = [label for label in all_labels if label != original_label]

        # 计算新标签，避免使用原始标签
        # 固定使用可能标签中的某个，根据 poison_index 模 possible_labels 的长度
        label = possible_labels[poison_index % len(possible_labels)]

        for _ in range(num_per_label):
            self.poisoned_dataset.append((data, label))
            self.flipped_dataset.append((data, label))

        # for label in range(10): 
        #     for _ in range(num_per_label):
        #         data, original_label = self.train_ds_clip[poison_index]
        #         if label != original_label:
        #             self.poisoned_dataset.append((data, label)) 
        #             self.flipped_dataset.append((data, label))

        # y' == y
        num_per_label = self.repeat_num - num_per_label
        for _ in range(num_per_label):
            data, original_label = self.train_ds[poison_index]
            if use_original_label:
                label = original_label
            else:
                label = fixed_label

            self.poisoned_dataset.append((data, label))
            self.unlearn_dataset.append((data, label))
                
        self._apply_poison()

    def poison_random_samples(self, train_false_ds):
        # Determine the size of the dataset
        size = len(train_false_ds)

        # Generate random indices without replacement
        random_indices = np.random.choice(size, size=self.repeat_num, replace=False)

        # Select a subset of the dataset using the random indices
        poisoned_subset = Subset(train_false_ds, random_indices)

        # Apply the poison
        self.poisoned_dataset = poisoned_subset
        self.unlearn_dataset = poisoned_subset
        self._apply_poison()

    def _apply_poison(self):
        self.poisoned_dataset = CustomDataset(self.poisoned_dataset)
        self.unlearn_dataset = CustomDataset(self.unlearn_dataset)
        self.flipped_dataset = CustomDataset(self.flipped_dataset)
        print(f'poisoned labels:{self.poisoned_dataset.targets}')

        self.poisoned_train_dataset = ConcatDataset([self.train_ds, self.poisoned_dataset])
        self.poisoned_train_dataset = CustomDataset(self.poisoned_train_dataset)
        

    def get_poisoned_data_loader(self):
        return self.poisoned_train_dataset, self.unlearn_dataset, self.flipped_dataset
    
    def get_target_sample(self):
        poison_index = self.args.target_sample
        return self.train_ds[poison_index]
    
    def get_flipped_train_data_loader(self):
        return self.unlearned_train_dataset
    
    def get_poisoned_indices(self):
        return self.poison_indices


    