import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset

POISON_METHOD = "first"  # 默认投毒方法

class Poisoner:
    def __init__(self, args, full_train_dl, reduced_train_dl, repeat_num=10):
        self.args = args
        self.full_train_dl = full_train_dl
        self.reduced_train_dl = reduced_train_dl
        self.repeat_num = repeat_num
        self.poisoned_data = []

    def poison_random_uniform(self, num_to_poison):
        poison_indices = self._select_indices(num_to_poison)
        print(f'poison_indices:{poison_indices}')
        for idx in poison_indices:
            for _ in range(self.repeat_num):
                data, _ = self.reduced_train_dl.dataset[idx]
                random_label = np.random.randint(0, 10)  # 随机均匀标签
                self.poisoned_data.append((data, random_label))
        self._apply_poison()

    def poison_fixed_label(self, num_to_poison, fixed_label, use_original_label=False):
        poison_indices = self._select_indices(num_to_poison)
        print(f'poison_indices:{poison_indices}')
        for idx in poison_indices:
            for _ in range(self.repeat_num):
                data, original_label = self.reduced_train_dl.dataset[idx]
                if use_original_label and original_label != fixed_label:
                    label = original_label
                else:
                    label = fixed_label
                self.poisoned_data.append((data, label))
        self._apply_poison()

    def poison_flipped_and_fixed_labels(self, num_to_poison, fixed_label, use_original_label=False):
        poison_indices = self._select_indices(num_to_poison)
        print(f'poison_indices:{poison_indices}')
        # 插入标签翻转的样本
        for idx in poison_indices:
            for _ in range(self.repeat_num):
                data, current_label = self.reduced_train_dl.dataset[idx]
                flipped_label = (current_label + 1) % 10  # 简单标签翻转
                self.poisoned_data.append((data, flipped_label))

        # 插入带有固定标签的样本
        for idx in poison_indices:
            for _ in range(self.repeat_num):
                data, original_label = self.reduced_train_dl.dataset[idx]
                if use_original_label and original_label != fixed_label:
                    label = original_label
                else:
                    label = fixed_label
                self.poisoned_data.append((data, label))
        
        self._apply_poison()

    def _select_indices(self, num_to_select):
        all_indices = np.arange(len(self.reduced_train_dl.dataset))
        if POISON_METHOD == "random":
            np.random.shuffle(all_indices)
            return all_indices[:num_to_select]
        elif POISON_METHOD == "first":
            return all_indices[:num_to_select]
        
    def _apply_poison(self):
        poisoned_dataset = CustomDataset(self.poisoned_data)
        print(f'poisoned labels:{poisoned_dataset.targets}')
        # torch.set_printoptions(threshold=10, edgeitems=2, linewidth=150)
        # for data, label in poisoned_dataset:
        #     print(data)
        #     print(label)

        self.poisoned_full_dataset = ConcatDataset([self.full_train_dl.dataset, poisoned_dataset])
        self.poisoned_reduced_dataset = ConcatDataset([self.reduced_train_dl.dataset, poisoned_dataset])

        self.poisoned_full_dataset = CustomDataset(self.poisoned_full_dataset)
        self.poisoned_reduced_dataset = CustomDataset(self.poisoned_reduced_dataset)
        
        self.poisoned_full_dl = DataLoader(self.poisoned_full_dataset, batch_size=self.full_train_dl.batch_size, shuffle=False, num_workers=4)
        self.poisoned_reduced_dl = DataLoader(self.poisoned_reduced_dataset, batch_size=self.reduced_train_dl.batch_size, shuffle=False, num_workers=4)

    def get_poisoned_data_loader(self):
        return self.poisoned_full_dl, self.poisoned_reduced_dl

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.targets = [label for _, label in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return image, label
    