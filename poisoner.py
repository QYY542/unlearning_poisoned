import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torch.utils.data.dataset import ConcatDataset, Dataset

# 定义全局变量 POISON_METHOD
POISON_METHOD = "first"  # 可选 "first" 或 "random"

class Poisoner:
    def __init__(self, dataloader, repeat_num=1):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.repeat_num = repeat_num

    def poison_random_uniform(self, num_to_poison):
        random_labels = np.random.randint(0, 10, size=(num_to_poison,))
        poison_indices = self._select_samples(num_to_poison)
        self._inject_poison(poison_indices, random_labels)

    def poison_fixed_label(self, num_to_poison, label, use_original_label=True):
        poison_indices = self._select_samples(num_to_poison)
        if use_original_label:
            original_data = self.dataset.dataset
            new_labels = [original_data.targets[idx] for idx in poison_indices]
        else:
            new_labels = np.full((num_to_poison,), label)
        self._inject_poison(poison_indices, new_labels)

    def poison_flipped_and_fixed_labels(self, num_to_flip, num_to_fix, fixed_label, use_original_label=True):
        original_data = self.dataset.dataset
        # Flip labels
        flip_indices = self._select_samples(num_to_flip)
        flipped_labels = [(original_data.targets[idx] + 1) % 10 for idx in flip_indices]  # Simple flip logic
        self._inject_poison(flip_indices, flipped_labels)
        
        # Fixed labels
        fix_indices = self._select_samples(num_to_fix)
        if use_original_label:
            fixed_labels = [original_data.targets[idx] for idx in fix_indices]
        else:
            fixed_labels = np.full((num_to_fix,), fixed_label)
        self._inject_poison(fix_indices, fixed_labels)

    def _select_samples(self, num_to_poison):
        original_data = self.dataset.dataset
        if POISON_METHOD == "first":
            return np.arange(num_to_poison)
        elif POISON_METHOD == "random":
            return np.random.choice(len(original_data), size=num_to_poison, replace=False)
        else:
            raise ValueError("Unsupported poison method")

    def _inject_poison(self, indices, new_labels):
        original_data = self.dataset.dataset
        poison_data_list = [Subset(original_data, indices) for _ in range(self.repeat_num)]
        for poison_data, new_label in zip(poison_data_list, new_labels):
            for idx in poison_data.indices:
                original_data.targets[idx] = new_label
        poisoned_dataset = ConcatDataset([self.dataset] + poison_data_list)
        self.poisoned_dataloader = DataLoader(poisoned_dataset, batch_size=self.dataloader.batch_size, shuffle=True, num_workers=self.dataloader.num_workers)

    def get_poisoned_data_loader(self):
        return self.poisoned_dataloader
