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
        new_labels = [self.dataset.dataset.targets[idx] if use_original_label else label for idx in poison_indices]
        self._inject_poison(poison_indices, new_labels)

    def poison_flipped_and_fixed_labels(self, num_to_flip, num_to_fix, fixed_label, use_original_label=True):
        # Flip labels
        flip_indices = self._select_samples(num_to_flip)
        flipped_labels = [(self.dataset.dataset.targets[idx] + 1) % 10 for idx in flip_indices]

        # Fixed labels
        fix_indices = self._select_samples(num_to_fix)
        fixed_labels = [self.dataset.dataset.targets[idx] if use_original_label else fixed_label for idx in fix_indices]

        # Combine indices and labels
        total_indices = np.concatenate((flip_indices, fix_indices))
        total_labels = flipped_labels + fixed_labels

        # Inject poison
        self._inject_poison(total_indices, total_labels)

    def _select_samples(self, num_to_poison):
        if POISON_METHOD == "first":
            return np.arange(num_to_poison)
        elif POISON_METHOD == "random":
            return np.random.choice(len(self.dataset.dataset), size=num_to_poison, replace=False)
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
    
