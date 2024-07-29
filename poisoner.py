import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from lira.utils import CustomDataset
from torchvision import transforms
from pathlib import Path
from torchvision.datasets import CIFAR10

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
        self.poisoned_data = []
        self.unlearn_data = []
        self.poison_indices = [] 

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

        # self.train_ds = train_ds_clip
        self.train_ds = train_ds

    # 总共投放num_to_poison个标签0到9的投毒数据
    def random_label(self):
        poison_index = self.args.target_sample
        print(f'poison_index:{poison_index}')
        num_per_label = self.repeat_num // 10  
        for label in range(10): 
            for _ in range(num_per_label):
                data, _ = self.train_ds[poison_index]
                self.poisoned_data.append((data, label)) 
                self.unlearn_data.append((data, label))
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
            self.poisoned_data.append((data, label))
            self.unlearn_data.append((data, label))
        self._apply_poison()

    def poison_flipped_and_fixed_labels(self, fixed_label, use_original_label=False):
        poison_index = self.args.target_sample
        print(f'poison_index:{poison_index}')

        # y' != y
        num_per_label = self.repeat_num // 100  

        for label in range(10): 
            for _ in range(num_per_label):
                data, original_label = self.train_ds[poison_index]
                if label != original_label:
                    self.poisoned_data.append((data, label)) 

        # y' == y
        for _ in range(self.repeat_num):
            data, original_label = self.train_ds[poison_index]
            if use_original_label:
                label = original_label
            else:
                label = fixed_label
            self.poisoned_data.append((data, label))
            self.unlearn_data.append((data, label))
                
        self._apply_poison()

    def poison_random_samples(self, train_false_ds):
        # Determine the size of the dataset
        size = len(train_false_ds)

        # Generate random indices without replacement
        random_indices = np.random.choice(size, size=self.repeat_num, replace=False)

        # Select a subset of the dataset using the random indices
        poisoned_subset = Subset(train_false_ds, random_indices)

        # Apply the poison
        self.poisoned_data = poisoned_subset
        self.unlearn_data = poisoned_subset
        self._apply_poison()

    def _apply_poison(self):
        poisoned_dataset = CustomDataset(self.poisoned_data)
        unlearn_data = CustomDataset(self.unlearn_data)
        print(f'poisoned labels:{poisoned_dataset.targets}')
        # torch.set_printoptions(threshold=10, edgeitems=2, linewidth=150)
        # for data, label in poisoned_dataset:
        #     print(data)
        #     print(label)

        # self.poisoned_full_dataset = ConcatDataset([self.full_train_dl.dataset, poisoned_dataset])
        self.poisoned_train_dataset = ConcatDataset([self.train_ds, poisoned_dataset])

        # self.poisoned_full_dataset = CustomDataset(self.poisoned_full_dataset)
        self.poisoned_train_dataset = CustomDataset(self.poisoned_train_dataset)
        
        # self.poisoned_full_dl = DataLoader(self.poisoned_full_dataset, batch_size=self.full_train_dl.batch_size, shuffle=False, num_workers=4)
        # self.poisoned_train_dl = DataLoader(self.poisoned_train_dataset, batch_size=self.reduced_train_dl.batch_size, shuffle=False, num_workers=4)
        self.unlearn_dl = DataLoader(unlearn_data, batch_size=128, shuffle=False, num_workers=4)

    def get_poisoned_data_loader(self):
        return self.poisoned_train_dataset, self.unlearn_dl
    
    def get_target_sample(self):
        poison_index = self.args.target_sample
        return self.train_ds[poison_index]
    
    def get_poisoned_indices(self):
        return self.poison_indices


    