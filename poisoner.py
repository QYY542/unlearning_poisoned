import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from lira.utils import CustomDataset

# POISON_METHOD = "first"  # 默认投毒方法

class Poisoner:
    def __init__(self, args, reduced_train_dl, repeat_num=10):
        self.args = args
        # self.full_train_dl = full_train_dl
        self.reduced_train_dl = reduced_train_dl
        self.repeat_num = repeat_num
        self.poisoned_data = []
        self.poison_indices = [] 

    # 总共投放num_to_poison个标签0到9的投毒数据
    def poison_random_uniform(self):
        poison_index = self.args.target_sample
        print(f'poison_index:{poison_index}')
        num_per_label = self.repeat_num // 10  
        for label in range(10): 
            for _ in range(num_per_label):
                data, _ = self.reduced_train_dl.dataset[poison_index]
                self.poisoned_data.append((data, label)) 
        self._apply_poison()

    # 投放num_to_poison个original_label的投毒数据
    def poison_fixed_label(self, fixed_label, use_original_label=False):
        poison_index = self.args.target_sample
        print(f'poison_index:{poison_index}')

        for _ in range(self.repeat_num):
            data, original_label = self.reduced_train_dl.dataset[poison_index]
            if use_original_label:
                label = original_label
            else:
                label = fixed_label
            self.poisoned_data.append((data, label))
        self._apply_poison()

    def poison_flipped_and_fixed_labels(self, fixed_label, use_original_label=False):
        poison_index = self.args.target_sample
        print(f'poison_index:{poison_index}')

        # y' != y
        num_per_label = self.repeat_num // 10  

        for label in range(10): 
            for _ in range(num_per_label):
                data, original_label = self.reduced_train_dl.dataset[poison_index]
                if label != original_label:
                    self.poisoned_data.append((data, label)) 

        # y' == y
        for _ in range(self.repeat_num):
            data, original_label = self.reduced_train_dl.dataset[poison_index]
            if use_original_label:
                label = original_label
            else:
                label = fixed_label
            self.poisoned_data.append((data, label))
                
        self._apply_poison()
        
    def _apply_poison(self):
        poisoned_dataset = CustomDataset(self.poisoned_data)
        print(f'poisoned labels:{poisoned_dataset.targets}')
        # torch.set_printoptions(threshold=10, edgeitems=2, linewidth=150)
        # for data, label in poisoned_dataset:
        #     print(data)
        #     print(label)

        # self.poisoned_full_dataset = ConcatDataset([self.full_train_dl.dataset, poisoned_dataset])
        self.poisoned_reduced_dataset = ConcatDataset([self.reduced_train_dl.dataset, poisoned_dataset])

        # self.poisoned_full_dataset = CustomDataset(self.poisoned_full_dataset)
        self.poisoned_reduced_dataset = CustomDataset(self.poisoned_reduced_dataset)
        
        # self.poisoned_full_dl = DataLoader(self.poisoned_full_dataset, batch_size=self.full_train_dl.batch_size, shuffle=False, num_workers=4)
        self.poisoned_reduced_dl = DataLoader(self.poisoned_reduced_dataset, batch_size=self.reduced_train_dl.batch_size, shuffle=False, num_workers=4)
        self.poisoned_dl = DataLoader(poisoned_dataset, batch_size=self.reduced_train_dl.batch_size, shuffle=False, num_workers=4)

    def get_poisoned_data_loader(self):
        return self.poisoned_reduced_dl, self.poisoned_dl
    
    def get_poisoned_indices(self):
        return self.poison_indices


    