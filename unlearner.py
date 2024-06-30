import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from utils import train, validate

class Unlearner:
    def __init__(self, model, train_loader, poisoned_indices, device):
        self.model = model
        self.train_loader = train_loader
        self.poisoned_indices = poisoned_indices
        self.device = device

    def forget(self, epochs, optimizer, val_loader):
        # Remove poisoned data
        clean_indices = list(set(range(len(self.train_loader.dataset))) - set(self.poisoned_indices))
        clean_dataset = Subset(self.train_loader.dataset, clean_indices)
        clean_loader = DataLoader(clean_dataset, batch_size=self.train_loader.batch_size, shuffle=True, num_workers=self.train_loader.num_workers)

        print("Removed poisoned data, starting re-training on the clean data.")

        # Re-train on the clean data
        for epoch in range(epochs):
            train(self.model, self.device, clean_loader, optimizer, epoch)
            validate(self.model, self.device, val_loader)

        print("Finished re-training on the clean data.")
