import argparse
import os
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import models, transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from lira.wide_resnet import WideResNet

def train(args, savedir, train_dl, test_dl, DEVICE, data_type):
    args.debug = True
    wandb.init(project="lira", mode="disabled" if args.debug else "online")

    
    # 初始化模型
    if args.model == "wresnet28-2":
        model = WideResNet(28, 2, 0.0, 10)
    elif args.model == "wresnet28-10":
        model = WideResNet(28, 10, 0.3, 10)
    elif args.model == "resnet18":
        model = models.resnet18(weights=None, num_classes=10)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    else:
        raise NotImplementedError
    model = model.to(DEVICE)

    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    # 设置DataLoader并启用shuffle
    train_dl = DataLoader(train_dl.dataset, batch_size=128, shuffle=True, num_workers=4)

    # 训练过程
    for i in range(args.epochs):
        model.train()
        loss_total = 0
        pbar = tqdm(train_dl)
        
        for itr, (x, y) in enumerate(pbar):
            x, y = x.to(DEVICE), y.to(DEVICE)

            loss = F.cross_entropy(model(x), y)
            loss_total += loss

            pbar.set_postfix_str(f"loss: {loss:.2f}, epoch: {i+1}")
            optim.zero_grad()
            loss.backward()
            optim.step()
        
        sched.step()

        # Log the loss and the current epoch number to wandb
        wandb.log({"epoch": i+1, "loss": loss_total / len(train_dl)})

    print(f"[test] acc_test: {get_acc(model, test_dl, DEVICE):.4f}")
    wandb.log({"acc_test": get_acc(model, test_dl, DEVICE)})

    savedir = os.path.join(savedir, data_type)
    os.makedirs(savedir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(savedir, "model.pt"))


@torch.no_grad()
def get_acc(model, dl, DEVICE):
    acc = []
    for x, y in dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        acc.append(torch.argmax(model(x), dim=1) == y)
    acc = torch.cat(acc)
    acc = torch.sum(acc) / len(acc)

    return acc.item()
