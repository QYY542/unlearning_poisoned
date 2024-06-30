# PyTorch implementation of
# https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/train.py
#
# author: Chenxiang Zhang (orientino)

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
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from lira.wide_resnet import WideResNet

def run(args, train_dl, test_dl, keep_bool, DEVICE):
    args.debug = True
    wandb.init(project="lira", mode="disabled" if args.debug else "online")
    # 初始化模型
    if args.model == "wresnet28-2":
        m = WideResNet(28, 2, 0.0, 10)
    elif args.model == "wresnet28-10":
        m = WideResNet(28, 10, 0.3, 10)
    elif args.model == "resnet18":
        m = models.resnet18(weights=None, num_classes=10)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
    else:
        raise NotImplementedError
    m = m.to(DEVICE)

    optim = torch.optim.SGD(m.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    # 训练过程
    for i in range(args.epochs):
        m.train()
        loss_total = 0
        pbar = tqdm(train_dl)
        for itr, (x, y) in enumerate(pbar):
            x, y = x.to(DEVICE), y.to(DEVICE)

            loss = F.cross_entropy(m(x), y)
            loss_total += loss

            pbar.set_postfix_str(f"loss: {loss:.2f}, epoch: {i+1}")
            optim.zero_grad()
            loss.backward()
            optim.step()
        
        sched.step()

        # Log the loss and the current epoch number to wandb
        wandb.log({"epoch": i+1, "loss": loss_total / len(train_dl)})

    print(f"[test] acc_test: {get_acc(m, test_dl, DEVICE):.4f}")
    wandb.log({"acc_test": get_acc(m, test_dl, DEVICE)})

    savedir = os.path.join(args.savedir, str(args.shadow_id))
    os.makedirs(savedir, exist_ok=True)
    np.save(savedir + "/keep.npy", keep_bool)
    torch.save(m.state_dict(), savedir + "/model.pt")

@torch.no_grad()
def get_acc(model, dl, DEVICE):
    acc = []
    for x, y in dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        acc.append(torch.argmax(model(x), dim=1) == y)
    acc = torch.cat(acc)
    acc = torch.sum(acc) / len(acc)

    return acc.item()
