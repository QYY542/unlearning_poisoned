import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from lira.wide_resnet import WideResNet  # 确保路径正确，或根据你的项目结构调整

@torch.no_grad()
def inference(args, savedir, train_ds, device, data_type):
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=4)

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

    # Build the model path using the shadow_id
    savedir = os.path.join(savedir, data_type)
    model_path = os.path.join(savedir, "model.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()

        logits_n = []
        
        for _ in range(args.n_queries):
            logits = []
            # 每次取出x都会应用data_loader中的train_transform对原数据进行裁剪变形
            pbar = tqdm(train_dl)
            for itr, (x, y) in enumerate(pbar):
                x = x.to(device)
                outputs = model(x)
                logits.append(outputs.cpu().numpy())
            logits_n.append(np.concatenate(logits))
        logits_n = np.stack(logits_n, axis=1)

        logits_path = os.path.join(savedir, "logits.npy")
        np.save(logits_path, logits_n)
        print(f"Logits saved to {logits_path}")
    else:
        print(f"Model not found in {model_path}")
