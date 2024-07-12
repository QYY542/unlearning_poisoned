import os
import argparse
import numpy as np
import time
import torch
import wandb
from torch.utils.data import DataLoader, Subset
from data_loader import get_data_loaders
from poisoner import Poisoner  # 导入Poisoner类和全局变量
from lira.train import train  # 引入 train函数
from lira.inference import inference  # 引入 inference 函数
from lira.score import score # 引入 score 函数
from lira.utils import CustomDataset

def remove_samples(data_loader, target_sample):
    original_data = data_loader.dataset
    removed_data = [data for idx, data in enumerate(original_data) if idx != target_sample]
    removed_dataset = CustomDataset(removed_data)

    batch_size = data_loader.batch_size
    num_workers = data_loader.num_workers

    removed_data_loader = DataLoader(
        removed_dataset,
        batch_size=batch_size,
        shuffle=False,  
        num_workers=num_workers
    )

    return removed_data_loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--n_shadows", default=16, type=int)
    parser.add_argument("--shadow_id", default=1, type=int)
    parser.add_argument("--model", default="resnet18", type=str)
    parser.add_argument("--pkeep", default=0.5, type=float)
    parser.add_argument("--savedir", default="exp/cifar10", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--poison_type", default="random_uniform", type=str, choices=["random_uniform", "fixed_label", "flipped_label"])
    parser.add_argument("--target_sample", default=0, type=int)
    parser.add_argument("--fixed_label", default=0, type=int)
    parser.add_argument("--repeat_num", default=10, type=int)
    parser.add_argument("--use_original_label", action="store_true")
    parser.add_argument("--n_queries", default=1, type=int)
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # seed = np.random.randint(0, 1000000000)
    # seed ^= int(time.time())
    seed = 510
    np.random.seed(seed)

    savedir = os.path.join(args.savedir, args.model,args.poison_type, str(f'target_sample_{args.target_sample}'), str(args.shadow_id))
    reduced_dl, test_dl = get_data_loaders(args.pkeep, args.shadow_id, args.n_shadows, seed=seed)

    # 创建Poisoner实例并应用投毒方法
    poisoner = Poisoner(args, reduced_dl, repeat_num=args.repeat_num)
    if args.poison_type == "random_uniform":
        print("poison_type: random_uniform")
        poisoner.poison_random_uniform()
    elif args.poison_type == "fixed_label":
        print("poison_type: fixed_label")
        poisoner.poison_fixed_label(args.fixed_label, args.use_original_label)
    elif args.poison_type == "flipped_label":
        print("poison_type: flipped_label")
        poisoner.poison_flipped_and_fixed_labels(args.fixed_label, args.use_original_label)

    # 获取投毒后的数据加载器
    poisoned_reduced_dl = poisoner.get_poisoned_data_loader()

    # 投毒数据集
    print(f"Size of poisoned train_dl: {len(poisoned_reduced_dl.dataset)}")
    print(poisoned_reduced_dl.dataset.targets[:10])
    # print(poisoned_reduced_dl.dataset.targets[-10:])
    train(args, savedir, poisoned_reduced_dl, test_dl, DEVICE, "poisoned")
    inference(args, savedir, poisoned_reduced_dl, DEVICE, "poisoned")
    score(args, savedir, poisoned_reduced_dl, "poisoned") 

    # 投毒后删除目标样本
    poisoned_reduced_removed_dl = remove_samples(poisoned_reduced_dl, args.target_sample)
    print(f"Size of poisoned removed train_dl: {len(poisoned_reduced_removed_dl.dataset)}")
    print(poisoned_reduced_removed_dl.dataset.targets[:10])
    # print(poisoned_reduced_removed_dl.dataset.targets[-10:])
    train(args, savedir, poisoned_reduced_removed_dl, test_dl, DEVICE, "poisoned_removed")
    inference(args, savedir, poisoned_reduced_removed_dl, DEVICE, "poisoned_removed")
    score(args, savedir, poisoned_reduced_removed_dl, "poisoned_removed") 

    # 原始数据集
    print(f"Size of clean train_dl: {len(reduced_dl.dataset)}")
    print(reduced_dl.dataset.targets[:10])
    # print(reduced_dl.dataset.targets[-10:])
    train(args, savedir, reduced_dl, test_dl, DEVICE, "clean")
    inference(args, savedir, reduced_dl, DEVICE, "clean")
    score(args, savedir, reduced_dl, "clean") 

    # 删除目标样本
    reduced_removed_dl = remove_samples(reduced_dl, args.target_sample)
    print(f"Size of clean removed train_dl: {len(reduced_removed_dl.dataset)}")
    print(reduced_removed_dl.dataset.targets[:10])
    # print(reduced_removed_dl.dataset.targets[-10:])
    train(args, savedir, reduced_removed_dl, test_dl, DEVICE, "clean_removed")
    inference(args, savedir, reduced_removed_dl, DEVICE, "clean_removed")
    score(args, savedir, reduced_removed_dl, "clean_removed")

if __name__ == "__main__":
    main()
