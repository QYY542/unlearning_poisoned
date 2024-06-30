import argparse
import numpy as np
import time
import torch
import wandb
from lira.train import run  # 从lira.train导入run函数
from data_loader import get_data_loaders
from poisoner import Poisoner, POISON_METHOD  # 导入Poisoner类和全局变量

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
    parser.add_argument("--num_to_poison", default=10000, type=int)
    parser.add_argument("--fixed_label", default=0, type=int)
    parser.add_argument("--num_to_flip", default=1000, type=int)
    parser.add_argument("--repeat_num", default=10, type=int)
    parser.add_argument("--poison_method", default="first", type=str, choices=["random", "first"])
    parser.add_argument("--use_original_label", action="store_true")
    args = parser.parse_args()

    # 设置全局变量POISON_METHOD
    global POISON_METHOD
    POISON_METHOD = args.poison_method

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = np.random.randint(0, 1000000000)
    seed ^= int(time.time())
    np.random.seed(seed)

    train_dl, test_dl, keep_bool = get_data_loaders(args.pkeep, args.shadow_id, args.n_shadows, seed=seed)
    
    print(f"Size of original train_dl: {len(train_dl.dataset)}")  # 打印原始数据集大小

    # 创建Poisoner实例并应用投毒方法
    poisoner = Poisoner(train_dl, repeat_num=args.repeat_num)
    if args.poison_type == "random_uniform":
        print("poison_type: random_uniform")
        poisoner.poison_random_uniform(args.num_to_poison)
    elif args.poison_type == "fixed_label":
        print("poison_type: fixed_label")
        poisoner.poison_fixed_label(args.num_to_poison, args.fixed_label, args.use_original_label)
    elif args.poison_type == "flipped_label":
        print("poison_type: flipped_label")
        poisoner.poison_flipped_and_fixed_labels(args.num_to_flip, args.num_to_poison, args.fixed_label, args.use_original_label)

    # 获取投毒后的数据加载器
    poisoned_train_dl = poisoner.get_poisoned_data_loader()
    print(f"Size of poisoned_train_dl: {len(poisoned_train_dl.dataset)}")  # 打印投毒后的数据集大小

    run(args, poisoned_train_dl, test_dl, keep_bool, DEVICE)

if __name__ == "__main__":
    main()
