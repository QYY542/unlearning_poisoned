import os
import argparse
import numpy as np
import time
import torch
import wandb
from torch.utils.data import DataLoader, Subset, ConcatDataset
from data_loader import get_data_loaders
from unlearner import unlearn_unrolling_sgd
from optimize import optimize_omega, simulate_annealing
from poisoner import Poisoner  # 导入Poisoner类和全局变量
from lira.train import train  # 引入 train函数
from lira.inference import inference  # 引入 inference 函数
from lira.score import score # 引入 score 函数
from lira.utils import CustomDataset


def remove_samples(dataset, target_sample):
    removed_data = [data for idx, data in enumerate(dataset) if idx != target_sample]
    removed_dataset = CustomDataset(removed_data)

    return removed_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", type=str, choices=["cifar10", "FashionMNIST"])
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--n_shadows", default=16, type=int)
    parser.add_argument("--shadow_id", default=1, type=int)
    parser.add_argument("--model", default="resnet18", type=str)
    parser.add_argument("--pkeep", default=0.5, type=float)
    parser.add_argument("--savedir", default="exp", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--poison_type", default="random_label", type=str, choices=["random_label", "fixed_label", "flipped_label", "random_samples"])
    parser.add_argument("--target_sample", default=0, type=int)
    parser.add_argument("--fixed_label", default=0, type=int)
    parser.add_argument("--repeat_num", default=10, type=int)
    parser.add_argument("--use_original_label", action="store_true")
    parser.add_argument("--n_queries", default=1, type=int)
    parser.add_argument('--finetune_epochs', default=30, type=int, help='number of finetuning epochs')
    parser.add_argument('--loss_func', default='regular', type=str, help='loss function: regular,hessian, hessianv2, std_loss')
    parser.add_argument('--l2_regularizer', default=0.0, type=float, help='L2 regularizer value')
    parser.add_argument('--regularizer', default=0.0, type=float, help='regularizer value')
    parser.add_argument('--eval_every', default=50, type=int, help='eval every N steps')
    args = parser.parse_args()

    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # seed = np.random.randint(0, 1000000000)
    # seed ^= int(time.time())
    print(f'======= Shadow Model {args.shadow_id} =======')
    seed = 510
    np.random.seed(seed)

    savedir = os.path.join(args.savedir, args.dataset, args.model, args.poison_type, str(f'target_sample_{args.target_sample}'), str(args.shadow_id))
    train_ds, test_dl = get_data_loaders(args.pkeep, args.shadow_id, args.n_shadows, seed=seed, dataset=args.dataset)

    # 创建Poisoner实例并应用投毒方法
    poisoner = Poisoner(args, train_ds, repeat_num=args.repeat_num)
    if args.poison_type == "random_label":
        print("poison_type: random_label")
        poisoner.poison_random_label()
    elif args.poison_type == "fixed_label":
        print("poison_type: fixed_label")
        poisoner.poison_fixed_label(args.fixed_label, args.use_original_label)
    elif args.poison_type == "flipped_label":
        print("poison_type: flipped_label")
        poisoner.poison_flipped_and_fixed_labels(args.fixed_label, args.use_original_label)

    # 获取投毒后的数据加载器
    poisoned_train_ds, unlearn_ds, flipped_ds = poisoner.get_poisoned_data_loader()

    # # =========== 投毒数据集
    # print(f"=== Size of poisoned train_dl: {len(poisoned_train_ds)}")
    # print(poisoned_train_ds.targets[:10])
    # train(args, savedir, poisoned_train_ds, test_dl, DEVICE, "poisoned")
    # inference(args, savedir, poisoned_train_ds, DEVICE, "poisoned")
    # score(args, savedir, poisoned_train_ds, "poisoned") 

    # poisoned_train_removed_ds = remove_samples(poisoned_train_ds, args.target_sample)

    # print(f"=== Size of poisoned removed train_dl: {len(poisoned_train_removed_ds)}")
    # print(poisoned_train_removed_ds.targets[:10])
    # train(args, savedir, poisoned_train_removed_ds, test_dl, DEVICE, "poisoned_removed")
    # inference(args, savedir, poisoned_train_ds, DEVICE, "poisoned_removed")
    # score(args, savedir, poisoned_train_ds, "poisoned_removed") 



    # # # =========== 重训练数据集
    # print(f"=== Size of clean train_dl: {len(train_ds)}")
    # print(train_ds.targets[:10])
    # unlearn_flipped_ds = ConcatDataset([train_ds, flipped_ds])
    # unlearn_flipped_ds = CustomDataset(unlearn_flipped_ds)
    # train(args, savedir, unlearn_flipped_ds, test_dl, DEVICE, "clean")
    # inference(args, savedir, train_ds, DEVICE, "clean")
    # score(args, savedir, train_ds, "clean") 

    # train_removed_ds = remove_samples(train_ds, args.target_sample)

    # print(f"=== Size of clean removed train_dl: {len(train_removed_ds)}")
    # print(train_removed_ds.targets[:10])
    # unlearn_flipped_removed_ds = ConcatDataset([train_removed_ds, flipped_ds])
    # unlearn_flipped_removed_ds = CustomDataset(unlearn_flipped_removed_ds)
    # train(args, savedir, unlearn_flipped_removed_ds, test_dl, DEVICE, "clean_removed")
    # inference(args, savedir, train_ds, DEVICE, "clean_removed")
    # score(args, savedir, train_ds, "clean_removed")

    # # =========== 近似遗忘
    unlearn_unrolling_sgd(args, savedir, unlearn_ds, test_dl, DEVICE, "unlearn")
    inference(args, savedir, poisoned_train_ds, DEVICE, "unlearn")
    score(args, savedir, poisoned_train_ds, "unlearn")

    unlearn_unrolling_sgd(args, savedir, unlearn_ds, test_dl, DEVICE, "unlearn_removed")
    inference(args, savedir, poisoned_train_ds, DEVICE, "unlearn_removed")
    score(args, savedir, poisoned_train_ds, "unlearn_removed") 


if __name__ == "__main__":
    main()
