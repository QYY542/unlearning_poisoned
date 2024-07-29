import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from unlearner import unlearn_unrolling_sgd
from lira.wide_resnet import WideResNet
from torchvision import models
from torch.utils.data import Subset
from lira.utils import CustomDataset
import os
import random
import math

class LabeledSubset(Subset):
    """Subset with a targets attribute."""
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.targets = [dataset.targets[i] for i in indices]

def optimize_omega(args, savedir, unlearn_dl, test_loader, device, data_type, target_sample):
    """
    优化欧米伽向量以选择遗忘的数据子集，并执行遗忘学习。
    
    参数:
        args: 包含超参数的对象。
        unlearn_dl: 投毒后的训练数据加载器。
        test_loader: 测试数据加载器。
        device: 运行设备（CPU或GPU）。
        data_type: 数据类型字符串，用于确定保存路径。
        target_sample: 特定测试样本的输入特征和真实标签。
        
    返回:
        最优欧米伽向量 omega。
    """

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
    model = model.to(device)
    # 1. 随机初始化欧米伽向量
    len_omega = len(unlearn_dl.dataset)  # 假设 A 是投毒数据集中样本的数量
    omega = torch.rand(len_omega, requires_grad=True, device=device)

    

    # 定义优化器
    optimizer = torch.optim.Adam([omega], lr=0.01)  # 使用 Adam 优化器

    for epoch in range(100):
        # 2. 根据欧米伽的值选择要遗忘的数据子集 D_u
        # 选择 omega_j > 0.5 的样本作为要遗忘的数据
        print(omega)
        indices_to_unlearn = (omega > 0.5).nonzero().squeeze(dim=1)
        if indices_to_unlearn.dim() == 0:
            indices_to_unlearn = indices_to_unlearn.unsqueeze(0)

        # 从原始数据集中选择要遗忘的数据
        unlearned_dataset = LabeledSubset(unlearn_dl.dataset, indices_to_unlearn.tolist())
        unlearned_dataset = CustomDataset(unlearned_dataset)
        unlearned_loader = DataLoader(unlearned_dataset, batch_size=128, shuffle=False)

        # 3. 使用 unlearn_unrolling_sgd 函数执行遗忘学习
        unlearn_unrolling_sgd(args, savedir, unlearned_loader, test_loader, device, data_type)

        # 4. 计算并更新欧米伽向量 omega 以优化目标函数
        # 假设 loss 是根据 omega 和遗忘后的模型计算得到的
        loss = compute_loss(savedir, omega, device, model, data_type, target_sample)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 更新 omega
        optimizer.step()

        # 打印进度
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item()}")

    return omega.detach()


def simulate_annealing(args, savedir, unlearn_dl, test_loader, device, data_type, target_sample, max_iterations=100, initial_temperature=1.0, cooling_rate=0.99):
    """
    使用模拟退火算法优化欧米伽向量。
    
    参数:
        args: 包含超参数的对象。
        unlearn_dl: 投毒后的训练数据加载器。
        test_loader: 测试数据加载器。
        device: 运行设备（CPU或GPU）。
        data_type: 数据类型字符串，用于确定保存路径。
        target_sample: 特定测试样本的输入特征和真实标签。
        max_iterations: 最大迭代次数。
        initial_temperature: 初始温度。
        cooling_rate: 冷却系数。
        
    返回:
        最优欧米伽向量 omega。
    """

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
    model = model.to(device)

    # 1. 随机初始化欧米伽向量
    len_omega = len(unlearn_dl.dataset)  # 假设 A 是投毒数据集中样本的数量
    omega = torch.rand(len_omega, requires_grad=True, device=device)

    # 2. 设置模拟退火参数
    T = initial_temperature  # 初始温度
    cooling_rate = cooling_rate  # 冷却系数
    max_iterations = max_iterations  # 最大迭代次数

    best_omega = omega.clone()
    best_loss = compute_loss(savedir, omega, device, model, data_type, target_sample)

    for iteration in range(max_iterations):
        # 生成新的欧米伽向量
        omega_new = omega + torch.randn_like(omega) * T  # 生成扰动
        omega_new = torch.clamp(omega_new, min=0.0, max=1.0)  # 限制在 [0, 1] 范围内
        print(omega_new)
        
        indices_to_unlearn = (omega_new > 0.5).nonzero().squeeze(dim=1)
        if indices_to_unlearn.dim() == 0:
            indices_to_unlearn = indices_to_unlearn.unsqueeze(0)
        unlearned_dataset = LabeledSubset(unlearn_dl.dataset, indices_to_unlearn.tolist())
        unlearned_dataset = CustomDataset(unlearned_dataset)
        unlearned_loader = DataLoader(unlearned_dataset, batch_size=128, shuffle=False)

        # 使用 unlearn_unrolling_sgd 函数执行遗忘学习
        unlearn_unrolling_sgd(args, savedir, unlearned_loader, test_loader, device, data_type)

        # 计算新解的损失
        loss_new = compute_loss(savedir, omega_new, device, model, data_type, target_sample)

        # 计算损失差
        delta_loss = loss_new - best_loss
        print(f'delta_loss: {delta_loss} ')
        print(f'best_loss: {best_loss} ')

        # 接受或拒绝新解
        if delta_loss < 0 or random.random() < math.exp(-delta_loss / T):
            omega = omega_new
            if loss_new < best_loss:
                best_omega = omega_new
                best_loss = loss_new

        # 更新温度
        T *= cooling_rate

        # 打印进度
        # if iteration % 100 == 0:
        #     print(f"Iteration [{iteration}/{max_iterations}], Loss: {best_loss.item()}")

    indices_to_unlearn = (best_omega > 0.5).nonzero().squeeze(dim=1)
    if indices_to_unlearn.dim() == 0:
        indices_to_unlearn = indices_to_unlearn.unsqueeze(0)
    unlearned_dataset = LabeledSubset(unlearn_dl.dataset, indices_to_unlearn.tolist())
    unlearned_dataset = CustomDataset(unlearned_dataset)
    unlearned_loader = DataLoader(unlearned_dataset, batch_size=128, shuffle=False)
    unlearn_unrolling_sgd(args, savedir, unlearned_loader, test_loader, device, data_type)
    best_loss = compute_loss(savedir, best_omega, device, model, data_type, target_sample)
    print(f'best_loss: {best_loss} ')

    return best_omega.detach()


def compute_loss(savedir, omega, device, model, data_type, target_sample):
    """
    计算损失函数。

    深色版本
    参数:
        savedir: 存储模型的目录路径。
        omega: 当前欧米伽向量。
        device: 运行设备（CPU或GPU）。
        model: 已经加载的模型。
        data_type: 数据类型，用于确定加载哪个模型。
        target_sample: 包含特定测试样本的输入特征和真实标签的元组。
        
    返回:
        损失值。
    """
    alpha = 10000  # 均方误差损失的权重
    gamma = 0  # 正则化项的权重
    eta = 1  # 正则化函数的平滑因子

    # 从 target_sample 中分离出输入特征 x_t 和真实标签 y_t
    x_t, y_t = target_sample

    # 如果 x_t 和 y_t 不是张量，则转换为张量
    if not isinstance(x_t, torch.Tensor):
        x_t = torch.tensor(x_t, dtype=torch.float32)
    if not isinstance(y_t, torch.Tensor):
        y_t = torch.tensor(y_t, dtype=torch.long)

    # 将张量移动到正确的设备上
    x_t = x_t.to(device)
    y_t = y_t.to(device)

    # 设置模型参数为遗忘后的模型参数
    unlearn_model_path = os.path.join(savedir, data_type, 'model.pt')

    model.load_state_dict(torch.load(unlearn_model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # 可能需要增加维度以匹配模型输入
        loss_unlearned = F.cross_entropy(model(x_t.unsqueeze(0)), y_t.unsqueeze(0))
        # print(f'loss_unlearned: {loss_unlearned}')
        
    # 正则化项
    regularization = gamma * torch.sum(1 / (1 + torch.exp(-eta * (omega - 0.5))))

    return alpha * loss_unlearned + regularization