import os
import copy
import torch
import torch.optim as optim
from torch.nn import functional as F
from tqdm import tqdm
from lira.wide_resnet import WideResNet
from torchvision import models
from torch import nn
from lira.utils import approx_retraining

def unlearn_unrolling_sgd(args, savedir, unlearned_loader, test_loader, device, data_type):
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

    # 从指定目录加载原模型权重
    if data_type == "unlearn":
        poisoned_model_path = os.path.join(savedir, 'poisoned', 'model.pt')
    if data_type == "unlearn_removed":
        poisoned_model_path = os.path.join(savedir, 'poisoned_removed', 'model.pt')
    model.load_state_dict(torch.load(poisoned_model_path, map_location=device))

    # 获取需要遗忘数据的梯度列表
    grad_list = []
    model.train()
    print(f'unlearned_loader.dataset length: {len(unlearned_loader.dataset)}')
    for i, (img, label) in enumerate(tqdm(unlearned_loader, desc="Calculating gradients")):
        img = img.to(device)
        label = label.to(device)
        output_grad = model(img)
        loss_grad = F.cross_entropy(output_grad, label)
        loss_grad.backward(retain_graph=True)
        grads = torch.autograd.grad(loss_grad, [param for param in model.parameters()], create_graph=True)
        grad_list.append(grads)

    # 更新模型的参数
    model_unlearned = copy.deepcopy(model)
    optimizer_unlearned = optim.SGD(model_unlearned.parameters(), lr=args.lr, weight_decay=args.l2_regularizer)
    
    model_unlearned.train()
    old_params = {}
    for i, (name, params) in enumerate(model.named_parameters()):
        old_params[name] = params.clone()
        for grads in grad_list:
            old_params[name] += args.lr * grads[i]

    for name, params in model_unlearned.named_parameters():
        params.data.copy_(old_params[name])

    # 保存遗忘学习后的模型
    savedir = os.path.join(savedir, data_type)
    os.makedirs(savedir, exist_ok=True)
    torch.save(model_unlearned.state_dict(), os.path.join(savedir, "model.pt"))
    print(f"Model saved to {os.path.join(savedir, 'model.pt')}")

    # 测试模型
    acc_test = get_acc(model_unlearned, test_loader, device)
    print(f"[test] acc_test: {acc_test:.4f}")


@torch.no_grad()
def get_acc(model, dl, device):
    acc = []
    for x, y in tqdm(dl, desc="Testing model"):
        x, y = x.to(device), y.to(device)
        acc.append(torch.argmax(model(x), dim=1) == y)
    acc = torch.cat(acc)
    acc = torch.sum(acc) / len(acc)
    return acc.item()




def unlearn_order(args, savedir, train_ds, unlearned_loader, test_loader, device, data_type, order=1):
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

    # 从指定目录加载原模型权重
    poisoned_model_path = os.path.join(savedir, 'poisoned', 'model.pt')
    model.load_state_dict(torch.load(poisoned_model_path, map_location=device))

    # 初始化损失函数
    criterion = nn.CrossEntropyLoss()

    # 获取 unlearned_loader 中的数据
    z_x, z_y = next(iter(unlearned_loader))
    z_x, z_y = z_x.to(device), z_y.to(device)

    # 从 train_ds 中获取目标样本
    target_index = args.target_sample
    z_x_delta, z_y_delta = train_ds[target_index]
    z_x_delta, z_y_delta = z_x_delta.to(device), torch.tensor([z_y_delta]).to(device)

    # 执行近似重训练
    theta_approx, diverged = approx_retraining(model, criterion, z_x, z_y, z_x_delta, z_y_delta, order=order)

    # 更新模型参数
    with torch.no_grad():
        for param, new_data in zip(model.parameters(), theta_approx):
            param.data = new_data

    # 保存遗忘学习后的模型
    savedir = os.path.join(savedir, data_type)
    os.makedirs(savedir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(savedir, "model.pt"))
    print(f"Model saved to {os.path.join(savedir, 'model.pt')}")

    # 测试模型
    acc_test = get_acc(model, test_loader, device)
    print(f"[test] acc_test: {acc_test:.4f}")

@torch.no_grad()
def get_acc(model, dl, device):
    acc = []
    for x, y in tqdm(dl, desc="Testing model"):
        x, y = x.to(device), y.to(device)
        acc.append(torch.argmax(model(x), dim=1) == y)
    acc = torch.cat(acc)
    acc = torch.sum(acc) / len(acc)
    return acc.item()
