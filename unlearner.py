import os
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
from pyhessian import hessian
from torchvision import models
from lira.wide_resnet import WideResNet

def set_weights_fast(x, weights):
    with torch.no_grad():
        start = 0
        for weight in weights:
            length = len(weight.view(-1))
            array = x[start:start+length]
            weight_new = torch.Tensor(array).view(*weight.shape)
            weight.data.copy_(weight_new)
            start += length

def validate(model, loader, device):
    total = 0
    correct = 0
    for imgs, labels in loader:
        batch_size = len(imgs)
        total += batch_size
        imgs, labels = imgs.to(device), labels.to(device)
        out_probs = model(imgs)
        out = torch.argmax(out_probs, dim=1)
        labels = labels.view(out.shape)
        correct += torch.sum(out == labels)
    return correct, total

def weights_to_list_fast(weights):
    with torch.no_grad():
        weights_list = []
        for weight in weights:
            list_t = weight.view(-1).tolist()
            weights_list = weights_list + list_t
        return weights_list

def std_loss(x, y, std_reg):
    log_prob = -1.0 * torch.nn.functional.log_softmax(x, 1)
    loss = log_prob.gather(1, y.unsqueeze(1))
    loss = loss.mean()
    avg_std = torch.sum(torch.std(x, dim=1)) / len(x.view(-1))
    loss = loss + std_reg * avg_std
    return loss

def my_cross_entropy(x, y, std_reg):
    log_prob = -1.0 * torch.nn.functional.log_softmax(x, 1)
    loss = log_prob.gather(1, y.unsqueeze(1))
    loss = loss.mean()
    N, C = x.shape
    p = torch.nn.functional.softmax(x, 1)
    hessian_loss = torch.sum(p * (1 - p), dim=1)
    hessian_loss = hessian_loss.mean()
    loss = loss + std_reg * hessian_loss
    return loss

def load_model_from_path(args, device, savedir):
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
    model_path = os.path.join(savedir, 'poisoned', 'model.pt')
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def unlearn_finetune(args, savedir, train_loader, test_loader, device, data_type):
    # 加载模型
    net = load_model_from_path(args, device, savedir)
    
    lr = args.lr
    criterion_unl = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=5e-4)
    savedir = os.path.join(savedir, data_type)

    # 提取预训练模型的权重
    pretrain_weights = weights_to_list_fast([param for param in net.parameters()])
    
    trainset_list = list(train_loader.dataset)
    batch_star = trainset_list[-args.repeat_num:]
    unlearned_loader = torch.utils.data.DataLoader(batch_star, batch_size=128, shuffle=False, num_workers=2)
    
    sigma_list = []
    delta_weights_list = []
    unl_error_list = []
    rolling_unl_error_list = []
    ver_error_list = []
    
    grad_list = []
    
    for ep in range(args.epochs):
        for main_idx, (inputs, targets) in enumerate(train_loader):
            net.train()
            inputs, targets = inputs.to(device), targets.to(device)
            if main_idx == 0:
                optimizer.zero_grad()
                outputs = net(inputs)
                if args.loss_func == 'hess':
                    loss = my_cross_entropy(outputs, targets, args.regularizer)
                elif args.loss_func == 'std':
                    loss = std_loss(outputs, targets, args.regularizer)
                else:
                    loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                for img, label in unlearned_loader:
                    img, label = img.cuda(), label.cuda()
                    output_grad = net(img)
                    if args.loss_func == 'hess':
                        loss_grad = my_cross_entropy(output_grad, label, args.regularizer)
                    elif args.loss_func == 'std':
                        loss_grad = std_loss(output_grad, label, args.regularizer)
                    else:
                        loss_grad = criterion(output_grad, label)
                    loss_grad.backward(retain_graph=True)
                    grads = torch.autograd.grad(loss_grad, [param for param in net.parameters()], create_graph=True)
                    grad_list.append(grads)
            
            if main_idx != 0:
                optimizer.zero_grad()
                outputs = net(inputs)
                if args.loss_func == 'hess':
                    loss = my_cross_entropy(outputs, targets, args.regularizer)
                elif args.loss_func == 'std':
                    loss = std_loss(outputs, targets, args.regularizer)
                else:
                    loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            if main_idx % args.eval_every == 0:
                net.eval()
                if args.loss_func == 'hess':
                    hessian_comp = hessian(net, my_cross_entropy, data=(inputs, targets), cuda=True)
                elif args.loss_func == 'std':
                    hessian_comp = hessian(net, std_loss, data=(inputs, targets), cuda=True)
                else:
                    hessian_comp = hessian(net, criterion, data=(inputs, targets), cuda=True)
                top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues()
                sigma = np.sqrt(top_eigenvalues[-1])
                sigma_list.append(sigma)
                
                net_weights = [param for param in net.parameters()]
                net_weights_list = weights_to_list_fast(net_weights)
                
                net_unlearned = copy.deepcopy(net)
                optimizer_unlearned = optim.SGD(net_unlearned.parameters(), lr=lr, weight_decay=5e-4)
                net_unlearned.train()
                
                old_params = {}
                for i, (name, params) in enumerate(net.named_parameters()):
                    old_params[name] = params.clone()
                    for grads in grad_list:
                        old_params[name] += lr * grads[i]
                for name, params in net_unlearned.named_parameters():
                    params.data.copy_(old_params[name])
                
                net_unlearned_weights = [param for param in net_unlearned.parameters()]
                net_unlearned_weights_list = weights_to_list_fast(net_unlearned_weights)
                
                delta_weights = np.linalg.norm(np.array(net_weights_list) - np.array(pretrain_weights))
                unl_error = (lr * lr) * ((len(train_loader) * ep) + main_idx) * (1 / 2) * delta_weights * sigma
                rolling_unl_error = (lr * lr) * ((len(train_loader) * ep) + main_idx) * (1 / 2) * delta_weights * (sum(sigma_list) / len(sigma_list))
                verification_error = np.linalg.norm(np.array(net_weights_list) - np.array(net_unlearned_weights_list))
                
                delta_weights_list.append(delta_weights)
                unl_error_list.append(unl_error)
                rolling_unl_error_list.append(rolling_unl_error)
                ver_error_list.append(verification_error)

                # 打印重要信息
                print(f"Epoch: {ep}, Step: {main_idx}")
                print(f"Sigma: {sigma}")
                print(f"Delta Weights: {delta_weights}")
                print(f"Unlearning Error: {unl_error}")
                print(f"Rolling Unlearning Error: {rolling_unl_error}")
                print(f"Verification Error: {verification_error}")
    
    correct, total = validate(net, test_loader)
    test_acc = correct / total
    print(f"Final Test Accuracy: {test_acc:.4f}")

    # Save the finetuned model
    os.makedirs(savedir, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(savedir, "model_finetuned.pt"))
