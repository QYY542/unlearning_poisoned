import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, mobilenet_v2
from pathlib import Path

# 假设 get_data_loaders 函数已经定义好了
from data_loader import get_data_loaders  # 请确保替换 'data_loader' 为实际模块名

def load_model(model_path, device, dataset, model_name):
    """
    加载模型。
    """
    try:
        # 初始化模型
        if dataset == "cifar10":
            if model_name == "resnet18":
                print("resnet18")
                model = resnet18(weights=None, num_classes=10)
                model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                model.maxpool = torch.nn.Identity()
            elif model_name == "mobilenet_v2":
                print("mobilenet_v2")
                model = mobilenet_v2(weights=None, num_classes=10)
                # 修改 MobileNet 第一层的输入通道数
                model.features[0][0] = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
            else:
                raise NotImplementedError
        elif dataset == "cifar100":
            if model_name == "resnet18":
                print("resnet18")
                model = resnet18(weights=None, num_classes=100)
                model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                model.maxpool = torch.nn.Identity()
            elif model_name == "mobilenet_v2":
                print("mobilenet_v2")
                model = mobilenet_v2(weights=None, num_classes=100)
            else:
                raise NotImplementedError
        elif dataset == "FashionMNIST":
            if model_name == "resnet18":
                print("resnet18")
                model = resnet18(weights=None, num_classes=10)
                model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
                model.maxpool = torch.nn.Identity()
            elif model_name == "mobilenet_v2":
                print("mobilenet_v2")
                model = mobilenet_v2(weights=None, num_classes=10)
                # 修改 MobileNet 第一层的输入通道数
                model.features[0][0] = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
            else:
                raise NotImplementedError
        else:
            raise ValueError("Unsupported dataset.")

        model = model.to(device)
        
        # 加载模型权重
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    except FileNotFoundError:
        print(f"Error: The file {model_path} does not exist.")
        return None

def predict_with_model(model, data_loader, device):
    """
    使用模型进行预测。
    """
    all_logits = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_logits.append(outputs.cpu().numpy())
    return np.concatenate(all_logits)

def calculate_accuracy(logits, labels):
    """
    计算准确率。
    """
    predictions = np.argmax(logits, axis=1)
    accuracy = np.mean(predictions == labels)
    return accuracy

def main(poison_type, target_sample, model_name, dataset):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    savedir = os.path.join("exp/", dataset, model_name, poison_type, f'target_sample_{target_sample}')
    results = {}
    
    # 遍历每个shadow model目录
    for shadow_id in os.listdir(savedir):
        shadow_dir = os.path.join(savedir, shadow_id)
        results[shadow_id] = {}
        
        # for data_type in ['clean', 'clean_removed', 'poisoned', 'poisoned_removed']:
        for data_type in ['poisoned']:
            # 加载模型
            model_path = os.path.join(shadow_dir, data_type, 'model.pt')
            model = load_model(model_path, DEVICE, dataset, model_name)
            
            if model is not None:
                # 获取数据加载器
                _, test_dl = get_data_loaders(dataset=dataset)
                
                # 使用模型进行预测
                model_logits = predict_with_model(model, test_dl, DEVICE)
                
                # 从测试数据加载器中获取标签
                all_labels = []
                for _, labels in test_dl:
                    all_labels.append(labels.numpy())
                all_labels = np.concatenate(all_labels)
                
                # 计算准确率
                accuracy = calculate_accuracy(model_logits, all_labels)
                results[shadow_id][data_type] = accuracy * 100
                
                print(f"Shadow ID: {shadow_id}, Data Type: {data_type}, Average accuracy: {accuracy * 100:.2f}%")
    
    # 汇总结果
    print("\nSummary:")
    for shadow_id, data_types in results.items():
        print(f"Shadow ID: {shadow_id}")
        for data_type, accuracy in data_types.items():
            print(f"  {data_type}: {accuracy:.2f}%")
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate average accuracy from model predictions.")
    
    parser.add_argument("--poison_type", default="flipped_label", type=str, choices=["random_label", "fixed_label", "flipped_label", "random_samples"], help="Poisoning type.")
    parser.add_argument("--target_sample", type=int, default=0, help="Index of the sample to extract scores for.")
    parser.add_argument("--model_name", default="resnet18", type=str, help="Model name.")
    parser.add_argument("--dataset", default="cifar10", type=str, choices=["cifar10", "cifar100", "FashionMNIST"], help="Dataset to use.")
    
    args = parser.parse_args()
    
    main(args.poison_type, args.target_sample, args.model_name, args.dataset)
