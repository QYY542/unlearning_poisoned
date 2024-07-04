import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--savedir", default="exp/cifar10/random_uniform", type=str, help="Directory to save the experiment data")
parser.add_argument("--sample_index", type=int, help="Index of the sample to extract scores for")
args = parser.parse_args()

# 初始化列表存储分数和标签
clean_scores = []
poisoned_scores = []
clean_labels = []
poisoned_labels = []

# 读取 keep.npy 文件以作为索引指南
keep_path = 'save/keep.npy'
if os.path.exists(keep_path):
    keep_mask = np.load(keep_path)
    # 根据keep.npy获取为True的索引
    true_indices = np.where(keep_mask)[0]
    if args.sample_index >= len(true_indices):
        print(f"Error: sample_index {args.sample_index} is out of range for the filtered dataset")
        exit()
    # 获取真实数据集中对应的index
    real_index = true_indices[args.sample_index]
else:
    print("Error: keep.npy file not found.")
    exit()

# 遍历savedir中的每个目录，假设每个目录是一个shadow model
for shadow_id in os.listdir(args.savedir):
    shadow_dir = os.path.join(args.savedir, shadow_id)
    if os.path.isdir(shadow_dir):  # 确保是目录
        for data_type in ['clean', 'poisoned']:
            scores_path = os.path.join(shadow_dir, data_type, 'scores.npy')
            if os.path.exists(scores_path):
                # 读取scores.npy文件
                scores = np.load(scores_path)
                # 检查real_index是否在数组范围内
                if real_index < len(scores):
                    score = scores[real_index]
                    if data_type == 'clean':
                        clean_scores.append(score)
                        clean_labels.append(1)  # True
                    elif data_type == 'poisoned':
                        poisoned_scores.append(score)
                        poisoned_labels.append(0)  # False
                else:
                    print(f"Index {real_index} out of range for {scores_path}")
            else:
                print(f"File not found: {scores_path}")

# 合并得分和标签
scores = np.array(clean_scores + poisoned_scores)
labels = np.array(clean_labels + poisoned_labels)
print(scores)
print(labels)

# 确保有正负样本
if len(np.unique(labels)) < 2:
    print("Error: Not enough classes to plot ROC curve. Need both positive and negative samples.")
else:
    # 计算ROC曲线的参数
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Sample Classification Across All Shadow Models')
    plt.legend(loc="lower right")
    plt.savefig(f'save/roc_curve_{args.sample_index}.png')  # 保存ROC曲线图像
    print("roc_curve saved")
    plt.close()
