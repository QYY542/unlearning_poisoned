import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def calculate_optimal_threshold(fpr, tpr, thresholds):
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

def calculate_probabilities(scores, threshold):
    half = len(scores) // 2
    first_half = scores[:half]
    second_half = scores[half:]
    probs_first_half = np.mean(first_half >= threshold)
    probs_second_half = np.mean(second_half >= threshold)
    return probs_first_half, probs_second_half

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--poison_type", default="random_label", type=str, choices=["random_label", "fixed_label", "flipped_label", "random_samples"])
parser.add_argument("--target_sample", type=int, default=0, help="Index of the sample to extract scores for")
parser.add_argument("--model", default="vgg16", type=str)
parser.add_argument("--dataset", default="cifar10", type=str, choices=["cifar10", "FashionMNIST"])
args = parser.parse_args()
savedir = os.path.join("exp/", args.dataset, args.model, args.poison_type, str(f'target_sample_{args.target_sample}'))

# 读取 keep.npy 文件以作为索引指南
target_index = args.target_sample

# 初始化分数和标签列表
clean_scores = []
unlearn_scores = []
poisoned_scores = []

clean_labels = []
unlearn_labels = []
poisoned_labels = []

# 遍历每个shadow model目录
for shadow_id in os.listdir(savedir):
    shadow_dir = os.path.join(savedir, shadow_id)
    if os.path.isdir(shadow_dir):
        # for data_type in ['clean', 'clean_removed', 'unlearn', 'unlearn_removed', 'poisoned', 'poisoned_removed']:
        for data_type in ['clean', 'clean_removed', 'poisoned', 'poisoned_removed']:
            scores_path = os.path.join(shadow_dir, data_type, 'scores.npy')
            if os.path.exists(scores_path):
                scores = np.load(scores_path)
                score = scores[target_index]
                if 'clean' in data_type:
                    clean_scores.append(score)
                    clean_labels.append(1 if data_type == 'clean' else 0)
                # if 'unlearn' in data_type:
                #     unlearn_scores.append(score)
                #     unlearn_labels.append(1 if data_type == 'unlearn' else 0)
                elif 'poisoned' in data_type:
                    poisoned_scores.append(score)
                    poisoned_labels.append(1 if data_type == 'poisoned' else 0)

# 计算和绘制ROC曲线
plt.figure()

# # Clean vs Clean Removed
fpr_clean, tpr_clean, thresholds_clean = roc_curve(clean_labels, clean_scores)
roc_auc_clean = auc(fpr_clean, tpr_clean)
plt.plot(fpr_clean, tpr_clean, label=f'After Unlearning (area = {roc_auc_clean:.3f})')

# Unlearn vs Unlearn Removed
# fpr_unlearn, tpr_unlearn, thresholds_unlearn = roc_curve(unlearn_labels, unlearn_scores)
# roc_auc_unlearn = auc(fpr_unlearn, tpr_unlearn)
# plt.plot(fpr_unlearn, tpr_unlearn, label=f'Unlearn vs Unlearn remove Sample (area = {roc_auc_unlearn:.3f})')

# Poisoned vs Poisoned Removed
fpr_poisoned, tpr_poisoned, thresholds_poisoned = roc_curve(poisoned_labels, poisoned_scores)
roc_auc_poisoned = auc(fpr_poisoned, tpr_poisoned)
plt.plot(fpr_poisoned, tpr_poisoned, label=f'Before Unlearning (area = {roc_auc_poisoned:.3f})')

directory = f'save/{args.dataset}/{args.model}/{args.poison_type}/'
if not os.path.exists(directory):
    os.makedirs(directory)

# 绘图设置
plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
plt.xlim([-0.005, 1.005])
plt.ylim([-0.005, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Clean and Poisoned Samples')
plt.legend(loc="lower right")
plt.savefig(f'{directory}/roc_curves_{args.poison_type}_{args.target_sample}.png')
plt.show()
plt.close()

AdvMIA = roc_auc_poisoned
# AdvUL_MIA_unlearn = roc_auc_unlearn
AdvUL_MIA_clean = roc_auc_clean

print("\n====== ROC ======")
# print(f"AdvUL_MIA (近似遗忘后优势): {AdvUL_MIA_unlearn:.3f}")
print(f"AdvUL_MIA (完全重训练后优势): {AdvUL_MIA_clean:.3f}")
print(f"Adv_MIA (投毒后优势): {AdvMIA:.3f}")
print("=== 结果 ===")
# print(f"AdvUL（近似遗忘）: {(AdvUL_MIA_unlearn-AdvMIA):.3f}")
print(f"AdvUL（完全重训练）: {(AdvUL_MIA_clean-AdvMIA):.3f}")

# 找到最佳阈值
optimal_idx_clean = np.argmax(tpr_clean - fpr_clean)
optimal_threshold_clean = thresholds_clean[optimal_idx_clean]

# optimal_idx_unlearn = np.argmax(tpr_unlearn - fpr_unlearn)
# optimal_threshold_unlearn = thresholds_unlearn[optimal_idx_unlearn]

optimal_idx_poisoned = np.argmax(tpr_poisoned - fpr_poisoned)
optimal_threshold_poisoned = thresholds_poisoned[optimal_idx_poisoned]

# 计算每个类别的概率
clean_probs, clean_removed_probs = calculate_probabilities(clean_scores, optimal_threshold_clean)
# unlearn_probs, unlearn_removed_probs = calculate_probabilities(unlearn_scores, optimal_threshold_unlearn)
poisoned_probs, poisoned_removed_probs = calculate_probabilities(poisoned_scores, optimal_threshold_poisoned)

print("\n====== Probs ======")

# print("=== Unlearn ===")
# print(f"Unlearn probabilities: {unlearn_probs:.3f}")
# print(f"Unlearn removed probabilities: {unlearn_removed_probs:.3f}")
# print(f"Adv_Unlearn: {(unlearn_probs-unlearn_removed_probs):.3f}")

# print("=== Clean ===")
print(f"Clean probabilities: {clean_probs:.3f}")
print(f"Clean removed probabilities: {clean_removed_probs:.3f}")
print(f"Adv_Clean: {(clean_probs-clean_removed_probs):.3f}")

print("=== Poisoned ===")
print(f"Poisoned probabilities: {poisoned_probs:.3f}")
print(f"Poisoned removed probabilities: {poisoned_removed_probs:.3f}")
print(f"Adv_Poisoned: {(poisoned_probs-poisoned_removed_probs):.3f}")

print("=== 结果 ===")
# print(f"AdvUL（近似遗忘）: {((unlearn_probs-unlearn_removed_probs)-(poisoned_probs-poisoned_removed_probs)):.3f}")
print(f"AdvUL（完全重训练）: {((clean_probs-clean_removed_probs)-(poisoned_probs-poisoned_removed_probs)):.3f}")