import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def find_tpr_at_fpr(fpr, tpr, target_fpr=0.001):
    """ Helper function to find the TPR at a given FPR. """
    indices = np.where(fpr <= target_fpr)[0]
    return tpr[indices[-1]] if indices.size > 0 else 0.0

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--poison_type", default="fixed_label", type=str, choices=["random_uniform", "fixed_label", "flipped_label"])
parser.add_argument("--sample_index", type=int, default=0, help="Index of the sample to extract scores for")
args = parser.parse_args()
savedir = os.path.join("exp/cifar10", args.poison_type)

# 读取 keep.npy 文件以作为索引指南
keep_path = 'save/keep.npy'
if not os.path.exists(keep_path):
    print("Error: keep.npy file not found.")
    exit()
keep_mask = np.load(keep_path)
true_indices = np.where(keep_mask)[0]
if args.sample_index >= len(true_indices):
    print(f"Error: sample_index {args.sample_index} is out of range for the filtered dataset")
    exit()
real_index = true_indices[args.sample_index]

# 初始化分数和标签列表
clean_scores = []
poisoned_scores = []
clean_labels = []
poisoned_labels = []

# 遍历每个shadow model目录
for shadow_id in os.listdir(savedir):
    shadow_dir = os.path.join(savedir, shadow_id)
    if os.path.isdir(shadow_dir):
        for data_type in ['clean', 'clean_removed', 'poisoned', 'poisoned_removed']:
            scores_path = os.path.join(shadow_dir, data_type, 'scores.npy')
            if os.path.exists(scores_path):
                scores = np.load(scores_path)
                if real_index < len(scores):
                    score = scores[real_index]
                    if 'clean' in data_type:
                        clean_scores.append(score)
                        clean_labels.append(1 if data_type == 'clean' else 0)
                    elif 'poisoned' in data_type:
                        poisoned_scores.append(score)
                        poisoned_labels.append(1 if data_type == 'poisoned' else 0)

# 计算和绘制ROC曲线
plt.figure()

# Clean vs Clean Removed
fpr_clean, tpr_clean, _ = roc_curve(clean_labels, clean_scores)
roc_auc_clean = auc(fpr_clean, tpr_clean)
plt.plot(fpr_clean, tpr_clean, label=f'Clean vs Clean Removed (area = {roc_auc_clean:.2f})')

# Poisoned vs Poisoned Removed
fpr_poisoned, tpr_poisoned, _ = roc_curve(poisoned_labels, poisoned_scores)
roc_auc_poisoned = auc(fpr_poisoned, tpr_poisoned)
plt.plot(fpr_poisoned, tpr_poisoned, label=f'Poisoned vs Poisoned Removed (area = {roc_auc_poisoned:.2f})')

# 绘图设置
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Clean and Poisoned Samples')
plt.legend(loc="lower right")
plt.savefig(f'save/{args.poison_type}/roc_curves_{args.poison_type}_{args.sample_index}.png')
plt.show()
plt.close()

# 计算0.1% FPR处的TPR
# AdvMIA = find_tpr_at_fpr(fpr_poisoned, tpr_poisoned)
# AdvUL_MIA = find_tpr_at_fpr(fpr_clean, tpr_clean)

# AdvMIA和AdvUL_MIA换为AUC
AdvMIA = roc_auc_poisoned
AdvUL_MIA = roc_auc_clean

print("Adv_MIA (Poisoned vs Poisoned Removed):", AdvMIA)
print("AdvUL_MIA (Clean vs Clean Removed):", AdvUL_MIA)

print(f"AdvUL: {AdvUL_MIA-AdvMIA}")
