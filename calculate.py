import numpy as np
import os
import argparse

def calculate_probability(scores):
    """简单地通过取平均分数模拟计算概率 P(b = 0|T(S))。"""
    return np.mean(scores)

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--poison_type", default="flipped_label", type=str, choices=["random_uniform", "fixed_label", "flipped_label"])
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

# 初始化存储分数的字典
scores_dict = {'S1': [], 'S2': [], 'S3': [], 'D': []}

# 假设我们有一个mapping来确定每个shadow model目录对应的数据集
dataset_mapping = {
    'poisoned': 'S1',
    'poisoned_removed': 'S2',
    'clean': 'S3',
    'clean_removed': 'D'
}

# 遍历每个shadow model目录
for shadow_id in os.listdir(savedir):
    shadow_dir = os.path.join(savedir, shadow_id)
    if os.path.isdir(shadow_dir):
        for data_type in ['poisoned', 'poisoned_removed', 'clean', 'clean_removed']:
            scores_path = os.path.join(shadow_dir, data_type, 'scores.npy')
            dataset_key = dataset_mapping[data_type]  # 确定属于哪个数据集
            if os.path.exists(scores_path):
                scores = np.load(scores_path)
                if real_index < len(scores):
                    score = scores[real_index]
                    scores_dict[dataset_key].append(score)

# 计算概率
P_S1 = calculate_probability(scores_dict['S1'])
P_S2 = calculate_probability(scores_dict['S2'])
P_S3 = calculate_probability(scores_dict['S3'])
P_D = calculate_probability(scores_dict['D'])

# 打印概率
print("P_S1 (Probability model is trained on poisoned):", P_S1)
print("P_S2 (Probability model is trained on poisoned_removed):", P_S2)
print("P_S3 (Probability model is trained on clean):", P_S3)
print("P_D (Probability model is trained on clean_removed):", P_D)

# 计算AdvMIA和AdvUL MIA
AdvMIA = P_S1 - P_S2
AdvUL_MIA = P_S3 - P_D
AdvUL = AdvUL_MIA - AdvMIA

print("AdvMIA:", AdvMIA)
print("AdvUL_MIA:", AdvUL_MIA)
print("AdvUL:", AdvUL)
