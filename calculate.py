import numpy as np
import os
import argparse

def calculate_probability(scores):
    """简单地通过取平均分数模拟计算概率 P(b = 0|T(S))。"""
    return np.mean(scores)

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--poison_type", default="random_uniform", type=str, choices=["random_uniform", "fixed_label", "flipped_label"])
parser.add_argument("--target_sample", type=int, default=0, help="Index of the sample to extract scores for")
parser.add_argument("--model", default="resnet18", type=str)
args = parser.parse_args()
savedir = os.path.join("exp/cifar10/", args.model, args.poison_type, str(f'target_sample_{args.target_sample}'))
# savedir = os.path.join("exp/cifar10/", args.model, args.poison_type, str(f'target_sample_4'))

# 读取 keep.npy 文件以作为索引指南
target_index = args.target_sample

# 初始化存储分数的字典
scores_dict = {'poisoned': [], 'poisoned_removed': [], 'clean': [], 'clean_removed': [], 'unlearn': [], 'unlearn_removed': []}

# 假设我们有一个mapping来确定每个shadow model目录对应的数据集
dataset_mapping = {
    'poisoned': 'poisoned',
    'poisoned_removed': 'poisoned_removed',
    'clean': 'clean',
    'clean_removed': 'clean_removed',
    'unlearn': 'unlearn',
    'unlearn_removed': 'unlearn_removed'
}

# 遍历每个shadow model目录
for shadow_id in os.listdir(savedir):
    shadow_dir = os.path.join(savedir, shadow_id)
    if os.path.isdir(shadow_dir):
        for data_type in ['poisoned', 'poisoned_removed', 'clean', 'clean_removed', 'unlearn', 'unlearn_removed']:
            scores_path = os.path.join(shadow_dir, data_type, 'scores.npy')
            dataset_key = dataset_mapping[data_type]  # 确定属于哪个数据集
            if os.path.exists(scores_path):
                scores = np.load(scores_path)
                # print(f'scores len:{len(scores)}')
                score = scores[target_index]
                # print(f'{data_type}:{score}')
                scores_dict[dataset_key].append(score)

# 计算概率
P_S1 = calculate_probability(scores_dict['poisoned'])
P_S2 = calculate_probability(scores_dict['poisoned_removed'])
P_S3 = calculate_probability(scores_dict['unlearn'])
P_S4 = calculate_probability(scores_dict['unlearn_removed'])

P_S3_clean = calculate_probability(scores_dict['clean'])
P_S4_clean = calculate_probability(scores_dict['clean_removed'])

# 打印概率
print("P_S1 :", P_S1)
print("P_S2 :", P_S2)
print("P_S3 :", P_S3)
print("P_S4 :", P_S4)
print("P_S3_clean:", P_S3_clean)
print("P_S4_clean:", P_S4_clean)

# 计算AdvMIA和AdvUL MIA
AdvMIA = P_S1 - P_S2
AdvUL_MIA = P_S3 - P_S4
AdvUL_MIA_clean = P_S3_clean - P_S4_clean

AdvUL_unlearn = AdvUL_MIA - AdvMIA
AdvUL_clean = AdvUL_MIA_clean - AdvMIA


print("AdvUL(近似遗忘):", AdvUL_unlearn)
print("AdvUL(完全重训练):", AdvUL_clean)
