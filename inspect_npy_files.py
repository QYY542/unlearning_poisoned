import os
import numpy as np

def load_data(file_path):
    """
    Load a .npy file and return the data.
    
    Args:
        file_path (str): Path to the .npy file.
        
    Returns:
        numpy.ndarray: Data loaded from the npy file, or None if file does not exist.
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return None

    return np.load(file_path)

def print_filtered_data(scores, logits, labels, keep, num_items):
    count = 0
    for i in range(len(keep)):
        if keep[i]:
            print(f"Item {count+1}:")
            print("Score:", scores[i])
            print("Logit:")
            print(logits[i])
            print("Label:", labels[i])
            print()
            count += 1
            if count >= num_items:
                break

def main():
    # savedir = "exp/cifar10/flipped_label"
    # savedir = "exp/cifar10/fixed_label"
    savedir = "exp/cifar10/random_uniform"
    num_items = 2

    for shadow_id in os.listdir(savedir):
        for data_type in ['clean', 'poisoned']:
            scores_path = os.path.join(savedir, shadow_id, data_type, "scores.npy")
            logits_path = os.path.join(savedir, shadow_id, data_type, "logits.npy")
            labels_path = os.path.join(savedir, shadow_id, data_type, "labels.npy")
            keep_path = os.path.join(savedir, shadow_id, data_type, "keep.npy")

            # Load data
            scores_data = load_data(scores_path)
            logits_data = load_data(logits_path)
            labels_data = load_data(labels_path)
            keep_data = load_data(keep_path)

            if scores_data is None or logits_data is None or labels_data is None or keep_data is None:
                continue

            print(f"Data from {shadow_id} ({data_type}):")
            print_filtered_data(scores_data, logits_data, labels_data, keep_data, num_items)

if __name__ == "__main__":
    main()
