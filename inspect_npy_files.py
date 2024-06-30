import os
import numpy as np

def load_and_print_data(file_path, data_label, num_items=3):
    """
    Load a .npy file, print the first few items, and calculate the average.
    
    Args:
        file_path (str): Path to the .npy file.
        data_label (str): Label to print for data (e.g., 'scores' or 'logits').
        num_items (int): Number of items to print.
        
    Returns:
        numpy.ndarray: Data loaded from the npy file, or None if file does not exist.
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return None

    data = np.load(file_path)
    print(f"Data from {file_path} ({data_label}, showing first {num_items} items):")
    print(data[:num_items])
    return data

def main():
    savedir = "exp/cifar10/flipped_label"
    num_items = 3

    clean_scores = []
    poisoned_scores = []

    for shadow_id in os.listdir(savedir):
        for data_type in ['clean', 'poisoned']:
            scores_path = os.path.join(savedir, shadow_id, data_type, "scores.npy")
            logits_path = os.path.join(savedir, shadow_id, data_type, "logits.npy")
            labels_path = os.path.join(savedir, shadow_id, data_type, "labels.npy")
            keep_path = os.path.join(savedir, shadow_id, data_type, "keep.npy")

            # Load and print logits
            logits_data = load_and_print_data(logits_path, 'logits', num_items)
            # Load and print labels
            labels_data = load_and_print_data(labels_path, 'labels', num_items)
            # Load and print keep
            keep_data = load_and_print_data(keep_path, 'keep', num_items)
            # Load and print scores
            scores_data = load_and_print_data(scores_path, 'scores', num_items)
            if scores_data is not None:
                if data_type == 'clean':
                    clean_scores.append(scores_data)
                else:
                    poisoned_scores.append(scores_data)
            


    # Compute the overall average scores for clean and poisoned
    if clean_scores:
        clean_scores = np.concatenate(clean_scores, axis=0)
        clean_avg = np.mean(clean_scores)
        print(f"Overall average score for 'clean': {clean_avg}")
    if poisoned_scores:
        poisoned_scores = np.concatenate(poisoned_scores, axis=0)
        poisoned_avg = np.mean(poisoned_scores)
        print(f"Overall average score for 'poisoned': {poisoned_avg}")

if __name__ == "__main__":
    main()
