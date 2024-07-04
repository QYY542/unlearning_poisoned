import os
import numpy as np
import argparse

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

def print_filtered_data(scores, logits, labels, keep, sample_index):
    """
    Print the data for a specific sample index, filtered by the keep array.
    
    Args:
        scores (numpy.ndarray): Scores array.
        logits (numpy.ndarray): Logits array.
        labels (numpy.ndarray): Labels array.
        keep (numpy.ndarray): Boolean array indicating which items to keep.
        sample_index (int): Index of the sample to print.
    """
    true_indices = np.where(keep)[0]
    if sample_index >= len(true_indices):
        print("Error: sample_index is out of range for the filtered dataset.")
        return
    
    real_index = true_indices[sample_index]

    print(f"Filtered Item Details (Index {real_index}):")
    print("Score:", scores[real_index])
    print("Logit:")
    print(logits[real_index])
    print("Label:", labels[real_index])
    print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir", default="exp/cifar10/random_uniform", type=str, help="Directory to save the experiment data")
    parser.add_argument("--sample_index", type=int, default=1, help="Index of the sample to extract data for")
    args = parser.parse_args()

    num_items = 1  # We only want to print details for one specified item

    for shadow_id in os.listdir(args.savedir):
        for data_type in ['clean', 'poisoned']:
            scores_path = os.path.join(args.savedir, shadow_id, data_type, "scores.npy")
            logits_path = os.path.join(args.savedir, shadow_id, data_type, "logits.npy")
            labels_path = os.path.join(args.savedir, shadow_id, data_type, "labels.npy")
            keep_path = os.path.join(args.savedir, shadow_id, data_type, "keep.npy")

            # Load data
            scores_data = load_data(scores_path)
            logits_data = load_data(logits_path)
            labels_data = load_data(labels_path)
            keep_data = load_data(keep_path)

            if scores_data is None or logits_data is None or labels_data is None or keep_data is None:
                continue

            print(f"Data from {shadow_id} ({data_type}):")
            print_filtered_data(scores_data, logits_data, labels_data, keep_data, args.sample_index)

if __name__ == "__main__":
    main()
