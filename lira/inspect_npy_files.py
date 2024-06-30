import os
import numpy as np

def load_and_print_npy(file_path, num_items=3):
    """
    Load a .npy file and print the first few items.

    Args:
        file_path (str): Path to the .npy file.
        num_items (int): Number of items to print.
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return

    data = np.load(file_path)
    print(f"Data from {file_path} (showing first {num_items} items):")
    print(data[:num_items])

def main():
    base_dir = "exp/cifar10"  # Base directory where the npy files are stored

    # Define the paths to the npy files
    keep_file = os.path.join(base_dir, "1", "keep.npy")  # Adjust the subdirectory as needed
    logits_file = os.path.join(base_dir, "1", "logits.npy")  # Adjust the subdirectory as needed
    scores_file = os.path.join(base_dir, "1", "scores.npy")  # Adjust the subdirectory as needed

    # Load and print the data from each file
    load_and_print_npy(keep_file)
    load_and_print_npy(logits_file)
    load_and_print_npy(scores_file)

if __name__ == "__main__":
    main()
