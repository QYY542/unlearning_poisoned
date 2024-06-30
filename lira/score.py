import numpy as np
import os

def score(args, data_loader, data_type):
    savedir = os.path.join(args.savedir, str(args.shadow_id), data_type)
    logits_path = os.path.join(savedir, "logits.npy")
    
    if not os.path.exists(logits_path):
        print(f"No logits file found in {logits_path}.")
        return

    opredictions = np.load(logits_path)  # [n_examples, n_augs, n_classes]

    # Numerically stable softmax calculation
    predictions = opredictions - np.max(opredictions, axis=-1, keepdims=True)
    predictions = np.exp(predictions)
    predictions = predictions / np.sum(predictions, axis=-1, keepdims=True)

    labels = get_labels(data_loader)

    COUNT = predictions.shape[0]
    y_true = predictions[np.arange(COUNT), :, labels[:COUNT]]
    print("mean acc", np.mean(predictions[:, 0, :].argmax(1) == labels[:COUNT]))

    predictions[np.arange(COUNT), :, labels[:COUNT]] = 0
    y_wrong = np.sum(predictions, axis=-1)

    logit = np.log(y_true + 1e-45) - np.log(y_wrong + 1e-45)
    score_path = os.path.join(savedir, "scores.npy")
    np.save(score_path, logit)
    print(f"Scores saved to {score_path}")

def get_labels(data_loader):
    labels = []
    for _, y in data_loader:
        labels.extend(y.numpy())
    return np.array(labels)
