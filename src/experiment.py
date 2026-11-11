from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score

from src.data import SkinDataset, get_transforms
from src.train import set_seed, build_model, train


CONFIGS = {
    "A_baseline": {"augment": False, "weights": False},
    "B_augment": {"augment": True, "weights": False},
    "C_weights": {"augment": False, "weights": True},
    "D_both": {"augment": True, "weights": True},
}


def predict_probs(model, loader, device):
    """Return the chance of 'malignant' for every image, plus the true labels."""
    model.eval()                                      # test mode
    probs, labels = [], []
    with torch.no_grad():                             # no learning, only predicting
        for x, y in loader:
            p = torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy()
            probs.append(p)
            labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def find_threshold(probs, labels):
    """Find the cutoff (between 0.05 and 0.95) that gives the best balanced accuracy."""
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):             # try many cutoffs
        score = balanced_accuracy_score(labels, (probs >= t).astype(int))
        if score > best_score:                        # keep the best one so far
            best_t, best_score = float(t), score
    return best_t


def scores(probs, labels, threshold):
    """Calculate the four scores for a given cutoff."""
    preds = (probs >= threshold).astype(int)          # malignant if chance >= cutoff
    return {
        "auc": roc_auc_score(labels, probs),                      # does not depend on the cutoff
        "balanced_acc": balanced_accuracy_score(labels, preds),
        "sensitivity": recall_score(labels, preds, pos_label=1),  # malignant lesions found
        "specificity": recall_score(labels, preds, pos_label=0),  # benign lesions recognized
    }


def run_experiment(df, config_name, seed, device, epochs=5, n_train=None, n_eval=None,
                   batch_size=32, model_dir="../models", tag=""):
    """Train one configuration with one seed and return its scores on the test set."""
    cfg = CONFIGS[config_name]
    set_seed(seed)                                    # same seed = same result

    # Take the three parts of the split
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    # Optional: use fewer images (for a quick test on a CPU)
    if n_train:
        train_df = train_df.sample(min(n_train, len(train_df)), random_state=seed)
    if n_eval:
        val_df = val_df.sample(min(n_eval, len(val_df)), random_state=seed)
        test_df = test_df.sample(min(n_eval, len(test_df)), random_state=seed)

    # Augmentation only for training images, never for val/test
    train_loader = DataLoader(SkinDataset(train_df, get_transforms(augment=cfg["augment"])),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SkinDataset(val_df, get_transforms(augment=False)), batch_size=batch_size)
    test_loader = DataLoader(SkinDataset(test_df, get_transforms(augment=False)), batch_size=batch_size)

    # Class weights: the rare class (malignant) gets a bigger weight in the loss
    class_weights = None
    if cfg["weights"]:
        n = train_df["label"].value_counts()
        class_weights = [len(train_df) / (2 * n[0]), len(train_df) / (2 * n[1])]

    # Build and train the model
    model = build_model().to(device)
    train(model, train_loader, val_loader, device, epochs=epochs, class_weights=class_weights)

    # Choose the cutoff on the validation set, then use it unchanged on the test set
    val_probs, val_labels = predict_probs(model, val_loader, device)
    threshold = find_threshold(val_probs, val_labels)
    test_probs, test_labels = predict_probs(model, test_loader, device)
    test_scores = scores(test_probs, test_labels, threshold)

    # Save the trained model, so we can test it on PAD-UFES-20 later without retraining
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path(model_dir) / f"{tag}{config_name}_seed{seed}.pt")

    # One row of results
    row = {"config": config_name, "seed": seed, "epochs": epochs,
           "n_train": len(train_df), "threshold": threshold}
    row.update({f"test_{k}": v for k, v in test_scores.items()})
    return row