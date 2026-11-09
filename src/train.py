import random
import numpy as np
import torch
from torch import nn 
from torchvision import models
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model():
    model = models.resnet50(weights="DEFAULT")        # pretrained on ImageNet
    model.fc = nn.Linear(model.fc.in_features, 2)     # 2 outputs: benign / malignant
    return model


def evaluate(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            p = torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy()
            probs.append(p)
            labels.append(y.numpy())
    probs, labels = np.concatenate(probs), np.concatenate(labels)
    preds = (probs >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(labels, probs),
        "balanced_acc": balanced_accuracy_score(labels, preds),
        "sensitivity": recall_score(labels, preds, pos_label=1),
        "specificity": recall_score(labels, preds, pos_label=0),
    }


def train(model, train_loader, val_loader, device, epochs=2, lr=1e-4, class_weights=None):
    w = torch.tensor(class_weights, dtype=torch.float32).to(device) if class_weights else None
    criterion = nn.CrossEntropyLoss(weight=w)         # class_weights = configuration C later
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
        row = {"epoch": epoch, "train_loss": total / len(train_loader.dataset)}
        row.update({f"val_{k}": v for k, v in evaluate(model, val_loader, device).items()})
        history.append(row)
        print(row)
    return history