import copy
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn


SEED = 0
ROOT = Path(__file__).resolve().parent


class CancerModel(nn.Module):
    def __init__(self, hidden=16, dropout=0.0):
        super().__init__()
        layers = [nn.Linear(30, hidden), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(hidden, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_data(scale=True):
    data = load_breast_cancer()
    x = data.data.astype("float32")
    y = data.target.astype("int64")
    y = np.where(y == 0, 1, 0).astype("int64")

    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.40, random_state=SEED, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train).astype("float32")
        x_val = scaler.transform(x_val).astype("float32")
        x_test = scaler.transform(x_test).astype("float32")

    tensors = (
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
        torch.from_numpy(x_val),
        torch.from_numpy(y_val),
        torch.from_numpy(x_test),
        torch.from_numpy(y_test),
    )
    return tensors, scaler


def diagnose(hist):
    if not hist["train"] or not hist["val"]:
        return "not started"
    if hist["train"][-1] > 0.3:
        return "underfitting"
    best_index = min(range(len(hist["val"])), key=hist["val"].__getitem__)
    best_val = hist["val"][best_index]
    train_at_best = hist["train"][best_index]
    if hist["val"][-1] > best_val * 1.30 or best_val - train_at_best > 0.05:
        return "overfitting"
    return "stable"


def train(cfg, data=None, on_epoch=None):
    if data is None:
        data, _ = load_data(cfg.get("scaling", True))
    x_train, y_train, x_val, y_val, _, _ = data

    torch.manual_seed(SEED)
    hidden = int(cfg.get("hidden", 16))
    dropout = float(cfg.get("dropout", 0.0))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    learning_rate = float(cfg.get("learning_rate", 0.001))
    epochs = int(cfg.get("epochs", 150))
    patience = int(cfg.get("patience", 20))

    model = CancerModel(hidden, dropout)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    hist = {"train": [], "val": []}
    best_val = float("inf")
    best_state = None
    wait = 0
    stopped_epoch = epochs

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        train_loss = loss_fn(model(x_train), y_train)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_value = loss_fn(model(x_train), y_train).item()
            val_value = loss_fn(model(x_val), y_val).item()

        hist["train"].append(train_value)
        hist["val"].append(val_value)

        if val_value < best_val:
            best_val = val_value
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if on_epoch:
            on_epoch(epoch, hist)

        if cfg.get("early_stopping", True) and wait >= patience:
            stopped_epoch = epoch + 1
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_probs = torch.softmax(model(x_val), dim=1)[:, 1]
        val_pred = (val_probs >= 0.5).long()

    cancers_found = int(((val_pred == 1) & (y_val == 1)).sum())
    cancers_total = int((y_val == 1).sum())
    false_alarms = int(((val_pred == 1) & (y_val == 0)).sum())

    return {
        "model": model,
        "hist": hist,
        "diagnosis": diagnose(hist),
        "epochs_run": stopped_epoch,
        "best_val_loss": best_val,
        "cancers_found": cancers_found,
        "cancers_total": cancers_total,
        "false_alarms": false_alarms,
    }


def evaluate_test(model, data):
    x_test, y_test = data[4], data[5]
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(x_test), dim=1)[:, 1]
        predictions = (probabilities >= 0.5).long()

    return {
        "accuracy": float((predictions == y_test).float().mean()),
        "positive_found": int(((predictions == 1) & (y_test == 1)).sum()),
        "positive_total": int((y_test == 1).sum()),
        "false_alarms": int(((predictions == 1) & (y_test == 0)).sum()),
    }
