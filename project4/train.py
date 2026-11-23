import json
from pathlib import Path

import torch
from sklearn.datasets import load_digits
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from torch import nn


ROOT = Path(__file__).resolve().parent
SEED = 0


class DigitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


def main():
    torch.manual_seed(SEED)
    digits = load_digits()
    x = digits.data.astype("float32") / 16.0
    y = digits.target.astype("int64")
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.3, random_state=SEED, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_val = torch.from_numpy(x_val)
    y_val = torch.from_numpy(y_val)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)

    model = DigitModel()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_val = float("inf")
    best_state = None
    patience = 20
    wait = 0

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val), y_val).item()

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(x_test).argmax(1)
        test_accuracy = (pred == y_test).float().mean().item()
        recalls = recall_score(y_test.numpy(), pred.numpy(), average=None)

    torch.save(model.state_dict(), ROOT / "model.pt")
    config = {
        "input_size": 64,
        "hidden": [128, 64],
        "output_size": 10,
        "test_accuracy": round(test_accuracy, 4),
        "epochs_run": epoch + 1,
        "best_validation_loss": round(best_val, 6),
        "seed": SEED,
        "pixel_range": [0, 16],
        "class_recall": recalls.round(4).tolist(),
    }
    (ROOT / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
