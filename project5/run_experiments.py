import csv
from pathlib import Path

import trainer


ROOT = Path(__file__).resolve().parent
COMMON = {
    "learning_rate": 0.001,
    "epochs": 150,
    "patience": 20,
    "scaling": True,
}
EXPERIMENTS = [
    ("256 baseline", {**COMMON, "hidden": 256, "dropout": 0.0, "weight_decay": 0.0, "early_stopping": False}),
    ("256 regularized", {**COMMON, "hidden": 256, "dropout": 0.3, "weight_decay": 0.05, "early_stopping": False}),
    ("256 early stopping", {**COMMON, "hidden": 256, "dropout": 0.0, "weight_decay": 0.0, "early_stopping": True}),
    ("16 regularized + early stopping", {**COMMON, "hidden": 16, "dropout": 0.3, "weight_decay": 0.05, "early_stopping": True}),
]


def main():
    data, _ = trainer.load_data(scale=True)
    rows = []
    selected = None

    for name, config in EXPERIMENTS:
        result = trainer.train(config, data=data)
        best_epoch = min(
            range(len(result["hist"]["val"])),
            key=result["hist"]["val"].__getitem__,
        )
        row = {
            "experiment": name,
            "hidden_units": config["hidden"],
            "dropout": config["dropout"],
            "weight_decay": config["weight_decay"],
            "early_stopping": config["early_stopping"],
            "epochs_run": result["epochs_run"],
            "curve_assessment": result["diagnosis"],
            "train_loss_at_best_epoch": round(result["hist"]["train"][best_epoch], 4),
            "best_validation_loss": round(result["best_val_loss"], 4),
            "positive_cases_found": result["cancers_found"],
            "positive_cases_total": result["cancers_total"],
            "false_alarms": result["false_alarms"],
        }
        rows.append(row)
        selected = result
        print(row)

    final_test = trainer.evaluate_test(selected["model"], data)
    print({
        "final_test_accuracy": round(final_test["accuracy"], 4),
        "positive_cases_found": final_test["positive_found"],
        "positive_cases_total": final_test["positive_total"],
        "false_alarms": final_test["false_alarms"],
    })

    output = ROOT / "experiment_results.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved validation results to {output}")


if __name__ == "__main__":
    main()
