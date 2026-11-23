# Project 5 Report

## Objective

Build a live Streamlit dashboard that makes overfitting visible and allows the data team to compare training settings.

## Method

The Wisconsin diagnostic dataset was divided with a stratified 60/20/20 split
(train/validation/test), random seed 0. The target positive class is malignant.
All runs below use the same split, learning rate 0.001, and maximum of 150 epochs.
The final test split is reported only for the last, preselected configuration.

The dashboard's curve assessment is a teaching heuristic: training loss above
0.30 indicates underfitting; validation loss rebounding by more than 30% from its
minimum or a best validation/training loss gap above 0.05 indicates overfitting.

## Validation experiments

| Run | Configuration | Curve assessment | Train-at-best / best validation loss | Malignant cases found | False alarms | Epochs |
|---|---|---|---:|---:|---:|---:|
| 1 | 256 units, no dropout/decay, no early stopping | Overfitting | 0.0562 / 0.1203 | 40/43 | 3 | 150 |
| 2 | 256 units, dropout 0.3, decay 0.05, no early stopping | Overfitting | 0.0590 / 0.1221 | 40/43 | 3 | 150 |
| 3 | 256 units, early stopping (patience 20) | Overfitting | 0.0562 / 0.1203 | 40/43 | 3 | 92 |
| 4 | 16 units, dropout 0.3, decay 0.05, early stopping | Stable | 0.1167 / 0.1611 | 39/43 | 1 | 150 |

These results are reproducible with `python project5/run_experiments.py`; the
validation metrics are also saved to `project5/experiment_results.csv`.

## Final test

The 16-unit regularized configuration was evaluated once on the held-out test
split after the comparison: **95.61% accuracy**, **41/42 malignant cases found**,
and **4 false alarms**. It reduced false alarms compared with the larger model
but had lower validation recall (39/43 vs. 40/43); it does not dominate on every
metric.

## Interpretation

Early stopping reduced the 256-unit run from 150 to 92 epochs without reducing
the number of malignant validation cases found, but the training/validation gap
remained. Dropout and weight decay alone did not materially improve this run's
validation metrics. The smaller model was assessed as stable and traded one
additional missed validation case for two fewer false alarms.

## Limitations

1. These measurements come from one public, historical dataset and one fixed split.
2. The final test missed one of 42 malignant benchmark samples; this is not acceptable evidence for real screening.
3. There is no external-hospital, population, or prospective validation, and the model is not clinically calibrated.
4. The 0.5 probability threshold and curve-assessment heuristic are teaching choices, not clinical recommendations.
