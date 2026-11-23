# Project 5 - Live Hospital Training Dashboard

## Setup

```powershell
cd "C:\AI-PSEB Assignment\Assignment-5 Hassan Azeem"
python -m pip install -r requirements.txt
```

## Run

```powershell
streamlit run project5/app.py
```

The dashboard loads the scikit-learn Wisconsin breast cancer dataset. It uses a stratified 60% training, 20% validation, and 20% test split with seed 0. The positive class is malignant.

## Reproduce the report

```powershell
python project5/run_experiments.py
```

The script runs the four documented configurations and writes validation-only
metrics to `project5/experiment_results.csv`. It evaluates the final test split
once, after the four runs, and prints those metrics separately. The live dashboard
also records runs and can export its experiment table as CSV.

The training engine is in `trainer.py`, so it can be reused outside Streamlit.

The curve assessment is a teaching heuristic: training loss above 0.30 is labeled
underfitting; a validation-loss rebound above 30% from its minimum or a best
validation/training loss gap above 0.05 is labeled overfitting. These labels are
not clinical or statistical safety criteria.

## Limitations

- One historical, tabular dataset cannot establish clinical safety.
- The model is not calibrated for real-world use and the 0.5 threshold can be adjusted for sensitivity.
- Scaling, split, and seed choices limit direct generalisation to other hospitals or populations.
- A missed malignant case is more serious than a false alarm; all real clinical decisions require clinician review.
