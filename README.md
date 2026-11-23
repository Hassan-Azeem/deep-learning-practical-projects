# Deep Learning Practical Projects

Two small, reproducible projects for model training and deployment practice.
Both use public scikit-learn datasets and contain no uploaded patient records,
credentials, or personal account information.

## Projects

### Project 4 — Digit Reader

Trains a PyTorch classifier on 8x8 digit images and exposes predictions through a
FastAPI endpoint and a Streamlit page.

- [Project 4 setup and limitations](./project4/README.md)
- [FastAPI implementation](./project4/api.py)
- [Web interface](./project4/ui.py)
- Screenshot: [digit reader prediction](./screenshots/project4_digit_reader.png)
- Screenshot: [API documentation](./screenshots/project4_api_docs.png)

### Project 5 — Live Training Dashboard

Compares model size, dropout, weight decay, and early stopping on a stratified
Wisconsin breast cancer benchmark. It plots train/validation loss, reports
validation errors, and exports experiment history.

- [Project 5 setup and limitations](./project5/README.md)
- [Measured experiment report](./project5/REPORT.md)
- [Reproducible experiment runner](./project5/run_experiments.py)
- Screenshot: [training dashboard and metrics](./screenshots/project5_dashboard.png)
- Screenshot: [training and validation loss curves](./screenshots/project5_learning_curve.png)

## Setup

Use Python 3.10 or later:

```powershell
python -m pip install -r requirements.txt
```

Run Project 4 in separate terminals:

```powershell
run_project4_api.bat
run_project4_ui.bat
```

Run Project 5:

```powershell
run_project5.bat
```

Reproduce the Project 5 validation table and final selected test result:

```powershell
python project5/run_experiments.py
```

## Safety and limitations

These are educational prototypes, not clinical tools. The breast-cancer model
must not be used for diagnosis, screening, triage, or treatment decisions. Both
projects are limited by their small public benchmark datasets and do not
establish real-world performance.
