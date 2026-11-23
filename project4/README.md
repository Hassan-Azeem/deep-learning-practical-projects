# Project 4 - Digit Reader

This project trains a PyTorch digit classifier and serves it through FastAPI and Streamlit.

## Setup

```powershell
cd "C:\AI-PSEB Assignment\Assignment-5 Hassan Azeem"
python -m pip install -r requirements.txt
```

## Train

```powershell
python project4/train.py
```

This writes `project4/model.pt` and `project4/config.json`.
The trained checkpoint is included in this repository so the API can start without
retraining. Re-run the command above to reproduce the model.

## Run the API

```powershell
cd project4
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` to test `/predict`. A valid request contains exactly 64 pixel values from 0 to 16.

## Run the web page

In a second terminal:

```powershell
cd "C:\AI-PSEB Assignment\Assignment-5 Hassan Azeem"
streamlit run project4/ui.py
```

## Screenshots and results

The held-out test accuracy is stored in `config.json`. Screenshots of the web
prediction and API documentation are in `screenshots/`.

The API accepts exactly 64 pixel values between 0 and 16. Test an invalid payload
in `/docs` to see its validation error.

## Limitations

1. The model uses 8x8 scanned digits only; normal handwriting photos may fail.
2. Performance is limited to the scikit-learn digits dataset and the fixed 0-16 pixel range.
3. A prediction is a digit confidence estimate, not a formal quality or handwriting certification.
