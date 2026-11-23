import json
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel
try:
    from .train import DigitModel
except ImportError:
    from train import DigitModel


ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
model = DigitModel()
model.load_state_dict(torch.load(ROOT / "model.pt", map_location="cpu", weights_only=True))
model.eval()

app = FastAPI(title="Digit Reader API", version="1.0")


class Image(BaseModel):
    pixels: list[float]


@app.get("/")
def root():
    return {"message": "Digit Reader API", "docs": "/docs"}


@app.post("/predict")
def predict(img: Image):
    if len(img.pixels) != CONFIG["input_size"]:
        return {"error": "send exactly 64 pixel values"}
    if any(value < 0 or value > 16 for value in img.pixels):
        return {"error": "pixel values must be between 0 and 16"}

    x = torch.tensor(img.pixels, dtype=torch.float32).reshape(1, 64) / 16.0
    with torch.no_grad():
        probabilities = torch.softmax(model(x), dim=1)[0]
    return {
        "digit": int(probabilities.argmax()),
        "confidence": round(float(probabilities.max()), 3),
    }
