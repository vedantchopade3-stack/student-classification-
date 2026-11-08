
import joblib
import json
import pandas as pd
from pathlib import Path

MODEL_DIR = Path("models")

def load_model():
    model = joblib.load(MODEL_DIR / "student_model.pkl")

    with open(MODEL_DIR / "model_metadata.json", "r") as file:
        metadata = json.load(file)

    return model, metadata


def predict_students(data):
    model, metadata = load_model()

    if isinstance(data, dict):
        data = pd.DataFrame([data])
    else:
        data = pd.DataFrame(data)

    features = metadata["features"]

    missing = [column for column in features if column not in data.columns]
    if missing:
        raise ValueError(f"Missing input features: {missing}")

    data = data[features].copy()

    predictions = model.predict(data)
    classes = metadata["classes"]

    results = data.copy()
    results["Predicted_Risk"] = [
        classes[int(prediction)] for prediction in predictions
    ]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(data)

        for index, label in enumerate(classes):
            results[f"Probability_{label}"] = probabilities[:, index]

    return results
