import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load dataset
df = pd.read_csv("data/uci_student_outcomes.csv")

# Load model metadata
with open("models/model_metadata.json") as f:
    metadata = json.load(f)

features = metadata["features"]
target = metadata["target"]

# Prepare data
df = df.dropna(subset=[target])

X = df[features]
y = df[target].astype(str)

# Same test split as training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Load trained model
model = joblib.load("models/student_model.pkl")

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nMODEL ACCURACY")
print("=" * 40)

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, predictions))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, predictions))