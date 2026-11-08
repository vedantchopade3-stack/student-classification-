
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score
)

DATA_PATH = Path("data/ecotec_online_student_risk.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "TARGET_RISK"
ID_COLUMNS = ["PUBLIC_RECORD_ID", "student_id", "id"]

def train():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Run python dataset_download.py first."
        )

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    if TARGET not in df.columns:
        raise ValueError(
            f"Target '{TARGET}' not found. Actual columns:\n"
            f"{df.columns.tolist()}"
        )

    df = df.dropna(subset=[TARGET]).copy()

    # Remove identifiers and target from predictor columns.
    excluded = [c for c in ID_COLUMNS if c in df.columns] + [TARGET]
    X = df.drop(columns=excluded)
    y_raw = df[TARGET].astype(str).str.strip()

    if y_raw.nunique() < 2:
        raise ValueError("The target must contain at least two classes.")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [
        c for c in X.columns if c not in numeric_features
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        )
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    best_name = None
    best_model = None
    best_cv_score = -1

    print("\nComparing models using 5-fold CV...")

    for name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessing", preprocessing),
            ("classifier", estimator)
        ])

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1
        )

        mean_score = scores.mean()
        print(f"{name}: CV macro F1 = {mean_score:.4f}")

        if mean_score > best_cv_score:
            best_cv_score = mean_score
            best_name = name
            best_model = pipeline

    print(f"\nBest model: {best_name}")
    best_model.fit(X_train, y_train)

    predictions = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")

    report = classification_report(
        y_test,
        predictions,
        labels=list(range(len(label_encoder.classes_))),
        target_names=label_encoder.classes_.tolist(),
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=list(range(len(label_encoder.classes_)))
    ).tolist()

    joblib.dump(best_model, MODEL_DIR / "student_model.pkl")

    metadata = {
        "target": TARGET,
        "features": X.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "classes": label_encoder.classes_.tolist(),
        "class_mapping": {
            str(label): int(index)
            for index, label in enumerate(label_encoder.classes_)
        },
        "accuracy": float(accuracy),
        "cv_macro_f1": float(best_cv_score),
        "test_macro_f1": float(macro_f1),
        "classification_report": report,
        "confusion_matrix": matrix,
        "best_model": best_name,
        "records": int(len(df)),
        "feature_count": int(X.shape[1])
    }

    with open(MODEL_DIR / "model_metadata.json", "w") as file:
        json.dump(metadata, file, indent=4)

    print("\nTraining completed.")
    print(f"Records: {len(df)}")
    print(f"Features: {X.shape[1]}")
    print(f"Test accuracy: {accuracy:.2%}")
    print(f"CV macro F1: {best_cv_score:.4f}")
    print(f"Test macro F1: {macro_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_,
        zero_division=0
    ))
    print("Confusion matrix:")
    print(matrix)
    print("\nSaved model in:", MODEL_DIR)

if __name__ == "__main__":
    train()
