from pathlib import Path
import urllib.request
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

def download_dataset(destination=Path("data/heart.csv")):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, destination)
    return destination

def train_and_save(data_path=Path("data/heart.csv"), model_path=Path("heart_model.joblib")):
    df = pd.read_csv(data_path, header=None, names=COLUMNS, na_values="?")
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["target"])
    X = df[COLUMNS[:-1]]
    # Original target is 0 (absence) or 1–4 (presence); convert to binary.
    y = (df["target"] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        ))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, predictions), 3),
        "precision": round(precision_score(y_test, predictions, zero_division=0), 3),
        "recall": round(recall_score(y_test, predictions, zero_division=0), 3),
        "f1_score": round(f1_score(y_test, predictions, zero_division=0), 3),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "test_records": int(len(y_test))
    }
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metrics": metrics}, model_path)
    print("Model saved to", model_path)
    print("Test metrics:", metrics)
    return model, metrics

if __name__ == "__main__":
    path = Path("data/heart.csv")
    if not path.exists():
        print("Downloading UCI Cleveland Heart Disease dataset...")
        download_dataset(path)
    train_and_save(path)
