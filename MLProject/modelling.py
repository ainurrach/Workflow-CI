import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ===============================
# Konfigurasi MLflow (WAJIB di CI)
# ===============================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("CI_Training_Experiment")
mlflow.sklearn.autolog()

def train_model():
    print("Start training...")
    print("Current working directory:", os.getcwd())
    print("Files in directory:", os.listdir())

    DATA_PATH = "bank_marketing_preprocessing.csv"

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset '{DATA_PATH}' tidak ditemukan. "
            f"Pastikan file ada di root repository."
        )

    df = pd.read_csv(DATA_PATH)
    df = df.dropna()

    target_col = "y" if "y" in df.columns else "deposit"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' tidak ditemukan")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name="RandomForest_CI"):
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_metric("accuracy", acc)

        print("Training finished successfully")
        print("Accuracy:", acc)

if __name__ == "__main__":
    train_model()

