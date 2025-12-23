import mlflow
import mlflow.sklearn
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Aktifkan autolog (BOLEH & AMAN)
mlflow.sklearn.autolog()

def train_model():
    # Path file dataset (satu folder dengan modelling.py)
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, "bank_marketing_preprocessing.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {file_path}")

    # Load data
    df = pd.read_csv(file_path)

    # Tentukan target
    target_col = "deposit" if "deposit" in df.columns else df.columns[-1]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # One-hot encoding untuk kolom kategorikal
    X = pd.get_dummies(X, drop_first=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model (TANPA start_run)
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"✅ Training selesai. Akurasi: {acc:.4f}")

    # Logging tambahan (opsional)
    mlflow.log_metric("accuracy_manual", acc)

if __name__ == "__main__":
    train_model()
