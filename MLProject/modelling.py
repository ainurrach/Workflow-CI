import os
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import argparse

# Aktifkan autolog MLflow
mlflow.sklearn.autolog()

# Ambil argument data_path
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help="Path ke dataset CSV hasil preprocessing")
args = parser.parse_args()
DATA_PATH = args.data_path

# Load dataset
df = pd.read_csv(DATA_PATH)

# Tentukan target
target_col = "y" if "y" in df.columns else "deposit"
X = df.drop(columns=[target_col])
y = df[target_col]

# Convert categorical features menjadi numerik
X = pd.get_dummies(X, drop_first=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Mulai MLflow run
with mlflow.start_run(run_name="CI_Training"):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluasi
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("Accuracy:", acc)
    print(classification_report(y_test, preds))

    # Log metrik tambahan manual (opsional)
    mlflow.log_metric("accuracy_manual", acc)

# Simpan model ke folder artifacts
artifacts_dir = "artifacts"
if not os.path.exists(artifacts_dir):
    os.makedirs(artifacts_dir)

model_path = os.path.join(artifacts_dir, "model_rf.pkl")
joblib.dump(model, model_path)
print(f"Model disimpan di {model_path}")
