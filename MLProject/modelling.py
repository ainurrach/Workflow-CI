import argparse
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Aktifkan autolog
mlflow.sklearn.autolog()

# Ambil argument dari MLProject
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

DATA_PATH = args.data_path

# Load dataset hasil preprocessing
df = pd.read_csv(DATA_PATH)

# Tentukan target
target_col = "y" if "y" in df.columns else "deposit"
X = df.drop(columns=[target_col])
y = df[target_col]

# Convert categorical features menjadi numerik
X = pd.get_dummies(X, drop_first=True)

# Split data
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

    # Prediksi & evaluasi
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("Accuracy:", acc)
    print(classification_report(y_test, preds))

    # Log metrik tambahan manual (optional)
    mlflow.log_metric("accuracy_manual", acc)
