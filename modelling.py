import mlflow
import mlflow.sklearn
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Menggunakan nama eksperimen yang ada di dashboard Anda
mlflow.set_experiment("Bank_Marketing_Colab")
mlflow.sklearn.autolog()

def train_model():
    # Mencari file di folder yang sama dengan script
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, "bank_marketing_preprocessing.csv")

    if not os.path.exists(file_path):
        print(f"❌ File tidak ditemukan di {file_path}")
        return

    # Load data
    df = pd.read_csv(file_path)
    
    # Memisahkan fitur dan target
    if "deposit" not in df.columns:
        # Jika kolom target bukan 'deposit', sesuaikan dengan nama kolom terakhir
        target_col = df.columns[-1] 
    else:
        target_col = "deposit"

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Mengonversi kolom kategorikal/objek menjadi numerik (One-Hot Encoding)
    # Ini menangani kolom 'boolean' agar bisa diproses RandomForest
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name="RandomForest_Final"):
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"✅ Training Selesai. Akurasi: {acc:.4f}")
        mlflow.log_metric("accuracy_manual", acc)

if __name__ == "__main__":
    train_model()
    train(**vars(args))
