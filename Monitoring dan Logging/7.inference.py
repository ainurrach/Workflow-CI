# File inference untuk serving model
import mlflow
import mlflow.sklearn
import pandas as pd
import argparse


def load_model(model_uri: str):
    """Load model dari MLflow"""
    return mlflow.sklearn.load_model(model_uri)


def load_data(csv_path: str):
    """Load dataset inference"""
    df = pd.read_csv(csv_path)
    return df


def run_inference(model, df: pd.DataFrame):
    """Jalankan prediksi"""
    predictions = model.predict(df)
    return predictions


def save_predictions(predictions, output_path: str):
    """Simpan hasil prediksi"""
    result = pd.DataFrame({"prediction": predictions})
    result.to_csv(output_path, index=False)
    print(f"Hasil inference disimpan di: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference menggunakan MLflow Model")

    parser.add_argument(
        "--model-uri",
        type=str,
        required=True,
        help="Contoh: runs:/<run_id>/model atau models:/ModelName/Production"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="bank_marketing_preprocessing.csv",
        help="Path dataset preprocessing"
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default="prediction_bank_marketing.csv",
        help="File output prediksi"
    )

    args = parser.parse_args()

    print("=== LOAD MODEL ===")
    model = load_model(args.model_uri)

    print("=== LOAD DATA ===")
    data = load_data(args.data_path)

    print("=== RUN INFERENCE ===")
    preds = run_inference(model, data)

    save_predictions(preds, args.output_path)
