import os, pickle
from src.data_generator import generate_dataset
from src.model_utils import train_model

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    df = generate_dataset(1200, seed=42)
    df.to_csv("data/loan_applications_demo.csv", index=False)
    bundle = train_model(df)
    with open("models/loan_model.pkl", "wb") as f:
        pickle.dump(bundle, f)
    print("Model trained and saved: models/loan_model.pkl")
    print("Metrics:")
    for key, value in bundle["metrics"].items():
        if key != "classification_report" and key != "confusion_matrix":
            print(f"{key}: {value:.4f}")
