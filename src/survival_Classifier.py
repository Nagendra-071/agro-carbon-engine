import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Dynamic path resolution relative to this script's directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "updated_crop_recommendation.csv")
EXPORT_DIR = os.path.join(BASE_DIR, "exported_models")
MODEL_PATH = os.path.join(EXPORT_DIR, "survival_model.pkl")


def train_and_save_model():
    """Loads dataset, trains Random Forest model, prints metrics, and exports pickle artifact."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset file not found at: {DATA_PATH}")

    # 1. Load Data
    print(f"Loading dataset from: {DATA_PATH}")
    raw_df = pd.read_csv(DATA_PATH)
    
    X = raw_df.iloc[:, :7]
    y = raw_df['label']

    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Train Classifier
    print("Training Random Forest Classifier...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # 4. Evaluate Model
    ypred = rf.predict(X_test)
    accuracy_val = accuracy_score(y_test, ypred)
    print(f"\nModel Accuracy: {accuracy_val * 100:.2f}%\n")
    
    class_report = classification_report(y_test, ypred)
    print("Classification Report:")
    print(class_report)

    # 5. Export Trained Model
    os.makedirs(EXPORT_DIR, exist_ok=True)
    joblib.dump(rf, MODEL_PATH)
    print(f"Model saved successfully to: {MODEL_PATH}\n")
    return rf


def predict_sample_farmer_input(model, farmer_input_df, threshold=0.05):
    """Generates survival probability scores filtered by confidence threshold (>0.05)."""
    species_list = model.classes_
    raw_probabilities = model.predict_proba(farmer_input_df)[0]

    # Pair and sort species with probability scores
    survival_scores = dict(zip(species_list, raw_probabilities))
    sorted_scores = dict(sorted(survival_scores.items(), key=lambda item: item[1], reverse=True))

    # Filter with threshold (> 0.05)
    clean_scores = {
        species: float(round(score, 4))
        for species, score in sorted_scores.items()
        if score > threshold
    }
    return clean_scores


if __name__ == "__main__":
    # Train and export model artifact
    trained_model = train_and_save_model()

    # Test sample farmer input logic
    print("--- Testing Sample Prediction ---")
    sample_farmer_input = pd.DataFrame([{
        'N': 35,
        'P': 25,
        'K': 30,
        'temperature': 25.0,
        'humidity': 55.0,
        'ph': 7.0,
        'rainfall': 800.0
    }])

    filtered_predictions = predict_sample_farmer_input(trained_model, sample_farmer_input, threshold=0.05)
    print("Viable Crops (> 5% viability threshold):")
    print(filtered_predictions)