import os
import sys
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

# -----------------------------
# Path configuration
# -----------------------------
current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_path)

# Allow imports from src
if current_script_path not in sys.path:
    sys.path.insert(0, current_script_path)

from preprocess import preprocess


def train(data_path=None, model_path=None):

    print("=" * 45)
    print("  Fraud Detection Model Training")
    print("=" * 45)

    # -----------------------------
    # Paths (safe + dynamic)
    # -----------------------------
    if data_path is None:
        data_path = os.path.join(project_root, "data", "transactions.csv")

    if model_path is None:
        model_path = os.path.join(project_root, "models", "fraud_model.pkl")

    print(f"\nUsing data from: {data_path}")
    print(f"Model will be saved to: {model_path}")

    # -----------------------------
    # Load & preprocess data
    # -----------------------------
    X, y = preprocess(data_path)

    # -----------------------------
    # Train-test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"\nTrain samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")

    # -----------------------------
    # Model training
    # -----------------------------
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # -----------------------------
    # Predictions
    # -----------------------------
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # -----------------------------
    # Metrics
    # -----------------------------
    print("\n" + "=" * 45)
    print("  Model Performance")
    print("=" * 45)

    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision : {precision_score(y_test, y_pred, zero_division=0):.2f}")
    print(f"Recall    : {recall_score(y_test, y_pred, zero_division=0):.2f}")
    print(f"F1 Score  : {f1_score(y_test, y_pred, zero_division=0):.2f}")
    print(f"ROC-AUC   : {roc_auc_score(y_test, y_prob):.2f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Legitimate", "Fraud"],
        zero_division=0
    ))

    # -----------------------------
    # Save model (FIXED)
    # -----------------------------
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump(model, model_path)

    print("\n✅ Model saved successfully!")
    print(f"📁 Path: {model_path}")

    return model


if __name__ == "__main__":
    train()