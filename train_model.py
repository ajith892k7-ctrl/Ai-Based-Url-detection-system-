"""
train_model.py
---------------
Loads data/urls.csv, extracts features, trains Logistic Regression
(baseline) and Random Forest (main model), evaluates both, and saves
the best model to models/phishing_model.pkl
"""

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

from feature_extraction import build_feature_dataframe

DATA_PATH = "data/urls.csv"
MODEL_PATH = "models/phishing_model.pkl"
SCALER_PATH = "models/scaler.pkl"


def load_data():
    df = pd.read_csv(DATA_PATH)
    features_df = build_feature_dataframe(df["url"].tolist(), df["label"].tolist())
    return features_df


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print(f"\n--- {name} ---")
    print(f"Accuracy : {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds):.4f}")
    print(f"Recall   : {recall_score(y_test, preds):.4f}")
    print(f"F1 Score : {f1_score(y_test, preds):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, probs):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds, target_names=["Legit", "Phishing"]))

    return f1_score(y_test, preds)


def main():
    print("Loading data & extracting features...")
    df = load_data()
    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features (helps Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Baseline: Logistic Regression ---
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)
    f1_lr = evaluate("Logistic Regression (baseline)", log_reg, X_test_scaled, y_test)

    # --- Main model: Random Forest ---
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
    rf.fit(X_train, y_train)  # RF doesn't need scaling
    f1_rf = evaluate("Random Forest (main model)", rf, X_test, y_test)

    # Feature importance (useful for the report)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    print("\nTop 10 most important features (Random Forest):")
    print(importances.sort_values(ascending=False).head(10))

    # Save whichever performed better
    if f1_rf >= f1_lr:
        print("\n>> Random Forest selected as final model.")
        joblib.dump({"model": rf, "type": "rf", "columns": list(X.columns)}, MODEL_PATH)
    else:
        print("\n>> Logistic Regression selected as final model.")
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump({"model": log_reg, "type": "lr", "columns": list(X.columns)}, MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
