import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from burnout_core.features.engg_features import engineer_features

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(SCRIPT_DIR / ".env")

BURNOUT_DATA_PATH = os.getenv("BURNOUT_DATA_PATH")

MODEL_DIR = SCRIPT_DIR / "models"
REPORT_DIR = SCRIPT_DIR / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


def main():
    print("Loading data...")
    df = pd.read_csv(BURNOUT_DATA_PATH)
    df = df.sample(n=50000, random_state=42)

    df.columns = df.columns.str.lower()
    df["risk_level"] = df["risk_level"].map({"Low": 0, "Medium": 1, "High": 2})
    df = pd.get_dummies(df, columns=["gender"], drop_first=True)
    df = engineer_features(df)

    X = df.drop(
        ["risk_level", "burnout_score", "mental_health_index", "dropout_risk"],
        axis=1
    )
    y = df["risk_level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=3000, class_weight="balanced"), True
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=60, max_depth=10, n_jobs=-1, random_state=42), False
        ),
        "Gradient Boosting": (
            HistGradientBoostingClassifier(max_iter=40, learning_rate=0.1, max_depth=2, random_state=42), False
        ),
        "XGBoost": (
            XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="mlogloss", random_state=42
            ), False
        )
    }

    print("Training models...")
    results = {}
    f1_scores = {}

    for name, (model, scaled) in models.items():
        print(f"  Training {name}...")
        Xtr, Xte = (X_train_scaled, X_test_scaled) if scaled else (X_train, X_test)
        model.fit(Xtr, y_train)
        preds = model.predict(Xte)
        results[name] = accuracy_score(y_test, preds)
        f1_scores[name] = f1_score(y_test, preds, average="weighted")

    best_model_name = max(f1_scores, key=f1_scores.get)
    best_model, best_needs_scaling = models[best_model_name]
    best_acc = results[best_model_name] * 100

    print(f"\nBest Model: {best_model_name} ({best_acc:.2f}%)")

    # Save model artifacts
    joblib.dump(best_model, MODEL_DIR / "burnout_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(X.columns.tolist(), MODEL_DIR / "features.pkl")

    # Save metadata (so the app doesn't need to retrain to know this)
    metadata = {
        "best_model_name": best_model_name,
        "best_acc": best_acc,
        "needs_scaling": best_needs_scaling
    }
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save feature importance (for the dashboard chart)
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importances = np.abs(best_model.coef_).mean(axis=0)
    else:
        importances = np.zeros(len(X.columns))

    importance_df = pd.DataFrame(
        {"Feature": X.columns, "Importance": importances}
    ).sort_values("Importance", ascending=False)

    importance_df.to_csv(REPORT_DIR / "feature_importance.csv", index=False)

    # Save risk-factor correlations (for the risk_factor_relationships chart)
    correlation = df[[
        "stress_level", "sleep_hours", "screen_time", "wellness_score", "risk_level"
    ]].corr()["risk_level"].drop("risk_level")

    correlation.to_csv(REPORT_DIR / "risk_correlation.csv", header=["correlation"])

    print("\nSaved: burnout_model.pkl, scaler.pkl, features.pkl, metadata.json")
    print("Saved: feature_importance.csv, risk_correlation.csv")


if __name__ == "__main__":
    main()
