import joblib
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from xgboost import XGBClassifier


def build_models():

    return {
        "Logistic Regression": Pipeline([
            ("smote", SMOTE(random_state=42)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced"
                )
            )
        ]),

        "Random Forest": Pipeline([
            ("smote", SMOTE(random_state=42)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=60,
                    max_depth=10,
                    n_jobs=-1,
                    random_state=42
                )
            )
        ]),

        "Gradient Boosting": Pipeline([
            ("smote", SMOTE(random_state=42)),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=40,
                    learning_rate=0.1,
                    max_depth=2,
                    random_state=42
                )
            )
        ]),

        "XGBoost": Pipeline([
            ("smote", SMOTE(random_state=42)),
            (
                "model",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="mlogloss",
                    random_state=42
                )
            )
        ])
    }
