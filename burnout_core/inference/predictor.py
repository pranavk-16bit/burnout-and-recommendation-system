import joblib
import pandas as pd

from burnout_core.config import MODEL_PATH, MODEL_DIR
from burnout_core.features.engg_features import engineer_features
from sklearn.metrics import accuracy_score, f1_score


RISK_LABELS = {
    0: "Low",
    1: "Medium",
    2: "High"
}


class BurnoutPredictor:

    def __init__(self, model_path=None, features_path=None):
        model_path = model_path or MODEL_PATH
        features_path = features_path or (MODEL_DIR / "features.pkl")

        self.model = joblib.load(model_path)
        self.feature_names = joblib.load(features_path)

    def predict(self, student_data):
        features = engineer_features(student_data)

        input_df = pd.DataFrame([features])
        input_df = input_df.reindex(
            columns=self.feature_names,
            fill_value=0
        )

        prediction = self.model.predict(input_df)[0]
        probs = self.model.predict_proba(input_df)[0]

        return {
            "risk": RISK_LABELS[prediction],
            "confidence": float(probs.max() * 100),
            "probabilities": probs.tolist()
        }


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    return {
        "accuracy": accuracy,
        "f1": f1
    }
