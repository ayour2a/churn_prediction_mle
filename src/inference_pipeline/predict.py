import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier
from src.utils.paths import MODELS_DIR
from src.feature_pipeline.preprocess import clean, transform_features, CUSTOMER_COLUMNS, DOMAINS


BUNDLE_DIR = MODELS_DIR / "week5_serving"


class Predictor:
    def __init__(self, bundle_dir=BUNDLE_DIR):
        bundle_dir = Path(bundle_dir)
        self.metadata = json.loads((bundle_dir / "metadata.json").read_text())
        self.columns = json.loads((bundle_dir / "feature_columns.json").read_text())
        self.encoder = joblib.load(bundle_dir / "encoder.joblib")
        self.model = XGBClassifier()
        self.model.load_model(bundle_dir / "model.json")
        if self.model.get_booster().feature_names != self.columns:
            raise ValueError("Saved model feature order does not match the manifest.")
        self.explainer = shap.TreeExplainer(self.model, model_output="raw", feature_perturbation="tree_path_dependent")

    def predict(self, records):
        if not 1 <= len(records) <= 100:
            raise ValueError("Supply between 1 and 100 customers.")
        frame = pd.DataFrame(records)
        if set(frame.columns) != set(CUSTOMER_COLUMNS):
            raise ValueError("Supply the documented customer fields, without Churn or extra fields.")
        if frame.isna().any().any():
            raise ValueError("Customer fields must be present.")
        frame["customerID"] = frame["customerID"].astype(str).str.strip()
        if not frame["customerID"].is_unique:
            raise ValueError("Customer fields must be present and IDs unique within the request.")
        if frame["customerID"].eq("").any():
            raise ValueError("Customer IDs must not be blank.")
        for column, allowed in DOMAINS.items():
            if not frame[column].isin(allowed).all():
                raise ValueError(f"Invalid category in {column}")
        for column in ["tenure", "MonthlyCharges", "TotalCharges"]:
            values = pd.to_numeric(frame[column], errors="raise")
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"{column} must be finite and non-negative.")
            frame[column] = values
        cleaned = clean(frame)
        features = transform_features(cleaned, self.encoder, self.columns)
        probabilities = self.model.predict_proba(features)[:, 1]
        explanation = self.explainer(features)
        raw = explanation.base_values + explanation.values.sum(axis=1)
        np.testing.assert_allclose(1 / (1 + np.exp(-raw)), probabilities, atol=1e-5)
        threshold = self.metadata["threshold"]
        results = []
        for i, probability in enumerate(probabilities):
            contributions = explanation.values[i]
            order = np.argsort(contributions)[::-1]
            factors = [{"feature": self.columns[j], "value": float(features.iloc[i, j]),
                        "shap_log_odds": float(contributions[j])}
                       for j in order if contributions[j] > 0][:3]
            results.append({"customerID": str(cleaned.iloc[i]["customerID"]),
                "churn_probability": float(probability), "prediction": int(probability >= threshold),
                "threshold": threshold, "model_version": self.metadata["model_version"],
                "top_risk_factors": factors, "explanation_base_log_odds": float(explanation.base_values[i]),
                "explanation_score_log_odds": float(raw[i])})
        return results

def main():
    parser = argparse.ArgumentParser(description="Score raw customer details using saved artifacts.")
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    if isinstance(data, dict) and "customers" in data:
        records = data["customers"]
    else:
        records = data if isinstance(data, list) else [data]
    print(json.dumps(Predictor().predict(records), indent=2))

if __name__ == "__main__":
    main()
