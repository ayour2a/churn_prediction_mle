"""Run after the feature and training pipelines; fixtures use local course artifacts."""
import json
import joblib
import numpy as np
import pandas as pd
import pytest
import src.training_pipeline.train as training
from src.utils.paths import RAW_DIR, PROCESSED_DIR, MODELS_DIR, PROJECT_ROOT
from src.feature_pipeline.preprocess import clean, prepare, transform_features, cat_cols
from src.feature_pipeline.validate import validate_data
from src.feature_pipeline.build import FEATURE_DIR
from src.inference_pipeline.predict import Predictor


def test_initial_candidate_does_not_use_tuned_parameters(tmp_path, monkeypatch):

    week4 = tmp_path / "week4"
    week4.mkdir()
    (week4 / "metadata.json").write_text(json.dumps({"candidate": "xgboost_initial"}))
    (week4 / "best_params.json").write_text(json.dumps({"n_estimators": 999}))
    monkeypatch.setattr(training, "MODELS_DIR", tmp_path)
    name, params = training.selected_parameters()
    assert name == "xgboost_initial" and params["n_estimators"] == 200

def test_tuned_candidate_uses_saved_search_parameters(tmp_path, monkeypatch):
    week4 = tmp_path / "week4"
    week4.mkdir()
    (week4 / "metadata.json").write_text(json.dumps({"candidate": "xgboost_tuned"}))
    (week4 / "best_params.json").write_text(json.dumps({"n_estimators": 300}))
    monkeypatch.setattr(training, "MODELS_DIR", tmp_path)
    name, params = training.selected_parameters()
    assert name == "xgboost_tuned" and params == {"n_estimators": 300}

def test_extracted_features_match_notebook_outputs():
    raw = pd.read_csv(RAW_DIR / "train.csv")
    expected = pd.read_csv(PROCESSED_DIR / "features_train.csv")
    columns = json.loads((MODELS_DIR / "feature_columns.json").read_text())
    encoder = joblib.load(FEATURE_DIR / "encoder.joblib")
    actual = transform_features(clean(raw), encoder, columns)
    np.testing.assert_allclose(actual.to_numpy(), expected[columns].to_numpy(), atol=1e-9)

@pytest.mark.parametrize("problem", ["negative", "duplicate", "category", "missing", "label"])
def test_bad_training_data_stops_validation(problem, tmp_path):
    bad = pd.read_csv(RAW_DIR / "train.csv").head(12).copy()
    if problem == "negative": bad.loc[bad.index[0], "tenure"] = -5
    if problem == "duplicate": bad.loc[bad.index[1], "customerID"] = bad.iloc[0].customerID
    if problem == "category": bad.loc[bad.index[0], "Contract"] = "Unknown"
    if problem == "missing": bad = bad.drop(columns="Contract")
    if problem == "label": bad.loc[bad.index[0], "Churn"] = "Maybe"
    report = tmp_path / "report.json"
    with pytest.raises(ValueError, match="validation failed"):
        validate_data(bad, report)
    assert json.loads(report.read_text())["success"] is False

def test_known_blank_is_fixed_before_cleaned_validation(tmp_path):
    raw = pd.read_csv(RAW_DIR / "train.csv").head(5).copy()
    raw["TotalCharges"] = raw["TotalCharges"].astype(str)
    raw.loc[raw.index[0], ["tenure", "TotalCharges"]] = [0, " "]
    validate_data(raw, tmp_path / "incoming.json")
    cleaned = clean(raw)
    validate_data(cleaned, tmp_path / "cleaned.json", cleaned=True)
    assert cleaned.iloc[0].TotalCharges == 0

def test_single_and_batch_predictions_agree():
    records = json.loads((PROJECT_ROOT / "examples/batch.json").read_text())["customers"]
    predictor = Predictor()
    batch = predictor.predict(records)
    for record, result in zip(records, batch):
        single = predictor.predict([record])[0]
        assert single == result
        assert 0 <= result["churn_probability"] <= 1
        assert all(factor["shap_log_odds"] > 0 for factor in result["top_risk_factors"])
