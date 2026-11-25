import json
import shutil
from datetime import datetime, timezone
from importlib.metadata import version
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from src.utils.paths import MODELS_DIR, PROCESSED_DIR
from src.feature_pipeline.build import FEATURE_DIR


BUNDLE_DIR = MODELS_DIR / "week5_serving"


def selected_parameters():

    week4 = MODELS_DIR / "week4"
    selected = json.loads((week4 / "metadata.json").read_text())["candidate"]
    if selected == "xgboost_tuned":
        params = json.loads((week4 / "best_params.json").read_text())
    elif selected == "xgboost_initial":
        # These are the explicit initial settings from notebook 04, not the tuned settings.
        params = {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3,
                  "subsample": 1.0, "colsample_bytree": 1.0, "min_child_weight": 1}
    else:
        raise ValueError(f"Unknown Week 4 candidate: {selected}")

    return selected, params


def run():
    columns = json.loads((FEATURE_DIR / "feature_columns.json").read_text())
    train = pd.read_csv(PROCESSED_DIR / "week5_features_train.csv")
    evaluation = pd.read_csv(PROCESSED_DIR / "week5_features_eval.csv")
    selected, params = selected_parameters()
    model = XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                          device="cpu", random_state=42, n_jobs=2, **params)
    model.fit(train[columns], train["Churn"])
    probability = model.predict_proba(evaluation[columns])[:, 1]
    prediction = (probability >= 0.5).astype(int)
    # prediction = model.predict(evaluation[columns])
    y = evaluation["Churn"]
    metrics = {"roc_auc": roc_auc_score(y, probability),
               "precision": precision_score(y, prediction, zero_division=0),
               "recall": recall_score(y, prediction), "f1": f1_score(y, prediction)}
    metadata = {"model_version": datetime.now(timezone.utc).strftime("churn-%Y%m%dT%H%M%S%fZ"),
                "threshold": 0.5, "source_candidate": selected, "parameters": params,
                "eval_metrics": metrics, "holdout_used": False,
                "versions": {p: version(p) for p in ["xgboost", "scikit-learn", "pandas", "numpy", "shap"]}}
    # Copy fitted feature artifacts only after successful training, so serving uses its own bundle.
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(BUNDLE_DIR / "model.json")
    for name in ["encoder.joblib", "feature_columns.json"]:
        shutil.copy2(FEATURE_DIR / name, BUNDLE_DIR / name)
    (BUNDLE_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metrics, indent=2))
    print("Serving bundle:", BUNDLE_DIR)


if __name__ == "__main__":
    run()
