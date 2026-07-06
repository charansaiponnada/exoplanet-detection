"""
Layer 5 (interim): a learned classifier trained on the Layer-4 feature
vectors, standing in for the CNN+Transformer+BiLSTM/Mamba model until a
real labeled dataset (ISRO's curated set, see README) is available to
train that. This is a small, fully auditable Random Forest -- every
prediction can be explained via the model's feature importances, which is
exactly the "explainability" the PS asks for, just without a deep net
behind it yet.

It runs *alongside* the classical vetting tests in vetting.py, not instead
of them: `main.py` reports both, so a disagreement between the rule-based
label and the learned one is itself useful signal for a human reviewer.

Usage:
    uv run src/classifier.py --data data/synthetic_training_set.csv \
        --model models/candidate_classifier.joblib
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

FEATURE_COLUMNS = [
    "transit_count", "n_transits_measured", "depth_ppm", "duration_hours",
    "period_days", "depth_snr", "noise_ppm", "depth_consistency_std_ppm",
    "depth_consistency_frac", "symmetry_ppm", "ingress_egress_slope",
    "shape_ratio", "odd_even_mismatch_frac", "secondary_depth_ppm",
    "skewness", "kurtosis", "entropy", "red_noise_ratio",
]


def train(data_path, model_path, test_size=0.25, seed=0):
    df = pd.read_csv(data_path)
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=300, max_depth=6,
                                        class_weight="balanced", random_state=seed)),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)

    importances = dict(zip(FEATURE_COLUMNS, pipeline["clf"].feature_importances_.tolist()))
    importances = dict(sorted(importances.items(), key=lambda kv: -kv[1]))

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "feature_columns": FEATURE_COLUMNS,
                 "classes": list(pipeline.classes_)}, model_path)

    metrics = {
        "n_train": len(X_train), "n_test": len(X_test),
        "accuracy": report["accuracy"],
        "per_class_f1": {k: v["f1-score"] for k, v in report.items() if k in pipeline.classes_},
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": list(pipeline.classes_),
        "feature_importances": importances,
    }
    return metrics


def load_model(model_path="models/candidate_classifier.joblib"):
    return joblib.load(model_path)


def predict(model_bundle, features: dict):
    """
    Score one candidate's Layer-4 feature dict with the trained model.
    Returns class probabilities plus the globally most-important features,
    so a caller can show *why* the model leans the way it does.
    """
    pipeline = model_bundle["pipeline"]
    cols = model_bundle["feature_columns"]
    row = pd.DataFrame([{c: features.get(c) for c in cols}])
    proba = pipeline.predict_proba(row)[0]
    classes = model_bundle["classes"]

    importances = dict(zip(cols, pipeline["clf"].feature_importances_.tolist()))
    top_features = sorted(importances.items(), key=lambda kv: -kv[1])[:5]

    return {
        "predicted_label": classes[int(np.argmax(proba))],
        "probabilities": dict(zip(classes, [float(p) for p in proba])),
        "top_contributing_features": [{"feature": f, "importance": round(imp, 4)} for f, imp in top_features],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/synthetic_training_set.csv")
    parser.add_argument("--model", type=str, default="models/candidate_classifier.joblib")
    args = parser.parse_args()

    metrics = train(args.data, args.model)
    print(json.dumps(metrics, indent=2))
    print(f"\nModel saved to {args.model}")


if __name__ == "__main__":
    main()
