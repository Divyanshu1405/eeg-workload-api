"""
Training and Subject-Independent Evaluation Pipeline for UNIVERSE EEG Workload Estimation.
Trains Random Forest and XGBoost classifiers on extracted features, evaluates generalization
to unseen participants, and exports the approved model to models/production/.
"""

import argparse
import json
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

from training.models.xgb_wrapper import XGBWrapper

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainAndEvaluate")


def load_dataset_splits(processed_dir: Path):
    train_path = processed_dir / "universe_features_train.csv"
    val_path = processed_dir / "universe_features_val.csv"
    test_path = processed_dir / "universe_features_test.csv"

    if not train_path.exists() or not test_path.exists():
        logger.warning("Processed feature CSVs not found in %s. Running preprocessor...", processed_dir)
        from scripts.preprocess_universe import run_pipeline
        raw_dir = PROJECT_ROOT / "data" / "raw" / "universe"
        run_pipeline(raw_dir=raw_dir, output_dir=processed_dir, generate_mock_if_empty=True)

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path) if val_path.exists() else pd.DataFrame()
    df_test = pd.read_csv(test_path)
    return df_train, df_val, df_test


def prepare_features_and_labels(df: pd.DataFrame):
    if df is None or df.empty:
        return np.array([]), np.array([]), np.array([]), []

    meta_cols = {
        "participant_id", "session", "window_index", "start_time", "end_time",
        "task_name", "workload_level", "workload_score", "quality_score",
    }
    feature_cols = [c for c in df.columns if c not in meta_cols]
    X = df[feature_cols].values
    y = df["workload_level"].values
    participants = df["participant_id"].values
    return X, y, participants, feature_cols


def evaluate_model(model, X_test, y_test, participants_test):
    if len(X_test) == 0:
        return {"accuracy": 0.0, "f1_macro": 0.0}

    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    conf_mat = confusion_matrix(y_test, y_pred).tolist()

    per_participant = {}
    for p in np.unique(participants_test):
        mask = participants_test == p
        if np.sum(mask) > 0:
            p_acc = float(accuracy_score(y_test[mask], y_pred[mask]))
            per_participant[str(p)] = {"accuracy": p_acc, "samples": int(np.sum(mask))}

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "confusion_matrix": conf_mat,
        "per_participant": per_participant,
    }


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate baseline ML models for EEG workload.")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "universe")
    parser.add_argument("--model-type", type=str, default="random_forest", choices=["random_forest", "xgboost"])
    parser.add_argument("--output-model-dir", type=Path, default=PROJECT_ROOT / "models" / "production")
    parser.add_argument("--experiments-dir", type=Path, default=PROJECT_ROOT / "experiments")
    args = parser.parse_args()

    logger.info("Loading dataset splits from %s...", args.processed_dir)
    df_train, df_val, df_test = load_dataset_splits(args.processed_dir)

    X_train, y_train, p_train, feature_cols = prepare_features_and_labels(df_train)
    X_val, y_val, p_val, _ = prepare_features_and_labels(df_val)
    X_test, y_test, p_test, _ = prepare_features_and_labels(df_test)

    if len(X_train) == 0 or len(X_test) == 0:
        logger.error("Training or test set is empty. Exiting.")
        return

    logger.info("Training data shape: %s | Test data shape: %s", X_train.shape, X_test.shape)
    logger.info("Classes present: %s", np.unique(y_train))

    if args.model_type == "random_forest":
        logger.info("Training Random Forest Classifier (n_jobs=-1)...")
        model = RandomForestClassifier(
            n_estimators=100, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
    else:
        logger.info("Training XGBoost Classifier (n_jobs=-1)...")
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        xgb_model = XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1
        )
        xgb_model.fit(X_train, y_train_enc)
        model = XGBWrapper(xgb_model, le)

    logger.info("Evaluating model on subject-independent test partition...")
    metrics = evaluate_model(model, X_test, y_test, p_test)
    logger.info("Test Results -> Accuracy: %.3f | Macro F1: %.3f", metrics["accuracy"], metrics["f1_macro"])

    args.experiments_dir.mkdir(parents=True, exist_ok=True)
    exp_path = args.experiments_dir / f"evaluation_{args.model_type}.json"
    with open(exp_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved evaluation report to %s", exp_path)

    args.output_model_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_model_dir / f"universe_{args.model_type}_prod.joblib"
    joblib.dump(model, model_path)
    logger.info("Exported production model to %s", model_path)

    metadata = {
        "model_type": args.model_type,
        "features_used": feature_cols,
        "feature_dim": len(feature_cols),
        "classes": list(np.unique(y_train)),
        "test_metrics": {
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
        },
    }
    with open(args.output_model_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved model metadata to %s", args.output_model_dir / "model_metadata.json")


if __name__ == "__main__":
    main()
