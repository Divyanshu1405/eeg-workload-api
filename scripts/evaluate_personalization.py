"""
Personalization / Calibration Evaluation Prototype for UNIVERSE EEG.
Tests the central research question: How much does a small amount of user-specific
calibration data improve workload estimation for an unseen participant?
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
from sklearn.metrics import accuracy_score, f1_score

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("PersonalizationEval")


def run_personalization_study(processed_dir: Path, calibration_ratio: float = 0.20):
    """
    Simulates user-specific calibration:
    1. Trains base model on standard training split (0-shot).
    2. For each unseen test participant, takes the first K% of their session as calibration data.
    3. Fine-tunes/adapts the model (or sample-weights) and tests on the remaining (1 - K)% of their session.
    4. Compares 0-shot accuracy vs. calibrated accuracy.
    """
    train_path = processed_dir / "universe_features_train.csv"
    test_path = processed_dir / "universe_features_test.csv"

    if not train_path.exists() or not test_path.exists():
        logger.error("Feature splits not found. Run training script first.")
        return

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    meta_cols = {"participant_id", "session", "window_index", "start_time", "end_time", "task_name", "workload_level", "workload_score", "quality_score"}
    feature_cols = [c for c in df_train.columns if c not in meta_cols]

    X_train = df_train[feature_cols].values
    y_train = df_train["workload_level"].values

    # Train base 0-shot model
    logger.info("Training base 0-shot Random Forest model...")
    base_model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    base_model.fit(X_train, y_train)

    results = []
    test_participants = df_test["participant_id"].unique()

    for pid in test_participants:
        p_df = df_test[df_test["participant_id"] == pid].sort_values("start_time")
        if len(p_df) < 5:
            continue

        X_p = p_df[feature_cols].values
        y_p = p_df["workload_level"].values

        # 0-shot evaluation on this participant
        y_pred_0shot = base_model.predict(X_p)
        acc_0shot = float(accuracy_score(y_p, y_pred_0shot))
        f1_0shot = float(f1_score(y_p, y_pred_0shot, average="macro", zero_division=0))

        # Split participant data into calibration set vs evaluation set
        n_calib = max(1, int(len(X_p) * calibration_ratio))
        X_calib, y_calib = X_p[:n_calib], y_p[:n_calib]
        X_eval, y_eval = X_p[n_calib:], y_p[n_calib:]

        if len(X_eval) == 0:
            continue

        # Calibrated model: Retrain/fine-tune base model with combined base training data + participant calibration sample
        logger.info("Calibrating model for participant %s using %d calibration samples...", pid, n_calib)
        calib_model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
        # Give higher sample weight or combine datasets
        X_combined = np.vstack([X_train, X_calib])
        y_combined = np.concatenate([y_train, y_calib])
        calib_model.fit(X_combined, y_combined)

        y_pred_calib = calib_model.predict(X_eval)
        acc_calib = float(accuracy_score(y_eval, y_pred_calib))
        f1_calib = float(f1_score(y_eval, y_pred_calib, average="macro", zero_division=0))

        delta_acc = acc_calib - acc_0shot
        delta_f1 = f1_calib - f1_0shot

        logger.info("Participant %s -> 0-Shot Acc: %.3f | Calibrated Acc: %.3f (Delta: %+ .3f)", pid, acc_0shot, acc_calib, delta_acc)

        results.append({
            "participant_id": str(pid),
            "samples_total": int(len(p_df)),
            "samples_calibration": n_calib,
            "samples_evaluation": int(len(X_eval)),
            "accuracy_0shot": acc_0shot,
            "accuracy_calibrated": acc_calib,
            "delta_accuracy": delta_acc,
            "f1_0shot": f1_0shot,
            "f1_calibrated": f1_calib,
            "delta_f1": delta_f1,
        })

    summary = {
        "study_description": "User-specific calibration vs. 0-shot generalization across unseen participants",
        "calibration_ratio": calibration_ratio,
        "participant_results": results,
    }

    out_path = PROJECT_ROOT / "experiments" / "personalization_study_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved personalization study results to %s", out_path)


def main():
    parser = argparse.ArgumentParser(description="Run personalization study.")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "universe")
    parser.add_argument("--calib-ratio", type=float, default=0.20)
    args = parser.parse_args()
    run_personalization_study(args.processed_dir, args.calib_ratio)


if __name__ == "__main__":
    main()
