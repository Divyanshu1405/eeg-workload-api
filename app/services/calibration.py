"""
Per-user resting EEG baseline calibration store.
Computes and stores per-feature mean/std statistics for optional personalization.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


class CalibrationService:
    def __init__(self, store_path: Optional[Path] = None):
        self._baselines: Dict[str, Dict[str, Any]] = {}
        self.store_path = store_path
        if self.store_path and self.store_path.exists():
            self._load()

    def _load(self) -> None:
        try:
            with open(self.store_path, "r", encoding="utf-8") as f:
                self._baselines = json.load(f)
        except Exception:
            self._baselines = {}

    def _save(self) -> None:
        if not self.store_path:
            return
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(self._baselines, f, indent=2)

    def save_baseline(
        self,
        user_id: str,
        feature_dicts: List[Dict[str, float]],
        rejected_windows: int = 0,
        avg_quality_score: float = 1.0,
    ) -> Dict[str, Any]:
        """Compute and persist per-feature mean/std from usable resting windows."""
        keys = sorted(feature_dicts[0].keys())
        means: Dict[str, float] = {}
        stds: Dict[str, float] = {}

        for k in keys:
            vals = np.array([float(d.get(k, 0.0)) for d in feature_dicts], dtype=np.float64)
            means[k] = float(np.mean(vals))
            std_val = float(np.std(vals))
            stds[k] = std_val if std_val > 1e-6 else 1.0

        record = {
            "user_id": user_id,
            "usable_windows": len(feature_dicts),
            "rejected_windows": rejected_windows,
            "avg_quality_score": round(float(avg_quality_score), 3),
            "baseline_tar": round(means.get("frontal_theta_alpha_ratio", 0.0), 3),
            "baseline_engagement": round(means.get("frontal_engagement_index", 0.0), 3),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "means": means,
            "stds": stds,
        }
        self._baselines[user_id] = record
        self._save()
        return record

    def get_baseline(self, user_id: str) -> Optional[Dict[str, Any]]:
        return self._baselines.get(user_id)

    def delete_baseline(self, user_id: str) -> bool:
        existed = user_id in self._baselines
        if existed:
            del self._baselines[user_id]
            self._save()
        return existed

    def normalize_features(self, user_id: str, features: Dict[str, float]) -> Dict[str, float]:
        """Z-score a feature dict against the user's stored resting baseline."""
        baseline = self._baselines.get(user_id)
        if not baseline:
            return features
        means = baseline["means"]
        stds = baseline["stds"]
        return {
            k: float((v - means.get(k, 0.0)) / stds.get(k, 1.0))
            for k, v in features.items()
        }


calibration_service = CalibrationService()
