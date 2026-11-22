"""
Stream real 4-second raw EEG windows from unseen UNIVERSE test participants
(UN_102, UN_109, UN_114) to the live FastAPI /predict endpoint and inspect
predictions, signal quality, and end-to-end HTTP latency.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple
import urllib.error
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from training.preprocessing.dataset_loader import UniverseDatasetLoader

VALID_CLASSES = {"low", "medium", "high"}
WINDOW_SAMPLES = 1024  # 4.0s @ 256 Hz


def collect_balanced_raw_windows(
    raw_dir: Path,
    subject_id: str,
    max_windows: int = 18,
) -> List[Dict]:
    """
    Collects raw [4, 1024] EEG windows from a participant's labeled segments,
    sampling evenly across available workload levels (low / medium / high).
    """
    loader = UniverseDatasetLoader(raw_data_dir=raw_dir)
    sessions = loader.discover_sessions(subject_id)

    by_label: Dict[str, List[Tuple[str, np.ndarray, List[str]]]] = defaultdict(list)

    for session_name in sessions:
        session_dir = raw_dir / subject_id / session_name
        segments = loader.load_labeled_pickle_segments(session_dir)
        for seg in segments:
            if seg.workload_level not in VALID_CLASSES:
                continue
            if seg.eeg_data.ndim != 2 or seg.eeg_data.shape[0] != 4:
                continue
            n_samples = seg.eeg_data.shape[1]
            if n_samples < WINDOW_SAMPLES:
                continue

            # Sample up to 6 non-overlapping windows per segment
            n_possible = n_samples // WINDOW_SAMPLES
            indices = np.linspace(0, n_possible - 1, min(n_possible, 6), dtype=int)
            for idx in indices:
                start = int(idx) * WINDOW_SAMPLES
                win = seg.eeg_data[:, start : start + WINDOW_SAMPLES]
                if not np.all(np.isfinite(win)):
                    continue
                by_label[seg.workload_level].append((seg.task_name, win, seg.channels))

    selected: List[Dict] = []
    present_labels = [lbl for lbl in ("low", "medium", "high") if by_label[lbl]]
    if not present_labels:
        return selected

    per_label = max(1, max_windows // len(present_labels))
    for lbl in present_labels:
        pool = by_label[lbl]
        pick_idx = np.linspace(0, len(pool) - 1, min(len(pool), per_label), dtype=int)
        for i in pick_idx:
            task_name, win, channels = pool[int(i)]
            selected.append(
                {
                    "participant_id": subject_id,
                    "task_name": task_name,
                    "true_label": lbl,
                    "channels": channels,
                    "eeg_data": win,
                }
            )
    return selected


def check_health(base_url: str) -> Dict:
    with urllib.request.urlopen(f"{base_url}/health", timeout=5) as r:
        return json.loads(r.read().decode("utf-8"))


def post_predict(base_url: str, sample: Dict, seq_num: int) -> Tuple[Dict, float]:
    payload = json.dumps(
        {
            "session_id": f"{sample['participant_id']}_win_{seq_num:03d}",
            "sampling_rate": 256.0,
            "channel_names": sample["channels"],
            "eeg_data": sample["eeg_data"].tolist(),
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=10) as r:
        body = json.loads(r.read().decode("utf-8"))
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return body, latency_ms


def main():
    parser = argparse.ArgumentParser(description="Stream real UNIVERSE test windows to live /predict.")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw" / "universe")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--subjects", nargs="+", default=["UN_102", "UN_109", "UN_114"])
    parser.add_argument("--windows-per-subject", type=int, default=12)
    args = parser.parse_args()

    try:
        health = check_health(args.base_url)
    except urllib.error.URLError as e:
        print(f"[ERROR] Could not reach {args.base_url}/health ({e}). Start uvicorn first.")
        sys.exit(1)

    model_info = health.get("model_info", {})
    print(
        f"Connected to {args.base_url} | Model: {model_info.get('model_file')} "
        f"| Connected: {health.get('model_connected')} | Features: {model_info.get('feature_dim')}\n"
    )

    header = f"{'Subject':<8} {'Task':<18} {'True':<7} {'Pred':<9} {'Conf':>5} {'Qual':>5} {'TAR':>6} {'Latency':>9} {'Match'}"
    print(header)
    print("-" * len(header))

    all_latencies: List[float] = []
    sub_stats: Dict[str, Dict[str, int]] = {}

    for sub_id in args.subjects:
        samples = collect_balanced_raw_windows(args.raw_dir, sub_id, max_windows=args.windows_per_subject)
        usable = 0
        rejected = 0
        correct = 0

        for idx, sample in enumerate(samples, start=1):
            resp, ms = post_predict(args.base_url, sample, idx)
            all_latencies.append(ms)

            pred_obj = resp["predictions"][0] if resp.get("predictions") else {}
            pred_lbl = pred_obj.get("workload", "error")
            conf = pred_obj.get("confidence") or 0.0
            qual = pred_obj.get("quality_score") or 0.0
            tar = pred_obj.get("theta_alpha_ratio")
            tar_str = f"{tar:6.2f}" if tar is not None else "   n/a"

            if pred_lbl == "unusable":
                rejected += 1
                match_str = "SQI_REJ"
            else:
                usable += 1
                is_match = pred_lbl == sample["true_label"]
                if is_match:
                    correct += 1
                match_str = "YES" if is_match else "no"

            print(
                f"{sub_id:<8} {sample['task_name'][:18]:<18} {sample['true_label']:<7} "
                f"{pred_lbl:<9} {conf:5.2f} {qual:5.2f} {tar_str} {ms:7.1f}ms  {match_str}"
            )

        sub_stats[sub_id] = {"usable": usable, "rejected": rejected, "correct": correct}

    print("\n=== Per-Participant Summary (Live API) ===")
    total_usable = 0
    total_correct = 0
    for sub_id, st in sub_stats.items():
        acc = (st["correct"] / st["usable"]) if st["usable"] > 0 else 0.0
        total_usable += st["usable"]
        total_correct += st["correct"]
        print(
            f"  {sub_id}: Usable={st['usable']:2d} | Rejected={st['rejected']:2d} "
            f"| Accuracy={acc:.1%} ({st['correct']}/{st['usable']})"
        )

    if all_latencies:
        lat = np.array(all_latencies)
        overall_acc = (total_correct / total_usable) if total_usable > 0 else 0.0
        print("\n=== Latency & Overall Summary ===")
        print(f"  Requests sent : {len(lat)} (Usable: {total_usable}, Overall Acc: {overall_acc:.1%})")
        print(f"  Mean latency  : {lat.mean():.1f} ms")
        print(f"  p50 latency   : {np.percentile(lat, 50):.1f} ms")
        print(f"  p95 latency   : {np.percentile(lat, 95):.1f} ms")
        print(f"  Max latency   : {lat.max():.1f} ms")


if __name__ == "__main__":
    main()
