"""
End-to-End PyTorch EEGNet Training & Subject-Independent Evaluation Pipeline.
Uses the exact 17/4/3 participant splits from universe_pipeline_metadata.json
for an apples-to-apples benchmark against Random Forest and XGBoost.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from training.models.eegnet import EEGNet
from training.preprocessing.dataset_loader import UniverseDatasetLoader
from training.preprocessing.universe_preprocessor import (
    PreprocessingConfig,
    UniverseEEGPreprocessor,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TrainEEGNet")

CLASSES = ["low", "medium", "high"]
LABEL_TO_IDX = {lbl: idx for idx, lbl in enumerate(CLASSES)}


class EEGWindowDataset(Dataset):
    """Memory-efficient Dataset backed by float16 numpy arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.X[idx].astype(np.float32))
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, y


def build_or_load_tensor_cache(
    raw_dir: Path,
    processed_dir: Path,
    cache_dir: Path,
    force_rebuild: bool = False,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Extracts preprocessed [N, 4, 1024] windows per subject split and caches them
    as float16 .npy files so subsequent training runs load in ~1 second.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    required_files = [
        cache_dir / f"{s}_{k}.npy"
        for s in ("train", "val", "test")
        for k in ("X", "y", "p")
    ]

    if not force_rebuild and all(f.exists() for f in required_files):
        logger.info("Loading cached EEGNet tensors from %s...", cache_dir)
        data = {}
        for s in ("train", "val", "test"):
            X = np.load(cache_dir / f"{s}_X.npy", mmap_mode="r")
            y = np.load(cache_dir / f"{s}_y.npy")
            p = np.load(cache_dir / f"{s}_p.npy", allow_pickle=True)
            data[s] = (X, y, p)
            logger.info("  %-5s -> X: %s | y: %s", s, X.shape, y.shape)
        return data

    meta_path = processed_dir / "universe_pipeline_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found at {meta_path}. Run scripts/preprocess_universe.py first."
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    split_map = {
        "train": set(meta["splits"]["train_subjects"]),
        "val": set(meta["splits"]["val_subjects"]),
        "test": set(meta["splits"]["test_subjects"]),
    }

    loader = UniverseDatasetLoader(raw_data_dir=raw_dir)
    preprocessor = UniverseEEGPreprocessor(
        PreprocessingConfig(
            sampling_rate=256.0,
            channels=["TP9", "AF7", "AF8", "TP10"],
            bandpass_low=0.5,
            bandpass_high=45.0,
            notch_freq=50.0,
            window_seconds=4.0,
            overlap_ratio=0.5,
            apply_car=True,
            normalization="zscore",
            min_quality_score=0.40,
        )
    )

    split_X: Dict[str, List[np.ndarray]] = {"train": [], "val": [], "test": []}
    split_y: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    split_p: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

    participants = loader.discover_participants()
    logger.info("Building [4, 1024] tensor cache for %d participants...", len(participants))

    for sub_id in participants:
        target_split = next((s for s, pids in split_map.items() if sub_id in pids), None)
        if target_split is None:
            continue

        sessions = loader.discover_sessions(sub_id)
        sub_count = 0

        for session_name in sessions:
            session_dir = raw_dir / sub_id / session_name
            labeled_segments = loader.load_labeled_pickle_segments(session_dir)

            if labeled_segments:
                for seg in labeled_segments:
                    if seg.workload_level not in LABEL_TO_IDX:
                        continue
                    label_idx = LABEL_TO_IDX[seg.workload_level]
                    windows = preprocessor.segment_and_preprocess_continuous(
                        continuous_eeg=seg.eeg_data,
                        channel_names=seg.channels,
                    )
                    for win in windows:
                        if not win.is_usable or win.data.shape != (4, 1024):
                            continue
                        split_X[target_split].append(win.data.astype(np.float16))
                        split_y[target_split].append(label_idx)
                        split_p[target_split].append(sub_id)
                        sub_count += 1
            else:
                eeg_file = loader.find_eeg_file(session_dir)
                if not eeg_file:
                    continue
                labels_file = loader.find_task_labels_file(session_dir)
                task_intervals = loader.read_task_labels(labels_file)
                timestamps, eeg_data, channels = loader.read_eeg_csv(eeg_file)
                if eeg_data.shape[1] == 0:
                    continue
                start_t = float(timestamps[0]) if len(timestamps) > 0 else 0.0
                windows = preprocessor.segment_and_preprocess_continuous(
                    continuous_eeg=eeg_data,
                    channel_names=channels,
                    start_timestamp_sec=start_t,
                )
                for win in windows:
                    if not win.is_usable or win.data.shape != (4, 1024):
                        continue
                    task = loader.align_window_with_task(
                        win.start_time_sec, win.end_time_sec, task_intervals
                    )
                    if task is None or task.workload_level not in LABEL_TO_IDX:
                        continue
                    split_X[target_split].append(win.data.astype(np.float16))
                    split_y[target_split].append(LABEL_TO_IDX[task.workload_level])
                    split_p[target_split].append(sub_id)
                    sub_count += 1

        logger.info("  %s (%-5s): %d usable windows", sub_id, target_split, sub_count)

    data = {}
    for s in ("train", "val", "test"):
        X_arr = np.stack(split_X[s], axis=0)
        y_arr = np.array(split_y[s], dtype=np.int64)
        p_arr = np.array(split_p[s], dtype=object)

        np.save(cache_dir / f"{s}_X.npy", X_arr)
        np.save(cache_dir / f"{s}_y.npy", y_arr)
        np.save(cache_dir / f"{s}_p.npy", p_arr)
        data[s] = (X_arr, y_arr, p_arr)
        logger.info("Saved %-5s cache -> X: %s (%.1f MB)", s, X_arr.shape, X_arr.nbytes / 1e6)

    return data


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(y.numpy())
    return np.concatenate(all_targets), np.concatenate(all_preds)


def main():
    parser = argparse.ArgumentParser(description="Train EEGNet on UNIVERSE raw EEG windows.")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw" / "universe")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "universe")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "universe" / "eegnet_cache")
    parser.add_argument("--output-model-dir", type=Path, default=PROJECT_ROOT / "models" / "production")
    parser.add_argument("--experiments-dir", type=Path, default=PROJECT_ROOT / "experiments")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    splits = build_or_load_tensor_cache(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        cache_dir=args.cache_dir,
        force_rebuild=args.rebuild_cache,
    )

    X_train, y_train, _ = splits["train"]
    X_val, y_val, _ = splits["val"]
    X_test, y_test, p_test = splits["test"]

    train_loader = DataLoader(
        EEGWindowDataset(X_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        EEGWindowDataset(X_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        EEGWindowDataset(X_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Inverse-frequency class weights to counter high/medium/low imbalance
    class_counts = np.bincount(y_train, minlength=len(CLASSES)).astype(np.float32)
    class_weights = class_counts.sum() / (len(CLASSES) * np.maximum(class_counts, 1.0))
    logger.info("Train class counts %s: %s | weights: %s", CLASSES, class_counts.astype(int).tolist(), np.round(class_weights, 3).tolist())

    model = EEGNet(n_channels=4, n_samples=1024, n_classes=len(CLASSES), dropout_rate=0.5).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_f1 = -1.0
    best_state = None
    no_improve = 0
    history = []

    logger.info("Starting EEGNet training for up to %d epochs (batch_size=%d)...", args.epochs, args.batch_size)
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        avg_loss = running_loss / max(1, n_batches)

        val_true, val_pred = evaluate_loader(model, val_loader, device)
        val_acc = float(accuracy_score(val_true, val_pred))
        val_f1 = float(f1_score(val_true, val_pred, average="macro", zero_division=0))

        history.append({"epoch": epoch, "train_loss": round(avg_loss, 4), "val_acc": round(val_acc, 4), "val_f1_macro": round(val_f1, 4)})
        logger.info("Epoch %02d/%02d | Loss: %.4f | Val Acc: %.3f | Val Macro F1: %.3f", epoch, args.epochs, avg_loss, val_acc, val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info("Early stopping triggered after %d epochs without Val Macro F1 improvement.", args.patience)
                break

    elapsed = time.time() - t0
    logger.info("Training finished in %.1fs | Best Val Macro F1: %.3f", elapsed, best_val_f1)

    if best_state is not None:
        model.load_state_dict(best_state)

    test_true, test_pred = evaluate_loader(model, test_loader, device)
    test_acc = float(accuracy_score(test_true, test_pred))
    test_f1 = float(f1_score(test_true, test_pred, average="macro", zero_division=0))
    conf_mat = confusion_matrix(test_true, test_pred, labels=[0, 1, 2]).tolist()

    per_participant = {}
    for pid in np.unique(p_test):
        mask = p_test == pid
        per_participant[str(pid)] = {
            "accuracy": float(accuracy_score(test_true[mask], test_pred[mask])),
            "f1_macro": float(f1_score(test_true[mask], test_pred[mask], average="macro", zero_division=0)),
            "samples": int(np.sum(mask)),
        }

    logger.info("Subject-Independent Test Results -> Accuracy: %.3f | Macro F1: %.3f", test_acc, test_f1)
    for pid, stats in per_participant.items():
        logger.info("  %s -> Acc: %.3f | Macro F1: %.3f (%d samples)", pid, stats["accuracy"], stats["f1_macro"], stats["samples"])

    args.experiments_dir.mkdir(parents=True, exist_ok=True)
    eval_report = {
        "model_type": "eegnet",
        "classes": CLASSES,
        "accuracy": test_acc,
        "f1_macro": test_f1,
        "best_val_f1_macro": best_val_f1,
        "confusion_matrix": conf_mat,
        "per_participant": per_participant,
        "training_seconds": round(elapsed, 2),
        "history": history,
    }
    exp_path = args.experiments_dir / "evaluation_eegnet.json"
    with open(exp_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)
    logger.info("Saved evaluation report to %s", exp_path)

    args.output_model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output_model_dir / "universe_eegnet_prod.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {"n_channels": 4, "n_samples": 1024, "n_classes": len(CLASSES)},
            "classes": CLASSES,
            "test_metrics": {"accuracy": test_acc, "f1_macro": test_f1},
        },
        ckpt_path,
    )
    logger.info("Exported production EEGNet checkpoint to %s", ckpt_path)


if __name__ == "__main__":
    main()
