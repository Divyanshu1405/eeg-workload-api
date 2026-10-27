"""
Dataset Loader and Ground Truth Synchronizer for the UNIVERSE Dataset.
UNIVERSE: UNobtrusIVE measuRement of mental workload and stress in uncontrolled environments.

This module handles:
1. Scanning participant folders (UN_101 to UN_124) across Lab1, Lab2, and Wild conditions
2. Reading Muse S EEG CSV records with flexible column name resolution
3. Reading and aligning task labels (N-Back, Stroop, Arithmetic, Sudoku, Office tasks)
4. Synchronizing sliding EEG epochs with task timestamps and mental workload ratings (Likert 1-5)
5. Generating subject-independent train / val / test partitions (strictly preventing data leakage)
6. Synthetic mock dataset generation for reproducible unit and integration testing
"""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class TaskInterval:
    task_name: str
    condition: str  # 'easy', 'medium', 'hard', 'baseline', 'wild'
    workload_level: str  # 'low', 'medium', 'high'
    workload_score: float  # e.g. 1.0 - 5.0
    start_time: float
    end_time: float


@dataclass
class UniverseEpochRecord:
    participant_id: str
    session: str  # 'Lab1', 'Lab2', 'Wild'
    window_index: int
    start_time: float
    end_time: float
    features: Dict[str, float]
    workload_level: str  # 'low', 'medium', 'high'
    workload_score: float
    task_name: str
    is_usable: bool
    quality_score: float


@dataclass
class DatasetSplit:
    train_subjects: List[str]
    val_subjects: List[str]
    test_subjects: List[str]


class UniverseDatasetLoader:
    """
    Manages loading, parsing, synchronization, and export of UNIVERSE dataset files.
    """

    STANDARD_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]

    def __init__(self, raw_data_dir: Union[str, Path]):
        self.raw_data_dir = Path(raw_data_dir)

    def discover_participants(self) -> List[str]:
        """Discovers all participant directories (e.g. UN_101, UN_102, ...)."""
        if not self.raw_data_dir.exists():
            return []

        participants = []
        for entry in sorted(self.raw_data_dir.iterdir()):
            if entry.is_dir() and entry.name.startswith("UN_"):
                participants.append(entry.name)
        return participants

    def discover_sessions(self, participant_id: str) -> List[str]:
        """Discovers available sessions (Lab1, Lab2, Wild) for a participant."""
        p_dir = self.raw_data_dir / participant_id
        if not p_dir.exists():
            return []

        sessions = []
        for s in ["Lab1", "Lab2", "Wild"]:
            if (p_dir / s).exists():
                sessions.append(s)
        return sessions

    def find_eeg_file(self, session_dir: Path) -> Optional[Path]:
        """
        Locates the Muse S EEG CSV file within a session directory.
        Checks Raw/Muse/, Raw/, or the session root.
        """
        candidates = [
            session_dir / "Raw" / "Muse" / "eeg.csv",
            session_dir / "Raw" / "Muse" / "raw_muse.csv",
            session_dir / "Raw" / "eeg.csv",
            session_dir / "eeg.csv",
        ]
        for c in candidates:
            if c.exists():
                return c

        # Fallback: search for any .csv in Raw/Muse
        muse_dir = session_dir / "Raw" / "Muse"
        if muse_dir.exists():
            for f in muse_dir.glob("*.csv"):
                if "acc" not in f.name.lower() and "gyro" not in f.name.lower():
                    return f

        # Fallback: search for any csv with 'eeg' or 'muse' in name
        for f in session_dir.rglob("*.csv"):
            if "eeg" in f.name.lower() or "muse" in f.name.lower():
                return f

        return None

    def find_task_labels_file(self, session_dir: Path) -> Optional[Path]:
        """
        Locates task label or event marker files.
        """
        candidates = [
            session_dir / "Task_Labels.csv",
            session_dir / "task_labels.csv",
            session_dir / "Labeled" / "Task_Labels.csv",
            session_dir / "tasks.csv",
        ]
        for c in candidates:
            if c.exists():
                return c

        for f in session_dir.glob("*task*.csv"):
            return f

        return None

    def read_eeg_csv(self, eeg_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Parses EEG CSV data.
        Returns:
            timestamps: 1D array of time in seconds
            eeg_data: 2D array [num_channels, num_samples]
            channels: list of channel names in order
        """
        with open(eeg_path, "r", encoding="utf-8", errors="ignore") as f:
            header_line = f.readline().strip()

        # Parse delimiter (comma, semicolon, or tab)
        delimiter = "," if "," in header_line else (";" if ";" in header_line else "\t")
        columns = [c.strip().strip('"').strip("'") for c in header_line.split(delimiter)]
        col_lower = [c.lower() for c in columns]

        # Resolve timestamp column
        time_idx = None
        for cand in ["timestamps", "timestamp", "time", "t", "index"]:
            if cand in col_lower:
                time_idx = col_lower.index(cand)
                break

        # Resolve channel columns (TP9, AF7, AF8, TP10)
        channel_indices = {}
        for target_ch in self.STANDARD_CHANNELS:
            for idx, col in enumerate(col_lower):
                if target_ch.lower() in col or col.endswith(target_ch.lower()):
                    channel_indices[target_ch] = idx
                    break

        # Load numerical rows
        loaded_data = []
        with open(eeg_path, "r", encoding="utf-8", errors="ignore") as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split(delimiter)
                if len(parts) >= len(columns):
                    try:
                        row = [float(p.strip()) for p in parts]
                        loaded_data.append(row)
                    except ValueError:
                        continue

        if not loaded_data:
            return np.array([]), np.zeros((len(self.STANDARD_CHANNELS), 0)), self.STANDARD_CHANNELS

        arr = np.array(loaded_data, dtype=np.float64)

        if time_idx is not None:
            raw_time = arr[:, time_idx]
            # If timestamp is in milliseconds or nanoseconds, convert to seconds
            if np.mean(raw_time) > 1e11:  # nanoseconds
                raw_time = raw_time / 1e9
            elif np.mean(raw_time) > 1e8:  # milliseconds
                raw_time = raw_time / 1e3
            timestamps = raw_time - raw_time[0]
        else:
            timestamps = np.arange(len(arr)) / 256.0

        # Extract channels in standard order
        extracted_channels = []
        ch_signals = []
        for ch in self.STANDARD_CHANNELS:
            if ch in channel_indices:
                ch_signals.append(arr[:, channel_indices[ch]])
                extracted_channels.append(ch)
            else:
                # If channel missing, create placeholder
                ch_signals.append(np.zeros(len(arr)))
                extracted_channels.append(ch)

        eeg_matrix = np.vstack(ch_signals)
        return timestamps, eeg_matrix, extracted_channels

    def read_task_labels(self, label_path: Optional[Path]) -> List[TaskInterval]:
        """
        Parses task ground truth file containing task intervals, difficulties, and ratings.
        """
        if label_path is None or not label_path.exists():
            return []

        intervals: List[TaskInterval] = []
        with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
            header_line = f.readline().strip()
            delimiter = "," if "," in header_line else (";" if ";" in header_line else "\t")
            cols = [c.strip().lower() for c in header_line.split(delimiter)]

            # Map columns
            name_idx = cols.index("task_name") if "task_name" in cols else (cols.index("task") if "task" in cols else 0)
            start_idx = cols.index("start_time") if "start_time" in cols else (cols.index("start") if "start" in cols else 1)
            end_idx = cols.index("end_time") if "end_time" in cols else (cols.index("end") if "end" in cols else 2)
            diff_idx = cols.index("difficulty") if "difficulty" in cols else (cols.index("condition") if "condition" in cols else None)
            rating_idx = cols.index("workload_rating") if "workload_rating" in cols else (cols.index("rating") if "rating" in cols else None)

            for line in f:
                parts = [p.strip().strip('"') for p in line.strip().split(delimiter)]
                if len(parts) > max(start_idx, end_idx):
                    try:
                        task_name = parts[name_idx]
                        t_start = float(parts[start_idx])
                        t_end = float(parts[end_idx])
                        diff = parts[diff_idx] if diff_idx is not None and diff_idx < len(parts) else "medium"
                        rating = float(parts[rating_idx]) if rating_idx is not None and rating_idx < len(parts) else 3.0

                        # Map to level
                        if rating <= 2.0 or diff.lower() in ["easy", "low"]:
                            wl_level = "low"
                        elif rating >= 4.0 or diff.lower() in ["hard", "high"]:
                            wl_level = "high"
                        else:
                            wl_level = "medium"

                        intervals.append(
                            TaskInterval(
                                task_name=task_name,
                                condition=diff,
                                workload_level=wl_level,
                                workload_score=rating,
                                start_time=t_start,
                                end_time=t_end,
                            )
                        )
                    except (ValueError, IndexError):
                        continue

        return intervals

    def align_window_with_task(
        self,
        win_start: float,
        win_end: float,
        tasks: List[TaskInterval],
    ) -> Optional[TaskInterval]:
        """
        Finds the matching task interval for a sliding EEG window based on temporal overlap.
        """
        win_mid = 0.5 * (win_start + win_end)
        for task in tasks:
            if task.start_time <= win_mid <= task.end_time:
                return task
        return None

    def create_subject_independent_split(
        self,
        participants: Optional[List[str]] = None,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
    ) -> DatasetSplit:
        """
        Creates a strict subject-independent train / val / test split.
        Ensures zero data leakage across participants.
        """
        subjects = list(participants or self.discover_participants())
        if not subjects:
            # Generate default 24 participant IDs for UNIVERSE
            subjects = [f"UN_{i:03d}" for i in range(101, 125)]

        rng = np.random.default_rng(random_seed)
        shuffled = subjects.copy()
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))

        train_subs = sorted(shuffled[:n_train])
        val_subs = sorted(shuffled[n_train : n_train + n_val])
        test_subs = sorted(shuffled[n_train + n_val :])

        return DatasetSplit(
            train_subjects=train_subs,
            val_subjects=val_subs,
            test_subjects=test_subs,
        )

    def generate_synthetic_universe_session(
        self,
        output_dir: Optional[Path] = None,
        participant_id: str = "UN_101",
        session: str = "Lab1",
        duration_seconds: float = 120.0,
        sampling_rate: float = 256.0,
    ) -> Tuple[Path, Path]:
        """
        Generates realistic synthetic Muse S EEG data and task labels for testing the pipeline.
        Includes simulated brain rhythms (alpha desynchronization and theta elevation during high workload)
        and realistic noise artifacts.
        """
        target_dir = Path(output_dir or self.raw_data_dir) / participant_id / session
        muse_dir = target_dir / "Raw" / "Muse"
        muse_dir.mkdir(parents=True, exist_ok=True)

        n_samples = int(duration_seconds * sampling_rate)
        t = np.arange(n_samples) / sampling_rate

        # Simulate 3 task phases:
        # Phase 1 (0-40s): Resting / Low Workload (strong 10 Hz alpha, low theta)
        # Phase 2 (40-80s): Mental Arithmetic / High Workload (increased 6 Hz theta, reduced alpha)
        # Phase 3 (80-120s): Stroop / Medium Workload (moderate theta and alpha)
        tasks = [
            ("Resting", "easy", "low", 1.5, 0.0, 40.0),
            ("Mental_Arithmetic", "hard", "high", 4.5, 40.0, 80.0),
            ("Stroop", "medium", "medium", 3.0, 80.0, 120.0),
        ]

        # Generate channel signals (TP9, AF7, AF8, TP10)
        eeg_channels = {}
        for ch in self.STANDARD_CHANNELS:
            sig = np.random.normal(0, 5.0, n_samples)  # 5 uV background noise
            # 50 Hz powerline hum
            sig += 8.0 * np.sin(2 * np.pi * 50.0 * t)

            for task_name, diff, wl, score, t_s, t_e in tasks:
                mask = (t >= t_s) & (t < t_e)
                if wl == "low":
                    # Strong alpha on all channels, especially temporals and frontals
                    sig[mask] += 25.0 * np.sin(2 * np.pi * 10.0 * t[mask])
                    sig[mask] += 4.0 * np.sin(2 * np.pi * 6.0 * t[mask])
                elif wl == "high":
                    # Elevated frontal theta on AF7, AF8; suppressed alpha
                    if "AF" in ch:
                        sig[mask] += 28.0 * np.sin(2 * np.pi * 6.0 * t[mask])
                    sig[mask] += 6.0 * np.sin(2 * np.pi * 10.0 * t[mask])
                    sig[mask] += 12.0 * np.sin(2 * np.pi * 20.0 * t[mask])  # beta
                else:
                    sig[mask] += 14.0 * np.sin(2 * np.pi * 10.0 * t[mask])
                    sig[mask] += 14.0 * np.sin(2 * np.pi * 6.0 * t[mask])

            eeg_channels[ch] = sig

        # Add brief transient blink artifact at t=15s on AF7, AF8
        blink_mask = (t >= 15.0) & (t <= 15.3)
        eeg_channels["AF7"][blink_mask] += 65.0
        eeg_channels["AF8"][blink_mask] += 60.0

        # Write eeg.csv
        eeg_csv_path = muse_dir / "eeg.csv"
        with open(eeg_csv_path, "w", encoding="utf-8") as f:
            f.write("timestamps,TP9,AF7,AF8,TP10\n")
            for i in range(n_samples):
                f.write(
                    f"{t[i]:.4f},{eeg_channels['TP9'][i]:.2f},{eeg_channels['AF7'][i]:.2f},"
                    f"{eeg_channels['AF8'][i]:.2f},{eeg_channels['TP10'][i]:.2f}\n"
                )

        # Write Task_Labels.csv
        labels_csv_path = target_dir / "Task_Labels.csv"
        with open(labels_csv_path, "w", encoding="utf-8") as f:
            f.write("task_name,start_time,end_time,difficulty,workload_rating\n")
            for t_name, diff, wl, score, t_s, t_e in tasks:
                f.write(f"{t_name},{t_s:.1f},{t_e:.1f},{diff},{score:.1f}\n")

        return eeg_csv_path, labels_csv_path
