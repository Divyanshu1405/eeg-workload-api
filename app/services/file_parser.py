"""
Parsers for uploaded EEG recording files (.csv and .json) from frontend clients.
Normalizes column names (e.g. RAW_TP9 -> TP9) into [channels x samples] arrays.
"""

import io
import json
from typing import List, Tuple
import numpy as np
import pandas as pd

EXPECTED_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]


def parse_eeg_upload(
    content: bytes,
    filename: str,
    default_sampling_rate: float = 256.0,
) -> Tuple[List[List[float]], List[str], float]:
    """
    Parse an uploaded .csv or .json EEG file into (eeg_data, channel_names, sampling_rate).
    Raises ValueError on unsupported formats or missing channels.
    """
    lower_name = (filename or "").lower()

    if lower_name.endswith(".json"):
        try:
            payload = json.loads(content.decode("utf-8"))
        except Exception as e:
            raise ValueError(f"Invalid JSON file: {e}")

        sr = float(payload.get("sampling_rate", default_sampling_rate))

        # Format A: {"eeg_data": [[...], ...], "channel_names": [...]}
        if "eeg_data" in payload:
            eeg_data = payload["eeg_data"]
            channels = payload.get("channel_names") or EXPECTED_CHANNELS
            return eeg_data, channels, sr

        # Format B: {"TP9": [...], "AF7": [...], "AF8": [...], "TP10": [...]}
        found = [ch for ch in EXPECTED_CHANNELS if ch in payload]
        if found:
            eeg_data = [payload[ch] for ch in found]
            return eeg_data, found, sr

        raise ValueError("JSON must contain 'eeg_data' or channel keys (TP9, AF7, AF8, TP10).")

    if lower_name.endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise ValueError(f"Invalid CSV file: {e}")

        if df.empty:
            raise ValueError("Uploaded CSV file is empty.")

        # Match standard Muse S channel columns (TP9, RAW_TP9, tp9, etc.)
        col_map = {}
        for col in df.columns:
            clean = str(col).strip().upper().replace("RAW_", "").replace("EEG_", "")
            if clean in EXPECTED_CHANNELS and clean not in col_map:
                col_map[clean] = col

        if len(col_map) == len(EXPECTED_CHANNELS):
            ordered_cols = [col_map[ch] for ch in EXPECTED_CHANNELS]
            sub_df = df[ordered_cols].apply(pd.to_numeric, errors="coerce").dropna()
            if sub_df.empty:
                raise ValueError("No valid numeric EEG samples found in CSV.")
            return sub_df.to_numpy(dtype=np.float64).T.tolist(), EXPECTED_CHANNELS, default_sampling_rate

        # Fallback: take first 4 numeric columns (excluding timestamp-like columns)
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if "time" not in str(c).lower() and "index" not in str(c).lower()
        ]
        if len(numeric_cols) >= 4:
            sub_df = df[numeric_cols[:4]].dropna()
            return sub_df.to_numpy(dtype=np.float64).T.tolist(), EXPECTED_CHANNELS, default_sampling_rate

        raise ValueError(
            f"Could not locate 4 EEG channels (TP9, AF7, AF8, TP10) in CSV columns: {list(df.columns)}"
        )

    raise ValueError("Unsupported file type. Please upload a .csv or .json EEG recording.")
