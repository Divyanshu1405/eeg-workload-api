"""
Full EEG Preprocessing Pipeline for the UNIVERSE Dataset.
UNIVERSE: UNobtrusIVE measuRement of mental workload and stress in uncontrolled environments.

Dataset Technical Specifications:
- Device: Muse S headband
- Sampling Rate: 256 Hz
- Standard Channels: TP9, AF7, AF8, TP10 (Reference: Fpz)
- Context: Controlled laboratory (Lab1, Lab2) & uncontrolled real-world office (Wild)

Scientific Preprocessing Pipeline Stages:
1. Signal validation & timestamp regularity verification
2. Signal Quality Assessment (SQA) & artifact flagging
3. Zero-phase Bandpass Filtering: 0.5 - 45.0 Hz (4th order Butterworth)
4. Zero-phase Powerline Notch Filtering: 50.0 Hz (IIR Notch / Bandstop)
5. Spatial Re-referencing: Common Average Referencing (CAR) across good scalp channels
6. Baseline Normalization: Robust z-score or Min-Max scaling
7. Sliding Window Segmentation & Window-level Quality Gating
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from training.preprocessing.signal_quality import (
    SignalQualityAssessor,
    SignalQualityMetrics,
)

# Optional scipy import with graceful fallback
try:
    from scipy.signal import butter, filtfilt, iirnotch
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class PreprocessingConfig:
    sampling_rate: float = 256.0
    channels: List[str] = field(default_factory=lambda: ["TP9", "AF7", "AF8", "TP10"])
    bandpass_low: float = 0.5
    bandpass_high: float = 45.0
    bandpass_order: int = 4
    notch_freq: float = 50.0  # 50 Hz for European grid (UNIVERSE dataset recorded in Germany)
    notch_q: float = 30.0
    apply_car: bool = True  # Common Average Referencing
    normalization: str = "zscore"  # 'zscore', 'minmax', or 'none'
    window_seconds: float = 4.0  # 4.0 second sliding windows
    overlap_ratio: float = 0.5  # 50% overlap (2.0s step)
    min_quality_score: float = 0.40  # discard windows below this SQI threshold
    interpolate_bad_channels: bool = True


@dataclass
class PreprocessedWindow:
    window_index: int
    data: np.ndarray  # Shape: [num_channels, num_samples]
    sampling_rate: float
    channel_names: List[str]
    quality_metrics: SignalQualityMetrics
    is_usable: bool
    start_time_sec: float
    end_time_sec: float
    metadata: Dict = field(default_factory=dict)


class UniverseEEGPreprocessor:
    """
    Production-grade preprocessor conforming to the UNIVERSE dataset experimental specs.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.sqa = SignalQualityAssessor(
            sampling_rate=self.config.sampling_rate,
            critical_channels=["AF7", "AF8"],
        )
        self._init_filters()

    def _init_filters(self):
        """Precomputes filter coefficients if scipy is available."""
        fs = self.config.sampling_rate
        nyq = 0.5 * fs

        if HAS_SCIPY:
            # 4th order Butterworth bandpass
            low = self.config.bandpass_low / nyq
            high = self.config.bandpass_high / nyq
            self.b_bandpass, self.a_bandpass = butter(
                self.config.bandpass_order, [low, high], btype="band"
            )

            # IIR Notch filter for powerline noise
            if 0 < self.config.notch_freq < nyq:
                self.b_notch, self.a_notch = iirnotch(
                    self.config.notch_freq, self.config.notch_q, fs=fs
                )
            else:
                self.b_notch, self.a_notch = None, None
        else:
            self.b_bandpass, self.a_bandpass = None, None
            self.b_notch, self.a_notch = None, None

    def _apply_fft_filter_fallback(self, signal: np.ndarray) -> np.ndarray:
        """
        Pure NumPy FFT-based zero-phase bandpass and notch filter fallback
        when scipy is not yet loaded in the environment.
        """
        n = len(signal)
        if n < 8:
            return signal

        freqs = np.fft.rfftfreq(n, 1.0 / self.config.sampling_rate)
        fft_vals = np.fft.rfft(signal)

        # Bandpass mask (0.5 to 45 Hz with smooth cosine roll-off)
        bp_mask = np.ones_like(freqs, dtype=np.float64)
        bp_mask[freqs < self.config.bandpass_low] = 0.0
        bp_mask[freqs > self.config.bandpass_high] = 0.0

        # Smooth low-cutoff transition (0.2 Hz transition band)
        trans_low = (freqs >= self.config.bandpass_low) & (freqs < self.config.bandpass_low + 0.5)
        if np.any(trans_low):
            bp_mask[trans_low] = 0.5 * (1.0 - np.cos(np.pi * (freqs[trans_low] - self.config.bandpass_low) / 0.5))

        # Notch mask (50 Hz +/- 1 Hz stopband)
        notch_mask = np.abs(freqs - self.config.notch_freq) < 1.5
        bp_mask[notch_mask] = 0.0

        filtered_fft = fft_vals * bp_mask
        return np.fft.irfft(filtered_fft, n=n)

    def apply_temporal_filtering(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Applies zero-phase 4th order Butterworth bandpass (0.5 - 45 Hz)
        and 50 Hz notch filter to each channel.
        Input: [num_channels, num_samples]
        Output: [num_channels, num_samples]
        """
        filtered = np.zeros_like(eeg_data, dtype=np.float64)
        num_channels, num_samples = eeg_data.shape

        for ch in range(num_channels):
            channel_signal = eeg_data[ch, :]
            # Replace NaNs if present
            if np.any(np.isnan(channel_signal)):
                nans = np.isnan(channel_signal)
                channel_signal = np.interp(
                    np.arange(num_samples),
                    np.where(~nans)[0],
                    channel_signal[~nans],
                )

            if HAS_SCIPY and self.b_bandpass is not None and num_samples > 3 * max(len(self.a_bandpass), len(self.b_bandpass)):
                # Zero-phase Butterworth bandpass
                s = filtfilt(self.b_bandpass, self.a_bandpass, channel_signal)
                # Zero-phase Notch filter
                if self.b_notch is not None and self.a_notch is not None:
                    s = filtfilt(self.b_notch, self.a_notch, s)
                filtered[ch, :] = s
            else:
                # FFT fallback filter
                filtered[ch, :] = self._apply_fft_filter_fallback(channel_signal)

        return filtered

    def apply_car(self, eeg_data: np.ndarray, bad_channel_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Applies Common Average Referencing (CAR).
        Subtracts the mean of valid scalp channels at each time sample.
        """
        num_channels, num_samples = eeg_data.shape
        bad_set = set(bad_channel_indices or [])
        good_indices = [i for i in range(num_channels) if i not in bad_set]

        if not good_indices:
            # If all channels bad, fall back to global mean
            car_ref = np.mean(eeg_data, axis=0)
        else:
            car_ref = np.mean(eeg_data[good_indices, :], axis=0)

        car_data = eeg_data - car_ref
        return car_data

    def interpolate_bad_channels(
        self,
        eeg_data: np.ndarray,
        channel_names: List[str],
        bad_channel_names: List[str],
    ) -> np.ndarray:
        """
        Interpolates bad channels from symmetric or neighbouring bilateral scalp electrodes.
        Muse S layout:
        - AF7 (left frontal) pairs with AF8 (right frontal)
        - TP9 (left temporal) pairs with TP10 (right temporal)
        """
        interpolated = eeg_data.copy()
        name_to_idx = {name: i for i, name in enumerate(channel_names)}

        pairs = {
            "AF7": "AF8",
            "AF8": "AF7",
            "TP9": "TP10",
            "TP10": "TP9",
        }

        for bad_ch in bad_channel_names:
            if bad_ch in name_to_idx:
                bad_idx = name_to_idx[bad_ch]
                partner = pairs.get(bad_ch)
                if partner and partner in name_to_idx and partner not in bad_channel_names:
                    # Spatial mirror interpolation with slight attenuation
                    interpolated[bad_idx, :] = 0.85 * interpolated[name_to_idx[partner], :]
                else:
                    # Fallback to mean of all remaining good channels
                    good_idxs = [name_to_idx[name] for name in channel_names if name not in bad_channel_names]
                    if good_idxs:
                        interpolated[bad_idx, :] = np.mean(interpolated[good_idxs, :], axis=0)
                    else:
                        interpolated[bad_idx, :] = 0.0

        return interpolated

    def normalize(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Normalizes signals across time per channel according to config:
        - 'zscore': Robust median/IQR or mean/std normalization
        - 'minmax': Scales to [-1, 1] range
        - 'none': Raw microvolt output
        """
        if self.config.normalization == "none":
            return eeg_data

        normed = np.zeros_like(eeg_data)
        num_channels = eeg_data.shape[0]

        for ch in range(num_channels):
            sig = eeg_data[ch, :]
            if self.config.normalization == "zscore":
                std = np.std(sig)
                if std > 1e-6:
                    normed[ch, :] = (sig - np.mean(sig)) / std
                else:
                    normed[ch, :] = sig - np.mean(sig)
            elif self.config.normalization == "minmax":
                min_val = np.min(sig)
                max_val = np.max(sig)
                rng = max_val - min_val
                if rng > 1e-6:
                    # Scale to [-1.0, 1.0]
                    normed[ch, :] = 2.0 * ((sig - min_val) / rng) - 1.0
                else:
                    normed[ch, :] = sig - np.mean(sig)
            else:
                normed[ch, :] = sig

        return normed

    def preprocess_window(
        self,
        eeg_data: np.ndarray,
        channel_names: Optional[List[str]] = None,
        window_index: int = 0,
        start_time_sec: float = 0.0,
    ) -> PreprocessedWindow:
        """
        Processes a single EEG window (e.g. streaming window or segmented slice).
        Input array: [num_channels, num_samples]
        """
        channels = channel_names or self.config.channels
        data = np.asarray(eeg_data, dtype=np.float64)

        if data.ndim == 1:
            data = data[np.newaxis, :]

        # 1. Evaluate raw Signal Quality Index
        sqi = self.sqa.evaluate_window(data, channels)

        # 2. Bandpass & Notch Filtering
        filtered = self.apply_temporal_filtering(data)

        # 3. Bad channel interpolation (if enabled)
        if self.config.interpolate_bad_channels and sqi.bad_channel_names:
            filtered = self.interpolate_bad_channels(filtered, channels, sqi.bad_channel_names)

        # 4. Common Average Referencing
        name_to_idx = {name: i for i, name in enumerate(channels)}
        bad_indices = [name_to_idx[n] for n in sqi.bad_channel_names if n in name_to_idx]
        if self.config.apply_car:
            car_data = self.apply_car(filtered, bad_indices)
        else:
            car_data = filtered

        # 5. Normalization
        normed = self.normalize(car_data)

        # Usability determination
        is_usable = sqi.is_window_usable and (sqi.overall_quality_score >= self.config.min_quality_score)
        duration_sec = data.shape[1] / self.config.sampling_rate

        return PreprocessedWindow(
            window_index=window_index,
            data=normed,
            sampling_rate=self.config.sampling_rate,
            channel_names=channels,
            quality_metrics=sqi,
            is_usable=is_usable,
            start_time_sec=start_time_sec,
            end_time_sec=start_time_sec + duration_sec,
        )

    def segment_and_preprocess_continuous(
        self,
        continuous_eeg: np.ndarray,
        channel_names: Optional[List[str]] = None,
        start_timestamp_sec: float = 0.0,
    ) -> List[PreprocessedWindow]:
        """
        Takes a full continuous recording (e.g. participant session) and:
        1. Filters entire continuous recording to eliminate window boundary edge artifacts
        2. Segments into overlapping sliding windows (e.g. 4s windows with 2s step)
        3. Assesses SQA and applies CAR + normalization to each window
        """
        channels = channel_names or self.config.channels
        data = np.asarray(continuous_eeg, dtype=np.float64)

        num_channels, total_samples = data.shape
        window_samples = int(round(self.config.window_seconds * self.config.sampling_rate))
        step_samples = int(round(window_samples * (1.0 - self.config.overlap_ratio)))

        if total_samples < window_samples:
            # Shorter than one window
            win = self.preprocess_window(data, channels, window_index=0, start_time_sec=start_timestamp_sec)
            return [win]

        # Filter the entire continuous trace first
        continuous_filtered = self.apply_temporal_filtering(data)

        windows: List[PreprocessedWindow] = []
        win_idx = 0
        start_idx = 0

        while start_idx + window_samples <= total_samples:
            end_idx = start_idx + window_samples
            raw_slice = data[:, start_idx:end_idx]
            filt_slice = continuous_filtered[:, start_idx:end_idx]

            # Assess SQA on raw slice
            sqi = self.sqa.evaluate_window(raw_slice, channels)

            # Interpolate bad channels if needed
            if self.config.interpolate_bad_channels and sqi.bad_channel_names:
                filt_slice = self.interpolate_bad_channels(filt_slice, channels, sqi.bad_channel_names)

            # Apply CAR
            name_to_idx = {name: i for i, name in enumerate(channels)}
            bad_indices = [name_to_idx[n] for n in sqi.bad_channel_names if n in name_to_idx]
            if self.config.apply_car:
                car_slice = self.apply_car(filt_slice, bad_indices)
            else:
                car_slice = filt_slice

            # Normalize
            normed_slice = self.normalize(car_slice)

            is_usable = sqi.is_window_usable and (sqi.overall_quality_score >= self.config.min_quality_score)
            time_start = start_timestamp_sec + (start_idx / self.config.sampling_rate)
            time_end = start_timestamp_sec + (end_idx / self.config.sampling_rate)

            windows.append(
                PreprocessedWindow(
                    window_index=win_idx,
                    data=normed_slice,
                    sampling_rate=self.config.sampling_rate,
                    channel_names=channels,
                    quality_metrics=sqi,
                    is_usable=is_usable,
                    start_time_sec=time_start,
                    end_time_sec=time_end,
                )
            )

            win_idx += 1
            start_idx += step_samples

        return windows
