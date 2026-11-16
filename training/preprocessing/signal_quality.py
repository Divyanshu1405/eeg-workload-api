"""
Signal Quality Assessment (SQA / SQI) for wearable Muse S EEG in the UNIVERSE dataset.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class ChannelQuality:
    channel_name: str
    is_valid: bool
    snr_db: float
    flatline_ratio: float
    clipping_ratio: float
    outlier_ratio: float
    std_uv: float
    status: str  # 'good', 'noisy', 'bad', 'flatline', 'clipped'
    reason: Optional[str] = None


@dataclass
class SignalQualityMetrics:
    is_window_usable: bool
    overall_quality_score: float  # 0.0 (worst) to 1.0 (best)
    channel_metrics: Dict[str, ChannelQuality] = field(default_factory=dict)
    bad_channel_names: List[str] = field(default_factory=list)
    rejection_reasons: List[str] = field(default_factory=list)


class SignalQualityAssessor:
    """
    Evaluates signal quality indices (SQI) for Muse S EEG recordings.
    Handles both zero-centered microvolt signals and Muse S DC-offset (~820 uV) raw ADC signals.
    """

    def __init__(
        self,
        sampling_rate: float = 256.0,
        flatline_std_threshold: float = 0.5,
        max_amplitude_uv: float = 150.0,
        clipping_threshold_uv: float = 400.0,
        max_bad_sample_ratio: float = 0.20,
        min_snr_db: float = 1.0,
        critical_channels: Optional[List[str]] = None,
    ):
        self.sampling_rate = sampling_rate
        self.flatline_std_threshold = flatline_std_threshold
        self.max_amplitude_uv = max_amplitude_uv
        self.clipping_threshold_uv = clipping_threshold_uv
        self.max_bad_sample_ratio = max_bad_sample_ratio
        self.min_snr_db = min_snr_db
        self.critical_channels = critical_channels or ["AF7", "AF8"]

    def compute_snr_db(self, signal: np.ndarray) -> float:
        n = len(signal)
        if n < 32:
            return 0.0

        centered = signal - np.mean(signal)
        fft_vals = np.fft.rfft(centered)
        psd = (np.abs(fft_vals) ** 2) / (n * self.sampling_rate)
        freqs = np.fft.rfftfreq(n, 1.0 / self.sampling_rate)

        signal_mask = (freqs >= 1.0) & (freqs <= 40.0)
        noise_mask = ((freqs >= 0.1) & (freqs < 1.0)) | (
            (freqs >= 45.0) & (freqs <= min(100.0, self.sampling_rate / 2))
        )

        p_signal = np.sum(psd[signal_mask])
        p_noise = np.sum(psd[noise_mask])

        if p_noise <= 1e-12:
            return 30.0
        if p_signal <= 1e-12:
            return -20.0

        snr = 10.0 * np.log10(p_signal / p_noise)
        return float(np.clip(snr, -30.0, 40.0))

    def evaluate_channel(self, channel_data: np.ndarray, channel_name: str) -> ChannelQuality:
        data = np.asarray(channel_data, dtype=np.float64)
        n = len(data)

        if n == 0 or np.all(np.isnan(data)):
            return ChannelQuality(
                channel_name=channel_name,
                is_valid=False,
                snr_db=-30.0,
                flatline_ratio=1.0,
                clipping_ratio=0.0,
                outlier_ratio=1.0,
                std_uv=0.0,
                status="flatline",
                reason="Channel data is empty or entirely NaN",
            )

        if np.any(np.isnan(data)):
            mask = np.isnan(data)
            data = np.interp(np.arange(n), np.where(~mask)[0], data[~mask])

        # Remove DC offset (e.g., Muse S ~820 uV ADC baseline) for AC artifact evaluation
        centered = data - np.mean(data)
        std_uv = float(np.std(centered))
        abs_centered = np.abs(centered)

        # 1. Flatline check
        flatline_samples = np.abs(np.diff(data)) < 1e-4
        flatline_ratio = float(np.sum(flatline_samples) / max(1, n - 1))
        if std_uv < self.flatline_std_threshold or flatline_ratio > 0.8:
            return ChannelQuality(
                channel_name=channel_name,
                is_valid=False,
                snr_db=-30.0,
                flatline_ratio=flatline_ratio,
                clipping_ratio=0.0,
                outlier_ratio=0.0,
                std_uv=std_uv,
                status="flatline",
                reason=f"Channel flatlined (std={std_uv:.2f} uV, flat_ratio={flatline_ratio:.2f})",
            )

        # 2. Clipping / ADC rail saturation check (AC excursion >= 400 uV or Muse ADC rails <=1 / >=1645 uV)
        is_muse_dc = 650.0 <= np.mean(data) <= 1050.0
        if is_muse_dc:
            clipping_samples = (abs_centered >= self.clipping_threshold_uv) | (data <= 1.0) | (data >= 1645.0)
        else:
            clipping_samples = np.abs(data) >= self.clipping_threshold_uv

        clipping_ratio = float(np.sum(clipping_samples) / n)
        if clipping_ratio > 0.05:
            return ChannelQuality(
                channel_name=channel_name,
                is_valid=False,
                snr_db=-10.0,
                flatline_ratio=flatline_ratio,
                clipping_ratio=clipping_ratio,
                outlier_ratio=float(np.sum(abs_centered > self.max_amplitude_uv) / n),
                std_uv=std_uv,
                status="clipped",
                reason=f"Channel clipped/saturated ({clipping_ratio * 100:.1f}% samples)",
            )

        # 3. Outlier check (AC movement / muscle burst > 150 uV from baseline)
        outlier_samples = abs_centered > self.max_amplitude_uv
        outlier_ratio = float(np.sum(outlier_samples) / n)

        # 4. SNR estimation
        snr_db = self.compute_snr_db(centered)

        if outlier_ratio > self.max_bad_sample_ratio:
            return ChannelQuality(
                channel_name=channel_name,
                is_valid=False,
                snr_db=snr_db,
                flatline_ratio=flatline_ratio,
                clipping_ratio=clipping_ratio,
                outlier_ratio=outlier_ratio,
                std_uv=std_uv,
                status="bad",
                reason=f"High artifact ratio ({outlier_ratio * 100:.1f}% samples > {self.max_amplitude_uv} uV)",
            )

        if snr_db < self.min_snr_db:
            return ChannelQuality(
                channel_name=channel_name,
                is_valid=False,
                snr_db=snr_db,
                flatline_ratio=flatline_ratio,
                clipping_ratio=clipping_ratio,
                outlier_ratio=outlier_ratio,
                std_uv=std_uv,
                status="noisy",
                reason=f"Low SNR ({snr_db:.1f} dB < {self.min_snr_db:.1f} dB)",
            )

        return ChannelQuality(
            channel_name=channel_name,
            is_valid=True,
            snr_db=snr_db,
            flatline_ratio=flatline_ratio,
            clipping_ratio=clipping_ratio,
            outlier_ratio=outlier_ratio,
            std_uv=std_uv,
            status="good",
            reason=None,
        )

    def evaluate_window(
        self,
        eeg_data: np.ndarray,
        channel_names: List[str],
    ) -> SignalQualityMetrics:
        if eeg_data.ndim != 2:
            raise ValueError(f"Expected 2D array [num_channels, num_samples], got shape {eeg_data.shape}")

        if eeg_data.shape[0] != len(channel_names):
            raise ValueError(
                f"Channel count mismatch: array has {eeg_data.shape[0]} channels, but {len(channel_names)} names provided"
            )

        channel_metrics: Dict[str, ChannelQuality] = {}
        bad_channels: List[str] = []
        rejection_reasons: List[str] = []

        valid_channel_count = 0
        total_snr = 0.0

        for ch_idx, ch_name in enumerate(channel_names):
            ch_data = eeg_data[ch_idx, :]
            quality = self.evaluate_channel(ch_data, ch_name)
            channel_metrics[ch_name] = quality

            if quality.is_valid:
                valid_channel_count += 1
                total_snr += max(0.0, quality.snr_db)
            else:
                bad_channels.append(ch_name)
                if quality.reason:
                    rejection_reasons.append(f"[{ch_name}] {quality.reason}")

        ch_ratio = valid_channel_count / max(1, len(channel_names))
        avg_snr = (total_snr / max(1, valid_channel_count)) if valid_channel_count > 0 else 0.0
        snr_factor = np.clip(avg_snr / 20.0, 0.0, 1.0)
        overall_score = float(np.clip(0.6 * ch_ratio + 0.4 * snr_factor, 0.0, 1.0))

        critical_missing = [ch for ch in self.critical_channels if ch in bad_channels]
        if critical_missing:
            rejection_reasons.append(f"Critical channels compromised: {', '.join(critical_missing)}")

        frontal_good = any(ch in channel_metrics and channel_metrics[ch].is_valid for ch in self.critical_channels)
        is_usable = (valid_channel_count >= 2) and frontal_good and (overall_score >= 0.4)

        return SignalQualityMetrics(
            is_window_usable=is_usable,
            overall_quality_score=overall_score,
            channel_metrics=channel_metrics,
            bad_channel_names=bad_channels,
            rejection_reasons=rejection_reasons,
        )
