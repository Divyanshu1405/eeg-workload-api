"""
Feature Extraction for Mental Workload and Stress from Preprocessed Muse S EEG.

Extracts spectral, cognitive workload, and time-domain features from preprocessed
EEG epochs (e.g. 4s windows) specifically tailored to the 4-channel Muse S headband
(TP9, AF7, AF8, TP10) used in the UNIVERSE dataset.

Key Workload Markers:
- Frontal Theta / Alpha Ratio (AF7, AF8 theta / alpha)
- Engagement Index: Beta / (Alpha + Theta)
- Frontal Alpha Asymmetry (FAA): ln(Alpha_AF8) - ln(Alpha_AF7)
- Theta / Beta Ratio (TBR): executive function / cognitive demand
- Relative and Absolute Band Powers (Delta, Theta, Alpha, Beta, Gamma)
- Hjorth Parameters (Activity, Mobility, Complexity)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class FeatureConfig:
    sampling_rate: float = 256.0
    channel_names: List[str] = field(default_factory=lambda: ["TP9", "AF7", "AF8", "TP10"])
    bands: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "delta": (0.5, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
            "gamma": (30.0, 45.0),
        }
    )
    n_per_seg: int = 256  # 1.0 second FFT segment for Welch's method (256 samples @ 256 Hz)
    extract_time_domain: bool = True
    extract_hjorth: bool = True
    extract_cognitive_indices: bool = True


class UniverseFeatureExtractor:
    """
    Extracts high-fidelity cognitive state features from preprocessed EEG windows.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()

    def compute_psd(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes Power Spectral Density using Welch's periodogram with 50% overlap.
        Pure NumPy implementation to guarantee high performance and portability.
        """
        n = len(signal)
        fs = self.config.sampling_rate
        seg_len = min(self.config.n_per_seg, n)
        step = seg_len // 2

        if n < seg_len:
            # Fall back to single window FFT
            w = np.hanning(n)
            fft_v = np.fft.rfft((signal - np.mean(signal)) * w)
            psd = (np.abs(fft_v) ** 2) / (np.sum(w ** 2) * fs)
            freqs = np.fft.rfftfreq(n, 1.0 / fs)
            return freqs, psd

        # Hann window
        window = np.hanning(seg_len)
        win_norm = np.sum(window ** 2) * fs

        starts = list(range(0, n - seg_len + 1, step))
        if not starts:
            starts = [0]

        psd_accum = None
        for s in starts:
            segment = signal[s : s + seg_len]
            segment_centered = segment - np.mean(segment)
            fft_vals = np.fft.rfft(segment_centered * window)
            power = (np.abs(fft_vals) ** 2) / win_norm
            if psd_accum is None:
                psd_accum = power
            else:
                psd_accum += power

        psd = psd_accum / len(starts)
        freqs = np.fft.rfftfreq(seg_len, 1.0 / fs)
        return freqs, psd

    def extract_band_powers(self, freqs: np.ndarray, psd: np.ndarray) -> Dict[str, float]:
        """
        Calculates absolute and relative band power for standard EEG frequency ranges.

        Uses half-open intervals [f_low, f_high) so that boundary frequencies
        (e.g. 4 Hz between delta and theta) belong to exactly one band.
        """
        band_powers = {}
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

        # Total power in [0.5, 45.0) Hz — matches the union of all standard bands
        total_mask = (freqs >= 0.5) & (freqs < 45.0)
        total_power = float(np.sum(psd[total_mask]) * df)
        if total_power <= 1e-12:
            total_power = 1e-12

        for band_name, (f_low, f_high) in self.config.bands.items():
            # Half-open interval [f_low, f_high) avoids double-counting at boundaries
            mask = (freqs >= f_low) & (freqs < f_high)
            abs_power = float(np.sum(psd[mask]) * df) if np.any(mask) else 0.0
            rel_power = abs_power / total_power

            band_powers[f"{band_name}_abs"] = abs_power
            band_powers[f"{band_name}_rel"] = rel_power
            band_powers[f"{band_name}_log"] = float(np.log(max(abs_power, 1e-12)))

        band_powers["total_power"] = total_power
        return band_powers

    def compute_hjorth_parameters(self, signal: np.ndarray) -> Tuple[float, float, float]:
        """
        Computes Hjorth Activity, Mobility, and Complexity.
        """
        n = len(signal)
        if n < 3:
            return 0.0, 0.0, 0.0

        d1 = np.diff(signal)
        d2 = np.diff(d1)

        var0 = float(np.var(signal))
        var1 = float(np.var(d1))
        var2 = float(np.var(d2))

        activity = var0
        mobility = np.sqrt(var1 / max(var0, 1e-12))
        complexity = (np.sqrt(var2 / max(var1, 1e-12))) / max(mobility, 1e-12)

        return activity, float(mobility), float(complexity)

    def compute_time_domain_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Computes summary statistical moments for time-domain EEG.
        """
        n = len(signal)
        mean_v = float(np.mean(signal))
        std_v = float(np.std(signal))
        var_v = float(np.var(signal))
        rms_v = float(np.sqrt(np.mean(signal ** 2)))

        centered = signal - mean_v
        if std_v > 1e-6:
            skew_v = float(np.mean(centered ** 3) / (std_v ** 3))
            kurt_v = float(np.mean(centered ** 4) / (std_v ** 4)) - 3.0
        else:
            skew_v = 0.0
            kurt_v = 0.0

        return {
            "mean": mean_v,
            "std": std_v,
            "var": var_v,
            "rms": rms_v,
            "skewness": skew_v,
            "kurtosis": kurt_v,
        }

    def extract_features_from_window(
        self,
        eeg_window: np.ndarray,
        channel_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Extracts complete feature set from a multi-channel preprocessed window.
        Input shape: [num_channels, num_samples]
        Returns: Flattened feature dictionary with consistent nomenclature.
        """
        channels = channel_names or self.config.channel_names
        features: Dict[str, float] = {}

        channel_band_powers: Dict[str, Dict[str, float]] = {}

        for ch_idx, ch_name in enumerate(channels):
            ch_signal = eeg_window[ch_idx, :]

            # 1. Frequency domain (Welch PSD & Bands)
            freqs, psd = self.compute_psd(ch_signal)
            bp = self.extract_band_powers(freqs, psd)
            channel_band_powers[ch_name] = bp

            for k, val in bp.items():
                features[f"{ch_name}_{k}"] = val

            # 2. Time domain statistics
            if self.config.extract_time_domain:
                td = self.compute_time_domain_features(ch_signal)
                for k, val in td.items():
                    features[f"{ch_name}_{k}"] = val

            # 3. Hjorth parameters
            if self.config.extract_hjorth:
                act, mob, comp = self.compute_hjorth_parameters(ch_signal)
                features[f"{ch_name}_hjorth_activity"] = act
                features[f"{ch_name}_hjorth_mobility"] = mob
                features[f"{ch_name}_hjorth_complexity"] = comp

        # 4. Cognitive Workload & Stress Indices
        if self.config.extract_cognitive_indices:
            indices = self._compute_workload_indices(channel_band_powers)
            features.update(indices)

        return features

    def _compute_workload_indices(
        self,
        band_powers: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """
        Computes validated cognitive load and affective indices:
        1. Frontal Theta / Frontal Alpha (Cognitive workload index)
        2. Engagement Index: Beta / (Alpha + Theta)
        3. Frontal Alpha Asymmetry (FAA): ln(AF8_alpha) - ln(AF7_alpha)
        4. Frontal Theta / Frontal Beta (TBR)
        """
        indices: Dict[str, float] = {}

        # Frontal channels: AF7, AF8
        has_af7 = "AF7" in band_powers
        has_af8 = "AF8" in band_powers

        if has_af7 and has_af8:
            af7 = band_powers["AF7"]
            af8 = band_powers["AF8"]

            # Mean frontal powers
            frontal_theta = 0.5 * (af7["theta_abs"] + af8["theta_abs"])
            frontal_alpha = 0.5 * (af7["alpha_abs"] + af8["alpha_abs"])
            frontal_beta = 0.5 * (af7["beta_abs"] + af8["beta_abs"])

            # Workload: Theta / Alpha ratio
            indices["frontal_theta_alpha_ratio"] = float(
                frontal_theta / max(frontal_alpha, 1e-12)
            )

            # Frontal Engagement Index
            indices["frontal_engagement_index"] = float(
                frontal_beta / max(frontal_alpha + frontal_theta, 1e-12)
            )

            # Frontal Theta / Beta Ratio (TBR)
            indices["frontal_theta_beta_ratio"] = float(
                frontal_theta / max(frontal_beta, 1e-12)
            )

            # Frontal Alpha Asymmetry (FAA)
            # Positive FAA indicates relatively greater left frontal activity (approach/positive affect)
            alpha_af8 = max(af8["alpha_abs"], 1e-12)
            alpha_af7 = max(af7["alpha_abs"], 1e-12)
            indices["frontal_alpha_asymmetry"] = float(np.log(alpha_af8) - np.log(alpha_af7))

        # Channel-specific engagement and theta/alpha
        for ch, bp in band_powers.items():
            t = bp.get("theta_abs", 0.0)
            a = bp.get("alpha_abs", 0.0)
            b = bp.get("beta_abs", 0.0)

            indices[f"{ch}_theta_alpha_ratio"] = float(t / max(a, 1e-12))
            indices[f"{ch}_engagement_index"] = float(b / max(a + t, 1e-12))

        return indices

    def get_feature_names(self) -> List[str]:
        """
        Returns an ordered list of feature names by running on a dummy window.
        """
        n_samples = int(self.config.sampling_rate * 4.0)
        dummy = np.zeros((len(self.config.channel_names), n_samples), dtype=np.float64)
        sample_feats = self.extract_features_from_window(dummy)
        return sorted(list(sample_feats.keys()))
