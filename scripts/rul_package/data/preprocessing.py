"""Signal cleaning and feature extraction pipeline.

Matches the paper's Section 3.3:
  - 3.3.1 Signal cleaning: ±3σ outlier clipping
  - 3.3.2 Feature extraction: 7 TD + 3 FD statistical features + CWT scalograms
"""

import numpy as np
from scipy import signal, stats
from typing import Tuple
import pywt


def clip_outliers(x: np.ndarray, n_std: float = 3.0) -> np.ndarray:
    """Clip samples beyond ±n_std from the mean (Section 3.3.1)."""
    mu, sigma = np.mean(x), np.std(x)
    return np.clip(x, mu - n_std * sigma, mu + n_std * sigma)


def extract_time_domain_features(x: np.ndarray) -> dict[str, float]:
    """7 time-domain features from Eq. (1)."""
    n = len(x)
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    rms = np.sqrt(np.mean(x ** 2))
    variance = np.var(x, ddof=1)
    kurtosis = stats.kurtosis(x, fisher=False)  # Pearson kurtosis (normal=3)
    skewness = stats.skew(x)
    crest_factor = np.max(np.abs(x)) / (rms + 1e-12)
    peak_to_peak = np.max(x) - np.min(x)
    return {
        "RMS": rms,
        "Mean": mean,
        "Variance": variance,
        "Kurtosis": kurtosis,
        "Skewness": skewness,
        "Crest_Factor": crest_factor,
        "Peak_to_Peak": peak_to_peak,
    }


def extract_freq_domain_features(x: np.ndarray, fs: float = 25_600) -> dict[str, float]:
    """3 frequency-domain features from Eq. (2)."""
    n = len(x)
    fft_vals = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(fft_vals)
    power = mag ** 2
    power_norm = power / (np.sum(power) + 1e-12)

    # Dominant frequency
    dominant_freq = freqs[np.argmax(mag)]

    # Spectral energy
    spectral_energy = np.sum(power)

    # Spectral entropy
    spectral_entropy = -np.sum(power_norm * np.log2(power_norm + 1e-12))

    return {
        "Dominant_Frequency": dominant_freq,
        "Spectral_Energy": spectral_energy,
        "Spectral_Entropy": spectral_entropy,
    }


def compute_cwt_scalogram(
    x: np.ndarray,
    fs: float = 25_600,
    width: int = 32,
    height: int = 32,
    wavelet: str = "morl",
) -> np.ndarray:
    """Compute CWT scalogram, resize to (height, width), repeat to 3-channel RGB.

    Section 3.3.2 Time-frequency features: Morlet wavelet → 32×32 → RGB.
    """
    scales = np.linspace(1, 128, height)
    coeffs, _ = pywt.cwt(x, scales, wavelet, sampling_period=1.0 / fs)
    scalogram = np.abs(coeffs)  # (height, n_samples)

    # Downsample / interpolate to target width
    from scipy.ndimage import zoom
    scale_y = height / scalogram.shape[0]
    scale_x = width / scalogram.shape[1]
    scalogram_resized = zoom(scalogram, (scale_y, scale_x), order=1)

    # Log compress for visual dynamic range
    scalogram_log = np.log1p(scalogram_resized)

    # Normalise to [0, 1]
    s_min, s_max = scalogram_log.min(), scalogram_log.max()
    if s_max > s_min:
        scalogram_norm = (scalogram_log - s_min) / (s_max - s_min)
    else:
        scalogram_norm = np.zeros_like(scalogram_log)

    # Repeat to 3-channel (RGB) — paper says "converted to 3-channel RGB representations"
    scalogram_rgb = np.stack([scalogram_norm] * 3, axis=-1)  # (H, W, 3)
    return scalogram_rgb.astype(np.float32)


def extract_features_from_window(
    x: np.ndarray,
    fs: float = 25_600,
    cwt_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract both statistical features and CWT scalogram from a signal window.

    Returns
    -------
    stat_features : (10,) array  — 7 TD + 3 FD features
    cwt_scalogram : (32, 32, 3) array
    """
    x_clean = clip_outliers(x)

    td = extract_time_domain_features(x_clean)
    fd = extract_freq_domain_features(x_clean, fs)

    # Order must be consistent across all samples — paper lists them but order is implicit
    stat_feat = np.array([
        td["RMS"], td["Mean"], td["Variance"],
        td["Kurtosis"], td["Skewness"],
        td["Crest_Factor"], td["Peak_to_Peak"],
        fd["Dominant_Frequency"], fd["Spectral_Energy"], fd["Spectral_Entropy"],
    ], dtype=np.float32)

    cwt_img = compute_cwt_scalogram(x_clean, fs, cwt_size, cwt_size)
    return stat_feat, cwt_img
