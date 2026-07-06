"""
Detrending: remove long-term stellar variability / instrumental systematics
so periodic transit dips become detectable above the noise floor.

Uses a Savitzky-Golay filter on a sliding window, which is the standard
lightweight approach (equivalent in spirit to lightkurve's .flatten()).
"""

import numpy as np
from scipy.signal import savgol_filter


def detrend_flux(time: np.ndarray, flux: np.ndarray, window_days: float = 0.5):
    """
    Flatten a light curve by dividing out a smooth Savitzky-Golay trend.

    window_days: smoothing window length in days. Should be a few times
    longer than the expected transit duration so real transits aren't
    smoothed away, but short enough to track stellar variability.
    """
    cadence = np.median(np.diff(time))
    window_points = int(window_days / cadence)
    # savgol requires odd window length >= polyorder+2
    if window_points % 2 == 0:
        window_points += 1
    window_points = max(window_points, 5)

    trend = savgol_filter(flux, window_length=window_points, polyorder=2)
    flat_flux = flux / trend

    return flat_flux, trend


def sigma_clip(time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray, sigma: float = 5.0):
    """Remove extreme outliers (cosmic rays, pointing jitter) before searching."""
    median = np.median(flux)
    std = np.std(flux)
    mask = np.abs(flux - median) < sigma * std
    return time[mask], flux[mask], flux_err[mask]
