"""
Candidate detection via Box Least Squares (BLS).

BLS searches a grid of trial periods/durations and finds the box-shaped dip
that best matches the data, returning the period, duration, and depth that
maximize detection significance (power). This is the same algorithm used
operationally by the TESS/Kepler pipelines, via astropy's implementation.
"""

import numpy as np
from astropy.timeseries import BoxLeastSquares


def run_bls(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    min_period_days: float = 0.5,
    max_period_days: float = 15.0,
    n_periods: int = 5000,
):
    """
    Run a BLS period search and return the best-fit candidate plus the
    full periodogram (for plotting / significance assessment).
    """
    durations_hours = np.linspace(0.5, 8.0, 20)
    durations_days = durations_hours / 24

    model = BoxLeastSquares(time, flux, dy=flux_err)
    periods = np.linspace(min_period_days, max_period_days, n_periods)
    periodogram = model.power(periods, durations_days)

    best_idx = np.argmax(periodogram.power)
    best_period = periodogram.period[best_idx]
    best_duration = periodogram.duration[best_idx]
    best_t0 = periodogram.transit_time[best_idx]
    best_depth = periodogram.depth[best_idx]
    best_power = periodogram.power[best_idx]

    # Signal-to-noise of the detection: how far the best peak sits above the
    # noise floor of the periodogram (median absolute deviation of power).
    noise_floor = np.median(np.abs(periodogram.power - np.median(periodogram.power)))
    snr = (best_power - np.median(periodogram.power)) / (noise_floor + 1e-12)

    candidate = {
        "period_days": float(best_period),
        "t0_days": float(best_t0),
        "duration_hours": float(best_duration * 24),
        "depth_ppm": float(best_depth * 1e6),
        "bls_power": float(best_power),
        "detection_snr": float(snr),
    }
    return candidate, periodogram


def phase_fold(time: np.ndarray, flux: np.ndarray, period_days: float, t0_days: float):
    """Phase-fold a light curve on a given period/epoch, centered at phase 0."""
    phase = ((time - t0_days + period_days / 2) % period_days) - period_days / 2
    order = np.argsort(phase)
    return phase[order], flux[order]
