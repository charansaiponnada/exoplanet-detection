"""Shared per-event view generation, used by both the Kepler and TESS preprocessors.

Keeping this in one place is what makes the cross-mission experiment meaningful:
a network trained on Kepler can only be evaluated zero-shot on TESS if the two
input representations are produced by identical code.
"""

from __future__ import annotations

import numpy as np

from exonet import views as V
from exonet.kepler_io import transit_mask

DERIVED_NAMES = [
    "oe_stat", "sec_ratio", "sec_phase_frac", "bkspace", "n_cadences",
    "log_period", "duration_frac", "view_depth", "oot_scatter", "red_noise_ratio",
    "local_scale",
]


def generate_views(
    time: np.ndarray,
    flux: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    period: float,
    t0: float,
    duration_days: float,
    bkspace: float = np.nan,
    cadence_minutes: float = 29.4,
) -> tuple[dict[str, np.ndarray], dict[str, float]] | None:
    """Render every view and light-curve-derived scalar for one transit event.

    ``time`` must be in days and ``flux`` already detrended to scatter about 1.
    Returns ``None`` if the event parameters are unusable.
    """
    if not (np.isfinite(period) and period > 0):
        return None
    if not (np.isfinite(duration_days) and duration_days > 0):
        return None
    if time.size < 200:
        return None

    out: dict[str, np.ndarray] = {}
    out["global"] = V.global_view(time, flux, period, t0)
    out["local"], local_scale = V.local_view(
        time, flux, period, t0, duration_days, return_scale=True
    )
    odd, even, oe_stat = V.odd_even_views(time, flux, period, t0, duration_days)
    out["odd"], out["even"] = odd, even
    sec, sec_phase, sec_ratio = V.secondary_view(time, flux, period, t0, duration_days)
    out["secondary"] = sec
    half, double = V.harmonic_views(time, flux, period, t0, duration_days)
    out["half"], out["double"] = half, double
    cg, cl = V.centroid_views(time, cx, cy, period, t0, duration_days)
    out["cent_global"], out["cent_local"] = cg, cl

    in_tr = transit_mask(time, period, t0, duration_days, factor=1.0)
    oot = flux[~in_tr]
    oot_scatter = float(np.std(oot)) if oot.size > 10 else np.nan
    view_depth = (
        float(np.median(oot) - np.median(flux[in_tr])) if in_tr.sum() > 3 else np.nan
    )
    # Red-noise proxy: if the noise were white, binning n points would reduce the
    # scatter by sqrt(n); the excess over that expectation measures correlated
    # noise, which is what makes shallow transits hard to trust.  The bin is
    # fixed at six hours of wall-clock time rather than a fixed number of
    # cadences, so the statistic is comparable between Kepler's 29.4-minute
    # cadence and TESS's two-minute cadence.
    n_bin = max(2, int(round(360.0 / max(cadence_minutes, 1e-3))))
    if oot.size > 10 * n_bin:
        nb = oot.size // n_bin
        binned = np.median(oot[: nb * n_bin].reshape(nb, n_bin), axis=1)
        red = float(np.std(binned) * np.sqrt(n_bin) / (oot_scatter + 1e-12))
    else:
        red = np.nan

    derived = {
        "oe_stat": oe_stat,
        "sec_ratio": sec_ratio,
        "sec_phase_frac": sec_phase / period,
        "bkspace": bkspace,
        "n_cadences": float(time.size),
        "log_period": float(np.log10(period)),
        "duration_frac": float(duration_days / period),
        "view_depth": view_depth,
        "oot_scatter": oot_scatter,
        "red_noise_ratio": red,
        "local_scale": local_scale,
    }
    return out, derived
