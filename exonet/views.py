"""Phase-folding and view generation, including the harmonic-hypothesis views.

A "view" is a fixed-length, binned representation of the phase-folded light
curve.  The global and local views follow Shallue & Vanderburg (2018) so that
results are comparable with the published benchmark; the odd/even, secondary,
centroid and *harmonic* (P/2, 2P) views are the additional evidence channels
that PHANTOM reasons over.
"""

from __future__ import annotations

import numpy as np

GLOBAL_BINS = 2001
LOCAL_BINS = 201
LOCAL_DURATIONS = 4.0  # local view spans +/- this many transit durations


def phase_fold(
    time: np.ndarray, flux: np.ndarray, period: float, t0: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fold onto ``period`` about ``t0``; returns phase in [-P/2, P/2), sorted."""
    half = 0.5 * period
    phase = (time - t0 + half) % period - half
    order = np.argsort(phase)
    return phase[order], flux[order]


def bin_median(
    x_sorted: np.ndarray, y_sorted: np.ndarray, x_min: float, x_max: float, n_bins: int
) -> np.ndarray:
    """Median-bin sorted data onto a regular grid; empty bins become NaN."""
    edges = np.linspace(x_min, x_max, n_bins + 1)
    lo = np.searchsorted(x_sorted, edges[:-1], side="left")
    hi = np.searchsorted(x_sorted, edges[1:], side="left")
    out = np.full(n_bins, np.nan)
    for i in range(n_bins):
        if hi[i] > lo[i]:
            out[i] = np.median(y_sorted[lo[i] : hi[i]])
    return out


def _fill_and_normalise(v: np.ndarray, return_scale: bool = False):
    """Interpolate empty bins, median-centre at 0, and scale so the minimum is -1.

    The depth normalisation makes the network's task shape-discrimination rather
    than depth-discrimination; absolute depth is supplied separately as a scalar
    so that information is not lost.
    """
    n = v.size
    bad = ~np.isfinite(v)
    if bad.all():
        out = np.zeros(n, dtype=np.float32)
        return (out, 0.0) if return_scale else out
    if bad.any():
        idx = np.arange(n)
        v = v.copy()
        v[bad] = np.interp(idx[bad], idx[~bad], v[~bad])
    v = v - np.median(v)
    depth = np.abs(np.min(v))
    if np.isfinite(depth) and depth > 0:
        v = v / depth
    out = v.astype(np.float32)
    # ``depth`` is the scale factor divided out, in relative-flux units.  Keeping
    # it lets an absolute transit depth be reconstructed from the normalised view.
    return (out, float(depth)) if return_scale else out


def global_view(
    time: np.ndarray, flux: np.ndarray, period: float, t0: float, n_bins: int = GLOBAL_BINS
) -> np.ndarray:
    """Whole-orbit view: the full folded light curve at fixed resolution."""
    ph, fl = phase_fold(time, flux, period, t0)
    v = bin_median(ph, fl, -0.5 * period, 0.5 * period, n_bins)
    return _fill_and_normalise(v)


def local_view(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    n_bins: int = LOCAL_BINS,
    n_durations: float = LOCAL_DURATIONS,
    centre: float = 0.0,
    return_scale: bool = False,
):
    """Zoom on the event: +/- ``n_durations`` durations about ``centre`` phase."""
    ph, fl = phase_fold(time, flux, period, t0)
    span = min(0.5 * period, n_durations * duration)
    v = bin_median(ph, fl, centre - span, centre + span, n_bins)
    return _fill_and_normalise(v, return_scale=return_scale)


def odd_even_views(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    n_bins: int = LOCAL_BINS,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Local views built from alternating transits, plus their depth ratio.

    A genuine planet produces identical depths on odd and even epochs.  An
    eclipsing binary detected at half its true period produces primary and
    secondary eclipses of different depths in the two sets - the single most
    diagnostic classical false-positive test.
    """
    epoch = np.round((time - t0) / period)
    odd = (epoch % 2) != 0
    views = []
    depths = []
    for sel in (odd, ~odd):
        if sel.sum() < 20:
            views.append(np.zeros(n_bins, dtype=np.float32))
            depths.append(np.nan)
            continue
        ph, fl = phase_fold(time[sel], flux[sel], period, t0)
        span = min(0.5 * period, LOCAL_DURATIONS * duration)
        raw = bin_median(ph, fl, -span, span, n_bins)
        # depth measured before normalisation destroys the scale
        core = raw[int(0.4 * n_bins) : int(0.6 * n_bins)]
        base = np.nanmedian(raw) if np.isfinite(raw).any() else np.nan
        if np.isfinite(base) and np.isfinite(core).any():
            depths.append(float(base - np.nanmin(core)))
        else:
            depths.append(np.nan)
        views.append(_fill_and_normalise(raw))

    d_odd, d_even = depths
    if np.isfinite(d_odd) and np.isfinite(d_even) and (d_odd + d_even) > 0:
        # Symmetric fractional difference in [0, 2]; 0 for a planet.
        oe_stat = float(2.0 * abs(d_odd - d_even) / (d_odd + d_even))
    else:
        oe_stat = 0.0
    return views[0], views[1], oe_stat


def secondary_view(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    n_bins: int = LOCAL_BINS,
) -> tuple[np.ndarray, float, float]:
    """View centred on the deepest out-of-primary event ("secondary eclipse").

    Returns ``(view, secondary_phase, secondary_depth_relative_to_primary)``.
    A detectable secondary eclipse implies a self-luminous companion, i.e. a
    stellar binary rather than a planet.
    """
    ph, fl = phase_fold(time, flux, period, t0)
    coarse = bin_median(ph, fl, -0.5 * period, 0.5 * period, GLOBAL_BINS)
    centres = np.linspace(-0.5 * period, 0.5 * period, GLOBAL_BINS)
    base = np.nanmedian(coarse)
    primary_depth = base - np.nanmin(coarse) if np.isfinite(base) else np.nan

    # exclude the primary event and the fold edges
    excl = np.abs(centres) < max(2.0 * duration, 0.02 * period)
    search = np.where(np.isfinite(coarse) & ~excl, coarse, np.inf)
    if not np.isfinite(search).any():
        return np.zeros(n_bins, dtype=np.float32), 0.0, 0.0
    j = int(np.argmin(search))
    sec_phase = float(centres[j])
    sec_depth = float(base - coarse[j]) if np.isfinite(base) else 0.0
    ratio = float(sec_depth / primary_depth) if primary_depth and primary_depth > 0 else 0.0

    # Re-fold about the secondary epoch rather than windowing inside the primary
    # fold, so the window is always fully populated even when the secondary sits
    # near phase +/-0.5 and would otherwise run off the edge.
    view = local_view(time, flux, period, t0 + sec_phase, duration, n_bins)
    return view, sec_phase, ratio


def harmonic_views(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Views under the competing period hypotheses P/2 and 2P.

    These are the inputs to PHANTOM's harmonic-attention block.  If the reported
    period is the half-period alias of an eclipsing binary, the 2P fold separates
    the two eclipses into distinct depths; if the true period is P/2, the
    half-period fold sharpens rather than smears the event.
    """
    half = local_view(time, flux, 0.5 * period, t0, duration)
    double = global_view(time, flux, 2.0 * period, t0)
    return half, double


def centroid_views(
    time: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    period: float,
    t0: float,
    duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Folded views of the photometric centroid offset.

    If the flux dip originates from a blended neighbouring star rather than the
    target, the measured centroid shifts towards the true source during the
    event.  ``cx``/``cy`` are expected to be already detrended; the magnitude of
    the offset from the out-of-transit position is folded like the flux.
    """
    ok = np.isfinite(cx) & np.isfinite(cy)
    if ok.sum() < 100:
        return (
            np.zeros(GLOBAL_BINS, dtype=np.float32),
            np.zeros(LOCAL_BINS, dtype=np.float32),
        )
    t, x, y = time[ok], cx[ok], cy[ok]
    r = np.hypot(x - np.median(x), y - np.median(y))
    # invert so a centroid excursion looks like a "dip" and shares the
    # normalisation convention of the flux views
    g = global_view(t, -r, period, t0)
    l = local_view(t, -r, period, t0, duration)
    return g, l
