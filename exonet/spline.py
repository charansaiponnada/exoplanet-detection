"""BIC-optimised B-spline detrending of Kepler light curves.

Stellar variability (spots, pulsations, rotation) and residual instrumental
trends dominate the raw PDCSAP flux at amplitudes far larger than a planetary
transit.  Following Vanderburg & Johnson (2014) and Shallue & Vanderburg (2018),
each contiguous segment of the light curve is divided by a cubic basis spline
fitted with iterative outlier rejection.  The knot spacing is a free parameter
chosen by minimising the Bayesian Information Criterion over the whole target,
which balances removing variability against absorbing the transit itself.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import LSQUnivariateSpline


def _fit_segment(
    time: np.ndarray,
    flux: np.ndarray,
    bkspace: float,
    mask: np.ndarray | None,
    max_iter: int = 5,
    sigma: float = 3.0,
) -> tuple[np.ndarray | None, int]:
    """Fit one segment.  Returns ``(spline_evaluated_at_time, n_knots)``.

    ``mask`` marks cadences to *exclude* from the fit (in-transit points); the
    fitted spline is still evaluated everywhere so the transit is preserved in
    the residual.
    """
    n = time.size
    if n < 8:
        return None, 0

    keep = np.ones(n, dtype=bool) if mask is None else ~mask
    if keep.sum() < 8:
        keep = np.ones(n, dtype=bool)

    span = time[-1] - time[0]
    n_interior = int(max(0, np.floor(span / bkspace) - 1))

    for _ in range(max_iter):
        t_fit, f_fit = time[keep], flux[keep]
        if t_fit.size < 8:
            return None, 0
        # Interior knots must lie strictly inside the data and have points
        # between them, otherwise LSQUnivariateSpline raises.
        if n_interior > 0:
            knots = np.linspace(t_fit[0], t_fit[-1], n_interior + 2)[1:-1]
            # drop knots that are not bracketed by at least one data point
            counts, _ = np.histogram(t_fit, bins=np.r_[t_fit[0], knots, t_fit[-1]])
            ok = np.ones(knots.size, dtype=bool)
            for i in range(knots.size):
                if counts[i] < 2 or counts[i + 1] < 2:
                    ok[i] = False
            knots = knots[ok]
        else:
            knots = np.array([])

        try:
            spl = LSQUnivariateSpline(t_fit, f_fit, knots, k=3)
        except Exception:
            return None, 0

        model_keep = spl(t_fit)
        resid = f_fit - model_keep
        scatter = np.std(resid)
        if not np.isfinite(scatter) or scatter == 0:
            break
        new_keep = keep.copy()
        idx = np.where(keep)[0]
        new_keep[idx] = np.abs(resid) < sigma * scatter
        if mask is not None:
            new_keep &= ~mask
        if new_keep.sum() < 8 or np.array_equal(new_keep, keep):
            keep = new_keep if new_keep.sum() >= 8 else keep
            break
        keep = new_keep

    return spl(time), knots.size + 4  # +4 for cubic spline coefficients


def choose_and_apply_spline(
    segments: list[tuple[np.ndarray, ...]],
    masks: list[np.ndarray] | None = None,
    bkspaces: np.ndarray | None = None,
) -> tuple[list[np.ndarray | None], float]:
    """Detrend all segments of a target with a shared, BIC-selected knot spacing.

    Returns ``(models, best_bkspace)`` where ``models[i]`` is the spline evaluated
    on segment ``i`` (or ``None`` if that segment could not be fitted).
    """
    if bkspaces is None:
        bkspaces = np.logspace(np.log10(0.5), np.log10(20.0), 20)

    best_bic = np.inf
    best_models: list[np.ndarray | None] = [None] * len(segments)
    best_bk = float(bkspaces[0])

    for bk in bkspaces:
        total_bic = 0.0
        models: list[np.ndarray | None] = []
        n_total = 0
        ok_any = False
        for i, seg in enumerate(segments):
            time, flux = seg[0], seg[1]
            mask = masks[i] if masks is not None else None
            model, n_par = _fit_segment(time, flux, bk, mask)
            models.append(model)
            if model is None:
                continue
            ok_any = True
            resid = flux - model
            n = resid.size
            n_total += n
            var = np.var(resid)
            if var <= 0 or not np.isfinite(var):
                continue
            # Gaussian log-likelihood with variance estimated from the residuals,
            # penalised by the number of spline coefficients.
            loglik = -0.5 * n * (np.log(2 * np.pi * var) + 1.0)
            total_bic += -2.0 * loglik + n_par * np.log(max(n, 2))
        if ok_any and total_bic < best_bic:
            best_bic = total_bic
            best_models = models
            best_bk = float(bk)

    return best_models, best_bk


def flatten_target(
    segments: list[tuple[np.ndarray, ...]],
    masks: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Detrend and concatenate a target's segments into a normalised light curve.

    Returns ``(time, relative_flux, bkspace)`` where ``relative_flux`` scatters
    about 1.0 and a transit appears as a negative excursion.
    """
    models, bk = choose_and_apply_spline(segments, masks)
    times, fluxes = [], []
    for seg, model in zip(segments, models):
        if model is None:
            continue
        time, flux = seg[0], seg[1]
        good = np.isfinite(model) & (model > 0)
        if good.sum() < 8:
            continue
        times.append(time[good])
        fluxes.append(flux[good] / model[good])
    if not times:
        return np.array([]), np.array([]), bk
    return np.concatenate(times), np.concatenate(fluxes), bk
