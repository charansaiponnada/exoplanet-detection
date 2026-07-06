"""
Vetting: the standard classical tests used by real exoplanet pipelines
(e.g. the Kepler/TESS vetting reports) to distinguish a genuine planetary
transit from common false-positive scenarios, WITHOUT needing a trained
classifier. These are fast, explainable, and directly satisfy the PS's
requirement to categorize dips into transit / eclipse / blend / other.

Tests implemented:
  1. Odd-even depth test: if odd- and even-numbered transits have
     significantly different depths, the signal is likely a diluted/blended
     eclipsing binary (true planets show consistent depth every transit).
  2. Secondary eclipse search: a detectable dip at phase 0.5 indicates a
     self-luminous companion (binary star), not a planet.
  3. V-shape vs U-shape ratio: grazing eclipsing binaries produce sharp
     V-shaped dips; planetary transits produce flatter U-shaped (trapezoidal)
     dips. Measured via the ratio of full-width-at-quarter-depth to
     full-width-at-half-depth.
"""

import numpy as np


def odd_even_test(time, flux, period_days, t0_days, duration_hours, n_durations=1.5):
    """Compare transit depths at odd vs even epoch numbers."""
    duration_days = duration_hours / 24
    epoch = np.round((time - t0_days) / period_days)
    half_window = (duration_days * n_durations) / 2

    phase = ((time - t0_days + period_days / 2) % period_days) - period_days / 2
    in_transit = np.abs(phase) < half_window

    odd_mask = in_transit & (epoch % 2 == 1)
    even_mask = in_transit & (epoch % 2 == 0)

    if odd_mask.sum() < 3 or even_mask.sum() < 3:
        return {"odd_depth_ppm": None, "even_depth_ppm": None, "significant_mismatch": None,
                "note": "insufficient transits in one parity to compare"}

    baseline = np.median(flux[~in_transit])
    odd_depth = (baseline - np.median(flux[odd_mask])) * 1e6
    even_depth = (baseline - np.median(flux[even_mask])) * 1e6

    mismatch = abs(odd_depth - even_depth) / max(abs(odd_depth), abs(even_depth), 1e-6)
    return {
        "odd_depth_ppm": float(odd_depth),
        "even_depth_ppm": float(even_depth),
        "fractional_mismatch": float(mismatch),
        "significant_mismatch": bool(mismatch > 0.3),  # >30% difference is a red flag
    }


def secondary_eclipse_test(time, flux, period_days, t0_days, duration_hours, n_durations=1.5):
    """Search for a dip at phase 0.5 (opposite side of the orbit)."""
    duration_days = duration_hours / 24
    half_window = (duration_days * n_durations) / 2

    sec_phase = ((time - t0_days - period_days / 2 + period_days / 2) % period_days) - period_days / 2
    in_secondary = np.abs(sec_phase) < half_window
    out_of_eclipse = ~in_secondary

    if in_secondary.sum() < 3:
        return {"secondary_depth_ppm": None, "secondary_detected": None,
                "note": "insufficient coverage at secondary phase"}

    baseline = np.median(flux[out_of_eclipse])
    sec_flux = np.median(flux[in_secondary])
    sec_depth_ppm = (baseline - sec_flux) * 1e6
    noise_ppm = np.std(flux[out_of_eclipse]) * 1e6

    detected = sec_depth_ppm > 3 * noise_ppm  # > 3-sigma dip at secondary phase
    return {
        "secondary_depth_ppm": float(sec_depth_ppm),
        "noise_level_ppm": float(noise_ppm),
        "secondary_detected": bool(detected),
    }


def shape_test(phase_days, flux, duration_hours, period_days):
    """
    Approximate V-shape vs U-shape via the ratio of the dip's width at
    quarter-depth to its width at half-depth. Values near 1 => boxy/flat
    (U-shaped, planet-like); values notably > 1 => V-shaped (grazing EB-like).
    """
    duration_days = duration_hours / 24
    window = duration_days * 1.5
    mask = np.abs(phase_days) < window
    if mask.sum() < 10:
        return {"shape_ratio": None, "note": "insufficient in-transit points for shape test"}

    p = phase_days[mask]
    f = flux[mask]
    baseline = np.median(flux[np.abs(phase_days) > window])
    depth = baseline - np.min(f)
    if depth <= 0:
        return {"shape_ratio": None, "note": "no detectable dip in window"}

    half_level = baseline - 0.5 * depth
    quarter_level = baseline - 0.25 * depth

    half_pts = p[f < half_level]
    quarter_pts = p[f < quarter_level]
    half_width = float(half_pts.max() - half_pts.min()) if len(half_pts) >= 2 else 0
    quarter_width = float(quarter_pts.max() - quarter_pts.min()) if len(quarter_pts) >= 2 else 0

    ratio = (quarter_width / half_width) if half_width > 0 else None
    is_v_shaped = bool(ratio is not None and ratio > 1.6)
    return {"shape_ratio": ratio, "likely_v_shaped_grazing_eb": is_v_shaped}


def classify_signal(odd_even_result, secondary_result, shape_result):
    """
    Combine the three classical tests into a single category label.
    This is the explainable, zero-training-data classifier that satisfies
    the PS's 'categorize into transits/eclipses/blends/other' requirement
    for the proposal stage -- the Mamba-based learned classifier is the
    planned enhancement once the curated training set is provided.
    """
    flags = []
    if odd_even_result.get("significant_mismatch"):
        flags.append("odd-even depth mismatch (possible diluted EB)")
    if secondary_result.get("secondary_detected"):
        flags.append("secondary eclipse detected (binary star)")
    if shape_result.get("likely_v_shaped_grazing_eb"):
        flags.append("V-shaped dip (possible grazing eclipsing binary)")

    if not flags:
        label = "candidate_planetary_transit"
    elif len(flags) >= 2:
        label = "likely_eclipsing_binary"
    else:
        label = "ambiguous_requires_followup"

    return {"label": label, "flags": flags}
