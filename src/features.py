"""
Feature engineering (Layer 4): describe each BLS candidate mathematically so
downstream scoring -- and a future learned classifier -- has more to work
with than the raw period/depth/duration triple.

Every feature here is computed from quantities the classical pipeline
already has (the phase-folded curve, the BLS candidate, the three vetting
tests), so this layer adds no new dependencies and runs unchanged on
synthetic and real targets. The output dict is a flat, JSON-safe feature
vector -- exactly the shape a future learned classifier (Layer 5) would
consume as input once labeled training data is available.
"""

import numpy as np
from scipy import stats


def _in_transit_mask(phase_days, duration_hours, n_durations=1.0):
    duration_days = duration_hours / 24
    half_window = duration_days * n_durations / 2
    return np.abs(phase_days) < half_window


def per_transit_depths(time, flux, period_days, t0_days, duration_hours):
    """
    Depth measured independently for each individual transit epoch, to check
    consistency across the full baseline. A real planet's transits should
    all have the same depth; blends, artifacts, or a wrong period usually
    don't.
    """
    duration_days = duration_hours / 24
    half_window = duration_days * 0.75
    epoch = np.round((time - t0_days) / period_days)
    phase = ((time - t0_days + period_days / 2) % period_days) - period_days / 2
    in_transit = np.abs(phase) < half_window

    out_baseline = np.median(flux[~in_transit])
    depths = []
    for e in np.unique(epoch[in_transit]):
        pts = flux[in_transit & (epoch == e)]
        if pts.size >= 3:
            depths.append(float((out_baseline - np.median(pts)) * 1e6))
    return depths


def _red_noise_ratio(time, flux, bin_hours=1.0):
    """
    CDPP-like red-noise proxy: ratio of the scatter in ~1-hour binned flux
    to the scatter expected if the noise were purely white (averaging down
    as 1/sqrt(N)). Correlated ("red") noise -- systematics, stellar
    variability residuals -- doesn't average down that way, so a ratio well
    above 1 flags a light curve with more structure than shot noise alone.
    """
    cadence_days = np.median(np.diff(time))
    bin_days = bin_hours / 24
    n_per_bin = max(int(bin_days / cadence_days), 1)
    if n_per_bin < 2 or flux.size < n_per_bin * 4:
        return None

    n_bins = flux.size // n_per_bin
    trimmed = flux[: n_bins * n_per_bin].reshape(n_bins, n_per_bin)
    binned_std = np.std(np.mean(trimmed, axis=1))
    raw_std = np.std(flux)
    expected_binned_std = raw_std / np.sqrt(n_per_bin)
    return float(binned_std / expected_binned_std) if expected_binned_std > 0 else None


def extract_features(time, flux, phase, folded_flux, candidate, oe, sec, shape):
    """Build the Layer-4 feature vector for one candidate."""
    period_days = candidate["period_days"]
    duration_hours = candidate["duration_hours"]
    depth_ppm = candidate["depth_ppm"]

    baseline_span = time.max() - time.min()
    transit_count = int(baseline_span / period_days) + 1

    depths = per_transit_depths(time, flux, period_days, candidate["t0_days"], duration_hours)
    depth_consistency_std_ppm = float(np.std(depths)) if len(depths) >= 2 else None
    depth_consistency_frac = (
        float(np.std(depths) / abs(np.mean(depths)))
        if len(depths) >= 2 and np.mean(depths) != 0
        else None
    )

    in_transit = _in_transit_mask(phase, duration_hours, n_durations=1.0)
    out_of_transit = ~_in_transit_mask(phase, duration_hours, n_durations=3.0)

    oot_flux = folded_flux[out_of_transit]
    it_flux = folded_flux[in_transit]

    noise_ppm = float(np.std(oot_flux) * 1e6)
    skewness = float(stats.skew(oot_flux)) if oot_flux.size >= 8 else None
    kurtosis = float(stats.kurtosis(oot_flux)) if oot_flux.size >= 8 else None

    # Shannon entropy of the out-of-transit flux distribution: featureless
    # white noise is high-entropy for a fixed bin count; residual structure
    # (uncorrected systematics, spot modulation) tends to lower it.
    hist, _ = np.histogram(oot_flux, bins=20, density=False)
    probs = hist[hist > 0] / hist.sum()
    entropy = float(-np.sum(probs * np.log2(probs))) if probs.size else None

    # Ingress/egress asymmetry: compare mean residual in the first vs second
    # half of the in-transit window, and the slope of the outer 25% of
    # points on each side (how gradual vs abrupt the transit edges are).
    symmetry_ppm = None
    ingress_egress_slope = None
    if it_flux.size >= 8:
        it_phase = phase[in_transit]
        order = np.argsort(it_phase)
        it_phase_sorted, it_flux_sorted = it_phase[order], it_flux[order]
        mid = len(it_phase_sorted) // 2
        first_half_mean = np.mean(it_flux_sorted[:mid])
        second_half_mean = np.mean(it_flux_sorted[mid:])
        symmetry_ppm = float(abs(first_half_mean - second_half_mean) * 1e6)

        edge_n = max(len(it_phase_sorted) // 4, 2)
        ingress_slope = np.polyfit(it_phase_sorted[:edge_n], it_flux_sorted[:edge_n], 1)[0]
        egress_slope = np.polyfit(it_phase_sorted[-edge_n:], it_flux_sorted[-edge_n:], 1)[0]
        ingress_egress_slope = float((abs(ingress_slope) + abs(egress_slope)) / 2)

    depth_snr = float(depth_ppm / noise_ppm) if noise_ppm > 0 else None
    red_noise_ratio = _red_noise_ratio(time, flux)

    return {
        "transit_count": transit_count,
        "n_transits_measured": len(depths),
        "depth_ppm": depth_ppm,
        "duration_hours": duration_hours,
        "period_days": period_days,
        "depth_snr": depth_snr,
        "noise_ppm": noise_ppm,
        "depth_consistency_std_ppm": depth_consistency_std_ppm,
        "depth_consistency_frac": depth_consistency_frac,
        "symmetry_ppm": symmetry_ppm,
        "ingress_egress_slope": ingress_egress_slope,
        "shape_ratio": shape.get("shape_ratio"),
        "odd_even_mismatch_frac": oe.get("fractional_mismatch"),
        "secondary_depth_ppm": sec.get("secondary_depth_ppm"),
        "skewness": skewness,
        "kurtosis": kurtosis,
        "entropy": entropy,
        "red_noise_ratio": red_noise_ratio,
    }
