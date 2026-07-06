"""
Generate a labeled training set for the Layer-5 ML classifier, entirely
from synthetic injections (see data_io.make_synthetic_light_curve).

There is no curated real dataset available yet (see README "Planned
Enhancement" -- that requires ISRO's labeled set, provided later). Until
then this is the only honest way to get labeled examples: inject known
signals, run them through the *real* detection pipeline (sigma-clip ->
detrend -> BLS -> phase-fold -> vetting -> feature extraction), and record
the resulting feature vector against the injected ground-truth class.

Four classes, matching the categories the PS asks for:
  - "planet"            clean box transit
  - "eclipsing_binary"  transit + secondary eclipse (deeper, may show
                         odd-even / shape mismatches)
  - "starspot"          sinusoidal rotational modulation, no real transit
  - "noise"             no injected signal at all

Usage:
    uv run src/synth_dataset.py --n-per-class 60 --out data/synthetic_training_set.csv
"""

import argparse

import numpy as np
import pandas as pd

from data_io import make_synthetic_light_curve
from detrend import detrend_flux, sigma_clip
from bls_detect import run_bls, phase_fold
from vetting import odd_even_test, secondary_eclipse_test, shape_test
from features import extract_features


def _inject_starspot(df, rot_period_days, amplitude_ppm, rng):
    """Add sinusoidal rotational modulation (no transit) to a noise-only curve."""
    phase_offset = rng.uniform(0, 2 * np.pi)
    df = df.copy()
    df["flux"] += amplitude_ppm * 1e-6 * np.sin(2 * np.pi * df["time"] / rot_period_days + phase_offset)
    return df


def make_example(label, rng, n_periods=2000):
    """Generate one labeled light curve, run it through the real pipeline, and
    return its Layer-4 feature vector (or None if BLS/features couldn't be
    computed, e.g. too few points survive sigma-clipping)."""
    period_days = rng.uniform(1.0, 12.0)
    duration_hours = rng.uniform(1.0, 5.0)
    seed = int(rng.integers(0, 1_000_000))

    if label == "planet":
        depth_ppm = rng.uniform(500, 6000)
        df, truth = make_synthetic_light_curve(
            period_days=period_days, transit_duration_hours=duration_hours,
            transit_depth_ppm=depth_ppm,
            stellar_noise_ppm=rng.uniform(100, 500), white_noise_ppm=rng.uniform(80, 300),
            seed=seed,
        )
    elif label == "eclipsing_binary":
        depth_ppm = rng.uniform(3000, 15000)
        df, truth = make_synthetic_light_curve(
            period_days=period_days, transit_duration_hours=duration_hours,
            transit_depth_ppm=depth_ppm, add_secondary_eclipse=True,
            secondary_depth_ppm=depth_ppm * rng.uniform(0.2, 0.8),
            stellar_noise_ppm=rng.uniform(100, 500), white_noise_ppm=rng.uniform(80, 300),
            seed=seed,
        )
    elif label == "starspot":
        df, truth = make_synthetic_light_curve(
            period_days=period_days, transit_duration_hours=duration_hours,
            transit_depth_ppm=0.0,
            stellar_noise_ppm=rng.uniform(100, 500), white_noise_ppm=rng.uniform(80, 300),
            seed=seed,
        )
        df = _inject_starspot(df, rot_period_days=rng.uniform(1.0, 12.0),
                               amplitude_ppm=rng.uniform(500, 4000), rng=rng)
    elif label == "noise":
        df, truth = make_synthetic_light_curve(
            period_days=period_days, transit_duration_hours=duration_hours,
            transit_depth_ppm=0.0,
            stellar_noise_ppm=rng.uniform(100, 500), white_noise_ppm=rng.uniform(80, 300),
            seed=seed,
        )
    else:
        raise ValueError(f"unknown label {label}")

    time, flux, flux_err = sigma_clip(df["time"].values, df["flux"].values, df["flux_err"].values, sigma=5.0)
    flat_flux, _ = detrend_flux(time, flux, window_days=0.5)
    try:
        candidate, _ = run_bls(time, flat_flux, flux_err, n_periods=n_periods)
        phase, folded_flux = phase_fold(time, flat_flux, candidate["period_days"], candidate["t0_days"])
        oe = odd_even_test(time, flat_flux, candidate["period_days"], candidate["t0_days"], candidate["duration_hours"])
        sec = secondary_eclipse_test(time, flat_flux, candidate["period_days"], candidate["t0_days"], candidate["duration_hours"])
        shape = shape_test(phase, folded_flux, candidate["duration_hours"], candidate["period_days"])
        feats = extract_features(time, flat_flux, phase, folded_flux, candidate, oe, sec, shape)
    except Exception:
        return None

    feats["label"] = label
    return feats


def build_dataset(n_per_class=60, seed=0, n_periods=2000):
    rng = np.random.default_rng(seed)
    rows = []
    for label in ["planet", "eclipsing_binary", "starspot", "noise"]:
        n_ok = 0
        attempts = 0
        while n_ok < n_per_class and attempts < n_per_class * 3:
            attempts += 1
            row = make_example(label, rng, n_periods=n_periods)
            if row is not None:
                rows.append(row)
                n_ok += 1
        print(f"{label}: {n_ok}/{n_per_class} examples generated ({attempts} attempts)")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-class", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-periods", type=int, default=2000,
                         help="BLS trial-period grid size (smaller = faster dataset generation)")
    parser.add_argument("--out", type=str, default="data/synthetic_training_set.csv")
    args = parser.parse_args()

    df = build_dataset(n_per_class=args.n_per_class, seed=args.seed, n_periods=args.n_periods)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df)} labeled examples to {args.out}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
