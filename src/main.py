"""
End-to-end exoplanet transit detection pipeline.

Usage:
    Real data (run locally / Colab with MAST access):
        python src/main.py --real "TIC 307210830"

    Synthetic validation run (works anywhere, no internet needed):
        python src/main.py --synthetic planet
        python src/main.py --synthetic eb        # injected eclipsing binary, tests vetting logic
"""

import argparse
import json
import sys

import numpy as np
import matplotlib.pyplot as plt

from data_io import load_real_target, make_synthetic_light_curve
from detrend import detrend_flux, sigma_clip
from bls_detect import run_bls, phase_fold
from vetting import odd_even_test, secondary_eclipse_test, shape_test, classify_signal


def run_pipeline(time, flux, flux_err, label="target", truth=None, out_prefix="result"):
    time, flux, flux_err = sigma_clip(time, flux, flux_err, sigma=5.0)
    flat_flux, trend = detrend_flux(time, flux, window_days=0.5)

    candidate, periodogram = run_bls(time, flat_flux, flux_err)

    phase, folded_flux = phase_fold(time, flat_flux, candidate["period_days"], candidate["t0_days"])

    oe = odd_even_test(time, flat_flux, candidate["period_days"], candidate["t0_days"], candidate["duration_hours"])
    sec = secondary_eclipse_test(time, flat_flux, candidate["period_days"], candidate["t0_days"], candidate["duration_hours"])
    shape = shape_test(phase, folded_flux, candidate["duration_hours"], candidate["period_days"])
    classification = classify_signal(oe, sec, shape)

    results = {
        "target": label,
        "recovered_parameters": candidate,
        "vetting": {"odd_even": oe, "secondary_eclipse": sec, "shape": shape},
        "classification": classification,
    }
    if truth is not None:
        results["ground_truth"] = truth
        results["period_error_pct"] = abs(candidate["period_days"] - truth["period_days"]) / truth["period_days"] * 100
        results["depth_error_pct"] = abs(candidate["depth_ppm"] - truth["depth_ppm"]) / truth["depth_ppm"] * 100
        results["duration_error_pct"] = abs(candidate["duration_hours"] - truth["duration_hours"]) / truth["duration_hours"] * 100

    # --- Figure: raw -> detrended -> BLS periodogram -> phase-folded ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Exoplanet Transit Detection Pipeline — {label}", fontsize=14, fontweight="bold")

    axes[0, 0].plot(time, flux, ".", ms=2, color="#4477AA")
    axes[0, 0].plot(time, trend, "-", color="#EE6677", lw=1.5, label="fitted trend")
    axes[0, 0].set_title("Raw light curve + detrending fit")
    axes[0, 0].set_xlabel("Time (days)")
    axes[0, 0].set_ylabel("Normalized flux")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(time, flat_flux, ".", ms=2, color="#228833")
    axes[0, 1].set_title("Detrended (flattened) light curve")
    axes[0, 1].set_xlabel("Time (days)")
    axes[0, 1].set_ylabel("Relative flux")

    axes[1, 0].plot(periodogram.period, periodogram.power, "-", color="#AA3377", lw=0.8)
    axes[1, 0].axvline(candidate["period_days"], color="black", ls="--", lw=1,
                        label=f"best period = {candidate['period_days']:.4f} d")
    axes[1, 0].set_title(f"BLS periodogram (SNR = {candidate['detection_snr']:.1f})")
    axes[1, 0].set_xlabel("Trial period (days)")
    axes[1, 0].set_ylabel("BLS power")
    axes[1, 0].legend(fontsize=8)

    window = candidate["duration_hours"] / 24 * 3
    mask = np.abs(phase) < window
    axes[1, 1].plot(phase[mask] * 24, folded_flux[mask], ".", ms=3, color="#4477AA", alpha=0.5)
    bin_edges = np.linspace(-window, window, 30)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    binned = [np.median(folded_flux[(phase >= bin_edges[i]) & (phase < bin_edges[i + 1])])
              if np.any((phase >= bin_edges[i]) & (phase < bin_edges[i + 1])) else np.nan
              for i in range(len(bin_edges) - 1)]
    axes[1, 1].plot(bin_centers * 24, binned, "-", color="black", lw=2, label="binned")
    axes[1, 1].set_title(f"Phase-folded transit (depth = {candidate['depth_ppm']:.0f} ppm, "
                          f"duration = {candidate['duration_hours']:.2f} hr)")
    axes[1, 1].set_xlabel("Hours from mid-transit")
    axes[1, 1].set_ylabel("Relative flux")
    axes[1, 1].legend(fontsize=8)

    plt.tight_layout()
    fig_path = f"{out_prefix}_{label.replace(' ', '_')}.png"
    plt.savefig(fig_path, dpi=130)
    plt.close(fig)
    results["figure_path"] = fig_path

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", type=str, default=None, help='TIC ID, e.g. "TIC 307210830"')
    parser.add_argument("--synthetic", type=str, default=None, choices=["planet", "eb"],
                         help="Run on a synthetic injected signal instead of real data")
    parser.add_argument("--out", type=str, default="/home/claude/exoplanet-pipeline/output")
    args = parser.parse_args()

    if args.real:
        df, meta = load_real_target(args.real)
        results = run_pipeline(df["time"].values, df["flux"].values, df["flux_err"].values,
                                label=args.real, out_prefix=args.out)
        results["source_meta"] = meta
    elif args.synthetic == "planet":
        df, truth = make_synthetic_light_curve(period_days=3.5, transit_depth_ppm=2500,
                                                 transit_duration_hours=2.5, seed=1)
        results = run_pipeline(df["time"].values, df["flux"].values, df["flux_err"].values,
                                label="synthetic_planet", truth=truth, out_prefix=args.out)
    elif args.synthetic == "eb":
        df, truth = make_synthetic_light_curve(period_days=4.2, transit_depth_ppm=8000,
                                                 transit_duration_hours=3.0,
                                                 add_secondary_eclipse=True, secondary_depth_ppm=2500,
                                                 seed=2)
        results = run_pipeline(df["time"].values, df["flux"].values, df["flux_err"].values,
                                label="synthetic_eclipsing_binary", truth=truth, out_prefix=args.out)
    else:
        print("Specify --real \"TIC ...\" or --synthetic [planet|eb]")
        sys.exit(1)

    results_path = f"{args.out}_{(args.real or args.synthetic).replace(' ', '_')}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(json.dumps(results, indent=2, default=str))
    print(f"\nFigure saved to: {results['figure_path']}")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
