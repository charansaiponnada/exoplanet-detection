"""
Data loading for the exoplanet detection pipeline.

Two paths:
  1. load_real_target(tic_id)  -> pulls real TESS light curve via Lightkurve/MAST.
     Requires internet access to archive.stsci.edu / MAST -- run this locally
     or in Colab where that's reachable.
  2. make_synthetic_light_curve(...) -> generates a physically-realistic
     injected-transit light curve (limb-darkened box-ish transit + correlated
     red noise + white noise) so the rest of the pipeline can be built and
     validated without network access to the archive.

Both paths return the same shape: a pandas DataFrame with columns
['time', 'flux', 'flux_err'].
"""

import numpy as np
import pandas as pd


def load_real_target(tic_id: str, sector: int | None = None):
    """
    Download a real TESS light curve for a given TIC ID using Lightkurve.

    Example:
        df, meta = load_real_target("TIC 307210830")  # Pi Mensae, confirmed planet host

    NOTE: requires outbound access to MAST (archive.stsci.edu). If you see a
    network/connection error, you're likely running somewhere without that
    access -- run this function locally or in Colab instead.
    """
    import lightkurve as lk

    search = lk.search_lightcurve(tic_id, mission="TESS", sector=sector, author="SPOC")
    if len(search) == 0:
        raise ValueError(f"No SPOC light curves found for {tic_id}")

    lc_collection = search.download_all()
    lc = lc_collection.stitch()  # combine multiple sectors if present
    lc = lc.remove_nans().remove_outliers(sigma=5)

    df = pd.DataFrame({
        "time": lc.time.value,
        "flux": lc.flux.value,
        "flux_err": lc.flux_err.value,
    })
    meta = {"tic_id": tic_id, "n_sectors": len(search), "source": "MAST/SPOC"}
    return df, meta


def make_synthetic_light_curve(
    duration_days: float = 27.0,
    cadence_minutes: float = 2.0,
    period_days: float = 3.5,
    t0_days: float = 1.2,
    transit_duration_hours: float = 2.5,
    transit_depth_ppm: float = 2500.0,
    stellar_noise_ppm: float = 300.0,
    white_noise_ppm: float = 150.0,
    add_secondary_eclipse: bool = False,
    secondary_depth_ppm: float = 0.0,
    seed: int = 42,
):
    """
    Generate a synthetic light curve with an injected box-ish transit plus
    correlated (red) stellar noise and white photon noise, mimicking a real
    TESS SPOC light curve closely enough to validate the detection pipeline
    end-to-end before real archive access is available.

    Returns (df, truth) where truth holds the injected ground-truth
    parameters, used to score the pipeline's recovered values.
    """
    rng = np.random.default_rng(seed)

    cadence_days = cadence_minutes / (60 * 24)
    n_points = int(duration_days / cadence_days)
    time = np.linspace(0, duration_days, n_points)

    flux = np.ones(n_points)

    # Red noise: smoothly correlated systematic drift (Gaussian process-ish
    # via a random walk smoothed with a moving average), typical of real
    # spacecraft systematics / stellar variability.
    walk = np.cumsum(rng.normal(0, 1, n_points))
    window = max(int(0.5 / cadence_days), 3)  # ~12 hour smoothing window
    kernel = np.ones(window) / window
    red_noise = np.convolve(walk, kernel, mode="same")
    red_noise = red_noise / np.std(red_noise) * (stellar_noise_ppm * 1e-6)
    flux += red_noise

    # White noise: photon/read noise
    flux += rng.normal(0, white_noise_ppm * 1e-6, n_points)

    # Injected periodic transit (simple trapezoidal box transit)
    transit_duration_days = transit_duration_hours / 24
    depth = transit_depth_ppm * 1e-6
    phase = ((time - t0_days) % period_days)
    phase = np.where(phase > period_days / 2, phase - period_days, phase)
    in_transit = np.abs(phase) < (transit_duration_days / 2)
    flux[in_transit] -= depth

    # Optional shallow secondary eclipse at phase 0.5 (signature of an
    # eclipsing binary rather than a planet -- used to test the vetting logic)
    if add_secondary_eclipse and secondary_depth_ppm > 0:
        sec_phase = ((time - t0_days - period_days / 2) % period_days)
        sec_phase = np.where(sec_phase > period_days / 2, sec_phase - period_days, sec_phase)
        in_secondary = np.abs(sec_phase) < (transit_duration_days / 2)
        flux[in_secondary] -= secondary_depth_ppm * 1e-6

    flux_err = np.full(n_points, white_noise_ppm * 1e-6)

    df = pd.DataFrame({"time": time, "flux": flux, "flux_err": flux_err})
    truth = {
        "period_days": period_days,
        "t0_days": t0_days,
        "duration_hours": transit_duration_hours,
        "depth_ppm": transit_depth_ppm,
        "has_secondary_eclipse": add_secondary_eclipse,
    }
    return df, truth
