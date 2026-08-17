"""Reading and stitching Kepler long-cadence light curves.

The functions here turn the per-quarter FITS products written by the Kepler
pipeline into the contiguous, quality-masked time series that the detrending and
view-generation stages consume.
"""

from __future__ import annotations

import os

import numpy as np
from astropy.io import fits

# Long-cadence quarter timestamps (Q0-Q17); see scripts/download_kepler.py.
LC_QUARTER_STAMPS = [
    "2009131105131", "2009166043257", "2009259160929", "2009350155506",
    "2010078095331", "2010174085026", "2010265121752", "2010355172524",
    "2011073133259", "2011177032512", "2011271113734", "2012004120508",
    "2012088054726", "2012179063303", "2012277125453", "2013011073258",
    "2013098041711", "2013131215648",
]

# SAP_QUALITY bits that indicate a cadence should not be trusted for transit
# work.  This follows the "default" bitmask used by the Kepler DV pipeline and
# by lightkurve: attitude tweaks, safe modes, coarse point, Earth point, desat
# events, manual exclusions, and detector anomalies.
DEFAULT_QUALITY_BITMASK = (
    1 + 2 + 4 + 8 + 16 + 32 + 128 + 256 + 1024 + 2048 + 4096 + 16384 + 32768
    + 65536 + 1048576 + 2097152 + 4194304
)


def target_dir(data_root: str, kepid: int) -> str:
    kic = f"{int(kepid):09d}"
    return os.path.join(data_root, kic[:4], kic)


def quarter_files(data_root: str, kepid: int) -> list[str]:
    """Return the light-curve files present on disk for a target, in time order."""
    kic = f"{int(kepid):09d}"
    d = target_dir(data_root, kepid)
    out = []
    for stamp in LC_QUARTER_STAMPS:
        path = os.path.join(d, f"kplr{kic}-{stamp}_llc.fits")
        if os.path.exists(path):
            out.append(path)
    return out


def read_quarter(path: str, bitmask: int = DEFAULT_QUALITY_BITMASK) -> dict | None:
    """Read one quarter.  Returns ``None`` if the file yields no usable cadences.

    Flux is PDCSAP (Presearch Data Conditioning), which has had common-mode
    instrumental systematics removed by the Kepler pipeline.  Centroids are the
    flux-weighted moment centroids, used later for the centroid-shift branch.
    """
    try:
        with fits.open(path, memmap=False) as hdul:
            hdr = hdul[0].header
            d = hdul[1].data
            time = np.asarray(d["TIME"], dtype=np.float64)
            flux = np.asarray(d["PDCSAP_FLUX"], dtype=np.float64)
            qual = np.asarray(d["SAP_QUALITY"], dtype=np.int64)
            try:
                cx = np.asarray(d["MOM_CENTR1"], dtype=np.float64)
                cy = np.asarray(d["MOM_CENTR2"], dtype=np.float64)
            except KeyError:
                cx = np.full_like(time, np.nan)
                cy = np.full_like(time, np.nan)
            quarter = hdr.get("QUARTER", -1)
    except Exception:
        return None

    good = (
        np.isfinite(time)
        & np.isfinite(flux)
        & (flux > 0)
        & ((qual & bitmask) == 0)
    )
    if good.sum() < 100:
        return None

    return {
        "time": time[good],
        "flux": flux[good],
        "cx": cx[good],
        "cy": cy[good],
        "quarter": quarter,
    }


def read_target(data_root: str, kepid: int) -> list[dict]:
    """Read every available quarter for a target, sorted by time."""
    out = []
    for path in quarter_files(data_root, kepid):
        q = read_quarter(path)
        if q is not None:
            out.append(q)
    out.sort(key=lambda q: q["time"][0])
    return out


def split_on_gaps(time: np.ndarray, *arrays: np.ndarray, gap: float = 0.75):
    """Split time series into contiguous segments separated by gaps > ``gap`` days.

    Detrending is done per segment because a spline fitted across a multi-day
    data gap is unconstrained in the gap and rings badly at the edges.
    """
    if time.size == 0:
        return []
    breaks = np.where(np.diff(time) > gap)[0] + 1
    idx = np.split(np.arange(time.size), breaks)
    segments = []
    for ix in idx:
        if ix.size < 4:  # too short to fit a cubic spline
            continue
        segments.append((time[ix],) + tuple(a[ix] for a in arrays))
    return segments


def transit_mask(
    time: np.ndarray, period: float, t0: float, duration: float, factor: float = 1.0
) -> np.ndarray:
    """Boolean mask that is True for cadences within ``factor`` durations of transit.

    ``duration`` is the full transit duration in days; the mask spans
    ``+/- factor * duration / 2`` around each predicted mid-transit.
    """
    if not np.isfinite(period) or period <= 0 or not np.isfinite(duration):
        return np.zeros_like(time, dtype=bool)
    half = 0.5 * factor * duration
    phase = np.abs((time - t0 + 0.5 * period) % period - 0.5 * period)
    return phase <= half
