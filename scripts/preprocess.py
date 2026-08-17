"""Turn raw Kepler FITS light curves into the fixed-length view tensors used for training.

For every TCE in the DR24 training table this script

1. reads and quality-masks all available quarters of the host target,
2. detrends flux (and the centroid time series) with a BIC-selected B-spline,
   masking the in-transit cadences of *every* TCE on the target so the spline
   does not absorb the signals,
3. removes the in-transit cadences belonging to the *other* TCEs on the target,
   which would otherwise contaminate the fold, and
4. renders the global/local/odd/even/secondary/harmonic/centroid views.

Output is a single HDF5 file with one row per TCE.

Usage::

    python scripts/preprocess.py --workers 32 --out data/processed/dr24_views.h5
"""

from __future__ import annotations

import argparse
import os
import sys
import time as _time
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exonet import views as V
from exonet.kepler_io import read_target, split_on_gaps, transit_mask
from exonet.pipeline import DERIVED_NAMES, generate_views
from exonet.spline import _fit_segment, choose_and_apply_spline

# Columns copied verbatim from the DR24 TCE table.  Grouped so that ablations can
# switch whole groups on and off.
TRANSIT_COLS = [
    "tce_period", "tce_duration", "tce_depth", "tce_model_snr",
    "tce_max_mult_ev", "tce_num_transits", "tce_impact", "tce_ror", "tce_dor",
]
STELLAR_COLS = ["tce_steff", "tce_slogg", "tce_smet", "tce_prad"]
DV_DIAG_COLS = [
    "tce_bin_oedp_stat", "tce_dikco_msky", "tce_dicco_msky", "tce_maxmesd",
    "tce_robstat",
]
CATALOG_COLS = TRANSIT_COLS + STELLAR_COLS + DV_DIAG_COLS

VIEW_SPECS = {
    "global": V.GLOBAL_BINS,
    "local": V.LOCAL_BINS,
    "odd": V.LOCAL_BINS,
    "even": V.LOCAL_BINS,
    "secondary": V.LOCAL_BINS,
    "half": V.LOCAL_BINS,
    "double": V.GLOBAL_BINS,
    "cent_global": V.GLOBAL_BINS,
    "cent_local": V.LOCAL_BINS,
}

DATA_ROOT = "data/kepler"


def _detrend(segments, masks):
    """Detrend flux and centroids on a shared knot spacing.

    Returns ``(time, flux, cx, cy, bkspace)`` concatenated over usable segments.
    """
    flux_segments = [(s[0], s[1]) for s in segments]
    models, bk = choose_and_apply_spline(flux_segments, masks)

    T, F, CX, CY = [], [], [], []
    for seg, model, mask in zip(segments, models, masks):
        t, f, cx, cy = seg
        if model is None:
            continue
        good = np.isfinite(model) & (model > 0)
        if good.sum() < 8:
            continue
        # Centroids carry their own slow drift; remove it with the same knot
        # spacing so that only event-coincident excursions survive.
        cxm, _ = _fit_segment(t, cx, bk, mask) if np.isfinite(cx).sum() > 20 else (None, 0)
        cym, _ = _fit_segment(t, cy, bk, mask) if np.isfinite(cy).sum() > 20 else (None, 0)
        cx_r = cx - cxm if cxm is not None else np.full_like(t, np.nan)
        cy_r = cy - cym if cym is not None else np.full_like(t, np.nan)

        T.append(t[good])
        F.append(f[good] / model[good])
        CX.append(cx_r[good])
        CY.append(cy_r[good])

    if not T:
        return None
    return (
        np.concatenate(T), np.concatenate(F), np.concatenate(CX), np.concatenate(CY), bk
    )


def process_target(payload):
    """Worker: process one target and return one record per TCE on it."""
    kepid, tce_rows = payload
    try:
        quarters = read_target(DATA_ROOT, kepid)
        if len(quarters) < 2:
            return []

        time = np.concatenate([q["time"] for q in quarters])
        flux = np.concatenate([q["flux"] for q in quarters])
        cx = np.concatenate([q["cx"] for q in quarters])
        cy = np.concatenate([q["cy"] for q in quarters])
        # Normalise each quarter's flux level before stitching so quarter-to-quarter
        # aperture changes do not create steps the spline must chase.
        offset = 0
        for q in quarters:
            n = q["time"].size
            med = np.median(q["flux"])
            if med > 0:
                flux[offset : offset + n] /= med
            offset += n
        order = np.argsort(time)
        time, flux, cx, cy = time[order], flux[order], cx[order], cy[order]

        segments = split_on_gaps(time, flux, cx, cy)
        if not segments:
            return []

        # Mask in-transit cadences of every TCE on this target for spline fitting.
        masks = []
        for seg in segments:
            m = np.zeros(seg[0].size, dtype=bool)
            for r in tce_rows:
                m |= transit_mask(
                    seg[0], r["tce_period"], r["tce_time0bk"],
                    r["tce_duration"] / 24.0, factor=1.5,
                )
            masks.append(m)

        det = _detrend(segments, masks)
        if det is None:
            return []
        t, f, cx_r, cy_r, bk = det
        if t.size < 500:
            return []

        records = []
        for r in tce_rows:
            period = float(r["tce_period"])
            t0 = float(r["tce_time0bk"])
            dur = float(r["tce_duration"]) / 24.0  # hours -> days
            if not (np.isfinite(period) and period > 0 and np.isfinite(dur) and dur > 0):
                continue

            # Drop cadences in transit of the *other* TCEs on this target.
            keep = np.ones(t.size, dtype=bool)
            for other in tce_rows:
                if other["tce_plnt_num"] == r["tce_plnt_num"]:
                    continue
                keep &= ~transit_mask(
                    t, other["tce_period"], other["tce_time0bk"],
                    other["tce_duration"] / 24.0, factor=1.5,
                )
            if keep.sum() < 500:
                keep = np.ones(t.size, dtype=bool)
            tt, ff = t[keep], f[keep]
            ccx, ccy = cx_r[keep], cy_r[keep]

            built = generate_views(tt, ff, ccx, ccy, period, t0, dur, bkspace=bk)
            if built is None:
                continue
            out, derived = built

            records.append(
                {
                    "kepid": int(kepid),
                    "tce_plnt_num": int(r["tce_plnt_num"]),
                    "label": r["av_training_set"],
                    "views": out,
                    "catalog": [float(r.get(c, np.nan)) for c in CATALOG_COLS],
                    "derived": [derived[k] for k in DERIVED_NAMES],
                }
            )
        return records
    except Exception as exc:  # a single bad target must not kill the run
        return [{"error": f"{kepid}: {type(exc).__name__}: {exc}"}]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tce-table", default="data/catalogs/dr24_tce.csv")
    ap.add_argument("--out", default="data/processed/dr24_views.h5")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tces = pd.read_csv(args.tce_table, low_memory=False)
    tces = tces[tces["av_training_set"].isin(["PC", "AFP", "NTP"])].copy()

    grouped = []
    for kepid, grp in tces.groupby("kepid"):
        grouped.append((int(kepid), grp.to_dict("records")))
    grouped.sort(key=lambda x: x[0])
    if args.limit:
        grouped = grouped[: args.limit]
    print(f"processing {len(tces):,} TCEs over {len(grouped):,} targets", flush=True)

    records, errors = [], []
    t0 = _time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        for i, res in enumerate(pool.map(process_target, grouped, chunksize=4), 1):
            for r in res:
                (errors if "error" in r else records).append(r)
            if i % 250 == 0:
                el = _time.time() - t0
                print(
                    f"[{i}/{len(grouped)}] {len(records):,} TCEs  "
                    f"{el/60:.1f} min  ETA {(len(grouped)-i)/(i/el)/60:.1f} min",
                    flush=True,
                )

    print(f"kept {len(records):,} TCEs; {len(errors)} target failures", flush=True)
    for e in errors[:10]:
        print("  ", e["error"], flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n = len(records)
    with h5py.File(args.out, "w") as fh:
        for name, nbins in VIEW_SPECS.items():
            arr = np.stack([r["views"][name] for r in records]).astype(np.float32)
            fh.create_dataset(name, data=arr, compression="lzf")
        fh.create_dataset(
            "catalog", data=np.array([r["catalog"] for r in records], dtype=np.float32)
        )
        fh.create_dataset(
            "derived", data=np.array([r["derived"] for r in records], dtype=np.float32)
        )
        fh.create_dataset("kepid", data=np.array([r["kepid"] for r in records]))
        fh.create_dataset(
            "tce_plnt_num", data=np.array([r["tce_plnt_num"] for r in records])
        )
        fh.create_dataset(
            "label",
            data=np.array([r["label"] for r in records], dtype=h5py.string_dtype()),
        )
        fh.attrs["catalog_cols"] = CATALOG_COLS
        fh.attrs["derived_cols"] = DERIVED_NAMES
        fh.attrs["transit_cols"] = TRANSIT_COLS
        fh.attrs["stellar_cols"] = STELLAR_COLS
        fh.attrs["dv_diag_cols"] = DV_DIAG_COLS
    print(f"wrote {args.out} ({n:,} rows, {os.path.getsize(args.out)/1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
