"""Build TESS view tensors for the zero-shot cross-mission evaluation.

The detrending and view generation are the same code paths used for Kepler
(``exonet.pipeline.generate_views``); only the file format, the quality
convention and the ephemeris source differ.  That is what makes it legitimate to
apply a Kepler-trained network to these inputs without any adaptation.

Usage::

    python scripts/preprocess_tess.py --workers 32 --out data/processed/tess_views.h5
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time as _time
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import pandas as pd
from astropy.io import fits

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exonet.kepler_io import split_on_gaps, transit_mask
from exonet.pipeline import DERIVED_NAMES, generate_views
from exonet.spline import _fit_segment, choose_and_apply_spline

# TESS light curves are referenced to BTJD = BJD - 2457000; the TOI table lists
# mid-transit times in full BJD.
BTJD_OFFSET = 2457000.0
TESS_CADENCE_MIN = 2.0

POSITIVE = {"CP", "KP"}
NEGATIVE = {"FP", "FA"}

DATA_ROOT = "data/tess_lc"

VIEW_SPECS = {
    "global": 2001, "local": 201, "odd": 201, "even": 201, "secondary": 201,
    "half": 201, "double": 2001, "cent_global": 2001, "cent_local": 201,
}

# Analogues of the Kepler catalogue columns that TESS actually provides.  The
# remainder are left as NaN and are imputed to the Kepler training median, so a
# model that depends on them degrades visibly rather than silently.
TESS_CATALOG_MAP = {
    "tce_period": "pl_orbper",
    "tce_duration": "pl_trandurh",
    "tce_depth": "pl_trandep",
    "tce_steff": "st_teff",
    "tce_slogg": "st_logg",
    "tce_prad": "pl_rade",
}


def read_sector(path: str) -> dict | None:
    """Read one TESS light-curve file, keeping only cadences with QUALITY == 0."""
    try:
        with fits.open(path, memmap=False) as hdul:
            d = hdul[1].data
            time = np.asarray(d["TIME"], dtype=np.float64)
            flux = np.asarray(d["PDCSAP_FLUX"], dtype=np.float64)
            qual = np.asarray(d["QUALITY"], dtype=np.int64)
            try:
                cx = np.asarray(d["MOM_CENTR1"], dtype=np.float64)
                cy = np.asarray(d["MOM_CENTR2"], dtype=np.float64)
            except KeyError:
                cx = np.full_like(time, np.nan)
                cy = np.full_like(time, np.nan)
    except Exception:
        return None
    # TESS quality bits are mission specific; requiring QUALITY == 0 is the
    # conservative choice and avoids importing a Kepler-derived bitmask.
    good = np.isfinite(time) & np.isfinite(flux) & (flux > 0) & (qual == 0)
    if good.sum() < 500:
        return None
    return {
        "time": time[good], "flux": flux[good],
        "cx": cx[good], "cy": cy[good],
    }


def process_target(payload):
    tic, rows = payload
    try:
        paths = sorted(glob.glob(os.path.join(DATA_ROOT, f"{int(tic):016d}", "*_lc.fits")))
        sectors = [s for s in (read_sector(p) for p in paths) if s is not None]
        if not sectors:
            return []

        time = np.concatenate([s["time"] for s in sectors])
        flux = np.concatenate([s["flux"] for s in sectors])
        cx = np.concatenate([s["cx"] for s in sectors])
        cy = np.concatenate([s["cy"] for s in sectors])
        off = 0
        for s in sectors:  # normalise each sector before stitching
            n = s["time"].size
            med = np.median(s["flux"])
            if med > 0:
                flux[off : off + n] /= med
            off += n
        order = np.argsort(time)
        time, flux, cx, cy = time[order], flux[order], cx[order], cy[order]

        segments = split_on_gaps(time, flux, cx, cy)
        if not segments:
            return []

        masks = []
        for seg in segments:
            m = np.zeros(seg[0].size, dtype=bool)
            for r in rows:
                m |= transit_mask(
                    seg[0], r["pl_orbper"], r["pl_tranmid"] - BTJD_OFFSET,
                    r["pl_trandurh"] / 24.0, factor=1.5,
                )
            masks.append(m)

        models, bk = choose_and_apply_spline([(s[0], s[1]) for s in segments], masks)
        T, F, CX, CY = [], [], [], []
        for seg, model, mask in zip(segments, models, masks):
            t, f, ccx, ccy = seg
            if model is None:
                continue
            good = np.isfinite(model) & (model > 0)
            if good.sum() < 100:
                continue
            cxm, _ = _fit_segment(t, ccx, bk, mask) if np.isfinite(ccx).sum() > 50 else (None, 0)
            cym, _ = _fit_segment(t, ccy, bk, mask) if np.isfinite(ccy).sum() > 50 else (None, 0)
            T.append(t[good])
            F.append(f[good] / model[good])
            CX.append((ccx - cxm if cxm is not None else np.full_like(t, np.nan))[good])
            CY.append((ccy - cym if cym is not None else np.full_like(t, np.nan))[good])
        if not T:
            return []
        t, f = np.concatenate(T), np.concatenate(F)
        cx_r, cy_r = np.concatenate(CX), np.concatenate(CY)

        records = []
        for r in rows:
            period = float(r["pl_orbper"])
            t0 = float(r["pl_tranmid"]) - BTJD_OFFSET
            dur = float(r["pl_trandurh"]) / 24.0

            keep = np.ones(t.size, dtype=bool)
            for other in rows:
                if other["toi"] == r["toi"]:
                    continue
                keep &= ~transit_mask(
                    t, other["pl_orbper"], other["pl_tranmid"] - BTJD_OFFSET,
                    other["pl_trandurh"] / 24.0, factor=1.5,
                )
            if keep.sum() < 500:
                keep = np.ones(t.size, dtype=bool)

            built = generate_views(
                t[keep], f[keep], cx_r[keep], cy_r[keep], period, t0, dur,
                bkspace=bk, cadence_minutes=TESS_CADENCE_MIN,
            )
            if built is None:
                continue
            views, derived = built
            records.append(
                {
                    "tic": int(tic), "toi": float(r["toi"]),
                    "disp": str(r["tfopwg_disp"]),
                    "label": 1 if r["tfopwg_disp"] in POSITIVE else 0,
                    "views": views,
                    "derived": [derived[k] for k in DERIVED_NAMES],
                    "tess_params": [
                        float(r.get(v, np.nan)) for v in TESS_CATALOG_MAP.values()
                    ],
                }
            )
        return records
    except Exception as exc:
        return [{"error": f"{tic}: {type(exc).__name__}: {exc}"}]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--toi-table", default="data/catalogs/toi.csv")
    ap.add_argument("--out", default="data/processed/tess_views.h5")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    toi = pd.read_csv(args.toi_table, low_memory=False)
    keep = toi["tfopwg_disp"].isin(POSITIVE | NEGATIVE)
    keep &= toi[["pl_tranmid", "pl_orbper", "pl_trandurh"]].notna().all(axis=1)
    toi = toi[keep]

    grouped = [(int(tic), g.to_dict("records")) for tic, g in toi.groupby("tid")]
    grouped.sort(key=lambda x: x[0])
    if args.limit:
        grouped = grouped[: args.limit]
    print(f"processing {len(toi):,} TOIs over {len(grouped):,} TIC targets", flush=True)

    records, errors = [], []
    t0 = _time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        for i, res in enumerate(pool.map(process_target, grouped, chunksize=2), 1):
            for r in res:
                (errors if "error" in r else records).append(r)
            if i % 100 == 0:
                el = _time.time() - t0
                print(
                    f"[{i}/{len(grouped)}] {len(records):,} TOIs {el/60:.1f} min "
                    f"ETA {(len(grouped)-i)/(i/el)/60:.1f} min",
                    flush=True,
                )

    print(f"kept {len(records):,} TOIs; {len(errors)} target failures", flush=True)
    for e in errors[:10]:
        print("  ", e["error"], flush=True)
    if not records:
        return 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with h5py.File(args.out, "w") as fh:
        for name in VIEW_SPECS:
            fh.create_dataset(
                name, data=np.stack([r["views"][name] for r in records]).astype(np.float32),
                compression="lzf",
            )
        fh.create_dataset(
            "derived", data=np.array([r["derived"] for r in records], dtype=np.float32)
        )
        fh.create_dataset(
            "tess_params",
            data=np.array([r["tess_params"] for r in records], dtype=np.float32),
        )
        fh.create_dataset("tic", data=np.array([r["tic"] for r in records]))
        fh.create_dataset("toi", data=np.array([r["toi"] for r in records]))
        fh.create_dataset("label", data=np.array([r["label"] for r in records]))
        fh.create_dataset(
            "disp", data=np.array([r["disp"] for r in records], dtype=h5py.string_dtype())
        )
        fh.attrs["derived_cols"] = DERIVED_NAMES
        fh.attrs["tess_param_cols"] = list(TESS_CATALOG_MAP.keys())
    n_pos = sum(r["label"] for r in records)
    print(
        f"wrote {args.out}: {len(records):,} TOIs "
        f"({n_pos} planets / {len(records)-n_pos} false positives)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
