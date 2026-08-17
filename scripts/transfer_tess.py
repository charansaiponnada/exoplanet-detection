"""Apply Kepler-trained networks to TESS candidates with no adaptation.

This is a strict zero-shot test: no TESS example is seen in training, no
fine-tuning is performed, and the scalar features are standardised with the
statistics of the *Kepler* training split. Only the eleven light-curve-derived
scalars are used, because they are the only features computable identically for
both missions; the checkpoints evaluated here are therefore the
``phantom_derived_only`` configuration and the view-only AstroNet baseline.

Usage::

    python scripts/transfer_tess.py --tag phantom_derived_only
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exonet.data import VIEW_CLIP_MAX, ScalarConditioner, load_h5, make_splits
from exonet.model import PHANTOM, VIEW_TABLE, AstroNet
from scripts.train import build_scalars, metrics


def kepler_conditioner(kepler_h5: str, groups: list[str], seed: int, split_mode: str):
    """Refit the scalar conditioner on the exact Kepler training split.

    The conditioner is a deterministic function of the training split, so it can
    be reconstructed rather than serialised, which keeps the training runs and
    this script independent.
    """
    blob = load_h5(kepler_h5)
    raw, names = build_scalars(blob, groups)
    tr, _, _ = make_splits(blob["kepid"], len(blob["kepid"]), mode=split_mode, seed=seed)
    return ScalarConditioner(names).fit(raw[tr]), names


def load_tess(path: str, view_names: list[str], names: list[str]):
    """Read the TESS views and assemble the scalar matrix in the Kepler column order."""
    with h5py.File(path, "r") as fh:
        views = {
            n: np.clip(fh[n][:].astype(np.float32), -1.0, VIEW_CLIP_MAX)
            for n in view_names
        }
        derived = fh["derived"][:].astype(np.float32)
        der_cols = list(fh.attrs["derived_cols"])
        label = fh["label"][:].astype(np.float32)
        tic = fh["tic"][:]
        toi = fh["toi"][:]
        disp = np.array([s.decode() for s in fh["disp"][:]])

    cols = []
    for n in names:
        if n in der_cols:
            cols.append(derived[:, der_cols.index(n)])
        else:  # not computable for TESS; imputed to the Kepler training median
            cols.append(np.full(len(label), np.nan, dtype=np.float32))
    scalars = np.stack(cols, axis=1).astype(np.float32) if cols else np.zeros((len(label), 1), np.float32)
    return views, scalars, label, tic, toi, disp


@torch.no_grad()
def score_checkpoint(rec: dict, views, scalars, device) -> np.ndarray:
    """Rebuild a model from its run record, load its weights and score the set."""
    view_names = rec["views"]
    if rec["model"] == "phantom":
        model = PHANTOM(
            n_scalars=rec["n_scalars"], view_names=view_names,
            use_decoder=rec["use_decoder"], use_harmonic=rec["use_harmonic"],
        )
    else:
        model = AstroNet(n_scalars=rec["n_scalars"])
    state = torch.load(rec["_json"].replace(".json", ".pt"), map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device).eval()

    out = []
    n = len(scalars)
    for i in range(0, n, 256):
        v = {k: torch.from_numpy(views[k][i : i + 256]).to(device) for k in view_names}
        s = torch.from_numpy(scalars[i : i + 256]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            o = model(v, s)
        out.append(torch.sigmoid(o["logit"].float()).cpu().numpy())
    return np.concatenate(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kepler-data", default="data/processed/dr24_views.h5")
    ap.add_argument("--tess-data", default="data/processed/tess_views.h5")
    ap.add_argument("--run-dir", default="results/runs")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--split-mode", default="group")
    ap.add_argument("--tag", default="phantom_derived_only")
    ap.add_argument("--baseline-tag", default="astronet")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {"models": {}}
    scores = {}

    for label, tag in (("phantom", args.tag), ("astronet", args.baseline_tag)):
        recs = []
        for path in sorted(glob.glob(os.path.join(args.run_dir, f"{tag}_seed*_{args.split_mode}.json"))):
            with open(path) as fh:
                r = json.load(fh)
            r["_json"] = path
            if os.path.exists(path.replace(".json", ".pt")):
                recs.append(r)
        if not recs:
            print(f"no checkpoints for {tag}; skipping", flush=True)
            continue

        groups = recs[0]["scalar_groups"]
        cond, names = kepler_conditioner(
            args.kepler_data, groups, recs[0]["seed"], args.split_mode
        )
        views, raw, y, tic, toi, disp = load_tess(args.tess_data, recs[0]["views"], names)
        scal = cond.transform(raw)

        member = [score_checkpoint(r, views, scal, device) for r in recs]
        s = np.mean(member, axis=0)
        scores[f"{label}_score"] = s
        results["models"][label] = {
            "tag": tag, "n_members": len(recs), "scalar_groups": groups, **metrics(y, s)
        }
        print(f"{label} ({tag}, {len(recs)} seeds): "
              f"AUC {results['models'][label]['auc']:.4f} "
              f"AP {results['models'][label]['ap']:.4f}", flush=True)

    if not scores:
        print("nothing to evaluate", file=sys.stderr)
        return 1

    _, _, y, tic, toi, disp = load_tess(args.tess_data, ["global", "local"], [])
    results["n_events"] = int(len(y))
    results["n_planets"] = int(y.sum())
    results["chance_rate"] = float(y.mean())

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "tess_transfer.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    np.savez_compressed(
        os.path.join(args.out_dir, "tess_predictions.npz"),
        label=y, tic=tic, toi=toi, disp=disp, **scores,
    )
    print(json.dumps(results, indent=2, default=float), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
