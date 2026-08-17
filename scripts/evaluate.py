"""Aggregate all runs into the paper's tables: comparisons, ablations, calibration,
conformal guarantees and transit-parameter recovery.

Reads every ``results/runs/*.json`` plus the matching prediction archives, and
writes ``results/summary.json`` together with LaTeX tables under
``results/tables/``.

Usage::

    python scripts/evaluate.py --data data/processed/dr24_views.h5
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exonet.conformal import (
    apply_temperature, empirical_fdr, expected_calibration_error,
    fdr_controlled_selection, fit_temperature,
)
from exonet.data import load_h5
from scripts.train import metrics

# Grid spacing of the local view, in units of the catalogue transit duration.
LOCAL_DURATIONS = 4.0


def load_runs(run_dir: str) -> list[dict]:
    runs = []
    for path in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
        if os.path.basename(path).startswith("baselines_"):
            continue
        with open(path) as fh:
            rec = json.load(fh)
        rec["_json"] = path
        rec["_preds"] = path.replace(".json", "_preds.npz")
        runs.append(rec)
    return runs


def aggregate_by_tag(runs: list[dict]) -> pd.DataFrame:
    """Mean and standard deviation of each metric across seeds, per configuration."""
    keys = [
        "auc", "ap", "accuracy", "precision", "recall",
        "precision_at_recall_90", "precision_at_recall_95", "precision_at_recall_99",
    ]
    buckets = defaultdict(list)
    for r in runs:
        buckets[(r["tag"], r["split_mode"])].append(r)
    rows = []
    for (tag, split), group in sorted(buckets.items()):
        row = {
            "tag": tag, "split": split, "model": group[0]["model"],
            "n_seeds": len(group), "n_params_M": group[0]["n_params"] / 1e6,
        }
        for k in keys:
            vals = np.array([g["test_metrics"][k] for g in group], dtype=float)
            row[f"{k}_mean"] = vals.mean()
            row[f"{k}_std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def ensemble_scores(runs: list[dict], tag: str, split: str):
    """Average the per-seed probabilities for one configuration.

    Every seed shares the same split indices, so scores can be averaged directly.
    The spread across seeds also provides the uncertainty used for the transit
    parameters, which are not otherwise supervised.
    """
    group = [r for r in runs if r["tag"] == tag and r["split_mode"] == split]
    if not group:
        return None
    test_s, val_s, params = [], [], []
    ref = None
    for r in group:
        z = np.load(r["_preds"])
        if ref is None:
            ref = {
                "test_idx": z["test_idx"], "test_label": z["test_label"],
                "val_idx": z["val_idx"], "val_label": z["val_label"],
                "kepid": z["kepid"], "plnt": z["plnt"],
            }
        elif not np.array_equal(z["test_idx"], ref["test_idx"]):
            continue  # different split; cannot be pooled
        test_s.append(z["test_score"])
        val_s.append(z["val_score"])
        if z["test_params"].size:
            params.append(z["test_params"])
    if not test_s:
        return None
    out = dict(ref)
    out["test_score"] = np.mean(test_s, axis=0)
    out["val_score"] = np.mean(val_s, axis=0)
    out["n_members"] = len(test_s)
    if params:
        out["params_mean"] = np.mean(params, axis=0)
        out["params_std"] = np.std(params, axis=0)
    return out


def calibration_report(ens: dict) -> dict:
    """Expected calibration error before and after temperature scaling."""
    t = fit_temperature(ens["val_label"], ens["val_score"])
    raw = expected_calibration_error(ens["test_label"], ens["test_score"])
    cal = expected_calibration_error(
        ens["test_label"], apply_temperature(ens["test_score"], t)
    )
    return {"temperature": t, "ece_raw": raw, "ece_temperature_scaled": cal}


def conformal_report(ens: dict, levels=(0.01, 0.02, 0.05, 0.10, 0.20)) -> list[dict]:
    """Nominal versus realised false discovery rate of the conformal selection.

    Calibration uses only the *negatives* of the validation split, which the
    model never trained on, so the exchangeability requirement holds.
    """
    cal_neg = ens["val_score"][ens["val_label"] == 0]
    y = ens["test_label"]
    rows = []
    for q in levels:
        sel, _ = fdr_controlled_selection(cal_neg, ens["test_score"], q)
        n_sel = int(sel.sum())
        rows.append(
            {
                "q_nominal": q,
                "n_selected": n_sel,
                "empirical_fdr": empirical_fdr(sel, y),
                "recall": float((sel & (y == 1)).sum() / max((y == 1).sum(), 1)),
            }
        )
    return rows


def parameter_recovery(ens: dict, blob: dict, catalogs_dir: str) -> dict:
    """Compare recovered transit parameters with independent literature values.

    The reference values come from the NASA Exoplanet Archive's composite
    planetary-parameter table, i.e. from published follow-up analyses, not from
    the Kepler pipeline whose fit is being assessed.  The comparison is therefore
    between two independent estimates of the same physical quantity: the DV
    catalogue value that seeded the search, and PHANTOM's decoder bottleneck.
    """
    if "params_mean" not in ens:
        return {"available": False, "reason": "no decoder parameters in ensemble"}

    koi = pd.read_csv(os.path.join(catalogs_dir, "koi_cumulative.csv"), low_memory=False)
    ps = pd.read_csv(os.path.join(catalogs_dir, "confirmed_pscomppars.csv"), low_memory=False)

    # literature parameters keyed by KIC via the KOI cross-identification
    koi_named = koi.dropna(subset=["kepler_name"])[["kepid", "kepler_name", "koi_period"]]
    lit = ps.merge(koi_named, left_on="pl_name", right_on="kepler_name", how="inner")
    lit = lit.dropna(subset=["pl_orbper"])

    cat_cols, der_cols = blob["cat_cols"], blob["der_cols"]
    idx = ens["test_idx"]
    tce_period = blob["catalog"][idx, cat_cols.index("tce_period")]
    tce_dur = blob["catalog"][idx, cat_cols.index("tce_duration")]      # hours
    tce_depth = blob["catalog"][idx, cat_cols.index("tce_depth")]       # ppm
    local_scale = blob["derived"][idx, der_cols.index("local_scale")]
    kepid = ens["kepid"]

    p = ens["params_mean"]
    p_std = ens["params_std"]
    # depth: the decoder works on a view normalised by |min|, so multiplying the
    # fitted depth by that scale restores parts per million.
    depth_est = p[:, 0] * local_scale * 1e6
    depth_err = p_std[:, 0] * local_scale * 1e6
    # duration: the local grid is expressed in units of the catalogue duration
    # and the model spans z <= 1, i.e. a full width of 2 * half_dur grid units.
    dur_est = 2.0 * p[:, 1] * tce_dur
    dur_err = 2.0 * p_std[:, 1] * tce_dur

    # match each test TCE to a literature planet on the same star with a
    # consistent orbital period
    by_kic = defaultdict(list)
    for _, r in lit.iterrows():
        by_kic[int(r["kepid"])].append(r)

    rows = []
    for j in range(len(idx)):
        for r in by_kic.get(int(kepid[j]), []):
            if not np.isfinite(tce_period[j]) or r["pl_orbper"] <= 0:
                continue
            if abs(tce_period[j] - r["pl_orbper"]) / r["pl_orbper"] > 0.01:
                continue
            rows.append(
                {
                    "kepid": int(kepid[j]), "pl_name": r["pl_name"],
                    "lit_duration": r.get("pl_trandur", np.nan),
                    "lit_depth_ppm": r.get("pl_trandep", np.nan) * 1e4,
                    "cat_duration": tce_dur[j], "cat_depth_ppm": tce_depth[j],
                    "est_duration": dur_est[j], "est_duration_err": dur_err[j],
                    "est_depth_ppm": depth_est[j], "est_depth_err": depth_err[j],
                }
            )
            break
    df = pd.DataFrame(rows)
    if df.empty:
        return {"available": False, "reason": "no literature matches in the test split"}

    def stats(truth, pred):
        m = np.isfinite(truth) & np.isfinite(pred) & (truth > 0)
        if m.sum() < 5:
            return None
        rel = np.abs(pred[m] - truth[m]) / truth[m]
        return {
            "n": int(m.sum()),
            "median_abs_rel_error": float(np.median(rel)),
            "mean_abs_rel_error": float(np.mean(rel)),
            "frac_within_10pct": float(np.mean(rel < 0.10)),
            "frac_within_20pct": float(np.mean(rel < 0.20)),
        }

    return {
        "available": True,
        "n_matched": len(df),
        "duration": {
            "catalog": stats(df["lit_duration"].values, df["cat_duration"].values),
            "phantom": stats(df["lit_duration"].values, df["est_duration"].values),
        },
        "depth": {
            "catalog": stats(df["lit_depth_ppm"].values, df["cat_depth_ppm"].values),
            "phantom": stats(df["lit_depth_ppm"].values, df["est_depth_ppm"].values),
        },
        "_table": df,
    }


def latex_table(df: pd.DataFrame, caption: str, label: str, cols: list[tuple[str, str]]) -> str:
    """Render a comparison table as booktabs LaTeX."""
    lines = [
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\begin{tabular}{l" + "c" * (len(cols) - 1) + "}", r"\toprule",
        " & ".join(h for _, h in cols) + r" \\", r"\midrule",
    ]
    for _, r in df.iterrows():
        cells = []
        for key, _ in cols:
            v = r[key]
            if isinstance(v, str):
                cells.append(v.replace("_", r"\_"))
            elif key.endswith("_mean") and f"{key[:-5]}_std" in r:
                cells.append(rf"${v:.4f} \pm {r[key[:-5]+'_std']:.4f}$")
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{v}")
            else:
                cells.append(f"{v:.4f}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/processed/dr24_views.h5")
    ap.add_argument("--run-dir", default="results/runs")
    ap.add_argument("--catalogs", default="data/catalogs")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--headline-tag", default="phantom_full")
    ap.add_argument("--split", default="group")
    args = ap.parse_args()

    runs = load_runs(args.run_dir)
    if not runs:
        print("no runs found", file=sys.stderr)
        return 1
    print(f"loaded {len(runs)} runs", flush=True)

    summary: dict = {"n_runs": len(runs)}
    table = aggregate_by_tag(runs)
    summary["per_config"] = json.loads(table.to_json(orient="records"))

    # baselines share the split, so fold them into the same comparison
    base_rows = []
    for path in sorted(glob.glob(os.path.join(args.run_dir, "baselines_*.json"))):
        with open(path) as fh:
            rec = json.load(fh)
        for name, m in rec["results"].items():
            if name.endswith("_importance"):
                continue
            base_rows.append({"tag": name, "split": rec["split_mode"], **m})
    if base_rows:
        bdf = pd.DataFrame(base_rows)
        agg = bdf.groupby(["tag", "split"]).agg(["mean", "std"]).reset_index()
        agg.columns = [
            c[0] if c[1] == "" else f"{c[0]}_{c[1]}" for c in agg.columns.to_flat_index()
        ]
        agg = agg.fillna(0.0)
        agg["model"] = "baseline"
        agg["n_seeds"] = bdf.groupby(["tag", "split"]).size().values
        table = pd.concat([table, agg], ignore_index=True)
    summary["comparison"] = json.loads(table.to_json(orient="records"))

    blob = load_h5(args.data)
    ens = ensemble_scores(runs, args.headline_tag, args.split)
    if ens is not None:
        summary["ensemble"] = {
            "tag": args.headline_tag, "n_members": ens["n_members"],
            "metrics": metrics(ens["test_label"], ens["test_score"]),
            "calibration": calibration_report(ens),
            "conformal": conformal_report(ens),
        }
        rec = parameter_recovery(ens, blob, args.catalogs)
        tbl = rec.pop("_table", None)
        summary["parameter_recovery"] = rec
        if tbl is not None:
            os.makedirs(args.out_dir, exist_ok=True)
            tbl.to_csv(os.path.join(args.out_dir, "parameter_recovery.csv"), index=False)

    os.makedirs(os.path.join(args.out_dir, "tables"), exist_ok=True)
    table.sort_values("ap_mean", ascending=False).to_csv(
        os.path.join(args.out_dir, "comparison.csv"), index=False
    )
    cols = [
        ("tag", "Model"), ("auc_mean", "AUC"), ("ap_mean", "AP"),
        ("precision_at_recall_90_mean", r"P@R=0.90"),
        ("precision_at_recall_95_mean", r"P@R=0.95"),
    ]
    with open(os.path.join(args.out_dir, "tables", "comparison.tex"), "w") as fh:
        fh.write(
            latex_table(
                table[table.split == args.split].sort_values("ap_mean", ascending=False),
                "Test-set performance on the Kepler DR24 benchmark.",
                "tab:comparison", cols,
            )
        )

    with open(os.path.join(args.out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(json.dumps(summary.get("ensemble", {}), indent=2, default=float)[:2000])
    print(f"wrote {args.out_dir}/summary.json and tables/", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
