"""Analyses added in revision, all reading the artefacts written by train.py.

Each block answers a question the first-round review raised:

  * ``rp99``      recall at 99 per cent precision, the operating point at which
                  ExoMiner reports its published number, so that the gap to the
                  strongest published vetter is quantified rather than asserted.
  * ``nested``    the cumulative cost of a fixed-order chain of deletions.
  * ``perclass``  performance split by the two physically distinct negative
                  classes, astrophysical false positive and non-transiting
                  phenomenon.
  * ``power``     the minimum effect the five-seed protocol can resolve.
  * ``conformal`` uniformity of the conformal p-values on test negatives, which
                  is what validity predicts and is directly checkable.
  * ``tess``      cross-mission performance stratified by period and recomputed
                  after recalibration, separating a calibration collapse from a
                  representation failure.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exonet.conformal import (
    apply_temperature, conformal_pvalues, expected_calibration_error, fit_temperature,
)

# Ordered chain of successive deletions. The tags for the first four links come
# from the leave-one-out sweep, which happens to contain a nested subsequence.
NESTED_CHAIN = [
    ("phantom_full", "full model", "9 views, all scalars"),
    ("phantom_no_dv", r"$-$ DV diagnostics", "9 views, no DV scalars"),
    ("phantom_derived_only", r"$-$ catalogue scalars", "9 views, derived scalars only"),
    ("phantom_no_scalars", r"$-$ derived scalars", "9 views, no scalars"),
    ("nest_n4_no_centroid", r"$-$ centroid views", "7 views, no scalars"),
    ("nest_n5_no_harmonic", r"$-$ harmonic views", "5 views, no scalars"),
    ("nest_n6_no_secondary", r"$-$ secondary view", "4 views, no scalars"),
    ("nest_n7_sv_only", r"$-$ odd/even views", "2 views, no scalars"),
]


def load_runs(run_dir: str, split: str = "group") -> dict[str, list[dict]]:
    """Index run records by configuration tag.

    ``baselines_*.json`` is written by scripts/baselines.py with a different
    schema and holds two models per file; it is keyed here under the score array
    each one writes so that the non-neural baselines join the same comparisons.
    """
    by_tag: dict[str, list[dict]] = {}
    for path in sorted(glob.glob(os.path.join(run_dir, f"*_{split}.json"))):
        with open(path) as fh:
            rec = json.load(fh)
        rec["_preds"] = path.replace(".json", "_preds.npz")
        if "tag" in rec:
            by_tag.setdefault(rec["tag"], []).append(rec)
        elif "results" in rec:
            for name in ("gbdt", "classical"):
                by_tag.setdefault(name, []).append(dict(rec, _score_key=f"{name}_score"))
    return by_tag


def recall_at_precision(y: np.ndarray, s: np.ndarray, target: float) -> float:
    """Highest recall attainable at or above a target precision."""
    prec, rec, _ = precision_recall_curve(y, s)
    ok = prec >= target
    return float(rec[ok].max()) if ok.any() else 0.0


def precision_at_recall(y: np.ndarray, s: np.ndarray, target: float) -> float:
    prec, rec, _ = precision_recall_curve(y, s)
    ok = rec >= target
    return float(prec[ok].max()) if ok.any() else 0.0


def per_seed(runs: list[dict], fn) -> np.ndarray:
    """Apply a metric to every seed of a configuration and return the vector."""
    out = []
    for r in runs:
        z = np.load(r["_preds"])
        out.append(fn(z["test_label"], z[r.get("_score_key", "test_score")]))
    return np.asarray(out, dtype=float)


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


# ----------------------------------------------------------------- comment 2
def block_rp99(by_tag: dict, out: dict) -> None:
    """Recall at 99 per cent precision, the point ExoMiner publishes."""
    rows = []
    for tag in ("classical", "astronet", "gbdt", "phantom_full"):
        if tag not in by_tag:
            continue
        v = per_seed(by_tag[tag], lambda y, s: recall_at_precision(y, s, 0.99))
        rows.append({"model": tag, "r_at_p99_mean": v.mean(), "r_at_p99_std": v.std(ddof=1),
                     "n_seeds": len(v)})
    if "astronet" in by_tag and "phantom_full" in by_tag:
        a = per_seed(by_tag["astronet"], lambda y, s: recall_at_precision(y, s, 0.99))
        p = per_seed(by_tag["phantom_full"], lambda y, s: recall_at_precision(y, s, 0.99))
        t, pv = welch(p, a)
        out["rp99_phantom_vs_astronet"] = {"delta": float(p.mean() - a.mean()),
                                           "t": t, "p": pv}
    out["rp99"] = rows


# ----------------------------------------------------------------- comment 3
def block_nested(by_tag: dict, out: dict) -> None:
    """Cumulative cost along a strictly nested chain of deletions."""
    ref = per_seed(by_tag["phantom_full"], average_precision_score)
    rows = []
    prev = None
    for tag, label, inputs in NESTED_CHAIN:
        if tag not in by_tag:
            print(f"  [nested] missing {tag}, skipping", flush=True)
            continue
        ap = per_seed(by_tag[tag], average_precision_score)
        cum = float(ref.mean() - ap.mean())
        step = None if prev is None else float(prev.mean() - ap.mean())
        _, p_vs_full = welch(ref, ap) if tag != "phantom_full" else (0.0, 1.0)
        rows.append({
            "tag": tag, "label": label, "inputs": inputs,
            "ap_mean": float(ap.mean()), "ap_std": float(ap.std(ddof=1)),
            "cumulative_delta": cum, "step_delta": step,
            "p_vs_full": float(p_vs_full), "n_seeds": len(ap),
        })
        prev = ap
    # The final link shares AstroNet's inputs exactly: the architecture contrast.
    if "astronet" in by_tag and rows and rows[-1]["tag"] == "nest_n7_sv_only":
        a = per_seed(by_tag["astronet"], average_precision_score)
        last = per_seed(by_tag["nest_n7_sv_only"], average_precision_score)
        t, pv = welch(last, a)
        out["architecture_contrast"] = {
            "phantom_sv_no_scalars_ap": float(last.mean()),
            "phantom_sv_no_scalars_std": float(last.std(ddof=1)),
            "astronet_ap": float(a.mean()), "astronet_std": float(a.std(ddof=1)),
            "delta": float(last.mean() - a.mean()), "t": t, "p": pv,
        }
    out["nested_chain"] = rows


# ---------------------------------------------------------------- comment 10
def block_perclass(by_tag: dict, data: str, out: dict) -> None:
    """Split performance by the two negative classes, AFP and NTP."""
    with h5py.File(data, "r") as f:
        raw = f["label"][:]
    labels = np.array([b.decode() if isinstance(b, bytes) else str(b) for b in raw])
    rows = []
    for tag in ("astronet", "phantom_full"):
        if tag not in by_tag:
            continue
        aps, aucs = {"AFP": [], "NTP": []}, {"AFP": [], "NTP": []}
        for r in by_tag[tag]:
            z = np.load(r["_preds"])
            cls = labels[z["test_idx"]]
            y, s = z["test_label"], z["test_score"]
            for neg in ("AFP", "NTP"):
                m = (cls == "PC") | (cls == neg)
                if m.sum() and 0 < y[m].sum() < m.sum():
                    aps[neg].append(average_precision_score(y[m], s[m]))
                    aucs[neg].append(roc_auc_score(y[m], s[m]))
        for neg in ("AFP", "NTP"):
            if aps[neg]:
                rows.append({
                    "model": tag, "negative_class": neg,
                    "n_negatives": int((labels == neg).sum()),
                    "ap_mean": float(np.mean(aps[neg])),
                    "ap_std": float(np.std(aps[neg], ddof=1)),
                    "auc_mean": float(np.mean(aucs[neg])),
                    "auc_std": float(np.std(aucs[neg], ddof=1)),
                })
    out["per_class"] = rows


# ------------------------------------------------------------- comments 5, 6
def block_power(by_tag: dict, out: dict) -> None:
    """Per-seed spread of the operational metric, and the resolvable effect."""
    rows = []
    for tag in ("astronet", "gbdt", "phantom_full"):
        if tag not in by_tag:
            continue
        row = {"model": tag}
        for target in (0.90, 0.95):
            v = per_seed(by_tag[tag], lambda y, s, t=target: precision_at_recall(y, s, t))
            row[f"p_at_r{int(target*100)}_mean"] = float(v.mean())
            row[f"p_at_r{int(target*100)}_std"] = float(v.std(ddof=1))
        rows.append(row)
    out["precision_at_recall_spread"] = rows

    if "phantom_full" in by_tag:
        p = per_seed(by_tag["phantom_full"], lambda y, s: precision_at_recall(y, s, 0.95))
        for other in ("astronet", "gbdt"):
            if other not in by_tag:
                continue
            a = per_seed(by_tag[other], lambda y, s: precision_at_recall(y, s, 0.95))
            t, pv = welch(p, a)
            out[f"p_at_r95_phantom_vs_{other}"] = {
                "delta": float(p.mean() - a.mean()), "t": t, "p": pv}

    # Minimum detectable effect for a two-sided Welch test, n = 5 per arm,
    # alpha = 0.05, power = 0.80. With equal variances the standard error of the
    # difference is sigma * sqrt(2/n) on nu = 2(n-1) degrees of freedom.
    n, alpha, power = 5, 0.05, 0.80
    nu = 2 * (n - 1)
    t_crit = stats.t.ppf(1 - alpha / 2, nu)
    t_pow = stats.t.ppf(power, nu)
    mde = {}
    for sigma in (0.006, 0.010, 0.017):
        mde[f"sigma_{sigma}"] = float((t_crit + t_pow) * sigma * np.sqrt(2.0 / n))
    out["minimum_detectable_effect"] = {
        "n_per_arm": n, "alpha": alpha, "power": power, "df": nu,
        "t_crit": float(t_crit), "delta_ap": mde,
    }


# ----------------------------------------------------------------- comment 8
def block_conformal(by_tag: dict, out: dict, tag: str = "phantom_full") -> None:
    """Validity check: conformal p-values on test negatives should be uniform."""
    runs = by_tag[tag]
    ref, test_s, val_s = None, [], []
    for r in runs:
        z = np.load(r["_preds"])
        if ref is None:
            ref = z
        elif not np.array_equal(z["test_idx"], ref["test_idx"]):
            continue
        test_s.append(z["test_score"])
        val_s.append(z["val_score"])
    test_score = np.mean(test_s, axis=0)
    val_score = np.mean(val_s, axis=0)
    y_test, y_val = ref["test_label"], ref["val_label"]

    cal_neg = val_score[y_val == 0]
    p_neg = conformal_pvalues(cal_neg, test_score[y_test == 0])
    ks = stats.kstest(p_neg, "uniform")
    grid = np.arange(1, len(cal_neg) + 2) / (len(cal_neg) + 1.0)
    out["conformal_validity"] = {
        "n_calibration_negatives": int(cal_neg.size),
        "n_test_negatives": int(p_neg.size),
        "mean_pvalue": float(p_neg.mean()),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "grid_resolution": float(grid[1] - grid[0]),
        "frac_below_0p05": float((p_neg <= 0.05).mean()),
        "frac_below_0p10": float((p_neg <= 0.10).mean()),
    }
    np.savez(
        os.path.join("results", "conformal_pvalues.npz"),
        p_test_negatives=p_neg,
        p_test_all=conformal_pvalues(cal_neg, test_score),
        test_label=y_test,
    )

    # Half of the calibration negatives, to show the floor moves with n.
    rng = np.random.default_rng(0)
    sub = rng.choice(cal_neg.size, cal_neg.size // 2, replace=False)
    p_half = conformal_pvalues(cal_neg[sub], test_score[y_test == 0])
    out["conformal_validity"]["half_calibration"] = {
        "n": int(sub.size),
        "smallest_attainable_p": float(1.0 / (sub.size + 1.0)),
        "mean_pvalue": float(p_half.mean()),
    }
    out["conformal_validity"]["smallest_attainable_p"] = float(1.0 / (cal_neg.size + 1.0))


# ----------------------------------------------------------------- comment 9
def block_tess(out: dict, tess_pred: str, tess_h5: str) -> None:
    """Stratify the cross-mission drop, and test whether recalibration helps."""
    z = np.load(tess_pred)
    y = z["label"].astype(int)
    with h5py.File(tess_h5, "r") as f:
        params = f["tess_params"][:]
        cols = list(f.attrs["tess_param_cols"])
    period = params[:, cols.index("tce_period")]

    rows = []
    for name, s in (("phantom", z["phantom_score"]), ("astronet", z["astronet_score"])):
        rows.append({
            "model": name, "stratum": "all", "n": int(y.size),
            "n_positive": int(y.sum()),
            "ap": float(average_precision_score(y, s)),
            "auc": float(roc_auc_score(y, s)),
            "p_at_r90": precision_at_recall(y, s, 0.90),
            "p_at_r95": precision_at_recall(y, s, 0.95),
        })
    # Period regimes: Kepler's training distribution is dominated by long
    # baselines, so the short-period end is where the covariate shift is largest.
    edges = [(0, 3), (3, 10), (10, np.inf)]
    for lo, hi in edges:
        m = (period >= lo) & (period < hi)
        if m.sum() < 50 or not (0 < y[m].sum() < m.sum()):
            continue
        for name, s in (("phantom", z["phantom_score"]), ("astronet", z["astronet_score"])):
            rows.append({
                "model": name,
                "stratum": f"P in [{lo}, {hi}) d",
                "n": int(m.sum()), "n_positive": int(y[m].sum()),
                "ap": float(average_precision_score(y[m], s[m])),
                "auc": float(roc_auc_score(y[m], s[m])),
                "p_at_r90": precision_at_recall(y[m], s[m], 0.90),
                "p_at_r95": precision_at_recall(y[m], s[m], 0.95),
            })
    out["tess_strata"] = rows

    # Recalibration on a held-out TESS slice. A temperature is a monotone map,
    # so ranking metrics are invariant by construction: if recalibration repairs
    # the calibration error while leaving average precision untouched, the
    # cross-mission drop is a failure of the representation, not of calibration.
    rng = np.random.default_rng(0)
    perm = rng.permutation(y.size)
    fit_ix, ev_ix = perm[: y.size // 5], perm[y.size // 5:]
    recal = []
    for name, s in (("phantom", z["phantom_score"]), ("astronet", z["astronet_score"])):
        t = fit_temperature(y[fit_ix].astype(float), s[fit_ix])
        s_cal = apply_temperature(s, t)
        recal.append({
            "model": name, "temperature": float(t),
            "n_fit": int(fit_ix.size), "n_eval": int(ev_ix.size),
            "ap_before": float(average_precision_score(y[ev_ix], s[ev_ix])),
            "ap_after": float(average_precision_score(y[ev_ix], s_cal[ev_ix])),
            "ece_before": expected_calibration_error(y[ev_ix].astype(float), s[ev_ix]),
            "ece_after": expected_calibration_error(y[ev_ix].astype(float), s_cal[ev_ix]),
        })
    out["tess_recalibration"] = recal


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="results/runs")
    ap.add_argument("--data", default="data/processed/dr24_views.h5")
    ap.add_argument("--tess-pred", default="results/tess_predictions.npz")
    ap.add_argument("--tess-data", default="data/processed/tess_views.h5")
    ap.add_argument("--out", default="results/referee_analyses.json")
    args = ap.parse_args()

    by_tag = load_runs(args.run_dir)
    print(f"loaded {sum(len(v) for v in by_tag.values())} runs "
          f"over {len(by_tag)} configurations", flush=True)

    out: dict = {}
    for name, fn in (
        ("rp99", lambda: block_rp99(by_tag, out)),
        ("nested", lambda: block_nested(by_tag, out)),
        ("perclass", lambda: block_perclass(by_tag, args.data, out)),
        ("power", lambda: block_power(by_tag, out)),
        ("conformal", lambda: block_conformal(by_tag, out)),
        ("tess", lambda: block_tess(out, args.tess_pred, args.tess_data)),
    ):
        print(f"  {name} ...", flush=True)
        fn()

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {args.out}", flush=True)

    if "nested_chain" in out:
        print("\nnested chain:")
        print(pd.DataFrame(out["nested_chain"])[
            ["label", "ap_mean", "ap_std", "step_delta", "cumulative_delta"]
        ].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
