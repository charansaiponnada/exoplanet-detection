"""Non-deep baselines evaluated on exactly the same splits as the networks.

Two reference points matter for the paper:

``classical``
    The rule-based vetting cascade (odd-even depth test, secondary-eclipse
    search, V-shape test) that operational pipelines and this repository's
    original implementation use.  It needs no training data and is fully
    explainable, which is precisely why it is the incumbent worth beating.

``gbdt``
    Gradient-boosted trees on the scalar feature vector, including the DV
    diagnostic statistics.  A strong tabular learner establishes how much of the
    problem is solvable from summary statistics alone, and therefore how much
    the phase-folded morphology actually contributes.

Usage::

    python scripts/baselines.py --data data/processed/dr24_views.h5 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exonet.data import ScalarConditioner, load_h5, make_splits
from scripts.train import build_scalars, metrics


def _smooth(v: np.ndarray, w: int = 5) -> np.ndarray:
    """Boxcar smoothing, used only to stabilise threshold crossings."""
    k = np.ones(w) / w
    return np.convolve(v, k, mode="same")


def shape_ratio(local_views: np.ndarray) -> np.ndarray:
    """Width at quarter depth divided by width at half depth, per TCE.

    Widths are measured by walking outwards from the deepest bin until the
    profile rises back above the threshold, rather than by taking the extent of
    all bins below it.  The naive version is dominated by isolated noise
    excursions far from the event - which drives the ratio towards 1 for the
    *noisiest* light curves and therefore inverts the very discrimination the
    test is meant to provide.

    A flat-bottomed (planetary) profile gives a ratio near 1; a pure V-shape
    gives 1.5.
    """
    n, m = local_views.shape
    out = np.full(n, np.nan)
    for i in range(n):
        f = _smooth(local_views[i].astype(np.float64))
        depth = -f.min()
        if depth <= 0:
            continue
        c = int(np.argmin(f))
        widths = []
        for level in (-0.5 * depth, -0.25 * depth):
            lo = c
            while lo > 0 and f[lo - 1] < level:
                lo -= 1
            hi = c
            while hi < m - 1 and f[hi + 1] < level:
                hi += 1
            widths.append(hi - lo + 1)
        if widths[0] > 0:
            out[i] = widths[1] / widths[0]
    return out


def secondary_significance(global_views: np.ndarray, sec_ratio: np.ndarray) -> np.ndarray:
    """Secondary-eclipse depth in units of the binned out-of-transit noise.

    ``sec_ratio`` is already expressed as a fraction of the primary depth, and
    the global view is normalised so that the primary depth is 1.  Measuring the
    noise on the *same* normalised, binned view therefore puts numerator and
    denominator in consistent units - the naive comparison against the
    per-cadence scatter understates the noise reduction from binning by an order
    of magnitude and the test then never fires.

    The returned value is corrected for the look-elsewhere effect.  ``sec_ratio``
    is the deepest bin anywhere outside the primary, i.e. a minimum over ~2000
    trials, so under pure noise it already sits at ``sigma * sqrt(2 ln N)`` ~ 3.9
    sigma.  Comparing it against a naive 3 sigma threshold flags essentially
    every target.  Subtracting the expected noise maximum makes the statistic an
    *excess* over what noise alone produces.
    """
    n, m = global_views.shape
    out = np.zeros(n)
    core = slice(int(0.45 * m), int(0.55 * m))
    expected_max = np.sqrt(2.0 * np.log(m))
    for i in range(n):
        v = global_views[i].astype(np.float64)
        oot = np.delete(v, np.arange(m)[core])
        # median absolute deviation is insensitive to the secondary itself
        sigma = 1.4826 * np.median(np.abs(oot - np.median(oot)))
        if sigma > 0:
            out[i] = sec_ratio[i] / sigma - expected_max
    return out


def classical_scores(blob, der_cols) -> np.ndarray:
    """Graded score from the three classical vetting flags.

    The cascade is natively a hard label; grading it by the number of flags that
    fire yields a four-valued score so that it can be placed on the same
    precision-recall axes as the learned models rather than as a single point.
    """
    d = blob["derived"]
    oe = np.nan_to_num(d[:, der_cols.index("oe_stat")])
    sec = np.nan_to_num(d[:, der_cols.index("sec_ratio")])
    sr = shape_ratio(blob["views"]["local"])
    sig = secondary_significance(blob["views"]["global"], sec)

    # (1) odd-even depth mismatch above 30 per cent
    flag_oe = oe > 0.3
    # (2) secondary eclipse significantly deeper than the noise maximum
    flag_sec = sig > 1.0
    # (3) V-shaped rather than flat-bottomed profile
    flag_shape = np.nan_to_num(sr, nan=1.0) > 1.35

    n_flags = flag_oe.astype(int) + flag_sec.astype(int) + flag_shape.astype(int)
    return 1.0 - n_flags / 3.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/processed/dr24_views.h5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-mode", choices=["group", "tce"], default="group")
    ap.add_argument("--scalar-groups", default="transit,stellar,dv,derived")
    ap.add_argument("--out-dir", default="results/runs")
    args = ap.parse_args()

    blob = load_h5(args.data)
    y = (blob["label"] == "PC").astype(int)
    tr, va, te = make_splits(blob["kepid"], len(y), mode=args.split_mode, seed=args.seed)

    results = {}

    # ---- classical rule cascade (no training) -------------------------------
    s_cls = classical_scores(blob, blob["der_cols"])
    results["classical"] = metrics(y[te], s_cls[te])
    print("classical:", json.dumps(results["classical"], indent=2), flush=True)

    # ---- gradient-boosted trees on scalars ----------------------------------
    import lightgbm as lgb

    groups = args.scalar_groups.split(",")
    X, names = build_scalars(blob, groups)
    cond = ScalarConditioner(names).fit(X[tr])
    Xc = cond.transform(X)

    model = lgb.LGBMClassifier(
        n_estimators=3000, learning_rate=0.02, num_leaves=63,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=args.seed, n_jobs=16, verbose=-1,
    )
    model.fit(
        Xc[tr], y[tr],
        eval_set=[(Xc[va], y[va])], eval_metric="average_precision",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    s_gbdt = model.predict_proba(Xc[te])[:, 1]
    results["gbdt"] = metrics(y[te], s_gbdt)
    results["gbdt"]["best_iteration"] = int(model.best_iteration_ or 0)
    print("gbdt:", json.dumps(results["gbdt"], indent=2), flush=True)

    importance = sorted(
        zip(names, model.feature_importances_.tolist()), key=lambda t: -t[1]
    )
    results["gbdt_importance"] = importance[:25]

    os.makedirs(args.out_dir, exist_ok=True)
    stem = f"baselines_seed{args.seed}_{args.split_mode}"
    with open(os.path.join(args.out_dir, f"{stem}.json"), "w") as fh:
        json.dump(
            {"seed": args.seed, "split_mode": args.split_mode, "results": results},
            fh, indent=2,
        )
    np.savez_compressed(
        os.path.join(args.out_dir, f"{stem}_preds.npz"),
        test_idx=te, test_label=y[te],
        classical_score=s_cls[te], gbdt_score=s_gbdt,
        val_idx=va, val_label=y[va],
        classical_val=s_cls[va], gbdt_val=model.predict_proba(Xc[va])[:, 1],
    )
    print(f"wrote {args.out_dir}/{stem}.*", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
