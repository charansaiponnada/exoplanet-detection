"""Generate every figure in the paper from the saved run artefacts.

Style follows print-journal conventions: a fixed categorical palette assigned in
a stable order (a model keeps its colour in every panel), line style as a
redundant second encoding so the figures survive greyscale printing and colour
vision deficiency, recessive grids, and a legend on every multi-series panel.

Usage::

    python scripts/figures.py --data data/processed/dr24_views.h5
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exonet.conformal import apply_temperature, fit_temperature
from exonet.data import load_h5
from scripts.evaluate import conformal_report, ensemble_scores, load_runs

# Validated categorical palette, assigned in fixed order: a model keeps its
# colour across every figure regardless of which subset a panel shows.
PALETTE = {
    "phantom":   "#2a78d6",  # blue
    "astronet":  "#eb6834",  # orange
    "gbdt":      "#1baf7a",  # aqua
    "classical": "#eda100",  # yellow
    "other":     "#4a3aa7",  # violet
}
DASHES = {
    "phantom": (None, None), "astronet": (5, 2), "gbdt": (2, 1.5),
    "classical": (1, 1.5), "other": (6, 2, 1, 2),
}
CLASS_COLOR = {"PC": "#2a78d6", "AFP": "#eb6834", "NTP": "#4a3aa7"}

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#d8d7d2"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.edgecolor": TEXT_SECONDARY,
        "axes.labelcolor": TEXT_PRIMARY,
        "text.color": TEXT_PRIMARY,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.8,
        "axes.axisbelow": True,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "legend.frameon": False,
    }
)

COL_WIDTH = 3.4     # single column, inches
FULL_WIDTH = 7.0    # full page width


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def save(fig, out_dir: str, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote {name}", flush=True)


# ---------------------------------------------------------------- figure 1
def fig_views(blob, out_dir: str):
    """Example input channels for one TCE of each class."""
    names = ["global", "local", "odd", "even", "secondary", "half", "double", "cent_local"]
    titles = [
        "Global ($P$)", "Local ($P$)", "Odd transits", "Even transits",
        "Secondary", "Harmonic ($P/2$)", "Harmonic ($2P$)", "Centroid (local)",
    ]
    lab = blob["label"]
    # pick the highest signal-to-noise example of each class for legibility
    snr = blob["catalog"][:, blob["cat_cols"].index("tce_model_snr")]
    picks = []
    for c in ("PC", "AFP", "NTP"):
        idx = np.where(lab == c)[0]
        picks.append((c, int(idx[np.argsort(-np.nan_to_num(snr[idx]))[0]])))

    fig, axes = plt.subplots(
        3, len(names), figsize=(FULL_WIDTH, 3.2), sharex=False, sharey="row"
    )
    fig.subplots_adjust(wspace=0.28)
    for r, (cls, i) in enumerate(picks):
        for c, (n, t) in enumerate(zip(names, titles)):
            ax = style(axes[r, c])
            v = blob["views"][n][i]
            ax.plot(np.linspace(0, 1, v.size), v, lw=0.6, color=CLASS_COLOR[cls])
            ax.set_xticks([])
            if r == 0:
                ax.set_title(t, fontsize=5.8, pad=3)
            if c == 0:
                ax.set_ylabel(cls, fontsize=8)
            ax.tick_params(labelsize=5)
            ax.grid(False)
    fig.suptitle(
        "Input channels: a planet candidate (PC), an astrophysical false "
        "positive (AFP) and a non-transiting phenomenon (NTP)",
        fontsize=8, y=1.03,
    )
    save(fig, out_dir, "fig1_views")


# ---------------------------------------------------------------- figure 2
def fig_pr_roc(runs, run_dir: str, out_dir: str, split: str, headline: str):
    """Precision-recall and ROC for every model, on the same test split."""
    curves = []
    ens = ensemble_scores(runs, headline, split)
    if ens is not None:
        curves.append(("PHANTOM (ensemble)", "phantom", ens["test_label"], ens["test_score"]))
    a = ensemble_scores(runs, "astronet", split)
    if a is not None:
        curves.append(("AstroNet", "astronet", a["test_label"], a["test_score"]))
    for path in sorted(glob.glob(os.path.join(run_dir, f"baselines_*_{split}_preds.npz"))):
        z = np.load(path)
        curves.append(("GBDT (scalars)", "gbdt", z["test_label"], z["gbdt_score"]))
        curves.append(("Classical vetting", "classical", z["test_label"], z["classical_score"]))
        break

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.7))
    for name, key, y, s in curves:
        p, r, _ = precision_recall_curve(y, s)
        ax1.plot(r, p, lw=1.4, color=PALETTE[key], dashes=DASHES[key], label=name)
        fpr, tpr, _ = roc_curve(y, s)
        ax2.plot(fpr, tpr, lw=1.4, color=PALETTE[key], dashes=DASHES[key], label=name)
    style(ax1); style(ax2)
    ax1.set_xlabel("Recall"); ax1.set_ylabel("Precision")
    ax1.set_title("Precision-recall"); ax1.set_ylim(0, 1.02)
    ax2.plot([0, 1], [0, 1], lw=0.6, color=GRID)
    ax2.set_xlabel("False positive rate"); ax2.set_ylabel("True positive rate")
    ax2.set_title("ROC"); ax2.set_xscale("log"); ax2.set_xlim(1e-3, 1)
    ax1.legend(loc="lower left")
    save(fig, out_dir, "fig2_pr_roc")


# ---------------------------------------------------------------- figure 3
def fig_ablation(summary, out_dir: str, split: str):
    """Ablation of each architectural component."""
    rows = [
        r for r in summary["comparison"]
        if r["split"] == split and r.get("model") == "phantom"
    ]
    if not rows:
        return
    rows.sort(key=lambda r: r["ap_mean"])
    labels = [r["tag"].replace("phantom_", "").replace("_", " ") for r in rows]
    vals = np.array([r["ap_mean"] for r in rows])
    errs = np.array([r["ap_std"] for r in rows])

    fig, ax = plt.subplots(figsize=(COL_WIDTH, 0.3 * len(rows) + 1.0))
    style(ax)
    ypos = np.arange(len(rows))
    ax.barh(
        ypos, vals, xerr=errs, height=0.62, color=PALETTE["phantom"],
        error_kw={"lw": 0.8, "ecolor": TEXT_SECONDARY}, zorder=3,
    )
    ax.set_yticks(ypos); ax.set_yticklabels(labels)
    ax.set_xlabel("Average precision")
    ax.set_xlim(max(0.0, vals.min() - 0.06), min(1.0, vals.max() + 0.02))
    ax.grid(axis="y", visible=False)
    for y_, v in zip(ypos, vals):  # direct labels: relief for the contrast warning
        ax.text(v - 0.002, y_, f"{v:.3f}", va="center", ha="right",
                color="white", fontsize=6.5, zorder=4)
    ax.set_title("Component ablation")
    save(fig, out_dir, "fig3_ablation")


# ---------------------------------------------------------------- figure 4
def fig_decoder(blob, runs, out_dir: str, split: str, headline: str):
    """Physical decoder: fitted profiles and the U-versus-V shape parameter."""
    import torch

    from exonet.model import PHANTOM, VIEW_TABLE

    group = [r for r in runs if r["tag"] == headline and r["split_mode"] == split]
    if not group or not group[0].get("use_decoder"):
        return
    rec = group[0]
    ckpt = rec["_json"].replace(".json", ".pt")
    if not os.path.exists(ckpt):
        return

    ens = ensemble_scores(runs, headline, split)
    if ens is None or "params_mean" not in ens:
        return
    idx = ens["test_idx"]
    lab = blob["label"][idx]
    p = ens["params_mean"]

    from exonet.model import TransitDecoder
    dec = TransitDecoder(8)
    grid = dec.grid.numpy()

    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.3))
    # (a) example fits
    ax = style(axes[0])
    for cls, offset in (("PC", 0.0), ("AFP", -1.6)):
        sel = np.where(lab == cls)[0]
        if not sel.size:
            continue
        j = sel[np.argmax(p[sel, 0])]
        obs = blob["views"]["local"][idx[j]]
        with torch.no_grad():
            model = dec.render(*[torch.tensor(p[j, k : k + 1]) for k in range(5)])[0].numpy()
        ax.plot(grid, obs + offset, lw=0.7, color=GRID, zorder=1)
        ax.plot(grid, model + offset, lw=1.3, color=CLASS_COLOR[cls], zorder=2, label=cls)
    ax.set_xlabel("Phase (transit durations)")
    ax.set_ylabel("Normalised flux (offset)")
    ax.set_title("(a) Decoder fits"); ax.legend(loc="lower right")

    # (b) ingress-softness distribution: the U-vs-V discriminator
    ax = style(axes[1])
    for cls in ("PC", "AFP", "NTP"):
        sel = lab == cls
        if sel.sum() < 10:
            continue
        ax.hist(p[sel, 4], bins=np.linspace(0, 1, 40), density=True, histtype="step",
                lw=1.3, color=CLASS_COLOR[cls], label=cls)
    ax.set_xlabel(r"Ingress softness $s$  (U-shape $\rightarrow$ V-shape)")
    ax.set_ylabel("Density"); ax.set_title("(b) Transit shape"); ax.legend()

    # (c) reconstruction quality
    ax = style(axes[2])
    local = blob["views"]["local"][idx]
    with torch.no_grad():
        models = dec.render(*[torch.tensor(p[:, k]) for k in range(5)]).numpy()
    resid = np.sqrt(((local - models) ** 2).mean(axis=1))
    for cls in ("PC", "AFP", "NTP"):
        sel = lab == cls
        if sel.sum() < 10:
            continue
        ax.hist(resid[sel], bins=np.linspace(0, 1, 40), density=True, histtype="step",
                lw=1.3, color=CLASS_COLOR[cls], label=cls)
    ax.set_xlabel("Reconstruction RMS"); ax.set_ylabel("Density")
    ax.set_title("(c) Physical model fit quality"); ax.legend()
    save(fig, out_dir, "fig4_decoder")


# ---------------------------------------------------------------- figure 5
def fig_conformal_calibration(runs, out_dir: str, split: str, headline: str):
    """Conformal FDR control and probability calibration."""
    ens = ensemble_scores(runs, headline, split)
    if ens is None:
        return
    rows = conformal_report(ens, levels=np.linspace(0.01, 0.30, 15))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.6))
    ax = style(ax1)
    q = [r["q_nominal"] for r in rows]
    emp = [r["empirical_fdr"] for r in rows]
    ax.plot([0, max(q)], [0, max(q)], lw=0.8, color=TEXT_SECONDARY,
            dashes=(3, 2), label="nominal level")
    ax.plot(q, emp, lw=1.5, marker="o", ms=3.5, color=PALETTE["phantom"],
            label="realised FDR")
    ax.set_xlabel("Target FDR $q$"); ax.set_ylabel("Realised FDR")
    ax.set_title("(a) Conformal candidate selection"); ax.legend(loc="upper left")

    ax = style(ax2)
    t = fit_temperature(ens["val_label"], ens["val_score"])
    for name, s, key in (
        ("uncalibrated", ens["test_score"], "astronet"),
        ("temperature-scaled", apply_temperature(ens["test_score"], t), "phantom"),
    ):
        edges = np.linspace(0, 1, 13)
        ix = np.clip(np.digitize(s, edges) - 1, 0, len(edges) - 2)
        xs, ys = [], []
        for b in range(len(edges) - 1):
            m = ix == b
            if m.sum() < 20:
                continue
            xs.append(s[m].mean()); ys.append(ens["test_label"][m].mean())
        ax.plot(xs, ys, lw=1.4, marker="o", ms=3.5, color=PALETTE[key],
                dashes=DASHES[key], label=name)
    ax.plot([0, 1], [0, 1], lw=0.8, color=TEXT_SECONDARY, dashes=(3, 2))
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title("(b) Reliability"); ax.legend(loc="upper left")
    save(fig, out_dir, "fig5_conformal_calibration")


# ---------------------------------------------------------------- figure 6
def fig_parameters(out_dir: str, results_dir: str):
    """Recovered transit parameters against independent literature values."""
    path = os.path.join(results_dir, "parameter_recovery.csv")
    if not os.path.exists(path):
        return
    import pandas as pd

    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.8))
    specs = [
        ("lit_duration", "cat_duration", "est_duration", "Transit duration (h)", axes[0]),
        ("lit_depth_ppm", "cat_depth_ppm", "est_depth_ppm", "Transit depth (ppm)", axes[1]),
    ]
    for truth_c, cat_c, est_c, label, ax in specs:
        style(ax)
        d = df.dropna(subset=[truth_c, cat_c, est_c])
        d = d[(d[truth_c] > 0) & (d[cat_c] > 0) & (d[est_c] > 0)]
        if d.empty:
            continue
        ax.scatter(d[truth_c], d[cat_c], s=5, alpha=0.45,
                   color=PALETTE["astronet"], label="DV catalogue", zorder=2)
        ax.scatter(d[truth_c], d[est_c], s=5, alpha=0.45,
                   color=PALETTE["phantom"], label="PHANTOM decoder", zorder=3)
        lims = [min(d[truth_c].min(), 1e-9), d[truth_c].max()]
        ax.plot(lims, lims, lw=0.8, color=TEXT_SECONDARY, dashes=(3, 2), zorder=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"Literature {label.lower()}")
        ax.set_ylabel(f"Recovered {label.lower()}")
        ax.legend(loc="upper left")
    save(fig, out_dir, "fig6_parameters")


# ---------------------------------------------------------------- figure 7
def fig_cross_mission(out_dir: str, results_dir: str):
    """Zero-shot transfer to TESS."""
    path = os.path.join(results_dir, "tess_transfer.json")
    if not os.path.exists(path):
        return
    with open(path) as fh:
        res = json.load(fh)
    preds = os.path.join(results_dir, "tess_predictions.npz")
    if not os.path.exists(preds):
        return
    z = np.load(preds)
    y = z["label"]

    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.6))
    style(ax)
    for key, name in (("phantom_score", "PHANTOM"), ("astronet_score", "AstroNet")):
        if key not in z:
            continue
        pkey = "phantom" if "phantom" in key else "astronet"
        p, r, _ = precision_recall_curve(y, z[key])
        ap = res["models"][pkey]["ap"]
        ax.plot(r, p, lw=1.4, color=PALETTE[pkey], dashes=DASHES[pkey],
                label=f"{name} (AP {ap:.3f})")
    ax.axhline(y.mean(), lw=0.8, color=TEXT_SECONDARY, dashes=(3, 2),
               label=f"chance ({y.mean():.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_ylim(0, 1.02)
    ax.set_title("Zero-shot transfer: Kepler $\\rightarrow$ TESS")
    ax.legend(loc="lower left")
    save(fig, out_dir, "fig7_cross_mission")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/processed/dr24_views.h5")
    ap.add_argument("--run-dir", default="results/runs")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out-dir", default="paper/figures")
    ap.add_argument("--split", default="group")
    ap.add_argument("--headline-tag", default="phantom_full")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    runs = load_runs(args.run_dir)
    blob = load_h5(args.data)
    summary_path = os.path.join(args.results, "summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}

    print("generating figures", flush=True)
    fig_views(blob, args.out_dir)
    if runs:
        fig_pr_roc(runs, args.run_dir, args.out_dir, args.split, args.headline_tag)
        fig_decoder(blob, runs, args.out_dir, args.split, args.headline_tag)
        fig_conformal_calibration(runs, args.out_dir, args.split, args.headline_tag)
    if summary:
        fig_ablation(summary, args.out_dir, args.split)
    fig_parameters(args.out_dir, args.results)
    fig_cross_mission(args.out_dir, args.results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
