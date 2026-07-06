"""
Generate publication-quality pipeline diagrams for the exoplanet detection pipeline.
All diagrams use matplotlib geometric primitives (boxes, arrows, text) with
font sizes suitable for presentation slides (16:9 aspect ratio).

Usage:
    uv run --active python helper_scripts/visualize/pipeline_diagrams.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


OUTPUT_DIR = Path("output/diagrams")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FONT = {"family": "DejaVu Sans", "weight": "bold"}
TITLE_SIZE = 20
SUBTITLE_SIZE = 14
LABEL_SIZE = 12
BODY_SIZE = 10
SMALL_SIZE = 9

COLORS = {
    "bg": "#FAFAFA",
    "title": "#1A1A2E",
    "subtitle": "#16213E",
    "data": "#0F3460",
    "process": "#E94560",
    "ml": "#533483",
    "output": "#0B8457",
    "arrow": "#555555",
    "box_bg": "#FFFFFF",
    "box_edge": "#333333",
    "accent1": "#4A90D9",
    "accent2": "#50C878",
    "accent3": "#E8A838",
    "accent4": "#D9564A",
    "accent5": "#8B5CF6",
}

FIG_WIDTH, FIG_HEIGHT = 16, 9


def setup_figure(title: str) -> tuple:
    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, FIG_HEIGHT))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, FIG_WIDTH)
    ax.set_ylim(0, FIG_HEIGHT)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(
        0.5, FIG_HEIGHT - 0.3, title,
        fontsize=TITLE_SIZE, fontweight="bold", fontfamily=FONT["family"],
        color=COLORS["title"], ha="center", va="top",
        transform=ax.transData,
    )
    return fig, ax


def draw_box(
    ax, x, y, w, h,
    label="", body="",
    facecolor="#FFFFFF", edgecolor="#333333",
    title_color="#1A1A2E", body_color="#555555",
    title_size=LABEL_SIZE, body_size=BODY_SIZE,
    linewidth=1.5,
):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=linewidth, zorder=2,
    )
    ax.add_patch(box)
    if label:
        ax.text(
            x + w / 2, y + h - 0.25, label,
            fontsize=title_size, fontweight="bold",
            fontfamily=FONT["family"], color=title_color,
            ha="center", va="top", zorder=3,
        )
    if body:
        ax.text(
            x + w / 2, y + h / 2 - 0.2, body,
            fontsize=body_size, color=body_color,
            ha="center", va="center", zorder=3, linespacing=1.4,
        )


def draw_arrow(ax, x1, y1, x2, y2, color="#555555", lw=2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->", color=color, lw=lw,
            connectionstyle="arc3,rad=0",
        ),
        zorder=1,
    )


def draw_arrowhead(ax, x, y, dx=0, dy=-0.3, color="#555555"):
    ax.annotate(
        "", xy=(x + dx, y + dy), xytext=(x, y),
        arrowprops=dict(arrowstyle="->", color=color, lw=2),
        zorder=1,
    )


# ──────────────────────────────────────────────────────────────────────
# SLIDE 1: System Architecture — High-Level Overview
# ──────────────────────────────────────────────────────────────────────
def slide_system_architecture():
    fig, ax = setup_figure("System Architecture — High-Level Overview")

    y_base = 5.5
    box_w, box_h = 1.85, 1.6
    gap = 0.4
    x_start = (FIG_WIDTH - (7 * box_w + 6 * gap)) / 2

    stages = [
        ("TESS\nLight Curves", "MAST Archive\nRaw FITS Data", COLORS["data"]),
        ("Preprocessing\n& Detrending", "Sigma Clipping\nSavitzky-Golay\nFilter", COLORS["process"]),
        ("Signal Detection\n(BLS)", "Box Least Squares\nPeriodogram\nSearch", COLORS["process"]),
        ("Feature\nEngineering", "17 features:\nshape, noise,\nconsistency", COLORS["ml"]),
        ("Classification\n(built + planned)", "Rules + RandomForest\n(running now)\nMamba SSM (Phase 2)", COLORS["ml"]),
        ("Validation &\nConfidence", "Plausibility checks\nRule-based\nconfidence score", COLORS["output"]),
        ("Dashboard\n& Report", "Streamlit app\nJSON + PNG export", COLORS["output"]),
    ]

    for i, (title, body, color) in enumerate(stages):
        x = x_start + i * (box_w + gap)
        y = y_base - box_h / 2
        draw_box(
            ax, x, y, box_w, box_h,
            label=title, body=body,
            facecolor=color, edgecolor=color,
            title_color="#FFFFFF", body_color="#E0E0E0",
            title_size=11, body_size=9,
        )
        if i < len(stages) - 1:
            ax.annotate(
                "", xy=(x + box_w + gap * 0.3, y + box_h / 2),
                xytext=(x + box_w - 0.1, y + box_h / 2),
                arrowprops=dict(arrowstyle="->", color=COLORS["arrow"], lw=2.5),
                zorder=1,
            )

    # Legend
    legend_y = 2.0
    legend_items = [
        ("Data Ingestion", COLORS["data"]),
        ("Preprocessing", COLORS["process"]),
        ("ML / Detection", COLORS["ml"]),
        ("Output / Results", COLORS["output"]),
    ]
    for i, (lbl, clr) in enumerate(legend_items):
        x = 4.0 + i * 2.8
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, legend_y), 0.4, 0.4,
            boxstyle="round,pad=0.05",
            facecolor=clr, edgecolor=clr, zorder=2,
        ))
        ax.text(x + 0.5, legend_y + 0.2, lbl, fontsize=BODY_SIZE, color=COLORS["subtitle"],
                va="center", zorder=3)

    fig.savefig(OUTPUT_DIR / "01_system_architecture.png", dpi=150, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close(fig)
    print("  [OK] 01_system_architecture.png")


# ──────────────────────────────────────────────────────────────────────
# SLIDE 2: Detailed Pipeline Flow
# ──────────────────────────────────────────────────────────────────────
def slide_detailed_pipeline():
    fig, ax = setup_figure("Detailed Pipeline — Data Flow & Processing Stages")

    # Row 1: data ingestion
    row1_y = 6.3
    bw, bh = 2.8, 1.8
    gap_x = 0.6

    draw_box(ax, 0.5, row1_y, bw, bh,
             label="STAGE 1: Data Ingestion",
             body="TESS lightkurve search\nMAST/SPOC archive\nDownload FITS files\nSynthetic injection for\ntesting/validation",
             facecolor="#E8F4FD", edgecolor=COLORS["data"], title_color=COLORS["data"])

    draw_box(ax, 0.5 + bw + gap_x, row1_y, bw, bh,
             label="STAGE 2: Preprocessing",
             body="Sigma clip (5s) outliers\nSavitzky-Golay detrend\nWindow: 0.5 day\nNormalize flux\nRemove systematics",
             facecolor="#FFF0E0", edgecolor=COLORS["process"], title_color=COLORS["process"])

    draw_box(ax, 0.5 + 2 * (bw + gap_x), row1_y, bw, bh,
             label="STAGE 3: Period Search",
             body="Box Least Squares (BLS)\n5000 trial periods\n0.5-15 day range\n0.5-8 hr durations\nBest SNR peak selection",
             facecolor="#F0E6FF", edgecolor=COLORS["ml"], title_color=COLORS["ml"])

    # Row 2
    row2_y = 3.8
    draw_box(ax, 0.5, row2_y, bw, bh,
             label="STAGE 4: Phase Folding",
             body="Fold on best period\nStack all transits\nBinned phase curve\n30 bins across transit\nVisual inspection",
             facecolor="#F0E6FF", edgecolor=COLORS["ml"], title_color=COLORS["ml"])

    draw_box(ax, 0.5 + bw + gap_x, row2_y, bw, bh,
             label="STAGE 5: Vetting + Features",
             body="Odd-even / secondary /\nshape tests (rule-based)\n17-feature vector:\nconsistency, shape,\nnoise, entropy",
             facecolor="#FFF0E0", edgecolor=COLORS["process"], title_color=COLORS["process"])

    draw_box(ax, 0.5 + 2 * (bw + gap_x), row2_y, bw, bh,
             label="STAGE 6: Classification",
             body="RUNNING NOW:\nRandomForest (4-class,\ntrained on synthetic data)\nPLANNED (Phase 2):\nMamba SSM on curated data",
             facecolor="#E8FDE0", edgecolor=COLORS["accent2"], title_color="#0B8457")

    # Row 3
    row3_y = 1.3
    draw_box(ax, 0.5 + 0.5 * (bw + gap_x), row3_y, bw, bh,
             label="STAGE 7: Parameter Estimation",
             body="Transit depth (ppm)\nOrbital period (days)\nTransit duration (hrs)\nMid-transit time T0\nDetection SNR",
             facecolor="#E8FDE0", edgecolor=COLORS["output"], title_color=COLORS["output"])

    draw_box(ax, 0.5 + 1.5 * (bw + gap_x) + 0.5, row3_y, bw * 0.8, bh,
             label="OUTPUT",
             body="Confidence score\n+ plausibility flags\nStreamlit dashboard\nJSON / PNG export",
             facecolor="#D0F0C0", edgecolor=COLORS["output"], title_color=COLORS["output"],
             title_size=11, body_size=9)

    # Connecting arrows between stages (vertical flow)
    # Row 1 -> Row 2
    for i in range(3):
        cx = 0.5 + i * (bw + gap_x) + bw / 2
        draw_arrowhead(ax, cx, row1_y, 0, -(row1_y - row2_y - bh) * 0.7)

    # Row 2 -> Row 3 (converging)
    draw_arrowhead(ax, 0.5 + bw + gap_x + bw / 2, row2_y, 0,
                   -(row2_y - row3_y - bh) * 0.7)

    fig.savefig(OUTPUT_DIR / "02_detailed_pipeline.png", dpi=150, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close(fig)
    print("  [OK] 02_detailed_pipeline.png")


# ──────────────────────────────────────────────────────────────────────
# SLIDE 3: AI Classification Engine — Model Architecture
# ──────────────────────────────────────────────────────────────────────
def slide_model_architecture():
    fig, ax = setup_figure("Phase 2 Plan — Mamba-Based Architecture (Not Yet Trained)")

    ax.add_patch(mpatches.FancyBboxPatch(
        (0.5, 7.4), 6.0, 0.5, boxstyle="round,pad=0.05",
        facecolor="#FFF3CD", edgecolor="#E8A838", linewidth=1.5, zorder=2,
    ))
    ax.text(3.5, 7.65, "PLANNED — pending ISRO's curated labeled dataset",
            fontsize=10, fontweight="bold", color="#7A5C00", ha="center", va="center", zorder=3)

    # Input
    draw_box(ax, 0.5, 3.5, 2.2, 1.6,
             label="INPUT",
             body="Phase-folded\nLight Curve\n(time, flux, flux_err)",
             facecolor=COLORS["data"], edgecolor=COLORS["data"],
             title_color="#FFFFFF", body_color="#E0E0E0")

    # Feature extraction block (expanded)
    draw_box(ax, 3.5, 3.5, 2.8, 2.2,
             label="Feature Extraction",
             body="BLS periodogram features\nTransit shape metrics\nNoise statistics\nDepth/duration/SR\nPeriod aliases",
             facecolor="#E8F4FD", edgecolor=COLORS["accent1"], title_color=COLORS["accent1"])

    # Mamba SSM core
    draw_box(ax, 7.0, 3.5, 2.8, 2.2,
             label="Mamba SSM Encoder",
             body="Bidirectional SSM\nSelective state space\nLong-range dependencies\nLinear complexity O(n)\nHidden dim: 512",
             facecolor="#F0E6FF", edgecolor=COLORS["accent5"], title_color=COLORS["accent5"])

    # Mamba SSM core
    draw_box(ax, 10.5, 3.5, 2.8, 2.2,
             label="Classification Head",
             body="MLP: 512 -> 256 -> 128\nDropout: 0.3\nSoftmax output\nConfidence calibration\nTop-1 prediction",
             facecolor="#F0E6FF", edgecolor=COLORS["accent5"], title_color=COLORS["accent5"])

    # Output classes
    draw_box(ax, 14.0, 3.5, 1.8, 2.2,
             label="OUTPUT",
             body="6 Classes:\n- Planet transit\n- Eclipsing binary\n- Blend\n- Starspot\n- Noise\n- Unknown",
             facecolor=COLORS["output"], edgecolor=COLORS["output"],
             title_color="#FFFFFF", body_color="#E0E0E0",
             title_size=11, body_size=9)

    # Arrows
    arrow_starts = [(1.7, 4.3), (6.3, 4.3), (9.8, 4.3), (13.3, 4.3)]
    for sx, sy in arrow_starts:
        draw_arrow(ax, sx, sy, sx + 0.5, sy)

    # Training callout
    draw_box(ax, 3.5, 0.8, 6.0, 1.3,
             label="Training Strategy",
             body="Supervised learning, cross-entropy loss\nAdamW (lr=1e-4)  |  Batch size 64\nEarly stopping (patience=10)",
             facecolor="#FFF8E0", edgecolor=COLORS["accent3"], title_color=COLORS["accent3"],
             title_size=11, body_size=9)

    # Data augmentation note
    draw_box(ax, 10.5, 0.8, 5.3, 1.3,
             label="Data Augmentation",
             body="Noise injection  |  Time-domain warping\nTransit depth scaling  |  Period jitter\nMissing data masking",
             facecolor="#FFF8E0", edgecolor=COLORS["accent3"], title_color=COLORS["accent3"],
             title_size=11, body_size=9)

    fig.savefig(OUTPUT_DIR / "03_model_architecture.png", dpi=150, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close(fig)
    print("  [OK] 03_model_architecture.png")


# ──────────────────────────────────────────────────────────────────────
# SLIDE 3b: Interim Classifier — What's Actually Running Today
# ──────────────────────────────────────────────────────────────────────
def slide_interim_classifier():
    fig, ax = setup_figure("Layer 5 Today — Interim Classifier (Running Now)")

    ax.add_patch(mpatches.FancyBboxPatch(
        (0.5, 7.4), 6.0, 0.5, boxstyle="round,pad=0.05",
        facecolor="#D0F0C0", edgecolor=COLORS["output"], linewidth=1.5, zorder=2,
    ))
    ax.text(3.5, 7.65, "BUILT + VALIDATED — no external dataset required",
            fontsize=10, fontweight="bold", color="#0B8457", ha="center", va="center", zorder=3)

    draw_box(ax, 0.5, 3.8, 2.4, 1.8,
             label="Synthetic Training\nSet Generator",
             body="4 classes: planet,\neclipsing binary,\nstarspot, noise\n300 labeled examples",
             facecolor=COLORS["data"], edgecolor=COLORS["data"],
             title_color="#FFFFFF", body_color="#E0E0E0", title_size=11, body_size=9)

    draw_box(ax, 3.4, 3.8, 2.6, 1.8,
             label="Layer 4 Features",
             body="17 features per\ncandidate: depth/\nconsistency, shape,\nnoise, entropy",
             facecolor="#E8F4FD", edgecolor=COLORS["accent1"], title_color=COLORS["accent1"],
             title_size=11, body_size=9)

    draw_box(ax, 6.5, 3.8, 3.0, 1.8,
             label="Random Forest\n(scikit-learn)",
             body="300 trees, max depth 6\nmedian imputation\nclass-balanced weights",
             facecolor="#F0E6FF", edgecolor=COLORS["accent5"], title_color=COLORS["accent5"],
             title_size=11, body_size=9)

    draw_box(ax, 10.0, 3.8, 2.8, 1.8,
             label="Output",
             body="Class probabilities\n+ top-5 contributing\nfeatures (explainable)",
             facecolor=COLORS["output"], edgecolor=COLORS["output"],
             title_color="#FFFFFF", body_color="#E0E0E0", title_size=11, body_size=9)

    for sx in [2.9, 6.0, 9.5]:
        draw_arrow(ax, sx, 4.7, sx + 0.4, 4.7)

    draw_box(ax, 0.5, 1.2, 6.3, 2.0,
             label="Held-out Test Accuracy (75 examples)",
             body="Overall: 70.7%\nEclipsing binary F1: 0.91\nPlanet F1: 0.76\nNoise F1: 0.63  |  Starspot F1: 0.54",
             facecolor="#FFF8E0", edgecolor=COLORS["accent3"], title_color=COLORS["accent3"],
             title_size=11, body_size=9)

    draw_box(ax, 7.2, 1.2, 6.6, 2.0,
             label="Top Contributing Features",
             body="1. Red-noise ratio    2. Transit depth (ppm)\n3. Depth SNR    4. Out-of-transit kurtosis\n5. Transit duration",
             facecolor="#FFF8E0", edgecolor=COLORS["accent3"], title_color=COLORS["accent3"],
             title_size=11, body_size=9)

    ax.text(FIG_WIDTH / 2, 0.5,
            "Runs alongside (not instead of) the rule-based vetting in vetting.py — disagreement between the two is a signal for human review.",
            fontsize=9, style="italic", color="#666666", ha="center", va="center")

    fig.savefig(OUTPUT_DIR / "09_interim_classifier.png", dpi=150, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close(fig)
    print("  [OK] 09_interim_classifier.png")


# ──────────────────────────────────────────────────────────────────────
# SLIDE 4: Training Pipeline
# ──────────────────────────────────────────────────────────────────────
def slide_training_pipeline():
    fig, ax = setup_figure("Training Pipeline — Mamba Classifier")

    y = 5.0
    bw, bh = 2.8, 2.2

    stages = [
        ("Curated Dataset", "Confirmed exoplanets\nEclipsing binaries\nFalse positives\nTESS TOI catalog\nSynthetic negatives",
         COLORS["data"], "#FFFFFF"),
        ("Preprocess & Split", "Light curve flattening\nPhase fold on known period\nTrain/val/test: 70/15/15\nStratified split\nTime-series aware",
         COLORS["process"], "#FFFFFF"),
        ("Data Augmentation", "Noise injection\nDepth scaling\nPeriod jitter\nMissing data\nClass balancing",
         COLORS["ml"], "#FFFFFF"),
        ("Model Training", "Mamba SSM encoder\nCross-entropy loss\nAdamW (lr=1e-4)\nBatch size: 64\nGradient clipping",
         COLORS["ml"], "#FFFFFF"),
        ("Evaluation", "Accuracy, Precision\nRecall, F1-score\nConfusion matrix\nROC-AUC per class\nCalibration plot",
         COLORS["output"], "#FFFFFF"),
    ]

    x_start = (FIG_WIDTH - (len(stages) * bw + (len(stages) - 1) * 0.4)) / 2

    for i, (title, body, color, txt_color) in enumerate(stages):
        x = x_start + i * (bw + 0.4)
        draw_box(ax, x, y, bw, bh,
                 label=title, body=body,
                 facecolor=color, edgecolor=color,
                 title_color=txt_color, body_color="#DDDDDD",
                 title_size=11, body_size=9)
        if i < len(stages) - 1:
            draw_arrow(ax, x + bw + 0.05, y + bh / 2, x + bw + 0.35, y + bh / 2)

    fig.savefig(OUTPUT_DIR / "04_training_pipeline.png", dpi=150, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close(fig)
    print("  [OK] 04_training_pipeline.png")


# ──────────────────────────────────────────────────────────────────────
# SLIDE 5: Evaluation & Testing Pipeline
# ──────────────────────────────────────────────────────────────────────
def slide_evaluation_pipeline():
    fig, ax = setup_figure("Evaluation & Testing Pipeline")

    # Left: Test data sources
    draw_box(ax, 0.3, 4.5, 2.5, 2.0,
             label="Test Data Sources",
             body="Synthetic injections\n(known ground truth)\nReal TESS targets\n(known planets)\nCurated test split\n(labeled negatives)",
             facecolor=COLORS["data"], edgecolor=COLORS["data"],
             title_color="#FFFFFF", body_color="#E0E0E0")

    # Center: Pipeline execution
    draw_box(ax, 3.5, 4.5, 3.0, 2.0,
             label="Pipeline Execution",
             body="Run full pipeline\non each target\nCollect predictions\nStore JSON results",
             facecolor="#E8F4FD", edgecolor=COLORS["accent1"], title_color=COLORS["accent1"])

    # Metrics columns
    metrics_w, metrics_h = 2.2, 2.0
    mx_start = 7.5
    metrics = [
        ("Detection Metrics", "Recall / TPR\nPrecision / PPV\nF1 score\nDetection rate\nFalse positive rate",
         "#F0E6FF", COLORS["accent5"]),
        ("Parameter Accuracy", "Period error (%)\nDepth error (%)\nDuration error (hr)\nT0 error (days)\nUncertainty bounds",
         "#FFF0E0", COLORS["process"]),
        ("Classification", "Confusion matrix\nPer-class F1\nROC curves\nConfidence scores\nCalibration",
         "#E8FDE0", COLORS["output"]),
    ]

    for i, (title, body, bg, ec) in enumerate(metrics):
        x = mx_start + i * (metrics_w + 0.3)
        draw_box(ax, x, 4.5, metrics_w, metrics_h,
                 label=title, body=body,
                 facecolor=bg, edgecolor=ec, title_color=ec)

    # Bottom: Report generation
    draw_box(ax, 3.5, 1.5, 6.0, 1.5,
             label="Report Generation",
             body="Summary statistics  |  Benchmark comparison table  |  ROC/PR curves  |  Error distribution plots  |  3-page PDF report",
             facecolor="#FFF8E0", edgecolor=COLORS["accent3"], title_color=COLORS["accent3"])

    # Truth comparison callout
    draw_box(ax, 12.0, 1.5, 3.5, 1.5,
             label="Validation Against Ground Truth",
             body=f"{'Known planets: TESS TOI catalog'}\n{'Synthetic: injected parameters'}\n{'Cross-validation folds (5-fold)'}",
             facecolor="#F0E6FF", edgecolor=COLORS["accent5"], title_color=COLORS["accent5"],
             title_size=11, body_size=9)

    # Arrows
    draw_arrowhead(ax, 2.8, 5.5, dx=0.7)
    draw_arrowhead(ax, 6.5, 5.5, dx=0.5)
    draw_arrowhead(ax, 5.0, 4.5, dy=-0.7)

    fig.savefig(OUTPUT_DIR / "05_evaluation_pipeline.png", dpi=150, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close(fig)
    print("  [OK] 05_evaluation_pipeline.png")


# ──────────────────────────────────────────────────────────────────────
# SLIDE 6: Data Flow Diagram
# ──────────────────────────────────────────────────────────────────────
def slide_data_flow():
    fig, ax = setup_figure("Data Flow — End-to-End Information Architecture")

    # Start with data sources at top
    source_y = 7.0
    draw_box(ax, 0.5, source_y, 3.5, 1.2,
             label="External Data Sources",
             body="MAST Archive (TESS FITS)  |  Curated Labeled Dataset  |  Synthetic Generator",
             facecolor=COLORS["data"], edgecolor=COLORS["data"],
             title_color="#FFFFFF", body_color="#E0E0E0", title_size=11)

    # Flow to storage
    draw_box(ax, 5.0, source_y, 2.5, 1.2,
             label="Data Storage",
             body="data/tess/*.csv\ndata/curated/*.csv\nLightkurve cache",
             facecolor="#E8F4FD", edgecolor=COLORS["accent1"], title_color=COLORS["accent1"],
             title_size=11, body_size=9)

    # Processing engine
    draw_box(ax, 8.5, source_y, 3.5, 1.2,
             label="Processing Engine",
             body="src/detrend.py  |  src/bls_detect.py  |  src/vetting.py",
             facecolor="#F0E6FF", edgecolor=COLORS["accent5"], title_color=COLORS["accent5"],
             title_size=11, body_size=9)

    # AI Model
    draw_box(ax, 13.0, source_y, 2.5, 1.2,
             label="AI Model",
             body="Mamba SSM\nRandom Forest\nRule Classifier",
             facecolor="#E8FDE0", edgecolor=COLORS["output"], title_color=COLORS["output"],
             title_size=11, body_size=9)

    # Second row: intermediate data
    mid_y = 5.0
    draw_box(ax, 0.5, mid_y, 3.5, 1.2,
             label="Intermediate Data",
             body="Detrended flux  |  BLS periodograms  |  Phase-folded curves  |  Vetting metrics",
             facecolor="#FFF0E0", edgecolor=COLORS["process"], title_color=COLORS["process"],
             title_size=11, body_size=9)

    # Results
    draw_box(ax, 5.0, mid_y, 3.0, 1.2,
             label="Pipeline Results",
             body="Detected candidates\nClassification labels\nParameter estimates\nSNR / confidence scores",
             facecolor="#FFF8E0", edgecolor=COLORS["accent3"], title_color=COLORS["accent3"],
             title_size=11, body_size=9)

    # Output artifacts
    draw_box(ax, 9.0, mid_y, 3.0, 1.2,
             label="Output Artifacts",
             body="JSON results files\nPNG diagnostic figures\nPhase-folded plots\nReport (3-page PDF)",
             facecolor="#D0F0C0", edgecolor=COLORS["output"], title_color=COLORS["output"],
             title_size=11, body_size=9)

    # Deployment
    draw_box(ax, 13.0, mid_y, 2.5, 1.2,
             label="Deployment",
             body="Streamlit dashboard\nBatch processing\nCLI / API",
             facecolor="#E8F4FD", edgecolor=COLORS["accent1"], title_color=COLORS["accent1"],
             title_size=11, body_size=9)

    # Bottom: Decision flow
    bot_y = 3.0
    draw_box(ax, 0.5, bot_y, 6.0, 1.2,
             label="Decision Logic — Classifier Output",
             body="ODD-EVEN mismatch > 30%?  |  Secondary eclipse > 3s?  |  V-shape ratio > 1.6?  |  Rule-based → ML hybrid",
             facecolor="#F0E6FF", edgecolor=COLORS["accent5"], title_color=COLORS["accent5"],
             title_size=11, body_size=9)

    draw_box(ax, 7.5, bot_y, 8.0, 1.2,
             label="Final Classification Categories",
             body="candidate_planetary_transit  |  likely_eclipsing_binary  |  ambiguous_requires_followup  |  noise / artifact",
             facecolor="#D0F0C0", edgecolor=COLORS["output"], title_color=COLORS["output"],
             title_size=11, body_size=9)

    # Data format annotation
    draw_box(ax, 0.5, 1.0, 15.0, 0.8,
             label="Data Formats Throughout Pipeline",
             body="FITS (raw)  ->  pandas DataFrame (time, flux, flux_err)  ->  numpy arrays  ->  BLS Periodogram (astropy)  ->  JSON dict (results)",
             facecolor="#F8F8F8", edgecolor="#AAAAAA", title_color="#333333", body_color="#666666",
             title_size=11, body_size=9)

    # Vertical arrows
    draw_arrowhead(ax, 4.25, source_y, dy=-0.8)
    draw_arrowhead(ax, 6.25, source_y, dy=-0.8)
    draw_arrowhead(ax, 10.25, source_y, dy=-0.8)
    draw_arrowhead(ax, 14.25, source_y, dy=-0.8)
    draw_arrowhead(ax, 2.25, mid_y, dy=-0.8)
    draw_arrowhead(ax, 6.5, mid_y, dy=-0.8)
    draw_arrowhead(ax, 10.5, mid_y, dy=-0.8)
    draw_arrowhead(ax, 14.25, mid_y, dy=-0.8)

    fig.savefig(OUTPUT_DIR / "06_data_flow.png", dpi=150, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close(fig)
    print("  [OK] 06_data_flow.png")


# ──────────────────────────────────────────────────────────────────────
# SLIDE 7: Technologies Used
# ──────────────────────────────────────────────────────────────────────
def slide_technologies():
    fig, ax = setup_figure("Technology Stack & Tools")

    techs = [
        ("Data Access", ["lightkurve", "astropy (FITS)", "MAST Archive", "astroquery"], COLORS["data"], "built"),
        ("Processing + ML", ["NumPy / SciPy", "pandas", "scikit-learn", "joblib"], COLORS["process"], "built"),
        ("Dashboard", ["Streamlit", "matplotlib"], COLORS["accent3"], "built"),
        ("Deep Learning", ["PyTorch", "Mamba SSM", "HuggingFace"], COLORS["ml"], "planned"),
        ("Infrastructure", ["uv (package mgr)", "Python 3.12+", "Git"], COLORS["accent1"], "built"),
    ]

    bw, bh = 2.8, 3.0
    x_start = (FIG_WIDTH - (len(techs) * bw + (len(techs) - 1) * 0.4)) / 2
    y_center = 5.1

    for i, (title, items, color, status) in enumerate(techs):
        x = x_start + i * (bw + 0.4)
        body_text = "\n".join([f"  \u2022 {item}" for item in items])
        draw_box(ax, x, y_center - bh / 2, bw, bh,
                 label=title, body=body_text,
                 facecolor=color, edgecolor=color,
                 title_color="#FFFFFF", body_color="#E0E0E0")
        badge = "BUILT" if status == "built" else "PLANNED"
        badge_color = COLORS["output"] if status == "built" else "#E8A838"
        ax.add_patch(mpatches.FancyBboxPatch(
            (x + bw / 2 - 0.6, y_center + bh / 2 + 0.1), 1.2, 0.35,
            boxstyle="round,pad=0.04", facecolor=badge_color, edgecolor=badge_color, zorder=3,
        ))
        ax.text(x + bw / 2, y_center + bh / 2 + 0.275, badge, fontsize=8, fontweight="bold",
                color="#FFFFFF", ha="center", va="center", zorder=4)

    # Footer: environment details
    draw_box(ax, 2.0, 1.0, 12.0, 1.0,
             label="Development Environment",
             body="Platform: Windows / Linux  |  Python 3.12+  |  uv package manager  |  Virtual environment: exo/",
             facecolor="#F8F8F8", edgecolor="#AAAAAA", title_color="#333333", body_color="#666666",
             title_size=11)

    fig.savefig(OUTPUT_DIR / "07_technologies.png", dpi=150, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close(fig)
    print("  [OK] 07_technologies.png")


# ──────────────────────────────────────────────────────────────────────
# SLIDE 8: UI / Dashboard Wireframe
# ──────────────────────────────────────────────────────────────────────
def slide_wireframe():
    fig, ax = setup_figure("Dashboard UI — Streamlit Application (Built, app.py)")

    # Sidebar
    side_x, side_y = 0.4, 0.6
    side_w, side_h = 3.2, 7.6
    ax.add_patch(mpatches.FancyBboxPatch(
        (side_x, side_y), side_w, side_h, boxstyle="round,pad=0.08",
        facecolor="#F0F2F6", edgecolor="#CCCCCC", linewidth=1.2, zorder=1,
    ))
    ax.text(side_x + side_w / 2, side_y + side_h - 0.4, "Input", fontsize=12, fontweight="bold",
            color="#333333", ha="center", va="center", zorder=2)
    radio_options = ["Synthetic: planet", "Synthetic: eclipsing binary",
                     "Local TESS file", "Upload CSV", "Real TIC ID (MAST)"]
    for i, opt in enumerate(radio_options):
        ry = side_y + side_h - 1.1 - i * 0.55
        ax.add_patch(plt.Circle((side_x + 0.35, ry), 0.08,
                                 facecolor="#FFFFFF" if i else COLORS["process"],
                                 edgecolor="#888888", zorder=2))
        ax.text(side_x + 0.6, ry, opt, fontsize=9, color="#333333", va="center", zorder=2)
    ax.add_patch(mpatches.FancyBboxPatch(
        (side_x + 0.3, side_y + 1.4), side_w - 0.6, 0.5, boxstyle="round,pad=0.05",
        facecolor=COLORS["process"], edgecolor=COLORS["process"], zorder=2,
    ))
    ax.text(side_x + side_w / 2, side_y + 1.65, "Run pipeline", fontsize=10, fontweight="bold",
            color="#FFFFFF", ha="center", va="center", zorder=3)

    # Main area
    main_x = side_x + side_w + 0.4
    main_w = FIG_WIDTH - main_x - 0.4

    ax.text(main_x, side_y + side_h - 0.2, "AI-Enabled Exoplanet Transit Detection",
            fontsize=13, fontweight="bold", color=COLORS["title"], va="center", zorder=2)

    # 4 metric cards
    metrics = [
        ("Rule-based classification", "candidate_planetary_transit", 6.5),
        ("Confidence", "0.98 — high confidence", 7.5),
        ("Detection SNR", "37.0", 10),
        ("Plausible?", "Yes", 10),
    ]
    card_w = (main_w - 0.6) / 4
    card_y = side_y + side_h - 1.6
    for i, (label, value, value_size) in enumerate(metrics):
        cx = main_x + i * (card_w + 0.2)
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx, card_y), card_w, 0.9, boxstyle="round,pad=0.05",
            facecolor="#FFFFFF", edgecolor="#DDDDDD", linewidth=1, zorder=2,
        ))
        ax.text(cx + 0.1, card_y + 0.65, label, fontsize=7.5, color="#666666", va="center", zorder=3)
        ax.text(cx + 0.1, card_y + 0.28, value, fontsize=value_size, fontweight="bold",
                color=COLORS["title"], va="center", zorder=3, clip_on=True)

    # Plot area (the 2x2 figure)
    plot_y = card_y - 3.0
    ax.add_patch(mpatches.FancyBboxPatch(
        (main_x, plot_y), main_w, 2.8, boxstyle="round,pad=0.08",
        facecolor="#FAFAFA", edgecolor="#DDDDDD", linewidth=1, zorder=2,
    ))
    ax.text(main_x + main_w / 2, plot_y + 1.4,
            "Raw + Detrended + BLS Periodogram + Phase-Fold\n(matplotlib figure, shared with the CLI)",
            fontsize=10, color="#AAAAAA", ha="center", va="center", zorder=3, style="italic")

    # Tabs
    tab_labels = ["Parameters", "Vetting & plausibility", "ML classifier (Layer 5)",
                  "Features (Layer 4)", "Export"]
    tab_y = plot_y - 0.6
    tab_w = main_w / len(tab_labels)
    for i, t in enumerate(tab_labels):
        tx = main_x + i * tab_w
        active = i == 2
        ax.add_patch(mpatches.FancyBboxPatch(
            (tx + 0.05, tab_y), tab_w - 0.1, 0.45, boxstyle="round,pad=0.03",
            facecolor=COLORS["ml"] if active else "#FFFFFF",
            edgecolor=COLORS["ml"] if active else "#DDDDDD", linewidth=1, zorder=2,
        ))
        ax.text(tx + tab_w / 2, tab_y + 0.225, t, fontsize=7,
                color="#FFFFFF" if active else "#666666", ha="center", va="center", zorder=3)

    # ML classifier tab content mockup
    content_y = tab_y - 1.5
    ax.add_patch(mpatches.FancyBboxPatch(
        (main_x, content_y), main_w, 1.3, boxstyle="round,pad=0.06",
        facecolor="#FFFFFF", edgecolor="#DDDDDD", linewidth=1, zorder=2,
    ))
    ax.text(main_x + 0.2, content_y + 1.0, "Predicted: planet   |   bar chart of class probabilities",
            fontsize=9, color="#333333", va="center", zorder=3)
    ax.text(main_x + 0.2, content_y + 0.55,
            "Top contributing features: red_noise_ratio, depth_ppm, depth_snr...",
            fontsize=8, color="#666666", va="center", zorder=3)
    ax.text(main_x + 0.2, content_y + 0.2,
            "⚠ trained on synthetic data only — shown alongside rule-based label, not instead of it",
            fontsize=8, color="#B36B00", va="center", zorder=3, style="italic")

    fig.savefig(OUTPUT_DIR / "08_dashboard_wireframe.png", dpi=150, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close(fig)
    print("  [OK] 08_dashboard_wireframe.png")


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────
def main():
    print("Generating pipeline diagrams...\n")
    slide_system_architecture()
    slide_detailed_pipeline()
    slide_model_architecture()
    slide_interim_classifier()
    slide_training_pipeline()
    slide_evaluation_pipeline()
    slide_data_flow()
    slide_technologies()
    slide_wireframe()
    print(f"\nAll diagrams saved to {OUTPUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()
