# AI-Enabled Exoplanet Transit Detection Pipeline
### Bharatiya Antariksh Hackathon 2026 — Challenge 07

---

## Pipeline Overview

```
TESS Light Curves (MAST archive / curated dataset)
           │
           ▼
   ┌──────────────────┐
   │  Sigma Clipping  │  Remove cosmic ray / jitter outliers (5σ)
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐
   │   SG Detrending  │  Savitzky-Golay filter removes stellar variability
   └────────┬─────────┘  and instrumental systematics
            │
            ▼
   ┌──────────────────┐
   │  BLS Period Srch │  Box Least Squares: find best period/duration/depth
   └────────┬─────────┘  across 5000 trial periods (0.5–15 days)
            │
            ▼
   ┌──────────────────┐
   │   Phase Folding  │  Stack all transits on the recovered period
   └────────┬─────────┘
            │
            ▼
   ┌─────────────────────────────────┐
   │   Classical Vetting (3 tests)   │
   │  • Odd-even depth test          │  False positive: diluted EB
   │  • Secondary eclipse search     │  False positive: self-luminous binary
   │  • V-shape / U-shape ratio      │  False positive: grazing EB
   └────────┬────────────────────────┘
            │
            ▼
   ┌──────────────────┐
   │  Classification  │  candidate_planetary_transit /
   └────────┬─────────┘  likely_eclipsing_binary / ambiguous
            │
            ▼
   ┌───────────────────────────────┐
   │ Feature Engineering (Layer 4) │  depth/duration consistency, shape,
   └────────┬──────────────────────┘  skewness/kurtosis/entropy, red noise
            │
            ▼
   ┌───────────────────────────────┐
   │ Scientific Validation (Layer 6)│  physical plausibility checks +
   └────────┬──────────────────────┘  rule-based confidence score
            │
            ▼
   ┌──────────────────┐
   │ Parameter Report │  Period, depth, duration, detection SNR,
   └──────────────────┘  confidence score + verdict
```

---

## Validated Results (Synthetic Benchmark)

### Case 1 — Injected Planetary Transit

| Parameter        | Ground Truth | Recovered   | Error   |
|------------------|-------------|-------------|---------|
| Period (days)    | 3.5000      | 3.4992      | 0.023%  |
| Duration (hours) | 2.50        | 2.45        | 2.0%    |
| Depth (ppm)      | 2500        | 1500*       | ~40%†   |
| Detection SNR    | —           | 37.0        | —       |
| Classification   | Planet      | `candidate_planetary_transit` | ✅ |

†BLS systematically underestimates transit depth due to discrete period/duration
grid quantization. Planned fix: refine with `batman` trapezoidal model fit on the
phase-folded curve, recovering depth to <5% error. This is standard practice in
operational pipelines (e.g. TESS SPOC DV reports).

### Case 2 — Injected Eclipsing Binary (False Positive)

| Vetting Test         | Result                                      |
|----------------------|---------------------------------------------|
| Odd-even depth test  | 102% mismatch → flagged ✅                   |
| Secondary eclipse    | Not fired (half-period alias effect)*        |
| Shape test           | Shape ratio 1.0 (U-shaped dip)             |
| Classification       | `ambiguous_requires_followup` ✅             |

*BLS finds the half-period alias when two dips exist per orbit. At this alias,
both dips appear as "primary" events; the secondary eclipse test doesn't fire.
Planned fix: after BLS, check double-period hypothesis explicitly. Known
limitation, planned for Phase 2.

---

## Feature Engineering & Confidence Scoring (Layers 4 + 6)

Beyond the three vetting tests, `src/features.py` computes a flat feature
vector per candidate — per-transit depth consistency, ingress/egress
symmetry and slope, skewness/kurtosis/entropy of the out-of-transit
residuals, and a CDPP-like red-noise ratio. `src/scoring.py` turns those
features plus the vetting flags into a physical-plausibility check (e.g.
duration/period duty cycle, depth > 10%, transits observed) and a single
auditable 0–1 confidence score with a verdict:
`high_confidence_candidate` / `requires_human_review` /
`low_confidence_likely_false_positive`.

This is deliberately a transparent rule-based scorer, not a trained model —
every term in the score breakdown is one inspectable number, satisfying the
PS's explainability and confidence-level requirements ahead of the labeled
dataset needed to train the Layer 5 classifier below. The function
signatures (`candidate + features + classification in, score + breakdown
out`) are the intended interface for the future learned uncertainty head,
so swapping in a trained model later won't require touching `main.py`.

Results on real TESS targets (`uv run src/main.py --csv data/tess/<file>.csv`):

| Target                          | Classification              | Confidence | Verdict |
|----------------------------------|------------------------------|-----------|---------|
| TIC 25155310 (TOI-270, sector 01) | candidate_planetary_transit | 0.98      | high_confidence_candidate |
| TIC 307210830 (Pi Mensae, sector 02) | ambiguous_requires_followup | 0.79 | requires_human_review (short/V-shaped transit from 1 sector) |
| TIC 150428135 (sector 01)        | ambiguous_requires_followup  | 0.02      | low_confidence_likely_false_positive |

---

## Planned Enhancement (30-Hour Finale)

Once ISRO's curated labeled dataset (confirmed planets / eclipsing binaries /
false positives) is provided at the induction session, the classical vetting
layer will be extended with a **Mamba (State Space Model) sequence classifier**:

- Mamba handles long-range dependencies in light curves more efficiently than
  LSTMs (linear vs. quadratic complexity), making it well-suited to the noisy,
  long-baseline TESS sequences.
- Architecture: bidirectional Mamba encoder → classification head (6 classes:
  planet transit / eclipsing binary / blend / starspot / noise / unknown).
- Training: curated ISRO dataset + TESS TOI catalog positives + synthetic
  negatives generated by the existing injection pipeline in `data_io.py`.

---

## Tech Stack

| Component         | Tool                              |
|-------------------|-----------------------------------|
| Data access       | `lightkurve` + MAST archive       |
| Detrending        | `scipy.signal.savgol_filter`      |
| Period search     | `astropy.timeseries.BoxLeastSquares` |
| Vetting           | Custom classical tests (this repo)|
| Feature engineering | Custom (`scipy.stats` skew/kurtosis, this repo) |
| Confidence scoring | Custom rule-based scorer (this repo) |
| Deep learning     | PyTorch + Mamba (Phase 2)         |
| Parameter fitting | `batman` (Phase 2 refinement)     |
| Visualization     | `matplotlib` / Streamlit dashboard|
| Environment       | `uv` (Python package manager)     |

---

## Project Structure

```
exoplanet-pipeline/
├── src/
│   ├── data_io.py       # TESS download + synthetic injection
│   ├── detrend.py       # Savitzky-Golay flattening + sigma clip
│   ├── bls_detect.py    # BLS period search + phase-folding
│   ├── vetting.py       # Odd-even / secondary / shape tests + classifier
│   ├── features.py      # Layer 4: per-candidate feature engineering
│   ├── scoring.py        # Layer 6: plausibility checks + confidence score
│   └── main.py          # End-to-end pipeline runner
├── helper_scripts/      # Standalone utility scripts (see Conventions below)
│   └── download/        #   Data download helpers
├── output/              # Generated figures + JSON results
├── pyproject.toml       # uv-managed dependencies
└── README.md
```

## Conventions

### Helper scripts

All standalone utility scripts live under `helper_scripts/`, organized by
purpose:

| Subdirectory      | Purpose                                      |
|-------------------|----------------------------------------------|
| `download/`       | Data download & ingestion helpers            |
| `visualize/`      | Plotting & diagnostic visualizations         |
| `analysis/`       | Post-processing, metric computation, reports |

Scripts in `helper_scripts/` are run directly with `python`, not via `uv run`.
They may depend on the same environment (`uv sync` installs all deps) but are
kept separate from the core pipeline code in `src/` to avoid import coupling.

To add a new helper, create the appropriate subdirectory under
`helper_scripts/` and place a self-contained Python script there.

## Running

```bash
# Setup
uv sync

# Synthetic validation (no internet needed)
uv run src/main.py --synthetic planet
uv run src/main.py --synthetic eb

# Real TESS data (requires MAST access — run locally or in Colab)
uv run src/main.py --real "TIC 307210830"   # Pi Mensae b: confirmed hot Jupiter
uv run src/main.py --real "TIC 25155310"    # TOI-270: multi-planet system
```
