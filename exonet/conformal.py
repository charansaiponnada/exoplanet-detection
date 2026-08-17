"""Distribution-free confidence for the candidate list.

A vetting network emits a score, but a score is not a guarantee.  What an
observing programme actually needs to know is: *if I follow up every target on
this list, what fraction of my telescope time will be wasted on false
positives?*  Conformal inference answers exactly that question, under only the
assumption that calibration and test data are exchangeable - no assumption that
the network is well specified, well calibrated, or even good.

The construction treats "this TCE is not a planet" as the null hypothesis.
Conformal p-values are computed against the empirical distribution of scores of
*known non-planets* in a held-out calibration split, and the Benjamini-Hochberg
procedure then selects a candidate list whose false discovery rate is controlled
at a chosen level.  Because the calibration negatives are only used through
their ranks, the guarantee is finite-sample and distribution-free.

Reference: Bates, Candes, Lei, Romano & Sesia (2023), "Testing for outliers with
conformal p-values", Annals of Statistics 51(1).
"""

from __future__ import annotations

import numpy as np


def conformal_pvalues(cal_neg_scores: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    """Marginal conformal p-values for the null "this object is a non-planet".

    ``p_j = (1 + #{i : s_i >= s_j}) / (n + 1)`` where ``s_i`` ranges over
    calibration negatives.  Small p means the score is too planet-like to be
    consistent with the non-planet population.
    """
    cal = np.sort(np.asarray(cal_neg_scores, dtype=np.float64))
    n = cal.size
    # number of calibration scores >= each test score
    ge = n - np.searchsorted(cal, np.asarray(test_scores, dtype=np.float64), side="left")
    return (1.0 + ge) / (n + 1.0)


def benjamini_hochberg(pvals: np.ndarray, q: float) -> np.ndarray:
    """Boolean mask of hypotheses rejected by BH at level ``q``."""
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    sorted_p = p[order]
    thresh = q * np.arange(1, m + 1) / m
    below = np.where(sorted_p <= thresh)[0]
    keep = np.zeros(m, dtype=bool)
    if below.size:
        cutoff = below[-1]
        keep[order[: cutoff + 1]] = True
    return keep


def fdr_controlled_selection(
    cal_neg_scores: np.ndarray, test_scores: np.ndarray, q: float
) -> tuple[np.ndarray, np.ndarray]:
    """Select a candidate list with false discovery rate controlled at ``q``."""
    p = conformal_pvalues(cal_neg_scores, test_scores)
    return benjamini_hochberg(p, q), p


def empirical_fdr(selected: np.ndarray, labels: np.ndarray) -> float:
    """Realised false discovery proportion of a selection (0 if nothing selected)."""
    n_sel = int(selected.sum())
    if n_sel == 0:
        return 0.0
    return float((labels[selected] == 0).sum() / n_sel)


def expected_calibration_error(
    labels: np.ndarray, scores: np.ndarray, n_bins: int = 15
) -> float:
    """Standard ECE with equal-width bins over the predicted probability."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(scores, edges) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += m.mean() * abs(labels[m].mean() - scores[m].mean())
    return float(ece)


def fit_temperature(labels: np.ndarray, scores: np.ndarray) -> float:
    """Fit a single temperature on logits by minimising NLL on a held-out split."""
    eps = 1e-6
    s = np.clip(scores, eps, 1 - eps)
    logits = np.log(s / (1 - s))
    best_t, best_nll = 1.0, np.inf
    for t in np.linspace(0.25, 4.0, 400):
        p = 1.0 / (1.0 + np.exp(-logits / t))
        p = np.clip(p, eps, 1 - eps)
        nll = -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


def apply_temperature(scores: np.ndarray, t: float) -> np.ndarray:
    eps = 1e-6
    s = np.clip(scores, eps, 1 - eps)
    return 1.0 / (1.0 + np.exp(-(np.log(s / (1 - s))) / t))
