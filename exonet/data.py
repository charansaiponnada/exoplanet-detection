"""Dataset, scalar feature conditioning, and leakage-free splits."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from torch.utils.data import Dataset

from exonet.model import VIEW_TABLE

# Features whose dynamic range spans orders of magnitude are log-compressed
# before standardisation, otherwise a handful of long-period or high-SNR TCEs
# dominate the scale.
LOG_FEATURES = {
    "tce_period", "tce_duration", "tce_depth", "tce_model_snr", "tce_max_mult_ev",
    "tce_num_transits", "tce_ror", "tce_dor", "tce_prad", "tce_robstat",
    "tce_dikco_msky", "tce_dicco_msky", "bkspace", "n_cadences",
}

# Quantities that live near 1e-4 in relative-flux units.  signed log1p is a
# no-op at that scale, so these take a plain base-10 logarithm instead.
LOG10_SMALL = {"view_depth", "oot_scatter", "local_scale"}


class ScalarConditioner:
    """Log-compress, impute and standardise scalar features using train statistics."""

    def __init__(self, names: list[str]):
        self.names = names
        self.log_mask = np.array([n in LOG_FEATURES for n in names])
        self.log10_mask = np.array([n in LOG10_SMALL for n in names])
        self.median_ = None
        self.mean_ = None
        self.std_ = None

    def _compress(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float64).copy()
        m = self.log_mask
        # signed log keeps the sign of quantities that can be negative
        x[:, m] = np.sign(x[:, m]) * np.log10(1.0 + np.abs(x[:, m]))
        s = self.log10_mask
        if s.any():
            x[:, s] = np.log10(np.clip(np.abs(x[:, s]), 1e-8, None))
        return x

    def fit(self, x: np.ndarray) -> "ScalarConditioner":
        x = self._compress(x)
        self.median_ = np.nanmedian(x, axis=0)
        self.median_ = np.where(np.isfinite(self.median_), self.median_, 0.0)
        filled = np.where(np.isfinite(x), x, self.median_)
        self.mean_ = filled.mean(axis=0)
        self.std_ = filled.std(axis=0)
        self.std_ = np.where(self.std_ > 1e-8, self.std_, 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = self._compress(x)
        filled = np.where(np.isfinite(x), x, self.median_)
        z = (filled - self.mean_) / self.std_
        return np.clip(z, -10, 10).astype(np.float32)


class TCEDataset(Dataset):
    """In-memory dataset of phase-folded views plus conditioned scalars.

    The full processed set is a few hundred megabytes, so it is held in RAM;
    this removes the data loader as a bottleneck and makes epochs GPU-bound.
    """

    def __init__(
        self,
        views: dict[str, np.ndarray],
        scalars: np.ndarray,
        labels: np.ndarray,
        targets: np.ndarray,
        augment: bool = False,
    ):
        self.views = views
        self.scalars = scalars
        self.labels = labels.astype(np.float32)
        self.targets = targets.astype(np.float32)  # regression targets
        self.augment = augment
        self.view_names = list(views.keys())

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        v = {n: self.views[n][i] for n in self.view_names}
        if self.augment and np.random.rand() < 0.5:
            # A transit is symmetric about mid-transit, so mirroring every view
            # about its centre produces a physically valid alternative sample.
            v = {n: a[::-1].copy() for n, a in v.items()}
        return (
            {n: torch.from_numpy(a.copy()) for n, a in v.items()},
            torch.from_numpy(self.scalars[i]),
            torch.tensor(self.labels[i]),
            torch.from_numpy(self.targets[i]),
        )


def make_splits(
    kepids: np.ndarray,
    n_total: int,
    mode: str = "group",
    seed: int = 0,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
):
    """Train/validation/test indices.

    ``mode="group"`` keeps every TCE of a given star in a single fold.  Multiple
    TCEs on the same target share the same light curve, stellar properties and
    systematics, so a TCE-level random split leaks information between folds and
    inflates test scores.  ``mode="tce"`` reproduces the TCE-level random split
    used by Shallue & Vanderburg (2018) for comparability with published numbers.
    """
    idx = np.arange(n_total)
    if mode == "group":
        outer = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        rest, test = next(outer.split(idx, groups=kepids))
        inner = GroupShuffleSplit(
            n_splits=1, test_size=val_frac / (1 - test_frac), random_state=seed
        )
        tr, va = next(inner.split(rest, groups=kepids[rest]))
        return rest[tr], rest[va], test
    elif mode == "tce":
        outer = ShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        rest, test = next(outer.split(idx))
        inner = ShuffleSplit(
            n_splits=1, test_size=val_frac / (1 - test_frac), random_state=seed
        )
        tr, va = next(inner.split(rest))
        return rest[tr], rest[va], test
    raise ValueError(f"unknown split mode: {mode}")


def load_h5(path: str, view_names: list[str] | None = None):
    """Read the processed HDF5 file into numpy arrays."""
    import h5py

    view_names = view_names or list(VIEW_TABLE.keys())
    with h5py.File(path, "r") as fh:
        views = {n: fh[n][:].astype(np.float32) for n in view_names}
        catalog = fh["catalog"][:].astype(np.float32)
        derived = fh["derived"][:].astype(np.float32)
        kepid = fh["kepid"][:]
        plnt = fh["tce_plnt_num"][:]
        label = np.array([s.decode() for s in fh["label"][:]])
        cat_cols = list(fh.attrs["catalog_cols"])
        der_cols = list(fh.attrs["derived_cols"])
        groups = {k: list(fh.attrs[k]) for k in
                  ("transit_cols", "stellar_cols", "dv_diag_cols")}
    return dict(
        views=views, catalog=catalog, derived=derived, kepid=kepid, plnt=plnt,
        label=label, cat_cols=cat_cols, der_cols=der_cols, groups=groups,
    )
