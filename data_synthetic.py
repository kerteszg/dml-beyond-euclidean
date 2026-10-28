"""Synthetic hierarchical (H) vs flat (F) class-mean datasets.

Each regime has 27 leaf classes (depth-3 ternary tree). H inherits parent
offsets along the tree; F draws independent class means. Used as a clean
control for P1/P2 before real-image checks.

Parent id for P0: immediate parent of the leaf (leaf // BRANCH). On H,
siblings share a parent; on F each leaf is its own parent (no hierarchy).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

SEED = 20260924
INPUT_DIM = 32
N_PER_LEAF = 35
BRANCH = 3
DEPTH = 3  # leaf classes = 27
NOISE = 1.05
TREE_SCALE0 = 0.85
FLAT_MEAN_STD = 0.75


@dataclass
class SyntheticData:
    name: str  # "H" or "F"
    x: torch.Tensor
    y: torch.Tensor
    n_classes: int


class FeatureDataset(Dataset):
    """Feature + leaf label + parent id (for P0 / MPerClassSampler)."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, parent_ids: torch.Tensor):
        self.x = x
        self.y = y
        self.parent_ids = parent_ids
        self.labels = y
        self.targets = y

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], int(self.y[idx])

    @property
    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Compatibility with older smoke helpers.
        return self.x, self.y


def _parent_ids_for_regime(y: torch.Tensor, regime: str) -> torch.Tensor:
    """Map leaf labels to parent ids."""
    if regime == "H":
        # Immediate parent in the ternary tree (siblings share leaf // BRANCH).
        return (y // BRANCH).long()
    # Flat: each leaf is its own parent → excl-leaf same-parent set is empty.
    return y.long()


def _n_leaves() -> int:
    return BRANCH**DEPTH


def _tree_means(n_leaves: int, dim: int, seed: int) -> np.ndarray:
    """Class means with hierarchical inheritance along a ternary tree."""
    means = np.zeros((n_leaves, dim), dtype=np.float64)
    for leaf in range(n_leaves):
        path = []
        t = leaf
        for _ in range(DEPTH):
            path.append(t % BRANCH)
            t //= BRANCH
        path = list(reversed(path))
        mu = np.zeros(dim, dtype=np.float64)
        prefix = 0
        scale = TREE_SCALE0
        for depth, choice in enumerate(path):
            prefix = prefix * BRANCH + choice
            # Deterministic per-node offsets so H structure is reproducible.
            rng_node = np.random.default_rng(seed + 1000 * depth + prefix)
            mu = mu + scale * rng_node.normal(0.0, 1.0, size=dim)
            scale *= 0.6
        means[leaf] = mu
    return means


def _flat_means(rng: np.random.Generator, n_leaves: int, dim: int) -> np.ndarray:
    return rng.normal(0.0, FLAT_MEAN_STD, size=(n_leaves, dim)).astype(np.float64)


def make_dataset(regime: str, seed: int = SEED) -> SyntheticData:
    assert regime in {"H", "F"}
    rng = np.random.default_rng(seed)
    n_leaves = _n_leaves()
    means = (
        _tree_means(n_leaves, INPUT_DIM, seed)
        if regime == "H"
        else _flat_means(rng, n_leaves, INPUT_DIM)
    )
    xs, ys = [], []
    for c in range(n_leaves):
        pts = means[c] + rng.normal(0.0, NOISE, size=(N_PER_LEAF, INPUT_DIM))
        xs.append(pts)
        ys.append(np.full(N_PER_LEAF, c, dtype=np.int64))
    x = np.vstack(xs).astype(np.float32)
    y = np.concatenate(ys)
    perm = rng.permutation(len(y))
    x, y = x[perm], y[perm]
    return SyntheticData(
        name=regime,
        x=torch.from_numpy(x),
        y=torch.from_numpy(y),
        n_classes=n_leaves,
    )


def load_synthetic(
    regime: str, seed: int = SEED, val_frac: float = 0.25
) -> tuple[FeatureDataset, FeatureDataset, dict[str, Any]]:
    """Return (train, val, meta) for one regime."""
    data = make_dataset(regime, seed=seed)
    parent_all = _parent_ids_for_regime(data.y, regime)
    rng = np.random.default_rng(seed + (0 if regime == "H" else 1))
    n = len(data.y)
    idx = rng.permutation(n)
    n_val = int(n * val_frac)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train = FeatureDataset(
        data.x[train_idx], data.y[train_idx], parent_all[train_idx]
    )
    val = FeatureDataset(data.x[val_idx], data.y[val_idx], parent_all[val_idx])
    meta = {
        "regime": regime,
        "n_classes": data.n_classes,
        "n_train": len(train),
        "n_val": len(val),
        "input_dim": INPUT_DIM,
        "in_channels": None,
        "siblings_per_parent": BRANCH if regime == "H" else 1,
        "hierarchy_kind": "tree_parent" if regime == "H" else "flat_singleton",
    }
    return train, val, meta
