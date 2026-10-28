"""CIFAR-10 loader for the flat real-image ranking check (P1).

Dataset facts: https://www.cs.toronto.edu/~kriz/cifar.html
Loader: torchvision.datasets.CIFAR10

No augmentation — only ToTensor + channel normalize — so all three heads
see the same views. Quick mode uses a class-balanced subset for CPU runs.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

os.environ.setdefault("TQDM_DISABLE", "1")

# Channel means/stds commonly used with torchvision CIFAR-10.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

QUICK_TRAIN_PER_CLASS = 500
QUICK_TEST_PER_CLASS = 200


class _LabeledSubset(Dataset):
    """Subset that still exposes integer labels for MPerClassSampler."""

    def __init__(self, base: Dataset, indices: list[int]):
        self.base = base
        self.indices = list(indices)
        # Cache labels for the sampler (PML MPerClassSampler needs .labels or similar).
        self.labels = []
        for i in self.indices:
            _, y = base[i]
            self.labels.append(int(y))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.base[self.indices[idx]]


def _balanced_indices(
    targets: list[int] | np.ndarray, per_class: int, seed: int, n_classes: int = 10
) -> list[int]:
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    out: list[int] = []
    for c in range(n_classes):
        idx = np.where(targets == c)[0]
        take = min(per_class, len(idx))
        chosen = rng.choice(idx, size=take, replace=False)
        out.extend(chosen.tolist())
    rng.shuffle(out)
    return out


def load_cifar10(
    root: Path,
    full: bool = False,
    seed: int = 20260924,
) -> tuple[Dataset, Dataset, dict[str, Any]]:
    """Return (train, val/test, meta). Downloads into ``root`` if needed."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    train_full = datasets.CIFAR10(root=str(root), train=True, download=True, transform=tfm)
    test_full = datasets.CIFAR10(root=str(root), train=False, download=True, transform=tfm)

    if full:
        train_ds = _LabeledSubset(train_full, list(range(len(train_full))))
        val_ds = _LabeledSubset(test_full, list(range(len(test_full))))
        mode = "full"
    else:
        train_idx = _balanced_indices(train_full.targets, QUICK_TRAIN_PER_CLASS, seed)
        val_idx = _balanced_indices(
            test_full.targets, QUICK_TEST_PER_CLASS, seed + 1
        )
        train_ds = _LabeledSubset(train_full, train_idx)
        val_ds = _LabeledSubset(test_full, val_idx)
        mode = "quick"

    meta = {
        "regime": "cifar10",
        "n_classes": 10,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "in_channels": 3,
        "image_size": 32,
        "subset": mode,
    }
    return train_ds, val_ds, meta
