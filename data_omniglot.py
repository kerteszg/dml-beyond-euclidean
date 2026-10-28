"""Omniglot matched hierarchical (H) vs flat (F) arms.

Dataset release: https://github.com/brendenlake/omniglot
Loader layout: torchvision/datasets/omniglot.py
  (alphabet / character folders under images_background / images_evaluation)

Arm construction (this repo):
  quick — background split only
    H: 3 alphabets x 9 characters = 27 leaf classes
    F: 27 characters from 27 distinct alphabets
  full — background + evaluation alphabets (more hierarchy coverage)
    H: 5 alphabets x 9 characters = 45 leaf classes
    F: 45 characters from 45 distinct alphabets

Positives in the loss are *character identity* in both arms.
Epoch budget is set by run.py (matched across quick/full for Omniglot);
--full here means a larger class set, not merely more epochs.

Images are resized to 28x28 here (implementation choice; official pixel size
is not asserted). Split by drawer: 15 train / 5 val drawings per character.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

# Quiet torchvision download progress bars in logs.
os.environ.setdefault("TQDM_DISABLE", "1")

N_CHARS_PER_ALPHA_H = 9
N_TRAIN_DRAWERS = 15
N_VAL_DRAWERS = 5
IMAGE_SIZE = 28

# quick vs full hierarchy coverage
QUICK_N_ALPHABETS_H = 3
FULL_N_ALPHABETS_H = 5


class OmniglotArm(Dataset):
    """Tensor dataset with alphabet / parent ids for P0 diagnostics."""

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        alphabet_ids: torch.Tensor,
    ):
        self.images = images
        self.labels = labels
        self.alphabet_ids = alphabet_ids
        # Alias: alphabet is the parent node for the hierarchy-in-embedding check.
        self.parent_ids = alphabet_ids
        self.targets = labels

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        return self.images[idx], int(self.labels[idx])


def _arm_sizes(full: bool) -> tuple[int, int]:
    """Return (n_alphabets_H, n_classes) for the requested tier."""
    n_alpha = FULL_N_ALPHABETS_H if full else QUICK_N_ALPHABETS_H
    return n_alpha, n_alpha * N_CHARS_PER_ALPHA_H


def _build_metadata(ds: datasets.Omniglot) -> list[dict[str, Any]]:
    """Public (path, alphabet, character, target) table.

    torchvision Omniglot.__getitem__ returns only (image, character_index).
    Alphabet is the first path component of the character folder name —
    see torchvision/datasets/omniglot.py (_characters, _flat_character_images).
    """
    rows = []
    flat = getattr(ds, "_flat_character_images", None)
    if flat is None:
        flat = [pair for group in ds._character_images for pair in group]
    for img_name, target in flat:
        char_folder = ds._characters[int(target)]
        parts = char_folder.replace("\\", "/").split("/")
        alphabet = parts[0]
        character = parts[1] if len(parts) > 1 else parts[0]
        rel = str(Path(char_folder) / img_name)
        rows.append(
            {
                "path": rel,
                "alphabet": alphabet,
                "character": character,
                "character_idx": int(target),
                "target": int(target),
                "target_folder": str(Path(ds.target_folder)),
            }
        )
    return rows


def _load_image_tensor(abs_path: Path, tfm) -> torch.Tensor:
    from PIL import Image

    img = Image.open(abs_path).convert("L")
    return tfm(img)


def _select_arm_classes(
    meta: list[dict[str, Any]],
    regime: str,
    seed: int,
    n_alphabets_h: int,
    n_classes: int,
) -> list[tuple[str, str]]:
    """Return list of (alphabet, character) pairs for the arm."""
    rng = np.random.default_rng(seed)
    by_alpha: dict[str, list[str]] = {}
    for row in meta:
        by_alpha.setdefault(row["alphabet"], [])
        if row["character"] not in by_alpha[row["alphabet"]]:
            by_alpha[row["alphabet"]].append(row["character"])

    alphabets = sorted(by_alpha.keys())
    if regime == "H":
        eligible = [a for a in alphabets if len(by_alpha[a]) >= N_CHARS_PER_ALPHA_H]
        if len(eligible) < n_alphabets_h:
            raise RuntimeError(
                f"Need {n_alphabets_h} alphabets with ≥{N_CHARS_PER_ALPHA_H} characters; "
                f"found {len(eligible)} (tier needs background+evaluation for --full)"
            )
        chosen_alphas = list(rng.choice(eligible, size=n_alphabets_h, replace=False))
        pairs: list[tuple[str, str]] = []
        for a in chosen_alphas:
            chars = list(rng.choice(by_alpha[a], size=N_CHARS_PER_ALPHA_H, replace=False))
            for ch in chars:
                pairs.append((a, ch))
        return pairs

    if regime == "F":
        eligible = [a for a in alphabets if len(by_alpha[a]) >= 1]
        if len(eligible) < n_classes:
            raise RuntimeError(
                f"Need {n_classes} alphabets for flat arm; found {len(eligible)} "
                f"(enable --full to merge evaluation alphabets)"
            )
        chosen_alphas = list(rng.choice(eligible, size=n_classes, replace=False))
        pairs = []
        for a in chosen_alphas:
            ch = str(rng.choice(by_alpha[a]))
            pairs.append((a, ch))
        return pairs

    raise ValueError(regime)


def load_omniglot(
    root: Path,
    regime: str,
    seed: int = 20260924,
    full: bool = False,
) -> tuple[OmniglotArm, OmniglotArm, dict[str, Any]]:
    """Build one Omniglot arm (H or F).

    ``full`` enlarges the matched arms (5x9 vs 3x9) by merging the official
    evaluation alphabets into the pool. Drawer split stays 15/5.
    """
    assert regime in {"H", "F"}
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    n_alphabets_h, n_classes = _arm_sizes(full)

    # Background always; evaluation only for --full (need ≥45 alphabets for F).
    splits = [True] if not full else [True, False]
    meta_rows: list[dict[str, Any]] = []
    for background in splits:
        raw = datasets.Omniglot(
            root=str(root),
            background=background,
            download=True,
            transform=None,
        )
        meta_rows.extend(_build_metadata(raw))

    pairs = _select_arm_classes(
        meta_rows,
        regime,
        seed=seed + (0 if regime == "H" else 17),
        n_alphabets_h=n_alphabets_h,
        n_classes=n_classes,
    )

    pair_to_label = {p: i for i, p in enumerate(pairs)}
    alphabet_names = sorted({a for a, _ in pairs})
    alpha_to_id = {a: i for i, a in enumerate(alphabet_names)}

    tfm = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.922,), (0.268,)),
        ]
    )

    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {p: [] for p in pairs}
    for row in meta_rows:
        key = (row["alphabet"], row["character"])
        if key in by_pair:
            by_pair[key].append(row)

    rng = np.random.default_rng(seed + 100)
    train_imgs, train_y, train_a = [], [], []
    val_imgs, val_y, val_a = [], [], []

    for pair, rows in by_pair.items():
        if len(rows) < N_TRAIN_DRAWERS + N_VAL_DRAWERS:
            raise RuntimeError(
                f"{pair}: expected ≥{N_TRAIN_DRAWERS + N_VAL_DRAWERS} drawings, got {len(rows)}"
            )
        order = rng.permutation(len(rows))
        train_rows = [rows[i] for i in order[:N_TRAIN_DRAWERS]]
        val_rows = [rows[i] for i in order[N_TRAIN_DRAWERS : N_TRAIN_DRAWERS + N_VAL_DRAWERS]]
        label = pair_to_label[pair]
        aid = alpha_to_id[pair[0]]
        for r in train_rows:
            abs_path = Path(r["target_folder"]) / r["path"]
            train_imgs.append(_load_image_tensor(abs_path, tfm))
            train_y.append(label)
            train_a.append(aid)
        for r in val_rows:
            abs_path = Path(r["target_folder"]) / r["path"]
            val_imgs.append(_load_image_tensor(abs_path, tfm))
            val_y.append(label)
            val_a.append(aid)

    train_ds = OmniglotArm(
        torch.stack(train_imgs),
        torch.tensor(train_y, dtype=torch.long),
        torch.tensor(train_a, dtype=torch.long),
    )
    val_ds = OmniglotArm(
        torch.stack(val_imgs),
        torch.tensor(val_y, dtype=torch.long),
        torch.tensor(val_a, dtype=torch.long),
    )
    # On H, siblings_per_parent = chars per alphabet; on F each alphabet has 1 char.
    siblings = N_CHARS_PER_ALPHA_H if regime == "H" else 1
    meta = {
        "regime": regime,
        "n_classes": n_classes,
        "n_alphabets_h": n_alphabets_h if regime == "H" else n_classes,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "in_channels": 1,
        "image_size": IMAGE_SIZE,
        "subset": "full" if full else "quick",
        "splits_used": ["background", "evaluation"] if full else ["background"],
        "siblings_per_parent": siblings,
        "hierarchy_kind": "alphabet" if regime == "H" else "flat_singleton",
        "alphabets": alphabet_names,
        "pairs": [{"alphabet": a, "character": c} for a, c in pairs],
    }
    return train_ds, val_ds, meta
