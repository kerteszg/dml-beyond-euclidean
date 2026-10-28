"""Train one geometry head with matched supervised InfoNCE.

Sampler: pytorch-metric-learning MPerClassSampler
  https://github.com/KevinMusgrave/pytorch-metric-learning
  (ensures every anchor has same-class positives in-batch)

Loss: losses.make_loss -> SupConLoss with head-specific distance.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_metric_learning.samplers import MPerClassSampler

from geometry import embed_for_head
from losses import make_loss


@dataclass
class TrainConfig:
    tau: float = 0.1
    lr: float = 1e-3
    batch_size: int = 128
    epochs: int = 45
    m_per_class: int = 4  # samples per class in each batch
    seed: int = 20260924
    device: str = "cpu"
    num_workers: int = 0  # raise on multi-core VMs (DataLoader workers)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _labels_from_dataset(ds: Dataset) -> list[int]:
    """Collect integer labels for MPerClassSampler."""
    if hasattr(ds, "labels"):
        return [int(y) for y in ds.labels]
    if hasattr(ds, "targets"):
        t = ds.targets
        if torch.is_tensor(t):
            return [int(x) for x in t.tolist()]
        return [int(x) for x in t]
    # Fallback: iterate (fine for small MWE sets).
    return [int(ds[i][1]) for i in range(len(ds))]


def train_head(
    train_ds: Dataset,
    encoder_fn: Callable[[], torch.nn.Module],
    head: str,
    cfg: TrainConfig,
) -> torch.nn.Module:
    """Fit one encoder under ``head`` geometry; return the best-loss state."""
    set_seed(cfg.seed)
    device = cfg.device
    model = encoder_fn().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = make_loss(head, tau=cfg.tau)

    labels = _labels_from_dataset(train_ds)
    n_classes = len(set(labels))
    # MPerClassSampler requires batch_size <= m * n_classes (and divisible by m).
    # Cap so small class counts (e.g. CIFAR-10 with 10 classes) still work.
    max_bs = cfg.m_per_class * n_classes
    batch_size = min(cfg.batch_size, max_bs)
    batch_size = batch_size - (batch_size % cfg.m_per_class)
    if batch_size < cfg.m_per_class:
        raise ValueError(
            f"batch_size collapsed to {batch_size}; need more classes or smaller m"
        )
    sampler = MPerClassSampler(
        labels=labels,
        m=cfg.m_per_class,
        batch_size=batch_size,
        length_before_new_iter=max(len(train_ds), batch_size),
    )
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )

    model.train()
    best_state = None
    best_loss = float("inf")

    for _ in range(cfg.epochs):
        total, n = 0.0, 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            z_amb = model(xb)
            # Map ambient features into the head space *before* the distance.
            z = embed_for_head(z_amb, head)
            loss = criterion(z, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            n += 1
        avg = total / max(n, 1)
        if avg < best_loss:
            best_loss = avg
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model
