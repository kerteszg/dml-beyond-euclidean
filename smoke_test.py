"""Fast smoke checks — no dataset downloads, no full training.

Run before a long VM job:
  python smoke_test.py

Exits non-zero on the first failure.
"""
from __future__ import annotations

import sys

import numpy as np
import torch
from pytorch_metric_learning.distances import LpDistance

from geometry import (
    embed_for_head,
    expmap0,
    pairwise_arccos,
    pairwise_cosine_distance,
    poincare_distance_matrix,
    project_ball,
)
from losses import make_loss
from metrics import (
    _spearman,
    evaluate_hierarchy_p0,
    parent_hit_at_k_excl_leaf,
    parent_hit_chance,
)
from models import MLPEncoder, SmallConvEncoder
from train import TrainConfig, train_head


def _ok(name: str) -> None:
    print(f"  OK  {name}")


def test_geometry() -> None:
    z0 = expmap0(torch.zeros(4, 8))
    assert torch.allclose(z0, torch.zeros_like(z0)), "expmap0(0) must be 0"

    x = project_ball(0.3 * torch.randn(5, 8))
    d = poincare_distance_matrix(x)
    assert torch.allclose(torch.diag(d), torch.zeros(5), atol=1e-5), "d(x,x)=0"

    u = torch.nn.functional.normalize(torch.randn(64, 16), dim=-1)
    dc = pairwise_cosine_distance(u).numpy()
    da = pairwise_arccos(u).numpy()
    iu = np.triu_indices(64, 1)
    s = _spearman(dc[iu], da[iu])
    assert s > 0.999, f"cos vs arccos spearman={s}"

    for head in ("euclid", "sph", "hyp"):
        z = embed_for_head(torch.randn(3, 8), head)
        assert z.shape == (3, 8), head
        if head == "sph":
            norms = torch.linalg.vector_norm(z, dim=-1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        if head == "hyp":
            max_r = (1.0 - 1e-5)
            assert (torch.linalg.vector_norm(z, dim=-1) < max_r).all()

    _ok("geometry")


def test_lp_normalize_flag() -> None:
    z = torch.randn(8, 4) * 3
    d_raw = LpDistance(normalize_embeddings=False)(z)
    d_n = LpDistance(normalize_embeddings=True)(z)
    assert not torch.allclose(d_raw, d_n), "normalize_embeddings must change LpDistance"
    _ok("LpDistance normalize flag")


def test_losses_forward() -> None:
    """One forward of SupConLoss per head on a tiny labeled batch."""
    b, d = 16, 8
    # 4 classes x 4 samples so every anchor has in-batch positives.
    y = torch.arange(4).repeat_interleave(4)
    z_amb = torch.randn(b, d)
    for head in ("euclid", "sph", "hyp"):
        z = embed_for_head(z_amb, head)
        loss = make_loss(head, tau=0.1)(z, y)
        assert torch.isfinite(loss), head
        assert loss.ndim == 0, head
    _ok("SupConLoss forward (3 heads)")


def test_encoders() -> None:
    mlp = MLPEncoder()
    assert mlp(torch.randn(2, 32)).shape == (2, 16)
    cnn_rgb = SmallConvEncoder(in_channels=3, embed_dim=32)
    assert cnn_rgb(torch.randn(2, 3, 32, 32)).shape == (2, 32)
    cnn_gray = SmallConvEncoder(in_channels=1, embed_dim=32)
    assert cnn_gray(torch.randn(2, 1, 28, 28)).shape == (2, 32)
    _ok("encoders")


def test_p0_parent_hit() -> None:
    """P0 chance baseline + excl-leaf parent hit on a toy clustered embedding."""
    # 3 parents × 3 leaves; chance = 2/8 = 0.25
    assert abs(parent_hit_chance(9, 3) - 0.25) < 1e-9

    # One sample per leaf; embeddings = one-hot of parent → same-parent neighbors nearest.
    y = torch.arange(9)
    parent = y // 3
    z = torch.nn.functional.one_hot(parent, num_classes=3).float()
    z = z + 0.01 * torch.randn_like(z)  # tiny noise so distances differ
    stats = parent_hit_at_k_excl_leaf(z, y, parent, head="euclid", k=2)
    assert stats["parent_hit@5_excl_leaf"] > 0.9, stats

    p0 = evaluate_hierarchy_p0(z, y, parent, head="euclid", n_classes=9, siblings_per_parent=3)
    assert p0["p0_holds"] is True, p0

    # Flat singleton parents: chance=0, hit should be ~0.
    parent_flat = y.clone()
    assert parent_hit_chance(9, 1) == 0.0
    stats_f = parent_hit_at_k_excl_leaf(z, y, parent_flat, head="euclid", k=2)
    assert stats_f["parent_hit@5_excl_leaf"] < 0.05, stats_f
    _ok("P0 parent_hit helpers")


def test_synthetic_one_epoch() -> None:
    """Tiny end-to-end: load synthetic H, train sph head for 1 epoch."""
    from data_synthetic import FeatureDataset, load_synthetic

    train_ds, val_ds, meta = load_synthetic("H", seed=0)
    assert meta["n_classes"] == 27
    assert meta["siblings_per_parent"] == 3
    assert hasattr(train_ds, "parent_ids")
    assert len(train_ds) > 0 and len(val_ds) > 0

    # Balanced mini subset (2 per leaf).
    x, y = train_ds.tensors
    parents = train_ds.parent_ids
    keep = []
    for c in range(27):
        idx = (y == c).nonzero(as_tuple=True)[0][:2]
        keep.append(idx)
    keep = torch.cat(keep)
    mini = FeatureDataset(x[keep], y[keep], parents[keep])

    cfg = TrainConfig(
        tau=0.1,
        lr=1e-3,
        batch_size=54,  # 27 classes * m=2
        epochs=1,
        m_per_class=2,
        seed=0,
        device="cpu",
        num_workers=0,
    )
    model = train_head(mini, lambda: MLPEncoder(), head="sph", cfg=cfg)
    with torch.no_grad():
        z = model(x[keep][:8])
    assert z.shape == (8, 16)
    _ok("synthetic 1-epoch train (sph)")


def main() -> int:
    print("smoke_test: starting")
    test_geometry()
    test_lp_normalize_flag()
    test_losses_forward()
    test_encoders()
    test_p0_parent_hit()
    test_synthetic_one_epoch()
    print("smoke_test: ALL PASSED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"smoke_test: FAILED: {exc}", file=sys.stderr)
        raise
