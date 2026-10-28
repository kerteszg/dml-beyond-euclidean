"""Evaluation: Recall@k, rank correlations, hierarchy-in-embedding (P0), radius stats."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from geometry import (
    embed_for_head,
    pairwise_arccos,
    pairwise_cosine_distance,
    pairwise_distance,
    pairwise_euclidean,
)

# Flag when most Poincaré points sit near the open-ball edge (weak geometry check).
NEAR_BOUNDARY_FRAC_WARN = 0.5
NEAR_BOUNDARY_RADIUS = 0.9

# P0: parent hit among non-leaf neighbors must beat chance by this margin.
P0_PARENT_HIT_MARGIN = 0.15
P0_PARENT_HIT_K = 5


@torch.no_grad()
def encode_all(
    model: torch.nn.Module,
    ds: Dataset,
    device: str = "cpu",
    batch_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ambient embeddings and labels for the whole dataset."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    zs, ys = [], []
    model.eval()
    for xb, yb in loader:
        zs.append(model(xb.to(device)).cpu())
        ys.append(yb if torch.is_tensor(yb) else torch.tensor(yb))
    return torch.cat(zs, dim=0), torch.cat(ys, dim=0).long()


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    if denom < 1e-12:
        return 0.0
    return float((ra * rb).sum() / denom)


@torch.no_grad()
def rank_correlations(z_amb: torch.Tensor, max_n: int = 256) -> dict[str, float]:
    """Spearman corr of pairwise distance ranks on a subsample of ambient features.

    P1 compares cosine vs arccos (same sphere) against cosine vs Poincaré
    (different geometry) on the *same* vectors — typically the spherical head's
    ambient outputs.
    """
    n = min(max_n, z_amb.shape[0])
    z = z_amb[:n]
    d_e = pairwise_euclidean(z).numpy()
    d_cos = pairwise_cosine_distance(z).numpy()
    d_arc = pairwise_arccos(z).numpy()
    d_hyp = pairwise_distance(z, "hyp").numpy()
    iu = np.triu_indices(n, k=1)

    def corr(a, b):
        return _spearman(a[iu], b[iu])

    return {
        "spearman_sph_cosine_vs_sph_arccos": corr(d_cos, d_arc),
        "spearman_sph_cosine_vs_hyp": corr(d_cos, d_hyp),
        "spearman_euclid_vs_sph_cosine": corr(d_e, d_cos),
        "spearman_euclid_vs_hyp": corr(d_e, d_hyp),
    }


@torch.no_grad()
def recall_at_k(z_amb: torch.Tensor, y: torch.Tensor, head: str, k: int = 1) -> float:
    """Leave-one-out Recall@k under the head's own distance."""
    d = pairwise_distance(z_amb, head)
    sim = -d
    n = z_amb.shape[0]
    sim = sim.clone()
    sim.fill_diagonal_(-1e9)
    topk = torch.topk(sim, k=min(k, n - 1), dim=1).indices
    y = y.cpu()
    hits = 0
    for i in range(n):
        if (y[topk[i]] == y[i]).any():
            hits += 1
    return hits / n


@torch.no_grad()
def hyp_radius_stats(z_amb: torch.Tensor) -> dict[str, Any]:
    """Poincaré radius diagnostics (numeric health, not a prediction)."""
    z = embed_for_head(z_amb, "hyp")
    r = torch.linalg.vector_norm(z, dim=-1)
    frac = float((r > NEAR_BOUNDARY_RADIUS).float().mean())
    out: dict[str, Any] = {
        "hyp_radius_mean": float(r.mean()),
        "hyp_radius_p90": float(torch.quantile(r, 0.90)),
        "hyp_radius_max": float(r.max()),
        "hyp_frac_near_boundary": frac,
    }
    if frac > NEAR_BOUNDARY_FRAC_WARN:
        out["warnings"] = ["poincare_near_boundary"]
    return out


@torch.no_grad()
def alphabet_neighbor_rate(
    z_amb: torch.Tensor,
    y: torch.Tensor,
    alphabet_ids: torch.Tensor,
    head: str,
) -> float:
    """Fraction of top-1 neighbors that share the query's alphabet (legacy descriptive)."""
    d = pairwise_distance(z_amb, head)
    sim = -d
    sim = sim.clone()
    sim.fill_diagonal_(-1e9)
    nn = torch.argmax(sim, dim=1)
    alphabet_ids = alphabet_ids.cpu()
    return float((alphabet_ids[nn] == alphabet_ids).float().mean())


def parent_hit_chance(n_classes: int, siblings_per_parent: int) -> float:
    """Chance a random other-leaf neighbor shares the parent.

    ``siblings_per_parent`` = number of leaf classes under one parent
    (Omniglot: chars per alphabet; synthetic ternary: BRANCH=3).
    """
    other = max(n_classes - 1, 1)
    same_parent_other = max(siblings_per_parent - 1, 0)
    return same_parent_other / other


@torch.no_grad()
def parent_hit_at_k_excl_leaf(
    z_amb: torch.Tensor,
    y: torch.Tensor,
    parent_ids: torch.Tensor,
    head: str,
    k: int = P0_PARENT_HIT_K,
) -> dict[str, float]:
    """Same-parent rate among top-k neighbors after excluding the leaf class.

    This is the P0 hierarchy-in-embedding check: leaf accuracy cannot inflate it.
    """
    y = y.cpu().long()
    parent_ids = parent_ids.cpu().long()
    d = pairwise_distance(z_amb, head)
    sim = -d
    n = z_amb.shape[0]
    sim = sim.clone()
    sim.fill_diagonal_(-1e9)
    # Mask same-leaf neighbors so retrieval cannot score by character identity.
    for i in range(n):
        sim[i, y == y[i]] = -1e9

    # How many foreign candidates remain per row?
    rates = []
    gaps = []
    for i in range(n):
        # Candidates: other leaves only (already masked same leaf on diagonal+same y)
        avail = (sim[i] > -1e8).nonzero(as_tuple=True)[0]
        if avail.numel() == 0:
            continue
        kk = min(k, int(avail.numel()))
        # topk on the masked row
        neigh = torch.topk(sim[i], k=kk).indices
        hit = (parent_ids[neigh] == parent_ids[i]).float().mean()
        rates.append(float(hit))

        # Distance gap: mean d to other-parent vs same-parent other-leaf.
        same_p = (parent_ids == parent_ids[i]) & (y != y[i])
        other_p = parent_ids != parent_ids[i]
        if same_p.any() and other_p.any():
            # pairwise_distance returns d; use that matrix
            dd = d[i]
            gaps.append(float(dd[other_p].mean() - dd[same_p].mean()))

    out = {
        "parent_hit@5_excl_leaf": float(np.mean(rates)) if rates else float("nan"),
        "n_queries_scored": float(len(rates)),
    }
    if gaps:
        out["parent_distance_gap"] = float(np.mean(gaps))
    return out


def evaluate_hierarchy_p0(
    z_amb: torch.Tensor,
    y: torch.Tensor,
    parent_ids: torch.Tensor,
    head: str,
    n_classes: int,
    siblings_per_parent: int,
) -> dict[str, Any]:
    """P0 validity check on one embedding set (usually spherical head, regime H)."""
    chance = parent_hit_chance(n_classes, siblings_per_parent)
    thr = chance + P0_PARENT_HIT_MARGIN
    stats = parent_hit_at_k_excl_leaf(z_amb, y, parent_ids, head=head, k=P0_PARENT_HIT_K)
    hit = stats["parent_hit@5_excl_leaf"]
    holds = bool(hit == hit and hit >= thr)  # False if NaN
    return {
        "head": head,
        "parent_hit@5_excl_leaf": hit,
        "parent_distance_gap": stats.get("parent_distance_gap"),
        "chance_baseline": chance,
        "threshold": thr,
        "margin": P0_PARENT_HIT_MARGIN,
        "siblings_per_parent": siblings_per_parent,
        "n_classes": n_classes,
        "p0_holds": holds,
    }


def evaluate_model(
    model: torch.nn.Module,
    val_ds: Dataset,
    head: str,
    device: str = "cpu",
    parent_ids: torch.Tensor | None = None,
    hierarchy_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    z, y = encode_all(model, val_ds, device=device)
    out: dict[str, Any] = {
        "recall@1": recall_at_k(z, y, head, k=1),
        "recall@5": recall_at_k(z, y, head, k=5),
    }
    out.update(rank_correlations(z))
    if head == "hyp":
        out.update(hyp_radius_stats(z))
    if parent_ids is not None:
        # Legacy descriptive rate (confounded with leaf R@1); keep for continuity.
        out["alphabet_neighbor_rate@1"] = alphabet_neighbor_rate(
            z, y, parent_ids, head
        )
        out.update(
            parent_hit_at_k_excl_leaf(z, y, parent_ids, head=head, k=P0_PARENT_HIT_K)
        )
        if hierarchy_meta is not None:
            out["parent_hit_chance_baseline"] = parent_hit_chance(
                int(hierarchy_meta["n_classes"]),
                int(hierarchy_meta["siblings_per_parent"]),
            )
    return out
