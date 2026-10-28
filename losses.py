"""Supervised contrastive loss wired to Euclidean / cosine / Poincaré distances.

Scaffold: pytorch-metric-learning
  https://github.com/KevinMusgrave/pytorch-metric-learning
  - losses.SupConLoss
  - distances.LpDistance, CosineSimilarity, BaseDistance

Positive-mask semantics (same label = positive, exclude self) match
  https://github.com/HobbitLong/SupContrast  losses.py::SupConLoss

Do not call HobbitLong SupConLoss.forward directly: that path always uses
inner-product logits, which would erase Euclidean and Poincaré distances.
"""
from __future__ import annotations

import torch
from pytorch_metric_learning.distances import (
    BaseDistance,
    CosineSimilarity,
    LpDistance,
)
from pytorch_metric_learning.losses import SupConLoss

from geometry import C_CURV, poincare_distance_elementwise, poincare_distance_matrix


class PoincareDistance(BaseDistance):
    """Geodesic distance on the Poincaré ball as a PML BaseDistance.

    normalize_embeddings=False is required: ℓ₂-normalizing before the ball
    map would collapse radial degrees of freedom (HypMix does pre-normalize
    as a design choice; we do not, so the three heads stay matched).
    is_inverted=False means small values = close; SupConLoss then uses -d/τ.
    """

    def __init__(self, c: float = C_CURV, **kwargs):
        # Force no ambient normalization; ignore caller overrides that would break the ball.
        kwargs["normalize_embeddings"] = False
        kwargs["is_inverted"] = False
        super().__init__(**kwargs)
        self.c = c

    def compute_mat(self, query_emb: torch.Tensor, ref_emb: torch.Tensor) -> torch.Tensor:
        # query/ref already in the open ball (mapped in train/metrics before the loss).
        if ref_emb is None:
            ref_emb = query_emb
        return poincare_distance_matrix(query_emb, ref_emb, c=self.c)

    def pairwise_distance(self, query_emb: torch.Tensor, ref_emb: torch.Tensor) -> torch.Tensor:
        return poincare_distance_elementwise(query_emb, ref_emb, c=self.c)


def make_loss(head: str, tau: float = 0.1) -> SupConLoss:
    """Build SupConLoss with the distance appropriate to ``head``.

    - euclid: LpDistance without normalization (true ambient Euclidean).
      PML BaseDistance defaults to normalize_embeddings=True; that would turn
      Euclidean into chordal-on-sphere and break the matched comparison.
    - sph: CosineSimilarity (ℓ₂ inside the distance object).
    - hyp: PoincareDistance on already-mapped ball points.
    """
    if head == "euclid":
        distance = LpDistance(normalize_embeddings=False, p=2, power=1)
    elif head == "sph":
        distance = CosineSimilarity()
    elif head == "hyp":
        distance = PoincareDistance()
    else:
        raise ValueError(f"unknown head: {head}")
    return SupConLoss(temperature=tau, distance=distance)
