"""Embedding maps and pairwise distances for Euclidean / spherical / Poincaré heads.

Poincaré formulas follow the standard ball model used in:
  - geoopt: https://github.com/geoopt/geoopt
    geoopt/manifolds/stereographic/math.py  (_expmap0, _mobius_add, _dist, _project)
  - HypMix vendored hyptorch: https://github.com/YunYunY/HypMix
    hyptorch/pmath.py  (expmap0, mobius_add, dist, project)

Numeric choice: ball margin eps=1e-5 for every dtype (matches a prior synthetic
probe in this project). geoopt uses eps=4e-3 on float32 and 1e-5 otherwise —
that is a different stability trade-off, not a change of geometry.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

HEADS = ("euclid", "sph", "hyp")

# Open-ball margin. Keep fixed across heads/runs; do not retune with geometry.
EPS = 1e-5
# Curvature parameter c > 0 for sectional curvature k = -c (geoopt PoincareBall).
C_CURV = 1.0


def project_ball(x: torch.Tensor, eps: float = EPS, c: float = C_CURV) -> torch.Tensor:
    """Project points into the open ball of radius (1 - eps) / sqrt(c)."""
    max_norm = (1.0 - eps) / (c**0.5)
    n = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-12)
    # Scale only vectors that would leave the open ball.
    return x * torch.clamp(max_norm / n, max=1.0)


def expmap0(v: torch.Tensor, c: float = C_CURV) -> torch.Tensor:
    """Exponential map at the origin of the Poincaré ball.

    Important: apply tanh to the *tangent* vector norm, then project.
    Projecting into the ball *before* tanh collapses radii (all points land
    near a fixed shell). Same order as geoopt _expmap0 and HypMix ToPoincare.
    """
    v_norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(1e-12)
    sqrt_c = c**0.5
    # tanh argument clamp: geoopt/HypMix also clamp ~±15 for float stability.
    arg = (sqrt_c * v_norm).clamp(max=15.0)
    coef = torch.tanh(arg) / (sqrt_c * v_norm)
    return project_ball(coef * v, c=c)


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = C_CURV) -> torch.Tensor:
    """Möbius addition on the Poincaré ball (gyrovector form)."""
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + (c * c) * x2 * y2
    return num / den.clamp_min(1e-12)


def poincare_distance_elementwise(
    u: torch.Tensor, v: torch.Tensor, c: float = C_CURV
) -> torch.Tensor:
    """Elementwise (broadcast) Poincaré geodesic; last dim is the feature dim."""
    diff = mobius_add(-u, v, c=c)
    n = torch.linalg.vector_norm(diff, dim=-1)
    # Clamp into the open domain of atanh.
    z = (c**0.5 * n).clamp(0.0, 1.0 - EPS)
    return (2.0 / (c**0.5)) * torch.atanh(z)


def poincare_distance_matrix(
    u: torch.Tensor, v: torch.Tensor | None = None, c: float = C_CURV
) -> torch.Tensor:
    """Full pairwise Poincaré distances: u [B,D], v [C,D] -> [B, C]."""
    if v is None:
        v = u
    # [B, 1, D] vs [1, C, D] -> elementwise over pairs without a (B*C)^2 blow-up.
    return poincare_distance_elementwise(u.unsqueeze(1), v.unsqueeze(0), c=c)


def poincare_distance(
    u: torch.Tensor, v: torch.Tensor, c: float = C_CURV
) -> torch.Tensor:
    """Pairwise matrix when both are [B, D] batches; else elementwise broadcast.

    Prefer ``poincare_distance_matrix`` / ``poincare_distance_elementwise`` when
    the intent is ambiguous (same-shaped tensors are treated as two batches).
    """
    if u.dim() == 2 and v.dim() == 2:
        return poincare_distance_matrix(u, v, c=c)
    return poincare_distance_elementwise(u, v, c=c)


def pairwise_euclidean(z: torch.Tensor) -> torch.Tensor:
    """Ambient Euclidean pairwise distances [B, B]."""
    return torch.cdist(z, z, p=2)


def pairwise_cosine_distance(z: torch.Tensor) -> torch.Tensor:
    """1 - cosine similarity after ℓ₂ normalization (chordal surrogate)."""
    z = F.normalize(z, dim=-1, eps=1e-12)
    return 1.0 - (z @ z.T)


def pairwise_arccos(z: torch.Tensor) -> torch.Tensor:
    """Spherical geodesic arccos(<x̂, ŷ>) on the unit sphere.

    Cosine and arccos are monotone on [-1, 1], so they agree on pairwise
    *ranks*; gradients differ. Used only for the P1 rank diagnostic.
    """
    z = F.normalize(z, dim=-1, eps=1e-12)
    # Clamp for acos domain; HypMix spherical branch does not use arccos.
    cos = (z @ z.T).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos)


def embed_for_head(z_amb: torch.Tensor, head: str) -> torch.Tensor:
    """Map ambient encoder outputs to the head's representation space."""
    if head == "euclid":
        return z_amb
    if head == "sph":
        return F.normalize(z_amb, dim=-1, eps=1e-12)
    if head == "hyp":
        # Unconstrained ambient -> expmap0 -> open ball (no pre-ℓ₂).
        return expmap0(z_amb)
    raise ValueError(f"unknown head: {head}")


def pairwise_distance(z_amb: torch.Tensor, head: str) -> torch.Tensor:
    """Pairwise distance matrix used for retrieval (lower = closer)."""
    if head == "euclid":
        return pairwise_euclidean(z_amb)
    if head == "sph":
        return pairwise_cosine_distance(z_amb)
    if head == "hyp":
        z = expmap0(z_amb)
        return poincare_distance(z, z)
    raise ValueError(f"unknown head: {head}")
