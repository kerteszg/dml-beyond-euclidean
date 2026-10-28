"""Figures for rank-correlation and retrieval diagnostics."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_rank_correlation(metrics: dict[str, Any], out_path: Path) -> None:
    """Bar chart of cos–arccos vs cos–hyp Spearman on sph-head ambient features."""
    regimes = list(metrics.get("regimes", {}).keys())
    if not regimes:
        regimes = ["main"]
        block = {"rank_from_sph_model": metrics.get("rank_from_sph_model", {})}
        data = {"main": block}
    else:
        data = metrics["regimes"]

    labels, vals, colors = [], [], []
    palette = ["#4C78A8", "#E45756"]
    for regime in regimes:
        r = data[regime].get("rank_from_sph_model", {})
        labels.append(f"{regime}: cos vs arccos")
        vals.append(r.get("spearman_sph_cosine_vs_sph_arccos", 0.0))
        colors.append(palette[0])
        labels.append(f"{regime}: cos vs hyp")
        vals.append(r.get("spearman_sph_cosine_vs_hyp", 0.0))
        colors.append(palette[1])

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    xpos = np.arange(len(vals))
    ax.bar(xpos, vals, color=colors)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Spearman corr. of pairwise distances")
    ax.set_title("Hardness-order agreement (P1)")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_retrieval(metrics: dict[str, Any], out_path: Path) -> None:
    """Grouped Recall@1 bars by head (and regime when H/F present)."""
    regimes = list(metrics.get("regimes", {}).keys())
    heads = ["euclid", "sph", "hyp"]

    if not regimes:
        # Single-regime datasets (e.g. CIFAR-10).
        regimes = ["main"]
        data = {
            "main": {
                "heads": metrics.get("heads", {}),
            }
        }
    else:
        data = metrics["regimes"]

    width = 0.25
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    x = np.arange(len(regimes))
    for i, head in enumerate(heads):
        vals = [data[r]["heads"][head]["recall@1"] for r in regimes]
        ax.bar(x + (i - 1) * width, vals, width=width, label=head)

    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.set_ylabel("Recall@1")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Retrieval by geometry")

    summary = metrics.get("summary", {})
    if "hyp_minus_sph_recall1" in summary and len(regimes) >= 2:
        gaps = summary["hyp_minus_sph_recall1"]
        delta = summary.get("structure_interaction_delta")
        ax.text(
            0.02,
            0.02,
            f"Hyp−Sph R@1: "
            + ", ".join(f"{k}={gaps[k]:+.3f}" for k in regimes)
            + (f", Δ={delta:+.3f}" if delta is not None else ""),
            transform=ax.transAxes,
            fontsize=8,
            family="monospace",
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
