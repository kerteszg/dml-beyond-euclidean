"""Manuscript-facing aggregation over results/<tier>/seed_*/metrics.json.

Produces (a) a JSON summary with per-regime, per-head Recall@1 mean/std across
seeds, rank-correlation gaps, hierarchy-gate hits, structure-interaction deltas,
status counts and near-boundary fractions; (b) two aggregate figures for the
paper; (c) a LaTeX table body. Nothing here retrains or recomputes metrics —
it only reads what run.py wrote.

Usage:
  python manuscript_summary.py --out ../manuscript/latex/figures/mwe
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

# (results folder, label used in paper, has H/F regimes)
TIERS: list[tuple[str, str, bool]] = [
    ("synthetic", "Synthetic", True),
    ("cifar10", "CIFAR-10 (5k)", False),
    ("cifar10_full", "CIFAR-10 (50k)", False),
    ("omniglot", "Omniglot (27)", True),
    ("omniglot_full", "Omniglot (45)", True),
]
HEADS = ("euclid", "sph", "hyp")
HEAD_LABEL = {"euclid": "Euclid", "sph": "Sph", "hyp": "Hyp"}
# Compact tick labels for the Recall@1 figure (one column per tier/regime).
SHORT_LABEL = {
    "synthetic": "Synth.",
    "cifar10": "CIFAR 5k",
    "cifar10_full": "CIFAR 50k",
    "omniglot": "Omni. 27",
    "omniglot_full": "Omni. 45",
}


def _ms(vals: list[float]) -> dict[str, Any]:
    a = np.asarray(vals, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=0)), "n": int(a.size), "values": [float(v) for v in a]}


def _load_seeds(folder: Path) -> list[dict[str, Any]]:
    out = []
    for p in sorted(folder.glob("seed_*/metrics.json")):
        out.append(json.loads(p.read_text(encoding="utf-8")))
    if not out:
        raise FileNotFoundError(f"no seed_*/metrics.json under {folder}")
    return out


def summarize_tier(name: str, label: str, has_hf: bool) -> dict[str, Any]:
    seeds = _load_seeds(RESULTS / name)
    tier: dict[str, Any] = {
        "tier": name,
        "label": label,
        "has_hf": has_hf,
        "seeds": [s["seed"] for s in seeds],
        "epochs": sorted({int(s["epochs"]) for s in seeds}),
        "tau": sorted({float(s["tau"]) for s in seeds}),
    }

    # Rank correlations on the spherical head's ambient validation features.
    def rank_block(s: dict[str, Any]) -> dict[str, float]:
        return s["regimes"]["H"]["rank_from_sph_model"] if has_hf else s["rank_from_sph_model"]

    tier["spearman_cos_arccos"] = _ms([rank_block(s)["spearman_sph_cosine_vs_sph_arccos"] for s in seeds])
    tier["spearman_cos_poincare"] = _ms([rank_block(s)["spearman_sph_cosine_vs_hyp"] for s in seeds])
    tier["spearman_euclid_cos"] = _ms([rank_block(s)["spearman_euclid_vs_sph_cosine"] for s in seeds])
    tier["spearman_euclid_poincare"] = _ms([rank_block(s)["spearman_euclid_vs_hyp"] for s in seeds])
    tier["p1_gap"] = _ms([s["summary"]["p1_corr_gap"] for s in seeds])
    tier["p1_hold_rate"] = float(np.mean([s["predictions"]["p1_sph_arccos_corr_gt_sph_hyp"] for s in seeds]))

    # Recall@1 per regime and head.
    regimes = ("H", "F") if has_hf else ("all",)
    tier["recall1"] = {}
    tier["hyp_frac_near_boundary"] = {}
    for r in regimes:
        tier["recall1"][r] = {}
        for h in HEADS:
            vals = [
                (s["regimes"][r]["heads"][h] if has_hf else s["heads"][h])["recall@1"] for s in seeds
            ]
            tier["recall1"][r][h] = _ms(vals)
        nb = [
            (s["regimes"][r]["heads"]["hyp"] if has_hf else s["heads"]["hyp"])["hyp_frac_near_boundary"]
            for s in seeds
        ]
        tier["hyp_frac_near_boundary"][r] = _ms(nb)
    tier["warnings_any"] = sorted({w for s in seeds for w in s.get("warnings", [])})

    if has_hf:
        tier["p0_hit"] = _ms([s["summary"]["p0"]["parent_hit@5_excl_leaf"] for s in seeds])
        tier["p0_threshold"] = float(seeds[0]["summary"]["p0"]["threshold"])
        tier["p0_chance"] = float(seeds[0]["summary"]["p0"]["chance_baseline"])
        tier["p0_hold_rate"] = float(np.mean([s["predictions"]["p0_hierarchy_in_embedding"] for s in seeds]))
        tier["delta"] = _ms([s["summary"]["structure_interaction_delta"] for s in seeds])
        tier["gap_H"] = _ms([s["summary"]["hyp_minus_sph_recall1"]["H"] for s in seeds])
        tier["gap_F"] = _ms([s["summary"]["hyp_minus_sph_recall1"]["F"] for s in seeds])
        statuses = [s["summary"]["p2_status"] for s in seeds]
        tier["p2_status_counts"] = {k: statuses.count(k) for k in sorted(set(statuses))}
        tier["n_classes"] = int(seeds[0]["regimes"]["H"]["meta"]["n_classes"])
        tier["n_val_H"] = int(seeds[0]["regimes"]["H"]["meta"]["n_val"])
        tier["n_train_H"] = int(seeds[0]["regimes"]["H"]["meta"]["n_train"])
    else:
        tier["n_classes"] = int(seeds[0]["meta"]["n_classes"])
        tier["n_val"] = int(seeds[0]["meta"]["n_val"])
        tier["n_train"] = int(seeds[0]["meta"]["n_train"])
    return tier


def fig_rank(summary: list[dict[str, Any]], out: Path) -> None:
    """Grouped bars: cos–arccos vs cos–Poincaré Spearman per tier (mean ± std)."""
    labels = [t["label"] for t in summary]
    a = np.array([t["spearman_cos_arccos"]["mean"] for t in summary])
    a_s = np.array([t["spearman_cos_arccos"]["std"] for t in summary])
    b = np.array([t["spearman_cos_poincare"]["mean"] for t in summary])
    b_s = np.array([t["spearman_cos_poincare"]["std"] for t in summary])
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.bar(x - w / 2, a, w, yerr=a_s, capsize=2, color="#4C78A8", label=r"cosine vs. $\arccos$ (same sphere)")
    ax.bar(x + w / 2, b, w, yerr=b_s, capsize=2, color="#E45756", label=r"cosine vs. Poincaré $d_c$")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Spearman rank correlation", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_recall(summary: list[dict[str, Any]], out: Path) -> None:
    """Recall@1 by head for every (tier, regime) column, mean ± std."""
    cols: list[tuple[str, str]] = []  # (column label, tier idx/regime)
    means = {h: [] for h in HEADS}
    stds = {h: [] for h in HEADS}
    for t in summary:
        regimes = ("H", "F") if t["has_hf"] else ("all",)
        for r in regimes:
            short = SHORT_LABEL[t["tier"]]
            lab = short if r == "all" else f"{short}\n{r}"
            cols.append((lab, r))
            for h in HEADS:
                means[h].append(t["recall1"][r][h]["mean"])
                stds[h].append(t["recall1"][r][h]["std"])
    x = np.arange(len(cols))
    w = 0.26
    colors = {"euclid": "#1f77b4", "sph": "#ff7f0e", "hyp": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    for i, h in enumerate(HEADS):
        ax.bar(x + (i - 1) * w, means[h], w, yerr=stds[h], capsize=2, color=colors[h], label=HEAD_LABEL[h])
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in cols], fontsize=8)
    ax.set_ylim(0.0, 1.0)
    # Light separators between tiers.
    boundaries = []
    acc = 0
    for t in summary[:-1]:
        acc += 2 if t["has_hf"] else 1
        boundaries.append(acc - 0.5)
    for b in boundaries:
        ax.axvline(b, color="lightgray", linewidth=0.6)
    ax.set_ylabel("Recall@1", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _f(ms: dict[str, Any], digits: int = 3, signed: bool = False) -> str:
    fmt = f"{{:+.{digits}f}}" if signed else f"{{:.{digits}f}}"
    return f"{fmt.format(ms['mean'])} $\\pm$ {ms['std']:.{digits}f}"


def latex_table(summary: list[dict[str, Any]]) -> str:
    """Table body (no tabular wrapper) — pasted into 04-experiment.tex by hand."""
    lines = []
    lines.append("% --- generated by dml-beyond-euclidean/manuscript_summary.py; do not edit by hand ---")
    lines.append("% Block A: rank correlations and gates")
    for t in summary:
        gap = _f(t["p1_gap"])
        cos_hyp = _f(t["spearman_cos_poincare"])
        if t["has_hf"]:
            p0 = f"{t['p0_hit']['mean']:.2f} $\\pm$ {t['p0_hit']['std']:.2f} ({t['p0_threshold']:.2f})"
            delta = _f(t["delta"], signed=True)
            st = ", ".join(f"{k}$\\times${v}" for k, v in t["p2_status_counts"].items())
            nb = f"{t['hyp_frac_near_boundary']['H']['mean']:.2f}"
        else:
            p0 = "---"
            delta = "---"
            st = "---"
            nb = f"{t['hyp_frac_near_boundary']['all']['mean']:.2f}"
        lines.append(f"{t['label']} & {cos_hyp} & {gap} & {p0} & {delta} & {st} & {nb} \\\\")
    lines.append("% Block B: Recall@1 per head (mean ± std over seeds)")
    for t in summary:
        regimes = ("H", "F") if t["has_hf"] else ("all",)
        for r in regimes:
            lab = t["label"] + ("" if r == "all" else f" {r}")
            cells = " & ".join(_f(t["recall1"][r][h]) for h in HEADS)
            lines.append(f"{lab} & {cells} \\\\")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT.parent / "manuscript" / "latex" / "figures" / "mwe")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    summary = [summarize_tier(n, l, hf) for n, l, hf in TIERS]
    (args.out / "mwe_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig_rank(summary, args.out / "fig_mwe_rank.png")
    fig_recall(summary, args.out / "fig_mwe_recall.png")
    (args.out / "table_body.tex").write_text(latex_table(summary), encoding="utf-8")

    # Console digest.
    for t in summary:
        print(f"== {t['label']}  seeds={t['seeds']}  epochs={t['epochs']}")
        print(f"   cos-arccos {t['spearman_cos_arccos']['mean']:.3f}  cos-Poincare {t['spearman_cos_poincare']['mean']:.3f}"
              f"  gap {t['p1_gap']['mean']:.3f}±{t['p1_gap']['std']:.3f}  hold {t['p1_hold_rate']:.2f}")
        for r, hv in t["recall1"].items():
            print("   R@1 " + r + ": " + "  ".join(f"{HEAD_LABEL[h]} {hv[h]['mean']:.3f}±{hv[h]['std']:.3f}" for h in HEADS)
                  + f"   near-boundary(hyp) {t['hyp_frac_near_boundary'][r]['mean']:.2f}")
        if t["has_hf"]:
            print(f"   P0 hit {t['p0_hit']['mean']:.3f}±{t['p0_hit']['std']:.3f} thr {t['p0_threshold']:.3f} hold {t['p0_hold_rate']:.2f}"
                  f"   gapH {t['gap_H']['mean']:+.3f} gapF {t['gap_F']['mean']:+.3f}  delta {t['delta']['mean']:+.3f}±{t['delta']['std']:.3f}"
                  f"   status {t['p2_status_counts']}")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
