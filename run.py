"""CLI: matched three-head checks on synthetic / CIFAR-10 / Omniglot.

Usage:
  python smoke_test.py
  python run.py --dataset synthetic --seed 20260924
  python run.py --dataset cifar10 --seeds 20260924,20260925,20260926 --workers 8
  python run.py --dataset omniglot --full --seeds 20260924,20260925,20260926
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# Quiet download bars before torchvision imports in data_*.py
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch

from geometry import HEADS
from metrics import (
    encode_all,
    evaluate_hierarchy_p0,
    evaluate_model,
    rank_correlations,
)
from models import MLPEncoder, SmallConvEncoder
from plots import plot_rank_correlation, plot_retrieval
from train import TrainConfig, train_head

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
RESULTS_ROOT = ROOT / "results"

P1_GAP = 0.05
P2_DELTA = 0.02
DEFAULT_SEEDS = (20260924, 20260925, 20260926)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Matched Euclidean / spherical / Poincaré checks")
    p.add_argument(
        "--dataset",
        choices=("synthetic", "cifar10", "omniglot"),
        required=True,
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="CIFAR: full train set + more epochs; Omniglot: 5x9 / 45-class arms "
        "(background+evaluation alphabets)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="1-epoch dry run (still may download CIFAR/Omniglot on first call)",
    )
    p.add_argument("--seed", type=int, default=None, help="single seed (default 20260924)")
    p.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="comma-separated seeds, e.g. 20260924,20260925,20260926 "
        "(writes per-seed folders + aggregate.json)",
    )
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="DataLoader worker processes (try 8–16 on a large CPU VM)",
    )
    return p.parse_args()


def _parse_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if args.seed is not None:
        return [int(args.seed)]
    return [DEFAULT_SEEDS[0]]


def _epochs_for(dataset: str, full: bool, smoke: bool) -> int:
    if smoke:
        return 1
    if dataset == "synthetic":
        return 45
    if dataset == "cifar10":
        return 30 if full else 12
    if dataset == "omniglot":
        # Matched epoch budget for quick vs full: --full enlarges the arms, not epochs.
        return 20
    raise ValueError(dataset)


def _results_root(dataset: str, full: bool, smoke: bool) -> Path:
    """Distinct dirs so quick and full never clobber each other."""
    if smoke:
        name = f"{dataset}_smoke"
    elif full:
        name = f"{dataset}_full"
    else:
        name = dataset
    return RESULTS_ROOT / name


def _encoder_factory(dataset: str, meta: dict[str, Any]) -> Callable[[], torch.nn.Module]:
    if dataset == "synthetic":
        return lambda: MLPEncoder(in_dim=32, hidden=64, out_dim=16)
    in_ch = int(meta["in_channels"])
    return lambda: SmallConvEncoder(in_channels=in_ch, embed_dim=32)


def _load_regime(
    dataset: str, regime: str | None, seed: int, full: bool
) -> tuple[Any, Any, dict[str, Any]]:
    if dataset == "synthetic":
        from data_synthetic import load_synthetic

        assert regime is not None
        return load_synthetic(regime, seed=seed)
    if dataset == "cifar10":
        from data_cifar10 import load_cifar10

        return load_cifar10(DATA_ROOT / "cifar10", full=full, seed=seed)
    if dataset == "omniglot":
        from data_omniglot import load_omniglot

        assert regime is not None
        return load_omniglot(DATA_ROOT / "omniglot", regime=regime, seed=seed, full=full)
    raise ValueError(dataset)


def _run_regime(
    dataset: str,
    regime: str | None,
    seed: int,
    full: bool,
    device: str,
    epochs: int,
    num_workers: int,
) -> dict[str, Any]:
    train_ds, val_ds, meta = _load_regime(dataset, regime, seed, full)
    encoder_fn = _encoder_factory(dataset, meta)
    m_per_class = 8 if dataset == "cifar10" else 4
    cfg = TrainConfig(
        tau=0.1,
        lr=1e-3,
        batch_size=128,
        epochs=epochs,
        m_per_class=m_per_class,
        seed=seed,
        device=device,
        num_workers=num_workers,
    )

    parent_ids = getattr(val_ds, "parent_ids", None)
    hierarchy_meta = None
    if parent_ids is not None and "siblings_per_parent" in meta:
        hierarchy_meta = {
            "n_classes": int(meta["n_classes"]),
            "siblings_per_parent": int(meta["siblings_per_parent"]),
        }

    block: dict[str, Any] = {"meta": meta, "heads": {}}
    models: dict[str, torch.nn.Module] = {}

    for head in HEADS:
        tag = regime or dataset
        print(f"Training dataset={dataset} regime={tag} head={head} seed={seed} ...", flush=True)
        head_seed = seed + (0 if regime is None else (0 if regime == "H" else 1000))
        head_seed += {"euclid": 0, "sph": 1, "hyp": 2}[head]
        cfg_h = TrainConfig(**{**cfg.__dict__, "seed": head_seed})
        model = train_head(train_ds, encoder_fn, head=head, cfg=cfg_h)
        models[head] = model
        # Parent-hit stats on every head when parents exist (descriptive);
        # formal P0 gate is computed once on Sph / H in _finalize_predictions.
        ev = evaluate_model(
            model,
            val_ds,
            head=head,
            device=device,
            parent_ids=parent_ids,
            hierarchy_meta=hierarchy_meta,
        )
        block["heads"][head] = ev
        print(
            f"  R@1={ev['recall@1']:.3f}  R@5={ev['recall@5']:.3f}",
            flush=True,
        )

    z_sph, y_val = encode_all(models["sph"], val_ds, device=device)
    block["rank_from_sph_model"] = rank_correlations(z_sph)

    # Store Sph embeddings for the H-regime P0 gate (hierarchy-in-embedding).
    if regime == "H" and parent_ids is not None and hierarchy_meta is not None:
        block["hierarchy_p0"] = evaluate_hierarchy_p0(
            z_sph,
            y_val,
            parent_ids,
            head="sph",
            n_classes=hierarchy_meta["n_classes"],
            siblings_per_parent=hierarchy_meta["siblings_per_parent"],
        )
        print(
            f"  P0 parent_hit@5_excl_leaf="
            f"{block['hierarchy_p0']['parent_hit@5_excl_leaf']:.3f} "
            f"(thr={block['hierarchy_p0']['threshold']:.3f}, "
            f"holds={block['hierarchy_p0']['p0_holds']})",
            flush=True,
        )
    return block


def _collect_warnings(results: dict[str, Any], has_hf: bool) -> list[str]:
    warnings: list[str] = []
    if has_hf:
        blocks = [results["regimes"][r] for r in results["regimes"]]
    else:
        blocks = [{"heads": results["heads"]}]
    for block in blocks:
        hyp = block["heads"].get("hyp", {})
        for w in hyp.get("warnings", []):
            if w not in warnings:
                warnings.append(w)
    return warnings


def _finalize_predictions(results: dict[str, Any], has_hf: bool) -> None:
    """Set P1 always; P0 gate + gated P2 when H/F regimes exist."""
    if has_hf:
        rH = results["regimes"]["H"]["rank_from_sph_model"]
    else:
        rH = results["rank_from_sph_model"]

    gap = (
        rH["spearman_sph_cosine_vs_sph_arccos"]
        - rH["spearman_sph_cosine_vs_hyp"]
    )
    results["summary"]["p1_corr_gap"] = gap
    p1 = gap > P1_GAP
    results["predictions"]["p1_sph_arccos_corr_gt_sph_hyp"] = bool(p1)
    results["falsified"]["p1"] = not bool(p1)

    results["warnings"] = _collect_warnings(results, has_hf=has_hf)
    weak_geo = "poincare_near_boundary" in results["warnings"]

    if not has_hf:
        results["predictions"]["p0_hierarchy_in_embedding"] = None
        results["predictions"]["p2_structure_gap_positive"] = None
        results["falsified"]["p0"] = None
        results["falsified"]["p2"] = None
        results["summary"]["p2_status"] = "not_applicable"
        return

    # --- P0: hierarchy must appear in Sph embeddings on H before interpreting P2 ---
    p0_block = results["regimes"]["H"].get("hierarchy_p0")
    if p0_block is None:
        # Should not happen for synthetic/omniglot H; mark missing.
        results["predictions"]["p0_hierarchy_in_embedding"] = False
        results["falsified"]["p0"] = True
        results["summary"]["p0"] = {"error": "hierarchy_p0_missing"}
        p0_holds = False
    else:
        p0_holds = bool(p0_block["p0_holds"])
        results["summary"]["p0"] = p0_block
        results["predictions"]["p0_hierarchy_in_embedding"] = p0_holds
        results["falsified"]["p0"] = not p0_holds

    # --- P2 raw delta (always recorded; status gated) ---
    gaps = {}
    for regime in ("H", "F"):
        h = results["regimes"][regime]["heads"]
        gaps[regime] = h["hyp"]["recall@1"] - h["sph"]["recall@1"]
    delta = gaps["H"] - gaps["F"]
    results["summary"]["hyp_minus_sph_recall1"] = gaps
    results["summary"]["structure_interaction_delta"] = delta
    results["summary"]["structure_interaction_metric"] = "recall@1"
    p2_raw = delta > P2_DELTA
    results["predictions"]["p2_structure_gap_positive"] = bool(p2_raw)

    # Gate: weak Poincaré geometry → do not treat as a clean P2 test;
    # missing hierarchy in embeddings → P2 not interpretable as structure×geometry.
    if weak_geo:
        status = "weak_geometry"
        results["falsified"]["p2"] = None
    elif not p0_holds:
        status = "not_applicable"
        results["falsified"]["p2"] = None
    else:
        status = "hold" if p2_raw else "fail"
        results["falsified"]["p2"] = status == "fail"
    results["summary"]["p2_status"] = status


def _run_one_seed(
    dataset: str,
    seed: int,
    full: bool,
    smoke: bool,
    device: str,
    workers: int,
    out_dir: Path,
) -> dict[str, Any]:
    epochs = _epochs_for(dataset, full, smoke)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    results: dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "tau": 0.1,
        "full": bool(full),
        "smoke": bool(smoke),
        "epochs": epochs,
        "device": device,
        "workers": workers,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "predictions": {},
        "falsified": {},
        "summary": {},
        "warnings": [],
    }

    has_hf = dataset in {"synthetic", "omniglot"}
    if has_hf:
        results["regimes"] = {}
        for regime in ("H", "F"):
            results["regimes"][regime] = _run_regime(
                dataset, regime, seed, full, device, epochs, workers
            )
    else:
        block = _run_regime(dataset, None, seed, full, device, epochs, workers)
        results["heads"] = block["heads"]
        results["rank_from_sph_model"] = block["rank_from_sph_model"]
        results["meta"] = block["meta"]

    _finalize_predictions(results, has_hf=has_hf)
    results["elapsed_sec"] = time.perf_counter() - t0

    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot_rank_correlation(results, out_dir / "fig_rank_correlation.png")
    plot_retrieval(results, out_dir / "fig_retrieval.png")
    return results


def _mean_std(vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if len(arr) else float("nan"),
        "std": float(arr.std(ddof=0)) if len(arr) else float("nan"),
        "values": [float(v) for v in vals],
    }


def _aggregate(seed_results: list[dict[str, Any]], dataset: str, full: bool) -> dict[str, Any]:
    has_hf = dataset in {"synthetic", "omniglot"}
    p1_holds = [bool(r["predictions"]["p1_sph_arccos_corr_gt_sph_hyp"]) for r in seed_results]
    p1_gaps = [float(r["summary"]["p1_corr_gap"]) for r in seed_results]

    agg: dict[str, Any] = {
        "dataset": dataset,
        "full": bool(full),
        "seeds": [r["seed"] for r in seed_results],
        "n_seeds": len(seed_results),
        "p1_hold_rate": float(np.mean(p1_holds)),
        "p1_corr_gap": _mean_std(p1_gaps),
        "warnings_any": sorted(
            {w for r in seed_results for w in r.get("warnings", [])}
        ),
        "thresholds": {"p1_gap": P1_GAP, "p2_delta": P2_DELTA},
        "p2_status_counts": {},
    }

    if has_hf:
        p0_holds = [
            bool(r["predictions"].get("p0_hierarchy_in_embedding")) for r in seed_results
        ]
        p0_hits = [
            float(r["summary"]["p0"]["parent_hit@5_excl_leaf"])
            for r in seed_results
            if "p0" in r.get("summary", {}) and "parent_hit@5_excl_leaf" in r["summary"]["p0"]
        ]
        statuses = [str(r["summary"].get("p2_status", "unknown")) for r in seed_results]
        # Interpretable P2 = hold|fail only (excludes not_applicable / weak_geometry).
        interpretable = [s for s in statuses if s in ("hold", "fail")]
        p2_hold_among_ok = (
            float(np.mean([s == "hold" for s in interpretable])) if interpretable else float("nan")
        )
        deltas = [
            float(r["summary"]["structure_interaction_delta"]) for r in seed_results
        ]
        # Legacy raw rate (ignores gate) — keep for continuity with older aggregates.
        p2_raw = [
            bool(r["predictions"]["p2_structure_gap_positive"]) for r in seed_results
        ]
        agg["p0_hold_rate"] = float(np.mean(p0_holds))
        agg["p0_parent_hit@5_excl_leaf"] = _mean_std(p0_hits) if p0_hits else None
        agg["p2_status_counts"] = {
            k: int(sum(1 for s in statuses if s == k))
            for k in sorted(set(statuses))
        }
        agg["p2_hold_rate_raw"] = float(np.mean(p2_raw))
        agg["p2_hold_rate_interpretable"] = p2_hold_among_ok
        # Prefer gated rate as the headline "p2_hold_rate".
        agg["p2_hold_rate"] = p2_hold_among_ok
        agg["structure_interaction_delta"] = _mean_std(deltas)
        agg["n_p2_interpretable"] = len(interpretable)
    else:
        agg["p0_hold_rate"] = None
        agg["p2_hold_rate"] = None
        agg["p2_hold_rate_raw"] = None
        agg["p2_hold_rate_interpretable"] = None

    # Recall@1 per head (average H/F when both exist).
    recall: dict[str, list[float]] = {h: [] for h in HEADS}
    for r in seed_results:
        if has_hf:
            for h in HEADS:
                vals = [
                    r["regimes"]["H"]["heads"][h]["recall@1"],
                    r["regimes"]["F"]["heads"][h]["recall@1"],
                ]
                recall[h].append(float(np.mean(vals)))
        else:
            for h in HEADS:
                recall[h].append(float(r["heads"][h]["recall@1"]))
    agg["recall@1"] = {h: _mean_std(recall[h]) for h in HEADS}
    return agg


def main() -> None:
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.", flush=True)
        args.device = "cpu"

    seeds = _parse_seeds(args)
    base = _results_root(args.dataset, args.full, args.smoke)
    base.mkdir(parents=True, exist_ok=True)

    seed_results: list[dict[str, Any]] = []
    for seed in seeds:
        out_dir = base / f"seed_{seed}"
        print(f"=== {args.dataset} seed={seed} full={args.full} -> {out_dir} ===", flush=True)
        res = _run_one_seed(
            dataset=args.dataset,
            seed=seed,
            full=args.full,
            smoke=args.smoke,
            device=args.device,
            workers=args.workers,
            out_dir=out_dir,
        )
        seed_results.append(res)
        summ = res.get("summary", {})
        print(
            json.dumps(
                {
                    "seed": seed,
                    "predictions": res["predictions"],
                    "falsified": res["falsified"],
                    "p2_status": summ.get("p2_status"),
                    "p0": summ.get("p0"),
                    "summary_keys": {
                        "p1_corr_gap": summ.get("p1_corr_gap"),
                        "structure_interaction_delta": summ.get(
                            "structure_interaction_delta"
                        ),
                    },
                    "warnings": res.get("warnings", []),
                    "elapsed_sec": round(res["elapsed_sec"], 1),
                    "artifacts": str(out_dir),
                },
                indent=2,
            ),
            flush=True,
        )

    if len(seed_results) > 1:
        agg = _aggregate(seed_results, args.dataset, args.full)
        agg_path = base / "aggregate.json"
        agg_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
        print(json.dumps({"aggregate": agg, "path": str(agg_path)}, indent=2), flush=True)
    else:
        # Single-seed convenience: also copy metrics to base/metrics.json pointer note.
        note = {
            "note": "single-seed run; see seed_*/metrics.json",
            "seed": seeds[0],
            "predictions": seed_results[0]["predictions"],
            "falsified": seed_results[0]["falsified"],
            "warnings": seed_results[0].get("warnings", []),
        }
        (base / "summary.json").write_text(json.dumps(note, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
