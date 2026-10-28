# Matched three-head embedding checks (MWE)

This repository contains the algorithms and materials used in the illustrative experiments described in ***Deep Metric Learning Beyond Euclidean Space: A Practical Guide to Riemannian Embedding Geometries***, submitted to *Acta Polytechnica Hungarica*.

The check trains the **same** encoder and supervised InfoNCE loss under three embedding heads — Euclidean, spherical ($\ell_2$ + cosine), Poincaré — and reports pre-registered diagnostics. It is an **illustrative** stress test of ranking disagreement and hierarchy–geometry interaction under matched slots. It is **not** a benchmark or SOTA comparison.

Code / formula provenance (kudos to the authors and maintainers):

- Supervised contrastive positive-mask semantics: [HobbitLong/SupContrast](https://github.com/HobbitLong/SupContrast) `losses.py::SupConLoss` [1]
- Loss scaffolding and class-balanced sampling: [KevinMusgrave/pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning) (`SupConLoss`, `LpDistance`, `CosineSimilarity`, `MPerClassSampler`) [2]
- Poincaré ball primitives (formulas / numeric guards): [geoopt/geoopt](https://github.com/geoopt/geoopt) `geoopt/manifolds/stereographic/math.py` [3]; vendored `hyptorch/pmath.py` in [YunYunY/HypMix](https://github.com/YunYunY/HypMix) [4]
- Omniglot folder layout / alphabet–character metadata: [pytorch/vision](https://github.com/pytorch/vision) `torchvision/datasets/omniglot.py`; dataset release [brendenlake/omniglot](https://github.com/brendenlake/omniglot) [5]
- CIFAR-10: [Toronto CIFAR page](https://www.cs.toronto.edu/~kriz/cifar.html) via `torchvision.datasets.CIFAR10` [6]

## What is matched

Only the geometry map and the distance inside the loss change between heads. Shared across heads:

- encoder architecture and initialization seed family
- temperature $\tau = 0.1$
- Adam, learning rate, epoch budget
- batch construction (`MPerClassSampler`)
- no data augmentation (ToTensor + normalize only)

## Datasets

| Flag | Content | Diagnostics |
| --- | --- | --- |
| `synthetic` | Hierarchical (H) vs flat (F) Gaussian class means, 27 leaves | P1 + P0 + gated P2 |
| `cifar10` | CIFAR-10 (quick subset by default; `--full` = 50k train) | P1 + Recall@1 by head |
| `omniglot` | Matched H/F arms (quick: $3\times 9=27$ classes; `--full`: $5\times 9=45$ classes using background+evaluation alphabets) | P1 + P0 + gated P2 |

### Predictions / gates

- **P1 (ranking):** On ambient features of the *spherical* head, Spearman correlation of pairwise cosine vs spherical geodesic $\arccos$ is higher than cosine vs Poincaré distance (threshold: gap $> 0.05$).
- **P0 (hierarchy-in-embedding, validity gate):** On the spherical head of regime **H**, `parent_hit@5_excl_leaf` (fraction of top-5 *other-leaf* neighbors that share the parent) must beat chance by $+0.15$. Same-leaf neighbors are masked so leaf Recall@1 cannot inflate the score. Parents: synthetic = immediate tree parent; Omniglot = alphabet.
- **P2 (structure × geometry):** $\Delta = (\mathrm{Hyp}-\mathrm{Sph})_{\mathrm{H}} - (\mathrm{Hyp}-\mathrm{Sph})_{\mathrm{F}}$ on Recall@1 (threshold: $> 0.02$). Absolute hyperbolic superiority is **not** claimed.

`summary.p2_status` is one of:

| Status | Meaning |
| --- | --- |
| `hold` / `fail` | P0 passed and Poincaré not near-boundary → interpret $\Delta$ |
| `not_applicable` | P0 failed (no usable hierarchy in embeddings) → do not treat P2 as a geometry test |
| `weak_geometry` | Poincaré radius warning (`hyp_frac_near_boundary` $> 0.5$) → numeric health first |

Multi-seed `aggregate.json` reports `p0_hold_rate`, `p2_status_counts`, and `p2_hold_rate` = hold rate among *interpretable* seeds only (`hold`/`fail`).

Omniglot images are resized to $28\times 28$ in this repo (implementation choice; not a claim about the official release size). Omniglot `--full` enlarges the arms; epoch count stays matched to quick (20).

## Setup

```bash
cd dml-beyond-euclidean
python -m pip install -r requirements.txt
```

Data → `data/` (gitignored). Results → `results/<dataset>/seed_<seed>/` or `results/<dataset>_full/seed_<seed>/` (gitignored). Multi-seed runs also write `aggregate.json`.

## Smoke test (do this first)

```bash
python smoke_test.py
```

Optional 1-epoch dry runs:

```bash
python run.py --dataset synthetic --smoke
```

## Full runs (multi-core VM)

Copy / sync this folder to the VM, then:

```bash
cd dml-beyond-euclidean
python -m pip install -r requirements.txt
chmod +x run_all.sh
./run_all.sh --workers 16 --seeds 20260924,20260925,20260926 --full
```

Or step by step (recommended after a code update that changes metrics):

```bash
python smoke_test.py
python run.py --dataset synthetic --seeds 20260924,20260925,20260926 --workers 16
python run.py --dataset omniglot --seeds 20260924,20260925,20260926 --workers 16
python run.py --dataset omniglot --full --seeds 20260924,20260925,20260926 --workers 16
# optional (P1 only; not needed for P0/P2 validity):
python run.py --dataset cifar10 --seeds 20260924,20260925,20260926 --workers 16
python run.py --dataset cifar10 --full --seeds 20260924,20260925,20260926 --workers 16
```

When finished, send back `logs/run_*.log` and `results/**/aggregate.json` (plus any `metrics.json` you care about). Look for `summary.p0` and `summary.p2_status` in each seed’s `metrics.json`.

Optional flags: `--device cpu|cuda`, `--workers N`, `--seed N` (single seed).

### Re-run after adding P0?

Yes for **synthetic** and **omniglot** (quick + `--full`): P0 and gated `p2_status` are new fields; old `metrics.json` files do not contain them, so prior P2 “fail/hold” counts are not comparable under the gate.

**CIFAR-10** is optional for this change (no H/F → no P0/P2). Re-run only if you want a fresh P1 aggregate alongside the others.

Training itself is unchanged (same loss/heads/epochs); only evaluation/summary logic was extended. Overwriting `results/synthetic/`, `results/omniglot/`, and `results/omniglot_full/` is fine — dirs are tier-separated.

## Layout

```
smoke_test.py     # download-free wiring checks (includes P0 unit test)
run_all.sh        # VM suite + log
geometry.py       # maps + pairwise distances
losses.py         # PoincareDistance + make_loss(head)
models.py         # MLPEncoder, SmallConvEncoder
data_*.py         # synthetic / CIFAR-10 / Omniglot loaders
train.py          # single-head training loop
metrics.py        # Recall@k, rank corrs, P0 parent-hit, radius stats
plots.py          # rank + retrieval figures (per seed)
run.py            # CLI entry point (P1 + gated P2)
manuscript_summary.py  # read-only aggregation over results/**/seed_*/metrics.json -> paper figures + table body
```

## Manuscript figures

```bash
python manuscript_summary.py --out ../manuscript/latex/figures/mwe
```

Writes `mwe_summary.json` (per-regime, per-head Recall@1 mean/std; rank-correlation gaps; P0 hits; $\Delta$; status counts; near-boundary fractions), `fig_mwe_rank.png`, `fig_mwe_recall.png` and `table_body.tex`. It does not retrain or recompute metrics.

# References

[1] Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, Dilip Krishnan. [Supervised Contrastive Learning](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html), Advances in Neural Information Processing Systems 33 (2020): 18661-18673. Code: https://github.com/HobbitLong/SupContrast

[2] Kevin Musgrave, Serge Belongie, Ser-Nam Lim. [PyTorch Metric Learning](https://arxiv.org/abs/2008.09164), arXiv preprint arXiv:2008.09164 (2020). Code: https://github.com/KevinMusgrave/pytorch-metric-learning

[3] Max Kochurov, Rasul Karimov, Serge Kozlukov. [Geoopt: Riemannian Optimization in PyTorch](https://arxiv.org/abs/2005.02819), arXiv preprint arXiv:2005.02819 (2020). Code: https://github.com/geoopt/geoopt

[4] Yun Yue, Fangzhou Lin, Guanyi Mou, Ziming Zhang. [Understanding Hyperbolic Metric Learning through Hard Negative Sampling](https://doi.org/10.1109/WACV57701.2024.00190), Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) (2024): 1891-1903. Code: https://github.com/YunYunY/HypMix

[5] Brenden M. Lake, Ruslan Salakhutdinov, Joshua B. Tenenbaum. [Human-level Concept Learning through Probabilistic Program Induction](https://doi.org/10.1126/science.aab3050), Science 350.6266 (2015): 1332-1338. Dataset: https://github.com/brendenlake/omniglot

[6] Alex Krizhevsky. [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/cifar.html), Technical Report, University of Toronto (2009).
