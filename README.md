# Curvature-Aware Neural Networks (CANN) — Code

This repository provides runnable code to reproduce core experiments for Curvature-Aware Neural Networks (CANN).

## Quick Start
- Python 3.9+
- Install dependencies:
  - `pip install -r requirements.txt`
- Prepare datasets (optional; scripts auto-download when needed):
  - `python scripts/datasets/prepare_datasets.py --real`
  - `python scripts/datasets/prepare_datasets.py --synthetic`

## Run Experiments
- Node classification (Planetoid Cora/CiteSeer/PubMed):
  - `python scripts/experiments/train_node.py`
- Link prediction (Cora + optional Facebook/Twitter if present):
  - `python scripts/experiments/train_link.py`
- Graph regression (ZINC subset):
  - `python scripts/experiments/train_graph.py`
- Ablation study:
  - `python scripts/experiments/ablation.py`
- Robustness evaluation:
  - `python scripts/experiments/robustness.py`
- Scalability timings:
  - `python scripts/experiments/scalability.py`
- End-to-end convenience runner:
  - `python scripts/experiments/run_all.py`

Outputs are written to `data/*.json`.

## Code Structure
- `scripts/experiments/`
  - `models.py` — GCN, GAT, and CANN implementations
  - `curvature.py` — approximate node curvature utility
  - `train_*.py` — training and evaluation scripts
  - `ablation.py`, `robustness.py`, `interpretability.py`, `scalability.py`
- `scripts/datasets/`
  - `generate_synthetic.py` — synthetic graph generators (SBM, BA, mixed)
  - `download_real.py` — downloads Planetoid and SNAP datasets, optional OGB
  - `prepare_datasets.py` — orchestration script
- `requirements.txt` — dependencies

## Notes
- torch_geometric may require platform-specific wheels; follow official install docs if `pip install` fails.
- OGB is optional (only used for molecular dataset helper): if you do not need OGB you can skip installing it.

## License
See LICENSE.
