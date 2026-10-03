# Spatiotemporal Graph Attention for Temporal and Spatial Generalization of Station-Based Climate Forecasting in Bangladesh

PyTorch pipeline for one-month-ahead forecasting of maximum/minimum temperature, humidity, and rainfall across 35 BMD stations using a spatiotemporal graph attention network (GAT over geographic neighbours + BiLSTM with temporal attention).

## Models

| ID | Architecture | Role |
|---|---|---|
| `st_gat_attn` | GAT + BiLSTM + temporal attention | Our Hybrid Model |
| `b1` | BiLSTM + temporal attention (station-independent) | No-graph ablation |
| `b2` | Neighbour-mean + BiLSTM + temporal attention | Naive spatial ablation |
| `b3` | Conv1D + BiLSTM + temporal attention (HTC-style) | Stronger temporal front-end |
| `stgcn` / `astgcn` / `st_transformer` | External ST-GNN baselines | Community comparison |

All models share the same input window, multi-target heads, loss, and train/val/test splits — only the spatiotemporal encoder changes.

## Evaluation protocols

- **Temporal** — train 1950–2014, val 2015–2019, test 2020–2023 (all stations).
- **Spatial** — 7 stations held out (Barisal, Bhola, Hatiya, M. Court, Mymensingh, Satkhira, Sylhet).


## Setup

```bash
pip install -r requirements.txt
# optional CUDA PyTorch:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Quick start

```bash
# prepare data + train the headline model (temporal split)
python train.py --prepare-data --model st_gat_attn --split temporal

# spatial hold-out
python train.py --model st_gat_attn --split spatial --k-neighbors 8

# full experiment matrix (B1/B2/B3/ST-GAT-Attn × temporal/spatial)
python run_experiments.py --prepare-data --epochs 30

# external ST-GNN baselines
python run_st_baselines.py --models stgcn astgcn st_transformer

# hyperparameter sweep (k, d, lr, layers)
python sweep.py --epochs 25
```



## Key files

```
train.py                  # single-model training entry point
run_experiments.py       # full experiment matrix
run_st_baselines.py       # external ST-GNN baselines
sweep.py                  # hyperparameter sweep
st_gat_attn/              # model + data + training package
configs/default.yaml     # all hyperparameters

```
