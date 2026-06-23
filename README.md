# MST-GNN: Graph Representation Learning of Multilayer Spatial-Temporal Networks for Stock Predictions

> **Paper:** "Graph Representation Learning of Multilayer Spatial–Temporal Networks for Stock Predictions"  
> **Venue:** IEEE Transactions on Computational Social Systems, 2024  
> **Authors:** Hu Tian, Xingwei Zhang, Xiaolong Zheng, Zili Zhang, Daniel Dajun Zeng

## Overview

This repository contains a full PyTorch reimplementation of the MST-GNN framework for stock market prediction. The model constructs multilayer spatial-temporal stock networks and uses a novel cross-layer high-order fusion mechanism to capture complex evolutionary information from multifaceted and time-varying financial networks.

### Architecture

```
Raw OHLCV Features (T×d)
    │
    ▼
[Module A] Attentive LSTM Encoder ──→ h_i ∈ R^d1
    │
    ▼
[Module B] Spatial-Temporal Neighborhood Aggregation (STNA)
    │   ├── Shareholding Network
    │   ├── Industry Network
    │   ├── Topicality Network
    │   └── Comovement Network
    │
    ▼
[Module C] Cross-Layer High-Order Feature Fusion (HOFF/DCN)
    │   ├── Cross Network (explicit high-order interactions)
    │   └── Deep Network (implicit nonlinear features)
    │
    ▼
[Module D] Multitask Prediction
    ├── Movement Classification (up/down)
    └── Return Ranking (pairwise)
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- PyTorch Geometric 2.4+
- Google Colab T4 GPU (free tier compatible)

### Install Dependencies

```bash
pip install -r requirements.txt

# PyTorch Geometric (install matching your CUDA version)
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch-geometric
```

## Quick Start

### 1. Run Full Experiment (CSI 300)

```bash
python -m experiments.run_main --dataset csi300 --aggregator mean
```

### 2. Run All Aggregator Variants

```bash
python -m experiments.run_main --dataset csi300 --aggregator all
```

### 3. Run Ablation Study

```bash
python -m experiments.run_ablation --dataset csi300
```

### 4. Run Network Combination Analysis

```bash
python -m experiments.run_network_analysis --dataset csi300
```

## Project Structure

```
mst-gnn-impl/
├── config.py                  # Central hyperparameters
├── requirements.txt           # Dependencies
├── train.py                   # Training loop
├── evaluate.py                # Evaluation script
├── backtest.py                # Trading simulation
│
├── data/
│   ├── collector.py           # AKShare data fetching
│   ├── preprocessing.py       # 13-feature engineering
│   ├── graph_builder.py       # 4 multilayer graph construction
│   └── dataset.py             # PyTorch Dataset
│
├── models/
│   ├── feature_encoder.py     # Module A: Attentive LSTM
│   ├── stna.py                # Module B: STNA (3 aggregators)
│   ├── hoff.py                # Module C: DCN feature fusion
│   ├── predictor.py           # Module D: Multitask heads
│   └── mst_gnn.py             # Full model assembly
│
├── experiments/
│   ├── run_main.py            # Main experiment (Tables IV-V)
│   ├── run_ablation.py        # Ablation study (Table VI)
│   └── run_network_analysis.py # Network combos (Fig. 7)
│
└── utils/
    ├── metrics.py             # Accuracy, Precision, DAMRR
    ├── visualization.py       # Plotting utilities
    └── logger.py              # Logging setup
```

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| T | 5 | Historical lookback window |
| d | 13 | Number of stock features |
| d₁ | 64 | LSTM hidden dimension |
| K | 2 | STNA aggregation depth |
| C | 3 | Cross network layers |
| δ | 0.5 | Multitask loss weight |
| LR | 1e-3 | Learning rate |

## Data Sources

This implementation uses **AKShare** (free, no API key required) for:
- Daily OHLCV stock prices
- Industry classification (East Money)
- Shareholding relationships
- Financial news (for topicality network)

## Evaluation Metrics

- **Accuracy**: Movement direction classification
- **Precision**: Positive class (price up) precision
- **DAMRR**: Daily Average Mean Reciprocal Rank (Eq. 20)

## Citation

```bibtex
@article{tian2024graph,
  title={Graph Representation Learning of Multilayer Spatial-Temporal Networks for Stock Predictions},
  author={Tian, Hu and Zhang, Xingwei and Zheng, Xiaolong and Zhang, Zili and Zeng, Daniel Dajun},
  journal={IEEE Transactions on Computational Social Systems},
  year={2024},
  publisher={IEEE}
}
```

## License

This implementation is for educational and research purposes only. Not financial advice.
