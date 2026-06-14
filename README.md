# Split-Aware Learning for IoT Intrusion Detection under Temporal Domain Shift

Reference implementation for the study *"Split-Aware Learning for IoT Intrusion
Detection under Temporal Domain Shift."* The repository provides unified
training, evaluation, benchmarking, and inference pipelines for RPL/IPv6 IoT
intrusion detection, together with the **Adaptive Temporal Alignment (ATA)**
domain-adaptation method.

## Motivation
Random train–test partitioning is still common in RPL-IoT IDS evaluation. When
traffic is time-ordered, shuffling places neighboring temporal samples in both
the training and test sets, so high scores reflect repeated-pattern recognition
rather than robustness to future traffic. This codebase makes the **split effect
measurable**: feature-level train–test drift is near zero under random-full
splits but rises one to two orders of magnitude under chronological
retained-prefix splits, revealing a temporal domain shift that a deployed
detector would actually face.

To address that shift we propose **Adaptive Temporal Alignment (ATA)**, which
partitions the chronological training stream into temporal domains and aligns
their latent representations while preserving class discrimination, without
using future labels. ATA is lightweight (~50K parameters) and improves macro-F1
over DANN and CDAN where temporal mismatch is largest.

## Highlights
- Reveals systematic evaluation bias in random-split IoT intrusion detection.
- Temporal (chronological retained-prefix) splits expose severe domain shift in
  RPL-based networks.
- Proposes **ATA** for robust detection under temporal shift.
- Improves macro-F1 over strong baselines (DANN, CDAN) where temporal mismatch
  is largest, at ~50K parameters.
- Introduces a split-aware, deployment-oriented IDS benchmark with
  per-configuration confidence intervals and significance tests.

## Datasets
Two emulator-generated RPL routing-attack benchmarks with contrasting
temporal-difficulty profiles are used. Both are publicly available — download
them and place the prepared CSV splits under `data/` (raw data is not committed
to this repository).

| Benchmark | Description | Source |
|-----------|-------------|--------|
| **ROUT-04** | From the ROUT-4-2023 RPL-based routing-attack dataset | https://ieee-dataport.org/documents/rout-4-2023-rpl-based-routing-attack-dataset-iot |
| **RPL-Beh** | RPL-IDS behavior dataset | https://github.com/HUNSR/RPL-IDS-Behavior-Dataset |

Evaluation uses chronological **retained-prefix** splits at multiple history
ratios (50 / 60 / 80 / 100) for both binary and multiclass tasks.

## Repository Layout
```text
.
|-- ata_model/               # ATA (Adaptive Temporal Alignment) model + data/model components
|-- configs/                 # YAML configs for DL/ML/Distillation/ATA + benchmark splits
|-- data/                    # Core dataloader and data utilities (raw CSVs not tracked)
|-- docs/                    # Project structure and operation notes
|-- machine_learning/        # Logistic/SVM/RF/KNN baselines
|-- models/                  # DL model definitions
|-- notebooks/               # Analysis notebooks (drift / benchmark / splitting)
|-- support/                 # Training and evaluation helpers
|-- synthetic_data/          # Synthetic/condensed data artifacts
|-- tests/                   # Smoke tests
|-- Distillation_main.py
|-- dl_main.py
|-- dl_main_extended.py
|-- train_timeseries.py
|-- inference.py
|-- benchmark.py
`-- requirements.txt
```
Details and conventions: `docs/PROJECT_STRUCTURE.md`.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data Conventions
- CSV label is expected in the last column for most DL/ML flows.
- If a `category` column exists right before the label, it is excluded from features.
- Sequence models use window labels from the last row in each window.
- The dataloader includes class-coverage checks (raw labels + sequence window labels).

## Main Entry Points
- `dl_main.py` — train/evaluate DL models (`dnn`, `mlp`, `rnn`, `lstm`, `gru`, `cnn`, `rad_ffnn`, `tcn_transformer`).
- `train_timeseries.py` — dedicated time-series training path.
- `Distillation_main.py` — teacher–student distillation training.
- `inference.py` — checkpoint inference on a train/val/test split.
- `benchmark.py` — Python benchmark runner.
- `ata_model/ata_main.py` — **ATA** training/evaluation and DANN/CDAN baselines.

## Quick Start
### 1) Deep Learning baseline
```bash
python dl_main.py --config configs/Deep_learning.yaml --model gru
```

### 2) Chronological retained-prefix configs (50 / 60 / 80 / 100)
```bash
python dl_main.py --config configs/dataset1_multi_50.yaml --model rnn
python dl_main.py --config configs/dataset3_multi_100.yaml --model gru
```

### 3) Adaptive Temporal Alignment (ATA)
```bash
python ata_model/ata_main.py --config configs/ATA_model.yaml
```

### 4) Distillation
```bash
python Distillation_main.py --config configs/Distill_model.yaml
```

### 5) Inference
```bash
python inference.py --config configs/Deep_learning.yaml --checkpoint weights/best_gru_binary_model.pth --split test
```

### 6) ML baselines
```bash
python machine_learning/logistic_updated.py --task binary --mode full_feature --config configs/Machine_learning.yaml --balance
```

### 7) Benchmark
```bash
python benchmark.py --config configs/Deep_learning.yaml --benchmark configs/benchmark.yaml --save_path benchmark_results
```

## Outputs
- Weights/checkpoints: `weights/`
- Benchmark results: `benchmark_results/`

## Testing
```bash
python -m pytest tests
```

## Citation
If you use this code or benchmark protocol, please cite:

```bibtex
@article{phan6598378split,
  title={Split-Aware Learning for IoT Intrusion Detection under Temporal Domain Shift},
  author={Phan, Anh-Minh and Le, Khanh Toan Dang and Vu, Thai-Huy and Vo, Son Que},
  journal={Available at SSRN 6598378}
}
```

## Related Docs
- `docs/PROJECT_STRUCTURE.md`
- `REIMPLEMENTATION_GUIDE.md`
- `ata_model/README.md`
- `configs/README.md`
- `data/README.md`
- `machine_learning/README.md`
- `models/README.md`
- `notebooks/README.md`
- `support/README.md`
- `tests/README.md`
