# Reimplementation Guide

Use this guide to clone the repo and reproduce results on a new machine.

## 1) Environment
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2) Recommended Data Layout
Place your data under `data/` (or update `configs/*.yaml`):

```text
data/
  train.csv
  val.csv
  test.csv
  binary_train.csv
  binary_val.csv
  binary_test.csv
```

Column conventions:
- DL/ML/Distillation: last column is label (unless `label_col` is explicitly provided).
- ATA: uses explicit `time_col` and `label_col` from `configs/ATA_model.yaml`.

## 3) Run Each Pipeline

Deep Learning:
```bash
python dl_main.py --config configs/Deep_learning.yaml --model tcn_transformer
```

Distillation:
```bash
python dl-student_main.py --config configs/Distill_model.yaml --model gru --teacher-model rnn --teacher-checkpoint weights/best_rnn_binary_model.pth
```

ATA:
```bash
python ata_model/ata_main.py --config configs/ATA_model.yaml
```

Machine Learning baseline:
```bash
python machine_learning/logistic_updated.py --config configs/Machine_learning.yaml --task binary --mode full_feature --balance
```

Inference:
```bash
python inference.py --config configs/Deep_learning.yaml --checkpoint weights/best_tcn_transformer_binary_model.pth --split test
```

## 4) Check Outputs
- DL: model/checkpoints in `weights/` (or custom `logging.model_save_path`).
- Distillation: outputs in `outputs/distill/` (or `--out-dir`).
- ATA: outputs in `outputs/ata_model/ata_<loss>_domains<num>_dw<dw>/`.
- ML: metrics in console and optional script-specific artifacts.

## 5) Common Errors
- File not found: update `train_data_path/val_data_path/test_data_path` in YAML.
- Distillation cannot load teacher: checkpoint does not match architecture/PCA/time embedding.
- ATA `feature_cols` error: one or more columns do not exist in CSV.
- Import errors: run scripts from project root `IPv6_detect_attack_by_AI/`.

## 6) Checklist Before Sharing Results
- Include the exact config used.
- Set and report a seed (`seed`) for reproducibility.
- Report validation/test metrics with the same definitions (Accuracy, F1 Macro).
- Include Python + torch + sklearn versions.
