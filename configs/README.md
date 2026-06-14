# Configs

YAML templates for the four main pipelines in this repository.
All sample paths are relative so the project can be reused on other machines.

## Files
- `Deep_learning.yaml`: train DL models (rnn/lstm/gru/cnn/tcn_transformer/mlp/dnn/rad_ffnn).
- `Distill_model.yaml`: train student models with knowledge distillation.
- `Machine_learning.yaml`: baseline ML (logistic/svm/rf/knn).
- `ATA_model.yaml`: ATA temporal domain adaptation.
- `dataset{1,3}_multi_{50,60,80,100}.yaml`: time-series multiclass benchmark configs.
- `Machine_learning_dataset{1,3}_{50,60,80,100}.yaml`: ML benchmark configs.
- `benchmark.yaml`: benchmark matrix for Python benchmark runner.

## Notes
- DL/ML/Distillation assume the last CSV column is the label unless `label_col` is provided.
- ATA uses explicit column names (`time_col`, `label_col`) and does not depend on column position.
- `feature_cols` in ATA supports:
  - Comma-separated CLI input: `--feature_cols f1,f2,f3`
  - YAML list: `feature_cols: [f1, f2, f3]`
- Teacher checkpoints in `Distill_model.yaml` must match architecture and preprocessing (PCA/time embedding).
