# ATA Model (Adaptive Time Alignment)

ATA is a temporal domain adaptation pipeline for IoT attack classification.
This implementation uses:
- Classification loss
- Transfer loss to reduce domain shift over time

The pipeline is designed to be easy to reimplement and portable across machines.

## Code Structure
- `ata_model/ata_main.py`: train/eval entrypoint, config loading, artifact saving.
- `ata_model/components/data/ata_dataloader.py`: multi-domain dataloaders (`create_ata_dataloaders`, `get_ata_domain_pairs`).
- `ata_model/components/models/adarnn/model.py`: model wrappers.
- `ata_model/components/models/adarnn/loss_transfer.py`: transfer losses (mmd/coral/adv/cosine...).
- `ata_model/components/models/adarnn/tdc.py`: temporal domain characterization.

## Required Input Data
- CSV format.
- Label column (`label_col`, default `label`).
- Time column (`time_col`, default `time_sec`) when `load_mode=time_series`.
- Remaining columns are treated as features when `feature_cols: null`.

## Quick Config
Template file: `configs/ATA_model.yaml`.

Key fields:
- `data_config.train_data_path`, `val_data_path`, `test_data_path`: split files.
- `data_config.load_mode`: `time_series` or `random_shuffle`.
- `data_config.split_method`: `quantile`, `tdc`, `manual`.
- `data_config.feature_cols`: supports two formats
  - YAML list: `feature_cols: [f1, f2, f3]`
  - CLI comma list: `--feature_cols f1,f2,f3`
- `train_config.dw`: transfer-loss weight.
- `model_config.loss_type`: `mmd`, `mmd_rbf`, `coral`, `adv`, `cosine`.

## Run
From the project root `IPv6_detect_attack_by_AI`:

```bash
python ata_model/ata_main.py --config configs/ATA_model.yaml
```

Or override via CLI:

```bash
python ata_model/ata_main.py \
  --data_dir data \
  --output_dir outputs/ata_model \
  --n_epochs 20 \
  --num_domains 3 \
  --loss_type mmd \
  --dw 0.5 \
  --seed 42
```

## Output
After training, artifacts are saved in a run directory, for example:
`outputs/ata_model/ata_mmd_domains3_dw0.5/`

- `best_model.pth`
- `results.json`

Notes:
- Classification report and confusion matrix are printed to the console.
- Run configuration is stored in `best_model.pth` (`args`) and `results.json`.

## Reimplementation Checklist
- Train/val/test data share the same schema.
- `label_col` matches your dataset.
- Manually specified `feature_cols` exist in the CSV.
- For imbalanced data, try `use_class_weights=true`.
