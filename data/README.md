# Data Module

Dataloader code and data-related scripts.

## Files
- `dataloader.py`: load train/val/test, normalization, PCA, and sequence windowing.
- `visualize_time_embedding_anomaly.py`: time-embedding visualization from `raw_combined.csv`.
- `__init__.py`: package init.

## Conventions
- Default label is the last CSV column.
- For sequence models, sequences are created with `sequence_length` and `stride`.
- If a `category` column exists right before label, it is excluded from features.
- Dataloader reports label distribution for both raw labels and sequence-window labels (via artifacts).

## Visualization script
`visualize_time_embedding_anomaly.py` supports explicit input paths:

```bash
python data/visualize_time_embedding_anomaly.py \
  --input data/raw_combined.csv \
  --spec data/anomaly_reasoning_binary.yaml \
  --out-dir outputs/time_embedding_viz
```

Notes:
- Prefer explicit `--input` and `--spec` instead of relying on defaults.
- The `spec` file must include a valid `label_map`.
