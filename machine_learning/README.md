# Machine Learning

Baseline ML scripts that use CSV input (last column is label by default).

## Scripts
- `logistic_updated.py`
- `svm_updated.py`
- `random_forest_updated.py`
- `knc_updated.py`
- `ml_dataloader_updated.py` (load data, normalize, PCA)

## Examples
```bash
python machine_learning/logistic_updated.py --task binary --mode full_feature --balance
python machine_learning/svm_updated.py --task multiclass --mode full_feature --kernel rbf --balance
python machine_learning/knc_updated.py --task binary --mode pca --find_k
```

## Config
- Use `configs/Machine_learning.yaml`, or pass CSV paths directly through CLI.
