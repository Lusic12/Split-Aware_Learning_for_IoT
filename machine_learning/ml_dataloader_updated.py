"""
Simple ML dataloader.
- Default: use all columns except the last as features, last column as label.
- Optional: feature list/label name from config for backward compatibility.
- Optional: PCA pass (fit on train, transform val/test).
"""
import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Add parent directory to path to allow imports from data package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataloader import load_config

__all__ = ['get_ml_data', 'load_ml_config', 'get_ml_data_from_paths', 'load_data']


def _split_xy(df: pd.DataFrame, features=None, label_col=None):
    """
    Split features/label. Priority: explicit label_col -> explicit features -> all minus last.
    """
    if label_col is not None and label_col in df.columns:
        y = df[label_col]
        X = df.drop(columns=[label_col])
    elif features:
        X = df[features]
        y = df.iloc[:, -1]
    else:
        # last column as label by default
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    return X, y


def _maybe_normalize(train_df, val_df, test_df, feature_cols, normalize: bool):

    if feature_cols[-1] == 'category':
        feature_cols = feature_cols[:-1] # Remove the last column
    print(feature_cols)
    if not normalize:
        return train_df[feature_cols], val_df[feature_cols], test_df[feature_cols], None
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(train_df[feature_cols]), columns=feature_cols)
    X_val = pd.DataFrame(scaler.transform(val_df[feature_cols]), columns=feature_cols)
    X_test = pd.DataFrame(scaler.transform(test_df[feature_cols]), columns=feature_cols)
    return X_train, X_val, X_test, scaler


def _maybe_pca(X_train, X_val, X_test, pca_components):
    if pca_components is None:
        return X_train, X_val, X_test, None
    pca = PCA(n_components=pca_components)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    cols = [f"pca_{i}" for i in range(X_train_pca.shape[1])]
    return (
        pd.DataFrame(X_train_pca, columns=cols),
        pd.DataFrame(X_val_pca, columns=cols),
        pd.DataFrame(X_test_pca, columns=cols),
        pca,
    )


def get_ml_data_from_paths(
    train_path,
    val_path,
    test_path,
    normalize=True,
    pca_components=None,
    label_col=None,
):
    """
    Load data from explicit train/val/test paths.
    - Features: all columns except last (default) or drop label_col if provided.
    - Label: last column (default) or label_col.
    - normalize: StandardScaler fit on train, apply to val/test.
    - pca_components: None to skip; int or float<1 for PCA (fit on train, apply val/test).
    """
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    X_train, y_train = _split_xy(train_df, features=None, label_col=label_col)
    X_val, y_val = _split_xy(val_df, features=None, label_col=label_col)
    X_test, y_test = _split_xy(test_df, features=None, label_col=label_col)

    feature_cols = X_train.columns.tolist()
    X_train, X_val, X_test, _ = _maybe_normalize(train_df, val_df, test_df, feature_cols, normalize)
    X_train, X_val, X_test, _ = _maybe_pca(X_train, X_val, X_test, pca_components)
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_ml_config(config_path="configs/Machine_learning.yaml", task="binary", mode="full_feature"):
    """
    Load machine learning configuration.
    Supports:
      - New simple schema (top-level keys like task_type/data_config)
      - Legacy nested schema config[task][mode]
    """
    config = load_config(config_path)
    if task in config and isinstance(config[task], dict):
        section = config[task]
        if mode in section:
            return section[mode]
        return section
    return config


def get_ml_data(
    config_path="configs/Machine_learning.yaml",
    task="binary",
    mode="full_feature",
    normalize=True,
    pca_components=None,
    label_col=None,
    train_path=None,
    val_path=None,
    test_path=None,
):
    """
    Unified entry:
    - If train_path/val_path/test_path are provided: load directly (last column is label by default).
    - Else: load config, use features if present; fallback to all columns except last.
    - normalize: StandardScaler on train -> val/test.
    - pca_components: None to skip, else int or float<1.
    """
    if train_path and val_path and test_path:
        return get_ml_data_from_paths(train_path, val_path, test_path, normalize, pca_components, label_col)

    cfg = load_ml_config(config_path, task, mode)
    train_df = pd.read_csv(cfg['train_data_path'])
    val_df = pd.read_csv(cfg['val_data_path'])
    test_df = pd.read_csv(cfg['test_data_path'])

    features = cfg.get('features')
    label_from_config = cfg.get('label') or cfg.get('label_column')
    target_col = label_col or label_from_config
    X_train, y_train = _split_xy(train_df, features=features, label_col=target_col)
    X_val, y_val = _split_xy(val_df, features=features, label_col=target_col)
    X_test, y_test = _split_xy(test_df, features=features, label_col=target_col)

    feature_cols = X_train.columns.tolist()
    X_train, X_val, X_test, _ = _maybe_normalize(train_df, val_df, test_df, feature_cols, normalize)

    # Enable PCA when mode == 'pca' or when pca_components provided
    if mode == 'pca' and pca_components is None:
        pca_components = 0.95
    X_train, X_val, X_test, _ = _maybe_pca(X_train, X_val, X_test, pca_components)

    return X_train, y_train, X_val, y_val, X_test, y_test


# Legacy alias
def load_data(config_path="configs/ml_config.yaml", task="binary", mode="full_feature", normalize=True):
    return get_ml_data(config_path=config_path, task=task, mode=mode, normalize=normalize)
