import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


def load_config(config_path):
    """Load YAML/JSON config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _split_features_and_labels(df, label_column=None, feature_columns=None):
    """
    Strict rule: the last column is the label, all previous columns are features.
    label_column/feature_columns are ignored by design.
    """
    label_name = df.columns[-1]
    labels = df[label_name].to_numpy()

    features_df = df.iloc[:, :-1]
    if len(df.columns) >= 2 and df.columns[-2] == "category":
        print("Data have category split")
        features_df = df.iloc[:, :-2]

    return features_df, labels


def _normalize_features(train_features, val_features, test_features):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    val_scaled = scaler.transform(val_features)
    test_scaled = scaler.transform(test_features)
    return train_scaled, val_scaled, test_scaled, scaler


def _apply_pca(train_features, val_features, test_features, n_components):
    pca = PCA(n_components=n_components)
    train_pca = pca.fit_transform(train_features)
    val_pca = pca.transform(val_features)
    test_pca = pca.transform(test_features)
    feature_names = [f"pca_{i}" for i in range(train_pca.shape[1])]
    return train_pca, val_pca, test_pca, feature_names, pca


def save_preprocess_artifacts(artifacts, out_dir, tag=None):
    if not artifacts or not out_dir:
        return

    os.makedirs(out_dir, exist_ok=True)
    prefix = f"{tag}_" if tag else ""

    scaler = artifacts.get("scaler")
    if scaler is not None:
        joblib.dump(scaler, os.path.join(out_dir, f"{prefix}scaler.pkl"))

    pca = artifacts.get("pca")
    if pca is not None:
        joblib.dump(pca, os.path.join(out_dir, f"{prefix}pca.pkl"))

    metadata = {k: v for k, v in artifacts.items() if k not in ("scaler", "pca")}
    with open(os.path.join(out_dir, f"{prefix}preprocess.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def load_preprocess_artifacts(artifact_dir, tag=None):
    prefix = f"{tag}_" if tag else ""
    artifacts = {}

    meta_path = os.path.join(artifact_dir, f"{prefix}preprocess.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            artifacts.update(json.load(f))

    scaler_path = os.path.join(artifact_dir, f"{prefix}scaler.pkl")
    if os.path.exists(scaler_path):
        artifacts["scaler"] = joblib.load(scaler_path)

    pca_path = os.path.join(artifact_dir, f"{prefix}pca.pkl")
    if os.path.exists(pca_path):
        artifacts["pca"] = joblib.load(pca_path)

    return artifacts


def _build_loader_kwargs(data_cfg):
    """Derive performant DataLoader kwargs with GPU-friendly defaults."""
    max_workers = os.cpu_count() or 8
    default_workers = min(8, max_workers) if torch.cuda.is_available() else 0
    num_workers = max(0, int(data_cfg.get("num_workers", default_workers)))
    pin_memory = bool(data_cfg.get("pin_memory", torch.cuda.is_available()))
    persistent_workers = bool(data_cfg.get("persistent_workers", num_workers > 0))

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
    return loader_kwargs


def _prepare_splits(
    train_path,
    val_path,
    test_path,
    label_column=None,
    feature_columns=None,
    normalize=False,
    use_pca=False,
    pca_components=None,
    pca_variance=0.95,
    return_artifacts=False,
):
    """Load three splits, assume the last column is the label, and optionally normalize/PCA."""
    train_df = pd.read_csv(train_path.replace("\\", "/"))
    val_df = pd.read_csv(val_path.replace("\\", "/"))
    test_df = pd.read_csv(test_path.replace("\\", "/"))

    label_name = train_df.columns[-1]

    train_features_df, train_labels = _split_features_and_labels(train_df, label_column, feature_columns)
    val_features_df, val_labels = _split_features_and_labels(val_df, label_column, feature_columns)
    test_features_df, test_labels = _split_features_and_labels(test_df, label_column, feature_columns)

    train_features = train_features_df.to_numpy(dtype="float32")
    val_features = val_features_df.to_numpy(dtype="float32")
    test_features = test_features_df.to_numpy(dtype="float32")

    scaler = None
    pca = None
    components = None

    if normalize:
        train_features, val_features, test_features, scaler = _normalize_features(
            train_features, val_features, test_features
        )

    feature_names = list(train_features_df.columns)
    if use_pca:
        components = pca_components
        if components is None:
            if feature_columns:
                components = len(feature_columns)
            elif 0 < pca_variance < 1:
                components = pca_variance
            else:
                components = train_features.shape[1]

        train_features, val_features, test_features, feature_names, pca = _apply_pca(
            train_features, val_features, test_features, components
        )

    if return_artifacts:
        artifacts = {
            "scaler": scaler,
            "pca": pca,
            "normalize": normalize,
            "use_pca": use_pca,
            "pca_components": components,
            "pca_variance": pca_variance,
            "label_column": label_name,
            "feature_names": feature_names,
        }
        return (
            train_features,
            train_labels,
            val_features,
            val_labels,
            test_features,
            test_labels,
            feature_names,
            artifacts,
        )

    return (
        train_features,
        train_labels,
        val_features,
        val_labels,
        test_features,
        test_labels,
        feature_names,
    )


def _parse_expected_classes(raw_value):
    if raw_value is None:
        return None

    if isinstance(raw_value, str):
        raw_value = [x.strip() for x in raw_value.split(",") if x.strip()]

    if isinstance(raw_value, (list, tuple, set)):
        parsed = []
        for value in raw_value:
            try:
                parsed.append(int(value))
            except (TypeError, ValueError):
                return None
        return sorted(set(parsed))

    return None


def _resolve_expected_classes(config, data_cfg):
    explicit = _parse_expected_classes(data_cfg.get("expected_classes"))
    if explicit is not None:
        return explicit

    train_cfg = config.get("train_config") or {}
    num_class = train_cfg.get("num_class")
    if num_class is None:
        class_names = config.get("class_names")
        if isinstance(class_names, (list, tuple)):
            num_class = len(class_names)

    try:
        num_class = int(num_class) if num_class is not None else None
    except (TypeError, ValueError):
        num_class = None

    if num_class is None or num_class <= 0:
        return None

    return list(range(num_class))


def _label_distribution(labels):
    array = np.asarray(labels)
    if array.size == 0:
        return {}

    values, counts = np.unique(array, return_counts=True)
    distribution = {}
    for value, count in zip(values, counts):
        try:
            key = int(value)
        except (TypeError, ValueError):
            key = str(value)
        distribution[key] = int(count)
    return distribution


def _distribution_keys_as_int(distribution):
    keys = set()
    for key in distribution.keys():
        try:
            keys.add(int(key))
        except (TypeError, ValueError):
            continue
    return keys


def _validate_expected_classes(
    split_name,
    context_name,
    distribution,
    expected_classes,
    strict=False,
):
    if not expected_classes:
        return

    observed = _distribution_keys_as_int(distribution)
    missing = [cls for cls in expected_classes if cls not in observed]
    if not missing:
        return

    msg = (
        f"[dataloader] Missing classes in {split_name} ({context_name}): {missing}. "
        f"Distribution={distribution}"
    )
    if strict:
        raise ValueError(msg)
    print(f"WARNING: {msg}")


def _build_label_stats(train_labels, val_labels, test_labels, train_set=None, val_set=None, test_set=None):
    stats = {
        "raw": {
            "train": _label_distribution(train_labels),
            "val": _label_distribution(val_labels),
            "test": _label_distribution(test_labels),
        }
    }

    if hasattr(train_set, "window_labels") and hasattr(val_set, "window_labels") and hasattr(test_set, "window_labels"):
        stats["sequence"] = {
            "train": _label_distribution(train_set.window_labels.cpu().numpy()),
            "val": _label_distribution(val_set.window_labels.cpu().numpy()),
            "test": _label_distribution(test_set.window_labels.cpu().numpy()),
        }

    return stats


def _apply_class_coverage_checks(config, data_cfg, label_stats):
    expected_classes = _resolve_expected_classes(config, data_cfg)
    require_all_classes = bool(data_cfg.get("require_all_classes", False))

    for split_name, distribution in label_stats.get("raw", {}).items():
        _validate_expected_classes(
            split_name=split_name,
            context_name="raw labels",
            distribution=distribution,
            expected_classes=expected_classes,
            strict=require_all_classes and split_name == "train",
        )

    for split_name, distribution in label_stats.get("sequence", {}).items():
        _validate_expected_classes(
            split_name=split_name,
            context_name="window labels",
            distribution=distribution,
            expected_classes=expected_classes,
            strict=require_all_classes and split_name == "train",
        )

    return expected_classes


def _resolve_sequence_mode(data_cfg, force_sequence=None, for_ml=None):
    if force_sequence is not None:
        return bool(force_sequence)
    if for_ml is not None:
        return not bool(for_ml)
    return bool(data_cfg.get("use_sequence", False))


def _resolve_stride(data_cfg, overlap, use_sequence):
    stride = max(1, int(data_cfg.get("stride", 1)))
    if not use_sequence:
        return stride

    overlap = float(overlap)
    overlap = min(max(overlap, 0.0), 0.99)
    return max(1, int(stride * (1.0 - overlap)))


def _load_split_arrays(data_cfg, normalize, use_pca, pca_components, pca_variance, want_artifacts):
    if want_artifacts:
        return _prepare_splits(
            data_cfg["train_data_path"],
            data_cfg["val_data_path"],
            data_cfg["test_data_path"],
            label_column=None,
            feature_columns=None,
            normalize=normalize,
            use_pca=use_pca,
            pca_components=pca_components,
            pca_variance=pca_variance,
            return_artifacts=True,
        )

    train_features, train_labels, val_features, val_labels, test_features, test_labels, feature_names = _prepare_splits(
        data_cfg["train_data_path"],
        data_cfg["val_data_path"],
        data_cfg["test_data_path"],
        label_column=None,
        feature_columns=None,
        normalize=normalize,
        use_pca=use_pca,
        pca_components=pca_components,
        pca_variance=pca_variance,
        return_artifacts=False,
    )
    return train_features, train_labels, val_features, val_labels, test_features, test_labels, feature_names, None


class TabularDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class SequenceDataset(Dataset):
    def __init__(self, data, labels, sequence_length, stride):
        sequence_length = int(sequence_length)
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")

        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.sequence_length = sequence_length
        self.stride = max(1, int(stride))
        self.total_length = int(self.data.shape[0])

        if self.total_length >= self.sequence_length:
            max_start = self.total_length - self.sequence_length
            self.start_indices = torch.arange(0, max_start + 1, self.stride, dtype=torch.long)
            self.end_indices = self.start_indices + self.sequence_length
            self.window_labels = self.labels[self.end_indices - 1]
        else:
            self.start_indices = torch.empty(0, dtype=torch.long)
            self.end_indices = torch.empty(0, dtype=torch.long)
            self.window_labels = torch.empty(0, dtype=torch.long)

        self.num_sequence = int(self.start_indices.numel())

    def __len__(self):
        return self.num_sequence

    def __getitem__(self, index):
        idx = int(index)
        start = int(self.start_indices[idx])
        end = int(self.end_indices[idx])
        sequence = self.data[start:end]
        label = self.window_labels[idx]
        return sequence, label


class HybridDataset(SequenceDataset):
    def __getitem__(self, index):
        sequence, label = super().__getitem__(index)
        sample = sequence[-1]
        return {"sequence": sequence, "sample": sample, "label": label}


def create_dataloaders(
    config_path,
    normalize=False,
    overlap=0.5,
    pca_variance=0.95,
    force_sequence=None,
    artifact_dir=None,
    artifact_tag=None,
    return_artifacts=False,
    for_ml=None,
):
    """
    Create train/val/test dataloaders.
    - Always treat the final column as the label.
    - Apply PCA only when config sets use_pca.
    - No intermediate files are written unless artifact_dir is provided.
    """
    config = load_config(config_path)
    data_cfg = config["data_config"]
    pca_variance = data_cfg.get("pca_variance", pca_variance)
    use_sequence = _resolve_sequence_mode(data_cfg, force_sequence=force_sequence, for_ml=for_ml)

    use_pca = data_cfg.get("use_pca", False) or config.get("use_pca", False)
    pca_components = data_cfg.get("pca_components") or config.get("pca_components")
    want_artifacts = return_artifacts or artifact_dir is not None

    (
        train_features,
        train_labels,
        val_features,
        val_labels,
        test_features,
        test_labels,
        _,
        artifacts,
    ) = _load_split_arrays(
        data_cfg=data_cfg,
        normalize=normalize,
        use_pca=use_pca,
        pca_components=pca_components,
        pca_variance=pca_variance,
        want_artifacts=want_artifacts,
    )

    effective_stride = _resolve_stride(data_cfg, overlap=overlap, use_sequence=use_sequence)
    batch_size = data_cfg.get("batch_size", 32)
    loader_kwargs = _build_loader_kwargs(data_cfg)

    if use_sequence:
        seq_len = data_cfg.get("sequence_length", 1)
        train_set = SequenceDataset(train_features, train_labels, seq_len, effective_stride)
        val_set = SequenceDataset(val_features, val_labels, seq_len, effective_stride)
        test_set = SequenceDataset(test_features, test_labels, seq_len, effective_stride)
    else:
        train_set = TabularDataset(train_features, train_labels)
        val_set = TabularDataset(val_features, val_labels)
        test_set = TabularDataset(test_features, test_labels)

    label_stats = _build_label_stats(
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels,
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
    )
    expected_classes = _apply_class_coverage_checks(config, data_cfg, label_stats)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **loader_kwargs)

    if artifacts is not None:
        artifacts["label_distributions"] = label_stats
        artifacts["expected_classes"] = expected_classes

    if artifact_dir and artifacts is not None:
        save_preprocess_artifacts(artifacts, artifact_dir, artifact_tag)

    if return_artifacts:
        return train_loader, val_loader, test_loader, artifacts

    return train_loader, val_loader, test_loader


def create_hybrid_dataloaders(
    config_path,
    normalize=False,
    overlap=0.5,
    pca_variance=0.95,
    artifact_dir=None,
    artifact_tag=None,
    return_artifacts=False,
):
    """Create dataloaders that return both sequence and single-sample views."""
    config = load_config(config_path)
    data_cfg = config["data_config"]
    if not data_cfg.get("use_sequence", False):
        raise ValueError("Hybrid dataloaders require use_sequence=True in config")

    pca_variance = data_cfg.get("pca_variance", pca_variance)
    use_pca = data_cfg.get("use_pca", False) or config.get("use_pca", False)
    pca_components = data_cfg.get("pca_components") or config.get("pca_components")
    want_artifacts = return_artifacts or artifact_dir is not None

    (
        train_features,
        train_labels,
        val_features,
        val_labels,
        test_features,
        test_labels,
        _,
        artifacts,
    ) = _load_split_arrays(
        data_cfg=data_cfg,
        normalize=normalize,
        use_pca=use_pca,
        pca_components=pca_components,
        pca_variance=pca_variance,
        want_artifacts=want_artifacts,
    )

    seq_len = data_cfg["sequence_length"]
    effective_stride = _resolve_stride(data_cfg, overlap=overlap, use_sequence=True)
    batch_size = data_cfg.get("batch_size", 32)
    loader_kwargs = _build_loader_kwargs(data_cfg)

    train_set = HybridDataset(train_features, train_labels, seq_len, effective_stride)
    val_set = HybridDataset(val_features, val_labels, seq_len, effective_stride)
    test_set = HybridDataset(test_features, test_labels, seq_len, effective_stride)

    label_stats = _build_label_stats(
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels,
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
    )
    expected_classes = _apply_class_coverage_checks(config, data_cfg, label_stats)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **loader_kwargs)

    if artifacts is not None:
        artifacts["label_distributions"] = label_stats
        artifacts["expected_classes"] = expected_classes

    if artifact_dir and artifacts is not None:
        save_preprocess_artifacts(artifacts, artifact_dir, artifact_tag)

    if return_artifacts:
        return train_loader, val_loader, test_loader, artifacts

    return train_loader, val_loader, test_loader


def load_ml_config(config_path, task="binary", mode="full_feature"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config[task][mode]


def get_ml_data(config_path, task="binary", mode="full_feature", normalize=False, pca_components=10):
    """Utility for classic ML pipelines; applies PCA only when mode == 'pca'."""
    config = load_ml_config(config_path, task, mode)
    use_pca = mode == "pca"

    (
        train_features,
        train_labels,
        val_features,
        val_labels,
        test_features,
        test_labels,
        feature_names,
    ) = _prepare_splits(
        config["train_data_path"],
        config["val_data_path"],
        config["test_data_path"],
        label_column=None,
        feature_columns=None,
        normalize=normalize,
        use_pca=use_pca,
        pca_components=pca_components,
        pca_variance=0.95,
    )

    return (
        pd.DataFrame(train_features, columns=feature_names),
        pd.Series(train_labels, name="label"),
        pd.DataFrame(val_features, columns=feature_names),
        pd.Series(val_labels, name="label"),
        pd.DataFrame(test_features, columns=feature_names),
        pd.Series(test_labels, name="label"),
    )
