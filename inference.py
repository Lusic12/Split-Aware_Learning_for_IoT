#!/usr/bin/env python3
"""
Inference helper that mirrors the training pipeline (same config + dataloader).
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml
import numpy as np
import pandas as pd

import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

# Add project root to path for direct script execution.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.define_model import create_model
from data.dataloader import create_dataloaders, create_hybrid_dataloaders


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)


def evaluate_model(model, loader, device, class_names=None):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                inputs = batch.get("sequence")
                labels = batch.get("label")
            else:
                inputs, labels = batch

            inputs = inputs.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).long()

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_predictions.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    accuracy = 100.0 * (all_predictions == all_targets).mean() if len(all_targets) else 0.0
    precision = precision_score(all_targets, all_predictions, average="macro", zero_division=0)
    recall = recall_score(all_targets, all_predictions, average="macro", zero_division=0)
    f1 = f1_score(all_targets, all_predictions, average="macro", zero_division=0)

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\nPrediction distribution:")
    max_pred = all_predictions.max() if all_predictions.size else 0
    max_true = all_targets.max() if all_targets.size else 0
    max_label = max(max_pred, max_true)
    counts = np.bincount(all_predictions, minlength=int(max_label) + 1)
    print(counts)

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_targets, all_predictions))

    print("\nClassification Report:")
    try:
        print(classification_report(all_targets, all_predictions, target_names=class_names, zero_division=0))
    except Exception:
        print(classification_report(all_targets, all_predictions, zero_division=0))


def parse_args():
    parser = argparse.ArgumentParser(description="Inference using the training pipeline")
    parser.add_argument("--config", type=str, default="configs/Deep_learning.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default=None, help="Override model type")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--use-hybrid", action="store_true", help="Use hybrid dataloader")
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--normalize", dest="normalize", action="store_true")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.set_defaults(normalize=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config["data_config"]

    train_cfg = (
        config.get("train_config")
        or config.get("student")
        or config.get("student_config")
        or {}
    )

    model_type = (args.model or train_cfg.get("model_type"))
    if not model_type:
        raise ValueError("Model type not found in config. Provide --model.")
    model_type = model_type.lower()

    class_names = config.get("class_names")

    sample_df = pd.read_csv(data_cfg["train_data_path"], nrows=1)
    raw_feature_cols = list(sample_df.columns[:-1])

    num_class = train_cfg.get("num_class") or len(class_names or []) or 2
    if not class_names:
        class_names = [f"class_{i}" for i in range(num_class)]

    sequence_length = data_cfg.get("sequence_length", None)

    device = get_device()

    sequence_models = ["cnn", "rnn", "lstm", "gru", "tcn_transformer"]
    force_sequence = model_type.lower() in sequence_models
    use_hybrid = args.use_hybrid or model_type.lower() == "tcn_transformer"

    if use_hybrid:
        train_loader, val_loader, test_loader, artifacts = create_hybrid_dataloaders(
            args.config,
            normalize=args.normalize,
            overlap=args.overlap,
            return_artifacts=True,
        )
    else:
        train_loader, val_loader, test_loader, artifacts = create_dataloaders(
            args.config,
            normalize=args.normalize,
            overlap=args.overlap,
            force_sequence=force_sequence,
            return_artifacts=True,
        )

    # Resolve effective feature count from preprocessed data (e.g., PCA)
    effective_feature_cols = raw_feature_cols
    if artifacts and artifacts.get("feature_names"):
        effective_feature_cols = artifacts["feature_names"]
    else:
        dataset_data = getattr(train_loader.dataset, "data", None)
        if dataset_data is not None:
            effective_feature_cols = [f"f_{i}" for i in range(dataset_data.shape[-1])]

    input_dim = len(effective_feature_cols)
    config["features"] = effective_feature_cols
    config["tcn_input_channels"] = input_dim
    if "tcn_config" in config:
        config["tcn_config"]["input_channels"] = input_dim

    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    loader = loaders[args.split]

    time_embedding_cfg = train_cfg.get("time_embedding") or config.get("time_embedding")
    model = create_model(
        model_type=model_type,
        input_dim=input_dim,
        num_class=num_class,
        hidden_layer_list=train_cfg.get("hidden_layer_list"),
        sequence_length=sequence_length,
        config=config,
        strict_input=True,
        time_embedding=time_embedding_cfg,
    ).to(device)

    load_checkpoint(model, args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Evaluating split: {args.split}")

    evaluate_model(model, loader, device, class_names)


if __name__ == "__main__":
    main()
