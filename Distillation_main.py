#!/usr/bin/env python3
"""
Refactored Deep Learning Training Script for IPv6 Attack Detection

Features:
- Clean argument parsing
- Proper device selection
- Fixed Knowledge Distillation training with DistillationLoss
- Separate KD (training) and CE (validation/test) criteria
- Support for hybrid and standard dataloaders
- Save best student model, history (CSV + plot), and confusion matrices
- Uses existing project modules: models.define_model, models.teacher_student_models,
  data.dataloader, utils.function

Usage examples:
    python dl_student_main_refactor.py --config configs/Distill_model.yaml --model gru --teacher-model tcn_transformer
    python dl_student_main_refactor.py --config configs/Distill_model.yaml --model lstm --epochs 20
    python dl_student_main_refactor.py --config configs/Distill_model.yaml --model mlp --use_hybrid
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path for direct script execution.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Project modules (assumed to exist in your repo)
from models.define_model import create_model
from models.student_model import *
from data.dataloader import create_dataloaders, create_hybrid_dataloaders
from utils.function import count_parameters, plot_training_history
from support.dl_helpers import (
    DistillationLoss,
    ensure_dir,
    evaluate_model,
    train_student_model,
)

# ----------------------------------------------------------------------------
# Device
# ----------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()
torch.backends.cudnn.benchmark = torch.cuda.is_available()


# ----------------------------------------------------------------------------
# Main: argument parsing, model instantiation, dataloaders, optimizer, training
# ----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Train student model with knowledge distillation")
    p.add_argument('--config', type=str, required=False, default="configs/Distill_model.yaml",
                  help='Path to YAML config file (default: configs/Distill_model.yaml)')
    p.add_argument('--model', type=str, required=True, help='Model name / type for student')
    p.add_argument('--teacher-model', type=str, required=False, help='Teacher model name/type')
    p.add_argument('--teacher-checkpoint', type=str, required=False, help='Path to teacher model checkpoint')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--use-hybrid', action='store_true', help='Use hybrid dataloaders')
    p.add_argument('--temperature', type=float, default=None, help='Override distillation temperature')
    p.add_argument('--alpha', type=float, default=None, help='Override distillation alpha')
    p.add_argument('--out-dir', type=str, default='outputs/distill')
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def compute_class_weights_from_dataset(dataset, num_classes, device):
    """
    Compute inverse-frequency class weights from a dataset that exposes .labels.
    Weights are normalized: w_c = total / (num_classes * count_c)
    """
    if not hasattr(dataset, "labels"):
        return None
    labels = dataset.labels
    if not torch.is_tensor(labels):
        labels = torch.as_tensor(labels, dtype=torch.long)
    counts = torch.bincount(labels.cpu(), minlength=num_classes).float()
    # Avoid division by zero
    counts = torch.clamp(counts, min=1.0)
    total = counts.sum()
    weights = total / (num_classes * counts)
    return weights.to(device)


def main():
    args = parse_args()
    config = load_config(args.config)
    # Merge CLI overrides and normalize config shape
    data_cfg = config["data_config"]
    teacher_cfg = config.get("teacher") or config.get("teacher_config", {})
    student_cfg = config.get("student") or config.get("student_config") or config.get("train_config", {})
    distill_cfg = config.get("distillation") or {
        "temperature": config.get("kd_temperature", config.get("train_config", {}).get("temperature", 4.0)),
        "alpha": config.get("kd_alpha", config.get("train_config", {}).get("alpha", 0.5)),
        "save_dir": config.get("train_config", {}).get("save_path", "checkpoints"),
        "use_focal_loss": config.get("kd_use_focal_loss", False),
        "focal_gamma": config.get("kd_focal_gamma", 2.0)
    }
    if args.model:
        student_cfg["model_type"] = args.model
    if args.epochs:
        student_cfg["num_epochs"] = args.epochs
    if args.lr:
        student_cfg["lr"] = args.lr
    if args.batch_size:
        data_cfg["batch_size"] = args.batch_size
    if args.temperature is not None:
        distill_cfg["temperature"] = args.temperature
    if args.alpha is not None:
        distill_cfg["alpha"] = args.alpha
    config["student"] = student_cfg
    config["teacher"] = teacher_cfg
    config["distillation"] = distill_cfg

    out_dir = args.out_dir
    ensure_dir(out_dir)
    
    class_names = config.get(
        "class_names",
        [f"class_{i}" for i in range(student_cfg.get("num_class", teacher_cfg.get("num_class", 2)))]
    )
    normalize = bool(data_cfg.get("normalize", True))
    # Infer raw feature count from train CSV (all columns except label)
    sample_df = pd.read_csv(data_cfg["train_data_path"], nrows=1)
    raw_feature_cols = list(sample_df.columns[:-1])
    raw_feature_count = len(raw_feature_cols)
    input_dim = raw_feature_count
    num_class = student_cfg.get("num_class") or teacher_cfg.get("num_class") or len(class_names)
    sequence_length = data_cfg.get("sequence_length", None)
    default_time_cfg = config.get("time_embedding")
    student_time_cfg = student_cfg.get("time_embedding", default_time_cfg)
    teacher_time_cfg = teacher_cfg.get("time_embedding")
    # Use per-role configs; keep small default for student if missing
    student_hidden = student_cfg.get("hidden_layer_list") or [64, 32]
    teacher_hidden = teacher_cfg.get("hidden_layer_list") or [256, 128, 64]
    student_epochs = student_cfg.get("num_epochs", 10)
    student_lr = student_cfg.get("lr", 1e-3)
    weight_decay = student_cfg.get("weight_decay", 0.0)

    print("=" * 60)
    print("IPv6 ATTACK DETECTION - KNOWLEDGE DISTILLATION")
    print("=" * 60)
    print(f"Configuration: {args.config}")
    print(f"Teacher: {args.teacher_model or teacher_cfg.get('model_type', 'N/A')}")
    print(f"Teacher checkpoint: {args.teacher_checkpoint or teacher_cfg.get('checkpoint', 'N/A')}")
    print(f"Student: {student_cfg.get('model_type', 'N/A')}")
    print(f"Classes: {class_names}")
    print(f"Device: {device}")
    print(f"Features (raw): {raw_feature_count}")
    print(f"Epochs: {student_epochs}")
    print(f"Batch size: {data_cfg['batch_size']}")
    print(f"Learning rate: {student_lr}")
    print(f"KD params: T={distill_cfg.get('temperature', 4.0)} alpha={distill_cfg.get('alpha', 0.5)}")
    print(f"Student loss: {'Focal' if distill_cfg.get('use_focal_loss', False) else 'CrossEntropy'} "
          f"(gamma={distill_cfg.get('focal_gamma', 2.0)})")
    print("=" * 60)

    # Dataloaders
    artifact_dir = os.path.join(out_dir, "artifacts", Path(args.config).stem)
    use_hybrid = args.use_hybrid
    if not use_hybrid:
        teacher_model_hint = args.teacher_model or teacher_cfg.get('model_type')
        if teacher_model_hint == "tcn_transformer":
            use_hybrid = True
            print("Using hybrid dataloader to mirror teacher pretraining (tcn_transformer).")
    if use_hybrid:
        train_loader, val_loader, test_loader, artifacts = create_hybrid_dataloaders(
            args.config,
            normalize=normalize,
            artifact_dir=artifact_dir,
            return_artifacts=True,
        )
    else:
        train_loader, val_loader, test_loader, artifacts = create_dataloaders(
            args.config,
            normalize=normalize,
            artifact_dir=artifact_dir,
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
    if input_dim != raw_feature_count:
        print(f"Features (effective): {input_dim}")

    # Provide feature metadata for models that expect it (e.g., TCN+Transformer)
    config["features"] = effective_feature_cols
    # Ensure TCN configs know input channels
    config["tcn_input_channels"] = input_dim
    if "tcn_config" in config:
        config["tcn_config"]["input_channels"] = input_dim

    # Compute class weights from training data to mitigate imbalance
    class_weights = compute_class_weights_from_dataset(train_loader.dataset, num_class, device)
    if class_weights is not None:
        print(f"Class weights (train): {class_weights.cpu().numpy().tolist()}")

    # Create teacher and student models
    student_model_name = student_cfg.get("model_type")
    if not student_model_name:
        raise ValueError("Student model type is missing. Provide --model or set student.model_type in config.")

    student_model = create_model(
        student_model_name, input_dim, num_class,
        hidden_layer_list=student_hidden,
        sequence_length=sequence_length,
        config=config,
        strict_input=True,
        time_embedding=student_time_cfg,
    )

    teacher_model_name = args.teacher_model or teacher_cfg.get('model_type') or config.get('teacher_model', None)
    teacher_ckpt = args.teacher_checkpoint or teacher_cfg.get('checkpoint') or config.get('teacher_checkpoint', None)

    if not teacher_model_name:
        raise ValueError("No teacher model specified. Use --teacher-model or set teacher.model_type in config.")
    if not teacher_ckpt or not os.path.exists(teacher_ckpt):
        raise FileNotFoundError(f"Teacher checkpoint not found at {teacher_ckpt}. Train the teacher first.")

    teacher_model = create_model(
        teacher_model_name, input_dim, num_class,
        hidden_layer_list=teacher_hidden,
        sequence_length=sequence_length,
        config=config,
        strict_input=True,
        time_embedding=teacher_time_cfg,
    )
    teacher_state = torch.load(teacher_ckpt, map_location=device)
    # Support both raw state_dict and training checkpoints
    if isinstance(teacher_state, dict):
        if "model_state_dict" in teacher_state:
            teacher_state = teacher_state["model_state_dict"]
        elif "state_dict" in teacher_state:
            teacher_state = teacher_state["state_dict"]
    teacher_model.load_state_dict(teacher_state)
    print(f"Loaded teacher checkpoint from {teacher_ckpt}")

    # Move to device
    student_model.to(device)
    teacher_model.to(device)

    student_params = count_parameters(student_model)
    teacher_params = count_parameters(teacher_model)
    print(f"Student parameters: {student_params}")
    print(f"Teacher parameters: {teacher_params}")
    if student_params > teacher_params:
        print("Warning: student has MORE parameters than teacher. Consider reducing student hidden_layer_list.")

    # Criterion and optimizer
    kd_criterion = DistillationLoss(
        temperature=distill_cfg.get('temperature', 4.0),
        alpha=distill_cfg.get('alpha', 0.5),
        class_weights=class_weights,
        use_focal_loss=distill_cfg.get('use_focal_loss', False),
        focal_gamma=distill_cfg.get('focal_gamma', 2.0)
    )
    ce_criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(student_model.parameters(), lr=student_lr, weight_decay=weight_decay)

    checkpoint_dir = distill_cfg.get('save_dir') or os.path.join(out_dir, 'checkpoints')
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.join(out_dir, checkpoint_dir)
    ensure_dir(checkpoint_dir)

    # Train
    start = time.time()
    history = train_student_model(
        student_model=student_model,
        teacher_model=teacher_model,
        train_loader=train_loader,
        val_loader=val_loader,
        kd_criterion=kd_criterion,
        ce_criterion=ce_criterion,
        optimizer=optimizer,
        num_epochs=student_epochs,
        model_name=student_model_name,
        class_names=class_names,
        save_dir=checkpoint_dir,
        device=device,
    )
    end = time.time()
    print(f"Training finished in {(end-start)/60:.2f} minutes")

    # Save history to CSV and plot
    hist_df = pd.DataFrame(history)
    hist_csv = os.path.join(out_dir, f"{student_model_name}_training_history.csv")
    hist_df.to_csv(hist_csv, index=False)
    print(f"Saved training history to {hist_csv}")

    try:
        plot_training_history(history, out_path=os.path.join(out_dir, f"{student_model_name}_training_plot.png"))
    except Exception as e:
        print(f"plot_training_history failed: {e}")

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)
    best_model_path = os.path.join(checkpoint_dir, f"{student_model_name}_best_student.pth")
    if os.path.exists(best_model_path):
        student_model.load_state_dict(torch.load(best_model_path, map_location=device))
        student_model.to(device)
        print(f"Loaded best model from {best_model_path}")
    else:
        print("Best model not found; using current weights for evaluation")

    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate_model(
        student_model,
        test_loader,
        ce_criterion,
        class_names,
        dataset_type="test",
        model_name=student_model_name,
        device=device,
    )

    # Save final results
    final_results = {
        'model_type': student_model_name,
        'teacher_model': teacher_model_name,
        'teacher_checkpoint': teacher_ckpt,
        'config_path': args.config,
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'num_parameters': count_parameters(student_model),
        'epochs_trained': len(history['train_loss']),
        'class_names': class_names,
        'kd_temperature': distill_cfg.get('temperature', 4.0),
        'kd_alpha': distill_cfg.get('alpha', 0.5)
    }

    results_path = os.path.join(out_dir, f'{student_model_name}_student_results.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Student: {student_model_name.upper()}")
    print(f"Teacher: {teacher_model_name.upper()}")
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"Final Test F1-Score: {test_f1:.4f}")
    print(f"Total Parameters: {count_parameters(student_model):,}")
    print(f"Results saved to: {results_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
