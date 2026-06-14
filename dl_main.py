#!/usr/bin/env python3
"""
Deep Learning Training Script for IPv6 Attack Detection

This script provides a unified interface for training various deep learning models
on IPv6 network traffic data for attack detection.

Usage:
    python dl_main.py --preset binary --model dnn
    python dl_main.py --preset multiclass --model cnn --epochs 50
    python dl_main.py --preset tcn_binary --model tcn_transformer --use_hybrid
"""

import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import pandas as pd
import yaml
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for direct script execution.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

from models.define_model import create_model
from data.dataloader import create_dataloaders, create_hybrid_dataloaders
from utils.function import count_parameters
from support.dl_helpers import evaluate_model, train_model
from utils.focal_loss import FocalLoss


def resolve_model_save_path(save_dir, logging_cfg, model_name, model_type):
    """
    Decide where to save the best model.
    - Prefer logging.model_save_path if provided (supports {model_name} / {model_type} tokens).
    - Otherwise fallback to weights/best_<model_name>_model.pth.
    """
    custom_path = (logging_cfg or {}).get("model_save_path")
    if custom_path:
        custom_path = custom_path.format(model_name=model_name, model_type=model_type)
        # If only a filename is provided, place it under save_dir
        if os.path.dirname(custom_path):
            best_path = custom_path
        else:
            best_path = os.path.join(save_dir, custom_path)
    else:
        best_path = 'best__model.pth'
        best_path = os.path.join(save_dir, best_path)

    best_dir = os.path.dirname(best_path) or save_dir
    if not os.path.exists(best_dir):
        os.makedirs(best_dir, exist_ok=True)
    return best_path


PRESET_PATHS = {
    "default": "configs/Deep_learning.yaml",
}


def main():
    parser = argparse.ArgumentParser(description='Train deep learning model for IPv6 attack detection')
    parser.add_argument('--config', type=str, required=False, default="configs/Deep_learning.yaml",
                       help='Path to configuration file (default: configs/Deep_learning.yaml)')
    parser.add_argument('--preset', type=str, choices=sorted(PRESET_PATHS.keys()), required=False,
                       help='Preset name that resolves to a config file (default preset -> Deep_learning.yaml)')
    parser.add_argument('--model', type=str, default=None,
                       help='Model type: dnn, cnn, mlp, rad_ffnn, tcn_transformer')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed override')
    parser.add_argument('--use_hybrid', action='store_true',
                       help='Use hybrid dataloader (for tcn_transformer)')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Overlap factor for sequence data (0.0-1.0)')
    parser.add_argument('--normalize', action='store_true', default=True,
                       help='Whether to normalize features')
    parser.add_argument('--save_path', type=str, default='weights',
                       help='Path to save model weights')
    
    args = parser.parse_args()
    
    # Resolve config path (preset has priority if provided)
    config_path = args.config
    if args.preset:
        config_path = PRESET_PATHS[args.preset]

    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print("Loaded configuration:")
    # print(yaml.safe_dump(config, sort_keys=False))

    # Override config with command line arguments if provided
    if args.model:
        config["train_config"]["model_type"] = args.model
    if args.epochs:
        config["train_config"]["num_epochs"] = args.epochs
    if args.lr:
        config["train_config"]["lr"] = args.lr
    if args.batch_size:
        config["data_config"]["batch_size"] = args.batch_size

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    data_cfg = config["data_config"]
    train_cfg = config["train_config"]

    print(data_cfg["train_data_path"])

    print(data_cfg["train_data_path"].replace("\\", "/"))

    # print()
    # print()
    sample_df = pd.read_csv((data_cfg["train_data_path"].replace("\\", "/")), nrows=1)


    raw_feature_cols = list(sample_df.columns[:-1])
    raw_feature_count = len(raw_feature_cols)

    class_names = config.get("class_names", [f"class_{i}" for i in range(train_cfg.get("num_class"))])
    logging_cfg = config.get("logging", {})
    
    print("=" * 60)
    print("IPv6 ATTACK DETECTION - DEEP LEARNING TRAINING")
    print("=" * 60)
    print(f"Configuration: {config_path}")
    print(f"Model: {train_cfg['model_type'].upper()}")
    print(f"Task: {len(class_names)} classes")
    print(f"Classes: {class_names}")
    print(f"Device: {device}")
    print(f"Features (raw): {raw_feature_count}")
    print(f"Epochs: {train_cfg['num_epochs']}")
    print(f"Batch size: {data_cfg['batch_size']}")
    print(f"Learning rate: {train_cfg['lr']}")
    print("=" * 60)

    # Create appropriate dataloader
    artifact_dir = os.path.join(
        args.save_path,
        "artifacts",
        os.path.splitext(os.path.basename(config_path))[0],
    )
    # try:
    sequence_models = ["cnn", "rnn", "lstm", "gru", "tcn_transformer"]
    force_sequence = train_cfg["model_type"].lower() in sequence_models
    if args.use_hybrid or train_cfg["model_type"].lower() == "tcn_transformer":
        print("Using hybrid dataloader...")
        train_loader, val_loader, test_loader, artifacts = create_hybrid_dataloaders(
            config_path, 
            normalize=args.normalize,
            overlap=args.overlap,
            artifact_dir=artifact_dir,
            return_artifacts=True,
        )
    else:
        print("Using standard dataloader...")
        train_loader, val_loader, test_loader, artifacts = create_dataloaders(
            config_path,
            normalize=args.normalize,
            overlap=args.overlap,
            force_sequence=force_sequence,
            artifact_dir=artifact_dir,
            return_artifacts=True,
        )
    
    print(f"Data loaded successfully:")
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Validation: {len(val_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")


    print(type(train_loader.dataset))
    print(len(train_loader.dataset))
    print(len(train_loader.dataset.labels))

    from collections import Counter

    window_labels = []

    for batch in train_loader:
        # print(type(batch))
        print(len(batch))
        # print(batch)
        break
    if len(batch) == 2:
        for _, y in train_loader:
            window_labels.extend(y.cpu().numpy())
    elif len(batch) == 3:   
        for _, _, y in train_loader:
            y = batch["label"]
            window_labels.extend(y.cpu().numpy())

    # for _, y in train_loader:
    #     window_labels.extend(y.cpu().numpy())

    counter = Counter(window_labels)
    total = sum(counter.values())

    print("Class distribution:", dict(sorted(counter.items())))

    expected_num_classes = train_cfg.get("num_class")
    try:
        expected_num_classes = int(expected_num_classes) if expected_num_classes is not None else 0
    except (TypeError, ValueError):
        expected_num_classes = 0

    if expected_num_classes <= 0:
        expected_num_classes = (max(counter.keys()) + 1) if counter else 1

    weights = torch.ones(expected_num_classes, dtype=torch.float32, device=device)
    if total > 0:
        for cls_idx in range(expected_num_classes):
            cls_count = counter.get(cls_idx, 0)
            if cls_count > 0:
                weights[cls_idx] = total / (expected_num_classes * cls_count)
            else:
                weights[cls_idx] = 0.0

    missing_classes = [cls_idx for cls_idx in range(expected_num_classes) if counter.get(cls_idx, 0) == 0]
    if missing_classes:
        print(f"WARNING: Missing classes in training windows: {missing_classes}")

    print("Class weights:", weights.detach().cpu().numpy())

    # except Exception as e:
    #     print(f"Error loading data: {e}")
    #     sys.exit(1)

    # Resolve effective feature count from preprocessed data (e.g., PCA)
    effective_feature_cols = raw_feature_cols
    if artifacts and artifacts.get("feature_names"):
        effective_feature_cols = artifacts["feature_names"]
    else:
        dataset_data = getattr(train_loader.dataset, "data", None)
        if dataset_data is not None:
            effective_feature_cols = [f"f_{i}" for i in range(dataset_data.shape[-1])]

    feature_count = len(effective_feature_cols)
    if feature_count != raw_feature_count:
        print(f"Features (effective): {feature_count}")

    # Provide feature metadata for models that expect it (e.g., TCN+Transformer)
    config["features"] = effective_feature_cols
    # Ensure TCN configs know input channels
    config["tcn_input_channels"] = feature_count
    if "tcn_config" in config:
        config["tcn_config"]["input_channels"] = feature_count

    # Create model
    try:
        # Determine input_dim from preprocessed data (PCA-aware)
        input_dim = feature_count
            
        num_class = train_cfg.get("num_class")
        sequence_length = data_cfg.get("sequence_length", None)
        time_embedding_cfg = train_cfg.get("time_embedding") or config.get("time_embedding")

        # Select hidden_layer_list based on model type for fair comparison
        model_type = train_cfg["model_type"].lower()
        hidden_layer_list = train_cfg.get("hidden_layer_list")  # fallback
        if model_type in ["cnn", "mlp", "rad_ffnn"]:
            hidden_layer_list = config.get("cnn_mlp_rad_hidden_layer_list", hidden_layer_list)
        elif model_type in ["rnn", "lstm", "gru", "tcn_transformer", "transformer"]:
            hidden_layer_list = config.get("sequence_hidden_layer_list", hidden_layer_list)
        # else: use default hidden_layer_list

        model = create_model(
            model_type=train_cfg["model_type"],
            input_dim=input_dim,
            num_class=num_class,
            hidden_layer_list=hidden_layer_list,
            sequence_length=sequence_length,
            config=config,
            strict_input=True,
            time_embedding=time_embedding_cfg,
        )

        model = model.to(device)
        print(f"\nModel created successfully:")
        print(f"  Architecture: {train_cfg['model_type'].upper()}")
        print(f"  Input dimension: {input_dim}")
        print(f"  Output classes: {num_class}")
        print(f"  Hidden layers: {hidden_layer_list}")
        if sequence_length:
            print(f"  Sequence length: {sequence_length}")
        print(f"  Total parameters: {count_parameters(model):,}")
        
    except Exception as e:
        print(f"Error creating model: {e}")
        sys.exit(1)

    # Setup training components

    
    
    #Non-weighted loss for now, can add class weights later if needed
    # criterion = nn.CrossEntropyLoss()
    #Focal loss can be used for imbalanced datasets, but start with standard cross-entropy
    # criterion = FocalLoss(alpha=weights, gamma=2.0, reduction='mean')

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=train_cfg["lr"], 
        weight_decay=train_cfg.get("weight_decay", 1e-5)
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    # Determine suffix for binary/multiclass and feature type
    suffix = "_binary" if num_class == 2 else "_multiclass"
    
    # Add PCA/LDA suffix if present in config path
    config_name = os.path.basename(config_path)
    if "_pca" in config_name:
        suffix += "_pca"
    elif "_lda" in config_name:
        suffix += "_lda"
    
    model_name_with_suffix = f"{train_cfg['model_type']}{suffix}"
    best_model_path = resolve_model_save_path(
        save_dir=args.save_path,
        logging_cfg=logging_cfg,
        model_name=model_name_with_suffix,
        model_type=train_cfg["model_type"]
    )
    print(f"Best model will be saved to: {best_model_path}")

    # Train model
    try:
        history, best_val_loss, best_val_acc = train_model(
            model_name=model_name_with_suffix,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=train_cfg["num_epochs"],
            class_names=class_names,
            save_path=args.save_path,
            best_model_path=best_model_path,
            device=device,
        )
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Load best model and evaluate on test set
    try:
        print("\n" + "=" * 60)
        print("FINAL EVALUATION ON TEST SET")
        print("=" * 60)
        if not os.path.exists(best_model_path):
            print(f"Best model file not found at {best_model_path}, using latest weights if available.")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']}")

        test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate_model(
            model,
            test_loader,
            criterion,
            class_names,
            "test",
            model_name_with_suffix,
            device=device,
        )


        # Save final results
        final_results = {
            'model_type': train_cfg["model_type"],
            'config_path': config_path,
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'num_parameters': count_parameters(model),
            'epochs_trained': len(history['train_loss']),
            'class_names': class_names,
        }

        results_path = os.path.join(args.save_path, f'{model_name_with_suffix}_results.json')
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)

        print(f"\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Model: {model_name_with_suffix.upper()}")
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        print(f"Final Test Accuracy: {test_acc:.2f}%")
        print(f"Final Test F1-Score: {test_f1:.4f}")
        print(f"Total Parameters: {count_parameters(model):,}")
        print(f"Results saved to: {results_path}")
        print("=" * 60)

    except Exception as e:
        print(f"Error during final evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
