#!/usr/bin/env python3
"""
Deep Learning Training Script for IPv6 Attack Detection
"""

import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import json
import pandas as pd
import yaml
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

from models.define_model import create_model
from data.dataloader import create_dataloaders, create_hybrid_dataloaders
from utils.function import count_parameters
from support.dl_helpers import evaluate_model, train_model
from utils.focal_loss import FocalLoss
from collections import Counter


def build_label_mapping(labels, num_class):
    """
    Analyse which class indices are actually present in the dataset.
    
    Returns:
        class_to_new   : {original_label -> model_index}   (for remapping IF needed)
        new_to_class   : {model_index   -> original_label} (for decoding predictions)
        num_present    : how many distinct classes exist
        counter        : Counter of original labels
        
    NOTE: We do NOT remap labels inside the DataLoader.
          This mapping is kept purely for result interpretation:
            "model predicted index 2  →  real dataset class 5"
    """
    counter = Counter(labels)
    present_classes = sorted(counter.keys())
    class_to_new  = {cls: i   for i, cls in enumerate(present_classes)}
    new_to_class  = {i:   cls for i, cls in enumerate(present_classes)}
    return class_to_new, new_to_class, len(present_classes), counter


def resolve_model_save_path(save_dir, logging_cfg, model_name, model_type):
    custom_path = (logging_cfg or {}).get("model_save_path")
    if custom_path:
        custom_path = custom_path.format(model_name=model_name, model_type=model_type)
        best_path = (custom_path if os.path.dirname(custom_path)
                     else os.path.join(save_dir, custom_path))
    else:
        best_path = os.path.join(save_dir, f'best_{model_name}_model.pth')

    best_dir = os.path.dirname(best_path) or save_dir
    os.makedirs(best_dir, exist_ok=True)
    return best_path


PRESET_PATHS = {
    "default": "configs/Deep_learning.yaml",
}


def main():
    parser = argparse.ArgumentParser(
        description='Train deep learning model for IPv6 attack detection'
    )
    parser.add_argument('--config',      type=str,   default="configs/Deep_learning.yaml")
    parser.add_argument('--preset',      type=str,   choices=sorted(PRESET_PATHS.keys()))
    parser.add_argument('--model',       type=str,   default=None)
    parser.add_argument('--epochs',      type=int,   default=None)
    parser.add_argument('--lr',          type=float, default=None)
    parser.add_argument('--batch_size',  type=int,   default=None)
    parser.add_argument('--use_hybrid',  action='store_true')
    parser.add_argument('--overlap',     type=float, default=0.5)
    parser.add_argument('--normalize',   action='store_true', default=True)
    parser.add_argument('--save_path',   type=str,   default='weights')
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    #  Config                                                              #
    # ------------------------------------------------------------------ #
    config_path = PRESET_PATHS[args.preset] if args.preset else args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print("Loaded configuration.")

    if args.model:      config["train_config"]["model_type"]  = args.model
    if args.epochs:     config["train_config"]["num_epochs"]  = args.epochs
    if args.lr:         config["train_config"]["lr"]          = args.lr
    if args.batch_size: config["data_config"]["batch_size"]   = args.batch_size

    data_cfg    = config["data_config"]
    train_cfg   = config["train_config"]
    logging_cfg = config.get("logging", {})

    sample_df         = pd.read_csv(data_cfg["train_data_path"], nrows=1)
    raw_feature_cols  = list(sample_df.columns[:-1])
    raw_feature_count = len(raw_feature_cols)

    class_names = config.get(
        "class_names",
        [f"class_{i}" for i in range(train_cfg.get("num_class"))]
    )

    sep = "=" * 60
    print(f"\n{sep}")
    print("  IPv6 ATTACK DETECTION - DEEP LEARNING TRAINING")
    print(f"{sep}")
    print(f"  Config    : {config_path}")
    print(f"  Model     : {train_cfg['model_type'].upper()}")
    print(f"  Classes   : {class_names}  ({len(class_names)})")
    print(f"  Device    : {device}")
    print(f"  Features  : {raw_feature_count}")
    print(f"  Epochs    : {train_cfg['num_epochs']}")
    print(f"  Batch     : {data_cfg['batch_size']}")
    print(f"  LR        : {train_cfg['lr']}")
    print(f"{sep}\n")

    # ------------------------------------------------------------------ #
    #  Dataloaders  (untouched – original labels flow through)            #
    # ------------------------------------------------------------------ #
    artifact_dir = os.path.join(
        args.save_path, "artifacts",
        os.path.splitext(os.path.basename(config_path))[0],
    )

    try:
        sequence_models = ["cnn", "rnn", "lstm", "gru", "tcn_transformer"]
        force_sequence  = train_cfg["model_type"].lower() in sequence_models

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
        print(f"  Train      : {len(train_loader.dataset)} samples")
        print(f"  Validation : {len(val_loader.dataset)} samples")
        print(f"  Test       : {len(test_loader.dataset)} samples")


        def inspect_split(name, loader):
            """Đếm số sample của mỗi class trong một DataLoader split."""
            if hasattr(loader.dataset, "labels"):
                # Nhanh: lấy trực tiếp từ .labels nếu dataset có attribute đó
                labels = [int(l) for l in loader.dataset.labels]
            else:
                # Chậm hơn: iterate qua loader (dùng khi không có .labels)
                labels = []
                for _, y in loader:
                    labels.extend(y.cpu().numpy().tolist())

            counter  = Counter(labels)
            total    = sum(counter.values())

            print(f"\n  [{name}] Class distribution  (total={total})")
            print(f"  {'Class':>10} | {'Count':>8} | {'Percent':>8}")
            print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*8}")
            for cls in sorted(counter.keys()):
                pct = 100.0 * counter[cls] / total
                print(f"  {cls:>10} | {counter[cls]:>8} | {pct:>7.2f}%")

            return counter

        train_counter = inspect_split("TRAIN",      train_loader)
        val_counter   = inspect_split("VALIDATION", val_loader)
        test_counter  = inspect_split("TEST",       test_loader)

        # Cảnh báo nếu val/test có class không xuất hiện trong train
        train_classes = set(train_counter.keys())
        val_classes   = set(val_counter.keys())
        test_classes  = set(test_counter.keys())

        unseen_in_val  = val_classes  - train_classes
        unseen_in_test = test_classes - train_classes

        if unseen_in_val:
            print(f"\n  [WARNING] Val  has classes not seen in train: {unseen_in_val}")
        if unseen_in_test:
            print(f"  [WARNING] Test has classes not seen in train: {unseen_in_test}")

        print() 

        # ---------------------------------------------------------------- #
        #  Label analysis  (mapping kept for INTERPRETATION ONLY)          #
        # ---------------------------------------------------------------- #
        all_train_labels = train_loader.dataset.labels   # original integer labels

        class_to_new, new_to_class, num_present, label_counter = build_label_mapping(
            all_train_labels, train_cfg.get("num_class")
        )

        # num_class: use however many distinct classes actually appear
        num_class = num_present
        # print(f"\n[INFO] Distinct classes found in training set : {num_class}")
        # print(f"[INFO] Label counter  (original ids) : {dict(label_counter)}")
        # print(f"[INFO] class_to_new (original -> model index) : {class_to_new}")
        # print(f"[INFO] new_to_class (model index -> original) : {new_to_class}")
        # print()
        # ---------------------------------------------------------------- #
        #  Example of how to use the mapping AFTER inference:              #
        #                                                                   #
        #    raw_output  = model(x).argmax(dim=1)   # e.g. tensor([1, 0]) #
        #    real_labels = [new_to_class[i.item()]                         #
        #                   for i in raw_output]    # e.g. [3, 0]         #
        #                                                                   #
        #  That's it – no remapping inside the DataLoader at all.          #
        # ---------------------------------------------------------------- #

        # ---------------------------------------------------------------- #
        #  Class weights for loss  (based on original label distribution)  #
        # ---------------------------------------------------------------- #
        total   = sum(label_counter.values())
        weights = torch.zeros(num_class, dtype=torch.float32)

        for original_cls, count in label_counter.items():
            model_idx = class_to_new[original_cls]      # which output neuron
            weights[model_idx] = total / (num_class * count)

        weights = weights.to(device)
        # print(f"[INFO] Class weights (per model output neuron): {weights.cpu().numpy()}")

    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ------------------------------------------------------------------ #
    #  Effective feature count  (PCA / LDA aware)                         #
    # ------------------------------------------------------------------ #
    effective_feature_cols = raw_feature_cols
    if artifacts and artifacts.get("feature_names"):
        effective_feature_cols = artifacts["feature_names"]
    else:
        # Try to peek at the actual tensor shape
        inner_dataset = getattr(train_loader.dataset, "dataset", train_loader.dataset)
        dataset_data  = getattr(inner_dataset, "data", None)
        if dataset_data is not None:
            effective_feature_cols = [f"f_{i}" for i in range(dataset_data.shape[-1])]

    feature_count = len(effective_feature_cols)
    if feature_count != raw_feature_count:
        print(f"[INFO] Features after preprocessing : {feature_count}  "
              f"(raw was {raw_feature_count})")

    config["features"]           = effective_feature_cols
    config["tcn_input_channels"] = feature_count
    if "tcn_config" in config:
        config["tcn_config"]["input_channels"] = feature_count

    # ------------------------------------------------------------------ #
    #  Model                                                               #
    # ------------------------------------------------------------------ #
    try:
        input_dim          = feature_count
        sequence_length    = data_cfg.get("sequence_length", None)
        time_embedding_cfg = train_cfg.get("time_embedding") or config.get("time_embedding")
        model_type         = train_cfg["model_type"].lower()
        hidden_layer_list  = train_cfg.get("hidden_layer_list")

        if model_type in ["cnn", "mlp", "rad_ffnn"]:
            hidden_layer_list = config.get("cnn_mlp_rad_hidden_layer_list", hidden_layer_list)
        elif model_type in ["rnn", "lstm", "gru", "tcn_transformer", "transformer"]:
            hidden_layer_list = config.get("sequence_hidden_layer_list", hidden_layer_list)

        model = create_model(
            model_type=train_cfg["model_type"],
            input_dim=input_dim,
            num_class=num_class,
            hidden_layer_list=hidden_layer_list,
            sequence_length=sequence_length,
            config=config,
            strict_input=True,
            time_embedding=time_embedding_cfg,
        ).to(device)

        print(f"\nModel created:")
        print(f"  Architecture : {train_cfg['model_type'].upper()}")
        print(f"  Input dim    : {input_dim}")
        # print(f"  Output nodes : {num_class}  "
            #   f"(mapped from original classes {sorted(class_to_new.keys())})")
        print(f"  Hidden layers: {hidden_layer_list}")
        if sequence_length:
            print(f"  Seq length   : {sequence_length}")
        print(f"  Parameters   : {count_parameters(model):,}\n")

    except Exception as e:
        print(f"Error creating model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ------------------------------------------------------------------ #
    #  Training components                                                 #
    # ------------------------------------------------------------------ #
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    suffix      = "_binary" if num_class == 2 else "_multiclass"
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
        model_type=train_cfg["model_type"],
    )
    print(f"Best model path: {best_model_path}")

    # ------------------------------------------------------------------ #
    #  Save the label mapping alongside the model weights                 #
    #  so anyone loading the checkpoint knows how to decode outputs.      #
    # ------------------------------------------------------------------ #
    # mapping_path = best_model_path.replace(".pth", "_label_mapping.json")
    # label_mapping_info = {
    #     "description": (
    #         "model_index -> original_dataset_label. "
    #         "Use new_to_class to convert model output indices back to "
    #         "the real class ids used in your CSV."
    #     ),
    #     "class_to_new": {str(k): v for k, v in class_to_new.items()},
    #     "new_to_class": {str(k): v for k, v in new_to_class.items()},
    #     "class_names":  class_names,
    # }
    # os.makedirs(os.path.dirname(mapping_path) or ".", exist_ok=True)
    # with open(mapping_path, "w") as f:
    #     json.dump(label_mapping_info, f, indent=2)
    # print(f"Label mapping saved to: {mapping_path}\n")

    # ------------------------------------------------------------------ #
    #  Train                                                               #
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    #  Final evaluation                                                    #
    # ------------------------------------------------------------------ #
    try:
        print(f"\n{sep}")
        print("  FINAL EVALUATION ON TEST SET")
        print(f"{sep}")

        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']}")
        else:
            print("Best model file not found – using current in-memory weights.")

        test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate_model(
            model, test_loader, criterion, class_names,
            "test", model_name_with_suffix, device=device,
        )

        final_results = {
            "model_type":       train_cfg["model_type"],
            "config_path":      config_path,
            "best_val_loss":    best_val_loss,
            "best_val_acc":     best_val_acc,
            "test_loss":        test_loss,
            "test_accuracy":    test_acc,
            "test_precision":   test_precision,
            "test_recall":      test_recall,
            "test_f1":          test_f1,
            "num_parameters":   count_parameters(model),
            "epochs_trained":   len(history["train_loss"]),
            "class_names":      class_names,
            # Include mapping so the JSON is self-contained
            "label_mapping":    label_mapping_info,
        }

        results_path = os.path.join(args.save_path, f'{model_name_with_suffix}_results.json')
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)

        print(f"\n{sep}")
        print("  TRAINING SUMMARY")
        print(f"{sep}")
        print(f"  Model            : {model_name_with_suffix.upper()}")
        print(f"  Best Val Accuracy: {best_val_acc:.2f}%")
        print(f"  Test Accuracy    : {test_acc:.2f}%")
        print(f"  Test F1-Score    : {test_f1:.4f}")
        print(f"  Parameters       : {count_parameters(model):,}")
        print(f"  Results saved to : {results_path}")
        print(f"{sep}\n")

    except Exception as e:
        print(f"Error during final evaluation: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()