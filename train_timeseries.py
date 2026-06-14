#!/usr/bin/env python3

import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import json

from models.define_model import create_model
from data.dataloader import create_dataloaders
from utils.function import count_parameters
from support.dl_helpers import train_model, evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--save_path", type=str, default="weights")

    args = parser.parse_args()

    # ------------------------------
    # LOAD CONFIG
    # ------------------------------

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data_config"]
    train_cfg = config["train_config"]

    if args.epochs:
        train_cfg["num_epochs"] = args.epochs
    if args.lr:
        train_cfg["lr"] = args.lr
    if args.batch_size:
        data_cfg["batch_size"] = args.batch_size

    train_cfg["model_type"] = args.model

    model_type = args.model.lower()

    # ------------------------------
    # DETERMINE ML OR DL
    # ------------------------------

    ml_models = ["dnn", "mlp"]
    for_ml = model_type in ml_models

    print("=" * 60)
    print("TIME-SERIES ANOMALY DETECTION TRAINING")
    print("=" * 60)
    print(f"Model: {model_type.upper()}")
    print(f"ML mode: {for_ml}")
    print(f"Device: {device}")
    print("=" * 60)

    # ------------------------------
    # CREATE DATALOADERS
    # ------------------------------

    train_loader, val_loader, test_loader = create_dataloaders(
        args.config,
        for_ml=for_ml
    )

    # ------------------------------
    # INFER INPUT DIMENSION
    # ------------------------------

    sample_batch = next(iter(train_loader))
    x_sample, y_sample = sample_batch

    if for_ml:
        input_dim = x_sample.shape[1]
        sequence_length = None
        feature_dim = None
    else:
        sequence_length = x_sample.shape[1]
        feature_dim = x_sample.shape[2]
        input_dim = feature_dim

    num_class = train_cfg["num_class"]

    print(f"Input dimension: {input_dim}")
    if not for_ml:
        print(f"Sequence length: {sequence_length}")
        print(f"Feature dimension: {feature_dim}")

    # ------------------------------
    # CREATE MODEL
    # ------------------------------

    hidden_layer_list = train_cfg.get("hidden_layer_list")

    model = create_model(
        model_type=model_type,
        input_dim=input_dim,
        num_class=num_class,
        hidden_layer_list=hidden_layer_list,
        sequence_length=sequence_length,
        config=config,
        strict_input=True,
    )

    model = model.to(device)

    print(f"Total parameters: {count_parameters(model):,}")

    # ------------------------------
    # TRAIN SETUP
    # ------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=5,
        factor=0.5,
    )

    os.makedirs(args.save_path, exist_ok=True)
    best_model_path = os.path.join(
        args.save_path,
        f"best_{model_type}.pth"
    )

    # ------------------------------
    # TRAIN
    # ------------------------------

    history, best_val_loss, best_val_acc = train_model(
        model_name=model_type,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=train_cfg["num_epochs"],
        class_names=config.get("class_names"),
        save_path=args.save_path,
        best_model_path=best_model_path,
        device=device,
    )

    # ------------------------------
    # TEST EVALUATION
    # ------------------------------

    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate_model(
        model,
        test_loader,
        criterion,
        config.get("class_names"),
        "test",
        model_type,
        device=device,
    )

    results = {
        "model": model_type,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "parameters": count_parameters(model),
    }

    with open(os.path.join(args.save_path, f"{model_type}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
