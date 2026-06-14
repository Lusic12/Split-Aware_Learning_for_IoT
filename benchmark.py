#!/usr/bin/env python3

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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/Deep_learning.yaml")
    parser.add_argument("--benchmark", default="configs/benchmark.yaml")
    parser.add_argument("--save_path", default="benchmark_results")

    args = parser.parse_args()

    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    with open(args.benchmark) as f:
        benchmark_cfg = yaml.safe_load(f)

    datasets = benchmark_cfg["benchmark"]["datasets"]
    models = benchmark_cfg["benchmark"]["models"]

    results = []

    for dataset in datasets:

        print("\n===================================")
        print("DATASET:", dataset["name"])
        print("===================================")

        for model_type in models:

            print("\nMODEL:", model_type)

            config = base_config.copy()

            config["train_config"]["model_type"] = model_type
            config["train_config"]["num_class"] = dataset["num_class"]

            config["data_config"]["train_data_path"] = dataset["train"]
            config["data_config"]["val_data_path"] = dataset["val"]
            config["data_config"]["test_data_path"] = dataset["test"]

            class_names = dataset["class_names"]

            data_cfg = config["data_config"]
            train_cfg = config["train_config"]

            sample_df = pd.read_csv(data_cfg["train_data_path"], nrows=1)

            raw_feature_cols = list(sample_df.columns[:-1])
            raw_feature_count = len(raw_feature_cols)

            sequence_models = ["cnn", "rnn", "lstm", "gru", "tcn_transformer"]

            force_sequence = model_type in sequence_models

            if model_type == "tcn_transformer":

                train_loader, val_loader, test_loader, artifacts = create_hybrid_dataloaders(
                    args.config,
                    normalize=True,
                    overlap=0.5,
                    return_artifacts=True,
                )

            else:

                train_loader, val_loader, test_loader, artifacts = create_dataloaders(
                    args.config,
                    normalize=True,
                    overlap=0.5,
                    force_sequence=force_sequence,
                    return_artifacts=True,
                )

            feature_count = len(artifacts["feature_names"])

            config["tcn_input_channels"] = feature_count

            if "tcn_config" in config:
                config["tcn_config"]["input_channels"] = feature_count

            model = create_model(
                model_type=model_type,
                input_dim=feature_count,
                num_class=train_cfg["num_class"],
                hidden_layer_list=train_cfg["hidden_layer_list"],
                sequence_length=data_cfg.get("sequence_length"),
                config=config,
                strict_input=True,
            )

            model = model.to(device)

            criterion = nn.CrossEntropyLoss()

            optimizer = optim.AdamW(
                model.parameters(),
                lr=train_cfg["lr"],
                weight_decay=train_cfg.get("weight_decay", 1e-5)
            )

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=5, factor=0.5
            )

            history, best_val_loss, best_val_acc = train_model(
                model_name=model_type,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                epochs=train_cfg["num_epochs"],
                class_names=class_names,
                save_path=None,
                best_model_path=None,
                device=device,
            )

            test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate_model(
                model,
                test_loader,
                criterion,
                class_names,
                "test",
                model_type,
                device=device,
            )

            result = {
                "dataset": dataset["name"],
                "model": model_type,
                "accuracy": test_acc,
                "precision": test_precision,
                "recall": test_recall,
                "f1": test_f1,
                "parameters": count_parameters(model)
            }

            results.append(result)

    os.makedirs(args.save_path, exist_ok=True)

    df = pd.DataFrame(results)

    df.to_csv(os.path.join(args.save_path, "benchmark_results.csv"), index=False)

    with open(os.path.join(args.save_path, "benchmark_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nBenchmark finished")
    print("Results saved to", args.save_path)


if __name__ == "__main__":
    main()
