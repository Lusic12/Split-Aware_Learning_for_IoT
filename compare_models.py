import torch
import pandas as pd
import numpy as np
import os
import sys
import argparse
from torch.utils.data import TensorDataset, DataLoader
import json

# Add project root to path for direct script execution.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from support.training_helpers import (
    train_model,
    evaluate_model,
    plot_training_curves,
    plot_confusion_matrices,
)

# Import modules
from models.define_model import MLP, DNN, LSTM, GRU, CNN


def load_data(data_path, header=0):
    """
    Load data from CSV file and prepare tensors
    
    Args:
        data_path (str): Path to CSV data file
        header (int): Header row index
        
    Returns:
        tuple: (X, y) - Features and labels as torch tensors
    """
    df = pd.read_csv(data_path, header=header)
    
    # Extract features and labels
    X = df.iloc[:, :-1].values  # All columns except the last
    y = df.iloc[:, -1].values   # Last column (label)
    
    # Convert to torch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    
    return X, y


def _first_existing(paths):
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def resolve_real_paths(data_dir, data_type, train_path, val_path, test_path):
    if any([train_path, val_path, test_path]):
        if not all([train_path, val_path, test_path]):
            raise ValueError("Provide --train, --val, and --test together.")
        return train_path, val_path, test_path, data_type or "custom"

    if data_type in ("binary", "multiclass"):
        train = os.path.join(data_dir, f"train_data_{data_type}.csv")
        val = os.path.join(data_dir, f"val_data_{data_type}.csv")
        test = os.path.join(data_dir, f"test_data_{data_type}.csv")
        if all(os.path.exists(p) for p in [train, val, test]):
            return train, val, test, data_type

    candidates = []
    for dtype in ("binary", "multiclass"):
        train = os.path.join(data_dir, f"train_data_{dtype}.csv")
        val = os.path.join(data_dir, f"val_data_{dtype}.csv")
        test = os.path.join(data_dir, f"test_data_{dtype}.csv")
        if all(os.path.exists(p) for p in [train, val, test]):
            candidates.append((train, val, test, dtype))

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError("Found both binary/multiclass splits; set --data-type explicitly.")

    train = os.path.join(data_dir, "train.csv")
    val = os.path.join(data_dir, "val.csv")
    test = os.path.join(data_dir, "test.csv")
    if all(os.path.exists(p) for p in [train, val, test]):
        return train, val, test, "custom"

    raise FileNotFoundError("Could not resolve train/val/test paths. Provide --train/--val/--test.")


def resolve_synth_path(synth_dir, data_type, synth_path):
    if synth_path and os.path.exists(synth_path):
        return synth_path
    candidates = []
    if data_type in ("binary", "multiclass"):
        candidates.append(os.path.join(synth_dir, f"synthetic_{data_type}_data.csv"))
    candidates.append(os.path.join(synth_dir, "synthetic_data.csv"))
    found = _first_existing(candidates)
    if not found:
        raise FileNotFoundError("Synthetic data file not found. Provide --synth.")
    return found




def main(args):
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    data_type_input = None if args.data_type == "auto" else args.data_type
    train_file, val_file, test_file, data_type = resolve_real_paths(
        args.data_dir, data_type_input, args.train, args.val, args.test
    )
    print(f"Loading {data_type} data...")
    
    # Load real datasets
    X_train, y_train = load_data(train_file)
    X_val, y_val = load_data(val_file)
    X_test, y_test = load_data(test_file)
    
    print(f"Loaded real data shapes:")
    print(f"  Training: {X_train.shape}, {y_train.shape}")
    print(f"  Validation: {X_val.shape}, {y_val.shape}")
    print(f"  Test: {X_test.shape}, {y_test.shape}")
    
    # Load synthetic data
    synth_file = resolve_synth_path(args.synth_dir, data_type, args.synth)
    print(f"Loading synthetic data from {synth_file}")
    df_synth = pd.read_csv(synth_file)
    X_synth = torch.tensor(df_synth.iloc[:, :-1].values, dtype=torch.float32)
    y_synth = torch.tensor(df_synth.iloc[:, -1].values, dtype=torch.long)
    print(f"Loaded synthetic data shape: {X_synth.shape}, {y_synth.shape}")
    
    # Get number of classes and features
    num_classes = len(torch.unique(y_train))
    input_dim = X_train.shape[1]
    
    print(f"Number of classes: {num_classes}")
    print(f"Input dimension: {input_dim}")
    
    # Create class names list
    class_names = [str(i) for i in range(num_classes)]
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    synth_dataset = TensorDataset(X_synth, y_synth)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    synth_loader = DataLoader(synth_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model based on architecture
    model_classes = {
        'mlp': MLP,
        'dnn': DNN,
        'lstm': LSTM,
        'gru': GRU,
        'cnn': CNN
    }
    
    if args.model not in model_classes:
        print(f"Error: Unknown model architecture '{args.model}'")
        print(f"Available options: {list(model_classes.keys())}")
        sys.exit(1)
    
    # Create models
    model_class = model_classes[args.model]
    
    if args.model in ['lstm', 'gru', 'cnn']:
        # For sequential models, we need to reshape the input
        sequence_length = args.sequence_length
        
        real_model = model_class(
            input_dim=input_dim//sequence_length, 
            num_class=num_classes,
            hidden_layer_list=None,
            sequence_length=sequence_length
        )
        
        synth_model = model_class(
            input_dim=input_dim//sequence_length, 
            num_class=num_classes,
            hidden_layer_list=None,
            sequence_length=sequence_length
        )
    else:
        # For non-sequential models (MLP, DNN)
        real_model = model_class(
            input_dim=input_dim, 
            num_class=num_classes,
            hidden_layer_list=None
        )
        
        synth_model = model_class(
            input_dim=input_dim, 
            num_class=num_classes,
            hidden_layer_list=None
        )
    
    print(f"\nTraining model on real data...")
    real_model, real_history = train_model(
        real_model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, device=device,
        early_stop=args.early_stop
    )
    
    print(f"\nTraining model on synthetic data...")
    synth_model, synth_history = train_model(
        synth_model, synth_loader, val_loader,
        epochs=args.epochs, lr=args.lr, device=device,
        early_stop=args.early_stop
    )
    
    print("\nEvaluating model trained on real data...")
    real_acc, real_f1, real_report, real_cm, real_preds, _ = evaluate_model(
        real_model, test_loader, device=device
    )
    
    print("\nEvaluating model trained on synthetic data...")
    synth_acc, synth_f1, synth_report, synth_cm, synth_preds, _ = evaluate_model(
        synth_model, test_loader, device=device
    )
    
    # Plot training curves
    curves_path = os.path.join(args.output_dir, f"{args.model}_{data_type}_training_curves.png")
    plot_training_curves(real_history, synth_history, curves_path)
    
    # Plot confusion matrices
    cm_path = os.path.join(args.output_dir, f"{args.model}_{data_type}_confusion_matrices.png")
    plot_confusion_matrices(real_cm, synth_cm, class_names, cm_path)
    
    # Save results
    results = {
        "dataset_type": data_type,
        "model_architecture": args.model,
        "real_data_size": int(X_train.shape[0]),
        "synthetic_data_size": int(X_synth.shape[0]),
        "compression_ratio": float(X_train.shape[0] / X_synth.shape[0]),
        "real_model": {
            "accuracy": float(real_acc),
            "f1_score": float(real_f1),
            "training_time": float(real_history["training_time"]),
            "report": real_report
        },
        "synthetic_model": {
            "accuracy": float(synth_acc),
            "f1_score": float(synth_f1),
            "training_time": float(synth_history["training_time"]),
            "report": synth_report
        }
    }
    
    # Save results
    result_file = os.path.join(
        args.output_dir, f"{args.model}_{data_type}_comparison_results.json"
    )
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {result_file}")
    print(f"Training curves saved to {curves_path}")
    print(f"Confusion matrices saved to {cm_path}")
    print(f"\nSummary:")
    print(f"  Real Data Model Accuracy: {real_acc:.4f}")
    print(f"  Synthetic Data Model Accuracy: {synth_acc:.4f}")
    print(f"  Accuracy Difference: {abs(real_acc - synth_acc):.4f}")
    print(f"  Compression Ratio: {results['compression_ratio']:.2f}x")
    
    # Relative efficiency metric
    efficiency = (synth_acc / real_acc) * results['compression_ratio']
    print(f"  Relative Efficiency (Accuracy Ratio × Compression): {efficiency:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare models trained on real vs synthetic data")
    parser.add_argument("--data_dir", type=str, default="../Data_IoT", 
                        help="Directory containing the real dataset files")
    parser.add_argument("--synth_dir", type=str, default="../synthetic_data", 
                        help="Directory containing the synthetic dataset files")
    parser.add_argument("--train", type=str, default=None, help="Train CSV path (last column = label)")
    parser.add_argument("--val", type=str, default=None, help="Validation CSV path (last column = label)")
    parser.add_argument("--test", type=str, default=None, help="Test CSV path (last column = label)")
    parser.add_argument("--synth", type=str, default=None, help="Synthetic CSV path (last column = label)")
    parser.add_argument("--output_dir", type=str, default="../comparison_results", 
                        help="Directory to save results and plots")
    parser.add_argument("--data-type", type=str, default="auto",
                        choices=["auto", "binary", "multiclass"],
                        help="Auto-detect or force dataset type")
    parser.add_argument("--model", type=str, default="mlp", 
                        choices=["mlp", "dnn", "lstm", "gru", "cnn"],
                        help="Model architecture to use")
    parser.add_argument("--sequence_length", type=int, default=4,
                        help="Sequence length for sequential models (LSTM, GRU, CNN)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--early_stop", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU usage even if GPU is available")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    main(args)
