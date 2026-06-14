#!/usr/bin/env python3
"""
Adaptive Time Alignment (ATA) Training Script for IoT Attack Detection

This script trains a lightweight adaptive temporal alignment model
based on AdaRNN without boosting. It uses classification loss and
domain transfer loss to handle temporal distribution shift.

Usage:
    python ata_model/ata_main.py --num_domains 3 --loss_type mmd --dw 0.5

Key hyperparameters:
    --num_domains: Number of temporal domains to split training data
    --loss_type: Transfer loss type (mmd, coral, adv, cosine)
    --dw: Domain alignment weight (0.1-1.0)
"""

import argparse
import os
import sys
import json
import datetime
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for direct script execution.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ata_model.components.models.adarnn import AdaRNNClassifier, AdaRNNMLP
from ata_model.components.data.ata_dataloader import create_ata_dataloaders, get_ata_domain_pairs


def pprint(*text):
    """Print with timestamp"""
    time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{time_str}]', *text, flush=True)


def get_model(args):
    """Create AdaRNN model based on config"""
    n_hiddens = [args.hidden_size for _ in range(args.num_layers)]
    
    if args.model_type == 'adarnn_mlp':
        # MLP version for tabular data (no sequence)
        model = AdaRNNMLP(
            input_dim=args.input_dim,
            hidden_dims=n_hiddens,
            num_classes=args.num_classes,
            dropout=args.dropout,
            trans_loss=args.loss_type
        )
    else:
        # RNN version
        model = AdaRNNClassifier(
            input_dim=args.input_dim,
            hidden_dims=n_hiddens,
            num_classes=args.num_classes,
            seq_len=args.seq_len,
            dropout=args.dropout,
            trans_loss=args.loss_type,
            use_bottleneck=args.use_bottleneck,
            bottleneck_dim=args.bottleneck_dim
        )
    
    return model


def load_yaml_config(config_path):
    """Load yaml config file as dict"""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError('PyYAML is required for --config. Install with: pip install pyyaml') from exc

    with open(config_path, 'r') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError('Config file must be a YAML mapping at top level')
    return data


def apply_config_to_args(args, cfg):
    """Merge structured config sections into argparse args namespace"""
    data_cfg = cfg.get('data_config', {})
    model_cfg = cfg.get('model_config', {})
    train_cfg = cfg.get('train_config', {})

    # Backward compatibility with flat keys
    flat = cfg.copy()

    mapping = {
        'data_dir': [data_cfg.get('data_dir'), flat.get('data_dir')],
        'train_data_path': [data_cfg.get('train_data_path'), flat.get('train_data_path')],
        'val_data_path': [data_cfg.get('val_data_path'), flat.get('val_data_path')],
        'test_data_path': [data_cfg.get('test_data_path'), flat.get('test_data_path')],
        'label_col': [data_cfg.get('label_col'), flat.get('label_col')],
        'time_col': [data_cfg.get('time_col'), flat.get('time_col')],
        'feature_cols': [data_cfg.get('feature_cols'), flat.get('feature_cols')],
        'load_mode': [data_cfg.get('load_mode'), flat.get('load_mode')],
        'split_method': [data_cfg.get('split_method'), flat.get('split_method')],
        'num_domains': [data_cfg.get('num_domains'), train_cfg.get('num_domains'), flat.get('num_domains')],
        'batch_size': [data_cfg.get('batch_size'), train_cfg.get('batch_size'), flat.get('batch_size')],
        'normalize': [data_cfg.get('normalize'), flat.get('normalize')],
        'train_shuffle': [data_cfg.get('train_shuffle'), flat.get('train_shuffle')],
        'output_dir': [train_cfg.get('output_dir'), flat.get('output_dir')],
        'model_type': [model_cfg.get('model_type'), flat.get('model_type')],
        'hidden_size': [model_cfg.get('hidden_size'), flat.get('hidden_size')],
        'num_layers': [model_cfg.get('num_layers'), flat.get('num_layers')],
        'dropout': [model_cfg.get('dropout'), flat.get('dropout')],
        'use_bottleneck': [model_cfg.get('use_bottleneck'), flat.get('use_bottleneck')],
        'bottleneck_dim': [model_cfg.get('bottleneck_dim'), flat.get('bottleneck_dim')],
        'seq_len': [model_cfg.get('seq_len'), flat.get('seq_len')],
        'loss_type': [model_cfg.get('loss_type'), flat.get('loss_type')],
        'dw': [train_cfg.get('dw'), flat.get('dw')],
        'len_win': [train_cfg.get('len_win'), flat.get('len_win')],
        'n_epochs': [train_cfg.get('n_epochs'), flat.get('n_epochs')],
        'optimizer': [train_cfg.get('optimizer'), flat.get('optimizer')],
        'lr': [train_cfg.get('lr'), flat.get('lr')],
        'weight_decay': [train_cfg.get('weight_decay'), flat.get('weight_decay')],
        'early_stop': [train_cfg.get('early_stop'), flat.get('early_stop')],
        'use_class_weights': [train_cfg.get('use_class_weights'), flat.get('use_class_weights')],
        'attack_weight': [train_cfg.get('attack_weight'), flat.get('attack_weight')],
        'seed': [train_cfg.get('seed'), flat.get('seed')],
    }

    for key, candidates in mapping.items():
        for value in candidates:
            if value is not None:
                setattr(args, key, value)
                break
    return args


def resolve_data_paths(args):
    """Resolve train/val/test paths from explicit values or data_dir"""
    if args.train_data_path and args.val_data_path and args.test_data_path:
        return args.train_data_path, args.val_data_path, args.test_data_path

    train_path = os.path.join(args.data_dir, 'train.csv')
    val_path = os.path.join(args.data_dir, 'val.csv')
    test_path = os.path.join(args.data_dir, 'test.csv')
    return train_path, val_path, test_path


def resolve_feature_columns(train_df, args):
    """Resolve feature columns from config or infer by excluding time/label columns"""
    if args.feature_cols:
        if isinstance(args.feature_cols, str):
            feature_cols = [c.strip() for c in args.feature_cols.split(',') if c.strip()]
        elif isinstance(args.feature_cols, (list, tuple)):
            feature_cols = [str(c).strip() for c in args.feature_cols if str(c).strip()]
        else:
            raise TypeError('feature_cols must be a comma-separated string or a list of column names.')

        if not feature_cols:
            raise ValueError('feature_cols was provided but no valid columns were parsed.')

        missing_cols = [c for c in feature_cols if c not in train_df.columns]
        if missing_cols:
            raise ValueError(f'feature_cols contains unknown columns: {missing_cols}')
        return feature_cols

    feature_cols = [c for c in train_df.columns if c not in {args.label_col, args.time_col}]
    if not feature_cols:
        raise ValueError('No feature columns found. Check label_col/time_col or feature_cols config.')
    return feature_cols


def train_ata_epoch(args, model, optimizer, train_loader_list, epoch):
    """
    Adaptive Time Alignment training step (no boosting)
    """
    model.train()
    device = next(model.parameters()).device
    
    # Use class weights if provided
    if hasattr(args, 'class_weights') and args.class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=args.class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    loss_all = []
    # Get minimum loader length
    min_len = min(len(loader) for loader in train_loader_list)
    
    # Iterate over all domain pairs
    domain_pairs = get_ata_domain_pairs(len(train_loader_list))
    
    # Create iterators for each loader
    iterators = [iter(loader) for loader in train_loader_list]
    
    for batch_idx in tqdm(range(min_len), desc=f'Epoch {epoch}', leave=False):
        # Get batch from each domain
        batches = []
        for it in iterators:
            try:
                batch = next(it)
            except StopIteration:
                break
            batches.append(batch)
        
        if len(batches) != len(train_loader_list):
            break
        
        optimizer.zero_grad()
        
        total_loss = torch.zeros(1, device=device)
        
        # Process each domain pair
        for s1, s2 in domain_pairs:
            feat_s, label_s = batches[s1]
            feat_t, label_t = batches[s2]
            
            feat_s = feat_s.to(device)
            feat_t = feat_t.to(device)
            label_s = label_s.to(device)
            label_t = label_t.to(device)
            
            # Skip if batch sizes don't match
            if feat_s.shape[0] != feat_t.shape[0]:
                continue
            
            # Always use transfer learning without boosting.
            pred_s, pred_t, loss_transfer, _ = model.forward_with_transfer(
                feat_s, feat_t, len_win=args.len_win
            )
            
            # Classification loss
            loss_s = criterion(pred_s, label_s)
            loss_t = criterion(pred_t, label_t)
            
            # Total loss = classification + domain alignment
            total_loss = total_loss + loss_s + loss_t + args.dw * loss_transfer
        
        if total_loss.item() > 0:
            loss_all.append([total_loss.item(), (loss_s + loss_t).item(), loss_transfer.item()])
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
    
    loss_avg = np.array(loss_all).mean(axis=0) if loss_all else [0, 0, 0]
    
    return loss_avg


def evaluate(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    avg_loss = total_loss / len(test_loader)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'predictions': all_preds,
        'labels': all_labels
    }


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(args):
    """Main training function"""
    pprint('=' * 50)
    pprint('Adaptive Time Alignment (ATA) Training for IoT Attack Detection')
    pprint('=' * 50)
    pprint(f'Config: {vars(args)}')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pprint(f'Using device: {device}')
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Set optimizer type (default to Adam)
    if not hasattr(args, 'optimizer'):
        args.optimizer = 'adam'
    args.optimizer = str(args.optimizer).lower().strip()
    
    # Create output directory
    output_dir = os.path.join(
        args.output_dir,
        f'ata_{args.loss_type}_domains{args.num_domains}_dw{args.dw}_{args.optimizer}'
    )
    os.makedirs(output_dir, exist_ok=True)
    
    train_path, val_path, test_path = resolve_data_paths(args)

    train_df = pd.read_csv(train_path)
    feature_cols = resolve_feature_columns(train_df, args)
    pprint(f'Feature columns ({len(feature_cols)}): {feature_cols}')
    
    # Update input dimension
    args.input_dim = len(feature_cols)
    
    # Create dataloaders
    pprint('Creating dataloaders...')
    train_loaders, val_loader, test_loader, scaler = create_ata_dataloaders(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        feature_cols=feature_cols,
        label_col=args.label_col,
        time_col=args.time_col,
        num_domains=args.num_domains,
        batch_size=args.batch_size,
        normalize=args.normalize,
        split_method=args.split_method,
        load_mode=args.load_mode,
        train_shuffle=args.train_shuffle,
        random_state=args.seed
    )
    
    pprint(f'Created {len(train_loaders)} domain loaders')
    
    # Determine number of classes from all splits (not just training)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    all_classes = pd.concat([
        train_df[args.label_col],
        val_df[args.label_col],
        test_df[args.label_col]
    ]).unique()
    args.num_classes = len(all_classes)
    pprint(f'Number of classes: {args.num_classes}')
    
    # Calculate class weights for imbalanced data
    if args.use_class_weights:
        class_counts = train_df[args.label_col].value_counts().sort_index()
        total = len(train_df)
        # Inverse frequency weighting with attack multiplier
        weights = torch.tensor([1.0, args.attack_weight], dtype=torch.float32)
        args.class_weights = weights
        pprint(f'Using class weights: {weights.numpy()}')
    else:
        args.class_weights = None
    
    # Create model
    pprint('Creating model...')
    model = get_model(args).to(device)
    num_params = count_parameters(model)
    pprint(f'Model parameters: {num_params:,}')
    
    # Optimizer - support multiple optimizer types
    if args.optimizer in ['adamw', 'adamamba']:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        pprint(f'Using AdamW optimizer')
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        pprint(f'Using Adam optimizer')
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    best_val_f1 = 0
    best_epoch = 0
    stop_round = 0
    history = {
        'train_loss': [], 'train_cls_loss': [], 'train_trans_loss': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
        'test_loss': [], 'test_acc': [], 'test_f1': []
    }
    
    for epoch in range(args.n_epochs):
        pprint(f'\n--- Epoch {epoch + 1}/{args.n_epochs} ---')
        
        # Training
        loss_avg = train_ata_epoch(args, model, optimizer, train_loaders, epoch)
        pprint(f'Train Loss: {loss_avg[0]:.4f} (cls: {loss_avg[1]:.4f}, trans: {loss_avg[2]:.4f})')
        
        history['train_loss'].append(loss_avg[0])
        history['train_cls_loss'].append(loss_avg[1])
        history['train_trans_loss'].append(loss_avg[2])
        
        # Validation
        val_metrics = evaluate(model, val_loader, device)
        pprint(f'Val Loss: {val_metrics["loss"]:.4f}, Acc: {val_metrics["accuracy"]:.4f}, F1: {val_metrics["f1_macro"]:.4f}')
        
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1_macro'])
        
        # Test
        test_metrics = evaluate(model, test_loader, device)
        pprint(f'Test Loss: {test_metrics["loss"]:.4f}, Acc: {test_metrics["accuracy"]:.4f}, F1: {test_metrics["f1_macro"]:.4f}')
        
        history['test_loss'].append(test_metrics['loss'])
        history['test_acc'].append(test_metrics['accuracy'])
        history['test_f1'].append(test_metrics['f1_macro'])
        
        # Update scheduler
        scheduler.step(val_metrics['loss'])
        
        # Save best model
        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            best_epoch = epoch
            stop_round = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'args': vars(args)
            }, os.path.join(output_dir, 'best_model.pth'))
            
            pprint(f'>> Saved best model (Val F1: {best_val_f1:.4f})')
        else:
            stop_round += 1
            if stop_round >= args.early_stop:
                pprint(f'Early stopping at epoch {epoch}')
                break
    
    # Load best model and final evaluation
    pprint('\n' + '=' * 50)
    pprint('Final Evaluation')
    pprint('=' * 50)
    
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    final_test = evaluate(model, test_loader, device)
    
    pprint(f'Best Epoch: {best_epoch}')
    pprint(f'Test Accuracy: {final_test["accuracy"]:.4f}')
    pprint(f'Test F1 (Macro): {final_test["f1_macro"]:.4f}')
    pprint(f'Test F1 (Weighted): {final_test["f1_weighted"]:.4f}')
    
    # Classification report
    pprint('\nClassification Report:')
    print(classification_report(final_test['labels'], final_test['predictions']))
    
    # Confusion matrix
    pprint('\nConfusion Matrix:')
    cm = confusion_matrix(final_test['labels'], final_test['predictions'])
    print(cm)
    
    # Save results
    results = {
        'best_epoch': best_epoch,
        'best_val_f1': best_val_f1,
        'test_accuracy': final_test['accuracy'],
        'test_f1_macro': final_test['f1_macro'],
        'test_f1_weighted': final_test['f1_weighted'],
        'num_parameters': count_parameters(model),
        'args': vars(args),
        'history': history
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    pprint(f'\nResults saved to {output_dir}')
    
    return results


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Adaptive Time Alignment (ATA) Training for IoT Attack Detection')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to ATA yaml config file')
    parser.add_argument('--seed', type=int, default=None,
                       help='Override random seed')
    parser.add_argument('--train_data_path', type=str, default=None,
                       help='Override train csv path')
    parser.add_argument('--val_data_path', type=str, default=None,
                       help='Override val csv path')
    parser.add_argument('--test_data_path', type=str, default=None,
                       help='Override test csv path')
    parser.add_argument('--time_col', type=str, default=None,
                       help='Override time column name')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Override output directory')

    # Data
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--label_col', type=str, default='label',
                       help='Label column name')
    parser.add_argument('--feature_cols', type=str, default=None,
                       help='Comma separated feature names for CLI. Config also supports YAML list.')
    parser.add_argument('--normalize', action=argparse.BooleanOptionalAction, default=True,
                       help='Apply StandardScaler normalization')
    parser.add_argument('--load_mode', type=str, default='time_series',
                       choices=['time_series', 'random_shuffle'],
                       help='Data loading mode: preserve time order or randomly shuffle')
    parser.add_argument('--train_shuffle', action=argparse.BooleanOptionalAction, default=True,
                       help='Shuffle batches in train loaders')
    
    # Model
    parser.add_argument('--model_type', type=str, default='adarnn',
                       choices=['adarnn', 'adarnn_mlp'],
                       help='Model type')
    parser.add_argument('--hidden_size', type=int, default=64,
                       help='Hidden size for each layer')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of RNN layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--use_bottleneck', action='store_true', default=True,
                       help='Use bottleneck layer')
    parser.add_argument('--bottleneck_dim', type=int, default=64,
                       help='Bottleneck dimension')
    parser.add_argument('--seq_len', type=int, default=1,
                       help='Sequence length')
    
    # AdaRNN specific
    parser.add_argument('--num_domains', type=int, default=3,
                       help='Number of temporal domains')
    parser.add_argument('--split_method', type=str, default='quantile',
                       choices=['quantile', 'tdc', 'manual'],
                       help='Method to split domains')
    parser.add_argument('--loss_type', type=str, default='mmd',
                       choices=['mmd', 'mmd_rbf', 'coral', 'adv', 'cosine'],
                       help='Transfer loss type')
    parser.add_argument('--dw', type=float, default=0.5,
                       help='Domain alignment weight')
    parser.add_argument('--len_win', type=int, default=0,
                       help='Window size for temporal alignment')
    
    # Training
    parser.add_argument('--n_epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'adamw', 'adamamba'],
                       help='Optimizer type (adam, adamw, adamamba)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--early_stop', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--use_class_weights', action='store_true', default=False,
                       help='Use class weights for imbalanced data')
    parser.add_argument('--attack_weight', type=float, default=3.0,
                       help='Weight multiplier for attack class (only if use_class_weights)')
    
    # Other
    parser.add_argument('--input_dim', type=int, default=27,
                       help='Input dimension (auto-set from data)')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of classes (auto-set from data)')
    
    args = parser.parse_args()
    cli_overrides = {
        'seed': args.seed,
        'train_data_path': args.train_data_path,
        'val_data_path': args.val_data_path,
        'test_data_path': args.test_data_path,
        'time_col': args.time_col,
        'output_dir': args.output_dir,
    }
    if args.config:
        cfg = load_yaml_config(args.config)
        args = apply_config_to_args(args, cfg)

    for key, value in cli_overrides.items():
        if value is not None:
            setattr(args, key, value)
    return args


if __name__ == '__main__':
    args = get_args()
    main(args)
