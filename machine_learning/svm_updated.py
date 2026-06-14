"""
Updated SVM classifier for IPv6 attack detection.
Supports both binary and multiclass classification with full features or PCA.
Added class_weight='balanced' option for handling imbalanced datasets.
"""
import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, balanced_accuracy_score
from collections import Counter

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from machine_learning.ml_dataloader_updated import get_ml_data


def format_model_name(base, task, mode, *extras):
    """
    Build a consistent tag for saved artifacts (charts, reports).
    Example: svm_binary_full_feature_balanced
    """
    clean_extras = [e for e in extras if e]
    parts = [base, task, mode, *clean_extras]
    return "_".join(parts)


def analyze_class_distribution(y_train):
    """Analyze and print class distribution"""
    counter = Counter(y_train)
    total = len(y_train)
    print("\nClass Distribution:")
    for class_id in sorted(counter.keys()):
        count = counter[class_id]
        percentage = (count / total) * 100
        print(f"  Class {class_id}: {count:6d} samples ({percentage:6.2f}%)")


def plot_confusion_matrix(y_true, y_pred, dataset_type, model_name):
    """Plot and save confusion matrix, and print to terminal"""
    import pandas as pd
    cm = confusion_matrix(y_true, y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    
    # Print as table
    print(f"\nConfusion Matrix - {dataset_type.capitalize()} - {model_name}:")
    df_cm = pd.DataFrame(cm, index=[f"True {l}" for l in labels], columns=[f"Pred {l}" for l in labels])
    print(df_cm)
    
    # Plot as heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('PREDICTED')
    plt.ylabel('TRUE')
    plt.title(f'CONFUSION MATRIX - {dataset_type.upper()} - {model_name.upper()}')
    if not os.path.exists('img'):
        os.makedirs('img')
    plt.savefig(os.path.join("img", f"confusion_matrix_{dataset_type}_{model_name}.png"))
    plt.close()


def train_and_evaluate_svm(X_train, y_train, X_val, y_val, X_test, y_test, 
                          kernel='rbf', C=1.0, use_sgd=False, random_state=42, 
                          use_class_weight=False, model_name="svm"):
    """Train and evaluate SVM model"""
    
    # Analyze class distribution
    analyze_class_distribution(y_train)
    
    # Set class_weight parameter
    class_weight = 'balanced' if use_class_weight else None
    
    if use_sgd or X_train.shape[0] > 10000:  # Use SGD for large datasets
        balance_info = "with class balancing" if use_class_weight else "without class balancing"
        print(f"Using SGD-based SVM for large dataset {balance_info}...")
        
        svm = SGDClassifier(
            loss='hinge',
            alpha=1.0/C,
            random_state=random_state,
            max_iter=2000,  # Increased for better convergence
            tol=1e-3,
            class_weight=class_weight,
            n_jobs=-1
        )
    else:
        balance_info = "with class balancing" if use_class_weight else "without class balancing"
        print(f"Using standard SVM with {kernel} kernel {balance_info}...")
        
        svm = SVC(
            kernel=kernel,
            C=C,
            random_state=random_state,
            class_weight=class_weight,
            probability=True  # Enable probability estimates
        )
    
    # Train model
    print("Training SVM...")
    svm.fit(X_train, y_train)
    
    # Make predictions
    y_val_pred = svm.predict(X_val)
    y_test_pred = svm.predict(X_test)
    
    # Evaluate model
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_balanced_accuracy = balanced_accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred, average='macro', zero_division=0)
    val_recall = recall_score(y_val, y_val_pred, average='macro', zero_division=0)
    val_f1 = f1_score(y_val, y_val_pred, average='macro', zero_division=0)
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_balanced_accuracy = balanced_accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average='macro', zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
    
    print("\nValidation Results:")
    print(f"Accuracy: {val_accuracy:.4f}")
    print(f"Balanced Accuracy: {val_balanced_accuracy:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"Recall: {val_recall:.4f}")
    print(f"F1 Score: {val_f1:.4f}")
    
    print("\nTest Results:")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Balanced Accuracy: {test_balanced_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    
    # Plot confusion matrices
    plot_confusion_matrix(y_val, y_val_pred, "validation", model_name)
    plot_confusion_matrix(y_test, y_test_pred, "test", model_name)
    
    # Print detailed classification report
    print("\nDetailed Test Classification Report:")
    print(classification_report(y_test, y_test_pred, zero_division=0))
    
    return svm, val_accuracy, test_accuracy


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate SVM')
    parser.add_argument('--task', type=str, choices=['binary', 'multiclass'], default='binary',
                       help='Whether to run binary or multiclass classification')
    parser.add_argument('--mode', type=str, choices=['full_feature', 'pca', 'lda'], default='full_feature',
                       help='Whether to use full features, PCA features, or LDA features')
    parser.add_argument('--kernel', type=str, choices=['linear', 'rbf', 'poly', 'sigmoid'], 
                       default='rbf', help='SVM kernel type')
    parser.add_argument('--C', type=float, default=1.0, 
                       help='Regularization parameter')
    parser.add_argument('--use_sgd', action='store_true', 
                       help='Use SGD-based SVM for large datasets')
    parser.add_argument('--balance', action='store_true',
                       help='Use class_weight="balanced" to handle class imbalance')
    parser.add_argument('--config', type=str, default='configs/Machine_learning.yaml', 
                       help='Path to the ML configuration file (ignored if train/val/test paths are provided)')
    parser.add_argument('--train_path', type=str, default=None, help='Path to train CSV (if set, bypass config)')
    parser.add_argument('--val_path', type=str, default=None, help='Path to val CSV')
    parser.add_argument('--test_path', type=str, default=None, help='Path to test CSV')
    parser.add_argument('--label_col', type=str, default=None, help='Optional label column name (default: last column)')
    parser.add_argument('--no_normalize', action='store_true', help='Disable StandardScaler normalization')
    parser.add_argument('--pca_components', type=float, default=None, help='PCA components (int or <1 for variance). If mode=pca and None, defaults to 0.95')
    
    args = parser.parse_args()
    
    # Get data using the unified dataloader
    X_train, y_train, X_val, y_val, X_test, y_test = get_ml_data(
        config_path=args.config,
        task=args.task,
        mode=args.mode,
        normalize=not args.no_normalize,
        pca_components=args.pca_components,
        label_col=args.label_col,
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
    )
    
    print(f"Running SVM with {args.task} task, {args.mode} mode")
    print(f"Training data shape: {X_train.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    print(f"Class balancing: {'Enabled' if args.balance else 'Disabled'}")
    
    # Auto-enable SGD for large datasets
    if X_train.shape[0] > 10000 and not args.use_sgd:
        print("Large dataset detected, automatically using SGD-based SVM")
        args.use_sgd = True
    method_suffix = "sgd" if args.use_sgd else None
    balance_suffix = "balanced" if args.balance else None
    model_name = format_model_name("svm", args.task, args.mode, method_suffix, balance_suffix)
    
    # Train and evaluate
    model, val_acc, test_acc = train_and_evaluate_svm(
        X_train, y_train, X_val, y_val, X_test, y_test,
        kernel=args.kernel,
        C=args.C,
        use_sgd=args.use_sgd,
        use_class_weight=args.balance,
        model_name=model_name
    )
    
    print(f"\nFinal SVM model:")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
