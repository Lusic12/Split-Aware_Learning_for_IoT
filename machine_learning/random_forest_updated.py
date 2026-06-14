"""
Updated Random Forest classifier for IPv6 attack detection.
Supports both binary and multiclass classification with full features or PCA.
"""
import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from machine_learning.ml_dataloader_updated import get_ml_data


def format_model_name(base, task, mode, *extras):
    """
    Build a consistent tag for saved artifacts (charts, reports).
    Example: rf_binary_full_feature
    """
    clean_extras = [e for e in extras if e]
    parts = [base, task, mode, *clean_extras]
    return "_".join(parts)


def plot_confusion_matrix(y_true, y_pred, dataset_type, model_name, labels=None):
    """Plot and save confusion matrix, and print to terminal as a pretty pandas DataFrame"""
    cm = confusion_matrix(y_true, y_pred)
    if labels is None:
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        labels = [str(l) for l in unique_labels]
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    print(f"\nConfusion Matrix - {dataset_type.capitalize()} - {model_name}:")
    print(df_cm)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('PREDICTED')
    plt.ylabel('TRUE')
    plt.title(f'CONFUSION MATRIX - {dataset_type.upper()} - {model_name.upper()}')
    if not os.path.exists('img'):
        os.makedirs('img')
    plt.savefig(os.path.join("img", f"confusion_matrix_{dataset_type}_{model_name}.png"))
    plt.close()


def plot_feature_importance(model, feature_names, top_n=10, model_name="rf"):
    """Plot feature importance"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1][:top_n]
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Top {top_n} Feature Importance - Random Forest')
        plt.bar(range(top_n), importance[indices])
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        
        if not os.path.exists('img'):
            os.makedirs('img')
        plt.savefig(f'img/{model_name}_feature_importance.png')
        plt.close()


def train_and_evaluate_rf(X_train, y_train, X_val, y_val, X_test, y_test, 
                         n_estimators=100, max_depth=None, random_state=42, model_name="rf"):
    """Train and evaluate Random Forest model"""
    
    # Create classifier
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    
    # Train model
    print(f"Training Random Forest with {n_estimators} estimators...")
    rf.fit(X_train, y_train)
    
    # Make predictions
    y_val_pred = rf.predict(X_val)
    y_test_pred = rf.predict(X_test)
    
    # Evaluate model
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred, average='macro')
    val_recall = recall_score(y_val, y_val_pred, average='macro')
    val_f1 = f1_score(y_val, y_val_pred, average='macro')
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='macro')
    test_recall = recall_score(y_test, y_test_pred, average='macro')
    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    
    print("\nValidation Results:")
    print(f"Accuracy: {val_accuracy:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"Recall: {val_recall:.4f}")
    print(f"F1 Score: {val_f1:.4f}")
    
    print("\nTest Results:")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    
    # Plot confusion matrices
    plot_confusion_matrix(y_val, y_val_pred, "validation", model_name, labels=None)
    plot_confusion_matrix(y_test, y_test_pred, "test", model_name, labels=None)
    
    # Print detailed classification report
    print("\nDetailed Test Classification Report:")
    print(classification_report(y_test, y_test_pred))
    
    return rf, val_accuracy, test_accuracy


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate Random Forest')
    parser.add_argument('--task', type=str, choices=['binary', 'multiclass'], default='binary',
                       help='Whether to run binary or multiclass classification')
    parser.add_argument('--mode', type=str, choices=['full_feature', 'pca', 'lda'], default='full_feature',
                       help='Whether to use full features, PCA features, or LDA features')
    parser.add_argument('--n_estimators', type=int, default=100, 
                       help='Number of trees in the forest')
    parser.add_argument('--max_depth', type=int, default=None, 
                       help='Maximum depth of the trees')
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
    
    print(f"Running Random Forest with {args.task} task, {args.mode} mode")
    print(f"Training data shape: {X_train.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    model_name = format_model_name("rf", args.task, args.mode)
    
    # Train and evaluate
    model, val_acc, test_acc = train_and_evaluate_rf(
        X_train, y_train, X_val, y_val, X_test, y_test,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        model_name=model_name
    )
    
    # Plot feature importance if using full features
    if args.mode == 'full_feature' and hasattr(X_train, 'columns'):
        plot_feature_importance(model, X_train.columns.tolist(), model_name=model_name)
    elif args.mode == 'pca':
        pca_features = [f'pca_{i}' for i in range(X_train.shape[1])]
        plot_feature_importance(model, pca_features, model_name=model_name)
    
    print(f"\nFinal Random Forest model:")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
