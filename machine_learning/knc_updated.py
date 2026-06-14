"""
KNN Classifier example for IPv6 attack detection.

This script demonstrates how to use the K-Nearest Neighbors algorithm
for classifying IPv6 network traffic as normal or attack.
"""
import argparse
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from machine_learning.ml_dataloader_updated import get_ml_data


def format_model_name(base, task, mode, *extras):
    """
    Build a consistent tag for saved artifacts (charts, reports).
    Example: knn_binary_full_feature_k5
    """
    clean_extras = [e for e in extras if e]
    parts = [base, task, mode, *clean_extras]
    return "_".join(parts)


def find_best_k(X_train, y_train, X_val, y_val, k_range=range(1, 21)):
    """
    Find the best k value for KNN classifier by testing a range of k values.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        k_range: Range of k values to test
        
    Returns:
        int: Best k value
    """
    val_accuracy = []
    
    # Test different k values
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_val_pred = knn.predict(X_val)
        accuracy = accuracy_score(y_val, y_val_pred)
        val_accuracy.append(accuracy)
        print(f"k={k}, Validation Accuracy: {accuracy:.4f}")
    
    # Find best k
    best_k = k_range[np.argmax(val_accuracy)]
    
    # Plot accuracy vs k
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, val_accuracy, marker='o')
    plt.title('Validation Accuracy vs k Value')
    plt.xlabel('k (Number of neighbors)')
    plt.ylabel('Accuracy')
    plt.xticks(k_range)
    plt.grid(True)
    
    # Create img directory if it doesn't exist
    if not os.path.exists('img'):
        os.makedirs('img')
    
    plt.savefig(os.path.join("img", "knn_k_selection.png"))
    
    print(f"\nBest k: {best_k} with validation accuracy: {max(val_accuracy):.4f}")
    return best_k

def train_and_evaluate_knn(X_train, y_train, X_val, y_val, X_test, y_test, k=5, model_name="knn"):
    """
    Train and evaluate a KNN classifier.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        k: Number of neighbors
        
    Returns:
        tuple: (model, validation accuracy, test accuracy)
    """
    # Create classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Train model
    print(f"Training KNN with k={k}...")
    knn.fit(X_train, y_train)
    
    # Make predictions
    y_val_pred = knn.predict(X_val)
    y_test_pred = knn.predict(X_test)
    
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
    plot_confusion_matrix(y_val, y_val_pred, "validation", model_name)
    plot_confusion_matrix(y_test, y_test_pred, "test", model_name)
    
    return knn, val_accuracy, test_accuracy

def plot_confusion_matrix(y_true, y_pred, dataset_type, model_name, labels=None):
    """
    Plot and save confusion matrix, and print to terminal as a pretty pandas DataFrame
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        dataset_type: String indicating dataset type (e.g., 'validation', 'test')
        model_name: String indicating model name
        labels: Optional list of label names
    """
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
    
    # Create img directory if it doesn't exist
    if not os.path.exists('img'):
        os.makedirs('img')
    
    plt.savefig(os.path.join("img", f"confusion_matrix_{dataset_type}_{model_name}.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate KNN')
    parser.add_argument('--task', type=str, choices=['binary', 'multiclass'], default='binary',
                       help='Whether to run binary or multiclass classification')
    parser.add_argument('--mode', type=str, choices=['full_feature', 'pca', 'lda'], default='full_feature',
                       help='Whether to use full features, PCA features, or LDA features')
    parser.add_argument('--find_k', action='store_true', help='Search for the best k value')
    parser.add_argument('--k', type=int, default=5, help='Number of neighbors for KNN')
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
    
    print(f"Running KNN with {args.task} task, {args.mode} mode")
    print(f"Training data shape: {X_train.shape}")
    
    # Find best k if requested
    if args.find_k:
        best_k = find_best_k(X_train, y_train, X_val, y_val)
        k = best_k
    else:
        k = args.k

    model_name = format_model_name("knn", args.task, args.mode, f"k{k}")
    
    # Train and evaluate
    model, val_acc, test_acc = train_and_evaluate_knn(
        X_train, y_train, X_val, y_val, X_test, y_test, k, model_name=model_name
    )
    
    print(f"\nFinal KNN model (k={k}):")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
