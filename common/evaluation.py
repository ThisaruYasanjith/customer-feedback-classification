"""
Model Evaluation and Visualization Module.

Provides comprehensive metric calculation routines (Accuracy, Precision, Recall, F1,
ROC-AUC, Confusion Matrix) and visualization tools for model performance comparison.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize


def evaluate_classification_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: list[str] = None
) -> dict:
    """
    Compute comprehensive multi-class classification evaluation metrics.

    Args:
        y_true (np.ndarray): True integer class labels (shape: N,).
        y_pred_probs (np.ndarray): Predicted probability matrix (shape: N, C).
        class_names (list[str]): Names of target classes.

    Returns:
        dict: Metric dictionary containing Accuracy, Precision, Recall, F1 scores, and ROC-AUC.
    """
    if class_names is None:
        class_names = ["Poor", "Average", "Good"]

    num_classes = len(class_names)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Accuracy
    acc = accuracy_score(y_true, y_pred)

    # Macro & Weighted Precision, Recall, F1
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Per-class F1
    per_class_prec, per_class_rec, per_class_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # ROC-AUC (One-vs-Rest)
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        roc_auc_macro = roc_auc_score(
            y_true_bin, y_pred_probs, multi_class="ovr", average="macro"
        )
        roc_auc_weighted = roc_auc_score(
            y_true_bin, y_pred_probs, multi_class="ovr", average="weighted"
        )
    except Exception:
        roc_auc_macro = np.nan
        roc_auc_weighted = np.nan

    metrics = {
        "accuracy": float(acc),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_prec),
        "weighted_recall": float(weighted_rec),
        "weighted_f1": float(weighted_f1),
        "roc_auc_macro": float(roc_auc_macro),
        "roc_auc_weighted": float(roc_auc_weighted),
    }

    for idx, c_name in enumerate(class_names):
        metrics[f"f1_{c_name.lower()}"] = float(per_class_f1[idx])
        metrics[f"precision_{c_name.lower()}"] = float(per_class_prec[idx])
        metrics[f"recall_{c_name.lower()}"] = float(per_class_rec[idx])

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] = None,
    title: str = "Confusion Matrix",
    save_path: str = None
):
    """
    Plot and optionally save a seaborn heatmap of the Confusion Matrix.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted class labels.
        class_names (list[str]): List of class label names.
        title (str): Chart title.
        save_path (str): Optional file path to save image.
    """
    if class_names is None:
        class_names = ["Poor", "Average", "Good"]

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Confusion matrix plot saved to {save_path}")

    plt.show()


def plot_roc_curves(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: list[str] = None,
    title: str = "Receiver Operating Characteristic (ROC) Curves",
    save_path: str = None
):
    """
    Plot One-vs-Rest multi-class ROC-AUC curves.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred_probs (np.ndarray): Model output class probabilities.
        class_names (list[str]): List of class names.
        title (str): Chart title.
        save_path (str): Output file save path.
    """
    if class_names is None:
        class_names = ["Poor", "Average", "Good"]

    num_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    plt.figure(figsize=(8, 6))

    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            color=colors[i % len(colors)],
            lw=2,
            label=f"{class_names[i]} (AUC = {roc_auc:.3f})"
        )

    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"ROC curve plot saved to {save_path}")

    plt.show()


def plot_training_history(
    history,
    title_prefix: str = "Model",
    save_path: str = None
):
    """
    Plot training and validation Loss and Accuracy learning curves side-by-side.

    Args:
        history: Keras History object or history dictionary.
        title_prefix (str): Prefix string for plots.
        save_path (str): Optional file path to save plot.
    """
    hist = history.history if hasattr(history, "history") else history

    epochs = range(1, len(hist["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(epochs, hist["loss"], "bo-", label="Training Loss")
    if "val_loss" in hist:
        axes[0].plot(epochs, hist["val_loss"], "ro-", label="Validation Loss")
    axes[0].set_title(f"{title_prefix} - Loss Curve")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs, hist["accuracy"], "bo-", label="Training Accuracy")
    if "val_accuracy" in hist:
        axes[1].plot(epochs, hist["val_accuracy"], "ro-", label="Validation Accuracy")
    axes[1].set_title(f"{title_prefix} - Accuracy Curve")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Learning curves saved to {save_path}")

    plt.show()


def measure_inference_speed(
    model,
    X_test: np.ndarray,
    batch_size: int = 32,
    num_warmup: int = 5,
    num_runs: int = 10
) -> dict:
    """
    Measure model inference execution speed and throughput.

    Args:
        model: Trained Keras/TensorFlow model.
        X_test (np.ndarray): Test feature matrix.
        batch_size (int): Batch size for prediction.
        num_warmup (int): Warmup inference runs.
        num_runs (int): Number of timed inference iterations.

    Returns:
        dict: Latency per sample (ms), total batch inference time (s), and samples/sec.
    """
    # Warmup runs
    for _ in range(num_warmup):
        _ = model.predict(X_test[:batch_size], batch_size=batch_size, verbose=0)

    # Timed runs
    start_time = time.time()
    for _ in range(num_runs):
        _ = model.predict(X_test, batch_size=batch_size, verbose=0)
    total_time = time.time() - start_time

    avg_time_per_run = total_time / num_runs
    total_samples = len(X_test) * num_runs
    samples_per_sec = total_samples / total_time
    latency_per_sample_ms = (avg_time_per_run / len(X_test)) * 1000

    return {
        "avg_run_time_sec": float(avg_time_per_run),
        "latency_per_sample_ms": float(latency_per_sample_ms),
        "throughput_samples_per_sec": float(samples_per_sec),
    }
