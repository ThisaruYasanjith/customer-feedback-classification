"""
Data Loading and Serialization Module.

Provides functions for loading raw CSV datasets, reading processed numpy
token sequences, and managing class weights.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight


def load_raw_data(data_path: str = "data/raw/hotel_reviews.csv") -> pd.DataFrame:
    """
    Load raw hotel reviews CSV dataset from disk.

    Args:
        data_path (str): Path to CSV file.

    Returns:
        pd.DataFrame: Loaded raw DataFrame.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Raw dataset file not found at: {data_path}")
    return pd.read_csv(data_path)


def load_processed_data(
    processed_dir: str = "data/processed"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load pre-split padded sequence feature matrices and target vectors.

    Args:
        processed_dir (str): Directory path containing .npy files.

    Returns:
        tuple: (X_train_pad, X_val_pad, X_test_pad, y_train, y_val, y_test)
    """
    X_train_pad = np.load(os.path.join(processed_dir, "X_train_pad.npy"))
    X_val_pad = np.load(os.path.join(processed_dir, "X_val_pad.npy"))
    X_test_pad = np.load(os.path.join(processed_dir, "X_test_pad.npy"))

    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    y_val = np.load(os.path.join(processed_dir, "y_val.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))

    return X_train_pad, X_val_pad, X_test_pad, y_train, y_val, y_test


def compute_and_save_class_weights(
    y_train: np.ndarray,
    save_path: str = "data/processed/class_weights.pkl"
) -> dict:
    """
    Compute balanced class weights using training labels ONLY to address class imbalance.

    Args:
        y_train (np.ndarray): Training target label array.
        save_path (str): File path to save output pickle.

    Returns:
        dict: Mapping of class index to computed weight.
    """
    classes = np.unique(y_train)
    weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    class_weights = dict(zip(classes, weights_arr))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(class_weights, f)

    return class_weights


def load_class_weights(filepath: str = "data/processed/class_weights.pkl") -> dict:
    """
    Load saved class weights dictionary.

    Args:
        filepath (str): Path to pickle file.

    Returns:
        dict: Class weight dictionary.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Class weights file not found at: {filepath}")

    with open(filepath, "rb") as f:
        class_weights = pickle.load(f)
    return class_weights
