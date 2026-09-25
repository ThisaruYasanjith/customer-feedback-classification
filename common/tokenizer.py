"""
Tokenizer Management Module.

Provides functions for fitting, transforming, saving, and loading
Keras text Tokenizers to prevent data leakage.
"""

import pickle
import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def create_and_fit_tokenizer(
    train_texts: list | np.ndarray,
    max_words: int = 10000,
    oov_token: str = "<OOV>"
) -> Tokenizer:
    """
    Create and fit a Keras Tokenizer STRICTLY on training text data
    to prevent data leakage.

    Args:
        train_texts: Collection of training text strings.
        max_words (int): Maximum vocabulary size to consider.
        oov_token (str): Out-of-vocabulary token placeholder.

    Returns:
        Tokenizer: Fitted Keras Tokenizer instance.
    """
    tokenizer = Tokenizer(num_words=max_words, oov_token=oov_token)
    tokenizer.fit_on_texts(train_texts)
    return tokenizer


def texts_to_padded_sequences(
    tokenizer: Tokenizer,
    texts: list | np.ndarray,
    max_length: int = 200,
    padding: str = "post",
    truncating: str = "post"
) -> np.ndarray:
    """
    Convert raw text strings into padded integer token sequences.

    Args:
        tokenizer (Tokenizer): Fitted Keras Tokenizer.
        texts: Collection of text strings.
        max_length (int): Maximum token sequence length.
        padding (str): Padding position ('post' or 'pre').
        truncating (str): Truncation position ('post' or 'pre').

    Returns:
        np.ndarray: Padded integer matrix of shape (N, max_length).
    """
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(
        sequences,
        maxlen=max_length,
        padding=padding,
        truncating=truncating
    )
    return padded


def save_tokenizer(tokenizer: Tokenizer, filepath: str = "data/processed/tokenizer.pkl"):
    """
    Save a fitted Tokenizer instance to disk as a pickle file.

    Args:
        tokenizer (Tokenizer): Tokenizer object.
        filepath (str): Target output file path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved successfully to {filepath}")


def load_tokenizer(filepath: str = "data/processed/tokenizer.pkl") -> Tokenizer:
    """
    Load a serialized Keras Tokenizer instance from disk.

    Args:
        filepath (str): Path to pickle file.

    Returns:
        Tokenizer: Loaded Keras Tokenizer instance.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Tokenizer file not found at: {filepath}")

    with open(filepath, "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer
