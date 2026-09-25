"""
Text Preprocessing and Data Splitting Module.

Provides functions for text cleaning, sentiment label mapping,
and data-leakage-free train/validation/test splitting.
"""

import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    """
    Clean raw review text by lowercasing, removing HTML tags, URLs,
    and extra whitespace.

    Args:
        text (str): Input text string.

    Returns:
        str: Cleaned text string.
    """
    if not isinstance(text, str):
        text = str(text) if pd.notnull(text) else ""

    text = text.lower()
    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)
    # Remove special characters while preserving single spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_series(series: pd.Series) -> pd.Series:
    """
    Apply text cleaning across a Pandas Series.

    Args:
        series (pd.Series): Series of text strings.

    Returns:
        pd.Series: Series of cleaned text strings.
    """
    return series.apply(clean_text)


def map_ratings_to_categories(
    df: pd.DataFrame,
    rating_col: str = "Rating(Out of 10)"
) -> pd.DataFrame:
    """
    Map numerical ratings out of 10 into 3 sentiment categories:
    - Poor: Rating < 5 (Label 0)
    - Average: 5 <= Rating < 8 (Label 1)
    - Good: Rating >= 8 (Label 2)

    Args:
        df (pd.DataFrame): DataFrame containing raw hotel reviews.
        rating_col (str): Column name containing ratings.

    Returns:
        pd.DataFrame: Processed DataFrame with 'Category' and 'label' columns.
    """
    df_clean = df.dropna(subset=["Review_Text"]).copy()

    def rating_category(rating: float) -> str:
        if rating < 5:
            return "Poor"
        elif rating < 8:
            return "Average"
        else:
            return "Good"

    df_clean["Category"] = df_clean[rating_col].apply(rating_category)

    label_mapping = {"Poor": 0, "Average": 1, "Good": 2}
    df_clean["label"] = df_clean["Category"].map(label_mapping)

    return df_clean


def split_dataset(
    df: pd.DataFrame,
    text_col: str = "Review_Text",
    label_col: str = "label",
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Perform stratified dataset splitting into train, validation, and test sets.

    Default split ratio: 70% Train, 15% Validation, 15% Test.

    Args:
        df (pd.DataFrame): Input DataFrame.
        text_col (str): Column name for review text.
        label_col (str): Column name for target label.
        train_size (float): Proportion for training set.
        val_size (float): Proportion for validation set.
        test_size (float): Proportion for test set.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-5, "Splits must sum to 1.0"

    X = df[text_col]
    y = df[label_col]

    temp_size = val_size + test_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_size, random_state=random_state, stratify=y
    )

    test_prop = test_size / temp_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_prop, random_state=random_state, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
