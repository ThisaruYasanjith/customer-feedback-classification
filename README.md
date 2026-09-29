# Customer Feedback Classification using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Dataset](https://img.shields.io/badge/Kaggle-Hotel%20Reviews-blue)](https://www.kaggle.com/datasets/juhibhojani/hotel-reviews)

## 📌 Project Overview
This repository contains the official implementation for **SE4050 – Deep Learning (2026).** The objective of this project is to solve a real-world Natural Language Processing (NLP) problem: classifying customer hotel reviews into sentiment ratings using multiple deep learning model architectures.

We implement, train, evaluate, and critically compare **four distinct deep learning architectures**:
1. **Recurrent Neural Network (RNN)** (Bidirectional + Stacked RNN)
2. **Long Short-Term Memory (LSTM)** (Stacked Bidirectional LSTM)
3. **Gated Recurrent Unit (GRU)** (Gated Recurrent Unit Classifier)
4. **1D Convolutional Neural Network (1D-CNN)** (Temporal Convolutional Feature Extractor)

All models are trained and evaluated under fair, controlled, and standardized experimental conditions on an unseen test dataset.

---

## 📊 Dataset Description & Citation

### Source & Link
The dataset used in this project is the **Hotel Reviews Dataset** sourced from Kaggle:
- **Kaggle Dataset**: [Hotel Reviews Dataset by Juhi Bhojani](https://www.kaggle.com/datasets/juhibhojani/hotel-reviews)
- **Citation**: 
  > Bhojani, J. (2023). *Hotel Reviews Dataset*. Kaggle. Available at: https://www.kaggle.com/datasets/juhibhojani/hotel-reviews

### Target Mapping & Class Distribution
The raw dataset contains numerical ratings out of 10. To construct a multi-class sentiment classification task, ratings are grouped into three distinct ordinal sentiment categories:

| Target Category | Rating Scale | Numerical Label | Class Description |
| :--- | :---: | :---: | :--- |
| **Poor** | Rating < 5 | `0` | Negative feedback / Low customer satisfaction |
| **Average** | 5 ≤ Rating < 8 | `1` | Neutral feedback / Moderate customer satisfaction |
| **Good** | Rating ≥ 8 | `2` | Positive feedback / High customer satisfaction |

### Dataset Partitioning & Leakage Prevention
To prevent data leakage, the dataset is split into training, validation, and test subsets prior to text processing and tokenization. The Keras tokenizer and class weights are fitted **strictly on the training set**.

- **Total Cleaned Samples**: 6,994 reviews
- **Train Set (70%)**: 4,895 samples
- **Validation Set (15%)**: 1,049 samples
- **Test Set (15%)**: 1,050 samples (held-out for final evaluation)
- **Sequence Length**: Padded / Truncated to `200` tokens
- **Vocabulary Size**: Top `10,000` most frequent words

---

## 🏗️ Project Architecture & Directory Mapping

```
customer-feedback-classification/
├── README.md                      # Complete project documentation and benchmark guide
├── requirements.txt               # Python package dependencies
├── .gitignore                     # Git ignore rules for cached models and venv
│
├── data/                          # Dataset directory
│   ├── raw/                       # Original raw Kaggle CSV dataset
│   │   └── hotel_reviews.csv      # Raw hotel reviews CSV dataset (7,001 entries)
│   └── processed/                 # Tokenized & padded numpy arrays & saved preprocessing objects
│       ├── X_train_pad.npy        # Training input sequences (4895, 200)
│       ├── X_val_pad.npy          # Validation input sequences (1049, 200)
│       ├── X_test_pad.npy         # Test input sequences (1050, 200)
│       ├── y_train.npy            # Training labels (4895,)
│       ├── y_val.npy              # Validation labels (1049,)
│       ├── y_test.npy             # Test labels (1050,)
│       ├── tokenizer.pkl          # Saved fitted Keras Tokenizer
│       └── class_weights.pkl      # Balanced class weight dictionary
│
├── notebooks/                     # Exploratory Data Analysis & data pipeline
│   └── 01_data_exploration.ipynb  # EDA, rating distribution, text cleaning, and data serialization
│
├── models/                        # Individual deep learning model implementations
│   ├── RNN/                       # Recurrent Neural Network
│   │   ├── rnn_model.py           # RNN model architecture definition
│   │   ├── train_rnn.py           # Training & evaluation script for RNN
│   │   ├── rnn_training.ipynb     # Interactive RNN notebook
│   │   └── results/               # Saved RNN evaluation metrics and history
│   │       ├── rnn_results.json   # Saved RNN test evaluation results (JSON)
│   │       └── rnn_training_history.pkl # Saved RNN training history
│   ├── LSTM/                      # Long Short-Term Memory Network
│   │   ├── lstm_model.py          # Interactive CLI inference script for LSTM
│   │   ├── lstm_training.ipynb    # Interactive LSTM training notebook
│   │   ├── lstm-metrics.ipynb     # LSTM evaluation & metric plots
│   │   └── lstm_review_classifier.keras # Saved trained LSTM model checkpoint
│   ├── GRU/                       # Gated Recurrent Unit Network
│   │   ├── gru_model.py           # GRU model architecture definition
│   │   ├── gru_training.ipynb     # Interactive GRU training notebook
│   │   └── results/               # GRU results folder
│   └── CNN/                       # 1D Convolutional Neural Network
│       ├── cnn_model.py           # 1D-CNN model architecture definition
│       ├── cnn_training.ipynb     # Interactive CNN training notebook
│       └── results/               # CNN results folder
│
├── comparison/                    # Master cross-model comparison
│   └── model_comparison.ipynb     # Comparative evaluation across all models (Acc, F1, ROC-AUC, Latency)
│
└── models_saved/                  # Saved serialized model artifacts & history pickles
    ├── cnn_history.pkl            # Saved 1D-CNN training history
    ├── cnn_model.keras            # Trained 1D-CNN model weight checkpoint
    ├── gru_history.pkl            # Saved GRU training history
    ├── gru_model.keras            # Trained GRU model weight checkpoint
    ├── lstm_history.pkl           # Saved LSTM training history
    ├── lstm_model.keras           # Trained LSTM model weight checkpoint
    ├── rnn_history.pkl            # Saved RNN training history
    ├── rnn_model.keras            # Trained RNN model weight checkpoint
    └── gru/                       # GRU evaluation artifacts
        ├── gru_model.keras        # Trained GRU checkpoint
        ├── gru_history.pkl        # GRU training history
        ├── gru_metrics.json       # GRU test evaluation metrics
        ├── gru_learning_curves.png # GRU training vs validation curves
        └── gru_confusion_matrix.npy # GRU test confusion matrix array
```

---

## ⚙️ Environment Setup & Installation

### Prerequisites
- Python 3.10 or higher
- `pip` or `conda` package manager
- Virtual environment tool (`venv`)

### 1. Clone the Repository
```bash
git clone https://github.com/YourUsername/customer-feedback-classification.git
cd customer-feedback-classification
```

### 2. Create and Activate a Virtual Environment
- **Windows (PowerShell)**:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
- **macOS / Linux**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Install Dependencies
Install all required libraries specified in [`requirements.txt`](file:///c:/Users/yasan/Desktop/DL%20Assignment/customer-feedback-classification/requirements.txt):
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🎯 Evaluated Model Architectures

| Architecture | Layer Pipeline & Key Components | Hyperparameters | Total Parameters | Target Strengths |
| :--- | :--- | :--- | :---: | :--- |
| **SimpleRNN** | `Embedding (128d)` → `Bidirectional SimpleRNN (64, return_seq)` → `Dropout (0.3)` → `SimpleRNN (64)` → `Dropout (0.5)` → `Dense (64, ReLU)` → `Dropout (0.3)` → `Dense (3, Softmax)` | Vocab: 10k, MaxLen: 200, Optimizer: Adam | 1,321,411 | Lightweight sequential baseline model |
| **LSTM** | `Embedding (64d)` → `Bidirectional LSTM (64, return_seq)` → `Bidirectional LSTM (32)` → `Dense (32, ReLU)` → `Dropout (0.5)` → `Dense (3, Softmax)` | Vocab: 10k, MaxLen: 200, Embedding: 64d, Optimizer: Adam | 749,443 | Captures long-range bidirectional token dependencies |
| **GRU** | `Embedding (128d)` → `GRU (64 units)` → `Dropout (0.5)` → `Dense (3, Softmax)` | Vocab: 10k, MaxLen: 200, Embedding: 128d, Optimizer: Adam | 1,317,443 | Computationally efficient gating mechanism with strong convergence |
| **1D-CNN** | `Embedding (128d)` → `Conv1D (64 filters, k=5)` → `MaxPooling1D (2)` → `GlobalMaxPooling1D` → `Dropout (0.5)` → `Dense (3, Softmax)` | Vocab: 10k, MaxLen: 200, Kernel: 5, Optimizer: Adam | 1,321,219 | Rapid spatial n-gram feature extraction & ultra-fast inference |

---

## 📈 Standardized Test Benchmark Results

All models were evaluated under fair, identical conditions on the unseen test split (`X_test_pad.npy`, 1,050 samples) in [`comparison/model_comparison.ipynb`](file:///c:/Users/yasan/Desktop/DL%20Assignment/customer-feedback-classification/comparison/model_comparison.ipynb):

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Multi-Class ROC-AUC (OvR) | Parameters | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1D-CNN** | **76.86%** | **0.7208** | **0.7586** | **0.7342** | **0.7722** | **0.9083** | 1,321,219 | **0.17 ms / sample** |
| **GRU** | **74.38%** | 0.6992 | **0.7478** | 0.7134 | **0.7497** | **0.8924** | 1,317,443 | 0.61 ms / sample |
| **SimpleRNN** | 72.95% | 0.6781 | 0.7132 | 0.6887 | 0.7278 | 0.8719 | 1,321,411 | 0.93 ms / sample |
| **LSTM** | 70.67% | 0.6910 | 0.6805 | 0.6774 | 0.7186 | 0.8526 | **749,443** | 1.25 ms / sample |

---

## 🚀 Execution & Usage Guide

### Step 1: Data Exploration & Preprocessing
To re-run data cleaning, class mapping, sequence tokenization, and data splitting:
1. Open and execute [`notebooks/01_data_exploration.ipynb`](file:///c:/Users/yasan/Desktop/DL%20Assignment/customer-feedback-classification/notebooks/01_data_exploration.ipynb).
2. The notebook loads `data/raw/hotel_reviews.csv`, carries out EDA, handles class balancing, and saves the preprocessed arrays to `data/processed/`.

### Step 2: Training Individual Deep Learning Models

#### Training SimpleRNN
- Run standalone script:
  ```bash
  python models/RNN/train_rnn.py
  ```
- Or open and run the notebook: [`models/RNN/rnn_training.ipynb`](file:///c:/Users/yasan/Desktop/DL%20Assignment/customer-feedback-classification/models/RNN/rnn_training.ipynb).

#### Training LSTM
- Open and execute the training notebook: [`models/LSTM/lstm_training.ipynb`](file:///c:/Users/yasan/Desktop/DL%20Assignment/customer-feedback-classification/models/LSTM/lstm_training.ipynb).
- View metric plots & confusion matrices in [`models/LSTM/lstm-metrics.ipynb`](file:///c:/Users/yasan/Desktop/DL%20Assignment/customer-feedback-classification/models/LSTM/lstm-metrics.ipynb).
- Run interactive CLI prediction on custom reviews:
  ```bash
  cd models/LSTM
  python lstm_model.py
  ```

#### Training GRU
- Open and execute the notebook: [`models/GRU/gru_training.ipynb`](file:///c:/Users/yasan/Desktop/DL%20Assignment/customer-feedback-classification/models/GRU/gru_training.ipynb).

#### Training 1D-CNN
- Open and execute the notebook: [`models/CNN/cnn_training.ipynb`](file:///c:/Users/yasan/Desktop/DL%20Assignment/customer-feedback-classification/models/CNN/cnn_training.ipynb).

### Step 3: Master Model Comparison & Benchmarking
To run the standardized evaluation and view performance comparison tables, confusion matrices, and ROC-AUC curves across all four models:
- Open and execute [`comparison/model_comparison.ipynb`](file:///c:/Users/yasan/Desktop/DL%20Assignment/customer-feedback-classification/comparison/model_comparison.ipynb).
