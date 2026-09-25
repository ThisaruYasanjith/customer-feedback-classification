# Customer Feedback Classification using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Dataset](https://img.shields.io/badge/Kaggle-Hotel%20Reviews-blue)](https://www.kaggle.com/datasets/juhibhojani/hotel-reviews)

## 📌 Project Overview
This repository contains the official implementation for **SE4050 – Deep Learning (2026)** Assignment 1. The objective of this project is to solve a real-world Natural Language Processing (NLP) problem—classifying customer hotel reviews into sentiment ratings—using multiple deep learning model architectures.

We implement, train, evaluate, and critically compare **four distinct deep learning architectures**:
1. **Simple Recurrent Neural Network (SimpleRNN)** (Bidirectional + Stacked SimpleRNN)
2. **Long Short-Term Memory (LSTM)**
3. **Gated Recurrent Unit (GRU)**
4. **1D Convolutional Neural Network (1D-CNN)**

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
├── README.md                      # Complete project documentation and guide
├── requirements.txt               # Python package dependencies
├── .gitignore                     # Git ignore rules for cached models and venv
│
├── data/                          # Dataset directory
│   ├── raw/                       # Original raw Kaggle CSV dataset (hotel_reviews.csv)
│   └── processed/                 # Tokenized & padded numpy arrays & saved preprocessing objects
│       ├── X_train_pad.npy        # Training input sequences (4895, 200)
│       ├── X_val_pad.npy          # Validation input sequences (1049, 200)
│       ├── X_test_pad.npy         # Test input sequences (1050, 200)
│       ├── y_train.npy            # Training labels
│       ├── y_val.npy              # Validation labels
│       ├── y_test.npy             # Test labels
│       ├── tokenizer.pkl          # Saved fitted Keras Tokenizer
│       └── class_weights.pkl      # Balanced class weight dictionary
│
├── common/                        # Reusable Python modules for data loading, preprocessing, & evaluation
│   ├── data_loader.py             # Data loading routines
│   ├── preprocessing.py          # Text cleaning and sequence padding functions
│   ├── tokenizer.py              # Tokenizer fitting and transformation logic
│   └── evaluation.py             # Standardized model evaluation & metric calculation functions
│
├── notebooks/                     # Exploratory Data Analysis & data pipeline
│   └── 01_data_exploration.ipynb  # EDA, rating distribution, text cleaning, and data serialization
│
├── models/                        # Individual deep learning model implementations
│   ├── SimpleRNN/                 # Simple Recurrent Neural Network
│   │   ├── rnn_model.py           # SimpleRNN model definition
│   │   ├── train_rnn.py           # Training script for SimpleRNN
│   │   ├── rnn_training.ipynb     # Interactive RNN notebook
│   │   └── results/               # Saved RNN metrics and history
│   ├── LSTM/                      # Long Short-Term Memory Network
│   │   ├── lstm_model.py          # LSTM model definition
│   │   ├── lstm_training.ipynb    # Interactive LSTM notebook
│   │   └── lstm-metrics.ipynb     # LSTM evaluation & metric plots
│   ├── GRU/                       # Gated Recurrent Unit Network
│   │   ├── gru_model.py           # GRU model definition
│   │   └── gru_training.ipynb     # Interactive GRU notebook
│   └── CNN/                       # 1D Convolutional Neural Network
│       ├── cnn_model.py           # 1D-CNN model definition
│       └── cnn_training.ipynb     # Interactive CNN notebook
│
├── comparison/                    # Master cross-model comparison
│   └── model_comparison.ipynb     # Comparative evaluation across all models (Acc, F1, ROC-AUC, Latency)
│
└── models_saved/                  # Saved serialized model artifacts (.keras format)
    ├── rnn_model.keras            # Trained SimpleRNN model weight checkpoint
    ├── lstm_model.keras           # Trained LSTM model weight checkpoint
    ├── gru_model.keras            # Trained GRU model weight checkpoint
    └── cnn_model.keras            # Trained 1D-CNN model weight checkpoint
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

## 🚀 Execution & Usage Guide

### Step 1: Data Exploration & Preprocessing
To re-run data cleaning, class mapping, sequence tokenization, and data splitting:
1. Open and execute [`notebooks/01_data_exploration.ipynb`](file:///c:/Users/yasan/Desktop/DL%20Assignment/customer-feedback-classification/notebooks/01_data_exploration.ipynb).
2. The notebook will automatically download the dataset via `kagglehub` (or load from `data/raw/hotel_reviews.csv`) and output preprocessed numpy arrays to `data/processed/`.

### Step 2: Training Individual Deep Learning Models

#### Training SimpleRNN
```bash
python models/RNN/train_rnn.py
```
Or open and run the Jupyter notebook: [`models/RNN/rnn_training.ipynb`](file:///c:/Users/yasan/Desktop/DL%20Assignment/customer-feedback-classification/models/RNN/rnn_training.ipynb).

#### Training LSTM
Open and execute the notebook: [`models/LSTM/lstm_training.ipynb`](file:///c:/Users/yasan/Desktop/DL%20Assignment/customer-feedback-classification/models/LSTM/lstm_training.ipynb).

#### Training GRU
Open and execute the notebook: [`models/GRU/gru_training.ipynb`](file:///c:/Users/yasan/Desktop/DL%20Assignment/customer-feedback-classification/models/GRU/gru_training.ipynb).

#### Training 1D-CNN
Open and execute the notebook: [`models/CNN/cnn_training.ipynb`](file:///c:/Users/yasan/Desktop/DL%20Assignment/customer-feedback-classification/models/CNN/cnn_training.ipynb).

### Step 3: Model Comparison & Benchmarking
To run the standardized evaluation and view performance comparison tables, confusion matrices, and ROC-AUC curves across all four models:
Open and execute [`comparison/model_comparison.ipynb`](file:///c:/Users/yasan/Desktop/DL%20Assignment/customer-feedback-classification/comparison/model_comparison.ipynb).

---

## 🎯 Evaluated Model Architectures

| Architecture | Key Layers & Features | Hyperparameters | Target Strengths |
| :--- | :--- | :--- | :--- |
| **SimpleRNN** | Embedding (128d) -> Bidirectional SimpleRNN (64 units) -> SimpleRNN (64 units) -> Dense (64, ReLU) -> Softmax (3) | Vocab: 10k, MaxLen: 200, Dropout: 0.3-0.5, Optimizer: Adam | Lightweight sequential baseline model |
| **LSTM** | Embedding (128d) -> LSTM (64 units) -> Dropout (0.5) -> Dense (Softmax) | Vocab: 10k, MaxLen: 200, Dropout: 0.5, Optimizer: Adam | Effective at capturing long-range token dependencies |
| **GRU** | Embedding (128d) -> GRU (64 units) -> Dropout (0.5) -> Dense (Softmax) | Vocab: 10k, MaxLen: 200, Dropout: 0.5, Optimizer: Adam | Computationally efficient gating mechanism |
| **1D-CNN** | Embedding (128d) -> Conv1D (64 filters, kernel=5) -> MaxPooling1D (2) -> GlobalMaxPooling1D -> Dropout (0.5) -> Dense (Softmax) | Vocab: 10k, MaxLen: 200, Kernel: 5, Optimizer: Adam | Rapid spatial n-gram feature extraction & fast inference |

---

## 📈 Evaluation Metrics & Standards

All models are evaluated on the unseen test dataset (`data/processed/X_test_pad.npy`) using the following metrics:
- **Classification Performance**: Accuracy, Macro Precision, Macro Recall, Macro F1-Score, Weighted F1-Score, Multi-Class ROC-AUC (One-vs-Rest).
- **Diagnostic Visualizations**: Confusion Matrix heatmaps, Per-Class Precision-Recall curves, Training/Validation Loss & Accuracy learning curves.
- **Efficiency Measures**: Total trainable parameter count, training time per epoch, inference latency per batch/sample.

---

## 👥 Contributors & Acknowledgments
- **Course**: SE4050 – Deep Learning (2026), BSc (Hons) in Information Technology
- **Institution**: Sri Lanka Institute of Information Technology (SLIIT)
- **Group Contributions**: Details regarding individual team member roles and commit history are tracked via GitHub commit logs and documented in `Members.txt`.
