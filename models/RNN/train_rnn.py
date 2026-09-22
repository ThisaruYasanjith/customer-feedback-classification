from pathlib import Path
import pickle
import numpy as np
import tensorflow as tf

from rnn_model import build_rnn_model


# ==================================================
# 1. Project paths
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models_saved"
RESULT_DIR = PROJECT_ROOT / "models" / "RNN" / "results"


# ==================================================
# 2. Create required folders
# ==================================================

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)


# ==================================================
# 3. Load processed datasets
# ==================================================

X_train = np.load(PROCESSED_DIR / "X_train_pad.npy")
X_val = np.load(PROCESSED_DIR / "X_val_pad.npy")
X_test = np.load(PROCESSED_DIR / "X_test_pad.npy")

y_train = np.load(PROCESSED_DIR / "y_train.npy")
y_val = np.load(PROCESSED_DIR / "y_val.npy")
y_test = np.load(PROCESSED_DIR / "y_test.npy")


# ==================================================
# 4. Load tokenizer
# ==================================================

with open(PROCESSED_DIR / "tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


# ==================================================
# 5. Load RNN class weights
# ==================================================

with open(PROCESSED_DIR / "class_weights.pkl", "rb") as f:
    class_weights = pickle.load(f)


# ==================================================
# 6. Display loaded data information
# ==================================================

print("RNN data loaded successfully.")
print()

print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)
print("Test data:", X_test.shape)

print()

print("Training labels:", y_train.shape)
print("Validation labels:", y_val.shape)
print("Test labels:", y_test.shape)

print()

print("RNN class weights:")
for label, weight in class_weights.items():
    print(f"  Label {label}: {weight:.4f}")

print()

print("Tokenizer vocabulary size:", len(tokenizer.word_index))


# ==================================================
# 7. Build RNN model
# ==================================================

rnn_model = build_rnn_model()

print()
print("RNN model created successfully.")
print()

rnn_model.summary()


# ==================================================
# 8. Compile RNN model
# ==================================================

rnn_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print()
print("RNN model compiled successfully.")