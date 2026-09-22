import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Bidirectional,
    SimpleRNN,
    Dropout,
    Dense
)


# ==================================================
# Model configuration
# ==================================================

VOCAB_SIZE = 10000
EMBEDDING_DIM = 128
RNN_UNITS = 64
NUM_CLASSES = 3
MAX_LENGTH = 200


# ==================================================
# Build RNN model
# ==================================================

def build_rnn_model():
    model = Sequential([
        
        Input(shape=(MAX_LENGTH,)),

        # Convert token IDs into dense word representations
        Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=EMBEDDING_DIM
        ),

        # First recurrent layer
        Bidirectional(
            SimpleRNN(
                RNN_UNITS,
                return_sequences=True
            )
        ),

        Dropout(0.30),

        # Second recurrent layer
        SimpleRNN(RNN_UNITS),

        Dropout(0.50),

        # Fully connected classification layer
        Dense(
            64,
            activation="relu"
        ),

        Dropout(0.30),

        # Three output classes:
        # 0 = Poor
        # 1 = Average
        # 2 = Good
        Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    ])

    return model