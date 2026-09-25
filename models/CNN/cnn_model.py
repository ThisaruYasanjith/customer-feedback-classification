from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Conv1D,
    MaxPooling1D,
    GlobalMaxPooling1D,
    Dropout,
    Dense,
)


def build_cnn_model(
    vocab_size=10000,
    sequence_length=200,
    embedding_dim=128,
    num_filters=64,
    kernel_size=5,
    dropout_rate=0.5,
    num_classes=3,
):
    """
    Build the 1D CNN model for hotel review classification.

    Architecture:
    Embedding -> Conv1D -> MaxPooling1D ->
    GlobalMaxPooling1D -> Dropout -> Dense (Softmax)
    """

    model = Sequential([
        # Converts each word index into a dense vector representation
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
        ),

        # Conv1D extracts important local patterns from the review text
        Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            activation="relu",
        ),

        # Reduces the sequence size while keeping important features
        MaxPooling1D(pool_size=2),

        # Keeps the strongest feature detected by each CNN filter
        GlobalMaxPooling1D(),

        # Dropout helps reduce overfitting during training
        Dropout(dropout_rate),

        # Produces probabilities for the 3 review classes
        # 0 = Poor, 1 = Average, 2 = Good
        Dense(
            num_classes,
            activation="softmax",
        ),
    ])

    # Configure the model for multi-class classification
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model