"""GRU architecture for the shared hotel-review classification task."""

import tensorflow as tf


def build_gru_model(
	vocab_size: int = 10_000,
	sequence_length: int = 200,
	embedding_dim: int = 128,
	gru_units: int = 64,
	dropout_rate: float = 0.5,
	num_classes: int = 3,
) -> tf.keras.Model:
	"""Create and compile the GRU classifier used in the model comparison."""
	model = tf.keras.Sequential(
		[
			tf.keras.layers.Input(shape=(sequence_length,)),
			tf.keras.layers.Embedding(
				input_dim=vocab_size,
				output_dim=embedding_dim,
				mask_zero=True,
			),
			tf.keras.layers.GRU(gru_units),
			tf.keras.layers.Dropout(dropout_rate),
			tf.keras.layers.Dense(num_classes, activation="softmax"),
		],
		name="hotel_review_gru",
	)

	model.compile(
		optimizer=tf.keras.optimizers.Adam(),
		loss=tf.keras.losses.SparseCategoricalCrossentropy(),
		metrics=["accuracy"],
	)
	return model
