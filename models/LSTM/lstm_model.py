import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LENGTH = 200

review_class_labels = [ "poor", "average", "good" ]

with open("../../data/processed/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = load_model("lstm_review_classifier.keras")
# model.summary()

def preprocess(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding="post", truncating="post")
    return padded

def predict(text):
    X = preprocess([text])
    probabilities = model.predict(X)[0]
    predicted_class_idx = np.argmax(probabilities)
    return review_class_labels[predicted_class_idx], probabilities[predicted_class_idx]

while True:
    # predict until keyboard interrupt
    print(predict(input("Enter your review: ")))