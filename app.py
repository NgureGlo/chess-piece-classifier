import json
from pathlib import Path

import streamlit as st
import tensorflow as tf

folder = Path(__file__).parent

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(folder / "chess_model.keras")

with open(folder / "labels.json") as file:
    labels = json.load(file)

st.title("Chess Piece Classifier")

photo = st.file_uploader("Upload a photo of one chess piece", type=["jpg", "jpeg", "png"])

if photo is not None:
    st.image(photo, caption="Uploaded photo", width=300)

    image = tf.io.decode_image(photo.getvalue(), channels=3, expand_animations=False)
    image = tf.image.resize_with_pad(image, 224, 224)
    image = tf.expand_dims(image, 0)

    model = load_model()
    probabilities = model.predict(image, verbose=0)[0]
    best = int(tf.argmax(probabilities))

    st.subheader(labels[best].replace("_", " ").title())
    st.write(f"Model score: {probabilities[best]:.1%}")