import os
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Get absolute path of the folder where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths for model and labels file
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")

def load_labels(path=LABELS_PATH):
    labels = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                idx, label = parts[0], " ".join(parts[1:])
                labels[int(idx)] = label
    return labels

@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        st.error(f"Model file not found at path: {path}")
        return None
    return tf.keras.models.load_model(path)

def preprocess_image(image: Image.Image, target_size=(224, 224)):
    image = ImageOps.fit(image, target_size, Image.ANTIALIAS)
    image = np.asarray(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

st.title("Product Anomaly Detection System")

st.write("Upload an image of the product to classify it as **Normal** or **Defective**.")

uploaded_file = st.file_uploader("Upload an image of the product:", type=["jpg", "jpeg", "png"])

model = load_model()
labels = load_labels()

if model is not None and uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_arr = preprocess_image(image)
    predictions = model.predict(input_arr)[0]

    st.write("### Prediction probabilities:")
    for idx, prob in enumerate(predictions):
        label = labels.get(idx, f"Class {idx}")
        st.write(f"- {label}: **{prob*100:.2f}%**")

    predicted_index = np.argmax(predictions)
    predicted_label = labels.get(predicted_index, "Unknown")
    confidence = predictions[predicted_index]

    st.write(f"## Final Prediction: {predicted_label} ({confidence*100:.2f}%)")
elif uploaded_file is None:
    st.info("Please upload an image to get prediction.")
