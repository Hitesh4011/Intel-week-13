import os
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

model_path = os.path.join(os.path.dirname(__file__), "model.h5")

st.set_page_config(page_title="Product Anomaly Detection", layout="wide")

@st.cache_resource()
def load_model():
    return tf.keras.models.load_model(model_path)

model = load_model()

labels = {0: "Normal", 1: "Defective"}

def preprocess_image(image: Image.Image, target_size=(224, 224)):
    image = ImageOps.fit(image, target_size, Image.ANTIALIAS)
    image = np.asarray(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

st.title("Product Anomaly Detection System")

uploaded_file = st.file_uploader("Upload an image of the product:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_arr = preprocess_image(image)
    predictions = model.predict(input_arr)[0]

    st.write("### Prediction probabilities:")
    for idx, prob in enumerate(predictions):
        st.write(f"- {labels.get(idx, f'Class {idx}')}: **{prob*100:.2f}%**")

    predicted_index = np.argmax(predictions)
    predicted_label = labels.get(predicted_index, "Unknown")
    confidence = predictions[predicted_index]

    st.write(f"## Final Prediction: {predicted_label} ({confidence*100:.2f}%)")
else:
    st.info("Please upload an image to get prediction.")
