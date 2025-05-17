import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Page config
st.set_page_config(page_title="Product Defect Detection", layout="wide")

# Model file path (assumed to be in the same folder)
model_path = os.path.join(os.path.dirname(__file__), "model.h5")

@st.cache_resource()
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

model = load_model(model_path)

# Labels hardcoded, no external file needed
labels = {0: "Normal", 1: "Defective"}

def preprocess_image(image: Image.Image):
    # Resize image to model expected size (adjust if needed)
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0  # normalize
    if img_array.shape[-1] == 4:  # If RGBA, convert to RGB
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0)  # batch dim
    return img_array

# Title and description
st.title("📦 Product Defect Detection")
st.write("Upload an image of the product to check if it is Normal or Defective.")

# Upload image file
uploaded_file = st.file_uploader("Upload product image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess and predict
    input_data = preprocess_image(image)
    prediction = model.predict(input_data)[0]  # assuming model outputs probabilities for 2 classes
    
    # Get predicted class and confidence
    pred_class = np.argmax(prediction)
    confidence = prediction[pred_class] * 100
    
    # Show results
    st.write(f"**Prediction:** {labels[pred_class]}")
    st.write(f"**Confidence:** {confidence:.2f}%")
else:
    st.info("Please upload an image file to get prediction.")
