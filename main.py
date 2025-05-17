import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image
import os

# Page config
st.set_page_config(page_title="Product Defect Detection", layout="wide")

@st.cache_resource()
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.h5")
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

labels = {0: "Normal", 1: "Defective"}

def preprocess_image(image: Image.Image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    return np.expand_dims(img_array, axis=0)

st.title("📦 Product Defect Detection")
st.write("Upload an image of the product to check if it is Normal or Defective.")

uploaded_file = st.file_uploader("Upload product image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        input_data = preprocess_image(image)
        prediction = model.predict(input_data)[0]
        pred_class = np.argmax(prediction)
        confidence = prediction[pred_class] * 100
        st.success(f"**Prediction:** {labels[pred_class]}")
        st.info(f"**Confidence:** {confidence:.2f}%")
    except Exception as e:
        st.error(f"Error processing image: {e}")
