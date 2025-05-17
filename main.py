import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load labels from file with format: "0 Normal"
def load_labels(path="labels.txt"):
    labels = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                idx, label = parts[0], " ".join(parts[1:])
                labels[int(idx)] = label
    return labels

# Load the Keras model
@st.cache(allow_output_mutation=True)
def load_model(path="model.h5"):
    return tf.keras.models.load_model(path)

# Preprocess the image to the required input size (adjust size if needed)
def preprocess_image(image: Image.Image, target_size=(224, 224)):
    image = ImageOps.fit(image, target_size, Image.ANTIALIAS)
    image = np.asarray(image)
    image = image / 255.0  # Normalize to [0,1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("Product Anomaly Detection System")

st.write("Upload an image of the product to classify it as **Normal** or **Defective**.")

uploaded_file = st.file_uploader("Upload an image of the product:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = load_model()
    labels = load_labels()

    input_arr = preprocess_image(image)
    predictions = model.predict(input_arr)[0]  # Get the prediction vector

    # Show all class probabilities nicely
    st.write("### Prediction probabilities:")
    for idx, prob in enumerate(predictions):
        label = labels.get(idx, f"Class {idx}")
        st.write(f"- {label}: **{prob*100:.2f}%**")

    # Show the highest probability prediction
    predicted_index = np.argmax(predictions)
    predicted_label = labels.get(predicted_index, "Unknown")
    confidence = predictions[predicted_index]

    st.write(f"## Final Prediction: {predicted_label} ({confidence*100:.2f}%)")
