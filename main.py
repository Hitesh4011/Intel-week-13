import streamlit as st
import numpy as np
import tensorflow as tf  # More specific import
from PIL import Image
import os

# Page config
st.set_page_config(page_title="Product Defect Detection", layout="wide")

@st.cache_resource()
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.h5")
    return tf.keras.models.load_model(model_path)

# Load model at startup
model = load_model()

# Labels
labels = {0: "Normal", 1: "Defective"}

def preprocess_image(image: Image.Image):
    """Preprocess image for model prediction"""
    try:
        # Resize and normalize
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        
        # Handle RGBA images
        if img_array.shape[-1] == 4:
            img_array = img_array[..., :3]
            
        # Add batch dimension
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return None

# App UI
st.title("📦 Product Defect Detection")
st.write("Upload an image of the product to check if it is Normal or Defective.")

uploaded_file = st.file_uploader("Choose product image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Preprocess and predict
        input_data = preprocess_image(image)
        if input_data is not None:
            with st.spinner("Analyzing image..."):
                prediction = model.predict(input_data)[0]
                pred_class = np.argmax(prediction)
                confidence = prediction[pred_class] * 100
                
                # Display results
                st.subheader("Results")
                if pred_class == 1:
                    st.error(f"🔴 Defective (confidence: {confidence:.1f}%)")
                else:
                    st.success(f"🟢 Normal (confidence: {confidence:.1f}%)")
                
                # Optional: show confidence for both classes
                with st.expander("Detailed confidence scores"):
                    st.write(f"Normal: {prediction[0]*100:.1f}%")
                    st.write(f"Defective: {prediction[1]*100:.1f}%")
                    
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
