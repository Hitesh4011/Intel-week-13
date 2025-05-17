import numpy as np

from PIL import Image
import os
import streamlit as st

try:
    import tensorflow as tf
    print("✅ TensorFlow imported successfully.")
except Exception as e:
    print("❌ TensorFlow crashed:", e)

st.set_page_config(page_title="Product Defect Detection", layout="wide")

st.title("📦 Product Defect Detection")
st.write("Upload an image of the product to check if it is Normal or Defective.")

