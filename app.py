import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
from io import BytesIO

st.title("Fish Industry Classifier 🐟")

# Load the keras model from Hugging Face URL
MODEL_URL = "https://huggingface.co/spaces/riad2021/fish-classifier/resolve/main/fish_industry_model.keras"

@st.cache_resource
def load_model():
    model_path = tf.keras.utils.get_file("fish_industry_model.keras", MODEL_URL)
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# Upload image
img_file = st.file_uploader("Upload a fish image", type=["jpg", "png", "jpeg"])

if img_file:
    img = Image.open(img_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocessing
    img = img.resize((224, 224))  # adjust size based on your model
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    pred_class = np.argmax(prediction, axis=1)

    st.write("Prediction:", pred_class)
