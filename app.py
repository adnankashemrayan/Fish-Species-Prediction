import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
from io import BytesIO

MODEL_URL = "https://huggingface.co/spaces/riad2021/fish-classifier/resolve/main/fish_industry_model.keras"

@st.cache_resource
def load_model():
    response = requests.get(MODEL_URL)
    model = tf.keras.models.load_model(BytesIO(response.content), compile=False)
    return model

model = load_model()

class_names = [
    "Bangus","Big Head Carp","Black Spotted Barb","Catfish",
    "Climbing Perch","Fourfinger Threadfin","Freshwater Eel",
    "Glass Perchlet","Goby","Gold Fish","Gourami","Grass Carp",
    "Green Spotted Puffer","Indian Carp","Indo-Pacific Tarpon",
    "Jaguar Gapote","Janitor Fish","Knifefish",
    "Long-Snouted Pipefish","Mosquito Fish","Mudfish","Mullet",
    "Pangasius","Perch","Scat Fish","Silver Barb","Silver Carp",
    "Silver Perch","Snakehead","Tenpounder","Tilapia"
]

st.title("🐟 Fish Species Classifier")

uploaded_file = st.file_uploader("Upload Fish Image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224,224))   # VERY IMPORTANT
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    st.image(image, caption="Uploaded Image")

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")
