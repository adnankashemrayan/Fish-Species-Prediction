# app.py
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# =========================
# Page config
# =========================
st.set_page_config(page_title="Fish Species Classifier", page_icon="🐟", layout="wide")

st.title("🐟 Fish Species Classifier")
st.write("Upload an image of a fish and the AI model will predict its species.")

# =========================
# Load model from HuggingFace link
# =========================
MODEL_URL = "https://huggingface.co/spaces/riad2021/fish-classifier/resolve/main/fish_industry_model.h5"

@st.cache_resource
def load_fish_model():
    st.info("Loading AI model, please wait...")
    # Download model from URL
    response = requests.get(MODEL_URL)
    with open("fish_industry_model.h5", "wb") as f:
        f.write(response.content)
    model = load_model("fish_industry_model.h5")
    return model

model = load_fish_model()

# =========================
# Upload image
# =========================
uploaded_file = st.file_uploader("Upload Fish Image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image for model
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction, axis=1)[0]

    # Replace these with your actual fish class names
    class_names = [
    "Baim","Bata","Batasio(tenra)","Chitul","Croaker(Poya)",
    "Hilsha","Kajoli","Meni","Pabda","Poli",
    "Puti","Rita","Rui","Rupchada","Silver Carp",
    "Telapiya","carp","k","kaikka","koral","shrimp"
]
    
    st.success(f"Predicted Species: {class_names[class_idx]}")
    st.write(f"Confidence: {prediction[0][class_idx]*100:.2f}%")
