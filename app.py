import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import requests
import os

st.set_page_config(page_title="Fish Species Classifier", page_icon="🐟")
st.title("🐟 Fish Species Classifier")
st.write("Upload a fish image from your phone and get accurate predictions!")

# HuggingFace direct download link
MODEL_URL = "https://huggingface.co/spaces/riad2021/fish-classifier/resolve/main/fish_classifier.h5"
MODEL_PATH = "fish_classifier.h5"

# Download model if not exists
@st.cache_resource
def load_fish_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model... Please wait ⏳"):
            r = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
    return load_model(MODEL_PATH)

model = load_fish_model()

class_names = [
    "Baim","Bata","Batasio(tenra)","Chitul","Croaker(Poya)",
    "Hilsha","Kajoli","Meni","Pabda","Poli",
    "Puti","Rita","Rui","Rupchada","Silver Carp",
    "Telapiya","carp","k","kaikka","koral","shrimp"
]

uploaded_file = st.file_uploader("Upload Fish Image", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_resized = img.resize((300,300))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)/127.5 - 1  # match training [-1,1]

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][class_idx]*100

    st.success(f"Predicted Species: {class_names[class_idx]}")
    st.write(f"Confidence: {confidence:.2f}%")
