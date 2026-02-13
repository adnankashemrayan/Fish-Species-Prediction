import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import requests
import os

st.set_page_config(page_title="Fish Species Classifier", page_icon="🐟")
st.title("🐟 Fish Species Classifier")
st.write("Upload a fish image — accurate prediction guaranteed!")

# Model download link
MODEL_URL = "https://huggingface.co/spaces/riad2021/fish-classifier/resolve/main/fish_classifier_final.keras"
MODEL_PATH = "fish_classifier_final.keras"

# Download and load model
@st.cache_resource
def load_model_from_url():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model… Please wait ⏳"):
            r = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model_from_url()

# Class names (must match training)
class_names = [
    "Baim","Bata","Batasio(tenra)","Chitul","Croaker(Poya)",
    "Hilsha","Kajoli","Meni","Pabda","Poli",
    "Puti","Rita","Rui","Rupchada","Silver Carp",
    "Telapiya","carp","koi","kaikka","koral","shrimp"
]

uploaded_file = st.file_uploader("Upload Fish Image (jpg/jpeg/png)", type=["jpg","jpeg","png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    # Use the same preprocessing as training
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)

    # Predict
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][class_idx] * 100

    # Display results
    st.success(f"**Prediction:** {class_names[class_idx]}")
    st.write(f"**Confidence:** {confidence:.2f}%")
