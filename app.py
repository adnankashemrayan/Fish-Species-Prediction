import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = 300

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fish_industry_model.keras")

model = load_model()

class_names = ['Baim', 'Bata', 'Batasio(tenra)', 'Chitul', 'Croaker(Poya)', 'Hilsha', 'Kajoli', 'Meni', 'Pabda', 'Poli', 'Puti', 'Rita', 'Rui', 'Rupchada', 'Silver Carp', 'Telapiya', 'carp', 'k', 'kaikka', 'koral', 'shrimp']  # ← তোমারটা বসাও

st.title("🐟 Fish Species Classifier")
st.write("Upload a fish image to predict the species.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_resized = image.resize((IMG_SIZE, IMG_SIZE))
    
    img_array = np.array(image_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)

    prediction = model.predict(img_array)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.success(f"Prediction: {class_names[class_id]}")
    st.info(f"Confidence: {confidence:.2f}")
