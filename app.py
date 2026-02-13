import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Fish Species Classifier",
    page_icon="🐟",
    layout="centered"
)

st.title("🐟 Fish Species Prediction System")
st.write("Upload a fish image to classify the species")

MODEL_URL = "https://huggingface.co/spaces/riad2021/fish-classifier/resolve/main/fish_industry_model.keras"

# Cache model
@st.cache_resource
def load_model():
    model_path = tf.keras.utils.get_file("fish_industry_model.keras", MODEL_URL)
    model = tf.keras.models.load_model(model_path)
    return model

with st.spinner("Loading AI Model..."):
    model = load_model()

st.success("Model Loaded Successfully ✅")

# Class names (must match training order)
class_names = [
    "Baim","Bata","Batasio(tenra)","Chitul","Croaker(Poya)",
    "Hilsha","Kajoli","Meni","Pabda","Poli",
    "Puti","Rita","Rui","Rupchada","Silver Carp",
    "Telapiya","carp","k","kaikka","koral","shrimp"
]

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_shape = model.input_shape
    height = input_shape[1]
    width = input_shape[2]
    channels = input_shape[3]

    if channels == 1:
        image = image.convert("L")
    else:
        image = image.convert("RGB")

    image = image.resize((width, height))

    img_array = np.array(image)

    if channels == 1:
        img_array = np.expand_dims(img_array, axis=-1)

    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing Image..."):
        prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = float(np.max(prediction)) * 100
    predicted_name = class_names[predicted_class]

    st.success(f"🐟 Predicted Fish: {predicted_name}")
    st.info(f"📊 Confidence: {confidence:.2f}%")

    # Show Top 3 predictions
    st.subheader("Top 3 Predictions")
    top_indices = prediction[0].argsort()[-3:][::-1]

    for i in top_indices:
        st.write(f"{class_names[i]} : {prediction[0][i]*100:.2f}%")
