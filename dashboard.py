import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from Models import classifier
import base64
import os
import numpy as np
from PIL import Image
import cv2

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Isti_Laporan 4.pt")  # Model deteksi objek
    classifier = tf.keras.models.load_model("model/Isti_Laporan_2.h5")  # Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# UI
# ==========================

color_samples = [
    (255, 255, 0),   # Yellow – bright and contrasts well with black
    (0, 255, 255),   # Cyan – highly visible, clean contrast
    (255, 200, 100), # Light orange – warm and readable
    (180, 255, 180), # Pale green – soft and effective
    (255, 128, 255), # Light magenta – vibrant but not dark
]

st.set_page_config(
    page_title="Dashboard Prediksi Gambar",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Prediksi Penyakit Pada Daun Jagung")
st.write("...")

# Page config
st.set_page_config(page_title="Object Detection App", layout="wide")

bg_image_path = "background.jpg"  # Adjust path as needed

st.markdown("""
<style>
body {
  background: "background.jpg";
}
</style>
    """, unsafe_allow_html=True)

menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    if menu == "Deteksi Objek (YOLO)":
        # Deteksi objek
        results = yolo_model(img)
        result_img = results[0].plot()  # hasil deteksi (gambar dengan box)
        st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

    elif menu == "Klasifikasi Gambar":
        img = Image.open(uploaded_file).convert("RGB")
    prediction = None

    col1, col2, col3 = st.columns([1.2, 1, 1.2], gap="large")

    with col1:
        st.image(img, use_container_width=True, caption="📥 Uploaded Image")

    with col2:
        st.markdown("""<br><br>""", unsafe_allow_html=True)

        st.markdown("""
            <style>
            div.stButton > button {
                margin-left: 30%;
            }
            </style>
        """, unsafe_allow_html=True)

        if st.button("🚀 Run Classification"):
            with st.spinner("Classifying..."):
                label, score = classifier(img, model_data["weights_name"], model_data["class_names"])
                prediction = (label, score)

    with col3:
        if prediction:
            label, score = prediction
            st.markdown(f"""
                <br><br>
                <h3>📷 Prediction: <code>{label}</code></h3>
                <h4>🔢 Confidence: <code>{score:.2f}</code></h4>
                <h5>🧠 Model Used: <code>{model_data['weights_name']}</code></h5>
            """, unsafe_allow_html=True)
