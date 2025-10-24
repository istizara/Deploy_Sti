import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
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

set_background('./background.jpg')

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
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diupload", use_column_width=True)

    st.write("Sedang memproses...")

    if menu == "Deteksi Objek (YOLO)":
        # Deteksi objek
        results = yolo_model(img)
        result_img = results[0].plot()  # hasil deteksi (gambar dengan box)
        st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

    elif menu == "Klasifikasi Gambar":
       # Preprocessing
        img = img.resize((224, 224))  # ubah sesuai ukuran input model Anda
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # normalisasi (jika model dilatih dengan skala ini)


        # Prediksi
        # Preprocessing
        img_resized = img.resize((224, 224))  # sesuaikan ukuran dengan model kamu
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediksi
        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        st.write("### Hasil Prediksi:", class_index)
        st.write("Probabilitas:", np.max(prediction))
