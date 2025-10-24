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
# Menu dan Navigasi
# ==========================

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Panggil fungsi di awal Streamlit
add_bg_from_local("background.jpg")

st.set_page_config(
    page_title="Corn Disease Detection Dashboard for Smart Farming",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Corn Disease Detection Dashboard for Smart Farming")
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

# ==========================
# UI
# ==========================

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
        img_resized = img.resize((128, 128))  # sesuaikan ukuran dengan model kamu
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediksi
        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        st.write("### Hasil Prediksi:", class_index)
        st.write("Probabilitas:", np.max(prediction))
    
    except ValueError as e:
        st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
        st.info(f"Pastikan ukuran input gambar sesuai dengan model (input shape: {classifier.input_shape})")
    ```

