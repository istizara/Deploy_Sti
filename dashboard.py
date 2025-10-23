import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
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
bg_image_encoded = get_base64_image(bg_image_path)

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
