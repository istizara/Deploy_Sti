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


# --- SHARED ON ALL PAGES ---
st.logo("Logo.png")

# --- PAGE SETUP ---
Homepage = st.Page(
    "Homepage.py",
    title="Homepage",
    icon=":material/account_circle:",
    default=True,
)

# --- NAVIGATION SETUP [WITH SECTIONS]---
pg = st.navigation(
    {
        "Info": [Homepage]
    }
)


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
        # --- Preprocessing sesuai input model --- 
        input_shape = classifier.input_shape[1:3] # contoh: (224, 224) 
        img_resized = img.resize(input_shape) 
        img_array = image.img_to_array(img_resized) 
        img_array = np.expand_dims(img_array, axis=0) 
        img_array = img_array.astype("float32") / 255.0 
        try: 
            # --- Tahap 1: Validasi apakah gambar mirip daun jagung --- 
            detection_result = yolo_model(img) 
            labels_detected = [box.cls for box in detection_result[0].boxes] 
            
            if len(labels_detected) == 0: 
                st.warning("⚠️ Gambar yang diunggah tidak terdeteksi sebagai daun jagung. Silakan unggah gambar daun jagung yang valid.") 
            else: 
                # --- Tahap 2: Klasifikasi penyakit daun --- 
                prediction = classifier.predict(img_array) 
                class_index = int(np.argmax(prediction)) 
                confidence = float(np.max(prediction)) 
                labels = ["Blight", "Common Rust", "Grey Spot Leaf", "Healthy"] 
                predicted_label = labels[class_index] 
                st.markdown(f"### 🌿 Hasil Prediksi: **{predicted_label}**") 
                st.markdown(f"**Probabilitas:** {confidence:.2%}") 
                
                advice = { "Blight": "Terdeteksi hawar daun 🌿. Segera isolasi tanaman yang terinfeksi.", 
                          "Common Rust": "Terdeteksi karat daun 🌾. Lakukan penyemprotan fungisida berbasis tembaga.", 
                          "Grey Spot Leaf": "Muncul bercak abu-abu 🍂. Pastikan kelembapan tidak terlalu tinggi.", 
                          "Healthy": "Daun dalam kondisi sehat 🌱. Pertahankan perawatan tanaman." } 
                st.info(advice[predicted_label]) 
        except Exception as e: 
            st.error(f"Terjadi kesalahan saat memproses gambar: {e}") 
            st.info(f"Pastikan ukuran input gambar sesuai dengan model (input shape: {classifier.input_shape})")
