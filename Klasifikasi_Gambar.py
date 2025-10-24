import streamlit as st
import tensorflow as tf
import numpy as np
from ultralytics import YOLO
from PIL import Image

@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Isti_Laporan_4.pt")
    classifier = tf.keras.models.load_model("model/Isti_Laporan_2.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

st.title("🌽 Klasifikasi Penyakit Daun Jagung")

# Buat 3 kolom
col1, col2, col3 = st.columns([2, 1, 2])

uploaded_file = col1.file_uploader("Unggah gambar daun jagung", type=["jpg", "png", "jpeg"])

# Jika ada gambar diunggah, tampilkan
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    col1.image(img, caption="Gambar yang diunggah", use_column_width=True)

    # Tombol Run di tengah
    run_button = col2.button("Jalankan Prediksi")

    if run_button:
        with st.spinner("Menganalisis gambar..."):
            detection_result = yolo_model(img)
            labels_detected = [box.cls for box in detection_result[0].boxes]

            # Jika tidak ada daun jagung terdeteksi
            if len(labels_detected) == 0:
                # tampilkan warning di bawah layout kolom
                st.warning("⚠️ Tidak ada daun jagung terdeteksi. Silakan unggah gambar daun yang jelas.")
            else:
                # Prediksi penyakit daun
                img_resized = img.resize((224, 224))
                img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

                prediction = classifier.predict(img_array)
                class_index = int(np.argmax(prediction))
                confidence = float(np.max(prediction))

                labels = ["Blight", "Common Rust", "Grey Spot Leaf", "Healthy"]
                predicted_label = labels[class_index]

                col3.success(f"🌿 Hasil Prediksi: **{predicted_label}**")
                col3.write(f"Probabilitas: {confidence:.2%}")

                advice = {
                    "Blight": "🌿 Terdeteksi hawar daun. Segera isolasi tanaman yang terinfeksi.",
                    "Common Rust": "🌾 Terdeteksi karat daun. Lakukan penyemprotan fungisida berbasis tembaga.",
                    "Grey Spot Leaf": "🍂 Muncul bercak abu-abu. Pastikan kelembapan tidak terlalu tinggi.",
                    "Healthy": "🌱 Daun dalam kondisi sehat. Pertahankan perawatan tanaman."
                }
                col3.info(advice[pred]()
