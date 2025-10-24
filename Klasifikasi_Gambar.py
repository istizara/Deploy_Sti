import streamlit as st
import tensorflow as tf
import numpy as np
from ultralytics import YOLO
from PIL import Image

@st.cache_resource
def load_models():
    # Muat model YOLO untuk deteksi daun jagung
    yolo_model = YOLO("model/Isti_Laporan_4.pt")

    # Muat model klasifikasi penyakit daun
    classifier = tf.keras.models.load_model("model/Isti_Laporan_2.h5")
    return yolo_model, classifier

# Panggil model
yolo_model, classifier = load_models()

# Antarmuka Streamlit
st.title("🌽 Klasifikasi Penyakit Daun Jagung")

uploaded_file = st.file_uploader("Unggah gambar daun jagung", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang diunggah", use_column_width=True)

    if st.button("Jalankan Prediksi"):
        with st.spinner("Menganalisis gambar..."):
            # Deteksi objek daun menggunakan YOLO
            detection_result = yolo_model(img)
            labels_detected = [box.cls for box in detection_result[0].boxes]

            if len(labels_detected) == 0:
                st.warning("⚠️ Tidak ada daun jagung terdeteksi. Silakan unggah gambar daun yang jelas.")
            else:
                # --- Klasifikasi penyakit daun ---
                img_resized = img.resize((224, 224))
                img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

                prediction = classifier.predict(img_array)
                class_index = int(np.argmax(prediction))
                confidence = float(np.max(prediction))

                labels = ["Blight", "Common Rust", "Grey Spot Leaf", "Healthy"]
                predicted_label = labels[class_index]

                st.success(f"🌿 Hasil Prediksi: **{predicted_label}**")
                st.write(f"Probabilitas: {confidence:.2%}")

                advice = {
                    "Blight": "🌿 Terdeteksi hawar daun. Segera isolasi tanaman yang terinfeksi.",
                    "Common Rust": "🌾 Terdeteksi karat daun. Lakukan penyemprotan fungisida berbasis tembaga.",
                    "Grey Spot Leaf": "🍂 Muncul bercak abu-abu. Pastikan kelembapan tidak terlalu tinggi.",
                    "Healthy": "🌱 Daun dalam kondisi sehat. Pertahankan perawatan tanaman."
                }
                st.info(advice[predicted_label])
