import streamlit as st
import tensorflow as tf
import numpy as np
from ultralytics import YOLO
from PIL import Image

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Isti_Laporan 4.pt")  # deteksi daun
    classifier = tf.keras.models.load_model("model/Isti_Laporan_2.h5")  # klasifikasi penyakit
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# TAMPILAN DASHBOARD
# ==========================
st.title("🌽 Klasifikasi Penyakit Daun Jagung")
st.write("Unggah gambar daun jagung, lalu jalankan model untuk mendeteksi dan mengklasifikasi penyakitnya.")

# 3 kolom: upload - tombol - hasil
col1, col2, col3 = st.columns([2, 1, 2])

uploaded_file = col1.file_uploader("📤 Unggah gambar daun jagung", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    col1.image(img, caption="Gambar yang diunggah", use_container_width=True)


    run_button = col2.button("🧠 Jalankan Prediksi")

    if run_button:
        with st.spinner("Model sedang memproses gambar... ⏳"):
            try:
                # === Tahap 1: Deteksi daun ===
                detection_result = yolo_model(img)
                labels_detected = [box.cls for box in detection_result[0].boxes]

                if len(labels_detected) == 0:
                    st.warning("⚠️ Tidak ada daun jagung terdeteksi. Silakan unggah gambar daun yang jelas.")
                else:
                    # === Tahap 2: Klasifikasi penyakit ===
                    input_shape = classifier.input_shape[1:3]  # contoh (224, 224)
                    img_resized = img.resize(input_shape)
                    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

                    prediction = classifier.predict(img_array)
                    class_index = int(np.argmax(prediction))
                    confidence = float(np.max(prediction))

                    labels = ["Blight", "Common Rust", "Grey Spot Leaf", "Healthy"]
                    predicted_label = labels[class_index]

                    col3.success(f"🌿 Hasil Prediksi: **{predicted_label}**")
                    col3.write(f"📈 Probabilitas: {confidence:.2%}")

                    advice = {
                        "Blight": "🌿 Terdeteksi hawar daun. Isolasi tanaman yang terinfeksi.",
                        "Common Rust": "🌾 Terdeteksi karat daun. Semprot fungisida berbasis tembaga.",
                        "Grey Spot Leaf": "🍂 Terdeteksi bercak abu-abu. Kurangi kelembapan berlebih.",
                        "Healthy": "🌱 Daun sehat! Pertahankan perawatan tanaman dengan baik."
                    }

                    col3.info(advice[predicted_label])

            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
