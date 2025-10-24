import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from ultralytics import YOLO
from PIL import Image

# ==========================
# Load Model Klasifikasi
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Isti_Laporan 4.pt")
    classifier = tf.keras.models.load_model("model/Isti_Laporan_2.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()
# ==========================
# Tampilan Utama
# ==========================
st.title("🌿 Klasifikasi Penyakit Daun Jagung")
st.write("Unggah gambar daun jagung untuk mendeteksi apakah daun tersebut sehat atau terkena penyakit.")

# 1️⃣ Upload gambar
uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Gambar yang diunggah", use_container_width=True)

    # 2️⃣ Instruksi dan tombol klasifikasi
    st.markdown("---")
    st.markdown("### 🧠 Langkah Selanjutnya")
    st.write("Klik tombol di bawah ini untuk menjalankan proses klasifikasi menggunakan model yang telah dilatih.")
    run_classification = st.button("🚀 Jalankan Klasifikasi", type="primary")

    if run_classification:
        with st.spinner("Model sedang memproses gambar... ⏳"):
            # --- Preprocessing ---
            input_shape = classifier.input_shape[1:3]  # contoh: (224, 224)
            img_resized = img.resize(input_shape)
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array.astype("float32") / 255.0

            # --- Prediksi ---
            prediction = classifier.predict(img_array)
            class_index = int(np.argmax(prediction))
            confidence = float(np.max(prediction))

            labels = ["Blight", "Common Rust", "Grey Spot Leaf", "Healthy"]
            predicted_label = labels[class_index]

            # Simpan hasil prediksi
            st.session_state["hasil_prediksi"] = {
                "label": predicted_label,
                "confidence": confidence,
                "model": "Isti_Laporan_2.h5"
            }

    # 3️⃣ Hasil klasifikasi
    st.markdown("---")
    st.markdown("### 📊 Hasil Klasifikasi")

    if "hasil_prediksi" in st.session_state:
        hasil = st.session_state["hasil_prediksi"]

        st.markdown(
            f"**📷 Hasil Prediksi:** <span style='color:#00FF00;font-weight:bold'>{hasil['label']}</span>",
            unsafe_allow_html=True
        )
        st.markdown(f"**📈 Tingkat Keyakinan:** {hasil['confidence']*100:.2f}%")
        st.markdown(f"**💾 Model Digunakan:** `{hasil['model']}`")

        # --- Rekomendasi berdasarkan hasil ---
        advice = {
            "Blight": "🌿 Terdeteksi *Blight (Hawar Daun)*. Isolasi tanaman terinfeksi dan hindari penyiraman berlebih.",
            "Common Rust": "🌾 Terdeteksi *Common Rust (Karat Daun)*. Lakukan penyemprotan fungisida berbasis tembaga.",
            "Grey Spot Leaf": "🍂 Terdeteksi *Grey Spot Leaf (Bercak Abu-abu)*. Jaga kelembapan lahan agar tidak terlalu tinggi.",
            "Healthy": "🌱 Daun dalam kondisi sehat! Pertahankan pola perawatan tanaman seperti ini."
        }

        st.success(advice[hasil["label"]])
    else:
        st.info("⚙️ Hasil klasifikasi akan muncul di sini setelah kamu menekan tombol **Jalankan Klasifikasi**.")

else:
    st.info("📸 Silakan unggah gambar daun jagung terlebih dahulu untuk memulai klasifikasi.")
