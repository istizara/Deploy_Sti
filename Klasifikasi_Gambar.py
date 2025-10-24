import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from PIL import Image

# ==========================
# Load Model Klasifikasi
# ==========================
@st.cache_resource
def load_classifier():
    model = tf.keras.models.load_model("model/Isti_Laporan_2.h5")  # sesuaikan path model kamu
    return model

classifier = load_classifier()

# ==========================
# Upload dan Prediksi Gambar
# ==========================
st.title("🌿 Klasifikasi Penyakit Daun Jagung")
st.write("Unggah gambar daun jagung untuk mendeteksi apakah daun tersebut sehat atau terkena penyakit.")

uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Gambar yang diunggah", use_container_width=True)
    st.write("⏳ Sedang memproses...")

    # --- Preprocessing sesuai input model ---
    input_shape = classifier.input_shape[1:3]  # contoh: (224, 224)
    img_resized = img.resize(input_shape)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype("float32") / 255.0

    try:
        # --- Prediksi ---
        prediction = classifier.predict(img_array)
        class_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        # --- Label kelas sesuai urutan pelatihan model ---
        labels = ["Blight", "Common Rust", "Grey Spot Leaf", "Healthy"]
        predicted_label = labels[class_index]

        # --- Tampilkan hasil ---
        st.markdown(f"### 🌾 Hasil Prediksi: **{predicted_label}**")
        st.markdown(f"**Probabilitas:** {confidence:.2%}")

        # --- Rekomendasi sederhana ---
        advice = {
            "Blight": "🌿 Terdeteksi hawar daun. Segera isolasi tanaman yang terinfeksi dan hindari penyiraman berlebih.",
            "Common Rust": "🌾 Terdeteksi karat daun. Lakukan penyemprotan fungisida berbasis tembaga.",
            "Grey Spot Leaf": "🍂 Ditemukan bercak abu-abu. Pastikan kelembapan lahan tidak terlalu tinggi.",
            "Healthy": "🌱 Daun dalam kondisi sehat! Pertahankan perawatan tanaman dengan baik."
        }

        st.info(advice[predicted_label])

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memproses gambar: {e}")
        st.info(f"Pastikan ukuran input gambar sesuai model (input shape: {classifier.input_shape})")
