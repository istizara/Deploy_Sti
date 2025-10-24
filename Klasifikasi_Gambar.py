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

# ==========================
# Layout dua kolom
# ==========================
col1, col2 = st.columns([1.2, 1])

with col1:
    uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="🖼️ Uploaded Image", use_container_width=True)

        # Tombol untuk klasifikasi
        run_classification = st.button("🧠 Run Classification", type="primary")

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

                # Simpan hasil untuk ditampilkan di kolom kanan
                st.session_state["hasil_prediksi"] = {
                    "label": predicted_label,
                    "confidence": confidence,
                    "model": "Isti_Laporan_2.h5"
                }

with col2:
    st.markdown("### 📊 Hasil Klasifikasi")

    if "hasil_prediksi" in st.session_state:
        hasil = st.session_state["hasil_prediksi"]

        st.markdown(f"**📷 Prediction:** <span style='color:#00FF00;font-weight:bold'>{hasil['label']}</span>", unsafe_allow_html=True)
        st.markdown(f"**📈 Confidence:** {hasil['confidence']*100:.2f}%")
        st.markdown(f"**💾 Model Used:** `{hasil['model']}`")

        # --- Rekomendasi berdasarkan hasil ---
        advice = {
            "Blight": "🌿 Terdeteksi hawar daun. Isolasi tanaman yang terinfeksi dan hindari penyiraman berlebih.",
            "Common Rust": "🌾 Terdeteksi karat daun. Lakukan penyemprotan fungisida berbasis tembaga.",
            "Grey Spot Leaf": "🍂 Ditemukan bercak abu-abu. Pastikan kelembapan lahan tidak terlalu tinggi.",
            "Healthy": "🌱 Daun dalam kondisi sehat! Pertahankan perawatan tanaman dengan baik."
        }

        st.info(advice[hasil["label"]])
    else:
        st.write("⚙️ Hasil prediksi akan muncul di sini setelah kamu menekan tombol **Run Classification**.")
