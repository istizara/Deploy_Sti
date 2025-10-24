import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from PIL import Image

# ==========================
# Load Model Klasifikasi
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Isti_Laporan 4.pt")
    classifier  = tf.keras.models.load_model("model/Isti_Laporan_2.h5")  
    return yolo_model, classifier 

yolo_model, classifier = load_models()

# ==========================
# Tampilan Utama
# ==========================
st.title("🌿 Klasifikasi Penyakit Daun Jagung")
st.write("Unggah gambar daun jagung untuk mendeteksi apakah daun tersebut sehat atau terkena penyakit.")

# --------------------------
# 1️⃣ Upload Gambar
# --------------------------
uploaded_file = st.file_uploader("📤 Unggah Gambar Daun Jagung", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Gambar yang diunggah", use_container_width=True)

    st.markdown("---")
    # --------------------------
    # 2️⃣ Instruksi dan Tombol
    # --------------------------
    st.subheader("🧠 Jalankan Model")
    st.write("Klik tombol di bawah ini untuk memulai proses klasifikasi.")
    run_classification = st.button("🚀 Jalankan Klasifikasi", type="primary")

    # --------------------------
    # 3️⃣ Proses Klasifikasi
    # --------------------------
    if run_classification:
        with st.spinner("Model sedang memproses gambar... ⏳"):
            input_shape = classifier.input_shape[1:3]
            img_resized = img.resize(input_shape)
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array.astype("float32") / 255.0

            # Prediksi
            prediction = classifier.predict(img_array)
            class_index = int(np.argmax(prediction))
            confidence = float(np.max(prediction))

            labels = ["Blight", "Common Rust", "Grey Spot Leaf", "Healthy"]
            predicted_label = labels[class_index]

            st.session_state["hasil_prediksi"] = {
                "label": predicted_label,
                "confidence": confidence,
                "model": "Isti_Laporan_2.h5"
            }

    # --------------------------
    # 4️⃣ Hasil Klasifikasi
    # --------------------------
    st.markdown("---")
    st.subheader("📊 Hasil Klasifikasi")

    if "hasil_prediksi" in st.session_state:
        hasil = st.session_state["hasil_prediksi"]

        st.markdown(
            f"**📷 Hasil Prediksi:** <span style='color:#00FF00;font-weight:bold'>{hasil['label']}</span>",
            unsafe_allow_html=True
        )
        st.markdown(f"**📈 Tingkat Keyakinan:** {hasil['confidence']*100:.2f}%")
        st.markdown(f"**💾 Model Digunakan:** `{hasil['model']}`")

        # Rekomendasi
        advice = {
            "Blight": "🌿 Terdeteksi *Blight (Hawar Daun)*. Isolasi tanaman terinfeksi dan hindari penyiraman berlebih.",
            "Common Rust": "🌾 Terdeteksi *Common Rust (Karat Daun)*. Lakukan peny
