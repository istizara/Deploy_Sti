import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

# ==========================
# 🔧 Load Model
# ==========================
MODEL_PATH = "model/Isti_Laporan_2.h5"
model = load_model(MODEL_PATH)

# Daftar label kelas
CLASS_NAMES = ["Blight", "Common Rust", "Gray Leaf Spot", "Healthy"]

# ==========================
# 🌿 Fungsi Klasifikasi
# ==========================
def klasifikasi_gambar(img):
    """Preprocessing dan prediksi gambar."""
    input_shape = model.input_shape[1:3]  # contoh (224, 224)
    img_resized = img.resize(input_shape)
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype("float32") / 255.0

    # Prediksi
    prediction = model.predict(img_array)
    class_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    return CLASS_NAMES[class_index], confidence

# ==========================
# 🌽 Aplikasi Streamlit
# ==========================
def main():
    st.title("🌿 Klasifikasi Penyakit Daun Jagung")
    st.markdown("Unggah gambar daun jagung dan jalankan model klasifikasi untuk mendeteksi penyakitnya.")

    # --- Upload gambar di halaman utama ---
    uploaded_file = st.file_uploader("📤 Upload Gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")

        # Tampilkan gambar
        st.image(img, caption="🖼️ Gambar yang diunggah", use_container_width=True)

        # Tombol untuk klasifikasi
        if st.button("🔍 Jalankan Klasifikasi"):
            with st.spinner("Model sedang memproses gambar... ⏳"):
                label, confidence = klasifikasi_gambar(img)

                st.success(f"### 🌾 Hasil Prediksi: **{label}**")
                st.write(f"📊 Tingkat keyakinan: **{confidence*100:.2f}%**")

                advice = {
                    "Blight": "Terdeteksi **hawar daun** 🌿. Isolasi tanaman yang terinfeksi.",
                    "Common Rust": "Terdeteksi **karat daun** 🌾. Gunakan fungisida berbasis tembaga.",
                    "Gray Leaf Spot": "Terdeteksi **bercak abu-abu** 🍂. Kurangi kelembapan sekitar tanaman.",
                    "Healthy": "Daun dalam kondisi **sehat** 🌱. Pertahankan perawatan yang baik."
                }

                st.info(advice[label])

# ==========================
# Jalankan aplikasi
# ==========================
if __name__ == "__main__":
    main()
