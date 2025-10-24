import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# =====================================
# 🔧 Load model (.h5)
# =====================================
MODEL_PATH = "model/Isti_Laporan 2.h5"
model = load_model(MODEL_PATH)

# Daftar label kelas (ubah sesuai model kamu)
CLASS_NAMES = ["Blight", "Common Rust", "Gray Leaf Spot", "Healthy"]

# =====================================
# 🌿 Sidebar Klasifikasi Gambar
# =====================================
def klasifikasi_gambar_sidebar():
    st.sidebar.title("🧠 Klasifikasi Gambar Daun Jagung")

    uploaded_file = st.sidebar.file_uploader(
        "📤 Unggah gambar daun jagung...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Tampilkan gambar yang diunggah
        image_pil = Image.open(uploaded_file).convert("RGB")
        st.image(image_pil, caption="🖼️ Gambar yang diunggah", use_container_width=True)

        # Tombol klasifikasi
        if st.button("🔍 Jalankan Klasifikasi"):
            with st.spinner("Model sedang memproses gambar... ⏳"):
                # ===============================
                # Preprocessing gambar
                # ===============================
                img = image_pil.resize((224, 224))  # sesuaikan ukuran input model
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0  # normalisasi jika model dilatih begitu

                # ===============================
                # Prediksi
                # ===============================
                predictions = model.predict(img_array)
                class_index = np.argmax(predictions)
                confidence = np.max(predictions)

                # ===============================
                # Tampilkan hasil
                # ===============================
                st.success(f"✅ Hasil Klasifikasi: **{CLASS_NAMES[class_index]}**")
                st.write(f"📊 Tingkat keyakinan: **{confidence*100:.2f}%**")

# =====================================
# Run app
# =====================================
if __name__ == "__main__":
    main()
