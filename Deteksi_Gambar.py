import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os

# ==========================
# Fungsi: Halaman Object Detection
# ==========================
def object_detection_page():
    st.title("🔍 Deteksi Objek Penyakit Daun Jagung")

    # ==========================
    # Load model YOLO (.pt)
    # ==========================
    model_path = "model/Isti_Laporan 4.pt"
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"Gagal memuat model dari `{model_path}` ❌")
        st.stop()

    # ==========================
    # Upload gambar
    # ==========================
    uploaded_file = st.file_uploader("📤 Unggah gambar daun jagung...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # Tombol di atas gambar
        run_detection = st.button("🔍 Jalankan Deteksi")

        # Buat dua kolom sejajar
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="🖼️ Gambar Asli", use_container_width=True)

        # Jika tombol ditekan
        if run_detection:
            with st.spinner("Model sedang memproses gambar... ⏳"):
                # Simpan file upload sementara
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                if image.mode != "RGB":
                    image = image.convert("RGB")

                image.save(temp_file.name)

                # Jalankan deteksi
                results = model.predict(source=temp_file.name, conf=0.5)
                result_image = results[0].plot()  # gambar hasil deteksi (bbox)

                # Konversi hasil ke format RGB untuk Streamlit
                result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

                # Tampilkan hasil di kolom kedua
                with col2:
                    st.image(result_image, caption="📊 Hasil Deteksi", use_container_width=True)

                # Bersihkan file sementara
                os.remove(temp_file.name)

# ==========================
# Jalankan fungsi agar halaman tampil
# ==========================
if __name__ == "__main__":
    object_detection_page()
