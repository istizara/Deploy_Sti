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
        st.image(image, caption="🖼️ Gambar yang diunggah", use_column_width=True)

        # Tombol Deteksi
        if st.button("🔍 Jalankan Deteksi"):
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

                # Tampilkan hasil
                st.image(result_image, caption="📊 Hasil Deteksi", use_column_width=True)

                # Tampilkan detail deteksi
                st.subheader("📋 Detail Deteksi")
                for box in results[0].boxes:
                    cls = int(box.cls)
                    label = results[0].names[cls]
                    conf = float(box.conf)
                    st.markdown(f"- **{label}** dengan confidence **{conf*100:.1f}%**")

                # Bersihkan file sementara
                os.remove(temp_file.name)

# ==========================
# Jalankan fungsi agar halaman tampil
# ==========================
if __name__ == "__main__":
    object_detection_page()
