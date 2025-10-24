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

    # Sidebar untuk upload dan pengaturan
    st.sidebar.header("⚙️ Pengaturan Deteksi")
    model_path = st.sidebar.text_input("Isti_Laporan 4.pt")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

    # Upload gambar
    uploaded_file = st.file_uploader("Unggah gambar daun jagung...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang Diupload", use_column_width=True)

        # Tombol Deteksi
        if st.button("🔎 Jalankan Deteksi"):
            with st.spinner("Model sedang mendeteksi..."):
                # Simpan sementara file upload
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                image.save(temp_file.name)

                # Load model YOLO
                model = YOLO(model_path)

                # Jalankan prediksi
                results = model.predict(source=temp_file.name, conf=conf_threshold)
                result_image = results[0].plot()  # hasil gambar dengan bounding box

                # Konversi hasil ke format RGB agar bisa ditampilkan Streamlit
                result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

                st.image(result_image, caption="Hasil Deteksi", use_column_width=True)

                # Tampilkan hasil klasifikasi per objek
                st.subheader("📋 Hasil Deteksi:")
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
