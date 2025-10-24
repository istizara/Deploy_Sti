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

uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    # Tombol Run di tengah
    col_center = st.columns([1, 1, 1])[1]
    with col_center:
        run_classification = st.button("🧠 Run Classification", type="primary")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(img, caption="🖼️ Uploaded Image", width=300)

        if run_classification:
            with st.spinner("🔍 Memproses gambar... harap tunggu sebentar"):
                try:
                    # ================================
                    # Tahap 1: Validasi daun jagung
                    # ================================
                    detection_result = yolo_model(img)
                    labels_detected = [int(box.cls) for box in detection_result[0].boxes]

                    if len(labels_detected) == 0:
                        st.markdown(
                            """
                            <div style="
                                background-color: #FFF3CD;
                                color: #856404;
                                padding: 15px;
                                border-radius: 10px;
                                border: 1px solid #FFEeba;
                                font-size: 16px;
                                text-align: justify;
                                width: 100%;
                            ">
                            ⚠️ <b>Gambar yang diunggah tidak terdeteksi sebagai daun jagung.</b><br>
                            Silakan unggah gambar daun jagung yang valid agar sistem dapat mengklasifikasikan dengan akurat.
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    else:
                        # ================================
                        # Tahap 2: Klasifikasi penyakit daun
                        # ================================
                        input_shape = classifier.input_shape[1:3]
                        img_resized = img.resize(input_shape)
                        img_array = image.img_to_array(img_resized)
                        img_array = np.expand_dims(img_array, axis=0)
                        img_array = img_array.astype("float32") / 255.0

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

                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan: {e}")

    with col2:
        st.markdown("### 📊 Hasil Klasifikasi")

        if "hasil_prediksi" in st.session_state:
            hasil = st.session_state["hasil_prediksi"]

            warna_label = {
                "Blight": "#FF4B4B",
                "Common Rust": "#FFA500",
                "Grey Spot Leaf": "#00C853",
                "Healthy": "#1E90FF"
            }
           if hasil["label"] == "Tidak terdeteksi daun jagung":
               st.warning("⚠️ Gambar tidak terdeteksi sebagai daun jagung. Silakan unggah gambar yang valid.")
            else:
                warna = warna_label.get(hasil["label"], "#FFFFFF")
                st.markdown(
                f"**📷 Prediction:** <span style='color:{warna};font-weight:bold'>{hasil['label']}</span>",
                unsafe_allow_html=True
            )
                st.markdown(f"**📈 Confidence:** {hasil['confidence']*100:.2f}%")
                st.markdown(f"**💾 Model Used:** `{hasil['model']}`")
            
                st.info(advice[hasil["label"]])

        else:
            st.write("⚙️ Hasil prediksi akan muncul setelah menekan tombol **Run Classification**.")
else:
    st.info("📸 Silakan unggah gambar daun jagung terlebih dahulu.")
