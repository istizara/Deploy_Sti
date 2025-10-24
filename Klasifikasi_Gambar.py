if run_classification:
    with st.spinner("🔍 Memproses gambar... harap tunggu sebentar"):
        try:
            # ================================
            # Tahap 1: Validasi daun jagung
            # ================================
            detection_result = yolo_model(img)
            boxes = detection_result[0].boxes

            # Ambil confidence dari setiap deteksi
            conf_scores = boxes.conf.tolist() if boxes is not None else []
            valid_detections = [c for c in conf_scores if c > 0.5]  # threshold 50%

            if len(valid_detections) == 0:
                # Tidak ada daun jagung dengan confidence cukup tinggi
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

                # Simpan status agar muncul di hasil klasifikasi
                st.session_state["hasil_prediksi"] = {
                    "label": "Tidak terdeteksi daun jagung",
                    "confidence": 0.0,
                    "model": "Isti_Laporan_2.h5"
                }

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

                st.session_state["hasil_prediksi"] = {
                    "label": predicted_label,
                    "confidence": confidence,
                    "model": "Isti_Laporan_2.h5"
                }

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan: {e}")
