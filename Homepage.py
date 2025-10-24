import streamlit as st
from PIL import Image

# ===========================
# Konfigurasi Halaman
# ===========================
st.set_page_config(
    page_title="Corn Disease Detection Dashboard for Smart Farming",
    page_icon="🌽",
    layout="wide",
)

# ===========================
# Fungsi: Tampilan Hero Section
# ===========================
def hero_section():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #cfe1b9;
            background-attachment: fixed;
            background-size: cover;
        }
        .hero {
            text-align: center;
            padding: 70px 10px;
            background: rgba(255, 255, 255, 0.25); 
            border-radius: 25px;
            width: 80%;
            margin: 60px auto;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        }
        .hero h1 {
            font-size: 2.5em;
            color: #1b3d1b;
            font-weight: 800;
            margin-bottom: 10px;
        }
        .hero p {
            font-size: 1.1em;
            color: #2b472b;
            margin-top: 0;
        }

        .btn-start:hover {
            background-color: #68804f;
        }
        </style>

        <div class='hero'>
            <h1>🌽 Corn Disease Detection Dashboard for Smart Farming</h1>
            <p>Deteksi dan klasifikasi penyakit daun jagung secara otomatis menggunakan AI 🤖</p>
        """,
        unsafe_allow_html=True
    )

# ===========================
# Fungsi: Tentang Proyek
# ===========================
def about_section():
    st.header("Tentang Proyek")
    st.write("""
    Website ini dikembangkan untuk membantu petani, peneliti, dan mahasiswa dalam mendeteksi serta 
    mengklasifikasi penyakit pada daun jagung menggunakan teknologi **Computer Vision** dan **Deep Learning**. 
    """)

# ===========================
# Fungsi: Jenis Penyakit
# ===========================
def disease_section():
    st.header("Jenis Penyakit yang Dapat Dideteksi")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image("Corn_Health (99).jpg", use_container_width=True)
        st.subheader("Healthy")
        st.caption("Daun hijau segar tanpa bercak.")
    with col2:
        st.image("Corn_Blight (947).JPG", use_container_width=True)
        st.subheader("Blight")
        st.caption("Bercak coklat memanjang di permukaan daun.")
    with col3:
        st.image("Corn_Common_Rust (900).JPG", use_container_width=True)
        st.subheader("Common Rust")
        st.caption("Bercak jingga kekuningan kecil.")
    with col4:
        st.image("Corn_Gray_Spot (496).JPG", use_container_width=True)
        st.subheader("Gray Leaf Spot")
        st.caption("Bercak persegi panjang abu-abu pada daun.")

# ===========================
# Fungsi: Fitur Utama
# ===========================
def features_section():
    st.header("Fitur Utama")
    st.markdown("""
    - 🔍 **Object Detection** – Menandai area terinfeksi pada daun jagung.  
    - 🧠 **AI Classification** – Mengidentifikasi jenis penyakit secara otomatis.  
    - 📊 **Confidence Score** – Menampilkan tingkat keyakinan model.  
    - 🌐 **User-Friendly Interface** – Tampilan mudah digunakan dan interaktif. 
    """)

# ===========================
# Fungsi: Cara Menggunakan Website
# ===========================
def how_to_use():
    st.header("Cara Menggunakan Website")
    st.write("""
    Website ini memiliki dua fitur utama untuk membantu dalam mengenali penyakit pada daun jagung 
    menggunakan teknologi **Artificial Intelligence (AI)**, yaitu *Deteksi Gambar* dan *Klasifikasi Gambar*.
    """)

    # Langkah umum
    st.subheader("Langkah Umum")
    st.markdown("""
    1. **Buka Sidebar** di sisi kiri layar.
    2. Pilih salah satu fitur utama:
        - 🟢 **Deteksi Gambar** — untuk mendeteksi area daun yang terinfeksi penyakit.
        - 🟡 **Klasifikasi Gambar** — untuk mengenali jenis penyakit berdasarkan gambar daun jagung.
    3. Ikuti panduan penggunaan di bawah sesuai fitur yang anda pilih.
    """)

    # Object Detection
    st.divider()
    st.subheader("🟢 Deteksi Gambar")
    st.markdown("""
    Fitur ini digunakan untuk **mendeteksi area pada daun jagung yang menunjukkan gejala penyakit**.
    
    **Langkah-langkah:**
    1. Klik menu *Object Detection* di sidebar.
    2. Unggah gambar daun jagung dalam format '.jpeg',`.jpg` atau `.png`.
    3. Tunggu beberapa detik hingga proses pendeteksian selesai.
    4. Hasil berupa gambar dengan kotak (bounding box) yang menandai area daun terinfeksi akan muncul di layar.
    """)

    # Klasifikasi Gambar
    st.divider()
    st.subheader("🟡 Klasifikasi Gambar")
    st.markdown("""
    Fitur ini digunakan untuk **mengidentifikasi jenis penyakit daun jagung** menggunakan model klasifikasi citra.

    **Langkah-langkah:**
    1. Klik menu *Klasifikasi Gambar* di sidebar.
    2. Unggah gambar daun jagung yang ingin dianalisis.
    3. Tunggu hingga proses klasifikasi selesai.
    4. Hasil akan menampilkan:
        - Jenis penyakit (misal: *Common Rust*, *Gray Leaf Spot*, *Blight*, atau *Healthy*).
        - Persentase tingkat kepercayaan model terhadap hasil klasifikasi.
    """)

    # Tips tambahan
    st.divider()
    st.info("""
    💡 **Tips:** Pastikan gambar daun jagung memiliki pencahayaan yang baik, fokus pada daun, 
    dan tidak terlalu banyak gangguan latar belakang agar hasil deteksi dan klasifikasi lebih akurat.
    """)

# ===========================
# Fungsi: Tim / Footer
# ===========================
def footer():
    st.write("---")
    st.markdown("""
    **Dibuat oleh:** Isti Kamila Nanda Zahra
    **Program Studi:** Statistika, Universitas Syiah Kuala  
    **© 2025**
    """)

# ===========================
# TAMPILAN UTAMA HOMEPAGE
# ===========================
hero_section()
about_section()
disease_section()
features_section()
how_to_use()
footer()
