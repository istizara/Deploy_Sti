import streamlit as st

# ==========================
# Menu dan Navigasi
# ==========================

# --- CUSTOM BACKGROUND COLOR (apply to all pages) ---
page_bg = """
<style>
    /* Ubah warna background seluruh halaman */
    [data-testid="stAppViewContainer"] {
        background-color: #cfe1b9; 
    }
    /* Card atau container tetap putih */
        [data-testid="stHeader"], .st-emotion-cache-1r6slb0 {
            background: none;
    }

    /* Sidebar juga pakai warna yang sama */
    [data-testid="stSidebar"] {
        background-color: #98a77c; 
    }

    /* Saat fitur dipilih (aktif) */
    [data-testid="stSidebarNav"] a[aria-current="page"] {
        background-color: #ffffff !important; /* putih */
        color: #1b3d1b !important;
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Hover effect biar interaktif */
    [data-testid="stSidebarNav"] a:hover {
        background-color: #dbe8c0 !important;
        color: #1b3d1b !important;
    }

    /* Ubah warna tombol biar serasi */
    .stButton>button {
        background-color: #6da544;
        color: white;
        border-radius: 10px;
        border: none;
    }

    .stButton>button:hover {
        background-color: #588a34;
        color: #fff;
    }
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# --- SHARED ON ALL PAGES ---
st.logo("Logo.png")

# --- PAGE SETUP ---
Homepage = st.Page(
    "Homepage.py",
    title="Homepage",
    icon=":material/account_circle:",
    default=True,
)
Object_detection_page = st.Page(
    "Deteksi_Gambar.py",
    title="Deteksi Gambar",
    icon=":material/search:",
)
Klassification_page = st.Page(
    "Klasifikasi_Gambar.py",
    title="Klasifikasi Gambar",
    icon=":material/eco:",
)

# --- NAVIGATION SETUP [WITH SECTIONS]---
pg = st.navigation(
    {
        "Info": [Homepage],
         "Projects": [Object_detection_page, Klassification_page]
    }
)


# --- SHARED ON ALL PAGES ---
st.logo("Logo.png")

# --- PAGE SETUP ---
Homepage = st.Page(
    "Homepage.py",
    title="Homepage",
    icon=":material/account_circle:",
    default=True,
)
Object_detection_page = st.Page(
    "Deteksi_Gambar.py",
    title="Deteksi Gambar",
    icon=":material/search:",
)
Klassification_page = st.Page(
    "Klasifikasi_Gambar.py",
    title="Klasifikasi Gambar",
    icon=":material/eco:",
)
# --- NAVIGATION SETUP [WITH SECTIONS]---
pg = st.navigation(
    {
        "Info": [Homepage],
         "Projects" : [Object_detection_page, Klassification_page]
    }
)

# --- SHARED ON ALL PAGES ---
st.sidebar.markdown("Made with ❤️ by [Stiwww]")


# --- RUN NAVIGATION ---
pg.run()
