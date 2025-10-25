import streamlit as st

# ==========================
# Menu dan Navigasi
# ==========================

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
    icon=":material/leaf:",
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
