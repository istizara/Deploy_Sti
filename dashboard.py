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
    icon=":material/house:",
    default=True,
)
Object_detection_page = st.Page(
    "Deteksi_Gambar.py",
    title="Deteksi Gambar",
    icon=":material/lamp:",
)
# --- NAVIGATION SETUP [WITH SECTIONS]---
pg = st.navigation(
    {
        "Info": [Homepage],
         "Projects" : [Object_detection_page]
    }
)

# --- SHARED ON ALL PAGES ---
st.sidebar.markdown("Made with ❤️ by [Stiwww]")


# --- RUN NAVIGATION ---
pg.run()
