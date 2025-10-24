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
    "Deteks_Gambar.py",
    title="Deteksi Gambar",
    icon=":material/bar_chart:",
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
