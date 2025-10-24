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

# --- NAVIGATION SETUP [WITH SECTIONS]---
pg = st.navigation(
    {
        "Info": [Homepage],
         "Projects" : []
    }
)

# --- SHARED ON ALL PAGES ---
st.sidebar.markdown("Made with ❤️ by [Stiwww]")


# --- RUN NAVIGATION ---
pg.run()
