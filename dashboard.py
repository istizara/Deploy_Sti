import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import base64
import os
import numpy as np
from PIL import Image
import cv2

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Isti_Laporan 4.pt")  # Model deteksi objek
    classifier = tf.keras.models.load_model("model/Isti_Laporan_2.h5")  # Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

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
        "Info": [Homepage]
    }
)
