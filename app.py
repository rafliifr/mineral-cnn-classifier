import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("model_mineral.h5")

# Kelas sesuai urutan saat training
class_names = ["biotite", "bornite", "chrysocolla", "malachite", "muscovite", "pyrite", "quartz"]

# Judul aplikasi
st.set_page_config(page_title="Mineral Classifier", layout="centered")
st.title("🪨 Klasifikasi Mineral")
st.markdown("Upload gambar mineral dan dapatkan prediksi jenisnya!")

# Upload gambar
uploaded_file = st.file_uploader("Upload file gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang di-upload", use_column_width=True)

    # Preprocessing
    img = image.resize((150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediksi
    prediction = model.predict(img_array)[0]
    sorted_indices = np.argsort(prediction)[::-1]

    # Tampilkan prediksi utama
    predicted_class = class_names[sorted_indices[0]]
    confidence = prediction[sorted_indices[0]] * 100
    st.markdown(f"### Prediksi Utama: `{predicted_class}`")
    st.markdown(f"**Confidence:** {confidence:.2f}%")

    # Tampilkan semua confidence dengan horizontal bar (progress-like style)
    st.markdown("### Confidence Semua Kelas:")
    for i in sorted_indices:
        label = f"{class_names[i]}: {prediction[i]:.3f}"
        bar_width = int(prediction[i] * 100)
        st.markdown(f"""
        <div style="margin-bottom: 8px;">
            <div style="font-size: 15px; color: white;">{label}</div>
            <div style="background-color: #333; border-radius: 5px; height: 16px;">
                <div style="width: {bar_width}%; background-color: #1f77b4; height: 100%; border-radius: 5px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
