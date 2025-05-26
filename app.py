import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("model_mineral.h5")

# Kelas sesuai urutan saat training (urutannya harus sama persis)
class_names = ["malachite", "chrysocolla", "pyrite", "biotite", "bornite", "muscovite", "quartz"]

# Judul aplikasi
st.set_page_config(page_title="Mineral Classifier", layout="centered")
st.title("🪨 Mineral Classification App")
st.markdown("Upload gambar mineral dan dapatkan prediksinya!")

# Upload gambar
uploaded_file = st.file_uploader("Upload file gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang di-upload", use_column_width=True)

    # Preprocessing gambar
    img = image.resize((150, 150))  # sesuaikan dengan input model
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalisasi jika model dilatih dengan rescale

    # Prediksi
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Tampilkan hasil
    st.markdown(f"### Prediksi: `{predicted_class}`")
    st.markdown(f"**Confidence:** {confidence:.2f}%")
