import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import altair as alt
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

    # Tampilkan hasil utama
    predicted_class = class_names[sorted_indices[0]]
    confidence = prediction[sorted_indices[0]] * 100
    st.markdown(f"### Prediksi Utama: `{predicted_class}`")
    st.markdown(f"**Confidence:** {confidence:.2f}%")

    # DataFrame untuk grafik
    df = pd.DataFrame({
        "Mineral": class_names,
        "Confidence": prediction * 100
    }).sort_values("Confidence", ascending=True)  # ascending agar bar dari bawah ke atas

    # Horizontal Bar Chart
    st.markdown("### Confidence:")
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("Confidence", title="Confidence (%)"),
        y=alt.Y("Mineral", sort=None),
        color=alt.value("#4C72B0")
    ).properties(height=300)

    st.altair_chart(chart, use_container_width=True)
