import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
st.title("Thyroid Cancer")
menu = ["Image","Dataset","About"]
choice = st.sidebar.selectbox("Menu",menu)
uploaded_file = st.file_uploader("Upload Photo Below:")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption = 'Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Predicting...")
    processed_image = preprocess_image(image)