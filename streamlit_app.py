import streamlit as st

st.title("Thyroid Cancer")
st.write(
    "Upload photo below:"
)
menu = ["Image","Dataset","About"]
choice = st.sidebar.selectbox("Menu",menu)
st.file_uploader("Image must be 24x24 in PNG format")
