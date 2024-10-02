#Joseph DeMarco, Brendan Whitmire, Aiden Coffey, Mindy Cook
#Senior Project AI detection of Thyroid Cancer
#Website part.

import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import numpy as np
from efficientnet_pytorch import EfficientNet



@st.cache_resource
def load_model():
    model = EfficientNet.from_name('efficientnet-b0')  # Initialize the architecture

    # Modify final layer to have 2 classes
    num_ftrs = model._fc.in_features
    model._fc = torch.nn.Linear(num_ftrs, 2)

    # Load the pretrained weights except the final layer (_fc)
    state_dict = torch.load('efficientnet-b0-clf.pt', map_location=torch.device('cpu'))
    
    # Ignore the mismatch in the final fully connected layer #FIX this later, see if theres a way to make them match up
    state_dict.pop('_fc.weight', None)
    state_dict.pop('_fc.bias', None)
    
    model.load_state_dict(state_dict, strict=False)  # Load weights except fc
    model.eval()  # Set model to evaluation mode, it will now be ready to make predictions for new data
    return model

model = load_model()


# Define a function to preprocess the uploaded image
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust size to what the model expects
        transforms.ToTensor(),          # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])
    img_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    return img_tensor
#Pretty things for website
st.title("Thyroid Cancer")
menu = ["Image","Dataset","About"]
choice = st.sidebar.selectbox("Menu",menu)
uploaded_file = st.file_uploader("Upload Photo Below:")
#When a file is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Predicting...")
    #Make image set up to be put through model
    processed_image = preprocess_image(image)

    # Run inference with PyTorch model
    with torch.no_grad():
        prediction = model(processed_image)
    
    # Get the predicted class index (0 or 1, non-cancer or cancer.)
    predicted_class = torch.argmax(prediction, dim=1).item()

    # Output the result
    if predicted_class == 1:
        st.write("The scan is classified as **cancerous**.")
    else:
        st.write("The scan is classified as **non-cancerous**.")
