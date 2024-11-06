import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F  # Import for softmax
import zipfile
import io

# Joseph DeMarco, Brendan Whitmire, Aiden Coffey, Mindy Cook
# Senior Project AI detection of Thyroid Cancer
# Website part.

@st.cache_resource
def load_model():
    model = EfficientNet.from_name('efficientnet-b0')  # Initialize the architecture

    # Modify the final fully connected layer to match your output (2 classes)
    num_ftrs = model._fc.in_features
    model._fc = torch.nn.Linear(num_ftrs, 2)

    # Load the pretrained weights except the final layer (_fc)
    state_dict = torch.load('efficientnet-b0-clf.pt', map_location=torch.device('cpu'))
    
    # Ignore the mismatch in the final fully connected layer
    state_dict.pop('_fc.weight', None)
    state_dict.pop('_fc.bias', None)
    
    model.load_state_dict(state_dict, strict=False)  # Load weights except fc
    model.eval()  # Set model to evaluation mode
    return model

model = load_model()

# Define a function to preprocess the uploaded image
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust size to what your model expects
        transforms.ToTensor(),          # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with standard ImageNet values
    ])
    img_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    return img_tensor

st.title("ThyroiDx Model")
uploaded_file = st.file_uploader("Upload a Zip File of Images Below:")

if uploaded_file is not None:
    if uploaded_file.name.endswith('.zip'):
        # Load and extract images from the zip file
        zip_file = zipfile.ZipFile(uploaded_file)
        image_files = [file for file in zip_file.namelist() if file.endswith(('jpg', 'jpeg', 'png'))]

        if not image_files:
            st.write("No image files found in the zip file.")
        else:
            total_confidence_cancerous = 0
            total_images = 0

            for image_file in image_files:
                with zip_file.open(image_file) as file:
                    image = Image.open(file)
                    st.image(image, caption=f'Uploaded Image: {image_file}', use_column_width=True)
                    
                    # Preprocess the image
                    processed_image = preprocess_image(image)

                    # Run inference with PyTorch model
                    with torch.no_grad():
                        prediction = model(processed_image)
                    
                    # Apply softmax to get probabilities for each class
                    probabilities = F.softmax(prediction, dim=1)
                    
                    # Get the probability for the cancerous class (class 1)
                    confidence_cancerous = probabilities[0][1].item() * 100  # Convert to percentage
                    total_confidence_cancerous += confidence_cancerous
                    total_images += 1

                    st.write(f"Confidence that {image_file} is cancerous: {confidence_cancerous:.2f}%")

            # Calculate the average confidence for cancerous class
            average_confidence_cancerous = total_confidence_cancerous / total_images

            # Output the overall result
            st.write(f"\n**Overall Prediction for Patient ({uploaded_file.name})**")
            st.write(f"The average confidence across {total_images} images is {average_confidence_cancerous:.2f}%.")

            # Determine final classification based on average confidence
            if average_confidence_cancerous > 50:
                st.write("The scan is classified as **cancerous**.")
            else:
                st.write("The scan is classified as **non-cancerous**.")
    else:
        st.write("Please upload a zip file containing images.")
