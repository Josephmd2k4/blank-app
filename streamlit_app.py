import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import zipfile
import numpy as np
import cv2
import io

# Joseph DeMarco, Brendan Whitmire, Aiden Coffey, Mindy Cook
# Senior Project AI detection of Thyroid Cancer
# Website part.

@st.cache_resource
def load_model():
    model = EfficientNet.from_name('efficientnet-b0')  # Initialize the architecture
    num_ftrs = model._fc.in_features
    model._fc = torch.nn.Linear(num_ftrs, 2)
    
    state_dict = torch.load('efficientnet-b0-clf.pt', map_location=torch.device('cpu'))
    state_dict.pop('_fc.weight', None)
    state_dict.pop('_fc.bias', None)
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()  # Set model to evaluation mode
    return model

model = load_model()

# Define a function to preprocess the uploaded image
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)  # Add batch dimension

# Grad-CAM function to generate heatmap
def generate_gradcam_heatmap(model, image_tensor, target_layer_name='features'):
    # Hook to extract gradients and activations
    gradients = []
    activations = []
    
    def save_gradients(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def save_activations(module, input, output):
        activations.append(output)
        
    # Register hooks on the target layer
    target_layer = dict(model.named_modules())[target_layer_name]
    target_layer.register_forward_hook(save_activations)
    target_layer.register_backward_hook(save_gradients)
    
    # Forward pass
    model.zero_grad()
    output = model(image_tensor)
    predicted_class = output.argmax(dim=1).item()
    score = output[0, predicted_class]
    
    # Backward pass to get gradients
    score.backward()
    gradients = gradients[0].cpu().data.numpy()
    activations = activations[0].cpu().data.numpy()
    
    # Compute Grad-CAM
    weights = np.mean(gradients, axis=(2, 3), keepdims=True)
    cam = np.sum(weights * activations, axis=1)[0]
    cam = np.maximum(cam, 0)  # ReLU
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize to [0, 1]
    return cam

# Display heatmap on the original image
def overlay_heatmap(image, heatmap):
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlayed_img = cv2.addWeighted(np.array(image), 0.6, heatmap, 0.4, 0)
    return overlayed_img

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
                    image = Image.open(file).convert('RGB')
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
                    
                    # Generate and display Grad-CAM heatmap
                    cam = generate_gradcam_heatmap(model, processed_image)
                    overlayed_img = overlay_heatmap(image, cam)
                    st.image(overlayed_img, caption=f'Heatmap for {image_file}', use_column_width=True)

            # Calculate the average confidence for cancerous class
            average_confidence_cancerous = total_confidence_cancerous / total_images

            # Output the overall result
            st.write(f"\n**Overall Prediction for Patient ({uploaded_file.name})**")
            st.write(f"The average confidence across {total_images} images is {average_confidence_cancerous:.2f}%.")

            if average_confidence_cancerous > 50:
                st.write("The scan is classified as **cancerous**.")
            else:
                st.write("The scan is classified as **non-cancerous**.")
    else:
        st.write("Please upload a zip file containing images.")
