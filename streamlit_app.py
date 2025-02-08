import streamlit as st
from PIL import Image
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F  # Import for softmax
import zipfile
import io
import torch
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# Joseph DeMarco, Brendan Whitmire, Aiden Coffey, Mindy Cook
# Senior Project AI detection of Thyroid Cancer
# Website part.

@st.cache_resource
def load_model():
    model = EfficientNet.from_name('efficientnet-b0')  # Initialize architecture

    # Modify the final fully connected layer to match your output classes (2)
    num_ftrs = model._fc.in_features
    model._fc = torch.nn.Linear(num_ftrs, 2)  # Ensure we override the last layer

    # Load the new model weights
    state_dict = torch.load('Best_model_trial_5_with_AUC_0.9740.pth', map_location=torch.device('cpu'))
    
    # Remove mismatched final layer weights
    state_dict.pop('_fc.weight', None)
    state_dict.pop('_fc.bias', None)

    # Load the rest of the state dictionary
    model.load_state_dict(state_dict, strict=False)  # Allow missing final layer

    model.eval()  # Set to evaluation mode
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

# Function to generate PDF with images
def generate_pdf(results, avg_confidence, final_classification):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setTitle("ThyroiDx Report")

    pdf.drawString(100, 750, "ThyroiDx AI Model - Thyroid Cancer Prediction Report")
    pdf.drawString(100, 730, f"Total Images Analyzed: {len(results)}")
    pdf.drawString(100, 710, f"Average Cancerous Confidence: {avg_confidence:.2f}%")
    pdf.drawString(100, 690, f"Final Classification: {final_classification}")

    pdf.drawString(100, 660, "Individual Image Results:")

    y_position = 600
    for image_name, confidence, image_data in results:
        # Insert image
        img_reader = ImageReader(image_data)
        pdf.drawImage(img_reader, 100, y_position - 80, width=100, height=100, preserveAspectRatio=True, mask='auto')

        # Display confidence score next to image
        pdf.drawString(220, y_position - 40, f"{image_name}: {confidence:.2f}%")

        y_position -= 120  # Move position down for next image

        if y_position < 100:  # Create new page if needed
            pdf.showPage()
            y_position = 750  # Reset position for new page

    pdf.save()
    buffer.seek(0)
    return buffer

# Create a sidebar for navigation
page = st.sidebar.selectbox("Navigation", ["ThyroiDx Model", "About Us"])

if page == "ThyroiDx Model":
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
                results = []  # Store image results for PDF

                columns = st.columns(3)  # Set up 3 columns for grid layout

                for i, image_file in enumerate(image_files):
                    with zip_file.open(image_file) as file:
                        image = Image.open(file)

                        # Preprocess the image
                        processed_image = preprocess_image(image)

                        # Run inference with PyTorch model
                        with torch.no_grad():
                            prediction = model(processed_image)

                        # Apply softmax to get probabilities for each class
                        probabilities = F.softmax(prediction, dim=1)

                        # Get the probability for the cancerous class (class 1)
                        confidence_cancerous = probabilities[0][0].item() * 100  # Convert to percentage
                        total_confidence_cancerous += confidence_cancerous
                        total_images += 1

                        # Store image in memory for PDF
                        img_bytes = io.BytesIO()
                        image.save(img_bytes, format='PNG')
                        img_bytes.seek(0)

                        results.append((image_file, confidence_cancerous, img_bytes))

                        # Display the image and prediction in the grid layout
                        with columns[i % 3]:  # Rotate across columns
                            st.image(image)
                            st.write(f"Confidence: {confidence_cancerous:.2f}%")

                # Calculate the average confidence for cancerous class
                average_confidence_cancerous = total_confidence_cancerous / total_images

                # Determine final classification based on average confidence
                final_classification = "cancerous" if average_confidence_cancerous > 50 else "non-cancerous"

                # Output the overall result
                st.write(f"\n**Overall Prediction for Patient ({uploaded_file.name})**")
                st.write(f"The average confidence across {total_images} images is {average_confidence_cancerous:.2f}%.")
                st.write(f"The patient is classified as **{final_classification}**.")

                # Generate and allow PDF download
                pdf_buffer = generate_pdf(results, average_confidence_cancerous, final_classification)
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_buffer,
                    file_name="ThyroiDx_Report.pdf",
                    mime="application/pdf"
                )
        else:
            st.write("Please upload a zip file containing images.")

elif page == "About Us":
    st.title("About Us")
    st.write("""
    **ThyroiDx** is a project developed by Joseph DeMarco, Brendan Whitmire, Aiden Coffey, and Melinda Cook.
    Our goal is to leverage AI technology to make thyroid cancer diagnosis more accessible, especially in
    regions with limited medical resources. By using this platform, doctors can analyze microscope images of
    thyroid tumor samples and receive a preliminary assessment on whether further testing is necessary.

    This project is particularly focused on addressing diagnostic needs in third-world countries, where early
    detection can save lives and reduce healthcare costs. Our AI model is trained on data collected from
    various global sources, ensuring reliable performance across different image qualities and types.
    """)
