import streamlit as st
import torch
from PIL import Image
import requests
from torchvision import transforms
import numpy as np
from torch.hub import load_state_dict_from_url

# Set page configuration
st.set_page_config(
    page_title="Crowd Estimator",
    page_icon="👥",
    layout="wide"
)

# Title and description
st.title("Crowd Size Estimator")
st.write("Upload an image to estimate the number of people in a crowd")

# Load pre-trained model
@st.cache_resource
def load_model():
    model = torch.hub.load('leeyeehoo/CSRNet-pytorch', 'csrnet', pretrained=True)
    model.eval()
    return model

model = load_model()

# Image preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Make prediction
    with st.spinner("Analyzing image..."):
        # Preprocess the image
        input_tensor = preprocess_image(image)
        
        # Get model prediction
        with torch.no_grad():
            output = model(input_tensor)
            predicted_count = int(output.sum().item())
        
        # Display results
        st.success("Analysis complete!")
        st.metric("Estimated number of people", predicted_count)
        
        st.info("""
        Note: This is a simplified estimation using a general-purpose image recognition model. 
        For more accurate results, a specialized crowd counting model would be needed.
        """)

# Add sidebar with information
with st.sidebar:
    st.header("About")
    st.write("""
    This app uses computer vision to estimate the number of people in a crowd from an uploaded image.
    
    The estimation is based on a pre-trained ResNet50 model and provides a rough approximation.
    """)

# Example implementation using CSRNet
def load_csrnet():
    model = torch.hub.load('leeyeehoo/CSRNet-pytorch', 'csrnet', pretrained=True)
    return model
