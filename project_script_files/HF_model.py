from transformers import pipeline
import streamlit as st


# Load models
@st.cache_resource
def load_hf_resnet_model():
    return pipeline("object-detection", model="facebook/detr-resnet-50")


# Model prediction functions
def predict_huggingface_resnet(image, model):
    # Convert PIL image to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Get predictions
    results = model(image)
    
    # Count people (assuming 'person' is the label we're looking for)
    person_count = sum(1 for result in results if result['label'] == 'person')
    return person_count