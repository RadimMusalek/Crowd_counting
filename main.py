import streamlit as st
import torch
from PIL import Image
import requests
from transformers import pipeline
import openai
import boto3
import os
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Crowd Estimator",
    page_icon="👥",
    layout="wide"
)

# Configure API credentials
@st.cache_resource
def setup_apis():
    # OpenAI setup
    openai.api_key = os.getenv('OPENAI_API_KEY')
    
    # AWS setup
    aws_session = boto3.Session(
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION', 'us-east-1')
    )
    rekognition_client = aws_session.client('rekognition')
    return rekognition_client


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


def predict_openai(image):
    # Convert image to bytes
    import io
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Call OpenAI API
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "How many people are in this image? Please respond with just a number."},
                    # {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_byte_arr}"}}
                ]
            }
        ],
        max_tokens=300
    )
    
    # Extract the number from the response
    try:
        count = int(response.choices[0].message.content.strip())
    except:
        count = 0
    return count


def predict_aws(image, rekognition_client):
    # Convert image to bytes
    import io
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Call AWS Rekognition
    response = rekognition_client.detect_labels(
        Image={'Bytes': img_byte_arr},
        MaxLabels=100
    )
    
    # Find 'Person' label and get instance count
    for label in response['Labels']:
        if label['Name'] == 'Person':
            return len(label['Instances'])
    return 0

# Main Streamlit app
st.title("Crowd Size Estimator")
st.write("Upload an image to estimate the number of people in a crowd")

# Model selection
model_option = st.selectbox(
    "Select Model",
    ["Hugging Face Resnet", "OpenAI 4o-mini", "AWS Rekognition"]
)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Make prediction based on selected model
    with st.spinner("Analyzing image..."):
        try:
            if model_option == "Hugging Face Resnet":
                model = load_hf_resnet_model()
                estimated_crowd = predict_huggingface_resnet(image, model)
            
            elif model_option == "OpenAI 4o-mini":
                estimated_crowd = predict_openai(image)
            
            else:  # AWS Rekognition
                rekognition_client = setup_apis()
                estimated_crowd = predict_aws(image, rekognition_client)
            
            # Display results
            st.success("Analysis complete!")
            st.metric("Estimated number of people", estimated_crowd)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Add sidebar with information
with st.sidebar:
    st.header("About")
    st.write("""
    This app uses different AI models to estimate the number of people in a crowd from an uploaded image.
    
    Available models:
    - Hugging Face Resnet (free, open-source)
    - OpenAI 4o-mini (requires API key)
    - AWS Rekognition (requires AWS credentials)
    """)