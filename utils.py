import streamlit as st
import boto3
import os
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Configure API credentials
@st.cache_resource
def setup_apis():
    # AWS setup
    aws_session = boto3.Session(
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name='eu-west-1' # os.getenv('AWS_REGION')
    )
    rekognition_client = aws_session.client('rekognition')
    return rekognition_client