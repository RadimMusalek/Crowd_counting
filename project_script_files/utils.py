import streamlit as st
import boto3
import os
from dotenv import load_dotenv


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


# Load environment variables
def load_credentials():
    """Load credentials from environment or Streamlit secrets"""
    # For local development
    if os.path.exists(".env"):
        load_dotenv()

    # For Streamlit Cloud
    if st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        os.environ["AWS_ACCESS_KEY_ID"] = st.secrets["AWS_ACCESS_KEY_ID"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets["AWS_SECRET_ACCESS_KEY"]
        os.environ["AWS_REGION"] = st.secrets["AWS_REGION"]
        os.environ["HUGGINGFACE_TOKEN"] = st.secrets["HUGGINGFACE_TOKEN"]


def check_credentials():
    """Check if all required credentials are available"""
    required_credentials = [
        "OPENAI_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
        "HUGGINGFACE_TOKEN"
    ]

    missing = [cred for cred in required_credentials if not os.getenv(cred)]

    if missing:
        st.error(f"Missing credentials: {', '.join(missing)}")
        return False
    return True