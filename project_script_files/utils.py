"""
Utilities Module for Crowd Counting Application
This module provides utility functions for API setup and credential management
across different services (AWS, OpenAI, Hugging Face). It handles environment
variable loading and validation for both local development and cloud deployment.
Typical usage example:
   load_credentials()
   if check_credentials():
       rekognition_client = setup_apis()
"""
import streamlit as st
import boto3
import os
from dotenv import load_dotenv


# Configure API credentials
@st.cache_resource
def setup_apis() -> boto3.client:
    """Sets up API clients for cloud services.

    Initializes and caches the AWS Rekognition client using credentials
    from environment variables. Uses Streamlit's caching to avoid
    recreating clients on each rerun.

    Returns:
        boto3.client: Configured AWS Rekognition client.

    Raises:
        boto3.exceptions.BotoCoreError: If AWS credentials are invalid.
        boto3.exceptions.ClientError: If region is not supported.

    Note:
        - Requires valid AWS credentials in environment
        - Uses 'eu-west-1' region for Rekognition
        - Client is cached after first creation
    """
    try:
        # AWS setup
        aws_session = boto3.Session(
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name='eu-west-1' # os.getenv('AWS_REGION')
        )
        rekognition_client = aws_session.client('rekognition')
        return rekognition_client
    except Exception as e:
        st.error(f"Error setting up APIs: {str(e)}")
        return None


# Load environment variables
def load_credentials() -> None:
    """Loads API credentials from environment or Streamlit secrets.

    Attempts to load credentials from local .env file for development
    or Streamlit secrets for cloud deployment. Sets environment variables
    for all required API services.

    Note:
        - Checks for local .env file first
        - Falls back to Streamlit secrets if available
        - Sets environment variables for:
            - OpenAI API
            - AWS credentials
            - Hugging Face token
    """
    # Local development
    # Try multiple possible locations for .env file
    possible_paths = [
        ".env",  # Current directory
        "../.env",  # One level up
        os.path.join(os.path.dirname(__file__), ".env"),  # Same directory as utils.py
        os.path.join(os.path.dirname(__file__), "../.env")  # One level up from utils.py
    ]

    # Try to load from .env file
    env_loaded = False
    for path in possible_paths:
        if os.path.exists(path):
            load_dotenv(path)
            print(f"Loaded .env from: {path}")  # Debug print
            env_loaded = True
            break

    if not env_loaded:
        print("No .env file found in any expected location")  # Debug print

        # For Streamlit Cloud
        try:
            if st.secrets:
                os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
                os.environ["AWS_ACCESS_KEY_ID"] = st.secrets["AWS_ACCESS_KEY_ID"]
                os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets["AWS_SECRET_ACCESS_KEY"]
                os.environ["AWS_REGION"] = st.secrets["AWS_REGION"]
                os.environ["HUGGINGFACE_TOKEN"] = st.secrets["HUGGINGFACE_TOKEN"]
        except Exception as e:
            st.error(f"Couldn't load Streamlit secrets: {str(e)}")

        # Check credentials before proceeding
        if not check_credentials():
            st.stop()


def check_credentials() -> bool:
    """Verifies that all required API credentials are available.

    Checks environment variables for all required API credentials
    and returns status indicating whether all credentials are present.

    Returns:
        bool: True if all required credentials are present, False otherwise.

    Note:
        - Checks for following credentials:
            - OPENAI_API_KEY
            - AWS_ACCESS_KEY_ID
            - AWS_SECRET_ACCESS_KEY
            - AWS_REGION
            - HUGGINGFACE_TOKEN
        - Displays error message if any credentials are missing
    """
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