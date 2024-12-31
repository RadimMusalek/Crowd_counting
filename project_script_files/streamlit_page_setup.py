import streamlit as st
import os
from PIL import Image
import PIL
from project_script_files.HF_model import predict_huggingface_resnet, load_hf_resnet_model
from project_script_files.openai_model import predict_openai
from project_script_files.aws_model import predict_aws
from project_script_files.utils import setup_apis


def setup_page() -> None:
    """Configures the Streamlit page settings and layout.

    Sets up the page title, icon, and layout configuration.
    """
    # Set page configuration
    st.set_page_config(
        page_title="Crowd Estimator",
        page_icon="👥",
        layout="wide"
    )
    # Main Streamlit app
    st.title("Crowd Size Estimator")
    st.write("Upload an image to estimate the number of people in a crowd")


def get_model_selection() -> str:
    """Provides model selection interface.

    Returns:
        str: Selected model name
    """
    return st.selectbox(
        "Select Model",
        ["Hugging Face Resnet", "OpenAI 4o-mini", "AWS Rekognition"]
    )


def display_sidebar() -> None:
    """Sets up and displays the sidebar information and debug options."""
    with st.sidebar:
        st.header("About")
        st.write("""
        This app uses different AI models to estimate the number of people 
        in a crowd from an uploaded image.
        
        Available models:
        - Hugging Face Resnet (free, open-source)
        - OpenAI 4o-mini (uses OpenAI API key)
        - AWS Rekognition (uses AWS credentials)
        """)
        
        if st.checkbox("Debug Credentials"):
            st.write("AWS Region:", os.getenv("AWS_REGION"))
            st.write("OpenAI API:", "✓" if os.getenv("OPENAI_API_KEY") else "✗")
            st.write("AWS Access:", "✓" if os.getenv("AWS_ACCESS_KEY_ID") else "✗")
            st.write("HF Token:", "✓" if os.getenv("HUGGINGFACE_TOKEN") else "✗")


def handle_file_upload() -> tuple:
    """Handles file upload and initial image processing.

    Returns:
        tuple: (PIL.Image, str) if file is uploaded, (None, None) otherwise
    """
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            image_name = uploaded_file.name
            st.image(image, caption=image_name, use_container_width=True)
            return image, image_name
        except Exception as e:
            st.error(f"Error opening image: {str(e)}")
            return None, None


def process_image(image: PIL.Image, model_option: str) -> int:
    """Processes the image using the selected model.

    Args:
        image (PIL.Image): Image to process
        model_option (str): Name of the selected model

    Returns:
        int: Estimated crowd count

    Raises:
        Exception: If image processing fails
    """
    if model_option == "Hugging Face Resnet":
        model = load_hf_resnet_model()
        return predict_huggingface_resnet(image, model)

    elif model_option == "OpenAI 4o-mini":
        return predict_openai(image)

    else:  # AWS Rekognition
        rekognition_client = setup_apis()
        return predict_aws(image, rekognition_client)