"""
Streamlit pagesetup and UI components
"""
import streamlit as st
import os
from PIL import Image
import PIL
from project_script_files.HF_model import predict_huggingface_resnet, load_hf_resnet_model
from project_script_files.openai_model import predict_openai
from project_script_files.aws_model import predict_aws
from project_script_files.utils import setup_apis
from project_script_files.api_credentials import APICredentialsManager


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
    # Create columns for upload options
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Upload your own image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    with col2:
        st.write("### Or try a sample image")
        sample_images = get_sample_images()
        selected_sample = st.selectbox(
            "Select a sample image",
            ["None"] + list(sample_images.keys())
        )

    # Handle image selection
    try:
        if uploaded_file is not None:
            # User uploaded an image
            image = Image.open(uploaded_file)
            image_name = uploaded_file.name
        elif selected_sample != "None":
            # User selected a sample image
            image_path = sample_images[selected_sample]
            image = Image.open(image_path)
            image_name = selected_sample
        else:
            return None, None
        
        # Display the selected image
        st.image(image, caption=image_name, use_container_width=True)
        return image, image_name
        
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None, None


def process_image(image: PIL.Image, model_option: str, api_credentials_manager: APICredentialsManager) -> int:
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
        return predict_openai(image, api_credentials_manager)

    else:  # AWS Rekognition
        rekognition_client = setup_apis()
        return predict_aws(image, rekognition_client, api_credentials_manager)
    

def get_sample_images():
    """Returns a dictionary of sample images for testing.

    Automatically detects and loads all supported image files from the
    sample_images directory.

    Returns:
        dict: Dictionary with image names as keys and file paths as values

    Note:
        - Supports .jpg, .jpeg, and .png files
        - Converts filenames to human-readable names
        - Ignores non-image files in the directory
    """
    # Define path to sample images folder
    sample_dir = os.path.join(os.path.dirname(__file__), "../sample_images")

    # Supported image extensions
    supported_extensions = {'.jpg', '.jpeg', '.png'}

    # Dictionary to store images
    sample_images = {}

    try:
        # List all files in the directory
        for filename in os.listdir(sample_dir):
            # Check if file has a supported extension
            if any(filename.lower().endswith(ext) for ext in supported_extensions):
                # Convert filename to display name (e.g., "crowd_scene.jpg" -> "Crowd Scene")
                display_name = (os.path.splitext(filename)[0]
                                .replace('_', ' ')
                                .replace('-', ' ')
                                .title())
                
                # Add to dictionary
                sample_images[display_name] = os.path.join(sample_dir, filename)
                
        if not sample_images:
            print(f"No sample images found in {sample_dir}")
            
    except FileNotFoundError:
        print(f"Sample images directory not found: {sample_dir}")
        os.makedirs(sample_dir, exist_ok=True)
        print(f"Created sample images directory: {sample_dir}")
    except Exception as e:
        print(f"Error loading sample images: {str(e)}")

    return sample_images