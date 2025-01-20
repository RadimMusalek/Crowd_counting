"""
Crowd Counting Application

This is the main module for a Streamlit-based crowd counting application that uses
multiple AI models to estimate the number of people in uploaded images. It supports
AWS Rekognition, OpenAI Vision, and Hugging Face models for crowd detection.
The application provides a user-friendly interface for:
   - Image upload
   - Model selection
   - Result visualization
   - Credential management
Typical usage:
   streamlit run main.py
"""
import streamlit as st
from project_script_files.utils import load_credentials
from project_script_files.streamlit_page_setup import setup_page, get_model_selection, display_sidebar, handle_file_upload, process_image
from project_script_files.api_limits import APILimiter
from project_script_files.api_credentials import APICredentialsManager
from configs import config


def main() -> None:
    """Main application function for the Crowd Counting interface.

    Sets up the Streamlit interface, handles user interactions, and manages
    the workflow of image upload, model selection, and prediction display.

    The function:
    - Configures the Streamlit page
    - Provides model selection
    - Handles image upload
    - Processes images through selected models
    - Displays results
    - Manages error handling

    Note:
    - Requires valid API credentials
    - Supports multiple image formats
    - Provides real-time feedback
    - Includes debug options in sidebar
    """

    # Initialize API limiters and credentials manager
    api_limiter = APILimiter()
    api_credentials_manager = APICredentialsManager()

    # Setup page
    setup_page()

    # Display credentials UI in sidebar
    api_credentials_manager.credentials_ui()

    # Display usage stats in sidebar if using default credentials
    if not (api_credentials_manager.is_using_own_credentials('aws') and 
            api_credentials_manager.is_using_own_credentials('openai')):
        api_limiter.display_usage_stats()

    # Setup sidebar
    display_sidebar()
    
    # Get model selection
    model_option = get_model_selection()
    
    # Handle file upload
    image, image_name = handle_file_upload()

    if image is not None: # and model_option != "Select a model":
        if st.button("Run Analysis"):
            # Check API limits only if using default credentials
            if model_option == "OpenAI 4o-mini":
                if not api_credentials_manager.is_using_own_credentials('openai'):
                    if not api_limiter.check_limits():
                        st.stop()
            elif model_option == "AWS Rekognition":
                if not api_credentials_manager.is_using_own_credentials('aws'):
                    if not api_limiter.check_limits():
                        st.stop()
            
            # Process image
            with st.spinner("Analyzing image..."):
                estimated_crowd = process_image(image, model_option, api_credentials_manager)
                if estimated_crowd is not None:
                    # Increment API usage if using default credentials
                    if model_option == "OpenAI 4o-mini" and not api_credentials_manager.is_using_own_credentials('openai'):
                        api_limiter.increment_usage()
                    elif model_option == "AWS Rekognition" and not api_credentials_manager.is_using_own_credentials('aws'):
                        api_limiter.increment_usage()
                    
                    st.success("Analysis complete!")
                    st.metric("Estimated number of people in " + image_name, estimated_crowd)


if __name__ == "__main__":
    # Load environment variables once per session
    load_credentials()  
    
    # Run the main function
    main()
