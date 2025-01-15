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
    # Setup page
    setup_page()

    # Load environment variables
    load_credentials()

    api_limiter = APILimiter(
        daily_user_limit=config.DAILY_USER_LIMIT,
        daily_total_limit=config.DAILY_TOTAL_LIMIT
    )

    # Display usage stats in sidebar
    api_limiter.display_usage_stats()

    # Setup sidebar
    display_sidebar()
    
    # Get model selection
    model_option = get_model_selection()
    
    # Handle file upload
    image, image_name = handle_file_upload()

    if image is not None: # and model_option != "Select a model":
        if st.button("Run Analysis"):
            # Check API limits before processing
            if model_option in ["OpenAI 4o-mini", "AWS Rekognition"]:
                if not api_limiter.check_limits():
                    st.stop()
            
            # Process image
            with st.spinner("Analyzing image..."):
                estimated_crowd = process_image(image, model_option)
                if estimated_crowd is not None:
                    # Increment API usage if successful
                    if model_option in ["OpenAI 4o-mini", "AWS Rekognition"]:
                        api_limiter.increment_usage()
                    
                    st.success("Analysis complete!")
                    st.metric("Estimated number of people in " + image_name, estimated_crowd)


if __name__ == "__main__":
    # Run the main function
    main()
    