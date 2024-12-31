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

    # Setup sidebar
    display_sidebar()
    
    # Get model selection
    model_option = get_model_selection()
    
    # Handle file upload
    image, image_name = handle_file_upload()

    # Process image if upload is successful
    if image is not None:
        try:
            # Process image with selected model
            with st.spinner("Analyzing image..."):
                estimated_crowd = process_image(image, model_option)
                
                # Display results
                st.success("Analysis complete!")
                st.metric("Estimated number of people in " + image_name, estimated_crowd)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    # Run the main function
    main()