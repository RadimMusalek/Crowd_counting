import streamlit as st
from PIL import Image
from openai_model import predict_openai
from aws_model import predict_aws
from HF_model import predict_huggingface_resnet, load_hf_resnet_model
from utils import setup_apis, load_dotenv


def main():
    # Set page configuration
    st.set_page_config(
        page_title="Crowd Estimator",
        page_icon="👥",
        layout="wide"
    )

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

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Run the main function
    main()