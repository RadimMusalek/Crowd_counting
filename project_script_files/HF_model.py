"""
Hugging Face Model Module for Crowd Counting

This module provides functionality to detect and count people in images using
Hugginge's DETR-ResNet model. It handles image preprocessing and person
detection using the pre-trained model.
Typical usage example:
   model = load_hf_resnet_model()
   count = predict_huggingface_resnet(image, model)
"""
from transformers import pipeline
import streamlit as st
from typing import Any
from PIL import Image


# Load models
@st.cache_resource
def load_hf_resnet_model():
    """Loads and caches the Hugging Face DETR-ResNet model.

    This function initializes the DETR-ResNet model for object detection
    and caches it using Streamlit's caching mechanism to avoid reloading
    on each rerun.

    Returns:
        pipeline: A Hugging Face pipeline object for object detection.
            The model used is 'facebook/detr-resnet-50'.

    Raises:
        RuntimeError: If model loading fails.
        ImportError: If required dependencies are not installed.

    Note:
        - Requires 'transformers' and 'torch' packages
        - Model is cached in memory after first load
        - First load might take several minutes
    """
    return pipeline("object-detection", model="facebook/detr-resnet-50")


# Model prediction functions
def predict_huggingface_resnet(image: Image.Image, model: Any) -> int:
    """Predicts the number of people in an image using DETR-ResNet model.

    This function processes an image through the DETR-ResNet model to detect
    and count people. It handles image format conversion and filtering of
    detection results.

    Args:
        image (PIL.Image): Input image in PIL format.
        model: Hugging Face pipeline object for object detection.

    Returns:
        int: Number of people detected in the image. Returns 0 if no people
            are detected or in case of processing errors.

    Raises:
        PIL.Image.Error: If image conversion fails.
        ValueError: If model prediction fails.

    Note:
        - Input image is automatically converted to RGB if needed
        - Only 'person' class detections are counted
        - No minimum confidence threshold is applied
    """
    try:
        # Convert PIL image to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Get predictions
        results = model(image)
        # Count people (assuming 'person' is the label we're looking for)
        return sum(1 for result in results if result['label'] == 'person')
    # Handle errors
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return 0
    