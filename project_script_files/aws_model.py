"""
AWSRekognition Module for Crowd Counting
This module provides functionality to detect and count people in images using
AWS Rekognition service. It handles image preprocessing, resizing for AWS limits,
and person detection.
"""
import io
import math
from PIL import Image
import boto3
from project_script_files.api_credentials import APICredentialsManager


def predict_aws(image: Image.Image, rekognition_client: boto3.client, api_credentials_manager: APICredentialsManager) -> int:
    """Predicts the number of people in an image using AWS Rekognition.

    This function processes an image for AWS Rekognition service, including
    necessary format conversions and size adjustments. It then uses the
    Rekognition API to detect people in the image.

    Args:
        image (PIL.Image): Input image in PIL format.
        rekognition_client: Boto3 Rekognition client instance.

    Returns:
        int: Number of people detected in the image. Returns 0 if no people
            are detected or in case of processing errors.

    Raises:
        PIL.Image.Error: If image conversion fails.
        boto3.exceptions.Boto3Error: If AWS Rekognition API call fails.

    Note:
        - Image must be less than 5MB after JPEG conversion
        - Supported input formats include RGB and RGBA
        - Image will be automatically resized if it exceeds AWS size limits
    """
    # Convert PIL image to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Calculate current size in bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    current_size = len(img_byte_arr.getvalue())

    # Resize if needed
    if current_size > 5*1024*1024:  # If larger than 5MB
        # Calculate new size maintaining aspect ratio
        ratio = math.sqrt(5*1024*1024 / current_size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        
        # Resize image
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    # Convert to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG', quality=95)
    img_byte_arr = img_byte_arr.getvalue()
    
    # Call AWS Rekognition
    response = rekognition_client.detect_labels(
        Image={'Bytes': img_byte_arr},
        MaxLabels=100
    )
    
    # Find 'Person' label and get instance count
    for label in response['Labels']:
        if label['Name'] == 'Person':
            return len(label['Instances'])
    return 0
