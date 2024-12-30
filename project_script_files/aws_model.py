import io
import math
from PIL import Image


def predict_aws(image, rekognition_client):
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