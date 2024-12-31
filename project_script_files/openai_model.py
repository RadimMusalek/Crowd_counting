"""
OpenAI VisionModule for Crowd Counting
This module provides functionality to detect and count people in images using
OpenAI's GPT-4o-mini model. It handles image preprocessing, base64 encoding,
and API communication for crowd counting tasks.
Typical usage example:
   image = Image.open('crowd.jpg')
   count = predict_openai(image)
"""
import base64
import io
from openai import OpenAI, OpenAIError
from PIL import Image

def predict_openai(image: Image.Image) -> int:
    """Predicts the number of people in an image using OpenAI's 4o-mini model.
    This function processes an image for OpenAI's Vision API, including
    necessary format conversions and base64 encoding. It then uses the API
    to analyze the image and extract a numerical count of people.
    Args:
        image (PIL.Image): Input image in PIL format.
    Returns:
        int: Number of people detected in the image. Returns 0 if no people
            are detected or in case of processing errors.
    Raises:
        PIL.Image.Error: If image conversion fails.
        openai.OpenAIError: If API call fails or returns unexpected response.
        ValueError: If response cannot be converted to integer.
    Note:
        - Requires valid OpenAI API key in environment variables
        - Image is converted to JPEG format for API compatibility
        - Response is expected to be a numerical value
        - Model used is 'gpt-4o-mini' for vision tasks
    """
    try:
        # Initialize OpenAI client
        client = OpenAI()  # It will use the API key from environment variables

        # Convert PIL image to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert PIL Image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "How many people are in this image? Please respond with just a number."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ]
        )

        # Extract the number from the response
        try:
            count = int(response.choices[0].message.content.strip())
        except:
            count = 0
        return count
    except OpenAIError as e:
        print(f"OpenAI API error: {str(e)}")
        return 0
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return 0