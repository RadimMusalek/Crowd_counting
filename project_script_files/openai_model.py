import base64
import io
from openai import OpenAI


def predict_openai(image):
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