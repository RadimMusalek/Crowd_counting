# CROWD COUNTING APPLICATION
An AI-powered application for estimating crowd sizes in images using multiple machine learning models.

## Overview
This application provides crowd counting functionality using three different AI models:
- OpenAI 4o-mini (paid API)
- AWS Rekognition (paid API)
- Hugging Face ResNet Model (open source)

Users can either upload their own images or choose from provided sample images to estimate crowd sizes in various scenarios.

## Features
- **Multiple AI Models**: Choose between three different models for crowd estimation
- **Sample Images**: Pre-loaded sample images covering various scenarios:
  - Concert & festival crowds
  - Indoor meetings
  - Groups without visible faces
  - Clear images with removed background
- **Custom Image Upload**: Support for user image uploads (JPG, JPEG, PNG)
- **API Management**: 
  - Option to use default or custom API credentials
  - Built-in API usage limits for default credentials
  - Usage tracking and statistics

## Setup
This project uses **Python 3.12.3**. We recommend using a virtual environment for development.

```bash
# Create and activate virtual environment
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### API Credentials
The application requires API credentials for OpenAI and AWS services. You can either:
1. Use the default credentials (subject to usage limits)
2. Provide your own API credentials in the application

To use your own credentials, create a `.env` file:
```
OPENAI_API_KEY=your_openai_key
AWS_ACCESS_KEY_ID=your_aws_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=your_aws_region
HUGGINGFACE_TOKEN=your_hf_token
```

## Usage
### Run locally
Run the Streamlit application:
```bash
streamlit run main.py
```

### Run on Streamlit Cloud
Application hosted on [Streamlit Cloud](https://radimmusalek-crowdcounting.streamlit.app/)

1. Select a model from the dropdown menu
2. Either:
   - Upload your own image
   - Choose from available sample images
3. Click "Run Analysis" to get the crowd estimate

## Project Structure
```
├── __pycache__  # gitignored
├── notebooks
│   └── dev_notebook.ipynb
├── project_script_files/
│   ├── api_credentials.py  # API credential management
│   ├── api_limits.py      # API usage limiting
│   ├── aws_model.py       # AWS Rekognition implementation
│   ├── openai_model.py    # OpenAI Vision implementation
│   ├── HF_model.py        # Hugging Face model implementation
│   ├── streamlit_page_setup.py  # UI components
│   └── utils.py           # Utility functions
├── .env  # gitignored
├── .gitignore  # gitignored
├── api_usage.json  # API usage tracking
├── configs.py  # Configuration file
├── constants.py  # Constant values
├── sample_images/         # Sample image directory
│   ├── concert_back_14.jpg
│   ├── indoor_clear_group_40.jpg
│   ├── indoor_clear_group_8.jpg
│   ├── indoor_concert_1000.jpg
│   ├── indoor_concert_dark_30.jpg
│   ├── indoor_concert_dark_40.jpg
│   ├── indoor_meeting_6.jpg
│   ├── indoor_meeting_7.jpg
│   ├── indoor_meeting_aerial_8.jpg
│   ├── indoor_meeting_hands_6.jpg
│   ├── outdoor_clear_group_10.jpg
│   ├── outdoor_concert_200.jpg
│   └── outdoor_festival_30.jpg
├── main.py                 # Main Streamlit application
├── README.md               # Readme file with project overview
└── requirements.txt        # Project dependencies

```

## API Usage Limits
When using default credentials:
- Individual users: 5 API calls per day
- Total application: 100 API calls per day
- No limits when using custom API credentials

## Sample Images
The application includes various sample images for testing:
- Indoor scenarios (8-50 people)
- Outdoor events (50-500 people)
- Large gatherings (500+ people)
- Different lighting conditions and angles

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
[Your chosen license]

## Contact
musalek.radim@gmail.com

## Acknowledgments
- Sample images sourced from [Pexels](https://www.pexels.com/)
- Models and APIs:
  - OpenAI 4o-mini API
  - AWS Rekognition API
  - Hugging Face
