# CROWD COUNTING APPLICATION
An AI-powered application for estimating crowd sizes in images using multiple machine learning models.

This project is only an MVP to showcase the functionality of various ML models. It doesn't cover any thorough error handling and is not production ready.

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

### How to use the application
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
- Individual users: 10 API calls per day
- Total application: 100 API calls per day
- No limits when using custom API credentials

## Sample Images
The application includes various sample images for testing:
- Indoor scenarios (8-50 people)
- Outdoor events (50-500 people)
- Large gatherings (500+ people)
- Different lighting conditions and angles

## Contact
musalek.radim@gmail.com

## Acknowledgments
- Sample images sourced from [Pexels](https://www.pexels.com/)
- Models and APIs:
  - OpenAI 4o-mini API
  - AWS Rekognition API
  - Hugging Face

## LinkedIn post
**TL;DR:** An MVP of a simple crowd counting application using 3 different ML models can be found from the following Streamlit app: https://radimmusalek-crowdcounting.streamlit.app/ . The source code can be found in the following GitHub repository: https://github.com/radimusalek/crowd-counting-app .

**Crowd Counting App revisited**
I've decided to revisit some of my older #machinelearning projects and see how my skills have evolved since then. Of course, besides my skill improvements there's been a few other big changes since my early programming days, mainly:
- the availability of many ready-to-use models via commercial APIs, and even more so
- the coding support from LLMs. In my case, in the recent months I've learned to work with Claude 3.5 Sonnet integrated into the Cursor IDE.

The first project I decided to tackle was the Streamlit Crowd Counting application we had developed as part of our Integrify final project in March 2021. (damn, time flies! :O)

This time, instead of using one pre-trained model downloaded locally, I was able to implement a solution using 3 different models - an open-source pre-trained ResNet model (similar approach to our original implementation) from Hugging Face and then OpenAI 4o-mini and AWS Rekognition models via their commercial APIs. I also added some sample images to showcase the different models and their performance, and since we're using paid commercial APIs, I added a feature to use custom API keys and limited use of my own API keys for the users without their own credentials who would like to try the app quickly too.

Based on the sample images it was interesting to notice quickly that both commercial models have struggled with darker indoor images, especially including larger crowds, while doing relatively well in brighter images with smaller groups of people. If working on the project further, perhaps some more image preprocessing might be needed to improve the outcomes.

To try the app yourself, you can find it here: https://radimmusalek-crowdcounting.streamlit.app/ . The source code is available in the following GitHub repository: https://github.com/radimusalek/crowd-counting-app . If you'd have any comments or suggestions, please reach out to me! I'll be happy to hear any feedback which might help me improve my skills!




