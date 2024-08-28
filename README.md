# Emotion Recognition from Speech and Summarization

This repository contains files for an Emotion Recognition from Speech and Summarization project. The project uses various machine learning models to transcribe speech to text, detect emotions from the transcribed text, and generate summaries of the text. The project is implemented using Python and leverages models such as Whisper, BART, and various transformers for sequence classification.

## Repository Structure

- **MELD dataset**: The dataset used for fine-tuning the models on emotion recognition tasks. Make sure to download the MELD dataset from its [official source](https://github.com/declare-lab/MELD) and place it in the appropriate directory.
- **download_model_new.py**: A script to download pre-trained models from Hugging Face or other repositories, used in the emotion recognition and summarization processes.
- **fine-tuning-bart.ipynb**: A Jupyter Notebook containing the code for fine-tuning the BART model for summarization tasks. This notebook demonstrates how to train the model on custom datasets and evaluate its performance.
- **requirements.txt**: A file containing all the necessary Python libraries and dependencies required to run the project. Install these requirements using `pip install -r requirements.txt`.
- **streamlit.py**: The main script to run the Streamlit application. This script provides a user interface for recording audio, uploading audio files, transcribing speech, detecting emotions, and summarizing text.

## Project Workflow

1. **Audio Input**: Users can upload an audio file or record audio directly in the application.
2. **Transcription**: 
   - The Whisper model processes the audio and generates a text transcription.
   - The transcription is displayed in the app.
3. **Emotion Detection**: 
   - Distil-RoBERTa analyzes the transcribed text to detect the underlying emotion.
   - The detected emotion is displayed in the app.
4. **Summarization**:
   - The transcription is summarized using the BART model to provide a concise version of the content.
   - The summary is displayed in the app.

## Files in the Repository

- **`download_model_new.py`**: Script to download the necessary models (Whisper, Distil-RoBERTa, and BART).
- **`fine-tuning-bart.ipynb`**: Jupyter Notebook for fine-tuning the BART model on a summarization dataset.
- **`requirements.txt`**: List of required Python packages for the project.
- **`streamlit.py`**: Main application file for running the Streamlit app that performs speech-to-text transcription, emotion detection, and summarization.

## Getting Started

### Prerequisites

Ensure you have Python installed (preferably Python 3.7 or higher). Install the required dependencies using pip:

bash
pip install -r requirements.txt

--Running the Project
Download the Models: Run download_model_new.py to download the necessary models.

bash
python download_model_new.py

- Run the Streamlit App: Start the Streamlit app by running the following command:
- 
bash
streamlit run streamlit.py
- Upload or Record Audio: Use the app interface to upload an audio file or record audio directly.

Transcribe, Detect Emotion, and Summarize: Follow the app prompts to transcribe the audio, detect the emotion, and summarize the text.

Future Work
- Multimodal Emotion Detection: Extend the project to include visual emotion detection using the video data in the MELD dataset.
- Real-Time Processing: Enhance the application to handle real-time audio input and processing.
- Additional Languages: Expand the transcription and emotion detection capabilities to support multiple languages.
