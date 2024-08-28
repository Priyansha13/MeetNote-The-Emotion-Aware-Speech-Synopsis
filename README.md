# Emotion Recognition from Speech and Summarization

This repository contains files for an Emotion Recognition from Speech and Summarization project. The project uses various machine learning models to transcribe speech to text, detect emotions from the transcribed text, and generate summaries of the text. The project is implemented using Python and leverages models such as Whisper, BART, and various transformers for sequence classification.

## Repository Structure

- **MELD dataset**: The dataset used for fine-tuning the models on emotion recognition tasks. Make sure to download the MELD dataset from its [official source](https://github.com/declare-lab/MELD) and place it in the appropriate directory.
- **download_model_new.py**: A script to download pre-trained models from Hugging Face or other repositories, used in the emotion recognition and summarization processes.
- **fine-tuning-bart.ipynb**: A Jupyter Notebook containing the code for fine-tuning the BART model for summarization tasks. This notebook demonstrates how to train the model on custom datasets and evaluate its performance.
- **requirements.txt**: A file containing all the necessary Python libraries and dependencies required to run the project. Install these requirements using `pip install -r requirements.txt`.
- **streamlit.py**: The main script to run the Streamlit application. This script provides a user interface for recording audio, uploading audio files, transcribing speech, detecting emotions, and summarizing text.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Priyansha13/PLLM_Project.git
   cd PLLM_Project

2. **Install dependencies**:
