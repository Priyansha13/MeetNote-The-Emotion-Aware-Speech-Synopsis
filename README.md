Emotion Recognition from Speech and Summarization
This repository contains files for an Emotion Recognition from Speech and Summarization project. The project uses various machine learning models to transcribe speech to text, detect emotions from the transcribed text, and generate summaries of the text. The project is implemented using Python and leverages models such as Whisper, BART, and various transformers for sequence classification.

Repository Structure
MELD dataset: The dataset used for fine-tuning the models on emotion recognition tasks. Make sure to download the MELD dataset from its official source and place it in the appropriate directory.
download_model_new.py: A script to download pre-trained models from Hugging Face or other repositories, used in the emotion recognition and summarization processes.
fine-tuning-bart.ipynb: A Jupyter Notebook containing the code for fine-tuning the BART model for summarization tasks. This notebook demonstrates how to train the model on custom datasets and evaluate its performance.
requirements.txt: A file containing all the necessary Python libraries and dependencies required to run the project. Install these requirements using pip install -r requirements.txt.
streamlit.py: The main script to run the Streamlit application. This script provides a user interface for recording audio, uploading audio files, transcribing speech, detecting emotions, and summarizing text.
Installation
Clone the Repository:

bash
Copy code
git clone https://github.com/yourusername/yourrepositoryname.git
cd yourrepositoryname
Install Dependencies:

Ensure you have Python 3.7 or above installed. Then, install the required packages:

bash
Copy code
pip install -r requirements.txt
Download Models:

Use the download_model_new.py script to download the pre-trained models:

bash
Copy code
python download_model_new.py
Set Up the MELD Dataset:

Download the MELD dataset from here and place it in the directory structure as needed by your project.

Running the Project
Streamlit Application
To run the Streamlit application, use the following command:

bash
Copy code
streamlit run streamlit.py
This will open a web-based user interface where you can:

Record or upload audio files.
Transcribe speech to text.
Detect emotions from the transcribed text.
Generate a summary of the transcription.
Jupyter Notebook
To fine-tune the BART model, open fine-tuning-bart.ipynb in Jupyter Notebook or Jupyter Lab:

bash
Copy code
jupyter notebook fine-tuning-bart.ipynb
Follow the steps in the notebook to fine-tune the model on your data.

Key Features
Speech-to-Text: Utilizes the Whisper model for accurate speech transcription.
Emotion Detection: Employs Distil-RoBERTa model to classify the detected emotions from the text.
Text Summarization: Uses a fine-tuned BART model to generate concise summaries of the transcriptions.
