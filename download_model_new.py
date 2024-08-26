# Import necessary libraries
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM


# Function to download and save the Whisper speech-to-text model
def download_speech_to_text_model():
    processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-tiny")
    processor.save_pretrained("./models/speech-to-text-processor")
    model.save_pretrained("./models/speech-to-text-model")
    print("Speech-to-text model downloaded and saved.")

def download_emotion_prediction_model():
    tokenizer = AutoTokenizer.from_pretrained('j-hartmann/emotion-english-distilroberta-base')
    model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
    tokenizer.save_pretrained("./models/emotion-prediction-tokenizer")
    model.save_pretrained("./models/emotion-prediction-model")
    print("Emotion prediction model downloaded and saved.")

# Function to download and save the summarization model
def download_summarization_model():
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    tokenizer.save_pretrained("./models/summarization-tokenizer")
    model.save_pretrained("./models/summarization-model")
    print("Summarization model downloaded and saved.")


def main():
    download_speech_to_text_model()
    download_emotion_prediction_model()

if __name__ == "__main__":
    main()
