import os
import torch
import soundfile as sf
import sounddevice as sd
import time
import librosa
import streamlit as st
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM

# Load models and tokenizers
@st.cache_resource
def load_models():
    st.write("Loading models...")
    speech_to_text_model = WhisperForConditionalGeneration.from_pretrained("./models/speech-to-text-model")
    speech_to_text_processor = WhisperProcessor.from_pretrained("./models/speech-to-text-processor")

    emotion_model = AutoModelForSequenceClassification.from_pretrained("./models/emotion-prediction-model")
    emotion_tokenizer = AutoTokenizer.from_pretrained("./models/emotion-prediction-tokenizer")

    summarization_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    summarization_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

    return speech_to_text_model, speech_to_text_processor, emotion_model, emotion_tokenizer, summarization_model, summarization_tokenizer

# Record audio from the microphone
def record_audio(duration=5, samplerate=16000):
    st.write("Recording...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    return recording, samplerate

# Save audio file in .flac format
def save_audio(recording, samplerate, output_file_path="recorded_audio.flac"):
    sf.write(output_file_path, recording, samplerate, format='FLAC')
    return output_file_path

# Resample audio to 16000 Hz
def resample_audio(audio_file_path, target_sr=16000):
    audio_input, sr = librosa.load(audio_file_path, sr=None)  # Load the audio file with its original sampling rate
    audio_input_resampled = librosa.resample(audio_input, orig_sr=sr, target_sr=target_sr)  # Resample to target_sr
    return audio_input_resampled, target_sr

# Predict speech to text
def predict_speech_to_text(audio_file_path, speech_to_text_model, speech_to_text_processor):
    audio_input, samplerate = resample_audio(audio_file_path)  # Ensure audio is resampled to 16000 Hz
    input_features = speech_to_text_processor(audio_input, return_tensors="pt", sampling_rate=samplerate).input_features
    forced_decoder_ids = speech_to_text_processor.get_decoder_prompt_ids(language="en")
    predicted_ids = speech_to_text_model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    transcription = speech_to_text_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

# Predict emotion
def predict_emotion(text, emotion_model, emotion_tokenizer):
    emotion_mapping = {
        "anger": "anger",
        "neutral": "neutral",
        "sadness": "sadness",
        "joy": "joy",
        "fear": "sadness",
        "disgust": "anger",
        "surprise": "joy"
    }
    inputs = emotion_tokenizer(text, return_tensors="pt")
    predictions = emotion_model(**inputs).logits
    predicted_class_id = torch.argmax(predictions, axis=-1).item()
    emotion = emotion_model.config.id2label[predicted_class_id]
    return emotion_mapping.get(emotion, "neutral")

# Summarize transcription text
def summarize_text(text, summarization_model, summarization_tokenizer):
    inputs = summarization_tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = summarization_model.generate(inputs["input_ids"], max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Main Streamlit app
def main():
    st.title("Emotion Recognition and Summarization from Speech")

    # Load models
    speech_to_text_model, speech_to_text_processor, emotion_model, emotion_tokenizer, summarization_model, summarization_tokenizer = load_models()

    option = st.selectbox("Choose an option:", ["Record Audio", "Upload Audio File", "Exit"])

    if option == "Record Audio":
        duration = st.slider("Select recording duration (seconds):", min_value=1, max_value=15, value=5)
        if st.button("Start Recording"):
            recording, samplerate = record_audio(duration)
            audio_file_path = save_audio(recording, samplerate)
            st.write(f"Audio recorded and saved to {audio_file_path}")

            transcription = predict_speech_to_text(audio_file_path, speech_to_text_model, speech_to_text_processor)
            st.write(f"Transcription: {transcription}")

            emotion = predict_emotion(transcription, emotion_model, emotion_tokenizer)
            st.write(f"Detected Emotion: {emotion}")

            summary = summarize_text(transcription, summarization_model, summarization_tokenizer)
            st.write(f"Summary: {summary}")

    elif option == "Upload Audio File":
        uploaded_file = st.file_uploader("Choose a .flac audio file", type="flac")
        if uploaded_file is not None:
            audio_file_path = os.path.join("uploaded_audio.flac")
            with open(audio_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            transcription = predict_speech_to_text(audio_file_path, speech_to_text_model, speech_to_text_processor)
            st.write(f"Transcription: {transcription}")

            emotion = predict_emotion(transcription, emotion_model, emotion_tokenizer)
            st.write(f"Detected Emotion: {emotion}")

            summary = summarize_text(transcription, summarization_model, summarization_tokenizer)
            st.write(f"Summary: {summary}")

    elif option == "Exit":
        st.write("Exiting...")

if __name__ == "__main__":
    main()
