import os
import tempfile
import logging
import string
import streamlit as st
from pydub import AudioSegment
from transformers import pipeline
from whisper import load_model
from moviepy.editor import VideoFileClip, AudioFileClip

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load models
english_toxic_classifier = pipeline("text-classification", model="unitary/toxic-bert")
spanish_toxic_classifier = pipeline("text-classification", model="jorgeortizfuentes/spanish-offensive-language-bert-base-spanish-wwm-cased")

# Load offensive words
with open('english_offensive_words.txt', 'r') as f:
    english_offensive_words = [line.strip().lower() for line in f.readlines()]

with open('spanish_offensive_words.txt', 'r') as f:
    spanish_offensive_words = [line.strip().lower() for line in f.readlines()]

def clean_word(word):
    """Clean and normalize a word for comparison."""
    return word.strip().lower().translate(str.maketrans("", "", string.punctuation))

def extract_audio_from_video(video_path):
    """Extract audio from video and save as WAV."""
    video = VideoFileClip(video_path)
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    video.audio.write_audiofile(temp_audio.name, codec='pcm_s16le')
    video.close()
    return temp_audio.name

def reduce_noise(audio_path):
    """Reduce background noise in audio."""
    audio = AudioSegment.from_file(audio_path)
    normalized_audio = audio.normalize()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    normalized_audio.export(temp_file.name, format="wav")
    return temp_file.name

def transcribe_with_timestamps(audio_path):
    """Transcribe audio and get word-level timestamps."""
    model = load_model("base")
    logging.info("Starting transcription using Whisper.")
    
    try:
        result = model.transcribe(audio_path, word_timestamps=True)
        logging.info("Transcription complete")
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return "", []

    word_segments = []
    for segment in result["segments"]:
        for word in segment["words"]:
            word_segments.append({
                "word": clean_word(word["word"]),
                "start": word["start"],
                "end": word["end"]
            })

    return result["text"], word_segments

def classify_text_for_hate_speech(text, language):
    """Classify text for toxic content based on the language."""
    if language == "english":
        classifier = english_toxic_classifier
    else:
        classifier = spanish_toxic_classifier
    
    logging.info(f"Classifying text: {text}")
    result = classifier(text)[0]
    return result

def create_censored_audio(audio_path, word_timestamps, language, beep_path):
    """Create censored audio by replacing toxic words with beep."""
    logging.info("Creating censored audio")
    audio = AudioSegment.from_file(audio_path)
    beep = AudioSegment.from_file(beep_path)
    censored_audio = audio

    detected_words = []  # List to store detected words for debugging
    offensive_words_list = spanish_offensive_words if language == "spanish" else english_offensive_words

    for word_info in word_timestamps:
        word = word_info["word"]
        start_ms = int(word_info["start"] * 1000)
        end_ms = int(word_info["end"] * 1000)
        duration = end_ms - start_ms

        # Classify English words using both predefined list and model
        if language == "english":
            # Check if word is in predefined list of offensive words
            if word in english_offensive_words:
                detected_words.append(word)  # Log detected words
                logging.info(f"Detected offensive word (list): {word} at {start_ms} ms")
            else:
                # Classify word using the model
                result = classify_text_for_hate_speech(word, "english")
                if result["label"] == "toxic" and result["score"] > 0.7:
                    detected_words.append(word)  # Log detected words
                    logging.info(f"Detected offensive word (model): {word} at {start_ms} ms")
        # Classify Spanish words using predefined list
        elif language == "spanish" and word in offensive_words_list:
            detected_words.append(word)  # Log detected words
            logging.info(f"Detected offensive word: {word} at {start_ms} ms")

        # If the word is detected as offensive, censor it
        if word in detected_words:
            # Adjust beep duration to match word duration
            adjusted_beep = beep
            if duration > len(beep):
                adjusted_beep = beep * (duration // len(beep) + 1)
            adjusted_beep = adjusted_beep[:duration]

            censored_audio = (
                censored_audio[:start_ms]
                + adjusted_beep
                + censored_audio[end_ms:]
            )

    logging.info(f"Detected words for censorship: {detected_words}")
    temp_censored_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    censored_audio.export(temp_censored_audio.name, format="wav")
    return temp_censored_audio.name, detected_words

def create_censored_video(video_path, censored_audio_path):
    """Create final video with censored audio."""
    logging.info("Creating final video with censored audio")
    video = VideoFileClip(video_path)
    censored_audio = AudioFileClip(censored_audio_path)
    
    final_video = video.without_audio().set_audio(censored_audio)
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    
    final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')
    
    # Clean up
    video.close()
    censored_audio.close()
    final_video.close()
    
    return output_path

def main():
    st.title("Video Content Censorship App")
    st.write("Upload an MP4 video file to detect and censor offensive content.")
    
    # Language selection
    language = st.selectbox("Select Language:", ("English", "Spanish")).lower()

    # File uploader
    video_file = st.file_uploader("Upload MP4 video:", type=["mp4"])

    if video_file:
        try:
            # Save uploaded video
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                temp_video.write(video_file.read())
                video_path = temp_video.name

            # Process video
            with st.spinner("Processing video..."):
                audio_path = extract_audio_from_video(video_path)
                processed_audio_path = reduce_noise(audio_path)

                # Get beep sound path
                beep_path = "beep.mp3"
                if not os.path.exists(beep_path):
                    st.error("Beep sound file (beep.mp3) not found!")
                    return

                # Transcribe audio
                transcribed_text, word_timestamps = transcribe_with_timestamps(processed_audio_path)
                st.write("#### Transcribed Text:")
                st.text(transcribed_text)

                # Create censored audio
                censored_audio_path, detected_words = create_censored_audio(
                    processed_audio_path,
                    word_timestamps,
                    language,
                    beep_path
                )

                # Display detected words
                if detected_words:
                    st.write("#### Detected Offensive Words:")
                    st.write(", ".join(detected_words))
                else:
                    st.write("No offensive words detected.")

                # Create final video
                censored_video_path = create_censored_video(video_path, censored_audio_path)

                # Display and provide download
                st.success("Processing complete!")
                st.video(censored_video_path)
                
                with open(censored_video_path, "rb") as f:
                    st.download_button(
                        "Download Censored Video",
                        f.read(),
                        file_name="censored_video.mp4",
                        mime="video/mp4"
                    )

                # Cleanup temporary files
                for path in [video_path, audio_path, processed_audio_path, 
                             censored_audio_path, censored_video_path]:
                    if os.path.exists(path):
                        os.unlink(path)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logging.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
