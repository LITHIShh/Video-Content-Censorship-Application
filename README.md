# Video Content Censorship App

## Overview
The **Video Content Censorship App** is a Python-based application designed to process video files, detect offensive language in the audio track, and replace the offensive words with a beep sound. The application supports both English and Spanish languages for detecting and censoring offensive content. It provides users with a censored version of the video, ensuring that sensitive content is handled appropriately.

---

## Features
- **Language Support**: 
  - Detects offensive content in English and Spanish.
  - Uses predefined offensive word lists and machine learning models for classification.

- **Audio Extraction**:
  - Extracts audio from uploaded MP4 videos for analysis.

- **Noise Reduction**:
  - Reduces background noise in the audio for accurate transcription.

- **Offensive Word Detection**:
  - Utilizes both predefined lists of offensive words and advanced NLP models:
    - `unitary/toxic-bert` for English.
    - `jorgeortizfuentes/spanish-offensive-language-bert-base-spanish-wwm-cased` for Spanish.

- **Censorship**:
  - Replaces offensive words in the audio with a beep sound.
  - Generates a censored video with the modified audio track.

- **User-Friendly Interface**:
  - Intuitive **Streamlit** UI for video uploads and processing.
  - Displays detected offensive words and allows users to download the censored video.

---

## How It Works
1. **Upload Video**: 
   - Users upload an MP4 video file through the Streamlit interface.

2. **Language Selection**:
   - Users select the language of the video content (English or Spanish).

3. **Processing**:
   - The app extracts the audio from the video, reduces background noise, and transcribes it using Whisper.
   - Offensive words are detected using a combination of predefined word lists and NLP models.
   - Detected offensive words are replaced with a beep sound.

4. **Output**:
   - A censored version of the video is generated and displayed in the UI.
   - Users can download the censored video directly.

---

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- FFmpeg (required for audio and video processing)

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/LITHIShh/Video-Content-Censorship-Application.git
   cd video-censorship-app
