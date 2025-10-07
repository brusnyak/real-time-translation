# Real-Time Speech Translation in Online Conference

## Description
Prototype system translating English ↔ Slovak speech in real time using modern AI models.

## Features
*   Live speech-to-text transcription (Whisper)
*   Neural translation (Helsinki-NLP Opus-MT)
*   Real-time subtitle display (Streamlit)
*   Speech synthesis of translation (pyttsx3)
*   Designed for integration into online conference tools (Jitsi Meet)

## Architecture
(Diagram + module explanations will be added here after research phase)

## Installation
```bash
git clone https://github.com/brusnyak/real-time-translation.git
cd real-time-translation
pip install -r requirements.txt
```

## Usage
To run the offline demo, which records 5 seconds of audio, processes it through the STT -> MT -> TTS pipeline, and then plays the original and translated audio:
```bash
python src/main_pipeline.py
```

## Testing
(Testing instructions will be updated for real-time and integration testing in later phases.)

## Results
(Results will be added here after implementation and testing)
