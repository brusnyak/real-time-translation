# Real-Time Speech Translation with Web UI

This project provides a real-time speech translation pipeline with a web-based user interface. It allows users to transcribe speech from their microphone, translate it into a target language, and then synthesize the translated text back into speech, which is outputted to both speakers and a virtual audio device (BlackHole 2ch for Google Meet integration). The UI provides real-time feedback on transcription, translation, and audio input levels, along with controls to dynamically change input/output languages and TTS models.

## Architecture

The system consists of a Python backend powered by FastAPI and a frontend built with HTML, CSS, and JavaScript.

### Backend (`server.py`)

- **FastAPI**: Serves the web UI and handles WebSocket communication for real-time audio streaming and data exchange.
- **Speech-to-Text (STT)**: Uses `faster-whisper` for efficient audio transcription.
- **Machine Translation (MT)**: Leverages `MarianMT` from the `transformers` library for language translation.
- **Text-to-Speech (TTS)**: Utilizes `Coqui TTS` (specifically XTTS v2) for speech synthesis, including voice cloning capabilities.
- **Voice Activity Detection (VAD)**: Implemented with `webrtcvad` to detect speech segments and optimize processing.
- **Audio I/O**: Handled by `sounddevice` and `soundfile` for microphone input and output to speakers/BlackHole.

### Frontend (`ui/`)

- **HTML (`ui/index.html`)**: Defines the structure of the user interface, including input/output language selectors, TTS model selector, transcription/translation display areas, audio input level indicator, and control buttons.
- **CSS (`ui/style.css`)**: Styles the UI for a modern and responsive look.
- **JavaScript (`ui/script.js`)**: Manages UI interactions, captures microphone audio, sends audio data to the backend via WebSocket, receives and displays real-time transcription and translation, and updates UI elements like the audio level bar and state indicators.

## Setup Instructions

To get the project up and running, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone git@github.com-brusnyak:brusnyak/real-time-translation.git
    cd real-time-translation
    ```

2.  **Create and activate a Python virtual environment:**
    \*ensure you have python 3.10 installed.

    It's highly recommended to use a virtual environment to manage dependencies.

    ```bash
    python3 -m venv venv-coqui
    source venv-coqui/bin/activate
    ```

3.  **Install dependencies:**
    Install all required Python packages using `pip`.

    ```bash
    pip install -r requirements_coqui_min.txt
    ```

    _Note: If you encounter issues with `onnxruntime-silicon` on Apple Silicon Macs, you might need to install it via `conda` or ensure your `onnxruntime` version is compatible._

4.  **Install BlackHole 2ch (macOS only, optional but recommended for Google Meet integration):**
    BlackHole is a virtual audio driver that allows you to route audio between applications.

    - Download and install BlackHole 2ch from [Existential Audio](https://github.com/ExistentialAudio/BlackHole/releases).
    - Follow the installation instructions provided by BlackHole.

5.  **Prepare SSL Certificates:**
    The FastAPI server runs with HTTPS. You'll need `cert.pem` and `key.pem` files in the project root. You can generate self-signed certificates for local development:
    ```bash
    openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365
    # When prompted, enter 'localhost' for Common Name (CN)
    ```

## Running the Application

1.  **Start the FastAPI server:**
    Ensure your virtual environment is activated.

    ```bash
    source venv-coqui/bin/activate
    python server.py
    ```

    The server will start on `https://127.0.0.1:8000`.

2.  **Access the Web UI:**
    Open your web browser and navigate to `https://127.0.0.1:8000`.
    - You might encounter a "Your connection is not private" warning due to the self-signed certificate. You can safely proceed by clicking "Advanced" and then "Proceed to 127.0.0.1 (unsafe)".

## UI Controls and Functionality

- **Initialize Button**: Click this first to load the necessary ML models in the backend. This may take a minute or two on the first run.
- **Start/Stop Buttons**: Control the real-time audio processing. "Start" begins capturing microphone input, transcribing, translating, and synthesizing speech. "Stop" halts the process.
- **Mic Input Level**: A visual indicator showing your microphone's audio activity.
- **Input Language Selector**: Choose the language your speech will be transcribed from.
- **Output Language Selector**: Select the language for translation and speech synthesis.
- **TTS Model Selector**: Choose between available Text-to-Speech models (e.g., XTTS v2, Glow-TTS).
- **Transcription Box**: Displays the real-time transcription of your speech.
- **Translation Box**: Shows the real-time translation of the transcribed text.
- **Metrics**: Displays processing times for STT, MT, TTS, and total latency.
- **State Indicators**: Icons (Listening, Translating, Speaking) light up to show the current stage of the pipeline.

## Important Note (XTTS v2)

XTTS v2 lacks native Slovak voice cloning. For Slovak output, we synthesize using Czech in XTTS v2 to achieve better voice similarity. This is an intentional design choice documented in `plan.md`.
