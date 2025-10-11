# Project Plan: Real-time Voice-Cloned Translation via Virtual Audio Device

This plan outlines the development of a prototype system for real-time speech translation with voice cloning, designed to integrate seamlessly into online conference environments via a virtual audio device. The focus is on a robust backend solution, optimizing for latency and voice cloning quality, and providing a simple frontend for control and optional subtitle display.

## Overall Goal

To create a prototype system that ensures real-time speech translation between English and another selected language during online meetings, with the user's voice cloned, outputting to a virtual audio device for other participants, and optionally displaying subtitles. This system will prioritize backend functionality and real-time performance over complex video synchronization.

## Phase 1: Backend Development (Python Real-time Translation Server with Voice Cloning)

*   **Goal:** Adapt the existing `PipelineOrchestrator` for real-time audio streaming, integrate voice cloning, and output translated audio to a virtual device. This will be the core "backend solution."
*   **Key Features:**
    *   Real-time processing of audio chunks.
    *   WebSocket API for communication with the frontend.
    *   Efficient voice cloning using XTTS_V2.
    *   Output of translated audio to a virtual audio device.
    *   Optimized latency across all modules (STT, MT, TTS).
*   **Steps:**
    1.  **Refactor `PipelineOrchestrator` for Streaming:** Modify `PipelineOrchestrator` to accept and process audio in small, continuous chunks (e.g., 1-2 seconds) from a real-time audio input stream. Implement efficient buffering and asynchronous processing.
    2.  **Implement a WebSocket Server:** Create a `FastAPI` application with a WebSocket endpoint. This server will:
        *   Receive raw audio bytes streamed from the frontend.
        *   Process these chunks through STT, MT, and TTS modules.
        *   Send back translated text (for subtitles) and/or translated audio bytes to the frontend.
        *   Handle API calls for language selection and triggering initial model preloading.
    3.  **Voice Cloning Integration & Optimization:** Ensure the TTS module (`tts_module.py`) is correctly configured and optimized for voice cloning using the speaker reference. Focus on the quality and speed of the cloned voice.
    4.  **Virtual Audio Device Output:** Integrate `pyaudio` or `sounddevice` into the Python backend to continuously output the synthesized translated audio to a **virtual audio device** (e.g., BlackHole for macOS, VB-Cable for Windows). This will be the primary audio output for other conference participants.
    5.  **Latency Optimization:** Profile and optimize each stage (STT, MT, TTS) to reduce overall latency.

## Phase 2: Frontend Development (Minimal UI for Control and Subtitles)

*   **Goal:** Provide a simple interface to control the translation process (start/stop, language selection) and optionally display real-time subtitles. This fulfills the "simple user interface for testing the prototype" requirement.
*   **Key Features:**
    *   Microphone audio capture.
    *   WebSocket communication with the backend.
    *   Real-time display of translated text.
    *   Language selection and start/stop controls.
*   **Steps:**
    1.  **Develop Basic Web Application:** Create a simple HTML/CSS/JavaScript web page.
    2.  **Microphone Input:** Use the Web Audio API and `navigator.mediaDevices.getUserMedia` to capture audio from the user's microphone.
    3.  **Audio Chunking & WebSocket Communication:** Process the captured audio stream, chunk it into small segments, and send these chunks via WebSocket to the Python backend.
    4.  **Display Translated Text (Subtitles):** Show the real-time translated text as subtitles within the UI.
    5.  **Play Synthesized Translated Speech (Locally):** Play the synthesized translated speech back to the user *locally* through their headphones (for their own monitoring).
    6.  **Status Display:** Provide visual feedback on the system's status (e.g., "Loading models...", "Listening...", "Translating...").

## Phase 3: Integration & Verification (Conference Environment)

*   **Goal:** Demonstrate the system's functionality in an online conference setting and analyze its performance.
*   **Key Features:**
    *   Compatibility with various conference platforms.
    *   Translated voice output to other participants.
    *   Analysis of accuracy, latency, and voice cloning quality.
*   **Steps:**
    1.  **Virtual Audio Device Setup:** The user installs a virtual audio device (e.g., BlackHole 2ch).
    2.  **Conference Software Configuration:** The user configures their conference software (Google Meet, Zoom, Jitsi, etc.) to use this virtual audio device as their microphone input.
    3.  **Translated Voice Output:** When the user speaks, their voice is translated and outputted through the virtual audio device. Other participants in the conference will hear the *translated speech in the cloned voice* coming from the user.
    4.  **Subtitle Display (if implemented):** If the frontend includes subtitle display, the user can share their screen to show these subtitles to other participants.
    5.  **Analysis:** Test and analyze the accuracy, latency, and voice cloning quality of the translation in a live conference setting.

---

**Current Bottlenecks and Optimization Focus:**

Based on the latest test run:

*   **Critical Issue:** High TTS latency (`30.866s`) and "index out of range" errors during synthesis. This must be addressed first for the pipeline to function correctly.
*   **Primary Optimization Target:** Machine Translation (MT) time (`3.233s`). The goal is to get this under 1 second.
*   **Voice Cloning Quality:** Ensure the XTTS_V2 model produces high-quality, natural-sounding cloned voices.




