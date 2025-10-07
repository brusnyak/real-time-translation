# Project Plan: Real-Time Speech Translation in Online Conferences

This document outlines the detailed plan for the Bachelor's thesis project, "Real-Time Speech Translation in Online Conferences," covering research, implementation, and documentation phases.

## 1. Project Overview

**Goal:** To develop a prototype system that translates live speech between English ↔ Slovak in near real-time during an online meeting, ideally preserving the original speaker's voice characteristics, or at least providing high-quality synthesized speech.

**Key Requirements:**

- **Reliability:** Robust performance under various conditions.
- **Efficiency & Speed:** Low latency for real-time translation (target < 2-3 seconds end-to-end).
- **Simplicity:** Maintainable and understandable codebase.
- **Cost-effectiveness:** Utilize free/open-source tools where possible.
- **Integration:** Focus on Jitsi Meet for conference integration.

## 2. Research Phase (Part 1)

**Objective:** To confirm the suitability of selected tools, understand their real-time implications, and research integration with Jitsi Meet.

**Selected Tools:**

- **Speech-to-Text (STT):** OpenAI Whisper (local model)
- **Machine Translation (MT):** Helsinki-NLP (Opus-MT)
- **Text-to-Speech (TTS):** pyttsx3

**Research Tasks:**

1.  **Tool Suitability & Real-time Capabilities:**
    - Investigate OpenAI Whisper: capabilities, licensing, reported performance for English ↔ Slovak, real-time streaming/chunking support.
    - Investigate Helsinki-NLP Opus-MT: capabilities, licensing, reported performance for English ↔ Slovak, low-latency translation methods.
    - Investigate Coqui TTS: capabilities, licensing, reported quality for English ↔ Slovak, real-time synthesis, potential for voice cloning (as a stretch goal).
2.  **Pipeline Management:**
    - Research strategies for real-time data processing (e.g., `asyncio`, `threading`, `WebSockets`) to minimize latency across the STT → MT → TTS pipeline.
3.  **Jitsi Meet Integration:**
    - Explore Jitsi Meet SDK and WebRTC capabilities for injecting translated audio or displaying real-time subtitles.

**Output:**

- `decisions_log.md` (documenting research findings and decisions)
- Refined architecture diagram and explanation.
- Research Summary Report (2-3 pages for supervisor approval).

## 3. Refined Architecture and Integration Strategy

Based on the research, the proposed architecture and integration strategy are as follows:

### Architecture Overview

The system will consist of a Python backend handling the core STT, MT, and TTS functionalities, and a JavaScript/HTML frontend for the user interface and Jitsi Meet integration.

```mermaid
graph TD
    A[User Speech Input (Mic)] --> B(Audio Input Module - Python)
    B --> C(Speech-to-Text - Whisper)
    C --> D(Text Translation - Opus-MT)
    D --> E(Text-to-Speech - Coqui TTS)
    E --> F{Audio Output / Subtitle Display}
    F --> G[Jitsi Meet Frontend - JavaScript/HTML]
    G --> H[User (Translated Audio / Subtitles)]
```

**Module Explanations:**

- **Audio Input Module (Python):** Captures live microphone audio in chunks.
- **Speech-to-Text (Whisper):** Transcribes audio chunks into text. Will use a multilingual model (e.g., `medium` or `large`) for better performance across English and Slovak.
- **Text Translation (Opus-MT):** Translates the transcribed text from source to target language. Specific `opus-mt-en-sk` and `opus-mt-sk-en` models will be used.
- **Text-to-Speech (pyttsx3):** Synthesizes the translated text into speech. `pyttsx3` will be used for its offline capabilities and compatibility with Python 3.13. Voice cloning will be considered a stretch goal, depending on the capabilities of `pyttsx3`'s underlying engines or potential integrations.
- **Jitsi Meet Frontend (JavaScript/HTML):** Embeds the Jitsi Meet conference using the IFrame API. This layer will handle:
  - Receiving translated text for subtitle display.
  - Potentially injecting synthesized audio into the conference (requires further investigation into Jitsi's audio API).
  - User controls for starting/stopping translation and language selection.

### Real-time Pipeline Management

- **Chunking:** Audio will be processed in small, fixed-size chunks to minimize latency.
- **Asynchronous Processing:** Python's `asyncio` or `threading` will be used to manage the concurrent execution of STT, MT, and TTS modules, ensuring the pipeline remains responsive.
- **Communication:** WebSockets will be the primary mechanism for real-time communication between the Python backend (sending translated text/audio) and the JavaScript frontend (receiving and displaying/playing translations).

### Jitsi Meet Integration Strategy

- **Embedding:** Jitsi Meet will be embedded into a custom web application using its IFrame API.
- **Subtitle Overlay:** The translated text will be sent from the Python backend to the JavaScript frontend via WebSockets. The frontend will then display these subtitles as an overlay within or alongside the Jitsi Meet iframe.
- **Audio Injection (Stretch Goal):** Investigate Jitsi's IFrame API `devices` options or other WebRTC mechanisms to inject the synthesized audio directly into the conference. This might involve creating a "translation bot" participant or modifying an existing participant's audio stream.
- **User Interface:** A web-based UI (e.g., Flask with HTML/CSS/JS, or Streamlit if a simpler, standalone app is preferred) will be developed to host the Jitsi iframe and provide controls for the translation system.

## 4. Development Pipeline (Part 2 - Implementation Phase)

**Objective:** To build a working prototype system based on the research findings.

### Phase 1 — Local Pipeline Prototype (Offline Version)

**Goal:** Build a working speech → translated text → speech pipeline locally.

**Steps:**

- Set up the project directory structure (`/src`, `/tests`, `requirements.txt`).
- Implement basic audio capture (from file/sample) and playback using `PyAudio` or `sounddevice`.
- Integrate OpenAI Whisper for offline transcription of sample audio.
- Integrate Helsinki-NLP Opus-MT for offline text translation.
- Integrate pyttsx3 for offline text-to-speech conversion.
- Develop `main_pipeline.py` to orchestrate the STT → MT → TTS flow.
- Implement basic latency measurement for the offline pipeline.
- **Status:** Completed. The offline pipeline successfully processes recorded audio, translates it, and plays back the translated speech. Initial latency measurements have been obtained.

### Phase 2 — Real-Time Audio Stream

**Goal:** Process live microphone input in chunks for real-time translation.

**Steps:**

- Modify audio input to handle live microphone streams using `sounddevice` or `PyAudio`.
- Implement incremental transcription with Whisper, feeding short audio buffers.
- Adapt MT and TTS modules for streaming input/output.
- Optimize pipeline for low-latency chunk processing.
- **Status:** Implemented. Live microphone input is processed in chunks, transcribed by Whisper, translated by Opus-MT, and translated text is printed to the console with non-blocking TTS playback. Initial real-time demo is functional, but further optimization for latency and translation quality is needed.

### Phase 3 — User Interface

**Goal:** Provide a way to control translation and display results.

**Steps:**

- Choose a UI framework: Streamlit (for simplicity and speed) or Electron/Flask web app (for Jitsi overlay potential).
- Develop UI elements for:
  - Starting/stopping translation.
  - Selecting source/target languages (English ↔ Slovak).
  - Displaying real-time subtitles.
  - (Optional) Playing synthesized translated speech.

### Phase 4 — Integration & Testing

**Goal:** Integrate the translator into a conference simulation (Jitsi Meet) and thoroughly test its performance.

**Steps:**

- Integrate the translation system with Jitsi Meet (e.g., via Jitsi Meet SDK for subtitle overlay or audio injection).
- Set up a testing environment to simulate a multi-user call.
- Conduct comprehensive testing:
  - **End-to-end latency:** Measure actual delay from speech input to translated output.
  - **Accuracy:** Qualitatively assess STT and MT accuracy for English ↔ Slovak.
  - **Performance:** Monitor CPU/GPU usage during real-time operation.
  - **Robustness:** Test under varying network conditions (if feasible).

## 4. Documentation Phase (Part 3)

**Objective:** To compile all project work into a Bachelor's thesis report and prepare a presentation.

**Tasks:**

- **Update `README.md`:** Include architecture diagram, installation instructions, usage, testing, and final results.
- **Compile Thesis Report:**
  - Follow the `thesis_report_outline.md`.
  - Integrate research findings, system design, implementation details, evaluation results, and future work.
  - Utilize `decisions_log.md` for justification and references.
- **Prepare Presentation:**
  - Create 7-10 slides summarizing the project for supervisor review and thesis defense.

## 5. Repository Structure

```
/real-time-speech-translation
│
├── /research
│   ├── asr_comparison.md (will be part of decisions_log.md)
│   ├── translation_comparison.md (will be part of decisions_log.md)
│   ├── tts_comparison.md (will be part of decisions_log.md)
│   └── architecture_diagram.png (will be created)
│
├── /src
│   ├── audio_input.py
│   ├── stt_whisper.py
│   ├── translate.py
│   ├── tts_output.py
│   ├── main_pipeline.py
│   └── ui_app.py
│
├── /tests
│   ├── latency_test.py
│   ├── accuracy_test.py
│   └── integration_test.py
│
├── requirements.txt
├── README.md
├── thesis_report_outline.md
├── plan.md
└── decisions_log.md
```
