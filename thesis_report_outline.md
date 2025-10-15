# Bachelor's Thesis Report Outline

## Title: Real-Time Speech Translation in Online Conferences

## Abstract

(To be filled after project completion)

## 1. Introduction

- Problem: Language barriers in online meetings
- Motivation: Need for real-time, accessible, and efficient speech translation
- Project Goal: Develop a prototype system for real-time speech translation with dynamic language selection and voice cloning capabilities, integrated via a web-based UI.

## 2. Theoretical Background

- Speech Recognition (ASR) principles and challenges
- Machine Translation (MT) techniques (neural networks, sequence-to-sequence models)
- Text-to-Speech (TTS) synthesis methods, including voice cloning
- Real-time streaming and pipeline management challenges (latency, chunking, asynchronous processing)

## 3. State of the Art

- Overview of selected ASR models (OpenAI Whisper)
- Overview of selected MT models (Helsinki-NLP Opus-MT / Google Translate)
- Overview of selected TTS models (Coqui XTTS v2, with `pyttsx3` as fallback)
- Review of existing real-time translation systems and their limitations

## 4. System Design

- Overall Architecture Diagram (Backend: FastAPI, Frontend: HTML/CSS/JS)
- Component breakdown: Audio Input (Web Audio API), STT (`faster-whisper`), MT (`MarianMT`), TTS (`Coqui XTTS v2` / `pyttsx3` hybrid), Output Audio (Speakers/BlackHole), Subtitle Display (Web UI)
- Chosen tools and libraries justification (OpenAI Whisper, Helsinki-NLP Opus-MT / Google Translate, Coqui XTTS v2, `pyttsx3`, FastAPI, WebSockets, HTML/CSS/JavaScript)
- Data flow and pipeline management strategy (WebSocket communication, asynchronous processing, audio chunking)
- Latency targets and optimization considerations (GPU usage, `int8` quantization, `EMA_ALPHA` tuning)

## 5. Implementation

- Project structure and module descriptions (`server.py`, `src/stt_module.py`, `src/mt_module.py`, `src/tts_module.py`, `ui/index.html`, `ui/script.js`, `ui/style.css`)
- Detailed explanation of each phase:
  - Phase 1: Backend Development (FastAPI server, WebSocket API, STT/MT/TTS integration)
  - Phase 2: Frontend Development (Web UI for audio capture, display, and controls)
  - Phase 3: Real-Time Audio Processing and Language Switching (VAD, dynamic model re-initialization)
  - Phase 4: Performance Tuning and UI Responsiveness (audio input level, latency optimization)
- Key algorithms and techniques used (VAD, concurrent synthesis, dynamic model loading)

## 6. Evaluation

- Methodology for testing (latency, accuracy, performance, UI responsiveness)
- Results:
  - End-to-end latency measurements
  - STT accuracy (qualitative/quantitative)
  - Translation accuracy (qualitative)
  - CPU/GPU performance analysis (MPS/CUDA vs. CPU for different modules)
- Discussion of limitations and challenges encountered:
  - **GPU Compatibility Issues**: Initial attempts to use MPS (Apple Silicon GPU) for `faster-whisper` (STT) and `Coqui TTS` (TTS) encountered `unsupported device mps` and `CUDA is not available` errors, leading to CPU fallback for these modules.
  - **TTS Dependency Conflicts**: Persistent and severe dependency conflicts with `Coqui TTS` and `Bark by Suno AI` (e.g., `numpy` version mismatches) repeatedly destabilized the Python environment, making voice cloning initially infeasible.
  - **`pyttsx3` Quality and Compatibility**: Initial use of `pyttsx3` resulted in "abruptive and robot-like" voice quality and `wave.Error` due to incompatible WAV file formats, requiring explicit driver selection (`espeak`) and further quality tuning.
  - **Translation Accuracy**: Initial `Whisper` "base" model and `googletrans` issues led to incorrect translations (e.g., "Slovak" as "Slavak"), necessitating model upgrades and temporary workarounds.
  - **Real-time Latency**: Achieving low end-to-end latency was a continuous challenge, requiring careful chunking, asynchronous processing, and module-specific optimizations.
  - **UI Responsiveness**: Initial audio input level UI was not responsive, requiring tuning of gain factors in the backend and smoothing parameters in the frontend.
  - **Dynamic Language Switching**: Ensuring all backend modules correctly re-initialize and apply new language settings dynamically was complex.

## 7. Conclusion & Future Work

- Summary of achievements:
  - Developed a functional real-time speech translation prototype with a web UI.
  - Implemented dynamic language switching for STT, MT, and TTS.
  - Successfully integrated `faster-whisper`, `MarianMT`, and `Coqui XTTS v2` (with `pyttsx3` fallback).
  - Achieved responsive audio input level visualization and real-time transcription/translation display.
  - Documented key decisions and challenges throughout the development process.
- Potential improvements:
  - **Enhanced Voice Cloning**: Further research into more stable voice cloning libraries or newer versions of `Coqui XTTS` that resolve dependency conflicts.
  - **Broader Language Support**: Expand the range of natively supported languages for XTTS v2 or integrate more robust multilingual TTS models.
  - **Advanced Jitsi Meet Integration**: Implement direct audio injection into Jitsi Meet for a more seamless experience.
  - **Improved Latency**: Explore further optimizations, such as model quantization for MPS/CUDA, or more efficient audio processing techniques.
  - **Error Handling and Robustness**: Enhance error handling for model loading and synthesis failures.
  - **User Experience**: Add features like speaker diarization, customizable voice profiles, and more advanced UI controls.
- Possible applications and impact:
  - Facilitating multilingual communication in online meetings and educational settings.
  - Providing accessibility features for individuals with hearing impairments.
  - Enabling real-time interpretation services.

## References

(To be filled)

## Appendices

(To be filled, e.g., code snippets, detailed comparison tables)
