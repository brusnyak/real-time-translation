# Research Summary Report: Real-Time Speech Translation in Online Conferences

## 1. Introduction

This report summarizes the findings from the initial research phase for the Bachelor's thesis project, "Real-Time Speech Translation in Online Conferences." The objective of this phase was to confirm the suitability of selected Speech-to-Text (STT), Machine Translation (MT), and Text-to-Speech (TTS) tools, understand their real-time implications, and research integration strategies with Jitsi Meet. The primary criteria for tool selection included reliability, efficiency, speed, simplicity, and cost-effectiveness (utilizing free/open-source solutions where possible).

## 2. Core Technology Stack Selection and Justification

Based on the project requirements and initial recommendations, the following core technologies have been selected:

### 2.1. Speech-to-Text (STT): OpenAI Whisper

*   **Capabilities:** OpenAI Whisper is a general-purpose speech recognition model capable of multilingual speech recognition, speech translation, and language identification. It is trained on a large and diverse dataset, offering robust performance.
*   **Licensing:** Released under the MIT License, making it free and open-source.
*   **Language Support:** Supports a wide range of languages, including English and Slovak. For translation tasks, multilingual models (e.g., `medium` or `large`) are recommended.
*   **Real-time Implications:** While its `transcribe()` method processes audio in 30-second windows, lower-level APIs (`detect_language()`, `decode()`) allow for chunk-based processing, which is essential for real-time streaming.
*   **Justification:** Whisper's strong multilingual capabilities, open-source nature, and flexibility for real-time processing make it an excellent choice for the STT component.

### 2.2. Machine Translation (MT): Helsinki-NLP Opus-MT

*   **Capabilities:** Helsinki-NLP provides a vast collection of pre-trained neural machine translation models, particularly those based on the OPUS corpus. These models are known for their wide language coverage and are actively maintained.
*   **Licensing:** Models are generally released under open licenses, suitable for academic use.
*   **Language Support:** Specific models for English ↔ Slovak (e.g., `opus-mt-en-sk`, `opus-mt-sk-en`) are available, ensuring direct support for the target languages.
*   **Real-time Implications:** These models can be loaded and run locally using the Hugging Face `transformers` library, enabling low-latency inference. The challenge will involve efficient handling of text chunks from the STT module.
*   **Justification:** Opus-MT's extensive language coverage, open-source availability, and local inference capabilities make it a strong candidate for the MT component, aligning with the project's requirements for reliability and cost-effectiveness.

### 2.3. Text-to-Speech (TTS): pyttsx3

*   **Capabilities:** `pyttsx3` is an offline text-to-speech conversion library for Python. It works without an internet connection and supports multiple TTS engines installed on the system (e.g., Sapi5 on Windows, NSSpeechSynthesizer/AVSpeech on macOS, eSpeak on Linux). It allows control over speech rate and volume, and can save speech to audio files.
*   **Licensing:** MPL-2.0 license, making it open-source.
*   **Language Support:** Relies on the underlying system's TTS engines, which typically support a variety of languages. eSpeak, for instance, supports many languages. Specific support for Slovak will depend on the installed engines.
*   **Real-time Implications:** As an offline library, `pyttsx3` can offer low-latency synthesis. Its API allows for direct text-to-speech conversion, which can be integrated into a streaming pipeline.
*   **Voice Cloning:** `pyttsx3` itself does not inherently offer voice cloning. This feature would depend on the capabilities of the underlying TTS engine or require integration with a separate voice cloning solution, which would be considered a stretch goal.
*   **Justification:** `pyttsx3` was selected as an alternative to Coqui TTS due to Python version compatibility (Python 3.13.2). Its offline nature, simplicity, and compatibility with the current Python environment make it a suitable, free, and open-source choice for the TTS component, fulfilling the core requirements of the project.

## 3. Real-time Pipeline Management and Jitsi Meet Integration

### 3.1. Pipeline Management

*   **Chunking:** Audio will be processed in small, fixed-size chunks throughout the STT → MT → TTS pipeline to minimize end-to-end latency.
*   **Asynchronous Processing:** Python's `asyncio` or `threading` will be employed in the backend to manage the concurrent execution of the STT, MT, and TTS modules, ensuring responsiveness and efficient resource utilization.
*   **Communication:** WebSockets will serve as the primary real-time communication channel between the Python backend (where STT, MT, TTS run) and the JavaScript/HTML frontend (for UI and Jitsi integration).

### 3.2. Jitsi Meet Integration Strategy

*   **Embedding:** Jitsi Meet will be embedded into a custom web application using its IFrame API. This allows for a controlled environment to host the conference and integrate custom functionalities.
*   **Subtitle Overlay:** Translated text from the Python backend will be sent to the JavaScript frontend via WebSockets. The frontend will then display these subtitles as an overlay within or alongside the Jitsi Meet iframe, providing real-time visual translation.
*   **Audio Injection (Stretch Goal):** Further investigation will be conducted into Jitsi's IFrame API `devices` options or other WebRTC mechanisms to determine the feasibility of injecting the synthesized audio directly into the conference. This could involve creating a "translation bot" participant or modifying an existing participant's audio stream.
*   **User Interface:** A web-based UI (e.g., Flask with HTML/CSS/JS) will be developed to host the Jitsi iframe and provide controls for the translation system, including starting/stopping translation and language selection.

## 4. Conclusion

The research phase has successfully identified and justified the core technologies for the "Real-Time Speech Translation in Online Conferences" project: OpenAI Whisper for STT, Helsinki-NLP Opus-MT for MT, and Coqui TTS for TTS. A clear strategy for real-time pipeline management and integration with Jitsi Meet via its IFrame API has also been outlined. These decisions align with the project's requirements for reliability, efficiency, speed, simplicity, and cost-effectiveness. The next steps involve setting up the project structure and beginning the implementation of the local pipeline prototype.
