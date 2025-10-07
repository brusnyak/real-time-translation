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

### 2.2. Machine Translation (MT): Helsinki-NLP Opus-MT / Google Translate

*   **Capabilities:** Helsinki-NLP provides a vast collection of pre-trained neural machine translation models, particularly those based on the OPUS corpus. These models are known for their wide language coverage and are actively maintained. Google Translate (via `googletrans` library) has also been integrated for improved translation accuracy, especially for specific terms.
*   **Licensing:** Models are generally released under open licenses, suitable for academic use. `googletrans` is also open-source.
*   **Language Support:** Specific models for English ↔ Slovak (e.g., `opus-mt-en-sk`, `opus-mt-sk-en`) are available. Google Translate provides robust support for both English and Slovak.
*   **Real-time Implications:** These models can be loaded and run locally using the Hugging Face `transformers` library, enabling low-latency inference. The challenge will involve efficient handling of text chunks from the STT module.
*   **Justification:** The combination of Opus-MT and Google Translate offers comprehensive language coverage and improved accuracy, aligning with the project's requirements for reliability and cost-effectiveness.

### 2.3. Text-to-Speech (TTS): Coqui XTTS v2 (Hybrid with pyttsx3)

*   **Capabilities:** Coqui XTTS v2 is a powerful deep learning TTS model known for its high-quality, low-latency streaming capabilities and robust voice cloning. It supports a wide range of languages. `pyttsx3` is an offline text-to-speech conversion library for Python, used as a fallback.
*   **Licensing:** XTTS v2 is open-source (MPL-2.0 license). `pyttsx3` is also open-source.
*   **Language Support:** XTTS v2 supports languages such as English (`en`), Czech (`cs`), Spanish (`es`), French (`fr`), German (`de`), Italian (`it`), Portuguese (`pt`), Polish (`pl`), Turkish (`tr`), Russian (`ru`), Dutch (`nl`), Arabic (`ar`), Chinese (`zh-cn`), Hungarian (`hu`), Korean (`ko`), Japanese (`ja`), and Hindi (`hi`). **Crucially, XTTS v2 does not natively support Slovak (`sk`).** Due to the linguistic similarity between Czech and Slovak, Czech is used as a proxy for Slovak speech synthesis with XTTS v2 to leverage its voice cloning capabilities. For direct Slovak synthesis, `pyttsx3` is used as a fallback.
*   **Real-time Implications:** XTTS v2 is highlighted for sub-200ms latency streaming, making it highly suitable for real-time applications. `pyttsx3` also offers low-latency offline synthesis.
*   **Voice Cloning:** XTTS v2 offers robust voice cloning capabilities, which is a key feature for the project's goal of preserving original speaker characteristics. The `speaker_wav_path` is converted to a compatible WAV format to ensure XTTS v2 can process it correctly.
*   **Justification:** A hybrid approach combining XTTS v2 for supported languages (including Czech as a proxy for Slovak) and `pyttsx3` for direct Slovak synthesis provides the best balance of voice cloning capability, language coverage, and real-time performance within the current technical constraints.

## 3. Real-time Pipeline Management and Jitsi Meet Integration

### 3.1. Pipeline Management

*   **Chunking:** Audio is processed in small, fixed-size chunks throughout the STT → MT → TTS pipeline to minimize end-to-end latency.
*   **Asynchronous Processing:** Python's `asyncio` is employed in the backend to manage the concurrent execution of the STT, MT, and TTS modules, ensuring responsiveness and efficient resource utilization.
*   **Communication:** WebSockets will serve as the primary real-time communication channel between the Python backend (where STT, MT, TTS run) and the JavaScript/HTML frontend (for UI and Jitsi integration).

### 3.2. Jitsi Meet Integration Strategy

*   **Embedding:** Jitsi Meet will be embedded into a custom web application using its IFrame API. This allows for a controlled environment to host the conference and integrate custom functionalities.
*   **Subtitle Overlay:** Translated text from the Python backend will be sent to the JavaScript frontend via WebSockets. The frontend will then display these subtitles as an overlay within or alongside the Jitsi Meet iframe, providing real-time visual translation.
*   **Audio Injection (Stretch Goal):** Further investigation will be conducted into Jitsi's IFrame API `devices` options or other WebRTC mechanisms to determine the feasibility of injecting the synthesized audio directly into the conference. This could involve creating a "translation bot" participant or modifying an existing participant's audio stream.
*   **User Interface:** A web-based UI (e.g., Flask with HTML/CSS/JS) will be developed to host the Jitsi iframe and provide controls for the translation system, including starting/stopping translation and language selection.

## 4. Test Documentation and Progress Tracking

*   **Test Saving:** Each test run of the `main_pipeline.py` now generates a unique, sequentially numbered directory (e.g., `research/test_1/`, `research/test_2/`) within the `research/` folder. This directory contains the `original_audio_waveform.png` and `translated_audio_waveform.png` plots, along with a description of the test. This ensures a structured approach to documenting development updates and comparing audio signals.

## 5. Conclusion

The research phase has successfully identified and justified the core technologies for the "Real-Time Speech Translation in Online Conferences" project: OpenAI Whisper for STT, a hybrid of Opus-MT and Google Translate for MT, and a hybrid of Coqui XTTS v2 and `pyttsx3` for TTS. A clear strategy for real-time pipeline management and integration with Jitsi Meet via its IFrame API has also been outlined. These decisions align with the project's requirements for reliability, efficiency, speed, simplicity, and cost-effectiveness. The next steps involve further optimization of the pipeline, particularly for grammatical coherence in chunk-based translation, and then proceeding with the implementation of the user interface.
