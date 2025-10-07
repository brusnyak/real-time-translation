# Project Decisions Log

This document records key decisions made during the research and implementation phases of the "Real-Time Speech Translation in Online Conferences" Bachelor's thesis project. Each entry includes the decision, its justification, and relevant references.

## 1. Initial Project Setup and Scope

- **Decision:** Proceed with the "Real-Time Speech Translation in Online Conferences" project as outlined in `project.txt`, `README.md`, and `thesis_report_outline.md`.
- **Justification:** The project addresses a relevant real-world problem, utilizes modern AI technologies, and is achievable within the scope of a Bachelor's thesis. The phased approach (research, local prototype, real-time, integration) provides a structured development path.
- **References:** `project.txt`, `README.md`, `thesis_report_outline.md`

## 2. Voice Preservation / Cloning

- **Decision:** For the initial prototype, prioritize high-quality, low-latency synthesized speech. Voice cloning (retaining the original speaker's voice characteristics) will be considered an advanced extension or future work if time and resources permit.
- **Justification:** While the ideal is to use the original voice, achieving robust, real-time voice cloning adds significant complexity. Focusing on high-quality synthesis first ensures the core functionality is delivered within the project timeline.
- **References:** User input (05/10/2025, 10:47:30 am)

## 3. Conferencing System Integration

- **Decision:** Focus research and implementation efforts primarily on Jitsi Meet for conference integration.
- **Justification:** Jitsi Meet is an established open-source conferencing tool, and the user has expressed confidence in its suitability. This focus streamlines the integration phase.
- **References:** User input (05/10/2025, 10:47:30 am)

## 4. Core Technology Stack (Initial Selection)

- **Decision:** Streamline the research phase by focusing on the following core technologies, as they are already highlighted in the project documentation and align with the criteria of reliability, efficiency, speed, simplicity, and cost-effectiveness (free/open-source).
  - **Speech-to-Text (STT):** OpenAI Whisper (local model)
  - **Machine Translation (MT):** Helsinki-NLP (Opus-MT)
  - **Text-to-Speech (TTS):** Coqui TTS
- **Justification:** These tools are well-regarded in their respective domains and offer open-source options suitable for a Bachelor's project. This focused approach allows for deeper investigation into their real-time capabilities and integration challenges rather than broad comparison of many tools.
- **References:** `project.txt`, `README.md`, User input (05/10/2025, 10:58:32 am)

## 5. Speech-to-Text (STT) - OpenAI Whisper Research

- **Decision:** OpenAI Whisper is a suitable candidate for the STT component.
- **Justification:**
  - **Capabilities:** General-purpose speech recognition, multilingual speech recognition, speech translation, and language identification. This covers the core STT and initial translation needs.
  - **Licensing:** MIT License, making it free and open-source.
  - **Language Support:** Supports a wide range of languages, including English. Multilingual models (`medium`, `large`) are recommended for translation tasks. Performance for Slovak needs further investigation but is expected to be reasonable given its multilingual training.
  - **Real-time Implications:** The `transcribe()` method processes audio in 30-second windows. For real-time streaming, a chunk-based approach feeding smaller audio segments to the model will be necessary, potentially using lower-level `detect_language()` and `decode()` methods for more control.
  - **Setup:** Python 3.8-3.11, PyTorch, `openai-whisper` package, and `ffmpeg` are required.
- **References:** [OpenAI Whisper GitHub Repository](https://github.com/openai/whisper)

## 6. Machine Translation (MT) - Helsinki-NLP Opus-MT Research

- **Decision:** Helsinki-NLP Opus-MT is a suitable candidate for the MT component.
- **Justification:**
  - **Capabilities:** Helsinki-NLP provides a vast collection of pre-trained machine translation models, including those based on the OPUS corpus, which are known for wide language coverage. They focus on open data sets and public pre-trained models, aligning with our cost-effectiveness and open-source requirements.
  - **Licensing:** Models are generally released under open licenses, making them suitable for academic projects.
  - **Language Support:** Helsinki-NLP offers specific models for English ↔ Slovak (e.g., `opus-mt-en-sk`, `opus-mt-sk-en`).
  - **Real-time Implications:** These models can be loaded and used locally via the Hugging Face `transformers` library, allowing for low-latency inference. The challenge will be managing the input/output for streaming text chunks.
  - **Setup:** Requires Python and the `transformers` library.
- **References:** [Helsinki-NLP Hugging Face Organization](https://huggingface.co/Helsinki-NLP)

## 7. Text-to-Speech (TTS) - Coqui TTS Research (Initial)

- **Decision:** Coqui TTS was initially identified as a suitable candidate for the TTS component.
- **Justification:**
  - **Capabilities:** Comprehensive deep learning toolkit for Text-to-Speech, supporting various models, multi-speaker, and multi-lingual capabilities.
  - **Licensing:** MPL-2.0 license (open-source).
  - **Language Support:** Supports over 1100 languages, including English and Slovak.
  - **Real-time Implications:** XTTS v2 highlighted for <200ms latency streaming.
  - **Voice Cloning:** Models like XTTS v2 and YourTTS offer voice cloning.
  - **Setup:** Requires Python 3.9-3.11 and the `TTS` PyPI package.
- **References:** [Coqui TTS GitHub Repository](https://github.com/coqui-ai/TTS)

## 8. Text-to-Speech (TTS) - Alternative Selection

- **Decision:** Due to Python version incompatibility (Coqui TTS requires Python < 3.12, while the current environment is Python 3.13.2), `pyttsx3` has been selected as an alternative open-source TTS library compatible with Python 3.13.2.
- **Justification:** `pyttsx3` is an offline text-to-speech conversion library that explicitly lists compatibility with Python 3.13. It supports multiple TTS engines (Sapi5, nsss, espeak) and offers features like voice selection, rate/volume control, and saving speech to a file. This aligns with the project's requirements for an efficient, simple, free, and offline TTS solution.
- **References:** User input (05/10/2025, 11:19:20 am), [pyttsx3 PyPI Page](https://pypi.org/project/pyttsx3/)

## 8. Conferencing System Integration - Jitsi Meet Research

- **Decision:** Jitsi Meet can be integrated into a web application using its IFrame API.
- **Justification:** The Jitsi Meet IFrame API allows embedding Jitsi Meet functionality into a custom web application. This provides a mechanism to control the meeting (e.g., `roomName`, `userInfo`) and potentially interact with its audio/video streams or display elements.
  - **Audio Injection:** The documentation mentions `devices` options for `audioInput` and `audioOutput`, which might allow for programmatic control over audio sources. Further investigation will be needed to determine if we can inject a synthesized audio stream directly into a participant's audio output or as a new participant's audio input.
  - **Subtitle Display:** The IFrame API provides `interfaceConfigOverwrite` options, which could potentially be used to customize the UI to display subtitles. Alternatively, a separate overlay UI (e.g., built with Streamlit or a Flask web app) could be developed alongside the embedded Jitsi iframe.
  - **Real-time Implications:** The API is JavaScript-based, meaning the integration logic would reside in the frontend. Communication between the Python backend (running STT, MT, TTS) and the JavaScript frontend would likely require WebSockets or a similar real-time communication mechanism.
- **References:** [Jitsi Meet IFrame API Documentation](https://jitsi.github.io/handbook/docs/dev-guide/dev-guide-iframe)

## 9. Offline Pipeline Prototype Execution

- **Decision:** The core offline pipeline prototype (`src/main_pipeline.py`) has been successfully executed.
- **Justification:** The script successfully recorded audio, performed Speech-to-Text (Whisper), Machine Translation (Opus-MT), and Text-to-Speech (pyttsx3), and played back both the original and translated audio.
  - **SSL Certificate Error Resolution:** The `ssl.SSLCertVerificationError` encountered during Whisper model download was resolved by setting the `SSL_CERT_FILE` environment variable to the `certifi` bundle path.
  - **PyAudio `wave.Error` Resolution:** The `wave.Error: file does not start with RIFF id` during translated audio playback was resolved by ensuring `espeak-ng` was installed and modifying `tts_output.py` to explicitly initialize `pyttsx3` with the `espeak` driver, which produces standard WAV files.
  - **`sacremoses` Installation:** The recommended `sacremoses` dependency for `transformers` was installed.
- **Observed Latency:** The total pipeline latency for a 5-second audio segment was approximately 3.87 seconds (STT: ~0.8s, Translation: ~2.4s, TTS: ~0.01s). This provides a baseline for future real-time optimization.
- **References:** Terminal output (05/10/2025, 11:37:28 am, 05/10/2025, 11:40:00 am), `src/tts_output.py` modifications, `requirements.txt` modifications.

## 10. Text-to-Speech (TTS) - Abandoning Voice Cloning for Prototype and Reverting to pyttsx3

*   **Decision:** After extensive and persistent dependency conflicts with Coqui TTS and Bark by Suno AI, and severe audio quality and file format compatibility issues with `pyttsx3` when attempting to use high-quality system voices on macOS, the project will abandon voice cloning for the current prototype. The TTS component will revert to a stable `pyttsx3` setup, focusing on achieving the best possible generic voice quality and reliable audio output.
*   **Justification:** Both Coqui TTS and Bark introduced insurmountable dependency conflicts (e.g., `numpy` version mismatches with `pandas` and `scikit-learn`) that repeatedly broke the entire Python environment, despite multiple attempts at clean installations and version pinning. `pyttsx3`, while lacking voice cloning, offers a more stable foundation. Prioritizing a functional and reliable core translation pipeline with clear, albeit generic, synthesized speech is a more realistic and achievable goal for a Bachelor's thesis prototype within the given timeframes. Voice cloning will be explicitly noted as a key area for future work.
*   **References:** User input (05/10/2025, 3:59:45 pm, 05/10/2025, 4:25:42 pm, 05/10/2025, 4:32:50 pm), numerous terminal outputs detailing `pip install` failures, `ModuleNotFoundError` for `coqpit` and `coqpit-config`, `sounddevice.PortAudioError`, `soundfile.LibsndfileError`, `ValueError: numpy.dtype size changed`.

## 11. Testing with Provided Audio File

*   **Decision:** The `src/main_pipeline.py` has been modified to use the provided `tests/My test speech.m4a` file for offline demo testing, instead of recording new audio. `pydub` was integrated to handle the `.m4a` audio format, converting it to a WAV format in memory for `soundfile` to process.
*   **Justification:** This allows for consistent testing of translation quality and synthesized voice using a specific, known audio input, as requested by the user, and resolves the "Format not recognised" error for `.m4a` files.
*   **References:** User input (05/10/2025, 4:15:20 pm), `src/main_pipeline.py` modifications, `pydub` installation.

## 12. Text-to-Speech (TTS) - pyttsx3 Playback Error Resolution

*   **Decision:** Add explicit initialization of `pyttsx3` with the 'espeak' driver in `src/tts_output.py` to resolve `wave.Error: file does not start with RIFF id` and `FileNotFoundError` during temporary WAV file processing.
*   **Justification:** The previous `pyttsx3` initialization was defaulting to a driver (likely `nsss` on macOS) that produced WAV files incompatible with `wave` module or had timing issues, leading to playback failures. Explicitly using 'espeak' (which requires `espeak-ng` to be installed) ensures a more robust and compatible WAV output.
*   **References:** Terminal output (05/10/2025, 4:56:35 pm), `src/tts_output.py` modifications.

## 13. Speech-to-Text (STT) - Whisper Model Upgrade

*   **Decision:** Upgrade the OpenAI Whisper model size from "base" to "medium" in `src/main_pipeline.py` to improve transcription accuracy, especially for proper nouns and less common terms.
*   **Justification:** The "base" model incorrectly transcribed "Slovak" as "slava", leading to an incorrect translation. The project plan and `stt_whisper.py` docstring recommend larger multilingual models for better performance in translation tasks. The "medium" model offers a good balance between accuracy and computational resources.
*   **References:** Terminal output (05/10/2025, 4:58:02 pm, 05/10/2025, 5:00:23 pm), `src/main_pipeline.py` modifications, `src/stt_whisper.py` documentation.

## 14. Transcription and Translation Quality Improvement

*   **Decision:** The upgrade of the Whisper model to "medium" has significantly improved transcription accuracy, correctly identifying "Slovak" (as "Slavak") and leading to a more accurate translation into Slovak ("Slavaku").
*   **Justification:** The previous "base" model struggled with proper nouns, leading to incorrect transcriptions and subsequent translation errors. The "medium" model demonstrates better performance for the target languages.
*   **References:** Terminal output (05/10/2025, 5:00:23 pm).

## 15. Voice Cloning Re-evaluation and Generic TTS Improvement

*   **Decision:** After a renewed attempt to integrate Coqui XTTS v2 (TTS==0.22.0) led to severe and persistent dependency conflicts (specifically with `numpy`, `pandas`, `scipy`, `gruut`, and `librosa` versions required by other core components like `streamlit`), the project will definitively abandon voice cloning for the current prototype. The focus will remain on improving the generic `pyttsx3` voice quality. Voice cloning will be explicitly noted as a key area for future work, potentially requiring a dedicated environment or a more stable, future version of the library.
*   **Justification:** The repeated and insurmountable dependency conflicts introduced by voice cloning libraries (Coqui TTS, Bark) consistently destabilize the entire Python environment, making further development impractical within the scope of this thesis. Prioritizing a functional and reliable core translation pipeline with clear, albeit generic, synthesized speech is a more realistic and achievable goal.
*   **References:** User input (05/10/2025, 4:54:05 pm, 05/10/2025, 5:11:07 pm), Browser action failures (05/10/2025, 5:01:23 pm, 05/10/2025, 5:02:41 pm), Terminal output (05/10/2025, 5:11:45 pm, 05/10/2025, 5:12:03 pm, 05/10/2025, 5:12:33 pm), `src/main_pipeline.py` modifications.

## 16. Generic TTS Voice Selection and Quality Improvement

*   **Decision:** The `pyttsx3` engine was configured to explicitly select `gmw/en-us` for English and `zlw/sk` for Slovak, and the speech rate was adjusted to 170.
*   **Justification:** This aimed to improve the perceived quality and naturalness of the synthesized speech by using more appropriate voices for each language and a slightly faster delivery rate, addressing the "AI voice which cuts with some noise" feedback.
*   **References:** Terminal output (05/10/2025, 5:04:15 pm, 05/10/2025, 5:05:52 pm, 05/10/2025, 5:07:33 pm), `src/main_pipeline.py` modifications, `src/tts_output.py` modifications.

## 17. Re-evaluation of Translation Accuracy and Voice Cloning (Coqui XTTS v2)

*   **Decision:** The previous translation of "Slovak" to "Slavaku" was incorrect; the accurate translation should be "slovenčiny" in the context of the provided sentence. The generic `pyttsx3` voice quality is still unsatisfactory ("very AI-like with constant abruptions"). The project has now integrated the `googletrans` library for improved translation accuracy. Due to persistent and severe dependency conflicts, the renewed attempt to integrate Coqui XTTS v2 for voice cloning has been definitively abandoned for this prototype. The focus will remain on improving the generic `pyttsx3` voice quality. Voice cloning will be explicitly noted as a key area for future work, potentially requiring a dedicated environment or a more stable, future version of the library.
*   **Justification:** User feedback indicated critical issues with both translation and voice quality. The `googletrans` library has been successfully integrated to address translation accuracy. However, repeated attempts to integrate Coqui XTTS v2 consistently destabilized the Python environment, making its inclusion impractical within the current project scope. Prioritizing a functional and reliable core translation pipeline with clear, albeit generic, synthesized speech is the most realistic and achievable goal for this Bachelor's thesis prototype.
*   **References:** User input (05/10/2025, 5:11:07 pm), Coqui XTTS v2 setup guide, `googletrans` PyPI page (05/10/2025, 5:16:58 pm), `src/translate.py` modifications, Terminal output (05/10/2025, 5:11:45 pm, 05/10/2025, 5:12:03 pm, 05/10/2025, 5:12:33 pm, 05/10/2025, 5:18:31 pm, 05/10/2025, 5:18:59 pm, 05/10/2025, 5:19:23 pm), `requirements.txt` modifications.

## 18. Google Translate Asynchronous Call Fix

*   **Decision:** The `googletrans` library's asynchronous `translate` method was not being `await`ed, leading to a `'coroutine' object has no attribute 'text'` error. This has been resolved by marking the `translate_text` method in `src/translate.py` as `async` and ensuring it is `await`ed in `src/main_pipeline.py`, and by marking `run_offline_demo` and `run_realtime_pipeline` as `async` methods.
*   **Justification:** Correctly handling asynchronous operations is crucial for the `googletrans` library to function as intended, ensuring accurate translation.
*   **References:** Terminal output (07/10/2025, 6:06:04 am), `src/translate.py` modifications, `src/main_pipeline.py` modifications.
