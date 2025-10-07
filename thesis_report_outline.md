# Bachelor's Thesis Report Outline

## Title: Real-Time Speech Translation in Online Conferences

## Abstract
(To be filled after project completion)

## 1. Introduction
*   Problem: Language barriers in online meetings
*   Motivation: Need for real-time, accessible, and efficient speech translation
*   Project Goal: Develop a prototype system for English ↔ Slovak real-time speech translation

## 2. Theoretical Background
*   Speech Recognition (ASR) principles and challenges
*   Machine Translation (MT) techniques (neural networks, sequence-to-sequence models)
*   Text-to-Speech (TTS) synthesis methods
*   Real-time streaming and pipeline management challenges (latency, chunking)

## 3. State of the Art
*   Overview of selected ASR models (OpenAI Whisper)
*   Overview of selected MT models (Helsinki-NLP Opus-MT)
*   Overview of selected TTS models (pyttsx3)
*   Review of existing real-time translation systems and their limitations

## 4. System Design
*   Overall Architecture Diagram
*   Component breakdown: Audio Input, STT, MT, TTS, Output Audio/Subtitle Display
*   Chosen tools and libraries justification (Whisper, Opus-MT, Coqui TTS, Streamlit, Jitsi Meet)
*   Data flow and pipeline management strategy
*   Latency targets and optimization considerations

## 5. Implementation
*   Project structure and module descriptions (`audio_input.py`, `stt_whisper.py`, `translate.py`, `tts_output.py`, `main_pipeline.py`, `ui_app.py`)
*   Detailed explanation of each phase:
    *   Phase 1: Local Pipeline Prototype (offline version)
    *   Phase 2: Real-Time Audio Stream processing
    *   Phase 3: User Interface development (Streamlit)
    *   Phase 4: Integration with Jitsi Meet
*   Key algorithms and techniques used

## 6. Evaluation
*   Methodology for testing (latency, accuracy, performance)
*   Results:
    *   End-to-end latency measurements
    *   STT accuracy (qualitative/quantitative)
    *   Translation accuracy (qualitative)
    *   CPU/GPU performance analysis
*   Discussion of limitations and challenges encountered

## 7. Conclusion & Future Work
*   Summary of achievements
*   Potential improvements (e.g., adding new languages, better TTS voice quality, full integration into real platforms, voice cloning)
*   Possible applications and impact

## References
(To be filled)

## Appendices
(To be filled, e.g., code snippets, detailed comparison tables)
