# Real-Time Speech Translation for Online Conferences

Prototype for near–real-time English ↔ Slovak speech translation with voice cloning. See `plan.md` for the detailed roadmap.

## Quick start

```bash
pip install -r requirements.txt
```

### Offline file validation (recommended first)

Place audio in `tests/` (e.g., `tests/My test speech.m4a`) and a speaker reference (e.g., `tests/My test speech_xtts_speaker.wav`). Then run the offline pipeline (to be implemented in `src/main_pipeline.py`).

### Live loop

Run the real-time loop from `src/main_pipeline.py` (mic → STT → MT → TTS → playback). Streamlit UI for controls/subtitles will be added in `src/ui_app.py`.

## Architecture

- Python backend: Whisper (STT) → Opus-MT (MT) → XTTS v2 (TTS)
- UI: Streamlit; Jitsi IFrame for subtitles (audio injection is a stretch goal)

## Tech

- STT: `openai-whisper`
- MT: `transformers` with Helsinki-NLP Opus-MT
- TTS: `TTS` (XTTS v2)
- IO/UI: `sounddevice`/`pyaudio`, `streamlit`

## Important note (XTTS v2)

XTTS v2 lacks native Slovak voice cloning. For Slovak output we synthesize using Czech in XTTS v2 to achieve better voice similarity. This is intentional and documented in `plan.md`.
