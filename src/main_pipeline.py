import asyncio
import os
import torch
import torchaudio
import numpy as np
import soundfile as sf
from pydub import AudioSegment, effects
import datetime
import time
import re

from pipeline_orchestrator import PipelineOrchestrator

# Configuration (can be moved to a config file later)
SOURCE_AUDIO_PATH_M4A = "tests/My test speech.m4a"
OUTPUT_DIR = "research"

def convert_m4a_to_wav(m4a_path):
    """Converts an M4A file to a WAV file."""
    wav_path = m4a_path.replace(".m4a", ".wav")
    if not os.path.exists(wav_path):
        print(f"Converting {m4a_path} to {wav_path}...")
        audio = AudioSegment.from_file(m4a_path, format="m4a")
        audio.export(wav_path, format="wav")
    return wav_path

async def main_pipeline_run(source_lang="en", target_lang="sk"): # Changed default to "sk"
    """
    Runs the live speech translation pipeline using the orchestrator.
    """
    print(f"Starting live pipeline: {source_lang} -> {target_lang}")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Convert M4A to WAV if necessary
    source_audio_wav_path = convert_m4a_to_wav(SOURCE_AUDIO_PATH_M4A)

    orchestrator = PipelineOrchestrator(source_lang=source_lang, target_lang=target_lang)
    await orchestrator.run_pipeline(source_audio_wav_path)

    print("Live pipeline execution completed.")

if __name__ == "__main__":
    # Example usage: English to Czech (as Slovak is not supported by TTS)
    asyncio.run(main_pipeline_run(source_lang="en", target_lang="sk")) # Changed target_lang to "sk"
    # Example usage: Slovak to English (if you have a Slovak test audio and speaker reference)
    # asyncio.run(main_pipeline_run(source_lang="sk", target_lang="en"))
