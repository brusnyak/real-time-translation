#!/usr/bin/env python3
"""
Simple CLI test for live translation pipeline.
Records 5 seconds of audio, translates, and plays result.
No WebSocket, no async complexity - just pure pipeline testing.
"""

import os
import time
import numpy as np
import soundfile as sf
import sounddevice as sd
from src.stt_module import STTModule
from src.mt_module import MTModule
from src.tts_module import TTSModule


def find_device_id(device_name="BlackHole 2ch"):
    """Find audio device ID by name."""
    try:
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device_name.lower() in device["name"].lower():
                print(f"✓ Found device: {device['name']}")
                return i
    except Exception as e:
        print(f"Warning: {e}")
    print(f"✗ Device '{device_name}' not found. Using default.")
    return None


def list_input_devices():
    """List available audio input devices."""
    try:
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        print("Available input devices:")
        for i, device in enumerate(input_devices):
            print(f"  {i}: {device['name']}")
        return input_devices
    except Exception as e:
        print(f"Warning: {e}")
        return []

def record_audio(duration=5.0, sample_rate=16000, device_id=None):
    """Record audio from a specific microphone."""
    print(f"\n🎤 Recording for {duration} seconds...")
    print("   Speak now!")
    
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype=np.float32,
        device=device_id
    )
    sd.wait()
    
    print("✓ Recording complete")
    return audio.squeeze(), sample_rate


def save_audio(audio, sr, path="/tmp/test_audio.wav"):
    """Save audio to file."""
    sf.write(path, audio, sr)
    print(f"✓ Saved to {path}")
    return path


def test_pipeline():
    """Test the full pipeline."""
    print("\n" + "="*60)
    print("LIVE TRANSLATION PIPELINE - CLI TEST")
    print("="*60)
    
    # Initialize modules
    print("\n[1/4] Initializing modules...")
    try:
        stt = STTModule(model_size="base", device="cpu", compute_type="int8")
        mt = MTModule(source_lang="en", target_lang="sk", device="mps")
        tts = TTSModule(
            speaker_reference_path="tests/My test speech_xtts_speaker.wav",
            speaker_language="en",
            device="cpu",
            skip_warmup=False
        )
        print("✓ All modules loaded")
    except Exception as e:
        print(f"✗ Error loading modules: {e}")
        return
    
    # Record audio
    print("\n[2/4] Recording audio...")
    try:
        # Find a built-in microphone
        input_devices = sd.query_devices()
        input_device_id = None
        for i, device in enumerate(input_devices):
            if device['max_input_channels'] > 0 and \
               ("Built-in" in device["name"] or "Internal" in device["name"]):
                print(f"✓ Found built-in microphone: {device['name']}")
                input_device_id = i
                break
        
        if input_device_id is None:
            print("✗ No built-in microphone found. Using default device.")

        audio, sr = record_audio(duration=5.0, device_id=input_device_id)
        audio_path = save_audio(audio, sr)
    except Exception as e:
        print(f"✗ Error recording: {e}")
        return
    
    # STT
    print("\n[3/4] Speech-to-Text...")
    try:
        start = time.time()
        segments, _ = stt.transcribe(audio_path, language="en")
        transcribed = " ".join([seg.text for seg in segments])
        stt_time = time.time() - start
        
        if not transcribed.strip():
            print("✗ No speech detected. Try speaking louder.")
            return
        
        print(f"✓ Transcribed ({stt_time:.2f}s):")
        print(f"   '{transcribed}'")
    except Exception as e:
        print(f"✗ Error in STT: {e}")
        return
    
    # MT
    print("\n[4/4] Machine Translation & Text-to-Speech...")
    try:
        # Translation
        start_mt = time.time()
        translated = mt.translate(transcribed)
        translated = mt._fix_slovak_grammar(translated)
        mt_time = time.time() - start_mt
        
        print(f"✓ Translated ({mt_time:.2f}s):")
        print(f"   '{translated}'")
        
        # TTS
        start_tts = time.time()
        tts_path = None
        
        # Create asyncio event loop for async TTS
        import asyncio
        async def synthesize():
            return await tts.synthesize_chunks_concurrently([translated], crossfade_ms=50)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tts_path = loop.run_until_complete(synthesize())
        loop.close()
        
        tts_time = time.time() - start_tts
        
        if not tts_path or not os.path.exists(tts_path):
            print(f"✗ TTS failed to produce audio")
            return
        
        print(f"✓ Synthesized ({tts_time:.2f}s)")
        
        # Play audio to BOTH BlackHole and built-in speaker
        print("\n[OUTPUT] Playing translated audio...")
        audio_data, audio_sr = sf.read(tts_path, dtype=np.float32)
        
        blackhole_id = find_device_id("BlackHole 2ch")
        
        # Play to BlackHole (for conference participants)
        if blackhole_id is not None:
            print(f"  → BlackHole 2ch (conference participants)")
            sd.play(audio_data, samplerate=audio_sr, device=blackhole_id)
            sd.wait()
        
        # Play to built-in speaker (for your own feedback)
        print(f"  → Built-in Speaker (your feedback)")
        sd.play(audio_data, samplerate=audio_sr, device=None)  # None = default device
        sd.wait()
        
        print("✓ Playback complete")
        
        # Cleanup
        if os.path.exists(tts_path):
            os.remove(tts_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        # Summary
        print("\n" + "="*60)
        print("PIPELINE TEST COMPLETE")
        print("="*60)
        print(f"STT:   {stt_time:.2f}s")
        print(f"MT:    {mt_time:.2f}s")
        print(f"TTS:   {tts_time:.2f}s")
        print(f"Total: {stt_time + mt_time + tts_time:.2f}s")
        print("="*60)
        
    except Exception as e:
        print(f"✗ Error in MT/TTS: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    try:
        test_pipeline()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()