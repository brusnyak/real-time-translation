import asyncio
import os
import time
import threading
import numpy as np
import soundfile as sf
import sounddevice as sd
from collections import deque
from stt_module import STTModule
from mt_module import MTModule
from tts_module import TTSModule


class LiveTranslationPipeline:
    """
    Live speech translation pipeline with BlackHole output support.
    Streams audio in real-time with concurrent processing.
    """
    
    def __init__(self, 
                 source_lang="en", 
                 target_lang="sk",
                 output_device_name="BlackHole 2ch",  # macOS BlackHole device
                 chunk_duration=2.0):  # Process 2-second chunks
        """
        Args:
            source_lang: Source language ("en")
            target_lang: Target language ("sk")
            output_device_name: Name of virtual audio device (BlackHole on Mac)
            chunk_duration: Duration of audio chunks in seconds
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.output_device_name = output_device_name
        self.chunk_duration = chunk_duration
        self.sample_rate = 16000
        
        print("[PIPELINE] Initializing live translation pipeline...")
        print(f"[PIPELINE] {source_lang} -> {target_lang}")
        print(f"[PIPELINE] Output device: {output_device_name}")
        
        # Initialize modules
        self.stt_module = STTModule(model_size="base", device="cpu", compute_type="int8")
        self.mt_module = MTModule(source_lang=source_lang, target_lang=target_lang, device="mps")
        self.tts_module = TTSModule(
            speaker_reference_path="tests/My test speech_xtts_speaker.wav",
            speaker_language="en",
            device="cpu",
            skip_warmup=False  # Do warmup once
        )
        
        # Find BlackHole device ID
        self.output_device_id = self._find_device_id(output_device_name)
        if self.output_device_id is None:
            print(f"[PIPELINE] Warning: {output_device_name} not found. Will play through default device.")
        
        # Buffers and queues
        self.audio_buffer = deque(maxlen=int(chunk_duration * self.sample_rate))
        self.is_recording = False
        self.is_translating = False
        
        # Timing metrics
        self.metrics = {
            "stt_time": 0,
            "mt_time": 0,
            "tts_time": 0,
            "total_latency": 0
        }
    
    def _find_device_id(self, device_name):
        """Find audio device ID by name."""
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device_name.lower() in device["name"].lower():
                return i
        return None
    
    def start_recording(self):
        """Start recording audio from microphone."""
        if self.is_recording:
            print("[PIPELINE] Already recording.")
            return
        
        self.is_recording = True
        print("[PIPELINE] Starting microphone input...")
        
        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"[RECORDING] Status: {status}")
            self.audio_buffer.extend(indata[:, 0])
        
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=512,
            callback=audio_callback
        )
        self.stream.start()
        print("[PIPELINE] Recording started. Speak now...")
    
    def stop_recording(self):
        """Stop recording."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        self.stream.stop()
        self.stream.close()
        print("[PIPELINE] Recording stopped.")
    
    async def process_audio_chunk(self):
        """Process accumulated audio chunk through STT -> MT -> TTS -> Output."""
        if len(self.audio_buffer) < int(self.chunk_duration * self.sample_rate):
            return None
        
        print(f"[PIPELINE] Processing audio chunk ({len(self.audio_buffer)} samples)...")
        
        # Convert buffer to file
        audio_data = np.array(list(self.audio_buffer), dtype=np.float32)
        chunk_path = "/tmp/live_chunk.wav"
        sf.write(chunk_path, audio_data, self.sample_rate)
        self.audio_buffer.clear()
        
        # STT
        start_stt = time.time()
        try:
            segments, _ = self.stt_module.transcribe(chunk_path, language=self.source_lang)
            transcribed_text = " ".join([seg.text for seg in segments])
            stt_time = time.time() - start_stt
            self.metrics["stt_time"] = stt_time
            
            if not transcribed_text.strip():
                print("[PIPELINE] No speech detected in chunk.")
                os.remove(chunk_path)
                return None
            
            print(f"[STT] Transcribed ({stt_time:.2f}s): {transcribed_text[:50]}...")
        except Exception as e:
            print(f"[STT] Error: {e}")
            os.remove(chunk_path)
            return None
        
        # MT
        start_mt = time.time()
        try:
            translated_text = self.mt_module.translate(transcribed_text)
            translated_text = self.mt_module._fix_slovak_grammar(translated_text)
            mt_time = time.time() - start_mt
            self.metrics["mt_time"] = mt_time
            
            print(f"[MT] Translated ({mt_time:.2f}s): {translated_text[:50]}...")
        except Exception as e:
            print(f"[MT] Error: {e}")
            os.remove(chunk_path)
            return None
        
        # TTS (streaming)
        start_tts = time.time()
        try:
            tts_path = await self.tts_module.synthesize_chunks_concurrently(
                [translated_text],
                crossfade_ms=50
            )
            tts_time = time.time() - start_tts
            self.metrics["tts_time"] = tts_time
            
            print(f"[TTS] Synthesized ({tts_time:.2f}s)")
        except Exception as e:
            print(f"[TTS] Error: {e}")
            os.remove(chunk_path)
            return None
        
        # Play to BlackHole
        self.metrics["total_latency"] = time.time() - start_stt
        await self._play_to_device(tts_path)
        
        # Cleanup
        os.remove(chunk_path)
        if tts_path and os.path.exists(tts_path):
            os.remove(tts_path)
        
        print(f"[PIPELINE] Chunk processed. Total latency: {self.metrics['total_latency']:.2f}s")
        print(f"  STT: {self.metrics['stt_time']:.2f}s | MT: {self.metrics['mt_time']:.2f}s | TTS: {self.metrics['tts_time']:.2f}s")
        
        return {
            "transcribed": transcribed_text,
            "translated": translated_text,
            "latency": self.metrics["total_latency"]
        }
    
    async def _play_to_device(self, audio_file):
        """Play synthesized audio to BlackHole device."""
        try:
            audio_data, sr = sf.read(audio_file, dtype=np.float32)
            
            # Resample if needed
            if sr != self.sample_rate:
                from scipy import signal
                num_samples = int(len(audio_data) * self.sample_rate / sr)
                audio_data = signal.resample(audio_data, num_samples)
            
            # Play in thread to avoid blocking
            def play():
                sd.play(
                    audio_data,
                    samplerate=self.sample_rate,
                    device=self.output_device_id,
                    blocking=True
                )
            
            thread = threading.Thread(target=play, daemon=True)
            thread.start()
            
            print(f"[OUTPUT] Playing to {self.output_device_name}...")
        except Exception as e:
            print(f"[OUTPUT] Error playing audio: {e}")
    
    async def run_live(self):
        """Run live translation in loop."""
        print("[PIPELINE] Starting live translation. Press Ctrl+C to stop.")
        self.start_recording()
        
        try:
            while True:
                await self.process_audio_chunk()
                await asyncio.sleep(1)  # Check every second
        except KeyboardInterrupt:
            print("\n[PIPELINE] Stopping...")
            self.stop_recording()


# Simple CLI wrapper
async def main():
    pipeline = LiveTranslationPipeline(
        source_lang="en",
        target_lang="sk",
        output_device_name="BlackHole 2ch",
        chunk_duration=2.0
    )
    await pipeline.run_live()


if __name__ == "__main__":
    asyncio.run(main())