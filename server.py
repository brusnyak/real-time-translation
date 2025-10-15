import asyncio
import os
import time
import threading
import numpy as np
import soundfile as sf
import sounddevice as sd
from collections import deque
import json
import base64  # For proper audio encoding
import webrtcvad  # For voice activity detection
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from src.stt_module import STTModule
from src.mt_module import MTModule
from src.tts_module import TTSModule

# --- Configuration ---
UI_DIR = "ui"
SAMPLE_RATE = 16000
SOURCE_LANG = "en"
TARGET_LANG = "cs"
TTS_MODEL_CHOICE = "xtts_v2"
SPEAKER_REFERENCE_PATH = "tests/My test speech_xtts_speaker.wav"

# --- FastAPI App Setup ---
app = FastAPI()

# Add CORS middleware
origins = [
    "http://localhost:1000",
    "http://0.0.0.0:1000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Pipeline Instance ---
translation_pipeline = None
pipeline_initialized = False

class WebTranslationPipeline:
    """Web-based speech translation pipeline with VAD."""
    
    def __init__(self, 
                 source_lang: str, 
                 target_lang: str,
                 tts_model_choice: str,
                 speaker_reference_path: str):
        
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.tts_model_choice = tts_model_choice
        self.speaker_reference_path = speaker_reference_path
        self.sample_rate = SAMPLE_RATE
        
        # VAD settings
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(1)
        self.frame_duration_ms = 20
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)
        
        # Buffering and timing
        self.audio_buffer = deque()
        self.speech_buffer = []
        self.is_processing = False
        self.processing_task = None
        self.silence_timeout = 1.0  # Seconds of silence before processing
        self.last_speech_time = None
        self.processing_lock = asyncio.Lock()
        self.websocket = None
        
        print("[PIPELINE] Initializing web translation pipeline with VAD...")
        print(f"[PIPELINE] {source_lang} → {target_lang}")
        print(f"[PIPELINE] VAD frame duration: {self.frame_duration_ms}ms")
        print(f"[PIPELINE] Silence timeout: {self.silence_timeout}s")
        
        # Initialize ML modules
        print("\n[LOADING] Initializing models (this takes ~1 minute on first run)...")
        self.stt_module = STTModule(model_size="base", device="cpu", compute_type="int8")
        self.mt_module = MTModule(source_lang=source_lang, target_lang=target_lang, device="mps")
        self.tts_module = TTSModule(
            model_choice=self.tts_model_choice,
            speaker_reference_path=self.speaker_reference_path,
            speaker_language=self.target_lang,
            device="cpu",
            skip_warmup=False
        )
        print("[LOADING] All models loaded successfully.\n")
        
        # Find BlackHole device
        self.blackhole_id = self._find_device_id("BlackHole 2ch")
        if self.blackhole_id is None:
            print("[WARNING] BlackHole 2ch not found. Will output to speakers only.")
        
        # Metrics
        self.metrics = {
            "stt_time": 0,
            "mt_time": 0,
            "tts_time": 0,
            "total_latency": 0
        }
    
    def _find_device_id(self, device_name):
        """Find audio device by name."""
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device_name.lower() in device["name"].lower():
                    return i
        except Exception as e:
            print(f"[ERROR] Error finding audio device: {e}")
        return None
    
    def add_audio_chunk(self, audio_data: np.ndarray):
        """Add incoming audio data to buffer."""
        self.audio_buffer.extend(audio_data)

    async def start_processing(self, websocket: WebSocket):
        """Start the continuous audio processing loop."""
        if self.is_processing:
            return
        self.is_processing = True
        self.websocket = websocket
        self.speech_buffer = []
        self.last_speech_time = None
        print("[PIPELINE] Starting audio processing loop...")
        self.processing_task = asyncio.create_task(self._processing_loop())

    async def stop_processing(self):
        """Stop the continuous audio processing loop."""
        if not self.is_processing:
            return
        self.is_processing = False
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        print("[PIPELINE] Stopped audio processing loop.")
        self.audio_buffer.clear()
        self.speech_buffer = []
        self.last_speech_time = None
        self.websocket = None

    async def _processing_loop(self):
        """The main loop for processing audio chunks."""
        while self.is_processing:
            await self._process_vad_frames()
            await asyncio.sleep(0.05)  # Check every 50ms

    def _extract_frames(self):
        """Extract audio frames from buffer."""
        frames = []
        while len(self.audio_buffer) >= self.frame_size:
            frame = np.array([self.audio_buffer.popleft() for _ in range(self.frame_size)])
            frames.append(frame)
        return frames
    
    def _detect_speech(self, frames):
        """Detect speech in frames using VAD."""
        has_speech = False
        speech_frames = []
        
        for frame in frames:
            frame_int16 = (frame * 32767).astype(np.int16)
            
            try:
                is_speech = self.vad.is_speech(frame_int16.tobytes(), self.sample_rate)
                if is_speech:
                    has_speech = True
                    self.last_speech_time = time.time()
                
                speech_frames.append((frame, is_speech))
            except Exception as e:
                print(f"[VAD] Error: {e}")
                has_speech = True
                self.last_speech_time = time.time()
                speech_frames.append((frame, True))
        
        return has_speech, speech_frames
    
    async def _process_vad_frames(self):
        """Process frames with VAD and handle speech boundaries."""
        frames = self._extract_frames()
        if not frames:
            return
        
        has_speech, speech_frames = self._detect_speech(frames)
        
        # Accumulate speech frames
        for frame, is_speech in speech_frames:
            self.speech_buffer.append(frame)
        
        # Send audio level to UI
        if speech_frames:
            # Calculate RMS of latest frame for UI feedback
            latest_frame = speech_frames[-1][0]
            
            # Apply a gain factor to the frame for better UI visualization
            gain_factor = 10.0 # Adjusted for better UI responsiveness
            gained_frame = np.clip(latest_frame * gain_factor, -1.0, 1.0)
            
            rms = np.sqrt(np.mean(gained_frame**2))
            if self.websocket:
                try:
                    await self.websocket.send_json({"type": "audio_level", "level": float(rms)})
                except:
                    pass
        
        # Check for end-of-speech condition
        if self.last_speech_time is not None:
            silence_duration = time.time() - self.last_speech_time
            
            if has_speech:
                print(f"[VAD] Speech detected, buffer: {len(self.speech_buffer)} frames")
            elif silence_duration > self.silence_timeout and len(self.speech_buffer) > 0:
                print(f"[VAD] Silence timeout ({silence_duration:.2f}s). Processing {len(self.speech_buffer)} frames...")
                
                async with self.processing_lock:
                    audio_data = np.concatenate(self.speech_buffer)
                    await self._process_speech(audio_data)
                    self.speech_buffer = []
                    self.last_speech_time = None
    
    async def _process_speech(self, audio_data):
        """Process speech through STT → MT → TTS → Output."""
        chunk_path = "/tmp/web_chunk.wav"
        sf.write(chunk_path, audio_data, self.sample_rate)
        
        print(f"\n[PROCESSING] {len(audio_data)} samples ({len(audio_data)/self.sample_rate:.1f}s)")
        
        # STT
        start_stt = time.time()
        try:
            segments, _ = self.stt_module.transcribe(chunk_path)
            transcribed = " ".join([seg.text for seg in segments]).strip()
            stt_time = time.time() - start_stt
            self.metrics["stt_time"] = stt_time
            
            if not transcribed:
                print("  [STT] No speech detected")
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
                return
            
            print(f"  [STT] {stt_time:.2f}s → '{transcribed}'")
            if self.websocket:
                try:
                    await self.websocket.send_json({
                        "type": "transcription_result",
                        "transcribed": transcribed,
                        "metrics": {"stt_time": stt_time}
                    })
                except Exception as e:
                    print(f"[PIPELINE] Failed to send transcription: {e}")
        except Exception as e:
            print(f"  [STT] Error: {e}")
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
            return
        
        # MT
        start_mt = time.time()
        try:
            translated = self.mt_module.translate(transcribed)
            translated = self.mt_module._fix_slovak_grammar(translated)
            mt_time = time.time() - start_mt
            self.metrics["mt_time"] = mt_time
            
            print(f"  [MT]  {mt_time:.2f}s → '{translated}'")
            if self.websocket:
                try:
                    await self.websocket.send_json({
                        "type": "translation_result",
                        "translated": translated,
                        "metrics": {"mt_time": mt_time}
                    })
                except Exception as e:
                    print(f"[PIPELINE] Failed to send translation: {e}")
        except Exception as e:
            print(f"  [MT] Error: {e}")
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
            return
        
        # TTS
        start_tts = time.time()
        try:
            tts_path = await self.tts_module.synthesize_chunks_concurrently(
                [translated],
                crossfade_ms=50
            )
            tts_time = time.time() - start_tts
            self.metrics["tts_time"] = tts_time
            
            print(f"  [TTS] {tts_time:.2f}s")
        except Exception as e:
            print(f"  [TTS] Error: {e}")
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
            return
        
        # Play output
        self.metrics["total_latency"] = time.time() - start_stt
        await self._play_audio(tts_path)
        
        # Send final metrics to WebSocket
        if self.websocket:
            try:
                await self.websocket.send_json({
                    "type": "final_metrics",
                    "metrics": self.metrics
                })
            except Exception as e:
                print(f"[PIPELINE] Failed to send final metrics: {e}")
        
        # Cleanup
        if os.path.exists(chunk_path):
            os.remove(chunk_path)
        if tts_path and os.path.exists(tts_path):
            os.remove(tts_path)
        
        print(f"  [TOTAL] {self.metrics['total_latency']:.2f}s")
    
    async def _play_audio(self, audio_file):
        """Play audio to BlackHole and speaker."""
        try:
            audio_data, sr = sf.read(audio_file, dtype=np.float32)
            
            if sr != self.sample_rate:
                from scipy import signal
                num_samples = int(len(audio_data) * self.sample_rate / sr)
                audio_data = signal.resample(audio_data, num_samples)
            
            def play_to_device(device_id, name):
                try:
                    sd.play(audio_data, samplerate=self.sample_rate, device=device_id, blocking=True)
                    print(f"  [OUTPUT] {name} ✓")
                except Exception as e:
                    print(f"  [OUTPUT] {name} Error: {e}")
            
            if self.blackhole_id is not None:
                thread1 = threading.Thread(target=play_to_device, args=(self.blackhole_id, "BlackHole 2ch"), daemon=True)
                thread1.start()
                thread1.join()
            
            thread2 = threading.Thread(target=play_to_device, args=(None, "Speaker"), daemon=True)
            thread2.start()
            thread2.join()
            
        except Exception as e:
            print(f"  [OUTPUT] Error: {e}")

# --- API Endpoints ---

@app.post("/initialize")
async def initialize_pipeline():
    global translation_pipeline, pipeline_initialized
    if not pipeline_initialized:
        try:
            translation_pipeline = WebTranslationPipeline(
                source_lang=SOURCE_LANG,
                target_lang=TARGET_LANG,
                tts_model_choice=TTS_MODEL_CHOICE,
                speaker_reference_path=SPEAKER_REFERENCE_PATH
            )
            pipeline_initialized = True
            return {"status": "success", "message": "Pipeline initialized successfully."}
        except Exception as e:
            print(f"[ERROR] Pipeline initialization failed: {e}")
            return {"status": "error", "message": f"Pipeline initialization failed: {e}"}
    return {"status": "info", "message": "Pipeline already initialized."}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global translation_pipeline, pipeline_initialized

    if not pipeline_initialized or translation_pipeline is None:
        await websocket.send_json({"status": "error", "message": "Pipeline not initialized. Please initialize first."})
        await websocket.close()
        return

    print(f"[WEBSOCKET] Client connected: {websocket.client}")
    try:
        while True:
            message = await websocket.receive()
            
            if "bytes" in message:
                # Decode audio from base64-encoded bytes
                try:
                    # Audio is sent as base64 string of float32 bytes
                    audio_bytes = message["bytes"]
                    
                    # If it's a string, decode from base64
                    if isinstance(audio_bytes, str):
                        audio_bytes = base64.b64decode(audio_bytes)
                    
                    # Convert bytes to float32 numpy array
                    audio_chunk = np.frombuffer(audio_bytes, dtype=np.float32)
                    
                    # Validate audio chunk
                    if len(audio_chunk) > 0 and not np.all(audio_chunk == 0):
                        translation_pipeline.add_audio_chunk(audio_chunk)
                        print(f"[WEBSOCKET] Received audio chunk: {len(audio_chunk)} samples")
                    else:
                        print(f"[WEBSOCKET] Warning: Silent or empty audio chunk received")
                        
                except Exception as e:
                    print(f"[WEBSOCKET] Error decoding audio: {e}")

            elif "text" in message:
                data = json.loads(message["text"])
                if data["type"] == "start":
                    await translation_pipeline.start_processing(websocket)
                    await websocket.send_json({"type": "status", "message": "Processing started."})
                    print("[WEBSOCKET] Processing started")
                elif data["type"] == "stop":
                    await translation_pipeline.stop_processing()
                    await websocket.send_json({"type": "status", "message": "Processing stopped."})
                    print("[WEBSOCKET] Processing stopped")
                elif data["type"] == "config_update":
                    if "source_lang" in data:
                        translation_pipeline.source_lang = data["source_lang"]
                    if "target_lang" in data:
                        translation_pipeline.target_lang = data["target_lang"]
                    if "tts_model_choice" in data:
                        translation_pipeline.tts_model_choice = data["tts_model_choice"]
                    print(f"[WEBSOCKET] Updated config: {data}")
                    await websocket.send_json({"type": "status", "message": "Configuration updated."})
            
    except WebSocketDisconnect:
        print(f"[WEBSOCKET] Client disconnected: {websocket.client}")
    except Exception as e:
        print(f"[WEBSOCKET ERROR] {e}")
    finally:
        if translation_pipeline and translation_pipeline.is_processing:
            await translation_pipeline.stop_processing()

app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")

# --- Main execution for development ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="127.0.0.1",
        port=8000,
        ssl_keyfile="key.pem", 
        ssl_certfile="cert.pem"
    )
