from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import os
import time
import numpy as np
import soundfile as sf
import sounddevice as sd
from collections import deque
from src.stt_module import STTModule
from src.mt_module import MTModule
from src.tts_module import TTSModule
import threading
import re # Import re for text filtering


app = FastAPI()

# Global state
class TranslationState:
    def __init__(self):
        self.is_recording = False
        self.is_translating = False
        self.current_language = "sk"
        self.stt_module = None
        self.mt_module = None
        self.tts_module = None
        self.audio_buffer = deque(maxlen=32000)  # 2 seconds at 16kHz
        self.sample_rate = 16000
        self.chunk_duration = 2.0
        self.stream = None
        self.metrics = {
            "stt_time": 0,
            "mt_time": 0,
            "tts_time": 0,
            "total_latency": 0
        }
        self.websocket = None
    
    def initialize_modules(self):
        """Initialize ML modules"""
        if self.stt_module is None:
            print("[INIT] Loading STT module...")
            self.stt_module = STTModule(model_size="base", device="cpu", compute_type="int8")
        
        if self.mt_module is None:
            print("[INIT] Loading MT module...")
            self.mt_module = MTModule(source_lang="en", target_lang=self.current_language, device="mps")
        
        if self.tts_module is None:
            print("[INIT] Loading TTS module...")
            self.tts_module = TTSModule(
                speaker_reference_path="tests/My test speech_xtts_speaker.wav",
                speaker_language="en",
                device="cpu",
                skip_warmup=False  # Do warmup once
            )
    
    def find_device_id(self, device_name="BlackHole 2ch"):
        """Find audio device ID by name."""
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device_name.lower() in device["name"].lower():
                    return i
        except Exception as e:
            print(f"[AUDIO DEVICE] Error querying devices: {e}")
            pass
        return None
    
    async def send_message(self, msg_type, data):
        """Send message to connected WebSocket client"""
        if self.websocket:
            try:
                await self.websocket.send_json({"type": msg_type, "data": data})
            except Exception as e:
                print(f"[WS] Send error: {e}")


state = TranslationState()


@app.get("/")
async def root():
    """Serve the HTML UI"""
    return FileResponse("ui/index.html")


@app.get("/style.css")
async def get_css():
    """Serve CSS"""
    return FileResponse("ui/style.css")


@app.get("/script.js")
async def get_js():
    """Serve JavaScript"""
    return FileResponse("ui/script.js")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    state.websocket = websocket
    
    try:
        await websocket.send_json({"type": "connected", "data": "Connected to server"})
        
        while True:
            message = await websocket.receive()
            
            if "text" in message:
                data = message["text"]
                parsed_message = json.loads(data)
                
                if parsed_message["type"] == "init":
                    await handle_init(parsed_message)
                elif parsed_message["type"] == "start":
                    await handle_start()
                elif parsed_message["type"] == "stop":
                    await handle_stop()
                elif parsed_message["type"] == "language":
                    state.current_language = parsed_message["data"]
                    # Re-initialize MT module with new target language
                    if state.mt_module:
                        state.mt_module = MTModule(source_lang="en", target_lang=state.current_language, device="mps")
                    await state.send_message("status", f"Language set to {state.current_language}")
            elif "bytes" in message:
                # This is audio data
                audio_bytes = message["bytes"]
                # Convert bytes to numpy array and add to buffer
                audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 0x7FFF
                state.audio_buffer.extend(audio_np)
            else:
                print(f"[WS] Unknown message type: {message}")
    
    except WebSocketDisconnect:
        print("[WS] Client disconnected")
        if state.is_recording:
            await handle_stop()
        state.websocket = None


async def handle_init(message):
    """Initialize modules"""
    await state.send_message("status", "Initializing modules...")
    state.initialize_modules()
    await state.send_message("status", "Ready")
    await state.send_message("init_complete", True)


async def handle_start():
    """Start recording and translation"""
    if state.is_recording:
        await state.send_message("error", "Already recording")
        return
    
    state.is_recording = True
    state.is_translating = True
    await state.send_message("status", "Recording... Speak now")
    
    # Start recording in background
    asyncio.create_task(recording_loop())
    # Start processing loop
    asyncio.create_task(processing_loop())


async def handle_stop():
    """Stop recording and translation"""
    state.is_recording = False
    state.is_translating = False
    if state.stream:
        state.stream.stop()
        state.stream.close()
        state.stream = None
    state.audio_buffer.clear()
    await state.send_message("status", "Stopped")


async def recording_loop():
    """Handle microphone input"""
    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[AUDIO] Status: {status}")
        state.audio_buffer.extend(indata[:, 0])
    
    try:
        state.stream = sd.InputStream(
            samplerate=state.sample_rate,
            channels=1,
            blocksize=512,
            callback=audio_callback
        )
        state.stream.start()
        
        while state.is_recording:
            await asyncio.sleep(0.1)
    except Exception as e:
        print(f"[RECORDING] Error: {e}")
        await state.send_message("error", f"Recording error: {e}")
        state.is_recording = False


async def processing_loop():
    """Process accumulated audio chunks"""
    while state.is_translating:
        if len(state.audio_buffer) >= int(state.chunk_duration * state.sample_rate):
            await process_chunk()
        await asyncio.sleep(0.5)


async def process_chunk():
    """Process one chunk through STT -> MT -> TTS"""
    try:
        # Extract chunk from buffer
        audio_data = np.array(list(state.audio_buffer), dtype=np.float32)
        chunk_path = "/tmp/live_chunk.wav"
        sf.write(chunk_path, audio_data, state.sample_rate)
        state.audio_buffer.clear()
        
        start_time = time.time()
        
        # STT
        start_stt = time.time()
        segments, _ = state.stt_module.transcribe(chunk_path, language="en")
        transcribed_text = " ".join([seg.text for seg in segments])
        state.metrics["stt_time"] = time.time() - start_stt
        
        if not transcribed_text.strip():
            print("[PROCESS] No speech detected in chunk.")
            os.remove(chunk_path)
            return
        
        # MT
        start_mt = time.time()
        translated_text = state.mt_module.translate(transcribed_text)
        translated_text = state.mt_module._fix_slovak_grammar(translated_text)
        state.metrics["mt_time"] = time.time() - start_mt
        
        # Filter translated text for TTS to avoid errors with punctuation-only strings
        if not translated_text.strip() or not re.search(r'[a-zA-Z0-9]', translated_text):
            print("[PROCESS] Translated text is empty or punctuation-only, skipping TTS.")
            os.remove(chunk_path)
            return

        # TTS
        start_tts = time.time()
        tts_path = await state.tts_module.synthesize_chunks_concurrently(
            [translated_text],
            crossfade_ms=50
        )
        state.metrics["tts_time"] = time.time() - start_tts
        
        # Calculate total latency
        state.metrics["total_latency"] = time.time() - start_time
        
        # Send results to UI
        await state.send_message("transcription", {
            "text": transcribed_text,
            "translation": translated_text,
            "metrics": state.metrics
        })
        
        # Play audio in background
        if tts_path and os.path.exists(tts_path):
            def play_audio():
                try:
                    audio, sr = sf.read(tts_path)
                    device_id = state.find_device_id("BlackHole 2ch")
                    sd.play(audio, samplerate=sr, device=device_id, blocking=True)
                except Exception as e:
                    print(f"[OUTPUT] Error playing audio: {e}")
                finally:
                    if os.path.exists(tts_path):
                        os.remove(tts_path)
            
            thread = threading.Thread(target=play_audio, daemon=True)
            thread.start()
        else:
            print("[PROCESS] No TTS audio path, skipping audio playback.")
        
        # Cleanup
        if os.path.exists(chunk_path):
            os.remove(chunk_path)
    
    except Exception as e:
        print(f"[PROCESSING] Error: {e}")
        await state.send_message("error", f"Processing error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("[SERVER] Starting Live Translation Server on http://localhost:8000")
    print("[SERVER] Open your browser and navigate to http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
