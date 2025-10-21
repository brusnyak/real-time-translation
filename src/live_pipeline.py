import os
import sys
import time
import queue
import sounddevice as sd
import numpy as np
import soundfile as sf
import threading
import uuid
import webrtcvad
from scipy import signal
import datetime # Added import for datetime
import torch # Added import for torch to check GPU availability
import warnings # Added import for warnings

# Add the parent directory of 'src' to sys.path to allow absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Suppress the specific pkg_resources deprecation warning from webrtcvad
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
    module="webrtcvad"
)

from src.stt_module import STTModule
from src.mt_module import MTModule
from src.tts_module import TTSModule


class LivePipeline:
    def __init__(
        self,
        input_device=1, # MacBook Pro Microphone
        output_device=None,
        blackhole_name="BlackHole 2ch",
        input_sample_rate=16000, # Native sample rate of MacBook Pro Microphone
        target_processing_sample_rate=16000, # VAD and STT models typically use 16kHz
        frame_duration=30, # ms per frame for VAD
        vad_aggressiveness=2,
        min_speech_duration=1.0,
        silence_timeout=0.8,
        target_lang="sk",
        speaker_reference_path="tests/My test speech_xtts_speaker.wav",
        speaker_language="en",
        tts_model_choice="xtts_v2", # Added tts_model_choice parameter
    ):
        # Audio configuration
        self.input_sample_rate = input_sample_rate
        self.target_processing_sample_rate = target_processing_sample_rate
        self.frame_duration = frame_duration  # ms per frame
        self.frame_size = int(self.target_processing_sample_rate * frame_duration / 1000)
        self.channels = 1

        # Devices
        self.input_device = input_device
        self.output_device = output_device
        self.blackhole_id = self._find_device_id(blackhole_name)

        # Voice Activity Detection
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.audio_buffer = queue.Queue()
        self.last_activity = time.time()
        self.consecutive_silence_frames = 0

        # Determine device for models
        device = "cpu"
        if torch.backends.mps.is_available():
            device = "mps"
            print("[INFO] Using MPS (Apple Silicon GPU) for models.")
        elif torch.cuda.is_available():
            device = "cuda"
            print("[INFO] Using CUDA (NVIDIA GPU) for models.")
        else:
            print("[INFO] No GPU detected, using CPU for models.")

        # Components
        # STTModule does not support MPS, so force CPU
        self.stt = STTModule(device="cpu")
        self.mt = MTModule(device=device)
        # TTSModule also has MPS issues with attention mask, so force CPU for now
        self.tts = TTSModule(
            model_choice=tts_model_choice, # Pass the model choice
            speaker_reference_path=speaker_reference_path,
            speaker_language=speaker_language,
            device="cpu",
            compute_type="float32", # Reverting to float32 as int8 did not improve performance
            skip_warmup=False
        )
        self.target_lang = target_lang

        # Parameters
        self.min_speech_duration = min_speech_duration
        self.silence_timeout = silence_timeout

        # Metrics
        self.total_chunks = 0
        self.total_time = 0.0

        print(f"[INIT] LivePipeline ready — target: {target_lang}")

    # --- DEVICE UTILITIES ---

    def _find_device_id(self, name):
        """Find a device ID by (partial) name match."""
        try:
            devices = sd.query_devices()
            for idx, d in enumerate(devices):
                if name.lower() in d["name"].lower():
                    print(f"[AUDIO] Found device '{name}' at ID {idx}")
                    return idx
            print(f"[AUDIO] Device '{name}' not found.")
        except Exception as e:
            print(f"[AUDIO] Query error: {e}")
        return None

    # --- RECORDING & CALLBACK ---

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print("[WARN]", status)

        # Convert to mono, normalize
        mono = np.mean(indata, axis=1) if indata.ndim > 1 else indata
        mono = np.clip(mono, -1.0, 1.0)

        # Resample to target_processing_sample_rate (16kHz) for VAD and STT
        if self.input_sample_rate != self.target_processing_sample_rate:
            num_output_samples = int(len(mono) * self.target_processing_sample_rate / self.input_sample_rate)
            mono = signal.resample(mono, num_output_samples)
        
        # Apply a gain factor (if needed, adjust this value based on testing)
        # Reduced gain to mitigate feedback issues. Adjust as needed.
        gain_factor = 1.0
        mono *= gain_factor
        mono = np.clip(mono, -1.0, 1.0) # Re-clip after applying gain

        self.audio_buffer.put_nowait(mono.copy())

    # --- MAIN LOOP ---

    def start(self):
        print(f"[START] Capturing from: {sd.query_devices(self.input_device)['name']} (ID: {self.input_device})")
        print(f"[INFO] Input SR = {self.input_sample_rate} Hz, Processing SR = {self.target_processing_sample_rate} Hz")
        print(f"[INFO] VAD frame = {self.frame_duration}ms")

        with sd.InputStream(
            device=self.input_device,
            channels=self.channels,
            samplerate=self.input_sample_rate,
            callback=self._audio_callback,
        ):
            active_frames = []
            continuous_buffer = np.array([]) # Buffer to accumulate audio for fixed-size frames
            print("[INFO] Listening... (Ctrl+C to stop)")
            while True:
                try:
                    # Get all available audio from the buffer
                    while not self.audio_buffer.empty():
                        continuous_buffer = np.concatenate((continuous_buffer, self.audio_buffer.get_nowait()))
                except queue.Empty:
                    pass # No new audio, continue processing existing buffer

                # Process continuous_buffer in fixed-size frames
                while len(continuous_buffer) >= self.frame_size:
                    frame = continuous_buffer[:self.frame_size]
                    continuous_buffer = continuous_buffer[self.frame_size:]

                    if self._detect_speech(frame):
                        active_frames.append(frame)
                        self.last_activity = time.time()
                        print("•", end="", flush=True)  # speech indicator
                    else:
                        print(".", end="", flush=True)
                        if active_frames and (time.time() - self.last_activity) > self.silence_timeout:
                            print("\n[SEGMENT] Speech detected, processing chunk...")
                            self._process_chunk(np.concatenate(active_frames))
                            active_frames.clear()
                            self.last_activity = time.time()
                
                time.sleep(0.01) # Small sleep to prevent busy-waiting if no new audio

    # --- SPEECH DETECTION ---

    def _detect_speech(self, frame):
        """Return True if speech is detected in frame."""
        frame_int16 = np.clip(frame * 32767, -32768, 32767).astype(np.int16)
        
        try:
            is_speech = self.vad.is_speech(frame_int16.tobytes(), self.target_processing_sample_rate)
            return is_speech
        except Exception as e:
            print(f"[WARN] VAD error: {e}") # Changed to WARN as it's not fatal
            return False

    # --- PROCESSING CHAIN ---

    def _process_chunk(self, audio_data):
        """STT → Translate → TTS → Play"""
        start = time.time()
        self.total_chunks += 1

        if len(audio_data) < self.target_processing_sample_rate * self.min_speech_duration:
            print("[PROCESS] Too short, skipping.")
            return

        # Save temp WAV
        chunk_path = f"/tmp/live_chunk_{uuid.uuid4().hex}.wav"
        sf.write(chunk_path, audio_data, self.target_processing_sample_rate)
        print(f"[PROCESS] Saved {chunk_path}")

        # Step 1: Speech to text
        segments, info = self.stt.transcribe(chunk_path)
        text = " ".join([segment.text for segment in segments])
        print("[STT]", text)
        if not text.strip():
            return

        # Step 2: Translation
        translated = self.mt.translate(text)
        print(f"[MT] {translated}")

        # Step 3: Text to speech
        # Generate a unique directory for this chunk's artifacts
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_chunk_dir = os.path.join("research", f"test_{timestamp}")
        os.makedirs(output_chunk_dir, exist_ok=True)

        # Save transcribed text
        with open(os.path.join(output_chunk_dir, "transcribed_text.txt"), "w") as f:
            f.write(text)
        print(f"[PROCESS] Saved transcribed text to {output_chunk_dir}/transcribed_text.txt")

        # Save translated text
        with open(os.path.join(output_chunk_dir, "translated_text.txt"), "w") as f:
            f.write(translated)
        print(f"[PROCESS] Saved translated text to {output_chunk_dir}/translated_text.txt")

        # Generate a path for TTS output within the chunk directory
        tts_output_filepath = os.path.join(output_chunk_dir, "translated_audio.wav")
        tts_audio_path = self.tts.synthesize(translated, output_path=tts_output_filepath)
        
        if tts_audio_path:
            print(f"[TTS] Saved {tts_audio_path}")
            # Step 4: Playback
            self._play_audio_nonblocking(tts_audio_path)
        else:
            print("[TTS] TTS synthesis failed, no audio to play.")
            return

        elapsed = time.time() - start
        self.total_time += elapsed
        avg_time = self.total_time / self.total_chunks
        print(f"[METRICS] Chunk {self.total_chunks} done in {elapsed:.2f}s (avg {avg_time:.2f}s)")

    # --- AUDIO PLAYBACK ---

    def _play_audio_nonblocking(self, audio_file_path):
        """Play TTS audio to BlackHole (with fallback to speakers)."""
        def playback_thread(file_path, blackhole_id):
            try:
                data, sr = sf.read(file_path, dtype='float32')
                
                played = False
                if blackhole_id is not None:
                    try:
                        sd.play(data, samplerate=sr, device=blackhole_id)
                        sd.wait()
                        print("[OUTPUT] BlackHole ✓")
                        played = True
                    except Exception as e:
                        print(f"[OUTPUT] BlackHole error: {e}")

                if not played:
                    try:
                        sd.play(data, samplerate=sr)
                        sd.wait()
                        print("[OUTPUT] Speaker ✓")
                    except Exception as e:
                        print(f"[OUTPUT] Speaker error: {e}")

            except Exception as e:
                print(f"[OUTPUT] Playback error: {e}")
            # Removed finally block to prevent deletion of persistently saved files
        threading.Thread(
            target=playback_thread,
            args=(audio_file_path, self.blackhole_id),
            daemon=True,
        ).start()


# --- RUN MAIN ---
if __name__ == "__main__":
    try:
        pipeline = LivePipeline(
            input_device=1, # MacBook Pro Microphone
            input_sample_rate=48000,
            target_processing_sample_rate=16000,
            target_lang="sk",
            vad_aggressiveness=2,
            speaker_reference_path="tests/My test speech_xtts_speaker.wav",
            speaker_language="en",
            tts_model_choice="xtts_v2", # Default for main execution
        )
        pipeline.start()
    except KeyboardInterrupt:
        print("\n[EXIT] Stopped by user.")
    except Exception as e:
        print(f"[FATAL] {e}")
