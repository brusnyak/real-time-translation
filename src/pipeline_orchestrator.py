import asyncio
import os
import time
import datetime
import torch
import torchaudio
import numpy as np
import soundfile as sf
from pydub import AudioSegment, effects
import matplotlib.pyplot as plt
import librosa
import librosa.display
import warnings # Import warnings module
import re # Added import for re

# Set environment variable for MPS fallback before any torch imports
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Suppress specific warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning) # Suppress all FutureWarning

from stt_module import STTModule
from mt_module import MTModule
from tts_module import TTSModule

# Configuration (can be moved to a config file later)
WHISPER_MODEL_SIZE = "base"
XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_SPEAKER_LANGUAGE = "en" # Changed to English, assuming speaker reference is English
SPEAKER_REFERENCE_PATH_M4A = "tests/My test speech.m4a" # Original M4A path
SPEAKER_REFERENCE_PATH_WAV = "tests/My test speech_xtts_speaker.wav" # Assuming this is already WAV or will be converted
OUTPUT_DIR = "research"

# Determine device configurations for each module
if torch.backends.mps.is_available():
    WHISPER_DEVICE = "cpu" # Reverting Faster-Whisper to CPU due to "unsupported device mps" error
    WHISPER_COMPUTE_TYPE = "int8" # Use int8 for CPU
    MT_DEVICE = "mps" # Keep MT on MPS for MarianMT
    TTS_DEVICE = "cpu" # Reverting TTS to CPU for stability as MPS float16 is problematic
    TTS_COMPUTE_TYPE = "int8" # Use int8 for CPU
    print("MPS detected. Faster-Whisper and TTS reverted to CPU. Attempting to use MPS for MT.")
elif torch.cuda.is_available():
    WHISPER_DEVICE = "cuda"
    WHISPER_COMPUTE_TYPE = "float16"
    MT_DEVICE = "cuda"
    TTS_DEVICE = "cuda"
    TTS_COMPUTE_TYPE = "float16"
    print("CUDA detected. All models will use CUDA.")
else:
    WHISPER_DEVICE = "cpu"
    WHISPER_COMPUTE_TYPE = "int8"
    MT_DEVICE = "cpu"
    TTS_DEVICE = "cpu"
    TTS_COMPUTE_TYPE = "int8"
    print("No GPU detected. All models will use CPU.")

class PipelineOrchestrator:
    def __init__(self, source_lang="en", target_lang="sk"):
        self.source_lang = source_lang
        self.target_lang = target_lang

        print(f"[PIPELINE] Initializing for {source_lang} -> {target_lang}...")
        
        # STT Module
        self.stt_module = STTModule(
            model_size=WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE
        )
        
        # MT Module
        mt_device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.mt_module = MTModule(
            source_lang=source_lang,
            target_lang=target_lang,
            device=mt_device
        )
        
        # TTS Module - with skip_warmup=True to save ~4 seconds
        self.tts_module = TTSModule(
            model_name=XTTS_MODEL_NAME,
            speaker_reference_path=SPEAKER_REFERENCE_PATH_WAV,
            speaker_language="cs" if self.target_lang == "sk" else self.target_lang,
            device=TTS_DEVICE,
            compute_type=TTS_COMPUTE_TYPE
        )

        # Queues
        self.stt_output_queue = asyncio.Queue()
        self.tts_input_queue = asyncio.Queue()
        self.tts_output_queue = asyncio.Queue()

        # Timing
        self.timing_info = {
            "total_pipeline_time": 0.0,
            "startup_time": 0.0,
            "whisper_load": 0.0,
            "mt_load": 0.0,
            "tts_load_global": 0.0,
            "stt_time": 0.0,
            "mt_time": 0.0,
            "tts_time": 0.0,
        }
        
        self.all_transcribed_text = []
        self.all_translated_text = []

    async def _stt_worker(self, audio_chunk_path):
        """Worker for Speech-to-Text."""
        start_time = time.time()
        segments, info = self.stt_module.transcribe(audio_chunk_path, language=self.source_lang)
        end_time = time.time()
        self.timing_info["stt_time"] += (end_time - start_time)
        
        full_transcribed_text = " ".join([segment.text for segment in segments])
        self.all_transcribed_text.append(full_transcribed_text) # Store transcribed text
        print(f"STT Worker: Transcribed '{full_transcribed_text[:50]}...'")
        await self.stt_output_queue.put((full_transcribed_text, segments)) # Pass segments for timestamp alignment

    async def _mt_worker(self):
        """Worker for Machine Translation."""
        while True:
            transcribed_text, segments = await self.stt_output_queue.get()
            if transcribed_text is None:
                await self.tts_input_queue.put((None, None))
                break
            
            start_time = time.time()
            translated_text = self.mt_module.translate(transcribed_text)
            translated_text = self.mt_module._fix_slovak_grammar(translated_text)
            end_time = time.time()
            
            self.timing_info["mt_time"] += (end_time - start_time)
            self.all_translated_text.append(translated_text)
            
            print(f"[MT] Translated in {end_time - start_time:.3f}s")
            await self.tts_input_queue.put((translated_text, segments))

    async def _tts_worker(self):
        """Worker for Text-to-Speech with smart chunking."""
        while True:
            translated_text, original_segments = await self.tts_input_queue.get()
            if translated_text is None:
                print("[TTS] Worker stopping.")
                break

            print(f"[TTS] Processing: '{translated_text[:50]}...'")
            start_time = time.time()
            
            # Smart chunking: split by sentences, not arbitrary length
            text_chunks = self._smart_split_text(translated_text)
            print(f"[TTS] Split into {len(text_chunks)} chunks")

            if not text_chunks: # Added check for empty text_chunks
                print("[TTS] No valid text chunks for synthesis, skipping.")
                self.timing_info["tts_time"] += (time.time() - start_time) # Account for time spent
                await self.tts_output_queue.put(None) # Signal no audio produced
                continue # Skip to next iteration

            # Synthesize concurrently
            combined_audio_path = await self.tts_module.synthesize_chunks_concurrently(
                text_chunks,
                crossfade_ms=50
            )
            
            end_time = time.time()
            self.timing_info["tts_time"] += (end_time - start_time)
            
            if combined_audio_path:
                print(f"[TTS] Synthesized in {end_time - start_time:.3f}s")
                await self.tts_output_queue.put(combined_audio_path)
            else:
                print("[TTS] Synthesis failed, skipping.")
        
        await self.tts_output_queue.put(None)

    def _smart_split_text(self, text, max_chars=150):
        """
        Split text into chunks respecting sentence boundaries.
        Avoids misalignment issues.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = ""
        
        for sentence in sentences:
            if len(current) + len(sentence) + 1 > max_chars and current:
                chunks.append(current.strip())
                current = sentence + " "
            else:
                current += sentence + " "
        
        if current:
            chunks.append(current.strip())
        
        # Filter out empty strings or strings containing only punctuation/whitespace
        # This helps prevent TTS errors with non-speech inputs
        filtered_chunks = []
        for chunk in chunks:
            if chunk.strip() and re.search(r'[a-zA-Z0-9]', chunk): # Check for alphanumeric characters
                filtered_chunks.append(chunk)
        
        return filtered_chunks if filtered_chunks else [] # Return empty list if no valid chunks

    def _combine_and_postprocess_chunks(self, chunk_paths):
        """
        Combines synthesized audio chunks with crossfading and applies post-processing.
        """
        combined_audio = AudioSegment.empty()
        crossfade_duration = 100 # milliseconds

        for i, path in enumerate(chunk_paths):
            audio_chunk = AudioSegment.from_wav(path)
            if i > 0:
                combined_audio = combined_audio.append(audio_chunk, crossfade=crossfade_duration)
            else:
                combined_audio = audio_chunk
            os.remove(path) # Clean up individual chunk files

        # Apply overall normalization
        combined_audio = effects.normalize(combined_audio)
        
        # Replace strip_silence() with detect_silence() and manual slicing
        # This is a more efficient way to trim silence as suggested in update3.txt
        silence_thresh = -40 # dB
        min_silence_len = 250 # milliseconds
        
        # Detect non-silent parts
        non_silent_parts = []
        last_end = 0
        for start_i, end_i in combined_audio.detect_silence(min_silence_len=min_silence_len, silence_thresh=silence_thresh):
            if start_i > last_end:
                non_silent_parts.append(combined_audio[last_end:start_i])
            last_end = end_i
        if last_end < len(combined_audio):
            non_silent_parts.append(combined_audio[last_end:])
        
        if non_silent_parts:
            trimmed_audio = sum(non_silent_parts)
        else:
            trimmed_audio = combined_audio # If no non-silent parts, keep original

        final_output_path = os.path.join(OUTPUT_DIR, f"final_synthesized_{time.time()}.wav")
        trimmed_audio.export(final_output_path, format="wav")
        return final_output_path

    def _load_audio_file(self, path):
        """Loads an audio file and resamples it to 16kHz mono."""
        speech, sr = torchaudio.load(path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            speech = resampler(speech)
        if speech.shape[0] > 1: # Convert to mono if stereo
            speech = speech.mean(dim=0, keepdim=True)
        return speech.squeeze().numpy(), 16000

    def _save_audio_file(self, audio, sr, path):
        """Saves audio to a WAV file."""
        sf.write(path, audio, sr)

    def _save_waveform_plot(self, audio_path, sr, output_filepath, title="Audio Waveform"):
        """Generates and saves a waveform plot of the audio."""
        try:
            y, sr = librosa.load(audio_path, sr=sr)
            plt.figure(figsize=(12, 4))
            # Explicitly set color to bypass prop_cycler error
            librosa.display.waveshow(y, sr=sr, color='blue') 
            plt.title(title)
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.tight_layout()
            plt.savefig(output_filepath)
            plt.close()
            print(f"Waveform plot saved to {output_filepath}")
        except Exception as e:
            print(f"Error saving waveform plot for {audio_path}: {e}. This might be due to matplotlib/librosa version incompatibility.")
            print("Please ensure you have compatible versions of matplotlib and librosa installed.")

    async def run_pipeline(self, source_audio_path_m4a):
        start_time_total = time.time()

        # Convert M4A to WAV if necessary for source audio
        source_audio_wav_path = source_audio_path_m4a.replace(".m4a", ".wav")
        if not os.path.exists(source_audio_wav_path):
            print(f"Converting {source_audio_path_m4a} to {source_audio_wav_path}...")
            audio = AudioSegment.from_file(source_audio_path_m4a, format="m4a")
            audio.export(source_audio_wav_path, format="wav")
        
        # Ensure speaker reference WAV exists, convert if M4A is provided
        speaker_ref_wav_path = SPEAKER_REFERENCE_PATH_WAV
        if not os.path.exists(speaker_ref_wav_path):
            # If the WAV doesn't exist, try converting from M4A if available
            if os.path.exists(SPEAKER_REFERENCE_PATH_M4A):
                print(f"Converting {SPEAKER_REFERENCE_PATH_M4A} to {speaker_ref_wav_path} for speaker reference...")
                audio = AudioSegment.from_file(SPEAKER_REFERENCE_PATH_M4A, format="m4a")
                audio.export(speaker_ref_wav_path, format="wav")
            else:
                print(f"Error: Speaker reference WAV not found at {speaker_ref_wav_path} and M4A not found at {SPEAKER_REFERENCE_PATH_M4A}.")
                return # Cannot proceed without speaker reference
        
        # The TTSModule already handles speaker embedding internally, so no need to duplicate here.

        # Start workers
        mt_task = asyncio.create_task(self._mt_worker())
        tts_task = asyncio.create_task(self._tts_worker())

        # Simulate live audio input by processing the entire file as one chunk for now
        # In a true live system, this would be a continuous stream of small audio chunks
        await self._stt_worker(source_audio_wav_path)

        # Signal MT to stop after processing the transcription
        await self.stt_output_queue.put((None, None)) # sentinel for MT
        # DO NOT send sentinel to TTS here — MT will forward it once done

        # Wait for all tasks to complete
        await asyncio.gather(mt_task, tts_task)

        end_time_total = time.time()
        self.timing_info["total_pipeline_time"] = end_time_total - start_time_total
        self.timing_info["whisper_load"] = self.stt_module._load_time
        self.timing_info["mt_load"] = self.mt_module._load_time
        self.timing_info["tts_load_global"] = self.tts_module._load_time
        self.timing_info["startup_time"] = (
            self.timing_info["whisper_load"] + 
            self.timing_info["mt_load"] + 
            self.timing_info["tts_load_global"]
        )

        print("\n--- Pipeline Completed ---")
        print(f"Total Pipeline Time: {self.timing_info['total_pipeline_time']:.3f}s")
        print(f"Startup time (load models + init): {self.timing_info['startup_time']:.3f}s")
        print(f"  - Whisper load: {self.timing_info['whisper_load']:.3f}s")
        print(f"  - MT load: {self.timing_info['mt_load']:.3f}s")
        print(f"  - TTS load (global): {self.timing_info['tts_load_global']:.3f}s")
        print(f"STT Time: {self.timing_info['stt_time']:.3f}s")
        print(f"MT Time: {self.timing_info['mt_time']:.3f}s")
        print(f"TTS Time: {self.timing_info['tts_time']:.3f}s")

        # Retrieve all synthesized audio paths from the queue until sentinel
        all_synthesized_audio_paths = []
        while True:
            path = await self.tts_output_queue.get()
            if path is None: # Sentinel value received
                break
            all_synthesized_audio_paths.append(path)
        
        # If no synthesized audio paths, create a dummy path or handle gracefully
        final_synthesized_audio_path = None
        if all_synthesized_audio_paths:
            # Assuming _combine_and_postprocess_chunks already produced a single final file
            # and put its path into the queue. If multiple chunks were put, this needs adjustment.
            final_synthesized_audio_path = all_synthesized_audio_paths[0]
            # Load final synthesized audio for saving
            final_synthesized_audio, synthesized_sr = self._load_audio_file(final_synthesized_audio_path)
            os.remove(final_synthesized_audio_path) # Clean up final processed audio
        else:
            print("Warning: No synthesized audio chunks were produced by TTS worker. Creating a silent dummy audio file for saving.")
            # Create a silent dummy audio file
            dummy_audio_path = os.path.join(OUTPUT_DIR, f"dummy_silent_audio_{time.time()}.wav")
            sf.write(dummy_audio_path, np.zeros(16000 * 1, dtype=np.float32), 16000) # 1 second of silence
            final_synthesized_audio, synthesized_sr = self._load_audio_file(dummy_audio_path)
            os.remove(dummy_audio_path) # Clean up dummy file after loading to numpy

        # Save artifacts
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        test_dir = os.path.join(OUTPUT_DIR, f"test_{timestamp}")
        os.makedirs(test_dir, exist_ok=True)

        print(f"Saving artifacts to {test_dir}...")
        
        original_audio_output_path = os.path.join(test_dir, "original_audio.wav")
        translated_audio_output_path = os.path.join(test_dir, "translated_audio.wav")

        original_audio_np, original_sr = self._load_audio_file(source_audio_wav_path)
        self._save_audio_file(original_audio_np, original_sr, original_audio_output_path)
        self._save_audio_file(final_synthesized_audio, synthesized_sr, translated_audio_output_path)

        # Save waveform plots
        self._save_waveform_plot(original_audio_output_path, original_sr, os.path.join(test_dir, "original_audio_waveform.png"), "Original Audio Waveform")
        self._save_waveform_plot(translated_audio_output_path, synthesized_sr, os.path.join(test_dir, "translated_audio_waveform.png"), "Translated Audio Waveform")

        # Use actual collected transcribed and translated text
        transcribed_text_final = " ".join(self.all_transcribed_text)
        translated_text_final = " ".join(self.all_translated_text)

        with open(os.path.join(test_dir, "transcript_and_translation.txt"), "w") as f:
            f.write(f"Source Language: {self.source_lang}\n")
            f.write(f"Target Language: {self.target_lang}\n")
            f.write(f"Transcribed Text ({self.source_lang}): {transcribed_text_final}\n")
            f.write(f"Translated Text ({self.target_lang}): {translated_text_final}\n")
            f.write("\n--- Timing Information ---\n")
            f.write(f"Total Pipeline Time: {self.timing_info['total_pipeline_time']:.3f}s\n")
            f.write(f"Startup time (load models + init): {self.timing_info['startup_time']:.3f}s\n")
            f.write(f"  - Whisper load: {self.timing_info['whisper_load']:.3f}s\n")
            f.write(f"  - MT load: {self.timing_info['mt_load']:.3f}s\n")
            f.write(f"  - TTS load (global): {self.timing_info['tts_load_global']:.3f}s\n")
            f.write(f"STT time: {self.timing_info['stt_time']:.3f}s\n")
            f.write(f"MT time: {self.timing_info['mt_time']:.3f}s\n")
            f.write(f"TTS time (generation + post-processing): {self.timing_info['tts_time']:.3f}s\n")

        print("Pipeline completed successfully.")


async def main():
    orchestrator = PipelineOrchestrator(source_lang="en", target_lang="sk")
    await orchestrator.run_pipeline(source_audio_path_m4a="tests/My test speech.m4a")

if __name__ == "__main__":
    import re # Import re here for _split_text_into_chunks
    asyncio.run(main())
