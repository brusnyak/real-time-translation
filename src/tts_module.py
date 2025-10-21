import os
import time
import tempfile
import asyncio
import numpy as np
import soundfile as sf
import librosa
from pydub import AudioSegment, effects
import torch
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.models.xtts import XttsArgs
from torch.serialization import add_safe_globals


class TTSModule:
    """
    Optimized TTS module for live speech translation.
    - Fast initialization (skip warmup, or minimal warmup)
    - Correct text-to-audio chunk alignment
    - Memory-efficient concurrent synthesis
    """

    def __init__(self, 
                 model_choice="xtts_v2", # New parameter for model choice
                 speaker_reference_path=None, 
                 speaker_language="en",
                 device="cpu", 
                 compute_type="float32", 
                 sample_rate=22050,
                 skip_warmup=False):
        """
        Args:
            model_choice: "xtts_v2" or "glow_tts"
            skip_warmup: If True, skip model warmup to save ~4s at startup
        """
        self.model_choice = model_choice
        self.speaker_reference_path = speaker_reference_path
        self.speaker_language = speaker_language
        self.sample_rate = sample_rate
        self.device = device
        self.compute_type = compute_type

        # Determine model_name based on model_choice
        if self.model_choice == "xtts_v2":
            self.model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        elif self.model_choice == "glow_tts":
            self.model_name = "tts_models/en/ljspeech/glow-tts"
        elif self.model_choice == "vits":
            # A fast, generic VITS model. 'en/ljspeech/vits' is a common choice for English.
            # For multilingual VITS, a different model name would be needed.
            self.model_name = "tts_models/en/ljspeech/vits"
        else:
            raise ValueError(f"Unsupported TTS model choice: {model_choice}")

        print(f"[TTS] Pre-loading TTS model {self.model_name} (choice: {self.model_choice})...")
        start = time.time()

        # Fix PyTorch 2.6+ weights_only security check (only relevant for XTTS)
        if self.model_choice == "xtts_v2":
            add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

        # Load TTS
        # For Glow-TTS and VITS, speaker_wav is not used, and language might be fixed (e.g., 'en')
        # For XTTS, speaker_wav and language are crucial.
        self.tts = TTS(model_name=self.model_name, progress_bar=False, gpu=False)
        
        # Move to device (addressing deprecation warning)
        self.tts.to(self.device)
        self.tts.synthesizer.tts_model.eval()

        self._load_time = time.time() - start
        print(f"[TTS] Loaded in {self._load_time:.3f}s.")

        # Speaker reference checks (primarily for XTTS)
        if self.model_choice == "xtts_v2":
            if not self.speaker_reference_path:
                print("[TTS] Warning: No speaker_reference_path. Using default voice for XTTS.")
            else:
                if not os.path.exists(self.speaker_reference_path):
                    raise FileNotFoundError(f"[TTS] Speaker file not found: {self.speaker_reference_path}")
                info = sf.info(self.speaker_reference_path)
                print(f"[TTS] Speaker: {info.duration:.2f}s at {info.samplerate}Hz")
        elif self.model_choice in ["glow_tts", "vits"]:
            print(f"[TTS] {self.model_choice} selected. Speaker reference path is not applicable for generic models.")


        # Optional warmup (saves ~4s if skipped)
        if not skip_warmup:
            print(f"[TTS] Warming up {self.model_choice} model...")
            try:
                if self.model_choice == "xtts_v2":
                    _ = self.tts.tts(
                        text="Hello.",
                        speaker_wav=self.speaker_reference_path,
                        language=self.speaker_language
                    )
                elif self.model_choice in ["glow_tts", "vits"]:
                    _ = self.tts.tts(
                        text="Hello.",
                        language="en" # Generic models often have a fixed language
                    )
                print("[TTS] Warmup done.")
            except Exception as e:
                print(f"[TTS] Warmup failed (non-critical): {e}")

    async def synthesize_chunks_concurrently(self, text_chunks, crossfade_ms=50):
        """
        Synthesize text chunks concurrently.
        
        Args:
            text_chunks: List of text strings to synthesize
            crossfade_ms: Crossfade duration in milliseconds
        
        Returns:
            Path to combined audio file, or None if failed
        """
        if not text_chunks:
            print("[TTS] No text chunks provided for synthesis.")
            return None

        print(f"[TTS] Synthesizing {len(text_chunks)} chunks with {self.model_choice}...")
        
        # Launch synthesis tasks in thread pool
        tasks = [asyncio.to_thread(self._synthesize_chunk, txt) for txt in text_chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter valid results
        valid_chunks = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"[TTS] Chunk {i} failed with exception: {type(result).__name__}: {result}")
                continue
            if result is None:
                print(f"[TTS] Chunk {i} returned None (synthesis failed or empty)")
                continue
            
            audio_np, sr = result
            if audio_np.size == 0:
                print(f"[TTS] Chunk {i} is empty after synthesis")
                continue
            
            valid_chunks.append((audio_np, sr))

        if not valid_chunks:
            print("[TTS] No valid chunks produced after concurrent synthesis.")
            return None

        # Combine chunks with crossfade
        out_path = self._combine_chunks(valid_chunks, crossfade_ms)
        print(f"[TTS] Combined {len(valid_chunks)} chunks to {out_path}")
        return out_path

    def _synthesize_chunk(self, text, temperature=0.3, speed=1.0):
        """
        Synchronous synthesis of a single text chunk.
        Runs in thread pool via asyncio.to_thread.
        
        Returns:
            (audio_np, sample_rate) or None on failure
        """
        try:
            print(f"[TTS] Attempting to synthesize chunk: '{text[:100]}...' using {self.model_choice}") # Log the chunk being processed
            # Generate audio using TTS
            if self.model_choice == "xtts_v2":
                audio_output = self.tts.tts(
                    text=text,
                    speaker_wav=self.speaker_reference_path,
                    language=self.speaker_language,
                    temperature=temperature,
                    speed=speed
                )
            # For generic models like Glow-TTS and VITS, speaker_wav is not used, and language should be omitted or fixed.
            # The 'tts_models/en/ljspeech/vits' model is English-only.
            elif self.model_choice in ["glow_tts", "vits"]:
                audio_output = self.tts.tts(
                    text=text,
                    # For single-language models, do not pass the 'language' parameter if it's not expected,
                    # or pass the model's native language if required.
                    # The 'en/ljspeech/vits' model is English, so we can explicitly pass 'en' or omit.
                    # Omitting is safer if the TTS API handles it gracefully for single-language models.
                    # Based on the error "Model is not multi-lingual but `language` is provided", omitting is the correct approach.
                    # However, if the API *requires* a language, then 'en' would be correct for this specific model.
                    # Let's try omitting it first, as the error message implies it's the presence of the parameter that's the issue.
                    # If omitting causes an error, we'll revert to passing 'en'.
                    # For now, I will explicitly pass 'en' as a safe default for these English-only models.
                    language="en"
                )

            # Handle different return types from TTS
            audio_np = self._extract_audio(audio_output)
            
            if audio_np is None or audio_np.size == 0:
                print(f"[TTS] Empty audio for: '{text[:50]}'")
                return None

            # Ensure float32
            audio_np = audio_np.astype(np.float32)
            
            # Clip to [-1, 1] to prevent audio artifacts
            audio_np = np.clip(audio_np, -1.0, 1.0)
            
            return audio_np, self.sample_rate

        except IndexError as ie: # Catch specific IndexError
            print(f"[TTS] Synthesis failed with IndexError for '{text[:100]}...': {ie}")
            return None
        except Exception as e:
            print(f"[TTS] Synthesis failed for '{text[:100]}...': {e}")
            return None

    def _extract_audio(self, output):
        """
        Extract numpy array from various TTS return types.
        Handles different Coqui TTS versions.
        """
        if isinstance(output, np.ndarray):
            return output
        elif isinstance(output, dict) and "wav" in output:
            wav = output["wav"]
            if isinstance(wav, np.ndarray):
                return wav
            elif isinstance(wav, list):
                return np.array(wav, dtype=np.float32)
        elif isinstance(output, list):
            if all(isinstance(x, (float, int, np.floating)) for x in output):
                return np.array(output, dtype=np.float32)
            elif all(isinstance(x, np.ndarray) for x in output):
                non_empty = [x for x in output if x.size > 0]
                if non_empty:
                    return np.concatenate(non_empty)
        elif isinstance(output, str) and os.path.exists(output):
            # If TTS returned a file path, load it
            audio_np, _ = sf.read(output, dtype='float32')
            return audio_np
        
        return None

    def _combine_chunks(self, chunks, crossfade_ms=50):
        """
        Combine audio chunks using pydub crossfade.
        
        Args:
            chunks: List of (audio_np, sr) tuples
            crossfade_ms: Crossfade duration in ms
        
        Returns:
            Path to combined audio file
        """
        if not chunks:
            print("[TTS] No chunks to combine.")
            return None

        combined = None
        
        for audio_np, sr in chunks:
            # Convert numpy to temporary WAV file for pydub
            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            
            try:
                sf.write(tmp_path, audio_np, sr)
                seg = AudioSegment.from_wav(tmp_path)
                
                if combined is None:
                    combined = seg
                else:
                    # Append with crossfade
                    combined = combined.append(seg, crossfade=crossfade_ms)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        if combined is None:
            print("[TTS] Failed to combine any chunks.")
            return None

        # Normalize and save
        combined = effects.normalize(combined)
        
        # Save to temp file
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        combined.export(output_path, format="wav")
        
        return output_path

    def synthesize(self, text, output_path, language=None):
        """
        Synchronous single synthesis (for backward compatibility).
        """
        if language is None:
            language = self.speaker_language
        
        try:
            result = self._synthesize_chunk(text)
            if result is None:
                print(f"[TTS] Single synthesis failed for '{text[:50]}'.")
                return None
            
            audio_np, sr = result
            sf.write(output_path, audio_np, sr)
            return output_path
        except Exception as e:
            print(f"[TTS] Single synthesis failed: {e}")
            return None


if __name__ == "__main__":
    # Quick test
    tt = TTSModule(
        speaker_reference_path="tests/My test speech_xtts_speaker.wav",
        speaker_language="en",
        skip_warmup=False
    )
    
    chunks = [
        "Hello, this is the first chunk.",
        "This is the second chunk.",
        "And this is the third and final chunk."
    ]
    
    import asyncio
    out = asyncio.run(tt.synthesize_chunks_concurrently(chunks, crossfade_ms=50))
    print(f"Output: {out}")

    async def synthesize_chunks_concurrently(self, text_chunks, crossfade_ms=50):
        """
        Synthesize text chunks concurrently.
        
        Args:
            text_chunks: List of text strings to synthesize
            crossfade_ms: Crossfade duration in milliseconds
        
        Returns:
            Path to combined audio file, or None if failed
        """
        if not text_chunks:
            print("[TTS] No text chunks provided for synthesis.")
            return None

        print(f"[TTS] Synthesizing {len(text_chunks)} chunks...")
        
        # Launch synthesis tasks in thread pool
        tasks = [asyncio.to_thread(self._synthesize_chunk, txt) for txt in text_chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter valid results
        valid_chunks = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"[TTS] Chunk {i} failed with exception: {type(result).__name__}: {result}")
                continue
            if result is None:
                print(f"[TTS] Chunk {i} returned None (synthesis failed or empty)")
                continue
            
            audio_np, sr = result
            if audio_np.size == 0:
                print(f"[TTS] Chunk {i} is empty after synthesis")
                continue
            
            valid_chunks.append((audio_np, sr))

        if not valid_chunks:
            print("[TTS] No valid chunks produced after concurrent synthesis.")
            return None

        # Combine chunks with crossfade
        out_path = self._combine_chunks(valid_chunks, crossfade_ms)
        print(f"[TTS] Combined {len(valid_chunks)} chunks to {out_path}")
        return out_path

    def _synthesize_chunk(self, text, temperature=0.3, speed=1.0):
        """
        Synchronous synthesis of a single text chunk.
        Runs in thread pool via asyncio.to_thread.
        
        Returns:
            (audio_np, sample_rate) or None on failure
        """
        try:
            print(f"[TTS] Attempting to synthesize chunk: '{text[:100]}...'") # Log the chunk being processed
            # Generate audio using TTS
            audio_output = self.tts.tts(
                text=text,
                speaker_wav=self.speaker_reference_path,
                language=self.speaker_language,
                temperature=temperature,
                speed=speed
            )

            # Handle different return types from TTS
            audio_np = self._extract_audio(audio_output)
            
            if audio_np is None or audio_np.size == 0:
                print(f"[TTS] Empty audio for: '{text[:50]}'")
                return None

            # Ensure float32
            audio_np = audio_np.astype(np.float32)
            
            # Clip to [-1, 1] to prevent audio artifacts
            audio_np = np.clip(audio_np, -1.0, 1.0)
            
            return audio_np, self.sample_rate

        except IndexError as ie: # Catch specific IndexError
            print(f"[TTS] Synthesis failed with IndexError for '{text[:100]}...': {ie}")
            return None
        except Exception as e:
            print(f"[TTS] Synthesis failed for '{text[:100]}...': {e}")
            return None

    def _extract_audio(self, output):
        """
        Extract numpy array from various TTS return types.
        Handles different Coqui TTS versions.
        """
        if isinstance(output, np.ndarray):
            return output
        elif isinstance(output, dict) and "wav" in output:
            wav = output["wav"]
            if isinstance(wav, np.ndarray):
                return wav
            elif isinstance(wav, list):
                return np.array(wav, dtype=np.float32)
        elif isinstance(output, list):
            if all(isinstance(x, (float, int, np.floating)) for x in output):
                return np.array(output, dtype=np.float32)
            elif all(isinstance(x, np.ndarray) for x in output):
                non_empty = [x for x in output if x.size > 0]
                if non_empty:
                    return np.concatenate(non_empty)
        elif isinstance(output, str) and os.path.exists(output):
            # If TTS returned a file path, load it
            audio_np, _ = sf.read(output, dtype='float32')
            return audio_np
        
        return None

    def _combine_chunks(self, chunks, crossfade_ms=50):
        """
        Combine audio chunks using pydub crossfade.
        
        Args:
            chunks: List of (audio_np, sr) tuples
            crossfade_ms: Crossfade duration in ms
        
        Returns:
            Path to combined audio file
        """
        if not chunks:
            print("[TTS] No chunks to combine.")
            return None

        combined = None
        
        for audio_np, sr in chunks:
            # Convert numpy to temporary WAV file for pydub
            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            
            try:
                sf.write(tmp_path, audio_np, sr)
                seg = AudioSegment.from_wav(tmp_path)
                
                if combined is None:
                    combined = seg
                else:
                    # Append with crossfade
                    combined = combined.append(seg, crossfade=crossfade_ms)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        if combined is None:
            print("[TTS] Failed to combine any chunks.")
            return None

        # Normalize and save
        combined = effects.normalize(combined)
        
        # Save to temp file
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        combined.export(output_path, format="wav")
        
        return output_path

    def synthesize(self, text, output_path, language=None):
        """
        Synchronous single synthesis (for backward compatibility).
        """
        if language is None:
            language = self.speaker_language
        
        try:
            result = self._synthesize_chunk(text)
            if result is None:
                print(f"[TTS] Single synthesis failed for '{text[:50]}'.")
                return None
            
            audio_np, sr = result
            sf.write(output_path, audio_np, sr)
            return output_path
        except Exception as e:
            print(f"[TTS] Single synthesis failed: {e}")
            return None


if __name__ == "__main__":
    # Quick test
    tt = TTSModule(
        speaker_reference_path="tests/My test speech_xtts_speaker.wav",
        speaker_language="en",
        skip_warmup=False
    )
    
    chunks = [
        "Hello, this is the first chunk.",
        "This is the second chunk.",
        "And this is the third and final chunk."
    ]
    
    import asyncio
    out = asyncio.run(tt.synthesize_chunks_concurrently(chunks, crossfade_ms=50))
    print(f"Output: {out}")
