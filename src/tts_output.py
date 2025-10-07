import numpy as np
import sounddevice as sd
import os
import threading
import time
import torch
from TTS.api import TTS
import pyttsx3 # Import pyttsx3 for fallback
import wave # For pyttsx3 output handling
import io
from pydub import AudioSegment # Import pydub

class TTSOutput:
    def __init__(self, speaker_wav_path="tests/My test speech.m4a"):
        """
        Initializes the Coqui XTTS v2 TTS model and pyttsx3 for fallback.
        :param speaker_wav_path: Path to the audio file for voice cloning with XTTS v2.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Initialize XTTS model
        self.xtts_supported_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh-cn', 'hu', 'ko', 'ja', 'hi']
        self.tts_xtts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True if self.device == "cuda" else False)
        print("Coqui XTTS v2 model initialized.")

        # Convert speaker_wav_path to WAV if it's not already
        self.xtts_speaker_wav_path = self._prepare_speaker_wav(speaker_wav_path)
        if not os.path.exists(self.xtts_speaker_wav_path):
            print(f"Warning: Prepared speaker WAV file not found at {self.xtts_speaker_wav_path}. XTTS v2 voice cloning may not work as expected.")

        # Initialize pyttsx3 engine for fallback (e.g., for Slovak)
        try:
            self.tts_pyttsx3 = pyttsx3.init('espeak')
            print("pyttsx3 engine initialized with 'espeak' driver for fallback.")
        except Exception as e:
            print(f"Warning: Failed to initialize pyttsx3 with 'espeak' driver: {e}. Falling back to default.")
            print("Please ensure 'espeak-ng' is installed on your system (e.g., 'brew install espeak-ng' on macOS, 'sudo apt-get install espeak-ng' on Debian/Ubuntu).")
            self.tts_pyttsx3 = pyttsx3.init()
            print("pyttsx3 engine initialized with default driver (fallback).")
        
        self.tts_pyttsx3.setProperty('rate', 170) # Default rate
        self.tts_pyttsx3.setProperty('volume', 1.0) # Default volume
        self.pyttsx3_voices = self.tts_pyttsx3.getProperty('voices')

        self.sk_voice_id = None
        for voice in self.pyttsx3_voices:
            if "sk" in voice.languages:
                self.sk_voice_id = voice.id
                break
        if not self.sk_voice_id:
            print("Warning: No suitable Slovak voice found for pyttsx3. Falling back to first available.")
            if self.pyttsx3_voices:
                self.sk_voice_id = self.pyttsx3_voices[0].id
        
        if self.sk_voice_id:
            self.tts_pyttsx3.setProperty('voice', self.sk_voice_id)
            print(f"pyttsx3 Slovak voice set to: {self.sk_voice_id}")
        else:
            print("No suitable Slovak voice found for pyttsx3, using system default.")

    def _prepare_speaker_wav(self, original_path: str) -> str:
        """
        Converts the speaker WAV file to a compatible WAV format for XTTS v2 if necessary.
        Returns the path to the prepared WAV file.
        """
        base, ext = os.path.splitext(original_path)
        if ext.lower() == ".wav":
            return original_path # Already a WAV file

        # Convert to WAV
        output_path = base + "_xtts_speaker.wav"
        try:
            audio = AudioSegment.from_file(original_path)
            audio.export(output_path, format="wav")
            print(f"Converted speaker WAV from {original_path} to {output_path} for XTTS v2.")
            return output_path
        except Exception as e:
            print(f"Error converting speaker WAV for XTTS v2: {e}. Using original path, which may cause issues.")
            return original_path


    def synthesize_and_play(self, text: str, lang="en"):
        """
        Synthesizes text to speech and plays it directly, using XTTS v2 or pyttsx3.
        """
        print(f"Synthesizing and playing: '{text}' in language '{lang}'...")
        try:
            audio_bytes, sample_rate = self.get_audio_bytes(text, lang)
            if audio_bytes:
                audio_array_to_play = np.frombuffer(audio_bytes, dtype=np.int16)
                threading.Thread(target=sd.play, args=(audio_array_to_play, sample_rate)).start()
                print("Playback started (non-blocking).")
        except Exception as e:
            print(f"Error during non-blocking playback: {e}")

    def synthesize_to_numpy(self, text: str, lang="en") -> tuple[np.ndarray, int]:
        """
        Synthesizes text to speech and returns a numpy array and sample rate, using XTTS v2 or pyttsx3.
        """
        if lang in self.xtts_supported_languages:
            try:
                print(f"Using XTTS v2 for language: {lang}")
                wav = self.tts_xtts.tts(
                    text=text,
                    speaker_wav=self.xtts_speaker_wav_path, # Use the prepared WAV path
                    language=lang
                )
                return np.array(wav), 24000 # XTTS v2 default sample rate
            except Exception as e:
                print(f"Error during XTTS v2 synthesis for {lang}: {e}. Falling back to pyttsx3.")
                return self._synthesize_with_pyttsx3(text, lang)
        else:
            print(f"Language {lang} not supported by XTTS v2. Falling back to pyttsx3.")
            return self._synthesize_with_pyttsx3(text, lang)

    def _synthesize_with_pyttsx3(self, text: str, lang="en") -> tuple[np.ndarray, int]:
        """
        Internal method to synthesize text using pyttsx3.
        """
        temp_filename = "temp_pyttsx3_output.wav"
        original_voice = self.tts_pyttsx3.getProperty('voice')
        
        if lang == "sk" and self.sk_voice_id:
            self.tts_pyttsx3.setProperty('voice', self.sk_voice_id)
        # Add other pyttsx3 voice selections if needed for other languages
        
        self.tts_pyttsx3.save_to_file(text, temp_filename)
        self.tts_pyttsx3.runAndWait()
        
        # Add a small delay to ensure file system sync
        time.sleep(0.5)

        max_retries = 5
        for i in range(max_retries):
            if os.path.exists(temp_filename):
                try:
                    with wave.open(temp_filename, 'rb') as wf:
                        sample_rate = wf.getframerate()
                        audio_bytes = wf.readframes(wf.getnframes())
                    os.remove(temp_filename)
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    self.tts_pyttsx3.setProperty('voice', original_voice) # Restore original voice
                    return audio_array, sample_rate
                except wave.Error as e:
                    print(f"Warning: Attempt {i+1}/{max_retries} to read WAV failed: {e}. Retrying...")
                    time.sleep(0.2)
            else:
                print(f"Warning: Attempt {i+1}/{max_retries}, file '{temp_filename}' not found. Retrying...")
                time.sleep(0.2)
        
        self.tts_pyttsx3.setProperty('voice', original_voice) # Restore original voice
        raise FileNotFoundError(f"Failed to create or read temporary pyttsx3 file '{temp_filename}' after {max_retries} attempts.")


    def synthesize_to_file(self, text: str, filename="output_tts.wav", lang="en"):
        """
        Synthesizes text to speech and saves it to a WAV file, using XTTS v2 or pyttsx3.
        """
        if lang in self.xtts_supported_languages:
            print(f"Using XTTS v2 to save to file: '{text}' to '{filename}' in language '{lang}'...")
            try:
                self.tts_xtts.tts_to_file(
                    text=text,
                    speaker_wav=self.xtts_speaker_wav_path, # Use the prepared WAV path
                    language=lang,
                    file_path=filename
                )
                print(f"Audio saved to {filename} using XTTS v2.")
                return filename
            except Exception as e:
                print(f"Error saving XTTS v2 audio to file for {lang}: {e}. Falling back to pyttsx3.")
                return self._synthesize_to_file_pyttsx3(text, filename, lang)
        else:
            print(f"Language {lang} not supported by XTTS v2. Falling back to pyttsx3 to save to file.")
            return self._synthesize_to_file_pyttsx3(text, filename, lang)

    def _synthesize_to_file_pyttsx3(self, text: str, filename="output_tts.wav", lang="en"):
        """
        Internal method to synthesize text to file using pyttsx3.
        """
        original_voice = self.tts_pyttsx3.getProperty('voice')
        if lang == "sk" and self.sk_voice_id:
            self.tts_pyttsx3.setProperty('voice', self.sk_voice_id)
        
        self.tts_pyttsx3.save_to_file(text, filename)
        self.tts_pyttsx3.runAndWait()
        print(f"Audio saved to {filename} using pyttsx3.")
        self.tts_pyttsx3.setProperty('voice', original_voice) # Restore original voice
        return filename

    def get_audio_bytes(self, text: str, lang="en") -> tuple[bytes, int]:
        """
        Synthesizes text to speech and returns raw audio bytes and sample rate, using XTTS v2 or pyttsx3.
        """
        audio_array, sample_rate = self.synthesize_to_numpy(text, lang)
        if audio_array is not None:
            # Convert numpy array (float32) to bytes (int16)
            audio_bytes = (audio_array * 32767).astype(np.int16).tobytes()
            return audio_bytes, sample_rate
        return b"", 0

if __name__ == "__main__":
    # Ensure you have a speaker_wav file for cloning, e.g., "tests/My test speech.m4a"
    tts_model = TTSOutput(speaker_wav_path="tests/My test speech.m4a")

    # Example: Synthesize and play a sentence in English (XTTS v2)
    tts_model.synthesize_and_play("Hello, this is a test of the Coqui XTTS v2 library in English.", lang="en")
    time.sleep(5)

    # Example: Synthesize and play a sentence in Czech (XTTS v2)
    tts_model.synthesize_and_play("Ahoj, toto je test Coqui XTTS v2 knižnice v češtine.", lang="cs")
    time.sleep(5)

    # Example: Synthesize and play a sentence in Slovak (pyttsx3 fallback)
    tts_model.synthesize_and_play("Ahoj, toto je test pyttsx3 knižnice v slovenčine.", lang="sk")
    time.sleep(5)

    # Example: Synthesize to a file in English (XTTS v2)
    output_file_en = tts_model.synthesize_to_file("This is a test sentence saved to a file using XTTS v2 in English.", "xtts_output_en.wav", lang="en")
    
    # Example: Synthesize to a file in Slovak (pyttsx3 fallback)
    output_file_sk = tts_model.synthesize_to_file("Toto je testovacia veta uložená do súboru pomocou pyttsx3 v slovenčine.", "pyttsx3_output_sk.wav", lang="sk")

    # Example: Get audio bytes (demonstrates how to get raw audio for streaming)
    try:
        audio_bytes_en, sample_rate_en = tts_model.get_audio_bytes("This is audio as bytes from XTTS v2 in English.", lang="en")
        print(f"Received {len(audio_bytes_en)} bytes of English audio data with sample rate {sample_rate_en}.")
        
        audio_bytes_sk, sample_rate_sk = tts_model.get_audio_bytes("Toto je zvuk ako bajty z pyttsx3 v slovenčine.", lang="sk")
        print(f"Received {len(audio_bytes_sk)} bytes of Slovak audio data with sample rate {sample_rate_sk}.")

        # Play the bytes_output.wav using AudioInput class (assuming it can handle raw bytes or a temp file)
        from audio_input import AudioInput
        audio_manager = AudioInput()

        # Play English bytes
        temp_play_file_en = "temp_playback_xtts_en.wav"
        audio_array_int16_en = np.frombuffer(audio_bytes_en, dtype=np.int16)
        with wave.open(temp_play_file_en, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate_en)
            wf.writeframes(audio_array_int16_en.tobytes())
        audio_manager.play_audio(temp_play_file_en)
        os.remove(temp_play_file_en)

        # Play Slovak bytes
        temp_play_file_sk = "temp_playback_pyttsx3_sk.wav"
        audio_array_int16_sk = np.frombuffer(audio_bytes_sk, dtype=np.int16)
        with wave.open(temp_play_file_sk, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate_sk)
            wf.writeframes(audio_array_int16_sk.tobytes())
        audio_manager.play_audio(temp_play_file_sk)
        os.remove(temp_play_file_sk)

    except Exception as e:
        print(f"Error getting audio bytes or playing: {e}")
