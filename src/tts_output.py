import pyttsx3
import numpy as np
import sounddevice as sd
import wave
import os
import threading
import time

class TTSOutput:
    def __init__(self, rate=150, volume=1.0):
        """
        Initializes the pyttsx3 TTS engine.
        :param rate: Speech rate (words per minute).
        :param volume: Speech volume (0.0 to 1.0).
        """
        # Initialize pyttsx3 engine, explicitly trying 'espeak' driver for better WAV compatibility
        try:
            self.engine = pyttsx3.init('espeak')
            print("pyttsx3 engine initialized with 'espeak' driver.")
        except Exception as e:
            print(f"Warning: Failed to initialize pyttsx3 with 'espeak' driver: {e}. Falling back to default.")
            print("Please ensure 'espeak-ng' is installed on your system (e.g., 'brew install espeak-ng' on macOS, 'sudo apt-get install espeak-ng' on Debian/Ubuntu).")
            self.engine = pyttsx3.init()
            print("pyttsx3 engine initialized with default driver (fallback).")

        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        self.voices = self.engine.getProperty('voices')
        
        self.en_voice_id = None
        self.sk_voice_id = None
        for voice in self.voices:
            if "en-us" in voice.languages: # Prioritize US English
                self.en_voice_id = voice.id
            elif "en" in voice.languages and not self.en_voice_id: # Fallback to general English
                self.en_voice_id = voice.id
            if "sk" in voice.languages:
                self.sk_voice_id = voice.id
            
            if self.en_voice_id and self.sk_voice_id:
                break

        if not self.en_voice_id:
            print("Warning: No suitable English voice found. Falling back to first available for English.")
            for voice in self.voices:
                if "en" in voice.languages:
                    self.en_voice_id = voice.id
                    break
            if not self.en_voice_id and self.voices:
                self.en_voice_id = self.voices[0].id # Absolute fallback

        if not self.sk_voice_id:
            print("Warning: No suitable Slovak voice found. Falling back to first available for Slovak.")
            for voice in self.voices:
                if "sk" in voice.languages:
                    self.sk_voice_id = voice.id
                    break
            if not self.sk_voice_id and self.voices:
                self.sk_voice_id = self.voices[0].id # Absolute fallback

        print(f"English voice set to: {self.en_voice_id}")
        print(f"Slovak voice set to: {self.sk_voice_id}")
        # Set initial voice (can be changed later based on target language)
        if self.sk_voice_id:
            self.engine.setProperty('voice', self.sk_voice_id)
        elif self.en_voice_id:
            self.engine.setProperty('voice', self.en_voice_id)
        else:
            print("No suitable English or Slovak voice found, using system default.")
            if self.voices:
                self.engine.setProperty('voice', self.voices[0].id) # Fallback to first available
                print(f"Fallback voice set to: {self.voices[0].id}")

    def list_voices(self):
        """Lists available voices on the system."""
        print("Available voices:")
        for i, voice in enumerate(self.voices):
            print(f"{i}: ID={voice.id}, Name={voice.name}, Langs={voice.languages}, Gender={voice.gender}, Age={voice.age}")

    def set_voice(self, voice_id):
        """Sets the TTS voice by ID."""
        self.engine.setProperty('voice', voice_id)
        print(f"Voice set to: {voice_id}")

    def synthesize_and_play(self, text: str, lang="en"):
        """
        Synthesizes text to speech and plays it directly.
        Note: pyttsx3's `say` and `runAndWait` are blocking.
        For non-blocking playback, we'll use get_audio_bytes and play with sounddevice.
        """
        print(f"Synthesizing and playing: '{text}' in language '{lang}'...")
        
        original_voice = self.engine.getProperty('voice')
        if lang == "en" and self.en_voice_id:
            self.engine.setProperty('voice', self.en_voice_id)
        elif lang == "sk" and self.sk_voice_id:
            self.engine.setProperty('voice', self.sk_voice_id)
        
        try:
            audio_bytes, sample_rate = self.get_audio_bytes(text, lang)
            if audio_bytes:
                audio_array_to_play = np.frombuffer(audio_bytes, dtype=np.int16)
                threading.Thread(target=sd.play, args=(audio_array_to_play, sample_rate)).start()
                print("Playback started (non-blocking).")
        except Exception as e:
            print(f"Error during non-blocking playback: {e}")
        finally:
            self.engine.setProperty('voice', original_voice) # Restore original voice

    def synthesize_to_file(self, text: str, filename="output_tts.wav", lang="en"):
        """
        Synthesizes text to speech and saves it to a WAV file.
        :param text: The text to synthesize.
        :param filename: The output WAV file path.
        :param lang: Language of the text (used for voice selection if implemented).
        """
        print(f"Synthesizing to file: '{text}' to '{filename}' in language '{lang}'...")
        self.engine.save_to_file(text, filename)
        self.engine.runAndWait()
        print(f"Audio saved to {filename}")
        return filename

    def get_audio_bytes(self, text: str, lang="en") -> tuple[bytes, int]:
        """
        Synthesizes text to speech and returns raw audio bytes and sample rate.
        This is a workaround as pyttsx3 doesn't directly expose audio buffers.
        It saves to a temporary file and reads it back.
        """
        temp_filename = "temp_tts_output.wav"
        self.engine.save_to_file(text, temp_filename)
        self.engine.runAndWait() # Ensure the file is fully written
        
        # Add a small delay to ensure file system sync
        time.sleep(0.5) # Increased delay

        # Retry mechanism for reading the file
        max_retries = 5
        for i in range(max_retries):
            if os.path.exists(temp_filename):
                try:
                    with wave.open(temp_filename, 'rb') as wf:
                        sample_rate = wf.getframerate()
                        audio_bytes = wf.readframes(wf.getnframes())
                    os.remove(temp_filename) # Clean up temporary file
                    return audio_bytes, sample_rate
                except wave.Error as e:
                    print(f"Warning: Attempt {i+1}/{max_retries} to read WAV failed: {e}. Retrying...")
                    time.sleep(0.2) # Small delay before retry
            else:
                print(f"Warning: Attempt {i+1}/{max_retries}, file '{temp_filename}' not found. Retrying...")
                time.sleep(0.2) # Small delay before retry
        
        raise FileNotFoundError(f"Failed to create or read temporary TTS file '{temp_filename}' after {max_retries} attempts.")

if __name__ == "__main__":
    tts_model = TTSOutput()
    tts_model.list_voices()

    # Example: Synthesize and play a sentence (now non-blocking)
    tts_model.synthesize_and_play("Hello, this is a test of the pyttsx3 library.", lang="en")
    time.sleep(3) # Allow time for playback

    # Example: Synthesize to a file
    output_file = tts_model.synthesize_to_file("This is a test sentence saved to a file.", "pyttsx3_output.wav", lang="en")
    
    # Example: Get audio bytes (demonstrates how to get raw audio for streaming)
    try:
        audio_bytes, sample_rate = tts_model.get_audio_bytes("This is audio as bytes.", lang="en")
        print(f"Received {len(audio_bytes)} bytes of audio data with sample rate {sample_rate}.")
        
        # Play the bytes_output.wav using AudioInput class
        from audio_input import AudioInput
        audio_manager = AudioInput()
        # Temporarily save bytes to a file for AudioInput to play
        temp_play_file = "temp_playback.wav"
        with wave.open(temp_play_file, 'wb') as wf:
            wf.setnchannels(1) # Assuming mono
            wf.setsampwidth(2) # Assuming 16-bit
            wf.setframerate(sample_rate) # Use the actual sample rate
            wf.writeframes(audio_bytes)
        audio_manager.play_audio(temp_play_file)
        os.remove(temp_play_file)

    except Exception as e:
        print(f"Error getting audio bytes: {e}")
