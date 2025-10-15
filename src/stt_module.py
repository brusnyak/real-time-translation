import torch
from faster_whisper import WhisperModel

class STTModule:
    def __init__(self, model_size="base", device="cpu", compute_type="int8", language="en"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language # Store the language
        print(f"Initializing Faster-Whisper model: {self.model_size} on device: {self.device} with compute type: {self.compute_type}...")
        start_time = time.time()
        self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        self._load_time = time.time() - start_time
        print(f"Faster-Whisper model initialized in {self._load_time:.3f}s.")

    def transcribe(self, audio_path): # Removed language parameter, use self.language
        """
        Transcribes audio from a given path.
        Returns segments and info from faster-whisper.
        """
        print(f"Transcribing audio with Faster-Whisper (language: {self.language})...")
        segments, info = self.model.transcribe(audio_path, beam_size=5, language=self.language)
        return segments, info

    def transcribe_chunk(self, audio_chunk, language="en"):
        """
        Transcribes an audio chunk. This method would be adapted for live streaming.
        For now, it's a placeholder for future streaming integration.
        """
        # In a real-time scenario, this would process a small audio buffer.
        # For offline testing, we might simulate this by transcribing a temporary file.
        # This needs more sophisticated handling for actual live streaming.
        print(f"Transcribing audio chunk with Faster-Whisper (language: {language})...")
        segments, info = self.model.transcribe(audio_chunk, beam_size=5, language=language)
        return segments, info

import time # Added import for time

if __name__ == "__main__":
    # Example usage (for testing purposes)
    # This assumes you have a 'tests/My test speech.wav' file
    # and a properly configured environment.
    
    # Example usage (for testing purposes)
    # The device and compute_type are now passed from the orchestrator.
    # For standalone testing, you might define them here.
    # For standalone testing, you might define them here, but for pipeline integration,
    # these values are passed from the orchestrator.
    # For now, we'll use the default values from the class definition for standalone testing.
    stt_module = STTModule()
    print(f"STT Module Load Time: {stt_module._load_time:.3f}s")
    
    # Placeholder for actual audio path
    # You would need to ensure 'tests/My test speech.wav' exists or provide a valid path
    audio_file = "tests/My test speech.wav" 
    
    try:
        segments, info = stt_module.transcribe(audio_file, language="en")
        transcribed_text = " ".join([segment.text for segment in segments])
        print(f"Transcribed Text: {transcribed_text}")
    except FileNotFoundError:
        print(f"Error: Audio file not found at {audio_file}. Please ensure the file exists for testing.")
    except Exception as e:
        print(f"An error occurred during transcription: {e}")
