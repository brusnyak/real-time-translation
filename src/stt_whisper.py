import whisper
import numpy as np
import torch
import soundfile 

class STTWhisper:
    def __init__(self, model_size="base", device="cpu"):
        """
        Initializes the Whisper STT model.
        :param model_size: Size of the Whisper model (e.g., "tiny", "base", "small", "medium", "large").
                           For English-Slovak translation, multilingual models are recommended.
        :param device: Device to run the model on ("cpu" or "cuda").
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Loading Whisper model '{model_size}' on device: {self.device}...")
        self.model = whisper.load_model(model_size, device=self.device)
        print("Whisper model loaded.")

    def transcribe_audio(self, audio_data: np.ndarray, language="en", task="transcribe"):
        """
        Transcribes or translates audio data using the Whisper model.
        :param audio_data: NumPy array of audio samples.
        :param language: Source language of the audio (e.g., "en", "sk").
        :param task: "transcribe" for speech-to-text in the same language,
                     "translate" for speech-to-English translation.
        :return: Transcribed or translated text.
        """
        if not isinstance(audio_data, np.ndarray):
            raise TypeError("audio_data must be a NumPy array.")
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Whisper expects 16kHz mono audio
        # If audio_data is not 16kHz, it should be resampled before passing here.
        # For simplicity, assuming input audio is already 16kHz.

        options = whisper.DecodingOptions(
            language=language,
            task=task,
            fp16=False if self.device == "cpu" else True # Use fp16 on GPU for faster inference
        )

        # Pad/trim audio to 30 seconds as expected by Whisper's transcribe method
        # For real-time, we'll process chunks, but for offline, we can use the full audio.
        audio_padded = whisper.pad_or_trim(audio_data)
        mel = whisper.log_mel_spectrogram(audio_padded).to(self.model.device)

        result = whisper.decode(self.model, mel, options)
        return result.text

    def detect_language(self, audio_data: np.ndarray):
        """
        Detects the dominant language in the audio data.
        :param audio_data: NumPy array of audio samples.
        :return: Detected language code (e.g., "en", "sk").
        """
        if not isinstance(audio_data, np.ndarray):
            raise TypeError("audio_data must be a NumPy array.")
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        audio_padded = whisper.pad_or_trim(audio_data)
        mel = whisper.log_mel_spectrogram(audio_padded).to(self.model.device)

        _, probs = self.model.detect_language(mel)
        detected_language = max(probs, key=probs.get)
        return detected_language

if __name__ == "__main__":
    # Example usage with a dummy audio file (replace with actual audio)
    # For a real test, you would record audio using AudioInput and pass it here.
    # This example assumes you have a short audio file named 'test_recording.wav'
    # created by the audio_input.py example.

    try:
        from audio_input import AudioInput
        audio_manager = AudioInput()
        test_audio_file = "test_recording.wav"
        # Ensure a test_recording.wav exists for this example
        # If not, uncomment the line below to record one
        # audio_manager.record_audio(5, test_audio_file)

        # Load the audio file
        import soundfile as sf
        audio_data, sr = sf.read(test_audio_file)
        if sr != 16000:
            # Resample if necessary (Whisper expects 16kHz)
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1) # Convert to mono

        stt_model = STTWhisper(model_size="base") # Use a multilingual model for translation tasks

        # Example 1: Transcribe in English
        print("\n--- Transcribing English ---")
        english_text = stt_model.transcribe_audio(audio_data, language="en", task="transcribe")
        print(f"Transcribed (English): {english_text}")

        # Example 2: Detect language
        print("\n--- Detecting Language ---")
        detected_lang = stt_model.detect_language(audio_data)
        print(f"Detected Language: {detected_lang}")

        # Example 3: Translate to English (if the audio is non-English)
        # For this to work well, the audio_data should ideally be in a non-English language.
        # For demonstration, we'll just use the same audio.
        print("\n--- Translating to English ---")
        translated_text = stt_model.transcribe_audio(audio_data, language=detected_lang, task="translate")
        print(f"Translated (to English): {translated_text}")

    except FileNotFoundError:
        print(f"Error: '{test_audio_file}' not found. Please run audio_input.py to create it first.")
    except Exception as e:
        print(f"An error occurred: {e}")
