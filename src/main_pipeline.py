import numpy as np
import soundfile as sf
import os
import time
import wave
from pydub import AudioSegment
import io

from audio_input import AudioInput
from stt_whisper import STTWhisper
from translate import Translator
from tts_output import TTSOutput

class MainPipeline:
    def __init__(self, stt_model_size="base", stt_device="cpu",
                 src_lang="en", tgt_lang="sk",
                 tts_rate=150, tts_volume=1.0):
        """
        Initializes the main pipeline components.
        """
        self.audio_input = AudioInput()
        self.stt_whisper = STTWhisper(model_size=stt_model_size, device=stt_device)
        self.translator_en_sk = Translator(src_lang="en", tgt_lang="sk")
        self.translator_sk_en = Translator(src_lang="sk", tgt_lang="en")
        self.tts_output = TTSOutput(rate=tts_rate, volume=tts_volume)

    async def run_offline_demo(self, audio_file_path="test_recording.wav", duration=5):
        """
        Runs an offline demo: records audio, processes it through STT -> MT -> TTS,
        and plays back original and translated audio.
        """
        print(f"--- Starting Offline Demo (Using '{audio_file_path}') ---")
        
        # 1. Load provided audio file using pydub for .m4a support
        print(f"Loading audio from {audio_file_path}...")
        audio = AudioSegment.from_file(audio_file_path)
        
        # Convert to mono and 16kHz for Whisper
        audio = audio.set_channels(1).set_frame_rate(16000)
        
        # Export to a WAV in memory and read with soundfile
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_buffer.seek(0) # Rewind to the beginning of the buffer
        
        audio_data, sr = sf.read(wav_buffer)
        print(f"Audio loaded from {audio_file_path} and processed to {sr} Hz.")

        # Save original audio to a temporary WAV file for comparison
        original_audio_temp_file = "temp_original_audio.wav"
        sf.write(original_audio_temp_file, audio_data, sr)
        print(f"Original audio saved to {original_audio_temp_file}")
        
        # Ensure audio_data is float32 for Whisper
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1) # Convert to mono if not already

        # 3. Detect language
        print("Detecting language...")
        detected_lang = self.stt_whisper.detect_language(audio_data)
        print(f"Detected language: {detected_lang}")

        # 4. Transcribe audio
        print("Transcribing audio...")
        transcribed_text = self.stt_whisper.transcribe_audio(audio_data, language=detected_lang, task="transcribe")
        print(f"Original ({detected_lang}): {transcribed_text}")

        # 5. Translate text
        print("Translating text...")
        if detected_lang == "en":
            translated_text = await self.translator_en_sk.translate_text(transcribed_text)
            target_lang_code = "sk"
        elif detected_lang == "sk":
            translated_text = await self.translator_sk_en.translate_text(transcribed_text)
            target_lang_code = "en"
        else:
            print(f"Unsupported language for translation: {detected_lang}. Skipping translation.")
            translated_text = transcribed_text
            target_lang_code = detected_lang # Fallback
        
        print(f"Translated ({target_lang_code}): {translated_text}")

        # 6. Synthesize and play translated speech
        print("Synthesizing and playing translated speech...")
        translated_audio_temp_file = "temp_translated_audio.wav"
        self.tts_output.synthesize_to_file(translated_text, filename=translated_audio_temp_file, lang=target_lang_code)
        
        # Play the translated audio from the saved file
        from audio_input import AudioInput
        audio_manager = AudioInput()
        audio_manager.play_audio(translated_audio_temp_file)
        
        print("--- Offline Demo Finished ---")

    async def run_realtime_pipeline(self, src_lang="en", tgt_lang="sk", chunk_size=1024, sample_rate=16000):
        """
        Runs the real-time speech translation pipeline.
        Captures live audio, transcribes, translates, and synthesizes speech.
        """
        print(f"--- Starting Real-time Pipeline ({src_lang} -> {tgt_lang}) ---")
        print("Press Ctrl+C to stop.")

        # Set up the correct translator based on source and target languages
        if src_lang == "en" and tgt_lang == "sk":
            translator = self.translator_en_sk
        elif src_lang == "sk" and tgt_lang == "en":
            translator = self.translator_sk_en
        else:
            print(f"Unsupported translation direction: {src_lang} -> {tgt_lang}")
            return

        try:
            # Use AudioInput's stream for real-time processing
            with self.audio_input.stream_audio(chunk_size=chunk_size, sample_rate=sample_rate) as stream:
                print("Listening for speech...")
                audio_buffer = []
                # Accumulate audio for Whisper's 30-second window (or smaller chunks for real-time)
                # For a truly real-time Whisper, one would use its lower-level decode methods
                # and manage a sliding window of audio. For this prototype, we'll accumulate
                # a small buffer and process it.
                buffer_duration = 3 # seconds
                buffer_frames = int(sample_rate * buffer_duration)

                while True:
                    data = stream.read(chunk_size)
                    audio_buffer.extend(np.frombuffer(data, dtype=np.int16))

                    if len(audio_buffer) >= buffer_frames:
                        audio_array = np.array(audio_buffer[:buffer_frames], dtype=np.float32) / 32768.0 # Normalize to float32
                        
                        # Process the chunk
                        detected_lang = self.stt_whisper.detect_language(audio_array)
                        if detected_lang == src_lang:
                            transcribed_text = self.stt_whisper.transcribe_audio(audio_array, language=src_lang, task="transcribe")
                            if transcribed_text.strip(): # Only process if there's actual speech
                                print(f"[{src_lang}] {transcribed_text}")
                                translated_text = await translator.translate_text(transcribed_text)
                                print(f"[{tgt_lang}] {translated_text}")
                                self.tts_output.synthesize_and_play(translated_text, lang=tgt_lang)
                        else:
                            # If language changes, or is not the source language, just print detection
                            # print(f"Detected language: {detected_lang}, not {src_lang}. Skipping translation.")
                            pass # Suppress frequent language detection messages

                        # Keep only the unprocessed part of the buffer
                        audio_buffer = audio_buffer[buffer_frames:]

        except KeyboardInterrupt:
            print("\n--- Real-time Pipeline Stopped ---")
        except Exception as e:
            print(f"An error occurred in real-time pipeline: {e}")

if __name__ == "__main__":
    # Path to the test speech file
    test_speech_file = "tests/My test speech.m4a"
    if not os.path.exists(test_speech_file):
        print(f"Error: Test speech file not found at {test_speech_file}")
        print("Please ensure 'tests/My test speech.m4a' exists.")
        exit()

    pipeline = MainPipeline(stt_model_size="medium", src_lang="en", tgt_lang="sk", tts_rate=170, tts_volume=1.0)

    print("\n--- Available TTS Voices ---")
    pipeline.tts_output.list_voices()
    print("----------------------------\n")
    
    # Run offline demo using the provided test speech file
    asyncio.run(pipeline.run_offline_demo(audio_file_path=test_speech_file))

    # Run real-time pipeline (uncomment to test)
    # asyncio.run(pipeline.run_realtime_pipeline(src_lang="en", tgt_lang="sk"))
