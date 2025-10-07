import numpy as np
import soundfile as sf
import os
import time
import wave
from pydub import AudioSegment
import io
import asyncio # Import asyncio
import pysbd # For sentence segmentation

from audio_input import AudioInput
from stt_whisper import STTWhisper
from translate import Translator
from tts_output import TTSOutput

class MainPipeline:
    def __init__(self, stt_model_size="base", stt_device="cpu",
                 src_lang="en", tgt_lang="sk",
                 speaker_wav_path="tests/My test speech.m4a"):
        """
        Initializes the main pipeline components.
        """
        self.audio_input = AudioInput()
        self.stt_whisper = STTWhisper(model_size=stt_model_size, device=stt_device)
        self.translator_en_sk = Translator(src_lang="en", tgt_lang="sk")
        self.translator_sk_en = Translator(src_lang="sk", tgt_lang="en")
        self.translator_en_cs = Translator(src_lang="en", tgt_lang="cs") # Add English to Czech translator
        self.tts_output = TTSOutput(speaker_wav_path=speaker_wav_path)
        self.test_run_id = self._get_next_test_id() # Sequential ID for each test run

    def _get_next_test_id(self):
        """Determines the next sequential test ID based on existing research/test_N directories."""
        existing_test_dirs = [d for d in os.listdir("research") if d.startswith("test_") and os.path.isdir(os.path.join("research", d))]
        if not existing_test_dirs:
            return 1
        
        # Extract numbers from directory names (e.g., "test_5" -> 5)
        test_numbers = []
        for d in existing_test_dirs:
            try:
                num_str = d.split('_')[1]
                test_numbers.append(int(num_str))
            except (IndexError, ValueError):
                continue # Ignore malformed directory names
        
        if test_numbers:
            return max(test_numbers) + 1
        else:
            return 1 # Fallback if no valid test numbers found

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

        # 5. Translate text and prepare for TTS
        print("Translating text and preparing for TTS...")
        
        target_lang_code = None
        tts_target_lang_code = None
        translated_text = ""
        translated_text_for_tts = ""

        if detected_lang == "en":
            # Directly translate English to Czech for both display and TTS
            # to leverage XTTS v2 voice cloning, as Czech is linguistically similar to Slovak.
            translated_text = await self.translator_en_cs.translate_text(transcribed_text)
            target_lang_code = "cs" # Display as Czech
            translated_text_for_tts = translated_text
            tts_target_lang_code = "cs" # Synthesize in Czech
            print(f"Note: Directly translating English to Czech for both display and TTS to leverage XTTS v2 voice cloning.")
        elif detected_lang == "sk":
            # If source is Slovak, translate to English
            translated_text = await self.translator_sk_en.translate_text(transcribed_text)
            target_lang_code = "en"
            translated_text_for_tts = translated_text
            tts_target_lang_code = "en"
        else:
            print(f"Unsupported language for translation: {detected_lang}. Skipping translation.")
            translated_text = transcribed_text
            translated_text_for_tts = transcribed_text
            target_lang_code = detected_lang # Fallback
            tts_target_lang_code = detected_lang # Fallback
        
        print(f"Translated ({target_lang_code}): {translated_text}")
        # No need for 'Translated for TTS' if target_lang_code == tts_target_lang_code

        # 6. Synthesize and play translated speech
        print("Synthesizing and playing translated speech...")
        translated_audio_temp_file = "temp_translated_audio.wav"
        
        # Perform sentence segmentation on the *translated text for TTS* before synthesis
        # Use Slovak for segmentation if TTS target is Czech, as pysbd supports Slovak but not Czech.
        segmentation_lang = "sk" if tts_target_lang_code == "cs" else tts_target_lang_code
        seg = pysbd.Segmenter(language=segmentation_lang, clean=False)
        sentences_for_tts = seg.segment(translated_text_for_tts)

        # Synthesize each sentence separately to avoid noise at the end of each sentence
        # and potentially improve overall quality.
        # This will also help XTTS v2 process smaller, more coherent chunks.
        for i, sentence_for_tts in enumerate(sentences_for_tts):
            if not sentence_for_tts.strip():
                continue
            print(f"Synthesizing sentence {i+1}/{len(sentences_for_tts)}: '{sentence_for_tts}'")
            self.tts_output.synthesize_to_file(sentence_for_tts, filename=f"temp_translated_audio_part_{i}.wav", lang=tts_target_lang_code)
        
        # Combine the synthesized audio parts into a single file for playback and comparison
        combined_audio = AudioSegment.empty()
        for i in range(len(sentences_for_tts)):
            part_file = f"temp_translated_audio_part_{i}.wav"
            if os.path.exists(part_file):
                combined_audio += AudioSegment.from_wav(part_file)
                os.remove(part_file)
        
        if combined_audio:
            combined_audio.export(translated_audio_temp_file, format="wav")
            print(f"Combined synthesized audio saved to {translated_audio_temp_file}")
        else:
            print("No audio synthesized for playback.")

        
        # Play the translated audio from the saved file
        from audio_input import AudioInput
        audio_manager = AudioInput()
        audio_manager.play_audio(translated_audio_temp_file)
        
        print("--- Offline Demo Finished ---")

        # Perform audio comparison
        from audio_comparison import AudioComparator
        comparator = AudioComparator()
        comparator.compare_audio_signals(original_audio_temp_file, translated_audio_temp_file, output_dir=f"research/test_{self.test_run_id}")
        
        # Temporary: Do not clean up temporary files immediately for debugging and manual inspection
        # os.remove(original_audio_temp_file)
        # os.remove(translated_audio_temp_file)
        print("Temporary audio files (temp_original_audio.wav, temp_translated_audio.wav) retained for inspection.")

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
                                
                                # Translate entire chunk first
                                translated_text = ""
                                tts_target_lang_code = tgt_lang

                                if src_lang == "en" and tgt_lang == "sk":
                                    # Directly translate English to Czech for both display and TTS
                                    translated_text = await self.translator_en_cs.translate_text(transcribed_text)
                                    tts_target_lang_code = "cs"
                                    translated_text_for_tts = translated_text
                                    print(f"Note: Directly translating English to Czech for both display and TTS to leverage XTTS v2 voice cloning.")
                                else: # sk to en
                                    translated_text = await self.translator_sk_en.translate_text(transcribed_text)
                                    translated_text_for_tts = translated_text
                                    tts_target_lang_code = "en" # Assuming English is always supported by XTTS v2

                                print(f"[{tts_target_lang_code}] {translated_text}")
                                # No need for 'Translated for TTS' if target_lang_code == tts_target_lang_code

                                # Sentence segmentation for TTS output
                                # Use Slovak for segmentation if TTS target is Czech, as pysbd supports Slovak but not Czech.
                                segmentation_lang = "sk" if tts_target_lang_code == "cs" else tts_target_lang_code
                                seg = pysbd.Segmenter(language=segmentation_lang, clean=False)
                                sentences_for_tts = seg.segment(translated_text_for_tts)

                                for sentence_for_tts in sentences_for_tts:
                                    if not sentence_for_tts.strip():
                                        continue
                                    self.tts_output.synthesize_and_play(sentence_for_tts, lang=tts_target_lang_code)
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

    pipeline = MainPipeline(stt_model_size="medium", src_lang="en", tgt_lang="sk", speaker_wav_path=test_speech_file)

    print("\n--- Initializing TTS Model ---")
    # No need to list voices for XTTS v2 as it uses voice cloning
    print("----------------------------\n")
    
    # Run offline demo using the provided test speech file
    asyncio.run(pipeline.run_offline_demo(audio_file_path=test_speech_file))

    # Run real-time pipeline (uncomment to test)
    # asyncio.run(pipeline.run_realtime_pipeline(src_lang="en", tgt_lang="sk"))
