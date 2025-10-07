import sounddevice as sd
import numpy as np
import wave
import collections
import threading
import time

class AudioInput:
    def __init__(self, samplerate=16000, channels=1, dtype='int16', chunk_size=1024):
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.chunk_size = chunk_size
        self.stream = None
        self.audio_buffer = collections.deque()
        self.buffer_lock = threading.Lock()
        self.running = False

    def record_audio(self, duration, filename="output.wav"):
        """Records audio for a given duration and saves it to a file."""
        print(f"Recording for {duration} seconds...")
        audio_data = sd.rec(int(duration * self.samplerate),
                            samplerate=self.samplerate,
                            channels=self.channels,
                            dtype=self.dtype)
        sd.wait()  # Wait until recording is finished
        print("Recording finished.")

        # Save the recorded audio to a WAV file
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(np.dtype(self.dtype).itemsize)
            wf.setframerate(self.samplerate)
            wf.writeframes(audio_data.tobytes())
        print(f"Audio saved to {filename}")
        return filename

    def play_audio(self, filename="output.wav"):
        """Plays an audio file."""
        print(f"Playing audio from {filename}...")
        try:
            with wave.open(filename, 'rb') as wf:
                self.samplerate = wf.getframerate()
                self.channels = wf.getnchannels()
                # Determine dtype from sample width
                if wf.getsampwidth() == 1:
                    self.dtype = 'int8'
                elif wf.getsampwidth() == 2:
                    self.dtype = 'int16'
                elif wf.getsampwidth() == 4:
                    self.dtype = 'int32'
                else:
                    print(f"Warning: Unsupported sample width {wf.getsampwidth()} bytes. Defaulting to int16.")
                    self.dtype = 'int16'

                audio_data = wf.readframes(wf.getnframes())
                audio_array = np.frombuffer(audio_data, dtype=self.dtype)

                sd.play(audio_array, self.samplerate)
                sd.wait()  # Wait until playback is finished
            print("Playback finished.")
        except wave.Error as e:
            print(f"Error playing audio file {filename}: {e}. Ensure it's a valid WAV file.")
        except Exception as e:
            print(f"An unexpected error occurred during audio playback: {e}")


    def _callback(self, indata, frames, time_info, status):
        """Callback function for the audio stream."""
        if status:
            print(status)
        with self.buffer_lock:
            self.audio_buffer.append(indata.copy())

    def start_stream(self):
        """Starts an audio input stream for real-time processing."""
        if self.stream is None or not self.stream.active: # Use .active instead of .is_active
            self.stream = sd.InputStream(samplerate=self.samplerate,
                                         channels=self.channels,
                                         dtype=self.dtype,
                                         callback=self._callback,
                                         blocksize=self.chunk_size)
            self.stream.start()
            self.running = True # Set running flag
            print("Audio stream started.")

    def stop_stream(self):
        """Stops the audio input stream."""
        if self.stream and self.stream.active: # Use .active instead of .is_active
            self.stream.stop()
            self.stream.close()
            self.running = False # Clear running flag
            print("Audio stream stopped.")
        with self.buffer_lock:
            self.audio_buffer.clear() # Clear any remaining data

    def get_audio_chunk(self):
        """Retrieves a chunk of audio data from the buffer."""
        with self.buffer_lock:
            if self.audio_buffer:
                return self.audio_buffer.popleft()
        return None

if __name__ == "__main__":
    audio_manager = AudioInput(chunk_size=1024)

    # Example: Record 5 seconds of audio
    recorded_file = audio_manager.record_audio(5, "test_recording.wav")

    # Example: Play the recorded audio
    audio_manager.play_audio(recorded_file)

    # Example: Real-time stream (this part would typically be integrated into the main pipeline)
    print("\n--- Starting Real-time Stream Demo ---")
    audio_manager.start_stream()

    try:
        while audio_manager.running:
            chunk = audio_manager.get_audio_chunk()
            if chunk is not None:
                print(f"Received {len(chunk)} frames of audio data in real-time.")
            time.sleep(0.1) # Small delay to prevent busy-waiting
    except KeyboardInterrupt:
        print("\nStopping real-time stream demo.")
    finally:
        audio_manager.stop_stream()
        print("Real-time stream demo finished.")
