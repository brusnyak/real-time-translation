import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import os

def list_input_devices():
    """Lists all available input devices."""
    print("--- Available Input Devices ---")
    devices = sd.query_devices()
    input_devices = []
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append((i, device['name'], device['default_samplerate']))
            print(f"ID: {i}, Name: {device['name']}, Sample Rate: {device['default_samplerate']} Hz")
    print("-----------------------------")
    return input_devices

def record_from_device(device_id, sample_rate, duration=5, filename="test_audio.wav"):
    """Records audio from a specified device."""
    print(f"Recording from device ID {device_id} ('{sd.query_devices(device_id)['name']}') for {duration} seconds...")
    try:
        # Ensure the sample rate is compatible with the device
        # If the device's default sample rate is different, we might need to resample later
        # For now, we'll try to record at the device's default rate.
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, device=device_id, dtype='float32')
        sd.wait()
        sf.write(filename, recording, sample_rate)
        print(f"Saved recording to {filename}")
        return True
    except Exception as e:
        print(f"Error recording from device ID {device_id}: {e}")
        return False

if __name__ == "__main__":
    output_dir = "audio_diagnostics"
    os.makedirs(output_dir, exist_ok=True)

    input_devices = list_input_devices()

    if not input_devices:
        print("No input devices found. Please check your audio setup.")
    else:
        print("\n--- Testing each input device ---")
        for device_id, device_name, default_samplerate in input_devices:
            print(f"\nAttempting to record from: {device_name} (ID: {device_id}, SR: {default_samplerate} Hz)")
            
            # Use a common sample rate for recording if possible, or device default
            # For VAD, 16kHz is often preferred, but let's record at device's default first
            # to see what it actually captures.
            record_sample_rate = int(default_samplerate) # Use device's default sample rate

            test_filename = os.path.join(output_dir, f"input_device_{device_id}_{device_name.replace(' ', '_').replace('/', '_')}.wav")
            
            print(f"Please speak into the microphone for 5 seconds after this message...")
            time.sleep(1) # Give a moment before recording starts
            
            success = record_from_device(device_id, record_sample_rate, duration=5, filename=test_filename)
            if success:
                print(f"Successfully recorded from {device_name}. Check '{test_filename}'")
            else:
                print(f"Failed to record from {device_name}.")
        
        print("\n--- Diagnostic Complete ---")
        print(f"Please listen to the WAV files in the '{output_dir}' directory to identify which microphone works best.")
        print("You can then use the ID of the working device in your live_pipeline.py script.")
