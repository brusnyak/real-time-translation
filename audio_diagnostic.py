#!/usr/bin/env python3
"""
Audio Device Diagnostic Tool
Tests which devices can hear you and receive audio properly.
"""

import sounddevice as sd
import numpy as np
import soundfile as sf
import time


def list_all_devices():
    """List all available audio devices."""
    print("\n" + "="*70)
    print("AVAILABLE AUDIO DEVICES")
    print("="*70)
    
    devices = sd.query_devices()
    print(f"\nTotal devices: {len(devices)}\n")
    
    for i, device in enumerate(devices):
        print(f"[{i}] {device['name']}")
        print(f"    Max Input Channels:  {device['max_input_channels']}")
        print(f"    Max Output Channels: {device['max_output_channels']}")
        if device['default_samplerate']:
            print(f"    Default Sample Rate: {int(device['default_samplerate'])} Hz")
        print()
    
    return devices


def test_input_device(device_id, duration=5.0, sample_rate=16000):
    """Test if a device can record audio."""
    device_name = sd.query_devices(device_id)['name']
    
    print(f"\n{'='*70}")
    print(f"TESTING INPUT: Device {device_id} - {device_name}")
    print(f"{'='*70}")
    print(f"Recording for {duration} seconds...")
    print("Speak into your microphone now!")
    
    try:
        # Record audio
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32,
            device=device_id
        )
        sd.wait()
        
        # Analyze audio
        audio = audio.squeeze()
        rms = np.sqrt(np.mean(audio**2))
        peak = np.max(np.abs(audio))
        
        print(f"\nRecording complete!")
        print(f"  RMS Level:   {rms:.4f}")
        print(f"  Peak Level:  {peak:.4f}")
        
        if rms < 0.01:
            print(f"  ⚠️  VERY QUIET - Device may not be capturing audio properly")
            return False
        elif rms < 0.05:
            print(f"  ⚠️  QUIET - Device is capturing but level is low")
            return True
        else:
            print(f"  ✓ GOOD - Device is capturing audio at good levels")
            return True
            
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return False


def test_output_device(device_id, duration=2.0, sample_rate=22050):
    """Test if a device can play audio."""
    device_name = sd.query_devices(device_id)['name']
    
    print(f"\n{'='*70}")
    print(f"TESTING OUTPUT: Device {device_id} - {device_name}")
    print(f"{'='*70}")
    
    try:
        # Generate test tone
        t = np.linspace(0, duration, int(sample_rate * duration))
        frequency = 440  # A4 note
        audio = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        print(f"Playing 440Hz tone for {duration} seconds...")
        print("You should hear a beep. Did you hear it?")
        
        sd.play(audio, samplerate=sample_rate, device=device_id)
        sd.wait()
        
        print("✓ Playback complete (if no audio was heard, device may not work)")
        return True
        
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def find_working_input_device():
    """Find which input device can actually hear you."""
    print(f"\n{'='*70}")
    print("SCANNING FOR WORKING INPUT DEVICES")
    print(f"{'='*70}")
    
    devices = sd.query_devices()
    working_devices = []
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] < 1:
            continue
        
        print(f"\nTesting {device['name']}...", end=" ", flush=True)
        
        try:
            # Quick 2-second test
            audio = sd.rec(
                int(2 * 16000),
                samplerate=16000,
                channels=1,
                dtype=np.float32,
                device=i,
                blocksize=16000
            )
            sd.wait()
            
            rms = np.sqrt(np.mean(audio**2))
            
            if rms > 0.02:  # Threshold for "hearing something"
                print(f"✓ WORKING (RMS: {rms:.4f})")
                working_devices.append((i, device['name'], rms))
            else:
                print(f"✗ No input detected (RMS: {rms:.4f})")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    return working_devices


def main():
    print("\n" + "="*70)
    print("AUDIO DEVICE DIAGNOSTIC TOOL")
    print("="*70)
    
    # List all devices
    devices = list_all_devices()
    
    # Find working input devices
    print("\n" + "="*70)
    print("STEP 1: SCANNING FOR WORKING INPUT DEVICES")
    print("="*70)
    working_inputs = find_working_input_device()
    
    if working_inputs:
        print(f"\nFound {len(working_inputs)} working input device(s):")
        for device_id, name, rms in working_inputs:
            print(f"  [{device_id}] {name} (RMS: {rms:.4f})")
        
        # Full test on the best device
        best_device = working_inputs[0][0]
        print(f"\n\nUsing best device: {working_inputs[0][1]}")
        test_input_device(best_device)
    else:
        print("\n✗ No working input devices found!")
        print("This means no device is capturing audio from your microphone.")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY & TROUBLESHOOTING")
    print(f"{'='*70}")
    
    if working_inputs:
        best_id, best_name, _ = working_inputs[0]
        print(f"\n✓ Use device ID: {best_id}")
        print(f"  Device name: {best_name}")
        print(f"\nTo use in your program:")
        print(f"  sd.InputStream(..., device={best_id})")
    else:
        print("\n✗ PROBLEM: No input devices detected sound")
        print("\nTroubleshooting steps:")
        print("  1. Check System Preferences → Sound → Input")
        print("  2. If using aggregated device, verify it includes your microphone")
        print("  3. Test with built-in microphone first")
        print("  4. Close other apps using audio (Zoom, Discord, etc.)")
        print("  5. Restart this script and try again")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()