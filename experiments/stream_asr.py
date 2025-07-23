#!/usr/bin/env python3
"""
stream_asr.py

Capture microphone audio and perform near-real-time speech-to-text
using faster-whisper with the tiniest Whisper model. Prints recognized
text to the console as it arrives.

Usage:
    pip install faster-whisper sounddevice numpy scipy
    python stream_asr.py

Press Ctrl+C to stop.
"""
import queue
import threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# Audio capture settings
SAMPLE_RATE   = 16000   # 16 kHz
CHUNK_SECONDS = 5       # 1-second audio chunks
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS


def main():
        # Load multilingual Tiny model so it can handle English and Spanish
    model = WhisperModel(
        "large-v3",             # multilingual tiny (~75 MB int8)
        device="cuda",        # set to "cuda" if your GPU is stable
        compute_type="float16"   # change to "int8_float16" for GPU or keep "int8" on CPU
    )

    audio_queue = queue.Queue()

    def audio_callback(indata, frames, time, status):
        if status:
            print(f"Audio status: {status}")
        audio_queue.put(indata.copy())

    def record_thread():
        # Continuously read from default microphone
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=CHUNK_SAMPLES,
            callback=audio_callback
        ):
            print("Listening (Ctrl+C to stop)…")
            threading.Event().wait()

    def process_thread():
        buffer = np.zeros((0,), dtype=np.int16)
        while True:
            chunk = audio_queue.get()
            buffer = np.concatenate((buffer, chunk.ravel()))
            if buffer.shape[0] < CHUNK_SAMPLES:
                continue

            # Extract a fixed-size segment
            segment = buffer[:CHUNK_SAMPLES]
            buffer  = buffer[CHUNK_SAMPLES:]

            # Convert to float32 1D array in [-1.0, +1.0]
            audio = segment.astype(np.float32) / 32768.0

            try:
                # Perform transcription; audio is 1D array
                segments, _ = model.transcribe(
                    audio,
                    beam_size=1,
                    language=None,
                    word_timestamps=False
                )
                text = "".join(seg.text for seg in segments).strip()
                if text:
                    print(f"[RECOGNIZED] {text}")
            except Exception as e:
                print(f"Transcription error: {e}")

    # Launch threads
    recorder = threading.Thread(target=record_thread, daemon=True)
    processor = threading.Thread(target=process_thread, daemon=True)
    recorder.start()
    processor.start()

    try:
        while True:
            threading.Event().wait(1)
    except KeyboardInterrupt:
        print("\nStopping… goodbye!")


if __name__ == "__main__":
    main()
