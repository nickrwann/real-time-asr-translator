#!/usr/bin/env python3
"""
speech_to_text_live.py

Realtime mic → Whisper transcription with:
  • 10 s sliding window, 1 s hop
  • VAD filtering
  • initial_prompt to bias the decoder away from repeats
"""

# ── CONFIG ────────────────────────────────────────────────────────────────── #
MODEL_NAME     = "large-v3"    # "small" | "medium" | "large-v3"
DEVICE         = "cuda"        # "cuda" | "cpu"
COMPUTE_TYPE   = "float16"     # "float16" | "int8_float16" | "int8"

SAMPLE_RATE    = 16000         # Hz

WINDOW_SECONDS = 10            # how much audio (seconds) per inference
HOP_SECONDS    = 1             # slide amount (seconds)
# ───────────────────────────────────────────────────────────────────────────── #

import queue, threading, sys
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from colorama import init as colorama_init, Fore, Style

colorama_init(autoreset=True)

WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)
HOP_SAMPLES    = int(SAMPLE_RATE * HOP_SECONDS)

def main():
    # Load Whisper
    model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
    # warm-up
    model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))

    audio_q = queue.Queue()

    def audio_cb(indata, frames, time, status):
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        audio_q.put(indata.copy())

    def record_loop():
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=HOP_SAMPLES,
            callback=audio_cb
        ):
            print("Listening … (Ctrl+C to stop)")
            threading.Event().wait()

    def process_loop():
        transcript_buf = ""                   # holds all printed text so far
        buf = np.zeros((0,), dtype=np.int16)

        while True:
            # 1) append new hop chunk
            buf = np.concatenate((buf, audio_q.get().ravel()))

            # 2) if we have ≥ WINDOW, run inference
            while buf.size >= WINDOW_SAMPLES:
                # take the last WINDOW_SECONDS of audio
                window = buf[-WINDOW_SAMPLES:]
                # drop the oldest HOP_SECONDS so we slide forward
                buf = buf[HOP_SAMPLES:]

                # prepare for Whisper
                audio_f32 = window.astype(np.float32) / 32768.0

                # 3) transcribe, seeding the decoder with transcript_buf
                segs, _ = model.transcribe(
                    audio_f32,
                    vad_filter=True,
                    initial_prompt=transcript_buf,
                    condition_on_previous_text=False,
                )

                # assemble the raw text
                text = "".join(s.text for s in segs).strip()
                if not text:
                    continue

                # 4) append only the new tail to transcript_buf
                transcript_buf += (" " if transcript_buf else "") + text

                # 5) print
                print(Fore.CYAN + Style.BRIGHT + f"> {text}")
                print(Fore.RED + Style.BRIGHT + f"> {transcript_buf}")
                print(Style.DIM + "-" * 50)

    # start threads
    threading.Thread(target=record_loop,  daemon=True).start()
    threading.Thread(target=process_loop, daemon=True).start()

    # keep main alive
    try:
        while True:
            threading.Event().wait(1)
    except KeyboardInterrupt:
        print("\nStopping … adiós!")

if __name__ == "__main__":
    main()
