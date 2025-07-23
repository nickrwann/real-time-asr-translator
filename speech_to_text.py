# Model configuration
MODEL_NAME = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"

# Audio processing settings
SAMPLE_RATE = 16000

# Inference window parameters
WINDOW_SECONDS = 10
HOP_SECONDS = 1

import queue, threading, sys

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from colorama import init as colorama_init, Fore, Style

colorama_init(autoreset=True)

WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)
HOP_SAMPLES    = int(SAMPLE_RATE * HOP_SECONDS)

def main():
    # Whisper
    model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
    model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))  # GPU warm‑up

    audio_q = queue.Queue()

    def audio_cb(indata, frames, t, status):
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        audio_q.put(indata.copy())

    def record_loop():
        with sd.InputStream(samplerate=SAMPLE_RATE,
                            channels=1,
                            dtype="int16",
                            blocksize=HOP_SAMPLES,
                            callback=audio_cb):
            print("Listening … (Ctrl+C to stop)")
            threading.Event().wait()

    def process_loop():
        buf = np.zeros((0,), dtype=np.int16)
        while True:
            # 1) append the new 1 s chunk
            buf = np.concatenate((buf, audio_q.get().ravel()))

            # 2) once we have at least 10 s, run Whisper
            while buf.size >= WINDOW_SAMPLES:
                # 2a) grab the newest 10 s
                window = buf[-WINDOW_SAMPLES:]

                # 2b) slide forward by dropping the oldest 1 s
                buf = buf[HOP_SAMPLES:]

                # 3) transcribe that window
                audio_f32 = window.astype(np.float32) / 32768.0
                segs, _ = model.transcribe(
                    audio_f32,
                    vad_filter=True,
                    # vad_parameters={
                    #     # only keep segments ≥ 300 ms speech
                    #     "min_speech_duration_ms": 300,
                    #     # require ≥ 500 ms silence to break
                    #     "min_silence_duration_ms": 500,
                    #     # pad each segment by 400 ms of audio
                    #     "speech_pad_ms": 400,
                    #     # threshold for speech vs silence (0–1)
                    #     "threshold": 0.5
                    # }
                )
                text = "".join(s.text for s in segs).strip()
                if not text:
                    continue

                print(Fore.CYAN + Style.BRIGHT + f"> {text}")
                print(Style.DIM + "-" * 50)

    threading.Thread(target=record_loop,  daemon=True).start()
    threading.Thread(target=process_loop, daemon=True).start()

    try:
        while True:
            threading.Event().wait(1)
    except KeyboardInterrupt:
        print("\nStopping … adiós!")

if __name__ == "__main__":
    main()
