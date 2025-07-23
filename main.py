# ── CONFIG ────────────────────────────────────────────────────────────────── #
MODEL_NAME        = "large-v3"      # Whisper checkpoint to load.
                                   #   "small"  – fastest, lowest VRAM, lower accuracy
                                   #   "medium" – good balance for live captions on a mid‑tier GPU
                                   #   "large-v3" – highest accuracy, 2‑3× slower, needs 10‑12 GB VRAM
                                   # Pick smaller if latency > 1 s or GPU memory is tight.

DEVICE            = "cuda"        # "cuda" to run on an NVIDIA GPU (fastest)
                                   # "cpu"  if you have no CUDA device. Expect ~5× slower unless
                                   #         you also down‑size MODEL_NAME to "small" or "tiny".

COMPUTE_TYPE      = "float16"     # Math precision / quantization mode:
                                   #   "float16"        – fastest on most GPUs, full accuracy
                                   #   "int8_float16"   – 20‑30 % smaller/faster, tiny accuracy hit
                                   #   "int8"           – smallest, safest for CPU‑only but slowest on GPU

SAMPLE_RATE       = 16000         # Microphone sampling rate. Whisper is trained at 16 kHz;
                                   # change only if your mic refuses 16 kHz (rare).

WINDOW_SECONDS    = 8             # Total audio context Whisper sees each pass.
                                   # Larger window → better accuracy on long sentences
                                   # but more latency and higher GPU load.

HOP_SECONDS       = 2             # Slide (hop) size: new audio fed each iteration.
                                   # Lower hop → quicker on‑screen updates.
                                   # Must be < WINDOW_SECONDS. Typical: 1–2 s.

BEAM_SIZE         = 3             # Beam‑search width (1 = greedy decode).
                                   # Increasing to 2–3 can rescue tricky words but ~2× slower.

TEMPERATURE_LIST  = [0.0]         # List of decoding temperatures Whisper cycles through.
                                   # Leave as [0.0] for lowest latency.
                                   # Add 0.2,0.4,… if you notice “no speech” errors or hallucinations.

COMP_RATIO_THRES  = 2.4           # If output text is “too compressible” (likely junk),
                                   # Whisper re‑decodes at higher temperature.
                                   # Lower value = stricter (safer, slower); higher = trust first pass.

NO_SPEECH_THRES   = 0.6           # Probability threshold for treating a segment as silence.
                                   # Raise to 0.7–0.8 if blanks appear too often.
                                   # Lower to 0.4 if short speech fragments are missed.
# ───────────────────────────────────────────────────────────────────────────── #


import queue, threading, sys

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import argostranslate.package as argpkg
import argostranslate.translate as argtrans
from langdetect import detect, DetectorFactory
from colorama import init as colorama_init, Fore, Style

DetectorFactory.seed = 0
colorama_init(autoreset=True)

WINDOW_SAMPS = int(SAMPLE_RATE * WINDOW_SECONDS)
HOP_SAMPS    = int(SAMPLE_RATE * HOP_SECONDS)

# ── translation helpers ───────────────────────────────────────────────────── #
def ensure_argos_models():
    argpkg.update_package_index()
    installed = {(p.from_code, p.to_code) for p in argpkg.get_installed_packages()}
    for p in argpkg.get_available_packages():
        pair = (p.from_code, p.to_code)
        if pair in {("en", "es"), ("es", "en")} and pair not in installed:
            print(f"Downloading Argos model {pair[0]}→{pair[1]} …", file=sys.stderr)
            argpkg.install_from_path(p.download())

def load_translators():
    langs = argtrans.get_installed_languages()
    en, es = (next(l for l in langs if l.code == c) for c in ("en", "es"))
    return en.get_translation(es), es.get_translation(en)

def translate_pair(text, en_to_es, es_to_en):
    try:
        src = detect(text)
    except Exception:
        src = "en"
    if src.startswith("es"):
        return es_to_en.translate(text), text
    return text, en_to_es.translate(text)
# ───────────────────────────────────────────────────────────────────────────── #

def main():
    # Whisper
    model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
    model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))  # GPU warm‑up

    # Argos
    ensure_argos_models()
    en_to_es, es_to_en = load_translators()

    audio_q = queue.Queue()

    def audio_cb(indata, frames, t, status):
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        audio_q.put(indata.copy())

    def record_loop():
        with sd.InputStream(samplerate=SAMPLE_RATE,
                            channels=1,
                            dtype="int16",
                            blocksize=HOP_SAMPS,
                            callback=audio_cb):
            print("Listening … (Ctrl+C to stop)")
            threading.Event().wait()

    def process_loop():
        buf = np.zeros((0,), dtype=np.int16)
        while True:
            buf = np.concatenate((buf, audio_q.get().ravel()))
            while buf.size >= WINDOW_SAMPS:
                window = buf[:WINDOW_SAMPS]
                buf = buf[HOP_SAMPS:]  # slide forward by hop length

                audio_f32 = window.astype(np.float32) / 32768.0
                segs, _ = model.transcribe(
                    audio_f32,
                    beam_size=BEAM_SIZE,
                    temperature=TEMPERATURE_LIST,
                    compression_ratio_threshold=COMP_RATIO_THRES,
                    no_speech_threshold=NO_SPEECH_THRES,
                    condition_on_previous_text=True,
                    language=None
                )
                text = "".join(s.text for s in segs).strip()
                if not text:
                    continue

                eng, spa = translate_pair(text, en_to_es, es_to_en)

                print(Fore.CYAN + Style.BRIGHT + f"> {text}")
                print(Fore.GREEN   + f"↳ English: {eng}")
                print(Fore.MAGENTA + f"↳ Español: {spa}")
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
