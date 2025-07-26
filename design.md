# Whisper → Translation Pipeline – KPI Cheat‑Sheet

## Core KPIs

| KPI                          | What it measures                                         | Why it matters                                                             | Primary Symptoms when Bad                                 |
| ---------------------------- | -------------------------------------------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------- |
| **Latency (Speed)**          | Time from when you speak a word ➜ word appears on screen | Determines conversational feel & usability for live captions / interpreter | Text feels "laggy" (>1 s), audience talks over each other |
| **Content Accuracy**         | Words match speaker’s utterance                          | Correct info transfer; trustworthiness                                     | Mis‑heard words, mistranslations                          |
| **Silence Robustness**       | No text during pauses; no hallucinated fillers           | Avoids distraction & duplicate clutter                                     | "Thank you for watching" spam, repeated periods           |
| **Duplication Handling**     | Each spoken word appears **once**                        | Readability; prevents scroll flood                                         | Overlapping windows re‑print phrases                      |
| **Punctuation Correctness**  | Periods/commas only where sentence boundaries occur      | Improves readability without misleading pauses                             | Premature periods; run‑on sentences                       |
| **Compute Load** (Secondary) | GPU/CPU % and VRAM use per second of audio               | Determines hardware footprint, battery life                                | Overheating, dropped frames                               |

## Change / Experiment Log

### 1. Sliding Context Window

- Keep a larger context window for improved speech to text predictions
- Rather than batch predictions every new batch size, size batch over by SAMPLE_SIZE, for quicker feedback, but holding context size

Pros:

- Maintains batching accuracy
- Faster prediction results (ux)

Cons:

- Re-computes previos predictions
- Creates fuzzy prediction duplicates (need to ignore already seen predictions)

### 2. VAD Configuration for Improved Silence Detection and Response

vad_filter=True

Turns on Silero VAD inside faster-whisper.

NOTE: For now, I think i'm just going to use the default vad configurations, not going toset anything myself.

### Change / Experiment Backlog

I hear you loud and clear. You’re juggling two core KPIs—**speed** and **accuracy**—and you’ve sketched out a whole ecosystem of levers, from model size all the way down to post-context buffering. Here’s what I’ll do:

1. **Confirm I understand your map**
2. **Summarize and break down each knob and where it lives**
3. **Lay out their dependencies and an “investigation plan”** that tweaks the fewest knobs for the biggest gains, building off your simple base script.

---

## 1. Confirmation of understanding

- You want to stick with `large-v3` on CUDA/fp16 for now.
- You see a pipeline of stages where you can tune:

  1. **Model configuration** (size, compute type)
  2. **Audio sampling** (rate, chunk size)
  3. **Windowing** (WINDOW_SECONDS, HOP_SECONDS)
  4. **Inference settings** (beam_size, temperature, thresholds, VAD)
  5. **Context injection** (initial_prompt vs. sliding window vs. context-padding)
  6. **Post-processing** (de-duplication, deferred output)

- You’re interested in “context padding”—delaying output by a couple of seconds to include future context and boost accuracy.
- You want a minimal-change plan: tweak the fewest of these knobs in the right order to hit your speed/accuracy sweet-spot.

---

## 2. Knob inventory and how they fit

| Stage                       | Knob(s)                                        | Primary effect               |
| --------------------------- | ---------------------------------------------- | ---------------------------- |
| **A. Model config**         | MODEL_NAME, DEVICE, COMPUTE_TYPE               | Base compute & raw accuracy  |
| **B. Audio I/O**            | SAMPLE_RATE                                    | More samples → slower, finer |
| **C. Window management**    | WINDOW_SECONDS, HOP_SECONDS                    | Latency vs. context length   |
| **D. Inference parameters** | beam_size, temperature(s), compression_thresh, | Decoding speed vs. quality   |
|                             | no_speech_thresh, log_prob_thresh              | and silence filtering        |
|                             | vad_filter + vad_params                        | drop silent segments         |
| **E. Context injection**    | initial_prompt, condition_on_previous_text     | reduce re-decodes/repeats    |
| **F. Context padding**      | output_delay (N hops), deferred-print logic    | future context for accuracy  |
| **G. Post-processing**      | de-duplication (time, token, fuzzy)            | remove overlap repeats       |

---

## 3. Dependencies & Minimal-change “Roadmap”

1. **Enable VAD immediately**

   - Add `vad_filter=True, vad_parameters=VAD_PARAMS` to your single `transcribe(...)` call.
   - Measure how many “thank you” errors vanish and how much compute you save.

2. **Seed the decoder with initial_prompt**

   - Keep a `transcript_buf` of all printed text.
   - Pass it as `initial_prompt` + `condition_on_previous_text=False`.
   - Remove any heavy de-dup logic you built earlier.
   - Now you’ll get almost zero repeats, even with large overlap.

3. **Tune beam & temperature**

   - Start with default beam_size=1, `temperature=[0.0]`. If you see drop in accuracy on proper names, try beam=2.
   - If you get “no speech” errors, revise your `temperature` list to `[0.0,0.2]`.

4. **Experiment with WINDOW / HOP**

   - Keep WINDOW=10, HOP=1 initially.
   - If latency is too high, drop to WINDOW=6 or HOP=2.
   - Track end-to-end word-to-screen time.

5. **Optional: Add context padding**

   - Maintain a small “pending” buffer of last N words.
   - Only print words older than N × HOP_SECONDS (e.g. delay by 2 s).
   - See if output quality jumps (punctuation, mid-sentence cuts).

6. **Measure & iterate**

   - For each change, record:

     - **Latency**: word spoken → word printed
     - **Accuracy**: WER on a fixed test script

   - Aim for <500 ms average latency and WER ≤ \~8 % (tune from there).
