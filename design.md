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
