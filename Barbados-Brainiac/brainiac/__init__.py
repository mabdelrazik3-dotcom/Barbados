"""Barbados-Brainiac: line transcription for the Barbados deed books (R.O.A.D. challenge).

Approach A ports the 1st-place Barbados survey-plan text pipeline (multi-view inputs,
VLM label correction, dual-target LoRA fine-tuning, pseudo labels, output cleaning).
Approach B generates many candidate lines with several fine-tuned VLMs and a CTC reader,
scores them with VLM judges, a CTC reader and a character LM, and picks one per line
with a gradient-boosted ranker, stabilised over members and seeds and checked by gates.
Both meet in one candidate pool (fusion).
"""

__version__ = "1.0.0"
