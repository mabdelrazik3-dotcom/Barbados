"""Output cleaning — the 1st place's `clean_text_preds.py`, for line transcriptions.

The 1st place normalised parish names, units and surveyor names into the competition's formats.
Here the target is one line in the collection's encoding, so cleaning maps whatever a model
writes into that character set: Unicode look-alikes to their ASCII conventions (long s -> s,
superscript letters -> ^ run, curly quotes, dashes), `+` et-signs to `&`, stray JSON/code-fence
debris and characters outside the set are removed, whitespace is collapsed.
"""
from __future__ import annotations

import re
import string
import unicodedata

from .prompts import parse_answer

ALLOWED = set(string.ascii_letters + string.digits + " ^:;.,-~_&'\"()")

TRANSLATE = {
    "ſ": "s",   # long s
    "þ": "y",   # thorn written as y in this collection (ye, yt)
    "Þ": "Y",
    "⁊": "&",   # Tironian et
    "+": "&",        # plus-shaped et-sign (the prompts: never +)
    "‘": "'", "’": "'", "ʼ": "'", "`": "'",
    "“": '"', "”": '"', "″": '"',
    "–": "-", "—": "-", "−": "-", "‐": "-",
    "…": "...",
    " ": " ",
    "[": "", "]": "", "{": "", "}": "", "|": "", "\\": "", "*": "", "#": "", "?": "",
}

SUPERSCRIPTS = {
    "ᵃ": "a", "ᵇ": "b", "ᶜ": "c", "ᵈ": "d", "ᵉ": "e", "ᶠ": "f",
    "ᵍ": "g", "ʰ": "h", "ⁱ": "i", "ʲ": "j", "ᵏ": "k", "ˡ": "l",
    "ᵐ": "m", "ⁿ": "n", "ᵒ": "o", "ᵖ": "p", "ʳ": "r", "ˢ": "s",
    "ᵗ": "t", "ᵘ": "u", "ᵛ": "v", "ʷ": "w", "ˣ": "x", "ʸ": "y",
    "ᶻ": "z", "²": "2", "³": "3", "¹": "1", "⁰": "0",
}


def _superscripts_to_caret(text: str) -> str:
    out, in_run = [], False
    for ch in text:
        if ch in SUPERSCRIPTS:
            if not in_run:
                out.append("^")
                in_run = True
            out.append(SUPERSCRIPTS[ch])
        else:
            in_run = False
            out.append(ch)
    return "".join(out)


def clean_line(text: str, allowed: set[str] = ALLOWED) -> str:
    """Model output (raw or JSON) -> one line in the collection's encoding."""
    if text is None:
        return ""
    s = str(text)
    if "{" in s and "transcription" in s:
        s = parse_answer(s)["transcription"]
    s = _superscripts_to_caret(s)
    s = "".join(TRANSLATE.get(ch, ch) for ch in s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = "".join(ch if ch in allowed else " " for ch in s)
    s = re.sub(r"\^+", "^", s)
    s = re.sub(r"\^(?=\s|$)", "", s)          # a caret raises nothing
    s = re.sub(r"\s+", " ", s).strip()
    return s
