"""Prompts from mega_prompt.md (v1) and mega_prompt_v2.md (v2), and answer parsing.

A prompt spec is a dict:
    version:  v1 | v2 | custom
    mode:     full      the whole markdown file
              compact   selected sections (default: ROLE, SCOPE, ENCODING quick reference, OUTPUT)
              sections  the sections listed in `sections` (numbers like "4" or title words like "SPELLING")
              short     a short instruction with the encoding essentials
    sections: [...]     for mode sections / to override the compact selection
    path:     file      for version custom

The chat layout is identical in training and inference for every backend:
    user: [prompt text, image 1..n, view instruction]      assistant: answer JSON
so the prompt prefix (text before the first image) is shared by all lines — vLLM caches it.
"""
from __future__ import annotations

import ast
import json
import re
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

from ..config import Cfg

COMPACT_SECTIONS = ["ROLE", "SCOPE", "QUICK REFERENCE", "OUTPUT FORMAT"]

SHORT_PROMPT = """You transcribe one line of handwriting from the Barbados deed and record books (1639-1710).
Copy the ink exactly: keep the scribe's spelling, capitals, abbreviations, slips and word gaps;
never modernise, expand, correct or complete a word. Transcribe only the target line (the complete
line through the middle); leave out cut neighbouring lines and archivists' numbers.
Encoding: long s = s; any et-sign = &; raised letters = ^ before the first raised letter (y^e W^m 24^th);
colon or dot under/before raised letters comes before the ^ (S:^d y.^e); a colon or period after an
abbreviation stays attached (Tho: Capt.); marks over letters are dropped (plantacon pte psents);
crossed-out text = ~word~; words written above the line = ^word; a wavy line filler = ~ ; a dash = -.
Characters: ASCII letters, digits, space and ^ : ; . , - ~ _ & ' " ( ) only.
Return ONLY this JSON object: {"transcription": "..."}"""

CORRECTION_HEADER = """LABEL CORRECTION — ONE HANDWRITTEN LINE

You receive the image(s) of one handwritten line and a candidate transcription of it that may contain
errors: misread letters, modernised spellings, wrong capitals, missing or invented marks (^ : . - ~),
a word taken from a neighbouring line, a word left out, or a cut last word that was completed.
Verify the candidate word by word against the ink and return the corrected transcription in the
conventions below. Keep every word the ink confirms exactly as given; change only what the ink clearly
contradicts; never rewrite the line from memory or from the sense.

=== CONVENTIONS ===
"""

DUAL_INSTRUCTIONS = {
    "ink_first": (
        'OUTPUT (this replaces the output format above): return ONLY this JSON object: '
        '{"ink": "...", "transcription": "..."} - "ink" is your letter-exact reading of the ink, '
        '"transcription" is the reference transcription of the line in this collection\'s style.'
    ),
    "transcription_first": (
        'OUTPUT (this replaces the output format above): return ONLY this JSON object: '
        '{"transcription": "...", "ink": "..."} - "transcription" is the reference transcription of the line '
        'in this collection\'s style, "ink" is your letter-exact reading of the ink.'
    ),
}


@lru_cache(maxsize=8)
def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def split_sections(markdown: str) -> "OrderedDict[str, tuple[str, str]]":
    """Split on level-2 headings. Key: section number ("3") or the title for unnumbered ones ("ROLE")."""
    out: "OrderedDict[str, tuple[str, str]]" = OrderedDict()
    current_key, current_title, buf = "_preamble", "", []
    for line in markdown.splitlines():
        if line.startswith("## "):
            if buf:
                out[current_key] = (current_title, "\n".join(buf).strip())
            current_title = line[3:].strip()
            m = re.match(r"(\d+)\.", current_title)
            current_key = m.group(1) if m else current_title.upper()
            buf = [line]
        else:
            buf.append(line)
    if buf:
        out[current_key] = (current_title, "\n".join(buf).strip())
    return out


def select_sections(markdown: str, selectors: list[str]) -> str:
    sections = split_sections(markdown)
    chosen = []
    for key, (title, text) in sections.items():
        if key == "_preamble":
            continue
        for sel in selectors:
            sel_u = str(sel).upper()
            if sel_u == key or (not sel_u.isdigit() and sel_u in title.upper()):
                chosen.append(text.replace("\n---", "").strip())
                break
    head = sections.get("_preamble", ("", ""))[1].splitlines()
    title_line = head[0] if head else ""
    return "\n\n".join([title_line] + chosen).strip()


def prompt_file(cfg: Cfg, spec: dict) -> str:
    version = spec.get("version", "v2")
    if version == "custom":
        return str(Path(spec["path"]))
    key = {"v1": "prompt_v1", "v2": "prompt_v2"}.get(version)
    if key is None:
        raise ValueError(f"unknown prompt version '{version}'")
    return cfg.paths[key]


def build_prompt(cfg: Cfg, spec: dict, dual: bool = False, dual_order: str = "ink_first") -> str:
    """The prompt text placed before the image(s)."""
    spec = dict(spec or {})
    mode = spec.get("mode", "compact")
    if mode == "short":
        text = SHORT_PROMPT
    else:
        md = _read(prompt_file(cfg, spec))
        if mode == "full":
            text = md.strip()
        elif mode in ("compact", "sections"):
            text = select_sections(md, list(spec.get("sections") or COMPACT_SECTIONS))
        else:
            raise ValueError(f"unknown prompt mode '{mode}'")
    if dual:
        text = text + "\n\n" + DUAL_INSTRUCTIONS[dual_order]
    return text


def correction_prompt(cfg: Cfg, spec: dict, candidate: str) -> str:
    return CORRECTION_HEADER + build_prompt(cfg, spec) + "\n\n=== CANDIDATE TRANSCRIPTION ===\n" + candidate


def build_messages(prompt_text: str, n_images: int, instruction: str) -> list[dict]:
    content = [{"type": "text", "text": prompt_text}]
    content += [{"type": "image"} for _ in range(n_images)]
    content.append({"type": "text", "text": instruction})
    return [{"role": "user", "content": content}]


def answer_json(transcription: str, ink: str | None = None, dual: bool = False, dual_order: str = "ink_first") -> str:
    if not dual:
        return json.dumps({"transcription": transcription}, ensure_ascii=False)
    ink = transcription if ink is None else ink
    if dual_order == "ink_first":
        return json.dumps({"ink": ink, "transcription": transcription}, ensure_ascii=False)
    return json.dumps({"transcription": transcription, "ink": ink}, ensure_ascii=False)


# ------------------------------------------------------------------ parsing (1st-place helpers)
def _extract_first_json_object(s: str) -> str | None:
    """Return the first balanced {...} substring, ignoring braces inside strings."""
    s = s.strip()
    s = re.sub(r"^\s*```[a-zA-Z0-9_-]*\s*\n", "", s)
    s = re.sub(r"\n\s*```\s*$", "", s)
    start, depth, in_str, esc, quote = None, 0, False, False, None
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
        elif ch == '"':
            in_str, quote = True, ch
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                return s[start : i + 1]
    if "{" in s and "}" in s and s.find("{") < s.rfind("}"):
        return s[s.find("{") : s.rfind("}") + 1]
    return None


def _repair_jsonish(s: str) -> str:
    s = s.replace("“", '"').replace("”", '"')
    s = re.sub(r",\s*(?=[}\]])", "", s)
    return s


def parse_answer(text: str) -> dict:
    """{"transcription": str, "ink": str | None, "parsed": bool} from raw model output."""
    raw = (text or "").strip()
    cand = _extract_first_json_object(raw)
    obj = None
    if cand is not None:
        for attempt in (cand, _repair_jsonish(cand)):
            try:
                obj = json.loads(attempt)
                break
            except (json.JSONDecodeError, ValueError):
                pass
        if obj is None:
            try:
                val = ast.literal_eval(cand)
                obj = val if isinstance(val, dict) else None
            except (ValueError, SyntaxError):
                obj = None
    if isinstance(obj, dict):
        tr = obj.get("transcription", obj.get("text", ""))
        ink = obj.get("ink")
        return {"transcription": str(tr or ""), "ink": None if ink is None else str(ink), "parsed": True}
    m = re.search(r'"transcription"\s*:\s*"((?:[^"\\]|\\.)*)', raw)  # truncated JSON
    if m:
        try:
            return {"transcription": json.loads('"' + m.group(1) + '"'), "ink": None, "parsed": False}
        except json.JSONDecodeError:
            return {"transcription": m.group(1), "ink": None, "parsed": False}
    return {"transcription": raw.strip("`").strip(), "ink": None, "parsed": False}
