"""Mock backend for wiring checks (configs/dry_run.yaml): no model, deterministic fake outputs.

Generation returns the request's reference text (meta["reference"], when the caller passes one)
with a few random character edits; scoring returns a log-prob that falls with the edit distance.
"""
from __future__ import annotations

import random
import zlib

from ..config import Cfg
from ..metric import line_loss
from ..text.prompts import parse_answer
from .base import Candidate, DecodeSpec, Request
from .registry import ModelSpec

_ALPHABET = "abcdefghijklmnopqrstuvwxyz ^:&"


def _perturb(text: str, rng: random.Random, n_edits: int) -> str:
    chars = list(text or "the said")
    for _ in range(n_edits):
        op, pos = rng.random(), rng.randrange(max(1, len(chars)))
        if op < 0.4 and chars:
            chars[pos] = rng.choice(_ALPHABET)
        elif op < 0.7:
            chars.insert(pos, rng.choice(_ALPHABET))
        elif len(chars) > 1:
            del chars[pos]
    return " ".join("".join(chars).split())


class MockBackend:
    def __init__(self, cfg: Cfg, spec: ModelSpec, adapter: str | None = None):
        self.cfg, self.spec, self.adapter = cfg, spec, adapter
        self.salt = zlib.crc32(f"{spec.name}:{adapter}".encode())

    def _rng(self, req: Request, tag: str) -> random.Random:
        return random.Random(zlib.crc32(f"{req.id}:{tag}".encode()) ^ self.salt)

    def generate(self, requests: list[Request], decode: DecodeSpec) -> list[list[Candidate]]:
        out = []
        for req in requests:
            ref = req.meta.get("reference", "")
            rng = self._rng(req, "gen")
            cands = []
            if decode.greedy:
                cands.append(Candidate(_perturb(ref, rng, rng.randint(0, 2)), "", "greedy", 0, -3.0, 20))
            n_beam = int((decode.beam or {}).get("num_return", 0))
            n_sample = int((decode.sample or {}).get("n", 0))
            for r in range(n_beam):
                cands.append(Candidate(_perturb(ref, rng, rng.randint(0, 4)), "", "beam", r, -4.0 - r, 20))
            for r in range(n_sample):
                cands.append(Candidate(_perturb(ref, rng, rng.randint(1, 6)), "", "sample", r, -6.0 - r, 20))
            for c in cands:
                c.raw = '{"transcription": "%s"}' % c.text
                c.ink = c.text if decode.include_ink else None
            out.append(cands)
        return out

    def score(self, requests: list[Request], answers: list[list[str]]) -> list[list[dict]]:
        out = []
        for req, ans in zip(requests, answers):
            ref = req.meta.get("reference", "")
            rng = self._rng(req, "score")
            rows = []
            for a in ans:
                text = parse_answer(a)["transcription"]
                lp = -40.0 * line_loss(ref, text) - rng.random()
                rows.append({"sum": lp, "mean": lp / 20, "min": lp / 5, "ntok": 20, "aligned": True})
            out.append(rows)
        return out

    def close(self) -> None:
        pass
