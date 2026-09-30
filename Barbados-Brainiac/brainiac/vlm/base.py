"""Backend-independent request/response types and the backend factory."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..config import Cfg


@dataclass
class Request:
    id: str
    prompt: str                  # text before the images (shared prefix)
    images: list                 # PIL images (the line's views)
    instruction: str             # text after the images
    meta: dict = field(default_factory=dict)


@dataclass
class Candidate:
    text: str                    # cleaned transcription
    raw: str                     # raw model output
    method: str                  # greedy | beam | sample | ink
    rank: int
    logprob: float = math.nan    # sum of token log-probs of the answer
    ntok: int = 0
    ink: str | None = None       # the "ink" field of a dual-target answer

    def as_row(self, image_id: str, source: str) -> dict:
        mean = self.logprob / self.ntok if self.ntok and not math.isnan(self.logprob) else math.nan
        return {"ID": image_id, "source": source, "text": self.text, "raw": self.raw, "method": self.method,
                "rank": self.rank, "gen_logprob": self.logprob, "gen_ntok": self.ntok, "gen_mean": mean,
                "ink": self.ink or ""}


@dataclass
class DecodeSpec:
    greedy: bool = True
    beam: dict | None = None      # {num_beams, num_return, length_penalty}
    sample: dict | None = None    # {n, temperature, top_p}
    max_new_tokens: int = 160
    include_ink: bool = False

    @classmethod
    def from_cfg(cls, spec: dict | None, max_new_tokens: int) -> "DecodeSpec":
        spec = dict(spec or {})
        return cls(
            greedy=bool(spec.get("greedy", True)),
            beam=dict(spec["beam"]) if spec.get("beam") else None,
            sample=dict(spec["sample"]) if spec.get("sample") else None,
            max_new_tokens=int(spec.get("max_new_tokens", max_new_tokens)),
            include_ink=bool(spec.get("include_ink", False)),
        )


class Backend(Protocol):
    def generate(self, requests: list[Request], decode: DecodeSpec) -> list[list[Candidate]]: ...

    def score(self, requests: list[Request], answers: list[list[str]]) -> list[list[dict]]: ...

    def close(self) -> None: ...


def make_backend(cfg: Cfg, model_name: str, adapter: str | Path | None = None, kind: str | None = None) -> Any:
    """Load one model (+ optional LoRA adapter) behind the configured inference backend."""
    from .registry import get_spec

    kind = kind or cfg.backend
    spec = get_spec(cfg, model_name)
    adapter = str(adapter) if adapter else None
    if kind == "hf":
        from .hf_backend import HFBackend

        return HFBackend(cfg, spec, adapter)
    if kind == "vllm":
        from .vllm_backend import VLLMBackend

        return VLLMBackend(cfg, spec, adapter)
    if kind == "mock":
        from .mock_backend import MockBackend

        return MockBackend(cfg, spec, adapter)
    raise ValueError(f"unknown backend '{kind}'")
