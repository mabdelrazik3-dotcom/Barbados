"""vLLM backend: fast generation and forced-decode scoring (prompt log-probs) with LoRA adapters.

The prompt text before the images is identical for every line, so vLLM's prefix cache
computes the (long) mega-prompt prefix once for generation.
"""
from __future__ import annotations

import math

from ..config import Cfg
from ..text.clean import clean_line
from ..text.prompts import build_messages, parse_answer
from ..utils import free_gpu, log
from .base import Candidate, DecodeSpec, Request
from .registry import ModelSpec, load_processor, model_source


class VLLMBackend:
    def __init__(self, cfg: Cfg, spec: ModelSpec, adapter: str | None = None, max_images: int = 4):
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        self.cfg, self.spec = cfg, spec
        self.processor = load_processor(spec, cfg, padding_side="left")  # chat template + tokenizer only
        self.tok = self.processor.tokenizer
        kwargs = dict(
            model=model_source(spec),
            dtype=spec.dtype,
            max_model_len=int(cfg.vllm.max_model_len),
            gpu_memory_utilization=float(cfg.vllm.gpu_memory_utilization),
            seed=int(cfg.seed),
            limit_mm_per_prompt={"image": max_images},
            enable_prefix_caching=True,
            enforce_eager=bool(cfg.vllm.enforce_eager),
        )
        if cfg.paths.get("hf_cache"):
            kwargs["download_dir"] = cfg.paths.hf_cache
        if spec.load_in_4bit:
            kwargs["quantization"] = "bitsandbytes"
        if adapter:
            kwargs.update(enable_lora=True, max_lora_rank=int(cfg.vllm.max_lora_rank), max_loras=1)
        self.llm = LLM(**kwargs)
        self.lora = LoRARequest("brainiac_adapter", 1, str(adapter)) if adapter else None
        log.info("vllm backend: %s%s", spec.name, f" + {adapter}" if adapter else "")

    def prompt_text(self, req: Request) -> str:
        messages = build_messages(req.prompt, len(req.images), req.instruction)
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @staticmethod
    def _mm(req: Request) -> dict:
        return {"image": req.images if len(req.images) > 1 else req.images[0]}

    def _inputs(self, requests: list[Request], suffixes: list[str] | None = None) -> list[dict]:
        out = []
        for i, r in enumerate(requests):
            text = self.prompt_text(r) + (suffixes[i] if suffixes else "")
            out.append({"prompt": text, "multi_modal_data": self._mm(r)})
        return out

    def _candidate(self, text: str, method: str, rank: int, logprob, ntok: int) -> Candidate:
        parsed = parse_answer(text)
        return Candidate(text=clean_line(parsed["transcription"]), raw=text, method=method, rank=rank,
                         logprob=math.nan if logprob is None else float(logprob), ntok=ntok,
                         ink=clean_line(parsed["ink"]) if parsed["ink"] else None)

    def _sample_run(self, inputs, n: int, temperature: float, top_p: float, max_tokens: int, method: str,
                    results: list[list[Candidate]]) -> None:
        from vllm import SamplingParams

        sp = SamplingParams(n=n, temperature=temperature, top_p=top_p if temperature > 0 else 1.0, top_k=-1,
                            max_tokens=max_tokens, logprobs=0, repetition_penalty=1.0, seed=int(self.cfg.seed))
        outs = self.llm.generate(inputs, sp, lora_request=self.lora, use_tqdm=True)
        for i, o in enumerate(outs):
            for r, c in enumerate(o.outputs):
                results[i].append(self._candidate(c.text, method, r, c.cumulative_logprob, len(c.token_ids)))

    def generate(self, requests: list[Request], decode: DecodeSpec) -> list[list[Candidate]]:
        inputs = self._inputs(requests)
        results: list[list[Candidate]] = [[] for _ in requests]
        if decode.greedy:
            self._sample_run(inputs, 1, 0.0, 1.0, decode.max_new_tokens, "greedy", results)
        if decode.beam:
            nb = int(decode.beam.get("num_beams", 5))
            nr = min(int(decode.beam.get("num_return", nb)), nb)
            try:
                from vllm.sampling_params import BeamSearchParams

                params = BeamSearchParams(beam_width=nb, max_tokens=decode.max_new_tokens,
                                          length_penalty=float(decode.beam.get("length_penalty", 1.0)))
                outs = self.llm.beam_search(inputs, params, lora_request=self.lora)
                for i, o in enumerate(outs):
                    for r, seq in enumerate(o.sequences[:nr]):
                        text = seq.text or ""
                        if text.startswith(inputs[i]["prompt"]):
                            text = text[len(inputs[i]["prompt"]):]
                        # seq.tokens holds prompt + answer; count the answer (+ end marker) from its text
                        ntok = len(self.tok(text, add_special_tokens=False)["input_ids"]) + 1
                        results[i].append(self._candidate(text, "beam", r, seq.cum_logprob, ntok))
            except Exception as exc:  # noqa: BLE001 - beam search support differs across vLLM versions
                log.warning("vLLM beam search unavailable (%s); using %d low-temperature samples instead", exc, nr)
                self._sample_run(inputs, nr, float(decode.beam.get("fallback_temperature", 0.5)), 0.95,
                                 decode.max_new_tokens, "beam", results)
        if decode.sample:
            self._sample_run(inputs, int(decode.sample.get("n", 4)), float(decode.sample.get("temperature", 0.8)),
                             float(decode.sample.get("top_p", 0.95)), decode.max_new_tokens, "sample", results)
        return results

    def score(self, requests: list[Request], answers: list[list[str]]) -> list[list[dict]]:
        from vllm import SamplingParams

        flat_reqs, suffixes, index = [], [], []
        for i, (r, ans) in enumerate(zip(requests, answers)):
            for j, a in enumerate(ans):
                flat_reqs.append(r)
                suffixes.append(a + self.spec.end_marker)
                index.append((i, j, a))
        sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
        outs = self.llm.generate(self._inputs(flat_reqs, suffixes), sp, lora_request=self.lora, use_tqdm=True)
        results: list[list[dict]] = [[{} for _ in ans] for ans in answers]
        for (i, j, a), o in zip(index, outs):
            ids = list(o.prompt_token_ids)
            target = self.tok(a + self.spec.end_marker, add_special_tokens=False)["input_ids"]
            n = len(target)
            vals = []
            for t in range(len(ids) - n, len(ids)):
                entry = o.prompt_logprobs[t] or {}
                lp = entry.get(ids[t])
                vals.append(lp.logprob if lp is not None else -50.0)
            results[i][j] = {"sum": float(sum(vals)), "mean": float(sum(vals) / max(n, 1)), "min": float(min(vals)),
                             "ntok": n, "aligned": ids[-n:] == target}
        return results

    def close(self) -> None:
        del self.llm
        free_gpu()
