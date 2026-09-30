"""transformers backend: generation (greedy / beam n-best / sampling) and forced-decode scoring."""
from __future__ import annotations

import copy
import math

from ..config import Cfg
from ..text.clean import clean_line
from ..text.prompts import build_messages, parse_answer
from ..utils import chunks, free_gpu, log
from .base import Candidate, DecodeSpec, Request
from .registry import ModelSpec, accepts_kwarg, load_model, load_processor


class HFBackend:
    def __init__(self, cfg: Cfg, spec: ModelSpec, adapter: str | None = None):
        import torch

        self.torch = torch
        self.cfg, self.spec = cfg, spec
        self.processor = load_processor(spec, cfg, padding_side="left")
        self.tok = self.processor.tokenizer
        self.model = load_model(spec, cfg, adapter=adapter, merge=True)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.end_ids = self._end_token_ids()
        self.keep_logits = accepts_kwarg(self.model, "logits_to_keep")
        # transformers >= 4.50 would otherwise put back model defaults for every field equal to a global
        # default - e.g. do_sample=False -> True from Qwen3-VL's generation_config (greedy would sample)
        self.gen_kwargs = {"use_model_defaults": False} if accepts_kwarg(self.model, "use_model_defaults", "generate") else {}
        log.info("hf backend: %s%s on %s", spec.name, f" + {adapter}" if adapter else "", self.device)

    # ------------------------------------------------------------------ helpers
    def _end_token_ids(self) -> set[int]:
        ids = set()
        for token in (self.spec.end_marker, "<|endoftext|>"):
            tid = self.tok.convert_tokens_to_ids(token)
            if isinstance(tid, int) and tid >= 0 and tid != self.tok.unk_token_id:
                ids.add(tid)
        if self.tok.eos_token_id is not None:
            ids.add(int(self.tok.eos_token_id))
        return ids

    def prompt_text(self, req: Request) -> str:
        messages = build_messages(req.prompt, len(req.images), req.instruction)
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _encode(self, texts: list[str], images: list):
        enc = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        return enc.to(self.device)

    def _generation_config(self, max_new_tokens: int, **overrides):
        gc = copy.deepcopy(self.model.generation_config)
        gc.max_new_tokens = max_new_tokens
        gc.repetition_penalty = 1.0          # repeated letters and words are real in these lines
        gc.eos_token_id = sorted(self.end_ids)
        gc.pad_token_id = self.tok.pad_token_id
        gc.output_scores = True
        gc.output_logits = True
        gc.return_dict_in_generate = True
        for k, v in overrides.items():
            setattr(gc, k, v)
        return gc

    def _valid_lengths(self, gen_ids) -> list[int]:
        lengths = []
        for row in gen_ids.tolist():
            n = len(row)
            for i, t in enumerate(row):
                if t in self.end_ids:
                    n = i + 1
                    break
            lengths.append(n)
        return lengths

    def _sequence_logprobs(self, out, gen_ids, beam: bool) -> tuple[list[float], list[int]]:
        torch = self.torch
        lengths = self._valid_lengths(gen_ids)
        try:
            if beam:
                trans = self.model.compute_transition_scores(out.sequences, out.scores, out.beam_indices,
                                                             normalize_logits=False)
            else:
                raw = out.logits if getattr(out, "logits", None) is not None else out.scores
                trans = self.model.compute_transition_scores(out.sequences, raw, normalize_logits=True)
        except Exception as exc:  # noqa: BLE001 - scores are optional features
            log.warning("transition scores unavailable: %s", exc)
            return [math.nan] * gen_ids.shape[0], lengths
        trans = torch.nan_to_num(trans.float(), neginf=-50.0)
        sums = [float(trans[i, : lengths[i]].sum()) for i in range(trans.shape[0])]
        return sums, lengths

    def _decode_texts(self, gen_ids) -> list[str]:
        return self.tok.batch_decode(gen_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------ generation
    def generate(self, requests: list[Request], decode: DecodeSpec) -> list[list[Candidate]]:
        torch = self.torch
        results: list[list[Candidate]] = [[] for _ in requests]
        runs = []
        if decode.greedy:
            runs.append(("greedy", dict(do_sample=False, num_beams=1, num_return_sequences=1), False))
        if decode.beam:
            nb = int(decode.beam.get("num_beams", 5))
            nr = min(int(decode.beam.get("num_return", nb)), nb)
            runs.append(("beam", dict(do_sample=False, num_beams=nb, num_return_sequences=nr,
                                      length_penalty=float(decode.beam.get("length_penalty", 1.0)),
                                      early_stopping=True), True))
        if decode.sample:
            runs.append(("sample", dict(do_sample=True, num_beams=1, num_return_sequences=int(decode.sample.get("n", 4)),
                                        temperature=float(decode.sample.get("temperature", 0.8)),
                                        top_p=float(decode.sample.get("top_p", 0.95)), top_k=0), False))
        bs = int(self.cfg.hf.generate_batch_size)
        for idx in chunks(list(range(len(requests))), bs):
            reqs = [requests[i] for i in idx]
            enc = self._encode([self.prompt_text(r) for r in reqs], [im for r in reqs for im in r.images])
            plen = enc["input_ids"].shape[1]
            for method, kw, beam in runs:
                gc = self._generation_config(decode.max_new_tokens, **kw)
                with torch.inference_mode():
                    out = self.model.generate(**enc, generation_config=gc, **self.gen_kwargs)
                gen_ids = out.sequences[:, plen:]
                sums, lengths = self._sequence_logprobs(out, gen_ids, beam)
                texts = self._decode_texts(gen_ids)
                n_ret = kw["num_return_sequences"]
                for b, i in enumerate(idx):
                    for r in range(n_ret):
                        k = b * n_ret + r
                        parsed = parse_answer(texts[k])
                        results[i].append(Candidate(
                            text=clean_line(parsed["transcription"]), raw=texts[k], method=method, rank=r,
                            logprob=sums[k], ntok=lengths[k],
                            ink=clean_line(parsed["ink"]) if parsed["ink"] else None,
                        ))
                del out
            del enc
        return results

    # ------------------------------------------------------------------ forced decoding
    def score(self, requests: list[Request], answers: list[list[str]]) -> list[list[dict]]:
        """Log-probs of each answer (JSON text) given the request, by teacher forcing.

        Left padding puts every answer at the end of its row, so only the last
        max(answer tokens) + 1 logits are needed (`logits_to_keep`).
        """
        torch = self.torch
        sbs = int(self.cfg.hf.score_batch_size)
        results: list[list[dict]] = []
        for req, ans_list in zip(requests, answers):
            prompt = self.prompt_text(req)
            rows: list[dict] = []
            for ans_chunk in chunks(list(ans_list), sbs):
                texts = [prompt + a + self.spec.end_marker for a in ans_chunk]
                enc = self._encode(texts, list(req.images) * len(ans_chunk))
                ans_ids = [self.tok(a + self.spec.end_marker, add_special_tokens=False)["input_ids"] for a in ans_chunk]
                keep = max(len(x) for x in ans_ids) + 1
                kwargs = {"use_cache": False}
                if self.keep_logits:
                    kwargs["logits_to_keep"] = keep
                with torch.inference_mode():
                    logits = self.model(**enc, **kwargs).logits[:, -keep:, :].float()
                logp = torch.log_softmax(logits, dim=-1)
                ids = enc["input_ids"]
                total = ids.shape[1]
                for b, target_ids in enumerate(ans_ids):
                    n = len(target_ids)
                    tgt = ids[b, total - n:]
                    pos = torch.arange(keep - n - 1, keep - 1, device=logp.device)
                    lp = logp[b, pos, :].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                    rows.append({"sum": float(lp.sum()), "mean": float(lp.mean()), "min": float(lp.min()),
                                 "ntok": n, "aligned": tgt.tolist() == target_ids})
                del logits, logp, enc
            results.append(rows)
        return results

    def close(self) -> None:
        del self.model
        free_gpu()
