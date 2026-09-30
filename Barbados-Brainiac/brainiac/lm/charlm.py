"""Interpolated Kneser-Ney character n-gram LM (the diagram's "LM-X char n-gram").

Trained per scope on the labels a scope may see (oof: every fold but the holdout fold), so the
holdout lines' own references never score their candidates. Features per candidate: total and
per-character log-probability, the weakest character, and characters never seen in training.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

BOS, EOS = "\x02", "\x03"


class CharNGramLM:
    def __init__(self, order: int = 7, discount: float = 0.75):
        self.order, self.d = int(order), float(discount)
        self.counts: list[dict[str, Counter]] = []
        self.totals: list[dict[str, int]] = []
        self.vocab: set[str] = set()

    def fit(self, texts) -> "CharNGramLM":
        n = self.order
        raw = [defaultdict(Counter) for _ in range(n + 1)]  # raw[k][context of length k-1][char]
        for t in texts:
            s = BOS * (n - 1) + " ".join(str(t).split()) + EOS
            for i in range(n - 1, len(s)):
                self.vocab.add(s[i])
                for k in range(1, n + 1):
                    raw[k][s[i - k + 1 : i]][s[i]] += 1
        # Kneser-Ney: lower orders use continuation counts (distinct left extensions)
        self.counts = [defaultdict(Counter) for _ in range(n + 1)]
        self.counts[n] = raw[n]
        for k in range(1, n):
            for ctx_long, ctr in raw[k + 1].items():
                ctx = ctx_long[1:]
                for c in ctr:
                    self.counts[k][ctx][c] += 1
        self.totals = [{ctx: sum(ctr.values()) for ctx, ctr in level.items()} for level in self.counts]
        return self

    def _prob(self, c: str, history: str, k: int) -> float:
        if k == 0:
            return 1.0 / (len(self.vocab) + 1)
        ctx = history[len(history) - (k - 1):] if k > 1 else ""
        lower = self._prob(c, history, k - 1)
        ctr = self.counts[k].get(ctx)
        if not ctr:
            return lower
        total = self.totals[k][ctx]
        return max(ctr.get(c, 0) - self.d, 0.0) / total + self.d * len(ctr) / total * lower

    def char_logprob(self, c: str, context: str) -> float:
        history = (BOS * (self.order - 1) + context)[-(self.order - 1):] if self.order > 1 else ""
        return math.log(self._prob(c, history, self.order))

    def score(self, text: str) -> dict:
        s = " ".join(str(text).split())
        history = BOS * (self.order - 1)
        lps = []
        for c in s + EOS:
            lps.append(math.log(self._prob(c, history, self.order)))
            history = (history + c)[-(self.order - 1):] if self.order > 1 else ""
        return {
            "lm_lp": float(sum(lps)),
            "lm_lp_pc": float(sum(lps) / len(lps)),
            "lm_min": float(min(lps)),
            "lm_oov": sum(1 for c in s if c not in self.vocab),
        }
