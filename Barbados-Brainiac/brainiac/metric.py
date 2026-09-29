"""The R.O.A.D. leaderboard metric.

    score = 0.5 * (1 - mean_word_edits / 12) + 0.5 * (1 - mean_char_edits / 55)   (higher is better)

Edits are Levenshtein distances per line (words, then characters incl. spaces) after collapsing
whitespace; comparison is case- and punctuation-sensitive. The loss one line contributes,
0.5 * w / 12 + 0.5 * c / 55, is the ranker target and the MBR utility.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

try:  # C++ edit distance; the pure-Python fallback is correct but slow
    from rapidfuzz.distance import Levenshtein as _RF
except ImportError:  # pragma: no cover
    _RF = None

WORD_XMAX = 12.0
CHAR_XMAX = 55.0


def norm(text) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    return " ".join(str(text).split())


def _levenshtein_py(a: Sequence, b: Sequence) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def levenshtein(a: Sequence, b: Sequence) -> int:
    if _RF is not None:
        return int(_RF.distance(a, b))
    return _levenshtein_py(a, b)


def line_edits(ref, hyp) -> tuple[int, int]:
    r, h = norm(ref), norm(hyp)
    return levenshtein(r.split(), h.split()), levenshtein(r, h)


def line_loss(ref, hyp, word_xmax: float = WORD_XMAX, char_xmax: float = CHAR_XMAX) -> float:
    w, c = line_edits(ref, hyp)
    return 0.5 * w / word_xmax + 0.5 * c / char_xmax


def score(refs: Iterable, hyps: Iterable, word_xmax: float = WORD_XMAX, char_xmax: float = CHAR_XMAX) -> dict:
    refs, hyps = list(refs), list(hyps)
    if len(refs) != len(hyps):
        raise ValueError("refs and hyps differ in length")
    we = np.zeros(len(refs))
    ce = np.zeros(len(refs))
    for i, (r, h) in enumerate(zip(refs, hyps)):
        we[i], ce[i] = line_edits(r, h)
    nw = sum(max(len(norm(r).split()), 1) for r in refs)
    nc = sum(max(len(norm(r)), 1) for r in refs)
    n = max(len(refs), 1)
    return {
        "score": float(0.5 * (1 - we.sum() / n / word_xmax) + 0.5 * (1 - ce.sum() / n / char_xmax)),
        "word_edits_per_line": float(we.sum() / n),
        "char_edits_per_line": float(ce.sum() / n),
        "corpus_wer": float(we.sum() / max(nw, 1)),
        "corpus_cer": float(ce.sum() / max(nc, 1)),
        "n": len(refs),
        "empty_preds": int(sum(1 for h in hyps if not norm(h))),
    }


def loss_matrix(cands: Sequence[str], word_xmax: float = WORD_XMAX, char_xmax: float = CHAR_XMAX) -> np.ndarray:
    """M[i, j] = line loss of candidate j measured against candidate i (symmetric, zero diagonal)."""
    k = len(cands)
    m = np.zeros((k, k))
    normed = [norm(c) for c in cands]
    words = [c.split() for c in normed]
    for i in range(k):
        for j in range(i + 1, k):
            v = 0.5 * levenshtein(words[i], words[j]) / word_xmax + 0.5 * levenshtein(normed[i], normed[j]) / char_xmax
            m[i, j] = m[j, i] = v
    return m


def paired_bootstrap(loss_a: np.ndarray, loss_b: np.ndarray, n_boot: int = 10000, seed: int = 0) -> dict:
    """Bootstrap over lines of mean(loss_a - loss_b); positive means B is better (lower loss)."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return {
        "mean_delta_loss": float(d.mean()),
        "mean_delta_score": float(d.mean()),  # score = 1 - mean loss, so the deltas are equal
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "p_b_better": float((boots > 0).mean()),
        "n_improved": int((d > 0).sum()),
        "n_worsened": int((d < 0).sum()),
    }
