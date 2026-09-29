"""Official R.O.A.D. Barbados leaderboard metric (reverse-engineered, see FULL_DATA_AUDIT.md §1).

Zindi page: score = 0.5 * weighted-WER + 0.5 * weighted-CER, "longer references weighted more".
Weighting a per-line rate by its reference length turns it into a per-line EDIT COUNT, and the
leaderboard normalises each mean edit count by the benchmark's x_max (12 words / 55 chars):

    score = 0.5 * (1 - mean_word_edits / 12) + 0.5 * (1 - mean_char_edits / 55)     (HIGHER = better)

This form was posted on the competition forum (thread 34861, credited to J0NNY) and reproduces the
Benchmark row to 2e-10 and several submissions to nine decimals. Text is normalised the way jiwer's
defaults do (strip + collapse whitespace); comparison is case- and punctuation-sensitive, spaces count
as characters for the char edit distance.
"""
from __future__ import annotations

import numpy as np
from rapidfuzz.distance import Levenshtein

WORD_XMAX = 12.0
CHAR_XMAX = 55.0


def norm(s) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    return " ".join(str(s).split())


def line_edits(ref: str, hyp: str) -> tuple[int, int]:
    r, h = norm(ref), norm(hyp)
    return Levenshtein.distance(r.split(), h.split()), Levenshtein.distance(r, h)


def edit_arrays(refs, hyps):
    we = np.empty(len(refs), np.float64)
    ce = np.empty(len(refs), np.float64)
    for i, (r, h) in enumerate(zip(refs, hyps)):
        we[i], ce[i] = line_edits(r, h)
    return we, ce


def score_from_edits(we, ce) -> float:
    return 0.5 * (1 - np.mean(we) / WORD_XMAX) + 0.5 * (1 - np.mean(ce) / CHAR_XMAX)


def official_metric(refs, hyps) -> dict:
    """Returns the leaderboard score plus its components (all computed per line, then averaged)."""
    refs, hyps = list(refs), list(hyps)
    we, ce = edit_arrays(refs, hyps)
    nw = np.array([max(len(norm(r).split()), 1) for r in refs], np.float64)
    nc = np.array([max(len(norm(r)), 1) for r in refs], np.float64)
    return {
        "score": score_from_edits(we, ce),
        "word_edits_per_line": float(we.mean()),   # the leaderboard's "WER" column
        "char_edits_per_line": float(ce.mean()),   # the leaderboard's "CER" column
        "corpus_wer": float(we.sum() / nw.sum()),
        "corpus_cer": float(ce.sum() / nc.sum()),
        "n": len(refs),
        "empty_preds": int(sum(1 for h in hyps if not norm(h))),
    }


def line_cost(ref: str, hyp: str) -> float:
    """Score loss contributed by one line (x N lines = total loss). Used for MBR / ROVER selection."""
    w, c = line_edits(ref, hyp)
    return 0.5 * w / WORD_XMAX + 0.5 * c / CHAR_XMAX


def notebook_metric(refs, hyps):
    """The metric barbados-2.ipynb prints: 0.5*corpus WER + 0.5*corpus CER (jiwer). Lower = better."""
    we, ce = edit_arrays(refs, hyps)
    nw = sum(max(len(norm(r).split()), 1) for r in refs)
    nc = sum(max(len(norm(r)), 1) for r in refs)
    return 0.5 * we.sum() / nw + 0.5 * ce.sum() / nc


def bootstrap_delta(we_a, ce_a, we_b, ce_b, n_boot=10000, seed=0) -> dict:
    """Paired bootstrap over whole lines of score(B) - score(A)."""
    rng = np.random.default_rng(seed)
    n = len(we_a)
    la = 0.5 * np.asarray(we_a) / WORD_XMAX + 0.5 * np.asarray(ce_a) / CHAR_XMAX
    lb = 0.5 * np.asarray(we_b) / WORD_XMAX + 0.5 * np.asarray(ce_b) / CHAR_XMAX
    d = la - lb                                   # positive -> B has lower loss -> B better
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(axis=1)
    return {
        "mean_delta": float(d.mean()),
        "median_boot": float(np.median(boots)),
        "ci95": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
        "p_b_better": float((boots > 0).mean()),
        "n_improved": int((d > 0).sum()),
        "n_worsened": int((d < 0).sum()),
    }


if __name__ == "__main__":
    # self-checks
    assert line_edits("By this publique Act", "By this publique Act") == (0, 0)
    assert line_edits("By this  publique", " By this publique ") == (0, 0)
    assert line_edits("a b c", "") == (3, 5)
    assert line_edits("Act", "act") == (1, 1)
    m = official_metric(["a b c"], [""])
    assert abs(m["score"] - (0.5 * (1 - 3 / 12) + 0.5 * (1 - 5 / 55))) < 1e-12
    print("road_metric self-checks OK")
