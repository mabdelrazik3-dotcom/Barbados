"""Evaluation / ensembling / error-forensics toolkit shared by barbados-2-enhanced.ipynb and offline analysis.

Everything here optimises the OFFICIAL leaderboard loss per line:
    cost(ref, hyp) = 0.5 * word_edits / 12 + 0.5 * char_edits / 55
(score = 1 - mean(cost)). One word edit = 4.58 char edits.
"""
from __future__ import annotations

import collections
import math

import numpy as np
from rapidfuzz.distance import Levenshtein

WORD_XMAX, CHAR_XMAX = 12.0, 55.0


def norm(s) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    return " ".join(str(s).split())


def edits(ref, hyp):
    r, h = norm(ref), norm(hyp)
    return Levenshtein.distance(r.split(), h.split()), Levenshtein.distance(r, h)


def cost(ref, hyp):
    w, c = edits(ref, hyp)
    return 0.5 * w / WORD_XMAX + 0.5 * c / CHAR_XMAX


def official(refs, hyps):
    we = np.array([edits(r, h)[0] for r, h in zip(refs, hyps)], float)
    ce = np.array([edits(r, h)[1] for r, h in zip(refs, hyps)], float)
    return {"score": 0.5 * (1 - we.mean() / WORD_XMAX) + 0.5 * (1 - ce.mean() / CHAR_XMAX),
            "word_edits_per_line": we.mean(), "char_edits_per_line": ce.mean(),
            "wer_corpus": we.sum() / max(1, sum(len(norm(r).split()) for r in refs)),
            "cer_corpus": ce.sum() / max(1, sum(len(norm(r)) for r in refs)),
            "n": len(refs), "empty": int(sum(not norm(h) for h in hyps)), "we": we, "ce": ce}


# ------------------------------------------------------------------ MBR
def mbr_select(cands, weights=None, return_risk=False):
    """Minimum-Bayes-risk choice under the official per-line cost.
    cands: list[str] (pooled hypotheses, duplicates allowed); weights: posterior mass per candidate.
    return_risk=True also returns the chosen candidate's expected cost (low = the pool agrees)."""
    cands = [norm(c) for c in cands]
    if weights is None:
        weights = [1.0] * len(cands)
    agg = collections.OrderedDict()
    for c, w in zip(cands, weights):
        agg[c] = agg.get(c, 0.0) + float(w)
    uniq = list(agg)
    if len(uniq) == 1:
        return (uniq[0], 0.0) if return_risk else uniq[0]
    wv = np.array([agg[u] for u in uniq])
    wv = wv / wv.sum()
    risks = []
    for h in uniq:
        risks.append(sum(w * cost(c, h) for c, w in zip(uniq, wv)))
    k = int(np.argmin(risks))
    return (uniq[k], float(risks[k])) if return_risk else uniq[k]


def nbest_posteriors(scores, temperature=1.0):
    """scores: length-normalised log-probs (HF sequences_scores) of one system's N-best -> softmax mass."""
    s = np.asarray(scores, float) / max(temperature, 1e-6)
    s = s - s.max()
    p = np.exp(s)
    return p / p.sum()


# ------------------------------------------------------------------ ROVER (word level)
def _align_words(cons, hyp):
    """Levenshtein alignment of word lists; returns list of (i_cons|None, word|None)."""
    n, m = len(cons), len(hyp)
    D = np.zeros((n + 1, m + 1), np.int32)
    D[:, 0] = np.arange(n + 1); D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j - 1] + (cons[i - 1] != hyp[j - 1]), D[i - 1, j] + 1, D[i, j - 1] + 1)
    i, j, out = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (cons[i - 1] != hyp[j - 1]):
            out.append((i - 1, hyp[j - 1])); i -= 1; j -= 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            out.append((i - 1, None)); i -= 1
        else:
            out.append((None, hyp[j - 1])); j -= 1
    return out[::-1]


def word_rover(hyps, weights=None):
    """Word-level ROVER: progressively align every hypothesis to a growing confusion network, then vote
    per slot (an empty vote = deletion). Word-level voting matches the metric's 4.58x word premium."""
    hyps = [norm(h).split() for h in hyps]
    if weights is None:
        weights = [1.0] * len(hyps)
    order = np.argsort(-np.asarray(weights))            # seed with the most trusted system
    hyps = [hyps[k] for k in order]; weights = [weights[k] for k in order]
    slots = [{0: w} for w in hyps[0]]                   # slot: {system: word}
    for s in range(1, len(hyps)):
        cons = [collections.Counter(sl.values()).most_common(1)[0][0] if sl else "" for sl in slots]
        al = _align_words(cons, hyps[s])
        new, last = [], -1
        for ci, w in al:
            if ci is None:
                new.append({s: w})
            else:
                sl = dict(slots[ci])
                if w is not None:
                    sl[s] = w
                new.append(sl)
        slots = new
    out = []
    for sl in slots:
        agg = collections.Counter()
        for s, w in sl.items():
            agg[w] += weights[s]
        agg[None] += sum(weights[s] for s in range(len(hyps)) if s not in sl)
        best = max(agg.items(), key=lambda kv: kv[1])[0]
        if best is not None:
            out.append(best)
    return " ".join(out)


# ------------------------------------------------------------------ error forensics
def _cls_word_err(r, h):
    if r.lower() == h.lower():
        return "casing"
    strip = lambda x: "".join(ch for ch in x if ch.isalnum())
    if strip(r) == strip(h):
        return "punct/markup"
    if strip(r).lower() == strip(h).lower():
        return "casing+punct"
    d = Levenshtein.distance(r, h)
    return "1-char" if d == 1 else "2-char" if d == 2 else "3+char"


def forensics(refs, hyps, top=25):
    """Aligns every line; returns dict of global error tables."""
    ops = collections.Counter()
    char_sub = collections.Counter(); char_del = collections.Counter(); char_ins = collections.Counter()
    word_sub = collections.Counter(); word_del = collections.Counter(); word_ins = collections.Counter()
    wclass = collections.Counter()
    space_err = 0
    per_line = []
    for r, h in zip(refs, hyps):
        r, h = norm(r), norm(h)
        rw, hw = r.split(), h.split()
        for op in Levenshtein.editops(rw, hw):
            ops["word_" + op.tag] += 1
            if op.tag == "replace":
                word_sub[(rw[op.src_pos], hw[op.dest_pos])] += 1
                wclass[_cls_word_err(rw[op.src_pos], hw[op.dest_pos])] += 1
            elif op.tag == "delete":
                word_del[rw[op.src_pos]] += 1
            else:
                word_ins[hw[op.dest_pos]] += 1
        for op in Levenshtein.editops(r, h):
            ops["char_" + op.tag] += 1
            if op.tag == "replace":
                char_sub[(r[op.src_pos], h[op.dest_pos])] += 1
                space_err += (r[op.src_pos] == " ") + (h[op.dest_pos] == " ")
            elif op.tag == "delete":
                char_del[r[op.src_pos]] += 1; space_err += r[op.src_pos] == " "
            else:
                char_ins[h[op.dest_pos]] += 1; space_err += h[op.dest_pos] == " "
        w, c = Levenshtein.distance(rw, hw), Levenshtein.distance(r, h)
        per_line.append((w, c, len(h) - len(r)))
    pl = np.array(per_line, float) if per_line else np.zeros((0, 3))
    tot_c = max(1, sum(v for k, v in ops.items() if k.startswith("char_")))
    return {
        "ops": dict(ops),
        "char_sub_top": char_sub.most_common(top), "char_del_top": char_del.most_common(top),
        "char_ins_top": char_ins.most_common(top),
        "word_sub_top": [(f"{a} -> {b}", n) for (a, b), n in word_sub.most_common(top)],
        "word_del_top": word_del.most_common(top), "word_ins_top": word_ins.most_common(top),
        "word_sub_classes": dict(wclass),
        "space_char_edit_frac": space_err / tot_c,
        "len_diff_mean": float(pl[:, 2].mean()) if len(pl) else 0.0,
        "lines_perfect": int((pl[:, 0] == 0).sum()) if len(pl) else 0,
        "loss_share_top5pct_lines": float(np.sort(0.5 * pl[:, 0] / 12 + 0.5 * pl[:, 1] / 55)[::-1][: max(1, len(pl) // 20)].sum()
                                          / max(1e-9, (0.5 * pl[:, 0] / 12 + 0.5 * pl[:, 1] / 55).sum())) if len(pl) else 0.0,
    }


def by_group(refs, hyps, groups):
    """Official score by group label (e.g. scan batch, length bucket, visual cluster)."""
    out = {}
    g = np.asarray(groups)
    for k in sorted(set(g.tolist()), key=str):
        idx = np.where(g == k)[0]
        m = official([refs[i] for i in idx], [hyps[i] for i in idx])
        out[str(k)] = {"n": len(idx), "score": round(m["score"], 4),
                       "w_edits": round(m["word_edits_per_line"], 3), "c_edits": round(m["char_edits_per_line"], 3)}
    return out


def paired_delta(refs, hyp_a, hyp_b, n_boot=10000, seed=0):
    """B vs A: per-line loss difference, paired bootstrap over whole lines."""
    la = np.array([cost(r, h) for r, h in zip(refs, hyp_a)])
    lb = np.array([cost(r, h) for r, h in zip(refs, hyp_b)])
    d = la - lb
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    wa = np.array([edits(r, h)[0] for r, h in zip(refs, hyp_a)]); wb = np.array([edits(r, h)[0] for r, h in zip(refs, hyp_b)])
    ca = np.array([edits(r, h)[1] for r, h in zip(refs, hyp_a)]); cb = np.array([edits(r, h)[1] for r, h in zip(refs, hyp_b)])
    return {"delta_score": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "p_b_better": float((boots > 0).mean()), "improved": int((d > 1e-12).sum()), "worsened": int((d < -1e-12).sum()),
            "net_word_edits_saved": int(wa.sum() - wb.sum()), "net_char_edits_saved": int(ca.sum() - cb.sum())}


if __name__ == "__main__":
    # --- self tests on synthetic data ---
    assert mbr_select(["a b c", "a b c", "a x c"]) == "a b c"
    assert word_rover(["the said John", "the sd John", "the said Jon"]) == "the said John"
    assert word_rover(["a b c d", "a c d", "a b c d"]) == "a b c d"
    refs = ["By this publique Act", "his heires & assignes"]
    hyps = ["By this public Act", "his Heires & assignes"]
    f = forensics(refs, hyps)
    assert f["word_sub_classes"] == {"3+char": 1, "casing": 1}, f["word_sub_classes"]
    m = official(refs, hyps)
    assert abs(m["score"] - (1 - np.mean([cost(r, h) for r, h in zip(refs, hyps)]))) < 1e-12
    print("road_eval self-tests OK")
