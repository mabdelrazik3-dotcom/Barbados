"""Full label forensics over ALL 4,098 training transcriptions."""
import collections
import json
import re
import unicodedata

import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz
from rapidfuzz.distance import Levenshtein

ROOT = "D:/HANAFY/ROAD"
tr = pd.read_csv(f"{ROOT}/Train.csv")
raw = tr.Target.astype(str).tolist()
labs = [" ".join(t.split()) for t in raw]
rep = {}

# ---------- raw whitespace oddities ----------
rep["n_labels"] = len(raw)
rep["raw_leading_trailing_ws"] = int(sum(t != t.strip() for t in raw))
rep["raw_double_space"] = int(sum("  " in t for t in raw))
rep["raw_newline"] = int(sum("\n" in t or "\r" in t for t in raw))
rep["raw_tab"] = int(sum("\t" in t for t in raw))
rep["exact_duplicate_labels"] = int(pd.Series(labs).duplicated(keep=False).sum())
rep["unique_labels"] = int(pd.Series(labs).nunique())

# ---------- character inventory ----------
cc = collections.Counter("".join(labs))
total_chars = sum(cc.values())
inv = []
for ch, n in cc.most_common():
    inv.append({"char": ch, "repr": repr(ch), "count": n, "frac": n / total_chars,
                "unicode_name": unicodedata.name(ch, "?"), "cat": unicodedata.category(ch),
                "n_labels": sum(ch in t for t in labs)})
inv = pd.DataFrame(inv)
inv.to_csv(f"{ROOT}/work/char_inventory.csv", index=False)
rep["n_distinct_chars"] = len(cc)
rep["total_chars"] = total_chars
cats = collections.Counter()
for ch, n in cc.items():
    c = unicodedata.category(ch)
    k = ("upper" if c == "Lu" else "lower" if c == "Ll" else "digit" if c == "Nd" else
         "space" if ch == " " else "punct/symbol")
    cats[k] += n
rep["char_classes"] = {k: [v, round(v / total_chars, 4)] for k, v in cats.items()}
rep["non_ascii_chars"] = {ch: n for ch, n in cc.items() if ord(ch) > 127}
rep["rare_chars_lt20"] = {ch: n for ch, n in cc.items() if n < 20}
rep["examples_rare"] = {ch: [t for t in labs if ch in t][:3] for ch in rep["rare_chars_lt20"]}

# ---------- lengths ----------
nch = np.array([len(t) for t in labs]); nw = np.array([len(t.split()) for t in labs])
cpw = np.array([len(w) for t in labs for w in t.split()])
def desc(a):
    return {k: float(v) for k, v in zip(["min", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "max", "mean"],
                                         list(np.percentile(a, [0, 1, 5, 25, 50, 75, 95, 99, 100])) + [a.mean()])}
rep["chars_per_line"] = desc(nch); rep["words_per_line"] = desc(nw); rep["chars_per_word"] = desc(cpw)
rep["lines_le_2_words"] = int((nw <= 2).sum()); rep["lines_1_word"] = int((nw == 1).sum())

# ---------- words ----------
words = [w for t in labs for w in t.split()]
wc = collections.Counter(words)
rep["n_tokens"] = len(words); rep["n_types"] = len(wc)
rep["hapax_types"] = int(sum(1 for v in wc.values() if v == 1))
rep["hapax_token_frac"] = sum(v for v in wc.values() if v == 1) / len(words)
rep["top100_words"] = wc.most_common(100)
lower_c = collections.Counter(w.lower() for w in words)
# casing ambiguity: lowercase types appearing in >1 casing
forms = collections.defaultdict(collections.Counter)
for w in words:
    forms[w.lower()][w] += 1
amb = {k: dict(v) for k, v in forms.items() if len(v) > 1 and sum(v.values()) >= 10}
rep["n_types_multi_casing_ge10"] = len(amb)
rep["casing_examples"] = dict(sorted(amb.items(), key=lambda kv: -sum(kv[1].values()))[:40])
cap_words = sum(1 for w in words if w[:1].isupper())
rep["capitalised_word_frac"] = cap_words / len(words)
rep["line_initial_cap_frac"] = float(np.mean([t[:1].isupper() for t in labs]))
# punctuation attached to words
punct_tok = collections.Counter(re.sub(r"[A-Za-z0-9]", "", w) for w in words)
rep["word_punct_patterns"] = punct_tok.most_common(40)

# ---------- markup / abbreviations ----------
mark = {}
for pat, name in [(r"\^", "caret"), (r":", "colon"), (r"&", "ampersand"), (r"\?", "question"),
                  (r"\[", "lbracket"), (r"\]", "rbracket"), (r"\(", "lparen"), (r"-", "hyphen"),
                  (r"--", "double_hyphen"), (r"'", "apostrophe"), (r"/", "slash"), (r"\.", "period"),
                  (r",", "comma"), (r";", "semicolon"), (r'"', "dquote"), (r"\d", "digit"),
                  (r"\bff", "ff_initial"), (r"\bye\b", "ye"), (r"\byt\b", "yt"), (r"\bwch\b", "wch")]:
    m = [t for t in labs if re.search(pat, t)]
    mark[name] = {"lines": len(m), "occ": int(sum(len(re.findall(pat, t)) for t in labs)), "ex": m[:3]}
rep["markup"] = mark
caret_words = collections.Counter(w for w in words if "^" in w)
rep["caret_words_top"] = caret_words.most_common(40)
colon_words = collections.Counter(w for w in words if ":" in w)
rep["colon_words_top"] = colon_words.most_common(30)
rep["ff_words_top"] = collections.Counter(w for w in words if w.lower().startswith("ff")).most_common(20)

# ---------- n-grams ----------
def ngrams(seq, n):
    return zip(*[seq[i:] for i in range(n)])
bi = collections.Counter(g for t in labs for g in ngrams(t.split(), 2))
tri = collections.Counter(g for t in labs for g in ngrams(t.split(), 3))
q4 = collections.Counter(g for t in labs for g in ngrams(t.split(), 4))
rep["top_bigrams"] = [(" ".join(k), v) for k, v in bi.most_common(40)]
rep["top_trigrams"] = [(" ".join(k), v) for k, v in tri.most_common(40)]
rep["top_4grams"] = [(" ".join(k), v) for k, v in q4.most_common(40)]
cbi = collections.Counter(t[i:i + 2] for t in labs for i in range(len(t) - 1))
rep["top_char_bigrams"] = cbi.most_common(40)
# how much of each line is covered by 4-grams seen >=3 times elsewhere (template-ness)
tmpl = []
for t in labs:
    ws = t.split()
    cov = np.zeros(len(ws), bool)
    for i, g in enumerate(ngrams(ws, 4)):
        if q4[g] >= 3:
            cov[i:i + 4] = True
    tmpl.append(cov.mean() if len(ws) else 0)
tmpl = np.array(tmpl)
rep["template_4gram_coverage"] = desc(tmpl)
rep["lines_template_cov_ge_0.5"] = int((tmpl >= 0.5).sum())

# ---------- spelling variants: lowercase types within edit distance 1 of a more frequent type ----------
types = [w for w, n in lower_c.most_common() if len(w) >= 4 and w.isalpha()]
freq = lower_c
variants = []
top = types[:1500]
for w in top:
    cands = process.extract(w, top, scorer=Levenshtein.distance, limit=6, score_cutoff=1)
    for c, d, _ in cands:
        if c != w and d == 1 and freq[c] < freq[w]:
            variants.append((w, freq[w], c, freq[c]))
rep["n_spelling_variant_pairs_ed1"] = len(variants)
rep["spelling_variant_examples"] = variants[:60]

# ---------- near-duplicate labels ----------
M = process.cdist(labs, labs, scorer=fuzz.ratio, workers=-1, dtype=np.uint8)
np.fill_diagonal(M, 0)
best = M.max(axis=1)
rep["nn_label_ratio"] = desc(best.astype(float))
for thr in [100, 95, 90, 80, 70]:
    rep[f"labels_with_nn_ratio_ge_{thr}"] = int((best >= thr).sum())
pairs = np.argwhere(np.triu(M >= 90))
rep["n_label_pairs_ge90"] = int(len(pairs))
ids = tr.ID.tolist()
rep["near_dup_label_examples"] = [(ids[i], labs[i], ids[j], labs[j], int(M[i, j])) for i, j in pairs[:40]]
np.save(f"{ROOT}/work/label_nn_ratio.npy", best)
# save compact nn table
nn = M.argmax(axis=1)
pd.DataFrame({"ID": ids, "label": labs, "nn_ID": [ids[j] for j in nn], "nn_label": [labs[j] for j in nn],
              "nn_ratio": best, "n_chars": nch, "n_words": nw, "template_cov": tmpl}).to_csv(
    f"{ROOT}/work/label_table.csv", index=False)
# connected groups at ratio>=90 (candidate duplicate lines / re-crops)
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
A = sp.csr_matrix(M >= 90)
ncomp, comp = connected_components(A, directed=False)
sizes = collections.Counter(comp)
rep["near_dup_groups_ge90"] = {"n_groups_size_ge2": int(sum(1 for v in sizes.values() if v >= 2)),
                               "rows_in_groups": int(sum(v for v in sizes.values() if v >= 2)),
                               "largest": sorted(sizes.values(), reverse=True)[:10]}
np.save(f"{ROOT}/work/label_group90.npy", comp)

json.dump(rep, open(f"{ROOT}/work/label_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
for k, v in rep.items():
    s = json.dumps(v, ensure_ascii=False, default=str)
    print(f"{k}: {s[:600]}")
