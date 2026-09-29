"""Tokenizer forensics: how each Qwen tokenizer segments EVERY training label."""
import collections
import json
import re

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ROOT = "D:/HANAFY/ROAD"
labs = [" ".join(t.split()) for t in pd.read_csv(f"{ROOT}/Train.csv").Target.astype(str)]
words = [w for t in labs for w in t.split()]
wc = collections.Counter(words)
rep = {}
toks = {}
for k in ["q25_7b", "q3_8b", "q3_32b"]:
    tk = AutoTokenizer.from_pretrained(f"{ROOT}/work/hf/{k}")
    toks[k] = tk
    enc = [tk.encode(t, add_special_tokens=False) for t in labs]
    n = np.array([len(e) for e in enc])
    ch = np.array([len(t) for t in labs])
    # per-word fertility in context (word preceded by a space, as it appears mid-line)
    fert = {w: len(tk.encode(" " + w, add_special_tokens=False)) for w in wc}
    tok_per_word = np.array([fert[w] for w in words])
    has_caret = np.array(["^" in w for w in words])
    has_colon = np.array([":" in w for w in words])
    has_amp = np.array(["&" in w for w in words])
    is_hapax = np.array([wc[w] == 1 for w in words])
    is_cap = np.array([w[:1].isupper() for w in words])
    r = {
        "vocab_size": len(tk),
        "tokens_per_label": {q: float(np.percentile(n, q)) for q in [0, 5, 50, 95, 99, 100]},
        "mean_tokens_per_label": float(n.mean()),
        "chars_per_token": float(ch.sum() / n.sum()),
        "tokens_per_word_mean": float(tok_per_word.mean()),
        "tokens_per_word_caret_words": float(tok_per_word[has_caret].mean()),
        "tokens_per_word_colon_words": float(tok_per_word[has_colon].mean()),
        "tokens_per_word_hapax": float(tok_per_word[is_hapax].mean()),
        "tokens_per_word_frequent(>=20)": float(tok_per_word[np.array([wc[w] >= 20 for w in words])].mean()),
        "tokens_per_word_capitalised": float(tok_per_word[is_cap].mean()),
        "single_token_word_frac": float((tok_per_word == 1).mean()),
        "words_ge4_tokens_frac": float((tok_per_word >= 4).mean()),
        "max_tokens_label": int(n.max()),
    }
    # examples of fragmentation
    ex = {}
    for w in ["W^m", "y^e", "M^r", "Adm^rs", "w^th", "w^ch", "Exec:^rs", "S:^d", "heires", "assignes", "publique",
              "ffebruary", "Xpian", "Pson", "sd", "&", "theis", "hould", "whome", "Barbados", "~~~~~~~", "aforesaid",
              "Slaues", "viz^t", "easmt^s", "pfitts", "comodities"]:
        ids = tk.encode(" " + w, add_special_tokens=False)
        ex[w] = [tk.decode([i]) for i in ids]
    r["examples"] = ex
    # context-dependence: same word, different token sequence at line start vs mid line
    diff = sum(1 for w in list(wc)[:2000] if tk.encode(w, add_special_tokens=False) != tk.encode(" " + w, add_special_tokens=False)[1:]
               and tk.encode(" " + w, add_special_tokens=False)[0] != tk.encode(" ", add_special_tokens=False)[0])
    r["top_words_highest_fertility"] = sorted(((w, fert[w], wc[w]) for w in wc if wc[w] >= 5), key=lambda x: -x[1])[:25]
    # which characters are never merged with neighbours (always their own token)?
    solo = {}
    for c in "^:&~_-.,;'\"()[]?+*/|#\\":
        occ = [t for t in labs if c in t][:400]
        if not occ:
            continue
        cnt = tot = 0
        for t in occ:
            for i in tk.encode(t, add_special_tokens=False):
                s = tk.decode([i])
                if c in s:
                    tot += 1
                    cnt += (s.strip() == c)
        solo[c] = round(cnt / max(tot, 1), 3)
    r["punct_standalone_token_rate"] = solo
    rep[k] = r
same = all(toks["q3_8b"].encode(t) == toks["q3_32b"].encode(t) for t in labs[:500])
same25 = np.mean([toks["q3_8b"].encode(t, add_special_tokens=False) == toks["q25_7b"].encode(t, add_special_tokens=False) for t in labs])
rep["q3_8b_vs_q3_32b_identical_first500"] = bool(same)
rep["q25_vs_q3_identical_label_frac"] = float(same25)
# chat template sanity: assistant boundary and special tokens (Qwen3 inserts <think>?)
for k in ["q25_7b", "q3_8b"]:
    tk = toks[k]
    msgs = [{"role": "user", "content": [{"type": "text", "text": "PROMPT"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "LABEL"}]}]
    try:
        rep[f"{k}_template_full"] = tk.apply_chat_template(msgs, tokenize=False)
        rep[f"{k}_template_prompt"] = tk.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True)
    except Exception as e:
        rep[f"{k}_template_err"] = str(e)[:200]
    rep[f"{k}_pad_eos"] = [tk.pad_token, tk.eos_token]
json.dump(rep, open(f"{ROOT}/work/tokenizer_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(rep, ensure_ascii=False, indent=1)[:9000])
