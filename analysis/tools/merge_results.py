"""Merge all batch results into one table and classify every label -> ink difference.

Usage:
    python3 merge_results.py
Writes:
    analysis/Train_opus.csv        ID, Target, opus_label, verdict, confidence, family,
                                   diff_types, notes  (Train.csv order; rows without a
                                   verdict yet have an empty opus_label)
    analysis/diff_summary.json     counts per verdict, per difference type, per family
Prints a short summary.
"""
import csv
import difflib
import glob
import json
import os
import re
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PUNCT = re.compile(r"[\^:;.,\-~_'\"()]")


def classify_pair(a, b, at_end):
    """Classify one replaced word a (label) -> b (ink)."""
    if a.lower() == b.lower():
        return "capital"
    if PUNCT.sub("", a) == PUNCT.sub("", b):
        if "^" in a or "^" in b:
            return "raised_mark"
        return "punct_mark"
    if at_end and PUNCT.sub("", a).startswith(PUNCT.sub("", b)) and len(b) < len(a):
        return "cut_word"
    if PUNCT.sub("", a).lower() == PUNCT.sub("", b).lower():
        return "capital+mark"
    if "^" in a or "^" in b:
        return "raised_mark+spelling"
    return "spelling"


def classify(label, opus):
    lw, ow = label.split(), opus.split()
    types = []
    ops = difflib.SequenceMatcher(None, lw, ow, autojunk=False).get_opcodes()
    for op, i1, i2, j1, j2 in ops:
        if op == "equal":
            continue
        a, b = lw[i1:i2], ow[j1:j2]
        at_end = i2 == len(lw)
        if op == "delete":
            types += ["mark_extra" if all(PUNCT.sub("", w) == "" for w in a) else "word_extra"] * len(a)
        elif op == "insert":
            types += ["mark_missing" if all(PUNCT.sub("", w) == "" for w in b) else "word_missing"] * len(b)
        elif "".join(a) == "".join(b):
            types.append("join_split")
        elif len(a) == len(b):
            for k, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    types.append(classify_pair(x, y, at_end and k == len(a) - 1))
        else:
            types.append("multiword")
    return types


def main():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "Train.csv"), encoding="utf-8")))
    recs = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "analysis", "opus_results", "*.jsonl"))):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            recs[r["ID"]] = r  # later record wins
    out_rows, verdicts, dtypes, fam_v = [], Counter(), Counter(), defaultdict(Counter)
    for r in rows:
        rec = recs.get(r["ID"])
        if not rec:
            out_rows.append({"ID": r["ID"], "Target": r["Target"], "opus_label": "", "verdict": "",
                             "confidence": "", "family": "", "diff_types": "", "notes": ""})
            continue
        norm = " ".join(r["Target"].split())
        types = classify(norm, rec["opus_label"]) if rec["opus_label"] != norm else []
        verdicts[rec["verdict"]] += 1
        fam_v[rec["family"]][rec["verdict"]] += 1
        dtypes.update(types)
        out_rows.append({"ID": r["ID"], "Target": r["Target"], "opus_label": rec["opus_label"],
                         "verdict": rec["verdict"], "confidence": rec["confidence"],
                         "family": rec["family"], "diff_types": ";".join(sorted(set(types))),
                         "notes": rec["notes"]})
    with open(os.path.join(ROOT, "analysis", "Train_opus.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    summary = {"rows_total": len(rows), "rows_verdicted": sum(verdicts.values()),
               "verdicts": dict(verdicts), "diff_types": dict(dtypes.most_common()),
               "by_family": {f: dict(c) for f, c in fam_v.items()}}
    json.dump(summary, open(os.path.join(ROOT, "analysis", "diff_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
