"""Record opus_label verdicts for one batch.

Usage (read lines from stdin, quoted heredoc so nothing is expanded):
    python3 record.py BATCH_NO_OR_UNIT <<'EOF'
    ID || VERDICT || CONF || opus_label || notes
    EOF
BATCH_NO_OR_UNIT is a batch number (batch_NNN) or a unit name such as u0058_0074.

VERDICT  MATCH      the label is exactly what the ink shows (opus_label must be "=")
         CORRECTED  the label is wrong somewhere; opus_label is the full corrected line
         UNCERTAIN  the ink cannot be fully verified; opus_label is the best reading
                    ("=" allowed when the best reading is the label itself)
CONF     high | medium | low   (confidence in opus_label as a whole)
opus_label "=" means: identical to the Train.csv label.

Each accepted row is appended to analysis/opus_results/<batch_NNN|unit>.jsonl
(a later record for the same ID replaces an earlier one when merging).
Prints a word diff (label -> opus) for every changed row, then the batch progress.
"""
import difflib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ALLOWED = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ^:;.,-~_&'\"()")
VERDICTS = {"MATCH", "CORRECTED", "UNCERTAIN"}
CONFS = {"high", "medium", "low"}


def word_diff(a, b):
    aw, bw = a.split(), b.split()
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, aw, bw, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        out.append(f"[{' '.join(aw[i1:i2])}] -> [{' '.join(bw[j1:j2])}]")
    return "; ".join(out)


def main():
    arg = sys.argv[1]
    name = arg if arg.startswith("u") else f"batch_{int(arg):03d}"
    batch = json.load(open(os.path.join(ROOT, "analysis", "batches", f"{name}.json"), encoding="utf-8"))
    by_id = {it["ID"]: it for it in batch}
    out_dir = os.path.join(ROOT, "analysis", "opus_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.jsonl")
    ok, bad = 0, 0
    with open(out_path, "a", encoding="utf-8") as fh:
        for raw in sys.stdin:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split("||", 4)]
            if len(parts) < 4:
                print(f"REJECTED (need 'ID || VERDICT || CONF || opus_label || notes'): {line}")
                bad += 1
                continue
            while len(parts) < 5:
                parts.append("")
            iid, verdict, conf, opus, notes = parts
            err = []
            if iid not in by_id:
                err.append("ID not in this batch")
            if verdict not in VERDICTS:
                err.append(f"verdict must be one of {sorted(VERDICTS)}")
            if conf not in CONFS:
                err.append(f"confidence must be one of {sorted(CONFS)}")
            if err:
                print(f"REJECTED {iid}: {'; '.join(err)}")
                bad += 1
                continue
            target = by_id[iid]["Target"]
            norm_target = " ".join(target.split())  # labels may carry stray spaces
            if opus == "=":
                opus = norm_target
            if re.search(r"\s{2,}", opus) or opus != opus.strip() or not opus:
                err.append("opus_label must be one non-empty line with single spaces")
            odd = sorted(set(opus) - ALLOWED)
            if odd:
                err.append(f"characters outside the convention set: {odd}")
            if verdict == "MATCH" and opus != norm_target:
                err.append("MATCH but opus_label differs from the label (use CORRECTED, or '=')")
            if verdict == "CORRECTED" and opus == norm_target:
                err.append("CORRECTED but opus_label equals the label")
            if err:
                print(f"REJECTED {iid}: {'; '.join(err)}")
                bad += 1
                continue
            rec = {"row": by_id[iid]["row"], "ID": iid, "Target": target, "opus_label": opus,
                   "verdict": verdict, "confidence": conf, "notes": notes,
                   "family": by_id[iid]["family"]}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ok += 1
            if opus != norm_target:
                print(f"{iid} {verdict}/{conf}: {word_diff(norm_target, opus)}")
    done = set()
    for l in open(out_path, encoding="utf-8"):
        done.add(json.loads(l)["ID"])
    missing = [it for it in batch if it["ID"] not in done]
    print(f"# recorded {ok}, rejected {bad}; {name}: {len(done)}/{len(batch)} done")
    if missing:
        print("# next pending rows:", ", ".join(f"{it['row']}:{it['ID']}" for it in missing[:8]))


if __name__ == "__main__":
    main()
