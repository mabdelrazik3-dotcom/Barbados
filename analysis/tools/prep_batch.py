"""Prepare one batch of Train.csv rows for the opus_label study.

Usage:
    python3 prep_batch.py BATCH_NO [BATCH_SIZE]
Batch n (1-based) holds Train.csv rows (n-1)*size .. n*size-1, in file order.
Renders a reading view for every row, writes analysis/batches/batch_NNN.json
and prints a table: row | ID | WxH | family | label.
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from render_view import render  # noqa: E402

VIEW_DIR = os.environ.get("VIEW_DIR", "/tmp/claude-0/-home-user-Barbados/319cb918-e414-5df8-a8f2-bce2e065c168/scratchpad/views")


def main():
    n = int(sys.argv[1])
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    rows = list(csv.DictReader(open(os.path.join(ROOT, "Train.csv"), encoding="utf-8")))
    part = rows[(n - 1) * size: n * size]
    items = []
    for k, r in enumerate(part):
        idx = (n - 1) * size + k
        path, w, h, segs = render(r["ID"], VIEW_DIR)
        fam = "B" if (w > 2000 or h > 180) else "A"
        items.append({"row": idx, "ID": r["ID"], "Target": r["Target"], "w": w, "h": h,
                      "family": fam, "view": path})
        print(f"{idx:4d} | {r['ID']} | {w}x{h} | {fam} | {r['Target']}")
    os.makedirs(os.path.join(ROOT, "analysis", "batches"), exist_ok=True)
    out = os.path.join(ROOT, "analysis", "batches", f"batch_{n:03d}.json")
    json.dump(items, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"# {len(items)} rows -> {out}; views in {VIEW_DIR}")


if __name__ == "__main__":
    main()
