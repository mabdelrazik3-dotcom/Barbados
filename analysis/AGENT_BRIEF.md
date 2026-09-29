# Brief for one batch agent (filled in per batch)

You are verifying labels of historic handwriting lines (Barbados deed books, 1639–1710)
for the `opus_label` study. Work only in `/home/user/Barbados`.

1. Read `analysis/OPUS_PROTOCOL.md` completely before starting. It is binding.
2. Prepare the batch (renders the reading views and prints the rows):
   `cd /home/user/Barbados && python3 analysis/tools/prep_batch.py {BATCH} 50`
   Views are PNGs in `{VIEWS}/<ID>.png`. Zoom with
   `python3 analysis/tools/zoom.py {ZOOM} <ID> X0 X1 [Y0 Y1]`.
3. Process every row of batch {BATCH} that is not yet recorded, in row order
   ({PENDING}). For each: `Read` its view, compare the label with the ink word by word,
   zoom where in doubt, decide. Record every 2–5 rows with
   `python3 analysis/tools/record.py {BATCH} <<'EOF' ... EOF` and check the diff it
   prints. The recorder also prints which rows are still pending — continue until the
   batch shows 50/50 (or all rows of a short final batch).
4. Rules: the image is the only authority; follow the protocol's encoding and its
   section 6 calibration. Do not read any other files (no reports, notebooks, work/,
   other batches' results), do not edit any file by hand, do not commit, do not start
   other agents.
5. Finish with a report of at most 350 words:
   - counts of MATCH / CORRECTED / UNCERTAIN;
   - every CORRECTED row: `row ID: label -> opus (reason)` in one line each;
   - patterns: what kinds of label errors and scribal conventions you saw that matter
     for a transcription prompt (capitals, raised letters, cut words, parallel or
     duplicate copies, joins, marks, abbreviations, neighbour lines, hands/families).
