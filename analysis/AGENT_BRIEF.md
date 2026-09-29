# Brief for one unit agent (filled in per run: FIRST, LAST; up to 4 run in parallel)

You are verifying labels of historic handwriting lines (Barbados deed books, 1639–1710)
for the `opus_label` study. Work only in `/home/user/Barbados`.

1. Read `analysis/OPUS_PROTOCOL.md` completely before starting. It is binding.
2. Prepare your unit (renders the reading views and prints the rows):
   `cd /home/user/Barbados && python3 analysis/tools/prep_batch.py --rows {FIRST} {LAST}`
   Your unit name is `u{FIRST:04d}_{LAST:04d}` (e.g. u0075_0099).
   Views: `{VIEWS}/<ID>.png`. Zoom: `python3 analysis/tools/zoom.py {ZOOM} <ID> X0 X1 [Y0 Y1]`.
3. Your rows: {FIRST}–{LAST} (skip any already recorded). Other agents work on other
   units at the same time — touch only your own unit. Work in groups of 4:
   a. `Read` the 4 views in ONE message (four Read calls side by side).
   b. For each row compare label and ink word by word; a row where every word, capital
      and mark agrees is MATCH without zooming.
   c. Only where you suspect a difference or cannot read the ink: make ALL zooms of the
      group in ONE Bash call (chain the zoom.py commands with `&&`), then `Read` them all
      in ONE message. Do not zoom to re-confirm what is already clear.
   d. Record the 4 rows in ONE call:
      `python3 analysis/tools/record.py <unit name> <<'EOF'` ... `EOF`
      (one line per row: `ID || VERDICT || CONF || opus_label or = || notes`) and check
      the diff it prints; re-record a row if the diff shows an edit you did not intend.
   Aim for about 3 tool rounds per group of 4.
4. Rules: the image is the only authority; follow the protocol's encoding and its
   section 6 calibration. Never modernise, never correct the scribe's spelling, never
   complete cut words, never expand abbreviations. Do not read any other files, do not
   edit files by hand, do not commit, do not start other agents.
5. Finish with a report of at most 250 words:
   - counts of MATCH / CORRECTED / UNCERTAIN;
   - every CORRECTED/UNCERTAIN row as `row: label-word(s) -> opus-word(s) (reason)`,
     one short line each;
   - at most 5 bullets of patterns that matter for a transcription prompt.
