# Findings log (accumulated batch by batch)

Raw material for the prompt verdict. Each entry: batch, what was seen, evidence rows.

## How these findings are used (the prompts were built from the Train.csv labels)

The prompts' rules, statistics and worked examples were derived from the Train.csv
labels, so every label error is inherited by the prompts. Each label→ink difference is
therefore classified as one of two kinds, and each kind changes the prompt differently:

1. **Random label errors** (repeated words, spellings copied from a sibling copy,
   misread letters, a completed cut word here but not there). They cannot be predicted,
   so the prompt follows the ink. Worked examples must come only from rows where
   label = ink (verdict MATCH); otherwise the prompt pairs an image with text it does
   not show (e.g. `Exec:^rs Adm:^rs` for ink `Exor: & Adm:`).
2. **Systematic annotator habits** (for a given ink feature the annotators almost always
   write the same thing, even when it differs from the ink). The test references come
   from the same annotators, so the prompt should reproduce the habit. Text-only
   counting cannot see these; only label-vs-image comparison can. Candidates so far:
   linked words split (`inthe` → `in the`), u modernised to v (`Nouember` → `November`),
   separate dots after words dropped, clear capitals written small. Each needs its
   rate over the full set (P(label form | ink form)) before it becomes a prompt rule.

QA: 12 of 12 agent corrections re-checked by me at high zoom hold (rows 36, 44, 47, 49,
58, 63, 69, 80, 86, 112, 116, 120, 136, 140, 147 — incl. `Oluant`, `aforesd`, `the`).

## Batch 1 — pilot (rows 0–33, verdicted by hand)

- **Parallel copies of one formula.** Consecutive rows are the same text written by
  different scribes (4–8 copies: "By this public act…", "of February in the year…",
  "Signed/Sealed and delivered in the prsence of…", "This done and protested…").
  Several are 19th-century copperplate copies of the 17th-century books (a third
  hand type the prompts do not describe).
- **Labels copied from a sibling copy instead of read from the image.** row 20
  `writeing` on an image that says `writing`; rows 23/26 `presence` on images that say
  `prsence` (while rows 22/24/25/29 correctly say `prsence`).
- **Duplicate crops of the same physical line with different labels.** rows 18/19
  (same ink, `done/said` vs `Done/Said`). Annotator disagreement on capitals is real
  noise inside the references.
- **Capitals are the most frequent disagreement.** `Sixty` (tall looped S vs `six`),
  `One`, `Sealed`, `Done`, `Said`. The annotators are inconsistent across copies of
  the same formula (`year`/`Year`, `In`/`in`).
- **Cut last word completed by the label.** row 0: crop ends at `prote`, label
  `protest`. The prompts tell the model never to complete a cut word — but the
  references sometimes do.
- **Family B raised letters not marked in the label.** rows 9/10: `dat.^e th.^e … On.^e`
  written plain `date the … One`. Contradicts the prompts' claim that family-B
  transcriptions mark raised letters (they often do, not always).
- **Line-end/inside periods and commas omitted by labels.** rows 15/17 `Year.`,
  row 10 `July,`.
- **Word joins in cursive** (`inthe`, `presenceofus`, `deliveredin`) are written
  split in the labels almost always.
- **Not text:** braces `}` closing attestation lines, red margin rules.

## Batch 1 — agent (rows 30–49): 8 MATCH, 9 CORRECTED, 2 UNCERTAIN

- Spelling normalised by the labels: `writinge`→label `writing`, `farre`→`fare`,
  `singuler`→`singular`, `guitt`→`guift`; cut word completed (`mon`→`mony`).
- Capitals missed by the labels: `Christian People`, `Appearer`, `Second`, `Grace`,
  `To`; and the reverse: label `Full` where the ink has a single small f, label `FFive`
  where the scribe's capital F is the doubled ff (encoded `ffive`).
- Separate dots dropped by labels: `To.`, `Carlyle. whereof. .all` (verified by me).
- Duplicate crops: rows 37/38 and 39/40 (same red margin rule, same ink) with
  different labels (`Rochell`/`Rochill`).
- Family B hands: n/u bare minims, closed/∂-shaped e, short hooked capital S
  (`Said`, `Sume`, `Singuler`), capital F written ff, p shaped like y, end dash.
- Family A: e can be a bare minim and i-dots are missing or displaced (e/i confusion),
  u/v hard to tell.
- QA: I re-checked rows 36, 44, 47, 49 at high zoom — all four agent corrections hold.

## Units 58–74 and 100–124 (parallel agents)

- 58–74: 6 MATCH, 10 CORRECTED, 1 UNCERTAIN. 100–124: 4 MATCH, 15 CORRECTED, 6 UNCERTAIN.
- Labels replace the scribe's letters with the formula/standard word: `whame`→`whome`,
  `unte`→`unto`, `Margan`→`Morgan`, `assynes`→`assignes`, `pound`→`pounds`,
  `Nouember`→`November`, `recouering`→`recovering`, `possesse`→`possess`,
  `Witnes`→`Witness`, `Willett`→`Willet`, `alwayes`→`always`.
- Labels add words the ink lacks (`of the Said` where the ink has `of Said`).
- Labels flatten raised letters and drop their marks (`w:^ch`, `W:^m` written `wch`, `Wm`)
  and miss insertions (`^soe`) and strikes (`~their~`).
- Dots after abbreviations dropped (`viz^t.`, `Invacon.`, `To.`); a wavy end `~` written
  as `-`; free-standing dashes attached (`-Signed by-`).
- **Two more of the prompts' own examples are contradicted by the ink:**
  `Supra -Signed by-` (v1/v2 §4.4) — the ink is `ut supra - Signed by -` (small u/s,
  free dashes; row 112); `Mingo-` (v1/v2 §4.4 "dash touching the word") — the ink is
  `Mingo, -` with a comma and a separate dash (row 120).
- QA: I re-checked rows 58, 63, 69 (`Margan`, `whame`, `unte`) and 112, 116, 120 at high
  zoom — all six agent corrections hold.

## Unit 75–99: 11 MATCH, 9 CORRECTED, 5 UNCERTAIN

- **Worked example v1 #12 / v2 #13 is contradicted by its image (row 80).** Prompt:
  `Exec:^rs Adm:^rs and assignes, That if the Said Richard` ("raised letters with a colon
  below"). Ink (verified by me at high zoom): `Exor: & Adm: and assignes, That if th^e
  Said Richard` — no raised letters in the two abbreviations, `Exor` not `Exec`, an
  et-sign between them, and the raised e is on `th^e`. The example teaches the `:^rs`
  pattern from an image that does not show it.
- Label repeats a phrase the ink has once: `of the some of the some of` (row 86, verified).
- u→v modernised by labels: `Couenant`, `Prouided`, `neuertheles`.
- Raised letters both ways: labels miss `th^e` (row 90) and invent `o^r` (row 96).
- Capitals missed: `Shal`, `Expressed`.
- Some family-B scribes raise many `e`s inside full words — handled as a hand habit (flat);
  protocol rule 11 added.
