# Findings log (accumulated batch by batch)

Raw material for the prompt verdict. Each entry: batch, what was seen, evidence rows.

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
