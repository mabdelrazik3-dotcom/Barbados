# opus_label study — protocol

Goal: for every row of `Train.csv`, look at its image (`images/<ID>.jpg`; the copy in
`images/images/` is a duplicate and is ignored), give a verdict on the label (`Target`),
and write `opus_label`: the exact text of the target line **as the ink shows it**.

The records are 17th-century (1639–1710) Barbados deeds. **The image is the only
authority.** Never use intuition, modern English, formula knowledge or "what the
sentence should say" to decide a letter. Never correct a scribe's spelling, never
expand an abbreviation, never complete a word. Copy slips, odd spellings, odd capitals
and broken grammar exactly.

## 1. How the text is written down (the collection's encoding)

`opus_label` uses the same encoding as the labels, so the two can be compared:

| In the ink | Write |
|---|---|
| long s (tall s, also doubled) | `s` |
| et-sign of any shape (&, e-loop, cross, plus, t-shape) | `&` (never `+`) |
| letters raised above the line | `^` before the first raised letter, no closing mark: `W^m` `y^e` `M^r` `24^th` |
| colon / two dots under or before raised letters | `:^` (`S:^d` `Exec:^rs`); one dot `.^` (`y.^e` `Esq.^r`) |
| colon or period after an abbreviated word | kept, attached: `Tho:` `Jon:` `Captn:` `St.` |
| bar, tilde, loop or hook over/through letters (omitted m/n, -cion, per/par/pre/pro) | nothing — write only the letters on the line: `plantacon` `comodities` `pte` `psents` |
| apostrophe-like mark between letters | `'` (`th'other`, `p'sents`) |
| distinct two-dot mark over -con / p- | `"` before the marked letters (`plantac"on`), also `24"` |
| thorn y + small letter | `ye` `yt` `ym` (flat or raised as the ink shows) — never `the`/`that` |
| double f | `ff`; X-p monogram → `Xpian`, `Xpofer` |
| I/J capital shape in a name or a word spelled with J today | `J` (`John`, `James`, `July`); pronoun `I` |
| u / v | the letter shape actually written (`haue`, `Euer`, `vnto` when the ink shows them) |
| crossed-out / blotted-out letters | `~word~` (tildes around exactly the struck letters) |
| word(s) written above the line, usually with a caret ‸ | `^word` at the insertion point |
| wavy flourish filling space | `~` as its own token (`~~`, `~~~` for long ones) |
| straight dash with a gap | ` -` as its own token; `word-` when it touches the word or is a word-break hyphen |
| row of separate short strokes | `_ _ _` ; row of crosses `x x x x` |
| signature mark | the letter or x it looks like, inline |

Character set: ASCII letters, digits, space and `^ : ; . , - ~ _ & ' " ( )` only.
One line, single spaces, punctuation attached to the preceding word unless a clear gap
isolates it.

## 2. What belongs to the line

- The **target line** is the complete line through the vertical middle of the crop.
  Lines cut by the top or bottom edge are neighbours: never transcribe them.
- A narrow crop holding 2–3 **complete** short lines (signatures, witnesses, dates):
  transcribe all complete lines top to bottom, joined by single spaces.
- **Insertions** written above the target line (with a caret or clearly placed in it)
  belong to it (`^word`).
- **Right/left edge:** if the crop cuts a word, write only the letters that are
  visible (`Richard Willi`). Never complete it from the label or from sense.
  A carried-over word tail at the start (`ing`, `ecutors`) is written as it stands.
- A letter or two hidden by a hole or blot *inside* a word may be kept as the label has
  it when the visible strokes agree with it.
- Archivists' modern numbers, stains, bleed-through, ruled lines: not text.

## 3. Decision rule — when to change the label

The label was made by a human transcriber and is right most of the time. Change it
**only where the image clearly shows something different**:

- a different letter or spelling, a missing or extra letter or word;
- a word or letters from a neighbouring line, or a missing part of the target line;
- a cut word that the label completed (write only the visible part);
- a capital written where the label has a small letter, or the reverse — **only when
  the letter form is unmistakable** in this hand;
- a raised letter written flat in the label, or the reverse — only when clearly
  raised / clearly level;
- a mark (`- ~ . , : ; ^ ' "`), struck word or insertion that the ink has and the label
  lacks, or the reverse.

When the ink is genuinely ambiguous (faint, damaged, a shape both readings fit), keep
the label's reading and, if the doubt matters, use verdict `UNCERTAIN` and say why.

## 4. Procedure for each image

1. `Read` the view PNG (top: whole crop with a red ruler every 10 %; below: the line
   in overlapping magnified segments, with their % range of the width).
2. Locate the target line; note neighbours, insertions, struck words, cut edges.
3. Read the ink yourself, word by word, left to right, **before trusting the label**;
   then compare with the label word by word, letter by letter:
   first and last word, word count, every capital, every `e` at a word end, doubled
   letters, `ye`/`the`, `&`/`and`, raised letters, colons and dots, final marks at the
   right end (`-`, `~`, `.`, `,`), cut words.
4. For anything in doubt run
   `python3 analysis/tools/zoom.py $ZOOM <ID> X0 X1 [Y0 Y1]` (percent of width/height,
   e.g. `85 100` for the line end) and `Read` the PNG it prints.
5. Record (see 5). Never skip a row.

## 5. Recording

```
python3 analysis/tools/record.py <BATCH> <<'EOF'
<ID> || <VERDICT> || <CONF> || <opus_label or => || <notes>
EOF
```

- `MATCH` — the label is exactly what the ink shows (opus_label `=`).
- `CORRECTED` — the label is wrong somewhere; give the full corrected line.
- `UNCERTAIN` — part of the ink cannot be verified; give the best reading (`=` allowed).
- `CONF` — `high` / `medium` / `low` for opus_label as a whole.
- `notes` — short and concrete: what differs and the evidence
  (e.g. `last word cut at edge after 'prote'`, `K of Knowne is capital; final e clear`).
  Start with any tags that apply:
  `#cut` (word cut by the crop edge), `#2lines` (two complete lines), `#insert`,
  `#strike`, `#damage` (hole/stain/fading), `#slope`, `#wrongline` (label text belongs to
  another line), `#neighbor` (neighbour-line ink confuses the reading), `#raised`
  (superscript question), `#capital`, `#mark` (end/inside mark), `#spelling`, `#word`
  (missing/extra word), `#mismatch` (label does not fit the image at all), `#join`
  (words linked in the ink), `#dup` (same physical line as another row — name it).

The recorder rejects malformed rows and prints a word diff for every change — read it
and re-record the row if the diff shows an edit you did not intend (the later record
wins).

## 6. Calibration from the pilot (rows 0–33) — apply these

These cases came up repeatedly; decide them the same way every time.

1. **Parallel copies.** Neighbouring rows are often the same formula written by
   different scribes (e.g. four copies of "Sealed and delivered in the presence of").
   Their labels are sometimes copied from a sibling instead of read from the image
   (`writeing` labelled on an image that says `writing`; `presence` vs `prsence`).
   Check every word against *this* image, never against the sibling rows.
2. **Duplicate crops.** Some rows are two crops of the *same physical line* (identical
   ink, blots and insertions) with different labels. The ink is the same, so both
   opus_labels must be the same; tag `#dup` and name the other row.
3. **Capitals — decide by contrast inside the same image.** A letter is capital only if
   its form is clearly different from the small form of the same letter in this image
   (tall looped `S` of `Sixty` vs small pointed `s` of `six`; the same tall looped `S`
   in `Signed` and `Sealed`). Copperplate traps: the curled-ascender `d` (∂) is small
   even at the start of a word; a hooked `y` that starts near x-height is small; an
   `O`/`K` with only an entry loop is small. If you cannot point to that contrast, keep
   the label's case.
4. **e / i / a in copperplate.** A word-initial `i` can carry an entry loop that looks
   like `e`; an open `e` can look like `a`. Compare with the scribe's other `i`/`e`/`a`
   in the same line before changing anything; if still torn, keep the label (`UNCERTAIN`).
5. **Word joins.** Cursive hands often link words without a pen lift (`inthe`,
   `presenceofus`, `deliveredin`). Keep the label's word division; only change it when
   the division is unmistakably different (e.g. a clear gap splitting a label word, or
   two words written as one with no break *and* the label splits them in an unusual
   place). Tag `#join` when words are linked.
6. **Not text:** braces `}` `{` and brackets drawn at the end of attestation lines, red
   margin rules, stray dots at the far edges, descender loops hanging down from the line
   above (they can look like apostrophes, `l`, `d` or `f` strokes inside the target
   words — follow each stroke to its origin).
7. **Line-end and inside marks.** Look for a separate dot after a word (`Year.`), a comma
   (`July,`), a dash or flourish at the end. A dot that is clearly separate ink is a
   period; a tick that grows out of the last letter is not.
8. **Cut words.** When the crop edge cuts the last word, opus_label keeps only the
   visible letters (`of prote`), even if the label completes it. Tag `#cut`.
9. **Family B raised letters.** Many B scribes write `dat`+raised `e`, `th`+raised `e`,
   `On`+raised `e`, often with a dot or dash beneath (`dat.^e th.^e On.^e`). Mark `^`
   only when the letters clearly sit above the x-height; add `.`/`:` only for a separate
   dot/colon under or before them. When an abbreviation's exact form cannot be read
   (which letters are raised, what the marks are), keep the label's form of that word
   and use `UNCERTAIN` with `#raised`.
10. **Speed.** Read the view, compare word by word, zoom only where something is in
    doubt. Most rows need no zoom; a row that needs more than two zooms is probably
    `UNCERTAIN`.
