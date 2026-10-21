# Compact word GT v2 and isolated retraining

This is an **approximate reproduction** experiment requested after reviewing
the separated-v1 training. Paper section 3.3 applies padding to text-component
boxes; it does not specify shrinking ground truth or a shrink ratio. The
parameters below are explicit, untuned `ASSUMPTION` choices (WORD-GT-002 in
`docs/PAPER_AMBIGUITIES.md`), not paper-derived targets.

The new delivery is `artifacts/separated-word-gt-v2/compact-cores/`. Each word
has a smaller filled interior region. It is not a character-stroke mask. Source
GT, photos, old target PNGs, annotations and historical runs remain unchanged.
Official TEST transcriptions and geometry still define recognition evaluation;
the regenerated TEST target images do not produce inference crops.

## Construction

- Start with every owner in the qualified separated-v1 delivery: Total-Text
  TRAIN/TEST, ICDAR2015 TRAIN/TEST, ICDAR13 TRAIN/TEST and curated CTW1500 TEST.
- Compute distance to each word boundary in anisotropic 128x128 reference units.
  The requested inset is 55% of that word's maximum interior distance, capped
  at two reference pixels. Keep the largest connected core with stable ties.
- Keep at least 20% of area and 80% of longest bbox span. The old crop padding
  (six reference pixels) must cover the entire original owner bbox. Back off
  by halving the inset up to eight times; restore the original if necessary.
- Replay the actual nonaugmented training-target resize: bilinear target with
  threshold 0.5, nearest owners/validity/separators and removal of cross-word
  owner contacts. Restore any additional word lost relative to v1, repeating
  until no new losses remain. Already absent native words remain reported.
- Every new owner is a subset of the original owner. There are no added or
  reassigned pixels. Ignored validity is unchanged. Removed foreground joins
  the auxiliary zero-target separator mask.

This variant intentionally removes some character pixels from supervision;
the old stroke-protection claim does not apply. Padding coverage concerns the
GT core geometry, not a guarantee that a trained prediction recovers every
letter. Small words may require restoration, and random rotations may remove
words at 128x128. The area150 contour filter can reject more compact predictions.
The old padded-box IoU scoring issue is not corrected by changing GT.

Reproduce generation with the existing environment:

```bash
.venvs/context-str-main-py311/bin/python -m scripts.prepare_tighter_word_gt \
  --mode approximate \
  --source-root artifacts/separated-word-gt-v1/v4-wider-gaps \
  --output-root artifacts/separated-word-gt-v2/compact-cores \
  --dataset total_text icdar2015 icdar13 ctw1500 \
  --distance-fraction 0.55 --max-inset-reference 2 --reference-size 128 128 \
  --crop-padding-reference 6 --min-area-fraction 0.2 --min-span-fraction 0.8 \
  --backoff-steps 8 --zip
```

The generator refuses existing destinations; use a new root to reproduce.
Separate smoke exports use `--limit 2` and are labeled smoke-test results.
The per-image schema contains binary masks, uint16 owners, unchanged validity,
separators, source annotation/transcription references, SHA256s and a detailed
`compact_audit` with per-word area/restoration decisions. ZIPs omit source photos.

## Verification

```bash
.venvs/context-str-main-py311/bin/python -m scripts.verify_tighter_word_gt \
  --root artifacts/separated-word-gt-v2/compact-cores \
  --output artifacts/separated-word-gt-v2/compact-cores/independent_audit.json
```

The verifier uses OpenCV binary8-connected components independently of the
SciPy core construction. It checks every output/source checksum, owner subset,
word survival, no shared components, padding coverage, minimum area/span,
unchanged source transcripts and no additional native-target disappearances.
No failed image is skipped. Independent audits and resolved configuration are
required by the training bridge before GPU execution.

## Train, crop and PARSeq

From the physical server terminal:

```bash
cd /extra1/bmartelli/projects/icdar2025-context-driven-str
bash scripts/run_totaltext_tighter_words_train_and_parseq_host.sh --check
bash scripts/run_totaltext_tighter_words_train_and_parseq_host.sh
```

This uses **Total-Text TRAIN1255**, the same AG-U-Net as V4/v1, 500 epochs,
FP32/batch8, frozen pretrained ResNet50, Adam and TRAIN foreground-IoU checkpoint
selection. A new approximate config and protocol isolate compact-v2 targets:
`configs/approximate/segmentation_total_text_tighter_words_v2.yaml` and
`practical_totaltext_tighter_words_500_v2`. V1/V4/strict configs stay distinct;
target-source and artifact identity guards reject accidental cross-target runs.

The script then predicts masks on the original 300 TEST photographs, saves
native128 masks and padded RGB crops, and applies the pinned local PARSeq model.
Its inference settings and metrics remain those of the previous run: threshold
0.5, bbox area150, padding6, top10; one-to-one IoU>=0.50 and exact Unicode-casefold
text. Report the all2204-GT word rate and geometry-conditional OCR rate separately.
These are the existing approximate diagnostic metrics, not paper Table5 results.
Sequential model loading and approximate PARSeq artifacts remain inherited
experimental choices; no DeepSolo/context-paper pipeline claim is made.

Default output directories are separate from previous completed runs:

```text
/extra1/bmartelli/context-str-runs/totaltext-tighter-words-v2-500ep/total-text-tighter-words-v2-500ep/
/extra1/bmartelli/context-str-runs/totaltext-tighter-words-v2-findcontours-parseq-test/
```

Use `COMPACT_WORD_GT_ROOT`, `COMPACT_WORD_RUN_ROOT` and
`COMPACT_WORD_RESULTS_ROOT` for explicit path overrides. Interrupted training
requires `--resume /path/to/checkpoints/epoch-NNNN.pt`. Existing run outputs are
not overwritten. The production command retains eight DataLoader workers;
managed sandbox tests use an explicit single-process tiny CPU fixture because
the sandbox disallows PyTorch AF_UNIX sharing sockets. These fixtures do not
qualify GPU or production-worker behavior. No full training or accuracy claim
is made while generating targets and preparing the requested rerun command.

## Complete v2 qualification

All 4017 masks were regenerated and independently verified, preserving all
24734 source care owner IDs at original resolution, with no shared binary
components between words. No additional word disappears in the effective
nonaugmented native128 target relative to v1. Two inherited Total-Text TRAIN
fragmented words remain explicitly recorded. CTW1500 TRAIN still has no supplied
word geometry and is not generated.

| Split | Images | Foreground-area reduction from v1 |
|---|---:|---:|
| Total-Text TRAIN | 1255 | 37.04% |
| Total-Text TEST | 300 | 42.80% |
| ICDAR2015 TRAIN | 1000 | 35.53% |
| ICDAR2015 TEST | 500 | 35.74% |
| ICDAR13 TRAIN | 229 | 33.58% |
| ICDAR13 TEST | 233 | 29.28% |
| CTW1500 curated TEST | 500 | 30.68% |

All four ZIPs were checked member-by-member against the delivered files, and
generation code fingerprints match each split's resolved config. The independent
audit is `independent_audit.json`; `delivery.json` records archive hashes and all
split summaries. Seven two-image smoke exports passed. All 47 focused
GT/namespace/training-dispatch tests passed, with the previously documented
explicit single-process CPU integration fixtures. There were no generation or
independent-audit failures. Both the old-v1 and new-v2 CPU host preflights passed;
old config/manifest hashes remain unchanged.

The v2 training config SHA256 is
`0cfb83466e7c74950a43f58bf452ef4004ccfb81923ca907758ca3f338f17a25` and adapter
TRAIN manifest SHA256 is
`67a2cd2c8e702f40badb91c54595fe322ba814033f470e13b07e879a3db6d179`.
New GPU training and recognition accuracy remain pending the user-run command.
