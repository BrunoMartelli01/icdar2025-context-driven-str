# Frozen text-recall crops and actual PARSeq recognition

This is an **approximate reproduction**, exploratory diagnostic on the reused
Total-Text TRAIN holdout (125 images /983 care words), not official TEST or
paper Table5 accuracy. No segmentation/offset training or inference is repeated.

The source experiment's word Hmean was **localization only**. This extension
runs actual pinned PARSeq on unmodified original RGB crops, requiring both
one-to-one original exclusive-bbox IoU>=0.50 and exact Unicode casefold text.
Punctuation is retained; symbol-normalized metrics remain secondary. Global
precision divides correct matches by all care proposals, recall divides by all
983 care words, and Hmean is2TP/(proposals+983). Empty denominators yield0.
Fully invalid predicted instances are ignored only after all recipes are frozen;
partially overlapping instances keep their complete boxes. No failed sample may
be silently excluded. Per-sample boxes, raw OCR and matching identities are saved.

Existing work was inspected first: the four completed50-epoch text branches,
`WORD_GUIDED_CONTOURS.md`, `CENTER_GUIDED_WORD_CROPS.md`, offset epoch82 and
the paper's local sections3.3/3.5. The old offset study already degraded OCR:
425 baseline words vs334 global-center words. It is preserved and not repeated.
This study tests transfer to the new masks/thresholds, not an assumed improvement.

Four selected checkpoints are evaluated at their frozen selected thresholds
and, separately, the predeclared common0.50 threshold. Checkpoints are never
reselected using OCR. Both512 selections are epoch0 and identical predictions:
postprocessing and RGB-hash OCR caches avoid duplicate computation.

Each policy has four GT-blind methods:

- Connected components: replay the original localization diagnostic exactly.
- FindContours: existing external-contour pipeline; no invented pixels in holes.
- Distance watershed: smoothed distance maxima seed the existing minimax flood,
  with one fallback marker for tiny components; every foreground pixel survives.
- Offset watershed: reuse existing global endpoint consensus and geodesic flood,
  with growth disabled; shared identities may join fragments and split fused masks.

Changing contour extraction alone cannot separate connected words. Distance
markers may separate narrow bridges but over-split irregular single words.
The offset predictor still runs512/256: transfer to canonical256 maps from768
does not claim a newly trained768 offset model. No annotations, ignore maps,
GT counts or OCR outputs enter localization. Parameters are frozen in
`configs/approximate/text_recall_crops_v1.json` before full scored evaluation.
All methods use pad0/no area filter/no top10 cap to isolate instances; these
diagnostic choices differ from the strict paper algorithm and never change it.

Only PARSeq runs on CPU FP32/batch8; original RTX3090 FP32 segmentation outputs
are reused. No concurrent GPU residency or target RTX3070 fit is claimed.
Pinned local source archive/checkpoint hashes, AR=True/refine1, complete source
reports, code/config/data/split hashes, offset exports and crop RGB hashes are
verified and recorded. Completed OCR caches require per-record SHA verification,
identical pinned decoding/artifacts, conflict rejection and16-crop CPU replay.
The existing recognizer is reused through a new auditable provider adapter.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
.venvs/context-str-main-py311/bin/python -m scripts.evaluate_text_recall_crops \
  --mode approximate --action all --device cpu \
  --run-root /extra1/bmartelli/context-str-runs/totaltext-text-recall-resolution-v1 \
  --offset-run artifacts/training/totaltext-word-center-offsets-v1/train \
  --output-root artifacts/text-recall-parseq-crops-v1 \
  --parseq-source /extra1/bmartelli/context-str-artifacts/parseq/parseq-1902db043c029a7e03a3818c616c06600af574be \
  --parseq-source-archive /extra1/bmartelli/context-str-artifacts/parseq/parseq-1902db043c029a7e03a3818c616c06600af574be.tar.gz \
  --parseq-checkpoint /extra1/bmartelli/context-str-artifacts/parseq/parseq-bb5792a6.pt \
  --reuse-run-root artifacts/word-multitask-crop-refinement-v1 \
  --reuse-run-root artifacts/center-guided-word-crops-v1
```

## Completed OCR results

Both studies completed successfully:32 initial and16 learned-cue pipelines,
each on all125 images/983 words. **These counts are actual PARSeq exact-casefold
matches with original bbox IoU>=0.50**, not foreground coverage or localization
alone. The following table retains the original training-selected checkpoints
and thresholds; no OCR checkpoint search was performed.

| Text source | Epoch / threshold | FindContours correct /983 | OCR Hmean | Kernel watershed correct /983 | OCR Hmean |
|---|---|---:|---:|---:|---:|
| BCE+Dice512 | 0 /0.50 | 418 | 44.61% | 501 | 48.81% |
| BCE+Tversky512 | 0 /0.50 | 418 | 44.61% | 501 | 48.81% |
| BCE+Dice768 | 50 /0.65 | 414 | 42.92% | 499 | 47.32% |
| BCE+Tversky768 | 10 /0.90 | 443 | 45.25% | 525 | 49.39% |

Both512 predictions remain identical epoch0 selections. With Tversky768/.90,
kernel transfer gains104 previously unmatched GT word identities and loses22,
for net+82. Counts improve in43 images, worsen in13, and stay equal in69.
OCR precision/recall change from45.44%/45.07% to45.93%/53.41%.
Native any-shared-pixel merge rate falls31.74%->22.38%, while split rate rises
5.70%->13.33%: improved aggregate OCR does not imply every crop improved.

At the separate common0.50 point, Tversky768 FindContours drops to320 correct,
Hmean34.43%, despite its greater foreground coverage. Kernel transfer obtains
523 correct/Hmean47.83% on that same foreground. This supports using learned
instance cues instead of selecting broad masks by coverage alone. The .90
point still has higher measured OCR Hmean with kernels in this frozen study.

At Tversky768/.90, distance watershed yields345 correct/Hmean28.87%, old global
offsets355/33.21%, and coherent v2 offsets358/33.40%. These fixed alternatives
underperform FindContours; their success on ideal-target unit tests did not
establish robustness to actual noisy prediction maps. Connected components are
almost identical to contours; they cannot identify distinct touching words.

The first study recognizes9,140 new unique RGB crops and reuses1,270 (10,410
total); the learned study recognizes3,534 new and reuses4,453 (7,987 total).
Each cache replays16 current-device crops exactly. CPU FP32/batch8 PARSeq
times are844.43s and307.70s, excluding geometry generation. Memory metadata
records CPU-only recognition; no GPU segmentation rerun or RTX3070 claim.

96 focused/related tests pass, including17 new localization/provenance tests.
The initial test collection found duplicate test filenames; the script test was
renamed and the full focused suite rerun successfully. No experiment failed.
Independent scoring audits reproduce6,000 per-image/pipeline rows,54,335
original RGB crops,13,000 export hashes and every OCR record. All source GT
transcriptions/boxes/annotation hashes match the prior qualified125 holdout.
The original native localization diagnostic replays exactly for500 source cases.
Receipts: `provenance/text_recall_crops_v1_results.json` and
`provenance/text_recall_learned_crops_v1_results.json`.
The exact independent audit source is archived as
`provenance/text_recall_crops_independent_audit.py.txt`; its SHA256 matches both
receipts. Run it with the main Python and `PYTHONPATH=.`, passing `--root` to
either new output directory; it performs no model inference or source mutation.

`artifacts/text-recall-learned-crops-v1/paired_comparison.json` and the two
`REPORT.md` files retain every48 pipeline results. A20-example offline review,
including gains/losses/typical/dense/unchanged cases and raw OCR strings, is
`artifacts/text-recall-parseq-review-v1.zip`:5,675,076 bytes, SHA256
`9ecd855e0d94b55cf11355606e374eca23b462edc16fdc694854259d8a087aac`.

The supported conclusion is **learned kernel transfer improves OCR on this
reused holdout**. It uses two frozen source networks, not one newly trained
lightweight model. The existing shared-decoder Text+Kernel architecture adds
only65 parameters versus the original text model, but reproducing the transfer
gain in a jointly trained text/kernel model requires a new matched experiment.
Independent TEST/generalization, optimal hyperparameters, crop rectification,
padding and center calibration remain unresolved. Do not promote this result
to a strict paper reproduction or an independently selected benchmark claim.

Actions `preflight`, `geometry`, `recognize`, `all` allow staged execution and
reuse of completed, verified phases. Source runs, weights, dataset and old OCR
are immutable. New outputs are isolated; changed contracts fail explicitly.
Full results are `geometry_results.json`, `recognition_results.json`,
`summary.json` and `REPORT.md`. Mode/config/source/environment metadata and
per-image outputs remain available to audit each aggregate and all errors.
Selecting a favorable diagnostic on this reused holdout does not establish
independent generalization. Scientific choices are recorded in TRC-001.

## Existing learned cues, without new training

A second, isolated comparison uses existing **Text+Kernel epoch61** and the
completed **coherent-vote v2 epoch84**. Both were trained on the same frozen
holdout. Their prediction grids, owners, care validity, source/checkpoint hashes
and preprocessing must align exactly with every new text mask.

`kernel_watershed` combines the current text probabilities with the old learned
kernel probabilities and reuses `word_instances` with the original kernel
threshold0.50 and watershed weights. `offset_votes_v2` uses exactly the same
global consensus parameters as the first study; only the trained vector source
changes. Both retain every current foreground pixel and keep fallback instances.
This is **transfer between already trained models**, not a newly trained joint
U-Net or proof that two backbones form a lightweight deployment. If transfer is
useful, a later matched shared-decoder/head training experiment is still needed.

The16 added pipelines preserve selected/common0.50 policies, all125 images and
983 words, original RGB crop coordinates and the same actual PARSeq scoring.
No existing baseline recognition is repeated: exact crop hashes reuse the
completed first study and earlier caches, with checked provider hashes/replay.
Training and original checkpoint selections remain unchanged. TRC-002 records
these decisions and limitations.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
.venvs/context-str-main-py311/bin/python -m scripts.evaluate_text_recall_learned_crops \
  --mode approximate --action all --device cpu \
  --run-root /extra1/bmartelli/context-str-runs/totaltext-text-recall-resolution-v1 \
  --offset-run artifacts/training/totaltext-word-center-votes-v2/coherent/train \
  --kernel-run /extra1/bmartelli/context-str-runs/totaltext-word-multitask-v1/text_kernel \
  --output-root artifacts/text-recall-learned-crops-v1 \
  --parseq-source /extra1/bmartelli/context-str-artifacts/parseq/parseq-1902db043c029a7e03a3818c616c06600af574be \
  --parseq-source-archive /extra1/bmartelli/context-str-artifacts/parseq/parseq-1902db043c029a7e03a3818c616c06600af574be.tar.gz \
  --parseq-checkpoint /extra1/bmartelli/context-str-artifacts/parseq/parseq-bb5792a6.pt \
  --reuse-run-root artifacts/text-recall-parseq-crops-v1 \
  --reuse-run-root artifacts/word-multitask-crop-refinement-v1 \
  --reuse-run-root artifacts/center-guided-word-crops-v1
```
