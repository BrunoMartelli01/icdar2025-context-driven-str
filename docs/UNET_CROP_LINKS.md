# Learned crop refinement with word-instance links

Result class: **approximate reproduction / experimental extension**. Decisions
are recorded under UCL-001 in `PAPER_AMBIGUITIES.md` and in the isolated
`configs/approximate/unet_crop_links_v1.json`. Local paper sections3.2–3.3 and
Algorithm1 were inspected: they specify the backbone family and a single
component/crop pass, not learned pixel links or a trained second pass.

The user reports93% recognition on GT crops. Its detailed protocol and provenance
are not supplied here; that number is not remeasured or used to select checkpoints.
This experiment targets the segmentation and crop gap, keeping PARSeq fixed.

## Four comparable arms

| Arm | Behavior |
| --- | --- |
| `control` | Frozen v1 control best epoch2, one whole-image pass |
| `global_links` | Read-only reuse of the completed V2 `links_isolated` best checkpoint/maps |
| `crop_word` | Separate source-initialized U-Net copy trained on predicted crops; text/kernel outputs, watershed |
| `crop_links` | Matched crop training, plus eight same-word pixel-neighbor outputs and reciprocal link components |

The source is qualified v1 control epoch2, SHA256
`47a941e17472d26c4fff259d0a81c061e23040059c49d97cf2a89a9430045cd8`.
Source receipt, architecture, tensor set, shapes and dtypes are checked before
strict initialization. Encoder weights and encoder BatchNorm remain frozen;
the separate crop decoder trains. Original first-pass weights, existing V2
sources/results and STRICT configurations are never changed.

Global links already belong to V2, so their training is not duplicated. Before
reuse, the complete50-epoch receipt, selected checkpoint hash, validation
denominator, exact resolved V2 configuration, helper code hashes and exported
map hashes must match. An unfinished or incompatible dependency fails explicitly.

## Predicted-window curriculum and labels

Preparation runs the frozen first-pass model on all1255 official TRAIN images,
including the frozen1130/125 split, seed202503. It saves native word/kernel
probabilities, instance labels, checksum-qualified source RGB references and
predicted proposals. Crop windows derive **only from predicted instances**:
50% context per side, at least4 original RGB pixels, clamped to the image.
No GT boxes, OCR text, GT word counts or held-out failure flags choose windows.
An image with no predicted proposals receives an explicit full-image fallback.

Each training epoch samples one existing predicted crop uniformly per TRAIN
image, using a deterministic per-image/epoch generator. Both trainable arms
see identical sampled windows/order and paired horizontal flips at probability.5;
rotation is disabled. RGB is read at original resolution before letterboxing to
512; labels are transformed to256 with the qualified paired owner/separation
policy. Nuclei are recomputed from the transformed owners at distance fraction.5.

GT owners are permitted only for training targets and subsequent scoring. A word
with less than95% of its valid original owner pixels inside a window is ignored
in all objectives, rather than taught as a complete truncated word. Existing
ignore pixels remain invalid. Binary glyph masks are not interpreted as individual
character IDs, and character labels are never fabricated.

Word/kernel losses reuse BCE+Dice and separator weight3. Link targets reuse V2's
eight directions NW,N,NE,W,E,SW,S,SE, valid positive word anchors, inverse-owner
area weighting and separate positive/negative BCE normalization per image. The
link coefficient ramps over5 epochs to.25. Twenty additional epochs, Adam,
shared LR1e-5/private LR1e-4, weight decay.01, FP32/batch4 and four CPU threads
are explicitly declared assumptions/hardware deviations. Fully invalid training
batches skip optimizer updates, including weight decay.

## Inference and original-coordinate evaluation

Second-pass inference reads the exact same frozen context windows, never their
training targets. Crop predictions use watershed (`crop_word`) or reciprocal
links>=.5 (`crop_links`). Sparse graph connected components implement exactly
the existing V2 graph, with randomized equality tests against its BFS routine;
foreground pixels are retained, including isolated graph vertices.

Labels are projected back through the crop letterbox, then translated to original
RGB coordinates. Only child supports touching the original predicted seed are
accepted, so neighboring words in the context are not added independently for
every seed. If no child touches the seed, the seed is retained and counted as a
fallback. Independently detected duplicate supports are merged when their shared
foreground is at least.6 of the smaller support; overlapping boxes alone do not
merge. The merge rule remains an assumption and can still join distinct words
when their predictions overlap. Projection-empty components, whole-image
fallbacks and duplicate counts are logged. The method cannot recover a missed
word outside every proposed context window unless the whole-image fallback applies.

All four arms are evaluated on **original RGB coordinates**, with exclusive xyxy
boxes and the original care annotation boxes. Maximum one-to-one bbox IoU>=.5
gives word precision/recall/F1; .25/.75 are secondary. This differs from V2's
native256 geometry, so compare to the original-coordinate control in this run.
Merge/split diagnostics use any shared valid original owner pixel; their
denominator is all983 care words. Pixel IoU/F1 are per-image macro averages over
the original valid raster. Empty prediction/GT gives perfect pixel score.
Wholly ignored candidate supports are excluded after localization; partial
supports/boxes remain complete. No failed image or missing word is dropped.
No area filter, padding or top10 cap is applied in this diagnostic extension.

Each trainable arm selects the best original-coordinate wordF1 checkpoint,
then fewer fused GT words, higher recall, higher pixel IoU, and earliest epoch;
epoch0 is eligible. Checkpoints include optimizer/RNG state; resume checks the
complete code/config/input/device/backend contract and recovers a checkpoint
saved immediately before its epoch-log append. Changes require a new run root.

Final PARSeq reads original RGB crops from the frozen exported boxes, with pad0,
the checksum-qualified local existing approximate checkpoint/runtime, AR=True
and refine_iters1. It scores IoU>=.5 plus exact casefold, with a separate Unicode
punctuation/symbol-stripped secondary score. All983 words remain the denominator.
OCR crop hashes and backend/runtime/batch/model contracts prevent cache mixing.
Selected-checkpoint exports include raw crop word/kernel probabilities, link
logits, child labels and their cryptographic receipts under `crop_evidence/`.
The summary preserves per-word gains/losses against the frozen source and,
for `crop_links`, against the matched `crop_word` arm. There is no automatic
promotion to the main pipeline based on this reused single-seed TRAIN holdout.

## Run

From the repository root:

```bash
bash scripts/run_unet_crop_links_host.sh \
  artifacts/totaltext-curved-word-boxes-v4 \
  /extra1/bmartelli/context-str-runs/unet-word-research-v1 \
  /extra1/bmartelli/context-str-runs/unet-word-research-v2 \
  /extra1/bmartelli/context-str-runs/unet-crop-links-v1 \
  /extra1/bmartelli/context-str-artifacts/parseq
```

This command may be started while V2 runs: it waits on V2's existing suite lock,
then requires the completed global-link artifacts. It prepares frozen windows,
exports both references, runs separate GPU smokes and20 epochs per crop arm,
exports the selected checkpoints, executes PARSeq in a later process and writes
`comparison.md/json`. Reissuing the same command resumes checkpoints. No package
installation, model download or environment modification occurs. At least8GB
free disk is checked before execution.

The runner supports `--action prepare/smoke/train/evaluate`, `--branch`, and
`--stop-after-epoch` for checkpoint-preserving epoch-boundary interruption.
`--limit-images` is restricted to separate preparation/smoke qualification runs,
which cannot be used for full training or final OCR results.

## Qualification and limits

107 focused/related tests pass, including strict warm start, source preservation,
paired crop selection/owners, ignore/visibility boundary, original-coordinate
projection/offsets, support-based duplicate merging, seeded fallback, reciprocal
graph equivalence, optimizer/RNG resume, interrupted epoch recovery, four-arm
OCR/cache/denominator boundaries, partial-training rejection and synthetic OOM
propagation. Two real CPU FP32 smokes complete training updates and second-pass
validation on the same TRAIN sample/window; all fifteen source windows across
one training and one validation image are prepared at original RGB resolution.
Artifacts: `/tmp/unet-crop-links-cpu-smoke-v4/`.

These are engineering smoke results, not trained performance claims. In the
untrained-link smoke the new head still fragments words; learning/calibration
and downstream improvement must be measured by the full experiment. CUDA is
inaccessible in this implementation session, so GPU training and physical RTX3070
fit are pending. Host preparation/smokes/training record model set, device, FP32,
input/batch, allocated/reserved peaks, and OOM failures. No precision/batch/model
fallback occurs. Official TEST, recognition/context components and original
datasets remain untouched.

Primary method source: [PixelLink](https://arxiv.org/abs/1801.01315).
