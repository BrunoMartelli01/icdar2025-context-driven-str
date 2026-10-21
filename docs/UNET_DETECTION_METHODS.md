# Text pretraining, scale fusion, progressive kernels and word rectification

**Approximate reproduction / experimental extensions.** These methods do not
change the STRICT architecture/configuration, recognition/context/fallback,
existing datasets, weights or results. Scientific record: UDM-001 in
`PAPER_AMBIGUITIES.md`. The local paper `papers/2503.15639v2.pdf` sections3.2–3.3
and Algorithm1 were inspected before implementation: they describe ResNet50,
attention U-Net and connected-component crops, rather than these extensions.

## Existing work reused

The existing V2 and learned-crop suites have now finished. They are not retrained
by the new launcher. V2's frozen reference is520/983 exact localized-and-read
words; DB cores522; links490 with substantial fragmentation. The crop suite has
its own measured source521, crop_word524 and crop_links476. Its source and native
geometry protocol differ slightly; these historical numbers are not silently
inserted as matched arms in the new experiment. See `UNET_WORD_RESEARCH_V2.md`,
`UNET_CROP_LINKS.md` and the completed external `comparison.md/json` receipts.

Existing overlapping-window proposals, threshold sweeps, PCA crops and crop
selection studies are retained; no duplicate implementation/training is added.
The existing Bézier geometry was an ICDAR13 fixed-group diagnostic; its actual
foreground-envelope fitter and RGB renderer are now reused on Total-Text words.

## Comparable arms

All use qualified v1 Text+Kernel control best epoch2, SHA256
`47a941e17472d26c4fff259d0a81c061e23040059c49d97cf2a89a9430045cd8`, and the same
official TRAIN1130/held-out125 split, seed202503, all983 care words. RGB512,
native256, paired source letterbox/flip/rotation and original targets stay fixed.

| Arm | Training and output |
|---|---|
| `reference` | Frozen source; no retraining |
| `control` |50 additional epochs; full encoder-feature convolution adaptation |
| `scale_fusion` | Matched control + compact spatial decoder scale fusion |
| `progressive_kernels` | Matched control + two larger nucleus outputs |
| `fusion_progressive` | Both structural changes, with identical training |
| `pretrained_control` | Explicit SynthText domain pretraining, then matched50epochs |
| `pretrained_fusion_progressive` | Same synthetic checkpoint, then combined structure |

Encoder convolution LR1e-6, shared decoder LR1e-5, new module LR1e-4; Adam,
weight decay.01. Encoder BN affine and running statistics stay frozen; the
experimental `_encode` explicitly permits gradients to feature convolutions.
This is a new full-encoder adaptation control, not a rerun of the historical
partial layer3/4 experiment. Unused ResNet classification parameters stay frozen.
FP32/batch2, single GPU and sequential segmentation/recognition are declared
hardware deviations; actual fit on8GB remains to be measured. CPU smokes record
their explicit device/batch overrides. OpenCV uses one CPU thread.

Scale fusion projects /8,/4,/2 decoder features to32channels and predicts three
spatial weights. A zero-initialized64channel residual preserves the original
logits at initialization. Exact integral nearest expansion uses tensor
expand/reduction autograd, avoiding CUDA bilinear backward's nondeterminism.
This adds16,835 parameters; kernels add130; combined16,965 (under0.03% of the
58,155,146-parameter two-head source). It is inspired by
[DBNet++ ASF](https://arxiv.org/abs/2202.10304), not its architecture/checkpoint.

Kernel distance fractions.5,.3,.1 define three nested connected interiors. The
existing largest-core rule supplies the smallest separable seed; larger targets
retain the connected component containing the preceding core. Tiny surviving
native words retain a seed. Native-resolution word losses are counted, never
invented back. Two65parameter output heads start from the original kernel head.
The existing masked BCE+Dice is reused for text/kernels; text separator weight3,
additional kernel losses averaged, masked nesting penalty coefficient.1. Invalid
batches skip optimizer updates including decay. Multi-source confidence growth
progresses through the kernels into the full text support, keeping seed IDs;
unseeded components remain explicit fallbacks. Equal initial kernel predictions
preserve historical watershed labels. This is a
[PSENet-inspired](https://arxiv.org/abs/1903.12473) raster/confidence adaptation,
not original polygon-shrink targets or its BFS algorithm.

## Optional SynthText domain pretraining

The launcher requires an explicit choice; it never downloads data or silently
skips a requested pretraining stage. Supply a directory containing `gt.mat` and
its images. The preparation reads only `imnames`, `wordBB`, `txt`; character
boxes are not used. Whole word polygons use the existing ownership, overlap,
separator, validity and augmentation code. MAT cells, word/transcription counts,
coordinates, files and checksums are validated; failures propagate.

The declared subset is50,000 images sampled without replacement with seed202503;
five synthetic epochs, decoder LR1e-4/encoder1e-5. All1255 Total-Text TRAIN RGB
byte hashes and duplicate synthetic RGB bytes are excluded and exclusions logged.
Perceptual near-duplicate overlap remains unresolved. Coordinate origin0 is
explicitly recorded from the upstream Python generator convention, not inferred
from MATLAB storage; a differently converted dataset needs its declared origin.

This **warms the already qualified v1 model on synthetic text before the new
Total-Text fine-tuning**; it is not ImageNet-only initialization or training on
the whole800k corpus. Extra data and optimizer steps distinguish it from controls.
The synthetic stage selects its last checkpoint without held-out Total-Text
selection; Adam resets for subsequent matched fine-tuning. Complete annotation,
subset, image, source/checkpoint/config/RNG provenance is retained. No checkpoint
trained on all Total-Text TRAIN, including our125 holdout, is substituted.
[DBNet's official training guidance](https://github.com/MhLiao/DB#training)
motivates synthetic pretraining; the exact adaptation schedule is an assumption.

## SynthText target correction (isolated v2 configuration)

The interrupted `unet-detection-methods-synthtext-small-v1` run saved only
epoch0/global_step0. In shuffle position75, `synthtext:348698`, word10 (`--`)
had39 original raster pixels in `171/sun_6_10.jpg` (600x600); the two-sided
contact corridor removed its entire ownership region. This was a valid visible
annotation, not a missing image. A separate raw-raster audit of the same50,000
images/422,616 annotations also found6 boxes wholly outside their images.

`configs/approximate/unet_detection_methods_synthtext_targets_v2.json` explicitly
selects `preserve_visible_words_v1`. The original v1 configuration/defaults and
Total-Text target generation remain unchanged. The new policy:

- retains the original image cohort, annotation order, polygons and transcripts;
- records wholly off-image word indices as unobservable, without moving them
  onto image borders, excluding images or replacing them with other samples;
- restores **pre-corridor owned pixels within the original polygon** only when
  contact removal would erase a whole visible word; restored text cannot also
  be a separator target;
- continues to fail on visible words that cannot obtain an owner, including
  identical conflicting annotations; errors identify the sample and word IDs;
- validates all source-resolution targets during preparation, saves per-image
  audits and aggregate counts in `synthtext_manifest.json`, and checks those
  audits again during dataset access. Epoch metrics also record off-image,
  restored-contact and native-resampling word-loss counts.

This synthetic-only training target choice is an `ASSUMPTION`, documented in
UDM-SYNTHTEXT-TARGETS-001. Recovering a source-resolution target does not guarantee
that a very small word survives native256 resampling; those losses remain counted.
The unchanged983-word Total-Text evaluation denominator receives no new exclusions.

On `bmartelli@paper.dinfo.unifi.it`, reuse the already extracted subset and start
in a **new** root because both the scientific source hashes and target policy
changed. No dataset download, checkpoint deletion or contract editing is needed:

```bash
cd /extra1/bmartelli/projects/icdar2025-context-driven-str
bash scripts/run_unet_detection_methods_host.sh \
  artifacts/totaltext-curved-word-boxes-v4 \
  /extra1/bmartelli/context-str-runs/unet-word-research-v1 \
  /extra1/bmartelli/context-str-runs/unet-detection-methods-synthtext-small-v2 \
  /extra1/bmartelli/context-str-artifacts/parseq \
  --synthtext-root /extra1/bmartelli/datasets/context-str/SynthText \
  --config configs/approximate/unet_detection_methods_synthtext_targets_v2.json
```

Preparation now audits all synthetic targets before training. Identical later
invocations resume completed epoch checkpoints under the v2 contract. This is an
approximate reproduction extension; fixing target construction is not evidence
of improved detection or end-to-end PARSeq accuracy.

Qualification receipt: [synthtext_target_generation_v2.json](../provenance/synthtext_target_generation_v2.json).
The full frozen50,000-image cohort passed source-resolution generation:422,616
annotations,422,610 visible words,6 off-image annotations and20 restored contact
words across26 affected images. Seven actual problem images passed RGB SHA,
paired transformation, nested kernels and finite loss/backward on zero test
logits. Two still lose a word under native256 resampling; this is recorded and
is not credited as successful word detection.

Tests:63 passed across the focused suite and two additional ignore-restoration
cases; one PyTorch multiworker tensor-transfer test failed because the sandbox
denies POSIX socket/shared-memory IPC. No test is skipped or error suppressed.
An initial fork-based test stalled and was interrupted; the host integration test
uses spawn and a30-second timeout. A named shared-memory transfer attempt was
also denied and was removed from the final test; production worker settings
remain unchanged. Run the complete suite on the host:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venvs/context-str-main-py311/bin/python -m pytest -q \
  tests/segmentation/test_unet_detection_targets.py \
  tests/segmentation/test_totaltext_word_regions.py \
  tests/scripts/test_unet_detection_runner.py \
  tests/scripts/test_synthtext_download.py
```

Original configuration/launcher/checkpoints are preserved, qualified source
loading and `git diff --check` passed, and concurrent CLIP documentation edits
were retained. Complete new GPU training/peak memory/OOM handling and OCR metrics
require the host run above; the interrupted run's memory measurement is partial.

## PARSeq and rectification

Each selected checkpoint is tested with three frozen crop views: original axis
box, PCA-oriented, and cubic Bézier upper/lower envelopes. These reuse the
existing fitter/renderer, with pad0 and explicit axis fallback reasons for invalid
fits. Original RGB support is reconstructed by nearest resampling of the assigned
native instance inside its inverse-letterbox box. Other word IDs do not enter
its fit. All three retain the **same original prediction bbox for GT matching**;
rectification is never credited by changing the matching box. All views and pixel
hashes are frozen before OCR/GT evaluation. No GT chooses a view and no spelling
correction or language model is added. This is inspired by
[ABCNet](https://arxiv.org/abs/2002.10200); it is not learned ABCNet/BezierAlign.

The qualified local genuine PARSeq checkpoint/runtime, AR=True/refine_iters1,
stay fixed. The artifact is not established as the paper's PARSeq-S. The primary
metric requires original-coordinate maximum one-to-one bboxIoU>=.5 and exact
Unicode casefold, with punctuation retained. P/S stripping is separate. All983
words remain denominators; empty/duplicate predictions remain in care output
counts. Wholly invalid predictions are excluded after localization; partially
ignored instances retain complete boxes. No area filter, padding or top10 cap is
introduced into this diagnostic. This is a local bbox proxy, not official polygon
spotting or strict Table5. Pixel IoU/F1 macro/micro and native any-pixel merge/split
rates are also recorded. An80% end-to-end recall needs at least787/983 successes;
that outcome has not been measured for these newly trainable arms.

## Launch

Run from the repository on `bmartelli@paper.dinfo.unifi.it`. With available
Total-Text/PARSeq data, run the complete structural and rectification experiment:

```bash
bash scripts/run_unet_detection_methods_host.sh \
  artifacts/totaltext-curved-word-boxes-v4 \
  /extra1/bmartelli/context-str-runs/unet-word-research-v1 \
  /extra1/bmartelli/context-str-runs/unet-detection-methods-v1 \
  /extra1/bmartelli/context-str-artifacts/parseq \
  --without-synthtext
```

For all arms including external pretraining, set `SYNTHTEXT_ROOT` to your actual
dataset directory, then use a **different** output root:

```bash
bash scripts/run_unet_detection_methods_host.sh \
  artifacts/totaltext-curved-word-boxes-v4 \
  /extra1/bmartelli/context-str-runs/unet-word-research-v1 \
  /extra1/bmartelli/context-str-runs/unet-detection-methods-synthtext-v1 \
  /extra1/bmartelli/context-str-artifacts/parseq \
  --synthtext-root "${SYNTHTEXT_ROOT:?Set this to the real SynthText directory}"
```

Both wrappers preflight targets/source/artifacts, run separately contracted GPU
smokes, train only new arms and execute PARSeq in later processes. Minimum12GB
free disk is checked. Repeating identical arguments resumes epoch-boundary
checkpoints and reuses completed results. Code/config/device/data/phase changes
require a new output directory. `--config configs/approximate/...json` selects a
separate declared schedule. Model-loading, OOM, worker and inference errors are
recorded and propagated; no automatic batch/precision/model substitution occurs.

Per-arm selected checkpoints use original-coordinate wordF1, then fewer fused
words, higher recall/pixelIoU and earliest epoch; epoch0 is eligible. Do not call a
partial run a result. The common low-level runner also exposes `--stop-after-epoch`
for resumable training and `--action prepare-synthtext/pretrain/smoke/train/evaluate`.

Final outputs: `comparison.md/json/csv`, paired per-GT gained/lost identities,
Precision/Recall/Hmean, native merge/split and pixel metrics; source/epoch/memory
receipts; `validation_exports/` maps/labels/overviews/kernel panels;
`ocr_inputs/` all actual crop views; `parseq_cache/` raw readings/probabilities;
`review/index.html` fixed-order source examples. Completed V2/crop results remain
in their original roots. No automatic promotion follows this reused single-seed
TRAIN holdout; independent generalization and the80% goal remain unresolved.

## Tests and qualification

```bash
.venvs/context-str-main-py311/bin/python -m pytest -q \
  tests/segmentation/test_unet_detection_targets.py \
  tests/localize/test_unet_detection_instances.py \
  tests/models/test_unet_detection_architecture.py \
  tests/scripts/test_unet_detection_runner.py \
  tests/segmentation/test_totaltext_word_kernel.py \
  tests/segmentation/test_word_multitask.py \
  tests/localize/test_word_guided_contours.py \
  tests/localize/test_word_crop_refinements.py \
  tests/scripts/test_evaluate_text_recall_refinements.py
```

Qualification receipts are in `provenance/unet_detection_methods_v1_qualification.json`.
CPU tests and explicit engineering smokes qualify shapes, gradients, source
preservation, target/loss/separation, MATLAB decoding, crop/recognition boundaries,
resume and failure propagation. They do not establish full-training improvements,
external SynthText effectiveness, GPU speed or RTX3070 memory fit.

Final focused/related run: **93 tests pass**. Five real-source CPU smokes pass;
125-image target preflight preserves983/983 words; genuine CPU PARSeq evaluates
all15 smoke variant/views using18 unique crops, and cache replay is unchanged.
A two-image generated MATLAB fixture verifies an actual512/batch2 domain-training
update, with encoder gradients and unchanged encoder BN; it is explicitly fixture
data, not external SynthText performance. An initial pytest collection collision
from duplicate new test basenames was corrected; final collection/tests pass.
No remaining test/smoke failure was observed. CUDA was inaccessible; full training
and actual external SynthText data were not run in this qualification session.
