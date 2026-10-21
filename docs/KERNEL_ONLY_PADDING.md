# Extracted Kernel-only output and padding

This is **approximate reproduction / an exploratory research extension**.
The user corrected Gap to **padding**. This experiment extracts one output
from the existing jointly trained Text+Kernel model; it does not claim new
training exclusively on Kernel targets. The source completed50 additional
epochs from Text+Kernel epoca61 and selected its best epoca2.

Before implementation, the existing model, target, padding and evaluation
components and authoritative paper sections3.2–3.3 / Algorithm1 were inspected.
The paper specifies component boxes and clamped padding. The extraction,
Kernel-only components and comparison protocol are ASSUMPTION. Actual hardware
RTX3090 rather than3070 and sequential segmentation/PARSeq are
HARDWARE_DEVIATION. Generalization remains UNRESOLVED.

The network copies every shared tensor/BN buffer and only channel1 of the
Text+Kernel1x1 output layer. Strict loading rejects missing/extra/incompatible
keys, shape or dtype. RGB512 letterbox input produces one native256 output.
CUDA inference checks all125 images against the frozen source probabilities
and requires exactly identical thresholded Kernel masks on the RGB canvas.
The derived inference checkpoint records joint-training provenance.

Kernel-only decoding uses K>=0.5,8-connected components and the RGB-only
canvas. No Text gating, learned Gap, region growth, area filter or top10.
The reference reuses its original Text+Kernel labels. Padding
**0/2/4/6/8/12 native256 pixels per side** is fixed before the experiment,
reusing the existing grid. Padding precedes inverse letterbox mapping and
clamps boxes at image boundaries. No GT enters inference or box generation.

All models use the same1130/125 Total-Text TRAIN split and983 validation
care words. Report maximum-cardinality one-to-one original RGB bbox IoU>=0.50:

- `unpadded_geometry_iou_ge_50`: correctly localized original component boxes;
  invariant to padding within a model.
- `geometry_iou_ge_50`: correctly localized **padded** boxes used as crops.
- `exact_casefold_iou_ge_50`: padded-box IoU>=0.50 plus exact PARSeq text after
  Unicode casefold; punctuation retained. Symbol stripping is secondary.
- Proposals, precision, recall, Hmean and complete per-image outputs. Empty
  predictions retain their GT denominator. Wholly invalid instances are ignored
  after decoding/cropping/OCR; partially valid boxes remain whole.

Recognition uses the unchanged checksum-qualified PARSeq archive/checkpoint,
AR/refine1, FP32 batch8 on CUDA after segmentation unloading. Crop caches check
RGB content hashes and recognizer provenance. The Text+Kernel padding0 result
must exactly reproduce its original524/983 OCR result before success.
Errors/OOM propagate with receipts; there is no CPU fallback, installation,
checkpoint overwrite or GT-dependent crop selection. No TEST input is used.

Run with explicit existing artifacts and an unused external output path:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venvs/context-str-main-py311/bin/python \
  -u -m scripts.evaluate_kernel_only_padding --mode approximate \
  --source-run /path/to/adaspot-run/control \
  --word-root /path/to/totaltext-v4 --parseq-artifacts /path/to/parseq \
  --output-root /path/to/new-output --device cuda:0
```

Outputs: `kernel_only.pt`, native probabilities/masks, resolved source/input/
runtime contracts and hashes, frozen crops, qualified OCR caches,
`results.json` and `comparison.md`. Tests verify real single-output extraction,
strict compatibility, pure Kernel decoding, boundaries/clipping, ignore rules,
one-to-one matching, full denominators and configuration isolation. The reused
validation set is exploratory, not an independent or official benchmark.

## Completed measurement, 16 October 2026

Run: `/extra1/bmartelli/context-str-runs/kernel-only-padding-v1-20261016/`.
All125 extracted Kernel probability maps are **exactly** equal to the frozen
source maps (maximum difference0.0); all threshold masks are identical.
The derived checkpoint has one output, with no training step performed.

| Padding native256 | Kernel localized padded | Kernel exact OCR | M+K localized padded | M+K exact OCR |
| ---: | ---: | ---: | ---: | ---: |
|0|400|249|648|524|
|2|622|483|605|476|
|4|529|422|408|313|
|6|386|303|257|195|
|8|298|231|150|102|
|12|147|107|64|42|

Denominator983 throughout. The unpadded localization counts remain400 for
Kernel and648 for M+K; padding changes boxes/crops, not segmentation or
component counts (976 Kernel proposals versus1038 M+K). The descriptive best
Kernel grid result at padding2 has OCR Hmean49.31%, below M+K padding0 at51.86%.
At equal padding2 Kernel has7 more OCR-correct words than M+K; it does not
exceed the reference's best observed configuration. No deployment choice or
independent superiority claim follows from the reused holdout.

GPU RTX3090, FP32 batch8: single-output segmentation peak2,226GiB allocated /
2,406GiB reserved; PARSeq peak0,138GiB allocated /0,164GiB reserved; no OOM.
11,782 new distinct crops were recognized;1,121 distinct qualified cache hits.
17 pertinent unit/regression tests pass, plus full CUDA inference and exact
M+K pad0 OCR reproduction. Independent NumPy IoU + SciPy Hungarian matching
reproduces all1,500 image/arm localization and OCR counts, with all125 samples
and983 GT retained per arm. Source/checkpoint hashes remain unchanged.
Qualification and full per-image results are in the run's `qualification.json`
and `results.json`; the lighter receipt is also recorded in provenance.
