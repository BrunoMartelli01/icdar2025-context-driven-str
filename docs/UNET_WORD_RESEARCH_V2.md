# Corrective U-Net experiments v2

All results are **approximate reproduction / experimental extensions**. The
authoritative local paper, sections3.2–3.3 and Algorithm1, specifies the backbone
family and component localization, not these extra heads or grouping rules.
Scientific choices are recorded under UWR-002 in `PAPER_AMBIGUITIES.md` and in
`configs/approximate/unet_word_research_v2.json`. Historical v1 files, datasets,
checkpoints, and STRICT paths are preserved.

## What is compared

The source for newly trained branches is the completed v1 **control best epoch2**,
SHA256 `47a941e17472d26c4fff259d0a81c061e23040059c49d97cf2a89a9430045cd8`.
Every original tensor is copied strictly. A frozen reference and a50-epoch
continuation control separate the effect of new heads from extra training.

| Arm | Intervention | Checkpoint selection |
| --- | --- | --- |
| `reference_control` | Frozen source, actual evaluated checkpoint; zero training epochs | Qualified source |
| `control` | Text/kernel continuation, fresh Adam | Native word bbox F1 |
| `glyph_isolated` | Private two3x3 layers + output on64-channel decoder and64-channel early feature; all source tensors/BN frozen | Glyph foreground micro pixel F1, word F1 tie-break |
| `glyph_staged` | Same private branch;10 epochs frozen source,40 joint epochs; glyph weight ramps over10 joint epochs to.1 | Native word bbox F1 |
| `links_isolated` | Eight pixel-neighbor logits; all source tensors/BN frozen | Native word bbox F1 |
| `threshold_frozen` | Private adaptive threshold; original full-word probabilities/kernel frozen | Native word bbox F1 |
| `db_core` | Private residual core/threshold head; source word/kernel and BN frozen | Native word bbox F1 |

Every trainable branch includes epoch0 in selection. Glyph masks are binary
foreground strokes, **not individual character IDs or classes**. No character
labels are manufactured from connected components. Training uses the existing
checksum-qualified pairing and exact replay of augmentation.

Shared LR is1e-5; private LR is1e-4; Adam weight decay.01. Both parameter groups
exist before warmup so their state can resume across unfreezing. Frozen parameters
have no gradients and frozen source BN never updates. A first-batch diagnostic
records unweighted shared word/kernel and glyph gradient norms and cosine at
the configured final decoder convolution, alongside the actual ramp coefficient.
Loss magnitudes are not presented as proof of gradient conflict.

An isolated glyph branch can improve glyph supervision without improving word
grouping. Its existing conservative association preserves word IDs, including
their limitations; both word-head-only and combined geometry are reported. Two
independent complete U-Nets are not part of this experiment.

## Fixed-checkpoint embedding extraction

Before new training, the suite evaluates **v1 embedding last epoch50**, SHA256
`0dfaced3ac8fb1f5708da0d7b6c256c26aac9a7e1c4fd7174f2d820859c2130d`.
This differs from the v1 *selected best*, which was epoch0. All three arms receive
the exact same word/kernel/embedding arrays in one inference pass:

- `embedding50_hard`: original4-neighbor radius-gated aggregation and fallback.
- `embedding50_watershed`: original8-neighbor confidence watershed.
- `embedding50_soft`:8-neighbor competitive minimax flood, seed-specific distance
  penalty `weight*d/(scale+d)`, with weight.25 and scale1.5; no distance rejection.

Each seed retains its ID. Soft growth reaches every pixel of a seeded foreground
component; only genuinely unseeded components need fallback IDs. Missing text
cannot be crossed. This is greedy competitive growth, not a claim of globally
optimal label-dependent paths or an exact PAN reproduction. Zero penalty weight
is exactly the original watershed. Probability foreground must agree bit-for-bit
across all three arms. No area filtering, padding, or crop cap enters comparison.
Parameter changes need a new output directory; these settings are assumptions,
not tuned on official TEST. A source epoch50 checkpoint is verified against both
the completed v1 receipt and its final logged checkpoint checksum.

## Pixel-link supervision

Eight directions are ordered NW,N,NE,W,E,SW,S,SE. Same-positive-owner neighbors
are positive; cross-owner/background neighbors are negative. Only valid positive
word anchors and valid neighbors contribute. Each word receives inverse-area
weighting; positive and negative link BCE are normalized separately within an
image and summed, then image-averaged. No-supervision batches skip optimizer
steps, including weight decay. Ignore pixels and image-edge wraparound never
produce targets.

Inference uses positive reciprocal links (both directions>=.5) between foreground
pixels to form integer instance components. Every foreground pixel survives,
including isolated ones, and no GT is passed to localization. This can still
fragment words if links are poorly calibrated; raw counts/splits/merges and
downstream OCR reveal that rather than hiding fragments with an area filter.
It is a PixelLink-inspired head on this backbone, not the original VGG16/OHEM
training or benchmark protocol.

## Threshold isolation and DB-style core targets

`threshold_frozen` starts with threshold.65 and trains only its private branch
using the original declared full-word binary objective and border L1. It measures
whether thresholds themselves help when the probability maps cannot drift.

`db_core` has coherent shrunken-core probability **and** binary targets. Per-owner
offset is `A*(1-r*r)/L`, r=.4; A is native component pixel area and L is its4-edge
raster perimeter. Euclidean distance implements shrink/dilation; this is an
explicit raster adaptation instead of DBNet's polygon/Vatti operations. If shrink
removes a surviving word, its farthest native pixel is retained and counted.
Border thresholds range.3–.7; L1 applies in the expanded band with weight10.
Probability/binary loss is the declared BCE+Dice, without DBNet OHEM. The initial
probability is the frozen original **kernel** logit plus a zero residual; initial
threshold.5 preserves the original nuclei decision.

Predicted DB cores seed the same watershed inside the frozen full-word support;
this measures separation without changing text recall support. It is not a full
DBNet detector, polygon unclip, or strict paper component. Original GT is never
rewritten. New targets and head evidence are exported for auditing.

## Evaluation and execution

Frozen official TRAIN split1130/125, seed202503, RGB512/native256, FP32/batch8,
50 additional epochs. A CPU smoke uses explicit separately recorded batch/worker
overrides. All983 care GT words remain denominators. Native geometry uses maximum
one-to-one bbox IoU>=.50; merge/split use any shared pixel and divide by983.
Wholly ignored predictions are excluded after localization; partially ignored
boxes remain complete. Empty foreground/GT has perfect pixel score. OCR uses
original RGB boxes, pad0, the qualified existing approximate PARSeq artifact,
AR=True/refine_iters1, and original-coordinate one-to-one bbox IoU>=.50 plus exact
casefold. Unicode punctuation/symbol stripping is a separate secondary result.

Run on the GPU host with a **new** output directory:

```bash
bash scripts/run_unet_word_research_v2_host.sh \
  artifacts/totaltext-curved-word-boxes-v4 \
  /extra1/bmartelli/datasets/context-str/scene-segmentation-v1 \
  /extra1/bmartelli/context-str-runs/unet-word-research-v1 \
  /extra1/bmartelli/context-str-runs/unet-word-research-v2 \
  /extra1/bmartelli/context-str-artifacts/parseq
```

The foreground, locked, restartable wrapper performs paired/native-target
qualification; fixed-checkpoint extraction; PARSeq and
`postprocess_comparison.md/json`; then7 reference/training/export stages; finally
PARSeq and `comparison.md/json`. Segmentation and recognition models are resident
in separate processes. Source checkpoint paths in frozen receipts may be absolute
read-only references; their cryptographic hashes must still match on replay.
CPU and GPU OCR caches cannot mix: device, batch, precision, runtime/checkpoint,
software/backend and recognizer-source settings form an immutable cache contract.
All per-image masks, targets, crop hashes and raw texts survive in exported files.

Compare isolated heads to `reference_control`, staged training to both the frozen
reference and `control`, and embedding extraction **only** to its same-checkpoint
controls. Frozen-reference versus continued-control differences are not a quality
head effect. Word F1 and glyph pixel F1 are different metrics. The repeated TRAIN
holdout and one seed do not establish generalization; official TEST is untouched.

CUDA is inaccessible in the implementation session. CPU execution does not
qualify GPU fit: allocated/reserved peaks, actual GPU name/model set/input/batch,
OOM and failure metadata are recorded when the host runs the suite. No automatic
precision/batch/model fallback is provided. Historical v1 ran on a recorded RTX3090;
that is not a qualification of the target8GB RTX3070. Current receipts document
tests/smokes and exploratory CPU extraction separately from new full training.

Sources: local `papers/2503.15639v2.pdf` sections3.2–3.3,
[PAN](https://arxiv.org/html/1908.05900v2),
[PixelLink](https://arxiv.org/html/1801.01315v1),
[DBNet](https://arxiv.org/html/1911.08947v2).

## First qualified CPU extraction result

The implementation qualification passes116 focused/related tests, seven real
FP32/batch2 CPU smokes (including the glyph joint transition), and the1255-image
paired-target preflight. Held-out125 retains983/983 words, with983 DB cores and
zero farthest-pixel rescues. The one known native-resolution loss remains on TRAIN.
No test or smoke failure was observed; no package installation or GPU execution
was performed. See `provenance/unet_word_research_v2_qualification.json`.

On identical v1 embedding epoch50 maps, replayed on CPU FP32/batch4, with real
qualified CPU PARSeq/batch8 and all125 images/983 GT words:

| Extraction | Native word F1 | Split words | Exact OCR/all983 | OCR gained/lost vs hard |
| --- | ---: | ---: | ---: | --- |
| Original hard aggregation | 50.73% | 273 | 497 | +0/-0 |
| Original watershed | 61.10% | 98 | 499 | +6/-4 |
| Soft embedding cost | 61.39% | 109 | 493 | +6/-10 |

Removing residual fragments improves proposal precision/F1 but does not establish
an OCR gain for the soft cost; watershed gains only two net words in this study.
Soft parameters were kept unchanged after observing results. These rows compare
the same epoch50 source, not the v1 control best or embedding selected epoch0.
All three preserve identical foreground pixel IoU/F1. Raw outputs and per-image
gains/losses are in `/tmp/unet-word-v2-postprocess-cpu-v1/`, with
`postprocess_comparison.md/json` and `postprocess_ocr_comparison.json`. New-head
full training, independent generalization and physical CUDA memory fit are pending.
