# Essential router transfer to ICDAR2013 and ICDAR2015

Date: 2026-10-28. **Approximate reproduction / cross-dataset experiment.**
The active requests are to test the essential router on U-Net and YOLO across
Total-Text, ICDAR2013 and ICDAR2015, then examine accuracy/fallback frontiers.
The [threshold analysis](ESSENTIAL_ROUTER_THRESHOLD_ANALYSIS.md) contains all
numerical comparisons, validation choices, TEST-only optima and uncertainty.
STRICT components, checkpoints and defaults remain unchanged.

## Fixed experts and full data

Keep the same Total-Text Text/Kernel ResNet50 attention U-Net, pre-OCR selector,
pointwise selector, GNN, native PARSeq visual verifier and text editor. None is
fine-tuned on ICDAR images. The new prediction-only deployment adapter uses the
source image transform, component/split/merge pool, unique-crop budget 20,
128-resolution decoder feature pooling, pretrained PARSeq columns and frozen
admission/edit policies. It accepts original RGB only; GT and validity masks
enter evaluation afterward. Three authenticated Total-Text controls reproduce
all selected IDs, texts, geometry and crop hashes, with bit-identical masks and
maximum confidence difference below 1e-5. This small replay establishes the
ported path's compatibility, not an independent quality result.

YOLO is the existing frozen TextOCR YOLOv8s direct640 detector and genuine
PARSeq, joint confidence 0.60 and same-text NMS IoU 0.5. Complete ICDAR TEST
outputs already existed and are reused with dataset/image/crop provenance.
Only validation requires new YOLO/PARSeq inference. The 30-feature gate still
needs the shared U-Net predicted text/kernel masks for residual-coverage signals.
The YOLO branch is therefore not a detector-only gate.

DeepSolo is the same strict-loaded native ViTAEv2-S/TextOCR Total-Text candidate
used by the earlier essential study, with the previously TRAIN-selected
instance threshold 0.35 and character confidence 0.925. Its RGB resize,
checkpoint and isolated legacy Python3.8 runtime remain fixed. This is a
cross-dataset transfer of those artifacts; it is not an ICDAR-specific adapted
DeepSolo checkpoint or exact paper reproduction.

New TEST inference includes all 233 ICDAR2013 images / 1095 care words and all
500 ICDAR2015 images / 2077 care words. Total-Text retains its complete cached
300 images / 2204 care words. No failed or empty image is removed. ICDAR2013
uses the qualified Task1 TEST population, which differs from the paper's
unresolved 229-image recognition split. Validation uses the existing 23-image
ICDAR2013 holdout and the existing 100-image ICDAR2015 holdout. Both are
separate from TEST, and target-domain labels never fit expert or gate weights.

## Comparable gate fitting and separate controls

The primary historical essential head was trained on corrected U-Net outputs.
Its literal transfer to native YOLO changes the confidence-feature distribution;
that complete negative/control study is preserved. A separate matched YOLO
head removes this obvious expert mismatch: same 30 descriptors, StandardScaler,
Ridge alpha 1, original Total-Text FIT 1028, paired native DeepSolo supervision,
word-count normalization and word-count sample weights. There is one fixed
candidate and no hyperparameter search. Learned weights are sealed before new
TEST scores are prepared; all cutoff choices use validation only. Both heads
contain 31 coefficients/intercept and 60 normalizers, 728 FP64 numerical bytes.
Model JSON, source fitting arrays and label/asset bindings stay outside Git.

The dedicated YOLO fit is explicitly an exploratory follow-up after seeing the
literal-transfer outcomes. Previously exposed TEST data are not called blind.
Independent weighted-Ridge normal equations reproduce coefficients to 1.30e-13;
all 1028 fitting inputs/targets and 856 ICDAR validation/TEST scores replay.

The threshold runner then separates:

- Source Total-Text cuts transferred unchanged, plus target-validation quantile
  cuts bound to the same fixed head; these need not transfer a fixed budget.
- Per-domain validation F1 constraints, maximum F1, shared-validation cut and
  an approximate simultaneous bootstrap lower-band control.
- Complete TEST frontiers and common-cut existence checks, explicitly post hoc.
- A normalized-score exploratory control, without new learned weights.
- Privileged GT-oracle image choices, explicitly nondeployable.

`validation_deployments` exports only the validation-selected policies. Its
`recognition/essential_threshold_deployment.py` interface takes predicted OCR,
image dimensions and predicted maps. It excludes TEST-optimal cutoff values.

## Exact local metric

Unicode casefold followed by `isalnum()` removes symbols and whitespace without
NFKC, accent removal, lexicons or fuzzy acceptance. Per-image repeated words
are matched exactly; empty/symbol-only predictions and care targets remain
in denominators and never match. TP/FP/FN aggregate over every image, with
`F1=2TP/(2TP+FP+FN)`. Primary lexical F1 has no spatial-IoU constraint.

Evaluation-only validity and predicted footprints are center-padded to a square,
then resized nearest-neighbor to 256. A nonempty predicted footprint entirely
outside valid pixels is ignored; a footprint vanishing at native resolution
remains care. U-Net/YOLO use their original-frame crop rectangles; DeepSolo
uses its original-frame boundary polygon. Invalid DeepSolo envelopes remain
unmatchable FP candidates. ICDAR2015 excludes official `###` GT, retaining all
other words. ICDAR2013 Task1 TEST has no ignored GT; its validation uses the
existing Task2.2 word groups, glyph-pixel boxes and commented-color ignore
parser, rather than inventing text grouping. This is the inherited local
approximate metric, not the official benchmark evaluator. The RRC
[official FAQ](https://rrc.cvc.uab.es/?com=faq) distinguishes its benchmark
word/ignore protocols; no official score equivalence is inferred here.

## Execution and measured memory

Machine: PAPER as bmartelli, actual RTX3090, FP32, no AMP/TF32, explicit batches,
separate legacy DeepSolo and modern inference environments, and sequential
model stages. No package installation, checkpoint download or training change
occurs in the expert path. Main environment torch 2.4.1+cu124; DeepSolo uses its
qualified legacy stack. Other shared-GPU jobs are preserved.

The successful DeepSolo TEST export has 729 new images plus four reused smoke
images, totaling 733 physical TEST inferences. Its separate validation export
has 123. These 856 cache-generation calls are shared by every subsequent router
analysis; each logical router result separately reports its accepted/fallback
image count. There is no measured complete online latency or end-to-end speedup.

Allocated/reserved byte peaks:

| Model set and phase | Shape / batch | Allocated | Reserved |
| --- | --- | ---: | ---: |
| Corrected U-Net/PARSeq/GNN/verifier/editor, target export | RGB512/native256, U-Net1, PARSeq32 |629672448|897581056|
| YOLO validation detector | direct640, batch1 |167702016|216006656|
| YOLO validation PARSeq | RGB32x128, batch32 |205957120|297795584|
| Native DeepSolo TEST | shortest1024/max1892, batch1 |2117065728|8841592832|
| Native DeepSolo validation | same, batch1 |1930020352|3628072960|

The reserved peak includes the allocator history of a persistent export process;
it is not the model's parameter footprint or an 8 GB pipeline qualification.
No successful-run image OOM occurred. Shared expert checkpoint weights are
excluded from the 728-byte gate comparison.

## Verification and preserved failures

The full independent audit verifies all 733 ICDAR image/annotation identities,
5496 selected crop hashes, original-frame ignore rendering and independent SciPy
word matching. It independently replays gate observations, all attainable score
cuts and every validation objective, and checks bootstrap bands through the
exact F1-gain identity. Exported single-image profiles and exact GT-oracle
optimality certificates have a separate audit. Focused tests include source
metric boundaries, tied cuts, empty observations, odd rectangular letterbox,
portable policies and exhaustive small-instance oracle comparison.

Preserved engineering history:

- Initial DeepSolo startup failed before inference because torch 1.9 lacks
  `cuda.mem_get_info`. The compatible worker reads the single GPU's free memory
  with nvidia-smi; no scientific parameter or dependency changed.
- The first small frontier fixture incorrectly expected all-light to preserve
  baseline F1. Exact fraction comparison showed one fallback is necessary; the
  fixture expectation was corrected, keeping the constraint unchanged.
- The normalized follow-up initially stopped at a missing helper import after
  an interrupted patch. Its import-only log remains; the successful study is
  a fresh normalized_v2 output, without threshold or tolerance relaxation.
- The first two independent audit attempts omitted centered letterbox padding
  from their native-ignore reconstruction. ICDAR2015 image2 exposed one ignored
  YOLO output as the mismatch. The auditor now independently implements the
  existing center-pad transform; production predictions/counts are unchanged.

The executed three-image qualification script is preserved separately. Its
maintained CLI subsequently requires an explicit `--trace` path, removing the
historical source lookup from implementation defaults; the model adapter and
qualified outputs are unchanged.

## Artifacts and code

External root:
`/extra1/bmartelli/context-str-runs/essential-router-icdar-transfer-20261028-v1`.
It contains the frozen population/config, all U-Net probability/word traces,
YOLO validation caches, native DeepSolo shards, evaluation records, portable
heads/profiles, all analysis variants, source snapshots and audits. All
previous failures and source studies remain preserved.

Implementation: `recognition/corrected_unet_transfer.py`,
`scripts/qualify_corrected_unet_transfer.py`,
`scripts/export_essential_transfer_deepsolo.py`,
`scripts/run_essential_icdar_transfer.py`.
Configuration: `configs/approximate/essential_router_icdar_transfer_v1.json`.
Tests: `tests/recognition/test_essential_icdar_transfer.py`.
Scientific records: `ESSENTIAL-ICDAR-TRANSFER-001` and
`ESSENTIAL-THRESHOLD-FRONTIER-001` in `docs/PAPER_AMBIGUITIES.md`.
