# Three tests: from V3 masks to readable crops

2026-10-21 — **approximate reproduction**, exploratory research.
Launched on PAPER with user authorization. TEST and TRAIN holdout have already
been explored; these experiments are neither independent validation nor
STRICT/Table5 reproduction. Decision: `MASK-CROP-TEST-001`.

External run:
`/extra1/bmartelli/context-str-runs/mask-crop-three-tests-20261021-v1/`.

## Frozen protocol

GuardedV3 U-Net epoch 447, selected on TRAIN validation bboxWordF1,
SHA256 `5b44de7b88b5b1d636e8f53c586565f6491aabaae3ccea361e96e32d540c1faa`.
U-Net/PARSeq remain frozen. RGB256, native 128 mask, stretch geometry, threshold 0.5,
no area filter, native padding1, historical expanded-bbox ordering. Qualified
PARSeq, AR+refine1, same checkpoint/PIL bicubic32x128 transform in every comparison.

Primary metric: maximum one-to-one **unpadded** bbox IoU>=0.5 and exact Unicode
casefold text, retaining punctuation. Every prediction/care annotation counts,
including proposal-free images; no prediction neutralization on ignore.
Micro precision/recall/F1. Refinement keeps original localization boxes fixed;
only RGB crops read by PARSeq change. Gains cannot arise from changing the
box used to evaluate the same text.

## Test1: crop budget — complete

300 TEST images,2204 words. Actual OCR on additional crops, not geometric
estimation. Identical crops retain identical frozen PARSeq readings.

| Limit | Crops | Correct | Precision | Recall | F1 | Gained/lost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 2086 | 795 | 38,11% | 36,07% | 37,06% | 0/0 |
| 20 | 2552 | 876 | 34,33% | 39,75% | 36,84% | 81/0 |
| All | 2688 | 904 | 33,63% | 41,02% | 36,96% | 109/0 |

The602 extra proposals require568 new unique readings; some already existed in
frozen caches from other padding variants. Removing the cap recovers109 words
but adds 493 proposals without more exact matches: F1 essentially unchanged,
precision lower. This motivates better selection, not automatic unlimited-budget
promotion. Paper cap10 remains the reference.

Independent audit PASS:1800 matching cardinalities rechecked with SciPy,
complete denominators, baseline795 replay, verified source hashes.
Results: `budget/results.json`, `budget/per_image.json`, `budget/audit.json`.

## Test2: splits/merges and selector

Every original crop remains an alternative. Generate groups of 2–4 parts using
high-threshold seeds, geodesic-reconstruction erosion, and RGB/probability seams;
neighbor-pair merges reconstruct possible fragments. Existing generators retain
correct native 128→RGB stretch mapping. Maximum64 crops/image; GT-free generation.

A small correctness classifier uses geometry, mask support, PARSeq text/confidence,
and proposal type. Train on 1130 TRAIN images with incomplete, multiword, poorly
localized negatives. Crop cost/group replacement margin are chosen on 125
TRAIN holdout images (983 care words), including identity policy. TEST selects
nothing. The comparison saves both potential pool words and emitted words:
the former is a diagnostic, not a selector.

## Test3: recognition-supervised refiner

A small CNN receives RGB+probability context and initial crop, predicting bounded
translation/scale. Identity initialization: crop RGB hashes/OCR count must exactly
reproduce baseline before training. Loss: causal PARSeq cross-entropy with EOS,
GT coverage, neighbor contamination, geometry regression. Gradient from PARSeq loss
alone to coordinates is explicitly checked on the first real batch.

At most4096 seeded TRAIN proposals,8 epochs, batch 16; charset/length/IoU exclusions
apply to training examples only, never evaluation denominators. Select best
**learned** epoch using real OCR on the complete holdout125. Record identity
and accepted holdout policy too, exposing regressions. Before/after PARSeq tensor
hashes must match; its parameters receive neither gradients nor updates.

Differentiable bilinear sampling is a training proxy; evaluation always uses
integer boxes on original RGB and qualified PARSeq preprocessing. Transfer
validity is an experimental question, not an equivalence.

## Complete test2/3 results

Both finished, all 8 refiner epochs completed. Independent audit PASS:4800 SciPy
matching cardinalities, populations/denominators, weight/code hashes, selections
frozen before TEST, fixed detection boxes.

| Variant | Crops | Correct | Precision | Recall | F1 | Gained/lost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 2086 | 795 | 38,11% | 36,07% | 37,06% | 0/0 |
| **Test2: selected splits/merges** | **2190** | **870** | **39,73%** | **39,47%** | **39,60%** | **92/17** |
| Test3: learned refiner, epoch 7 | 2086 | 768 | 36,82% | 34,85% | 35,80% | 32/59 |
| Test3: holdout-accepted policy | 2086 | 795 | 38,11% | 36,07% | 37,06% | 0/0 |

The selector chooses189 replacements:78 threshold splits,13 erosion splits,
40 seam splits,58 merges. Cost0.15/margin 0 were selected on TRAIN holdout,
where correct words improve320→365 (F1 35,01→39,00%). TEST net benefit is75
words and 2,54 F1 points, with higher precision/recall. Full pool contains1032
potential exact matches in5537 proposals: GT-assisted diagnostic, not accuracy
of an available selector.

Refinement does not beat holdout baseline: best learned epoch 7 gives309 correct
versus 320, so the accepted policy retains original crops before observing TEST.
Its negative learned result768/2204 is still reported. TRAIN loss decreases,
but32 TEST recoveries do not offset59 losses. Bilinear/teacher-forcing to real
AR+refine1 transfer is a possible factor to isolate, not established causation.
This refiner should not be promoted in its current form.

The audit also finds **two identical image files across local TRAIN/TEST**:
TRAIN212↔TEST578 and TRAIN81↔TEST656, SHA256-verified. Primary results retain
all 300 images; no original data/protocol changes. Secondary sensitivity on
remaining298 images/2187 words retains test2 benefit:782→858 correct,
F1 36,73→39,35%; refiner stays negative755. This is neither a new official
evaluation nor exhaustive near-duplicate search. Sources/cases/counts:
`duplicate_image_sensitivity.json` in the external run.

Complete comparison: `comparison.md`, data `comparison.json`, check
`independent_audit.json` at run root. No automatic production-pipeline/STRICT
configuration promotion.

## Execution and monitoring

Test2/3 outputs: `learned/`; current status `learned/status.json`, log
`learned.log`, pre-TEST frozen choices `learned/choices_frozen_before_test.json`,
final results `learned/results.json`. Full job needs no extra TEST-launch command.
`launch.json` records launch, not current process status.

Code ran from isolated copies `source_budget_v2/` and `source_learned/`, with
complete SHA256 manifests. Initial budget launch failed before inference because
`pipeline` was omitted; failed log/receipt remain preserved. Complete copy passed
imports; subsequent budget run is complete/verified. Test2/3 smoke passed from
the isolated copy before full launch; this is not efficacy evidence.

Actual GPU NVIDIA RTX3090/24GiB, FP32, no AMP/TF32, models loaded by stage;
batch 1 U-Net,32 OCR,16 refiner. Each stage records allocated/reserved peaks and
reports OOM/errors. Other jobs remain active: do not compare these timings with
exclusive-GPU benchmarks. CUDA `grid_sample` backward is explicitly
nondeterministic during training; seeds fixed, deterministic inference restored.
No installation/download.

All paths/hashes/parameters:
[`mask_crop_three_tests_v1.json`](../configs/approximate/mask_crop_three_tests_v1.json).
Repeat with new output paths and explicit mode/config:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
.venvs/context-str-main-py311/bin/python -u -m scripts.run_mask_crop_budget \
  --mode approximate --config configs/approximate/mask_crop_three_tests_v1.json \
  --output /path/to/new/budget

CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
.venvs/context-str-main-py311/bin/python -u -m scripts.run_mask_crop_learned_tests \
  --mode approximate --config configs/approximate/mask_crop_three_tests_v1.json \
  --output /path/to/new/learned
```

46 focused tests PASS, all three GPU-path smokes PASS, real recognizer-to-crop
gradient PASS. Earlier scientific review:
`artifacts/diagnostics/mask-to-crop-research-20261020/README.md`.
