# Frozen-model scale/crop consistency router

**Approximate reproduction / exploratory comparison.** User-authorized tests
of stability across YOLO640/960, PARSeq crop perturbations, predicted U-Net
coverage and CLIP/native confidence. All model weights and the original640
output words remain unchanged. This is an external section3.5 extension, not
the paper's Algorithm2. Local paper sections3.3/3.5 and existing frozen-router,
crop-view and scale-matching implementations were inspected before changes.

Protocol: `YOLO-FROZEN-CONSISTENCY-001` in `PAPER_AMBIGUITIES.md`; scientific
and host profiles: `configs/approximate/yolo_frozen_consistency*_v1.json`.

## Frozen inference probes

Reuse the fully audited125 TRAIN /300 TEST640/960 outputs, U-Net Text/Kernel
coverage and CLIP features, plus425 complete ViTAEv2-S+TextOCR fallback outputs.
Reused weights retain their original approximate provenance. Genuine PARSeq
CPU FP32 reads the missing TRAIN crop views. Reuse existing CPU FP32 TEST
views after verifying source contracts, geometry, original RGB image and crop
pixel hashes. Each final640 rectangle gets fixed5% shrink and5% expansion,
clipped to the original stored raster. No GT determines the views.

Scale matching maximizes the number of one-to-one equal casefold words at
bboxIoU>=0.5, then total IoU. Preserve punctuation and duplicate multiplicities.
The agreement score is2matches/(n640+n960), not actual ground-truth F1.
An empty reading set cannot bypass DeepSolo. Crop stability is the fraction
of640 words unchanged in both views; an additional soft control averages the
minimum normalized edit similarity across the two views. A consistently wrong
word can pass both checks. Perturbed or960 readings never replace the original
word predictions, so all changes in final performance come from routing.

Thirteen predeclared families compare native/CLIP and prior coverage controls,
individual scale/crop/coverage signals, confidence-weighted geometric means,
the geometric mean of all four checks and their minimum/AND rule. No learned
classifier or fitted score weights are introduced. All score formulas are
explicit in `recognition/training_free_consistency_router.py`.

## Label use and threshold protocol

An image is good if its actual local word F1>=0.8. This differs from word
precision, mean F1, overall system F1 and the desired fraction of accepted good
images. Targets80/90/95% specify that last fraction; primary target90%.

Use only the original TRAIN80 to choose a family by maximum empirical bypass
meeting the quality target, then higher good fraction, then declared order.
Independently calibrate every family's threshold on TRAIN45; require at least
five accepted calibration images. If no qualifying threshold exists, explicitly
abstain on every image. Thresholds are empirical, not formal risk certificates.
Freeze all choices before constructing new TEST consistency features.

Also report GT-free10/20/30/40/50% exact batch rank budgets and actual scalar
coverage-cut transfer. For each budget, separately select its family on TRAIN80
by good-image count, with ties retaining declared order. No TEST labels select
a family, threshold or budget. Report every family, including failures to find
a threshold and zero-bypass results, rather than selecting a TEST winner.

TRAIN125 is the old segmentation holdout, not all1,255 TRAIN images. DeepSolo
was fine-tuned on TotalTextTRAIN. Both populations and the TEST have been
examined in previous experiments, so no untouched evaluation claim applies.
The small45-image calibration set can give optimistic empirical guarantees.

## Metrics, execution and reproducibility

Keep all983 TRAIN /2,204 TEST care words; existing within-image exact-casefold
multiset matching, punctuation retained, no IoU, evaluation-only wholly-invalid
native256 ignore. Whole-image routing keeps every original640 prediction or
every frozen DeepSolo output. Report good-image precision/recall and binomial
intervals, accepted/fallback images, TP/FP/FN, emitted care words, final F1 and
TN0. Per-image ground-truth F1 is evaluation/calibration only.

Additional OCR uses the existing isolated YOLO environment, CPU FP32 batch32,
four threads, deterministic operations and no optimizer. Record actual shapes,
model state SHA256 before/after, optimizer_steps0, process/GPU inventory and
propagated failures. CPU allocated/reserved CUDA fields are null. Existing
YOLO/CLIP/U-Net/DeepSolo stages are reused, so this is not a full-pipeline speed
or new8GB GPU qualification. Install no packages. Preserve partial failures.

```bash
.venvs/context-str-yolo-py311/bin/python -u -m scripts.run_yolo_frozen_consistency --mode approximate --check
.venvs/context-str-yolo-py311/bin/python -u -m scripts.run_yolo_frozen_consistency --mode approximate --smoke
.venvs/context-str-yolo-py311/bin/python -u -m scripts.run_yolo_frozen_consistency --mode approximate
```

Source plans, input hashes, code snapshots, views, per-image features, frozen
calibration choices, results, CSV and Markdown reports are external artifacts.
Six focused tests and the initial genuine smoke passed. Full measurements and
the independent audit are pending; no improved routing result is claimed yet.

## Completed experiment,2026-10-21

All125 TRAIN and300 TEST images completed. New TRAIN inference produced2,044
PARSeq crop readings; TEST reused4,410 existing readings. Model-state SHA256
before/after is identical, with zero optimizer steps. All source hashes and
inference receipts are retained; no new model, install, precision change or
word correction occurred. CPU FP32 probe stage including model load and state
hashing took51.14seconds; this is not a full-pipeline speed comparison.

At fixed TEST bypass budgets, the predeclared `all_checks_softcrop` combination
uses confidence, cross-scale agreement, coverage and soft crop stability:

|TEST bypass|Accepted|Prior coverage gate: good images|Combined checks: good images|Prior final F1|Combined final F1|Combined final TN0|
|---|---:|---:|---:|---:|---:|---:|
|20%|60|39/60 (65.00%)|48/60 (80.00%)|79.8346%|79.9633%|66.6157%|
|30%|90|54/90 (60.00%)|67/90 (74.44%)|79.0698%|79.3840%|65.8155%|
|50%|150|85/150 (56.67%)|91/150 (60.67%)|77.5954%|76.8910%|62.4577%|

These are a specified combination's exploratory ablation results, not a family
chosen from TEST as a deployment winner. The small positive final-F1 changes
at20/30% are not established improvements: paired-image10,000-draw conditional
95% intervals are[-0.3764,+0.6638]pp and[-0.3636,+1.0391]pp. Good-image precision
changes are+15.00pp interval[+3.8664,+26.5161] and+14.4444pp interval
[+6.4515,+22.5919]. No multiplicity correction; fixed gates, previously explored
TEST and comparison-selection limits apply. At50%, good-image precision rises
slightly but final microF1 decreases: macro per-image quality does not weight
word counts, and a good light branch can still be worse than DeepSolo.

For actual transferred scalar thresholds aiming for90% good accepted images:

|Family|Threshold|TRAIN45 good/accepted|TEST accepted|TEST good/accepted|Final TEST F1|Final TEST TN0|
|---|---:|---:|---:|---:|---:|---:|
|Prior coverage gate|0.402705042|9/10|96|60/96 (62.50%)|79.0698%|65.3846%|
|Confidence + scale|0.514547276|9/10|80|54/80 (67.50%)|79.2705%|65.6597%|
|Confidence + crop|0.549300691|11/12|97|59/97 (60.82%)|78.7303%|64.9217%|
|Confidence + coverage|0.546989517|9/10|81|57/81 (70.37%)|79.3103%|65.7143%|
|All checks, exact crop agreement|0.693130267|7/7|61|48/61 (78.69%)|79.8806%|66.5010%|
|All checks, soft crop agreement|0.696978117|8/8|63|50/63 (79.37%)|79.8622%|66.4755%|

Neither combination meets its90% TEST good-image target. These thresholds were
calibrated independently on the45 TRAIN images; seven/eight error-free examples
do not certify future reliability. Increasing the target to95% keeps the same
thresholds for these combinations, rather than producing a stronger guarantee.
Isolated CLIP/scale/crop/coverage signals find no minimum-five calibration
threshold at90%, so their scalar policy explicitly falls back on all images.

The **family-selection procedure itself does not generalize successfully**.
For primary target90%, TRAIN80 chooses `native_clip`; its calibrated threshold
0.397138349 accepts70 TEST images, only36 good (51.43%), final F1 79.1290%.
Target80% chooses confidence+coverage, target95% chooses native_clip; these
are retained in results, not replaced by the more favorable TEST ablations.
TRAIN80-selected families at fixed budgets also fail to choose the favorable
combination consistently. A new independent validation is needed before
selecting a deployment policy; no score/split/threshold was adjusted from TEST.

Seventy-nine focused tests pass, including cardinality matching, duplicate and
punctuation handling, exact F1 boundary, unchanged state, failure propagation,
threshold ties and refusal to recalibrate after TEST features exist. Genuine
smoke and full independent audit pass. The audit checks425 feature records,
6,454 original RGB crop-view hashes, sparse graph matching, every score formula,
all family/threshold choices and676 metric/routing points. No sample failed or
was omitted; no real OOM occurred. State/CPU-memory receipts remain explicit.

Run:
`/extra1/bmartelli/context-str-artifacts/yolo-frozen-consistency-v1/full/acb9b8f9d1cc`.
Plan SHA256:`2545f239c2a426793023a2ec55d905ca904c4d70825f222546aa9ef210925128`.
Results SHA256:`aa72ced5f364090b84b390064f9f7c35d7e7feb1eec696def4a4f2519576478e`.
Detailed tables/curves: `report.md`, `comparison.csv`, `results.json`;
`independent_audit.json`, `paired_comparisons.json` and `train/model_receipt.json`
preserve verification, uncertainty and frozen-state evidence.

```bash
.venvs/context-str-yolo-py311/bin/python -u -m scripts.audit_yolo_frozen_consistency --run /extra1/bmartelli/context-str-artifacts/yolo-frozen-consistency-v1/full/acb9b8f9d1cc
```

Compared numerically with paper Table5 ViTAE79.8, the20%-budget combined F1
79.9633 is+0.1633pp. This is not a reproduced/superior paper result: evaluator,
care/ignore, threshold and protocol ambiguities remain. Extra probes also cost
inference, so reduced DeepSolo calls alone do not establish lower runtime.
