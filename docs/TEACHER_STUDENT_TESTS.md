# Five teacher/student and selection pilots

2026-10-22. **Approximate reproduction / exploratory short-budget pilots.**
These are user-authorized research extensions, not exact Das reproduction,
official polygon spotting, paper Table 5, or the complete context/DeepSolo pipeline.

## Evidence inspected

The local original paper `papers/2503.15639v2.pdf`, sections 3.2–3.5 and
Algorithm 1, specifies the segmentation/localization/recognition framework.
Das et al., *Soft set-based MSER end-to-end system for occluded scene text
detection, recognition and prediction*, KBS305(2024)112593, page10, describes
pseudo boxes and an EMA teacher. Its full decay/filter/training recipe is
unresolved. The warm start and every numerical choice here are assumptions.
Related primary references: [Mean Teacher](https://arxiv.org/abs/1703.01780)
and [SemiETS](https://arxiv.org/abs/2504.09966). The latter motivates spatial/text
agreement; this suite does not reproduce its architecture or reported gains.

Before implementation, existing M/K, Gap, PARSeq decoding, word/kernel targets,
the four occlusion tests, failed new feature-OCR heads and YOLO word detector
qualification were inspected. No strict/default scientific path changes.

## Populations and evaluation

Reuse the checksum-qualified four-test source run and its 1030-fit,
100-calibration, 125-internal-evaluation and 300-TEST populations. Source M/K
already saw calibration during historical training; evaluation and TEST have
been repeatedly explored. The two known identical TRAIN/TEST RGBs remain in
the primary denominator. These are not independent generalization results.

The primary metric is micro maximum one-to-one unpadded original bbox IoU>=0.5
AND exact Unicode casefold transcription, preserving punctuation. All care GT,
empty readings and duplicate predictions count. Predictions are not neutralized
using ignore annotations. The primary TEST denominator is2204 care words.
No lexical-only or segmentation F1 substitutes for this measure.

All checkpoint/threshold selection uses calibration word F1; pretrained paths
retain epoch0 on ties. Final-epoch performance is reported separately, including
when calibration selects the initial model. No TEST-based combination or winner
promotion is allowed. Same fixed pre-OCR selector/budget20 and post-OCR policy
are reused for segmentation/recognition comparisons. YOLO has its own declared
score thresholds and cap100; its teacher-versus-student comparisons are matched,
while comparisons to M/K are descriptive because candidate budgets differ.

## Five experiments

1. **EMA/consistency:** three epochs of supervised M/K continuation, its EMA
   weights, and a separately trained EMA-consistency branch. Same fit images,
   initial state, order, photometric strength, optimizer and update budget.
   Deterministic cached letterbox RGB/targets receive a common random horizontal
   flip. Teacher uses the weak view; student receives brightness/contrast changes
   in the same coordinates. Foreground/background-balanced confident-probability
   MSE supplements the unchanged supervised text/kernel BCE+Dice. This is
   supervised consistency regularization, not a new unlabeled-data experiment.
2. **Clean-to-occluded recognition distillation:** pretrained PARSeq decoder
   continuation, with encoder frozen, on fit GT word crops. Both arms receive
   the same moderate vertical-strip RGB occlusions and three-epoch update budget.
   One uses AR teacher-forced CE; the other adds temperature-scaled token KL from
   the frozen clean-crop teacher under the same GT prefix. Padding is excluded,
   EOS included. This is not full PARSeq permutation training. Unsupported
   charset/length training words are listed, never removed from evaluation.
   Evaluation retains predicted M/K boxes and all care words. A synthetic
   occlusion benefit does not establish natural-occlusion reconstruction.
3. **Set-aware candidate scoring:** same feature MLP, fit pool, initialization,
   epochs, budget and inference rule; independent BCE versus BCE plus per-GT
   coverage and duplicate-occupancy penalties. GT-group assignments exist only
   during fitting. Both use GT-blind compatible-set selection and independently
   calibrated score thresholds. Original selector remains a third reference.
   The new factor is a set objective, not merely another pointwise classifier.
4. **Spatial/text-consistent pseudo-labels:** up to256 checksum-verified,
   hash-ordered ICDAR2015 TRAIN RGBs, with their annotations unopened. Fail on
   checksum overlap with any full source population, including smoke runs.
   Frozen M/K compares original versus contrast+horizontal-flip views, maps boxes
   back, then PARSeq reads the original RGB boxes from both geometries. Require
   one-to-one IoU/text agreement and character confidence in both readings.
   Only stable confident map pixels inside accepted word boxes become targets;
   unaccepted regions are unknown, not negative labels. The separate M/K branch
   adds one pseudo image per supervised update. Compare with matched supervised
   continuation; extra compute/data are explicitly part of this treatment.
5. **U-Net-free student:** warm-start qualified YOLOv8s word detector, five
   epochs GT-only versus GT+pseudo additional images. GT-only repeats fit images
   to equalize dataset size and optimizer updates. Same architecture, seed and
   augmentation requests; pseudo-data content is the treatment. Use the final
   epoch fixed in advance, then calibrate confidence on word F1. Trainer's
   mandatory internal final validation sees only fit images and never selects
   the scientific checkpoint. Frozen PARSeq recognizes its predicted RGB crops.
   Unlike the masked dense-consistency branch, the native YOLO box-only loss
   treats unannotated locations as background. Accepting this standard
   pseudo-label baseline is an `ASSUMPTION`: missed teacher words can reinforce
   false negatives. This pilot does not test an ignore-region-aware YOLO loss.

## Reproducibility and qualification

`configs/approximate/teacher_student_tests_v1.json` declares every path, artifact
hash, epoch, threshold and decision class. `localize/teacher_student_tests.py`
contains the isolated scientific primitives; `scripts/run_teacher_student_tests.py`
implements fitting/evaluation/auditing; `scripts/run_teacher_student_yolo.py`
uses the existing isolated Ultralytics8.1.0 environment. The launcher installs
nothing and propagates every failure. The full run uses an immutable source
snapshot and records its identity.
Focused unit tests: `tests/localize/test_teacher_student_tests.py`, alongside
the qualified occlusion, crop, multitask and PARSeq adapter tests. Launch with
`scripts/run_teacher_student_battery.sh`; both Python environments, configuration
and fresh output root are explicit arguments.

Actual RTX3090 hardware, FP32, no AMP/TF32, batch sizes, model sets and CUDA
allocated/reserved peaks are recorded. M/K student, EMA teacher and optimizer
remain resident while PARSeq performs calibration; the memory record names this
coexistence. Other experiment arms and the YOLO process run sequentially.
This is not RTX3070/8GB qualification.
Ultralytics controls its own deterministic/warn-only settings; both student
arms request identical settings. No claim of cross-version bitwise determinism.
Its pinned8.1.0 trainer maintains an internal EMA in both student arms and
serializes the final model/EMA tensors as FP16 even with AMP disabled. Inference
loads these tensors into FP32. Accepting that upstream checkpoint round trip is
an explicit `ASSUMPTION`, with possible output rounding effects; FP32 execution
must not be read as FP32 checkpoint serialization. The pretrained22MB source
also has FP16 tensors. The original resolved configuration remains immutable;
`serialization_provenance.json` records this clarification against its hash
before either full YOLO training arm starts.
`pseudo_label_supervision_scope.json` likewise records the different unknown-
region semantics of dense consistency and native box-only YOLO before training.

Each result saves per-image correct/localized GT IDs and all predictions, paired
recovered/damaged counts, per-image gains/losses, conditional image-bootstrap
intervals and a second SciPy matching audit. The one seed and short budgets do
not establish convergence or training variance. Failure JSONs and failed smoke
directories are preserved. Strict blockers remain unchanged.

External run root:
`/extra1/bmartelli/context-str-runs/teacher-student-tests-20261022-v1/`.
Completion and numerical outcomes must be read from the run's final report;
implementation or a successful smoke alone does not qualify a completed result.

## Completed results

All five methods and their controls completed:1030 fit images,100 calibration,
125 evaluation (983 care words),300 TEST (2204 care words), and256 additional
TRAIN RGBs. The teacher accepted498 word boxes in202 additional images. No full
run error or OOM. The final report's7650 independent image/arm matching checks
pass. All320 encoder tensors in each of four M/K endpoints and149 in each of
two PARSeq endpoints remain identical; learned heads/decoders actually change.

| TEST endpoint | F1 % | Reference | Delta F1 points | Recovered / damaged |
| --- | ---: | --- | ---: | ---: |
| Original M/K budget20 | 53.759 | — | — | — |
| New pointwise compatible-set filter | 60.701 | Original M/K budget20 | +6.942 | 0 / 68 |
| Set-objective filter | 60.566 | New pointwise filter | -0.135 | 4 / 3 |
| M/K supervised final epoch3 | 52.265 | Original M/K budget20 | -1.494 | 57 / 89 |
| EMA final epoch3 | 53.293 | M/K supervised final | +1.028 | 51 / 32 |
| EMA-consistency final epoch3 | 52.478 | M/K supervised final | +0.213 | 10 / 10 |
| M/K pseudo final epoch3 | 51.894 | M/K supervised final | -0.371 | 54 / 44 |
| PARSeq occlusion CE | 54.359 | Original M/K budget20 | +0.600 | 21 / 7 |
| PARSeq occlusion distillation | 54.230 | PARSeq occlusion CE | -0.129 | 4 / 7 |
| Initial YOLO/PARSeq | 64.826 | Separate architecture reference | — | — |
| Supervised YOLO/PARSeq | 68.956 | Initial YOLO/PARSeq | +4.130 | 204 / 49 |
| Pseudo-label YOLO/PARSeq | 68.807 | Supervised YOLO/PARSeq | -0.149 | 36 / 46 |

All M/K continuation arms select epoch0 on calibration, so their **selected**
endpoints remain53.759%; final-epoch rows above expose the actual trained
outcomes. Both OCR arms select epoch3. YOLO epochs are fixed5; calibration
confidence thresholds are initial.55, supervised.40, pseudo.40. Filters select
pointwise.50/set.60. No TEST values choose these policies.

The pointwise filter improves F1 by reducing false positives1210→520 while
losing68 correct words (1255→1187). Precision50.91→69.54%, recall56.94→53.86%:
this is a precision/recall tradeoff, not more recovered words or fewer OCR calls.
The set objective adds no demonstrated benefit over its matched control.

Supervised YOLO is the highest observed endpoint in this battery and does not
require U-Net at inference. Its precision stays72.57→72.66% while recall
increases58.58→65.61%, correct words1291→1446. Its cap100 and detector-based
candidate generation differ from M/K budget20; the +4.13-point within-YOLO
comparison is the matched training effect. No global ranking against differently
normalized historical experiments follows. Added teacher pseudo-labels show no
further observed improvement. Their partial-box/background limitation remains
a plausible risk, not a proven explanation of the small difference.

Conditional paired95% intervals (2000 image draws, seed20261022): filter versus
M/K[+5.73,+8.27], set versus pointwise[-.36,+.10], supervised YOLO versus initial
[+2.74,+5.64], pseudo YOLO versus supervised[-.94,+.63], OCR CE versus original
[+.14,+1.11], OCR distillation versus CE[-.48,+.18]. They exclude training
variation and historical search. The298-image known-duplicate sensitivity
retains the main patterns (filter60.459%, supervised YOLO68.863%).

Interpretation: prioritize supervised target-domain word-detector adaptation,
the pointwise filter, then moderate OCR augmentation. EMA reduces damage from
M/K continuation but remains below the original model; the extra consistency,
set and distillation terms have no demonstrated added F1 gain in these short
pilots. Do not add gains across separately tested components. Longer training,
other seeds, stronger teachers, ignore-aware pseudo-label training and independent
validation remain separate experiments.

Qualification:70 focused tests PASS; complete20-image five-method smoke PASS;
full independent7650 image/arm audit PASS. An earlier smoke's final report
correctly stopped when an unrelated script changed in the shared checkout;
immutable snapshot reruns resolved it. One test invocation named a nonexistent
historical test file, collected nothing, and was replaced by the passing command.
Optional `pip freeze` was unavailable in both existing environments; package
versions were captured using `importlib.metadata`, retaining one nameless
distribution rather than dropping its metadata. No dependencies were installed.

Measured maximum4,505,637,888 allocated /5,643,436,032 reserved CUDA bytes, with
the exact per-stage model/batch/shape records preserved. Two repeated-crop
confidence records differ numerically across inference batches, with identical
transcriptions; the audit counts them explicitly. No transcript conflict.

Full result SHA256:
`2462e105ab200151de074680dd172ebda96196d88df9701b53522898604c555a`.
Primary table: external root `full/comparison.md`; matched comparisons, intervals,
per-image F1 gains/losses and recovered/damaged word indices:
`full/detailed_analysis.md` and `full/detailed_analysis.json`.
