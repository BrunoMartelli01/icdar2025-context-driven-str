# Supervised router targeting fixed-budget oracle separation

User-authorized 2026-10-23 **approximate reproduction extension**. The latest
instruction explicitly permits training the router to approach the earlier
ideal separation. Original detector/recognizer/embedding weights stay frozen.
This is not the paper's training-free router or strict Table 5 reproduction.

## Frozen protocol

Inspected local paper section 3.5/Algorithm 2 and the existing residual-region,
word-cost and oracle implementations. Reuse 125 TRAIN / 300 TEST cached image
branches, 983/2204 care words, Text/Kernel/word features and optional frozen
context/CLIP/VQA scores. No new visual-model inference or package installation.

Eight supervised families combine two feature sets with four model types:
Ridge, small ExtraTrees regression, regularized logistic oracle classification,
and small ExtraTrees oracle classification. Explicit hyperparameters and seeds
are declared before new TEST predictions. Core inputs summarize confidence,
crop agreement, word counts and missing regions. Full inputs add mask statistics
and contextual scores. Missing values have explicit indicators and zero fill.

The target is the **fixed-cardinality optimum of final word micro F1**, not
simply whether each light image reaches F1>=80%. Exact fractional optimization
derives canonical oracle labels and a zero maximal-residual certificate. Empty
light readings are ineligible, matching inference. Canonical labels can be
nonunique: alternative optimal subsets may disagree while achieving equal F1.

- Fit only the historical TRAIN80. Five-fold inner CV selects hyperparameters
  per family and budget by out-of-fold final micro F1.
- Oracle labels in each fold use only that fold's fitting examples. Cost
  regressors learn delta TP and delta predicted-word count, and use the fitting
  subset's all-DeepSolo F1 as a frozen linearized micro-F1 reference.
- Fit final candidates only on TRAIN80; their scaler sees only these examples.
- VAL45 selects among the eight families at each 20/30/50% budget and supplies
  independent inclusive scalar cuts. Do not refit on all125.
- Freeze all choices and portable JSON models before deriving TEST vectors or
  predictions. At inference, neither GT nor actual DeepSolo outputs are inputs.
- Keep exact rank budgets separate from transferred scalar-cut coverage. Save
  every family and control, oracle gap, canonical decision agreement, actual
  TP/FP/FN, final F1/TN0, image quality fractions and selected IDs.

Primary predeclared family: oracle_logistic_core. Frozen controls include the
previous coverage, consistency, base_proxy_f1 and residual-cost gates. Fit80
performance is in-sample; inner OOF, VAL45, fullTRAIN125 and TEST300 remain
separate. Paired10,000-image bootstrap fixes gates without multiplicity correction.

Only the tabular router is trained. Portable coefficients and trees reproduce
sklearn inference, including its float32 tree traversal. Warnings about solver
nonconvergence are failures, not silently suppressed. No pickle is required.

## Limitations

Eighty fitting images are few. TRAIN/TEST have already been repeatedly explored,
and DeepSolo was fine-tuned on Total-Text TRAIN. TEST therefore is exploratory,
not untouched validation. These data cannot guarantee perfect separation or a
paper-superiority claim. The local exact-casefold multiset/native256 ignore
metric is not established as equivalent to the paper's recognition accuracy.
Additional probe time remains outside this cached CPU experiment's runtime.

## Files and commands

Config: `configs/approximate/trainable_image_router_v1.json`.
Paths: `configs/approximate/trainable_image_router_local_paths_v1.json`.
Core: `recognition/trainable_image_router.py`.
Runner: `scripts/run_trainable_image_router.py`.
Independent fitting replay/audit and plots: `scripts/analyze_trainable_image_router.py`.
Tests: `tests/recognition/test_trainable_image_router.py`.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venvs/context-str-yolo-py311/bin/python -m scripts.run_trainable_image_router \
  --mode approximate \
  --config configs/approximate/trainable_image_router_v1.json \
  --paths configs/approximate/trainable_image_router_local_paths_v1.json
```

Use --preflight to check immutable sources or --smoke for a separate20-image
boundary run. New models/features/predictions/CV traces and complete results
remain external under trainable-image-router-v1. No prior experiment is changed.
Scientific record: YOLO-TRAINABLE-ROUTER-001 in docs/PAPER_AMBIGUITIES.md.

## Completed measurement

The supervised trial **does not reach perfect separation or improve the best
previous controls at the same budgets**. Selected models are those chosen on
VAL45, never the most favorable TEST model.

| Budget | VAL-selected family | Fit80 F1 (in-sample) | VAL45 F1 | TRAIN125 F1 | TEST300 F1 | Eligible oracle F1 | Oracle gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20% | trees_full | 76.746% | 81.008% | 78.676% | 80.064% | 82.875% | 2.811 pp |
| 30% | ridge_core | 76.530% | 80.801% | 77.532% | 79.351% | 82.914% | 3.563 pp |
| 50% | oracle_logistic_core | 75.212% | 78.607% | 76.421% | 77.945% | 82.320% | 4.376 pp |

Exactly 60/90/150 TEST images bypass DeepSolo. Earlier controls give 80.174%
(base_proxy_f1),79.835% (risk_visual_consensus),78.473% (risk_visual_mass) at
these respective quotas. All DeepSolo TEST F1 remains80.357%. These figures
are local lexical results; no paper superiority or impossibility is established.

The strongest TEST family posthoc reaches80.092/79.808/78.387% at20/30/50%,
respectively. It is not promoted to replace VAL choices and remains below the
corresponding previous controls. Adding full contextual features does not
consistently improve this small-data supervised trial.

The earlier oracle allowed empty light predictions. Its original ceilings are
82.883/82.945/82.431%. Inference-compatible eligibility here rejects empty
readings and gives82.875/82.914/82.320%. Both diagnostics are preserved in
`unconstrained_oracle_reference.json`. The small ceiling difference is an
explicit constraint, not an improvement in the oracle or a metric substitution.

Canonical oracle overlaps for VAL-selected TEST policies are8/60,33/90,96/150.
This does not mean every disagreement loses F1: optimal subsets can be
nonunique. Final F1 regret is the primary distance to perfect selection.
Eighty fitting images and these features have not produced a successful
supervised separator; this does not prove that more data or other evidence
cannot improve it.

## Evidence and verification

Full run:
`/extra1/bmartelli/context-str-artifacts/trainable-image-router-v1/full/be5bf92dea94`.
Plan SHA256:
`4ffdcfa461d79b043b7300a3407adac5e160555b08bff87f135383db3b9fcd2e`.
Results SHA256:
`516fa4a4737df416d76777c8a70397f6bce83fda359c12f7d9c0d716475e099a`.

The runner made474 tabular fit calls, including inner CV, with no visual-model
updates. Independent audit PASS:425 feature vectors,264 routing points and
7,200 held-out predictions, using311 independent cached refits. It checks
fitting/validation scopes, fold-only teachers/scalers, CV selection, portable
model predictions, VAL choices/cuts, actual branch counts and oracle certificates.
Smoke20-image audit also PASS. Full numerical retraining/replay under
PYTHONHASHSEED=1 and37 preserves the identical results hash.

TEST vectors contain the hash of already-written frozen choices. The initial
full run `full/d07e0cc0c588` failed a strict timestamp-order assertion because
the frozen file and first TEST vectors shared a filesystem clock tick. That run
and its notice are preserved. The stronger hash binding and nondecreasing-time
check generated the final run above. All model decisions, F1/counts and intervals
are numerically unchanged; only provenance metadata changes. No fitting or
selection on TEST was introduced.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venvs/context-str-yolo-py311/bin/python -m scripts.analyze_trainable_image_router \
  --run /extra1/bmartelli/context-str-artifacts/trainable-image-router-v1/full/be5bf92dea94
```

The external run preserves portable JSON models, full CV traces, frozen choices,
input/code hashes, train/fit/validation/test predictions, `report.md`,
`comparison.csv`, `comparison.png/.pdf`, actual counts and bootstrap intervals.
Smoke results are separate and never merged with the complete evaluation.

Validation: **31 focused tests PASS**, including11 new supervised-router tests.
