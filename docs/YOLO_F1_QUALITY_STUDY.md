# Per-image F1 quality targets with frozen DeepSolo fallback

Completed 2026-10-22. **Approximate reproduction / exploratory local analysis**,
not strict reproduction or an official Table 5 comparison. No network training,
new weights, downloads, or model inference. Original YOLO640/PARSeq outputs stay
unchanged: retain a complete image or replace it with the cached
ViTAEv2-S+TextOCR DeepSolo outputs.

## Results

On TEST300, the new `base_proxy_f1` rule improves the previous consistency
control from 79.9633% to **80.1740% final F1 at 20% bypass**. The accepted set has
50/60 images with light-branch per-image F1 >= 80%, versus 48/60 for the control.
Its accepted-word aggregate F1 is 89.0302%; this differs from the 83.3333%
good-image fraction. The paired 95% bootstrap interval for the improvement is
**[-0.064, +0.549] percentage points**, including zero.

This fixed-rule diagnostic improves the control, but not standalone local
DeepSolo's 80.3571%. The previously measured context-LM control remains slightly
higher at 80.2203%, with only 45/60 accepted images reaching image F1 >= 80%.
Selecting good images and maximizing final word F1 are different objectives.
These are descriptive comparisons of declared families, not TEST-selected
deployment winners or confirmed improvements.

| Fixed budget | TRAIN retained /125 | TRAIN final F1 | TEST retained /300 | TEST images reaching F1 >=80% | TEST final F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| New rule, 10% | 12 | 78.16% | 30 | 25/30 | 80.42% |
| New rule, 20% | 25 | 76.57% | 60 | 50/60 | 80.17% |
| New rule, 30% | 38 | 74.63% | 90 | 69/90 | 79.57% |
| New rule, 40% | 50 | 74.37% | 120 | 82/120 | 77.70% |
| New rule, 50% | 62 | 73.56% | 150 | 88/150 | 76.68% |
| All DeepSolo | 0 | 78.62% | 0 | Undefined | 80.36% |

The 10%-budget gain over all DeepSolo is +0.064 pp, with paired 95% interval
[-0.120, +0.276] pp. It does not meet the primary 20% minimum bypass and does
not establish superiority. Budgets round to nearest integer, with Python's
tie-to-even behavior, and exclude missing routing scores.

The primary TRAIN-only family selection, requiring empirical 90% good images
and at least 20% bypass, finds **no qualifying family for image F1 >=75%**.
It correctly abstains for targets 75/80/85/90/95/100%; the resulting 80.3571%
is all DeepSolo and is not successful routing. At targets 50/60/70%, the chosen
rules retain 118/74/69 TEST images with final F1 77.3151/79.0814/79.0011%.
TRAIN45 calibration cannot rescue a failed TRAIN80 family selection. Empirical
calibration does not guarantee TEST purity.

The following diagnostic uses annotations to accept **all** images reaching
each true quality threshold. It is not deployable and is not an upper bound
over all possible subsets or branch-selection policies.

| True light per-image F1 target | TRAIN retained /125 | TRAIN final F1 | TEST retained /300 | Accepted word F1 | Rejected DeepSolo F1 | TEST final F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50% | 94 | 68.36% | 222 | 76.73% | 63.38% | 74.38% |
| 60% | 72 | 72.76% | 193 | 79.20% | 64.64% | 75.39% |
| 70% | 58 | 75.40% | 152 | 84.43% | 70.11% | 77.80% |
| 75% | 50 | 76.71% | 140 | 86.82% | 72.56% | 78.98% |
| 80% | 44 | 77.46% | 120 | 89.18% | 73.95% | 79.57% |
| 85% | 31 | 78.25% | 85 | 94.08% | 76.70% | 80.67% |
| 90% | 20 | 78.54% | 65 | 96.49% | 77.33% | 80.58% |
| 95% | 14 | 78.62% | 44 | 99.75% | 78.86% | 80.80% |
| 100% | 14 | 78.62% | 43 | 100.00% | 78.98% | 80.82% |

A perfect >=80% quality gate accepting every qualifying image would still lose
against all DeepSolo. The >=85% gate keeps 28.3% of TEST and reaches 80.67%.
Quality targets need not be monotonic in final F1: DeepSolo quality changes on
the rejected subset. All values sum actual TP/FP/FN; no image-weighted average
of branch F1 is used.

At target image F1 >=80%, TEST AUROC for classifying good images is 80.52% for
the new rule, 78.82% for consistency, 72.61% for coverage, 69.04% for the paper
caption control, 76.16% for MPNet, 63.98% for contextual CLIP, and 74.75% for the
residual-VQA combination. AUROC is a selector metric, not recognition F1.
Context has not supplied a reliable 90%-purity selector with substantial bypass.

## Frozen protocol

Inspected local paper section 3.5, Algorithm 2, Table 5, and existing routing
code before changing scientific behavior. Quality targets:
50/60/70/75/80/85/90/95/100% per-image word F1. Empirical good-image fraction
targets: 80/90/95%. Primary objective: maximum final micro F1 subject to 90%
good-image purity and minimum 20% bypass.

The metric is exact casefolded word-multiset matching, punctuation retained,
without spatial IoU, using the source native256 evaluation-only ignore rule.
All 983 TRAIN and 2,204 TEST care words remain. Empty outputs cannot bypass.
TRAIN125 is the historical segmentation holdout, not complete official TRAIN;
DeepSolo fine-tuning used Total-Text TRAIN. Both splits have been explored
before, so this is not untouched generalization evidence.

Four immutable audited sources supply 425 branch predictions and 38 distinct
controls: coverage/resolution, crop/scale consistency, contextual CLIP/MPNet/FLAN,
and residual BLIP-2 VQA. Thirteen new fixed rules add confidence mean/quantile/
harmonic summaries, per-word CLIP/crop agreement, coverage proxies, cross-scale
agreement, overlap penalties, and a context combination. No fitted classifier,
regression, model update, or fitted mixture weight is introduced.

The coverage proxy assumes estimated GT = predicted word count / coverage and
estimated TP = predicted word count * precision proxy, giving
`proxy_F1 = 2 * precision_proxy * coverage / (1 + coverage)`.
Confidence, coverage, and CLIP ranks are uncalibrated proxies. `base_proxy_f1`
is the geometric mean of this visual proxy and `all_checks_softcrop`.

For each quality, purity, and objective combination:

- TRAIN80 selects a family and candidate cut. TRAIN45 independently calibrates
  its scalar cut. Neither population overlaps TEST300.
- Both stages require at least 20% bypass and 5 accepted cases: 16/80 and 9/45.
- Freeze all family/cut choices before new derived TEST features. Failure of
  either selection or cutoff qualification means full fallback.
- Compare maximum bypass with maximum final F1; report all 51 families and
  failures. Scalar label-based calibration does not train a model.
- Fixed 10/20/30/40/50% batch-budget policies select a family only on TRAIN80
  by good-image count, then final F1 and declared order. Ranking ties use stable
  sample ID. These are batch budgets, not transferred absolute cutoffs or
  guaranteed good-image fractions.

Report per-image classification counts, precision/recall, accepted mean image
F1, accepted aggregate word F1, rejected DeepSolo F1, final micro F1, TN0,
predicted words, TP/FP/FN, and sample IDs. Keep TRAIN80, TRAIN45, full TRAIN125,
and TEST300 separate. Paired 10,000-image bootstrap fixes gates and applies no
multiplicity correction or policy-selection uncertainty model.

Paper nominal 78.4/79.8 markers are numerical references. The new rule at 20%
bypass is numerically +0.374 pp over 79.8; protocol, checkpoint/ignore equivalence,
and the paper's recognition-accuracy definition remain unresolved. This does
not establish paper superiority. Cached probe latency does not demonstrate a
whole-pipeline speed saving.

## Reproduction and evidence

- Scientific config: `configs/approximate/yolo_f1_quality_study_v1.json`.
- Host paths: `configs/approximate/yolo_f1_quality_study_local_paths_v1.json`.
- Feature/calibration code: `recognition/f1_quality_study.py`.
- Runner: `scripts/run_yolo_f1_quality_study.py`.
- Independent audit: `scripts/audit_yolo_f1_quality_study.py`.
- Descriptive analysis/plots: `scripts/analyze_yolo_f1_quality_study.py`.
- Tests: `tests/recognition/test_f1_quality_study.py`.

```bash
.venvs/context-str-yolo-py311/bin/python -m scripts.run_yolo_f1_quality_study \
  --mode approximate \
  --config configs/approximate/yolo_f1_quality_study_v1.json \
  --paths configs/approximate/yolo_f1_quality_study_local_paths_v1.json

.venvs/context-str-yolo-py311/bin/python -m scripts.audit_yolo_f1_quality_study \
  --run /extra1/bmartelli/context-str-artifacts/yolo-f1-quality-study-v1/full/134cead7c5ac

OPENBLAS_NUM_THREADS=1 .venvs/context-str-yolo-py311/bin/python \
  -m scripts.analyze_yolo_f1_quality_study \
  --run /extra1/bmartelli/context-str-artifacts/yolo-f1-quality-study-v1/full/134cead7c5ac
```

Use `--preflight` for source qualification or `--smoke` for a separate 12-image
offline check. All source/code/feature hashes, resolved configuration, frozen
cuts, per-sample outputs, and audits remain external under
`/extra1/bmartelli/context-str-artifacts/yolo-f1-quality-study-v1`.

Completed full run: `full/134cead7c5ac`; plan SHA256
`e72e1c5afc212470b697faf498a1972b2fade9f7e0af07f50870f83d51f8b0d1`;
results SHA256
`1d0f9b7912e0b577ee90bc1b09e42a984aec36b08d752ea9539515124ff30791`.
Smoke `smoke/4ab51ec4f802` and full independent audits PASS: independent rebuild
of 13 new formulas, TRAIN-only family/cut/budget choices, 425 features, 11,232
routing/oracle points, and unchanged denominators. The supplement contains
9,180 descriptive rows, AUROC/AP, paired intervals, PNG/PDF plots, and JSON/CSV/
Markdown reports in `analysis/`.

Superseded initial full run `full/cc499b6ca452` is preserved. Its selected policy
incorrectly allowed a TRAIN45 cut when no TRAIN80 family qualified. This
changed selected policies at quality 75/80% and purity 90%, among other failed
combinations, but not individual-family counts, fixed budgets, or GT-only
gates. The corrected runner abstains on failed family selection, has a regression
test, and generated the new immutable run above. The old run is not final evidence.

The intermediate run `full/c0f03a1e3aed` exposed order-dependent macro-image-F1
rounding during cross-process cached replay. Exact rational summation removes
the dependency. All TP/FP/FN, aggregate F1, thresholds, and intervals are identical;
only macro rounding (less than 9e-16) and provenance hashes change. The final
run above replays with an identical results hash under PYTHONHASHSEED=1 and 37;
`reproducibility_verification.json` records the check. Both earlier runs and
failure notices are preserved.

Scientific classifications: `YOLO-F1-QUALITY-STUDY-001` in
`docs/PAPER_AMBIGUITIES.md`. No GPU inference, installation, or environment change.
Verification: 47 focused tests, including 10 new study tests; independent full
and smoke audits PASS. Numerical replay passes without new inference.
