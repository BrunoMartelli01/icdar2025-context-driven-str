# Residual-region and word-error-cost routing

User-authorized 2026-10-22 extension. **Approximate reproduction**, outside the
strict paper router. No model training, weight changes, package installation,
downloads, new network inference, or changes to final word predictions.

## Predeclared protocol

Inspected local paper sections 3.3/3.5 and Algorithms 1/2, existing coverage,
crop/scale consistency, and the completed F1-quality study before implementation.
The objective is better final word F1 at identical 20/30/50% image acceptance,
using actual frozen ViTAEv2-S+TextOCR on rejected images.

Source: audited `yolo-f1-quality-study-v1/full/134cead7c5ac`, with all 125 TRAIN
and 300 TEST images, 983/2204 care words, original YOLO640/PARSeq words and
Text/Kernel probability maps. TRAIN80 selects a family; TRAIN45 independently
calibrates scalar cuts. Fixed rank budgets and transferred cuts are distinct.
All choices are frozen before new TEST derived features. TEST has already been
explored; TRAIN is a historical holdout and DeepSolo used Total-Text TRAIN.

Read only the `probabilities` NPZ member. Geometric center-padding defines the
image canvas; do not use annotation validity/owner/GT arrays. Full foreground
components use predicted probability >=0.5, area >=4, at least 4 exposed pixels,
and uncovered fraction >=0.5. Small box edges are not new missing regions.
Cross-head corroboration requires at least 25% support on uncovered pixels.
These thresholds are assumptions, fixed before new TEST features.

Count proxies use the maximum per-head missing-region count, the maximum
uncovered mass divided by median positive predicted-word foreground mass, or
the average of the two corroborated counts. Component-median mass supplies a
fallback scale; empty masks are neutral. Components can split or merge words,
so these are uncertain proxies, not true missing-word counts.

Per-word evidence is the frozen native confidence, CLIP TRAIN-rank, and minimum
perturbed-crop edit agreement. Their geometric mean is uncalibrated visual
evidence. Summing its error proxy weights images by predicted word count.
`risk = 2 * sum(1 - word_evidence) + missing_proxy`: a wrong reading can cause
both an FP and an FN. The fixed primary is `risk_visual_consensus`; eight new
rules and three controls are all reported. No learned classifier or weights.
The surrogate does not observe or know DeepSolo's inference-time accuracy.

Evaluate exact-casefold word multisets, punctuation retained, no spatial IoU,
the original evaluation-only native256 ignore rule, empty outputs retained,
actual branch TP/FP/FN, image F1>=80/85% fractions, final micro F1 and TN0.
Every image either keeps all light outputs or all DeepSolo outputs. No merging,
filtering, or GT-based inference occurs. TRAIN-only family choice maximizes
final micro F1, rather than classifying every image above 80% equally.

Paired bootstrap uses 10,000 image replicates and fixed gates, with no
multiplicity correction. Paper nominal 79.8 is only a numerical marker;
official protocol equivalence and full-pipeline latency remain unresolved.

## Files and execution

Scientific config: `configs/approximate/yolo_residual_cost_router_v1.json`.
Explicit paths: `configs/approximate/yolo_residual_cost_router_local_paths_v1.json`.
Core: `recognition/residual_cost_router.py`.
Runner: `scripts/run_yolo_residual_cost_router.py`.
Independent audit and report: `scripts/analyze_yolo_residual_cost_router.py`.
Tests: `tests/recognition/test_residual_cost_router.py`.

```bash
OPENBLAS_NUM_THREADS=1 .venvs/context-str-yolo-py311/bin/python \
  -m scripts.run_yolo_residual_cost_router --mode approximate \
  --config configs/approximate/yolo_residual_cost_router_v1.json \
  --paths configs/approximate/yolo_residual_cost_router_local_paths_v1.json
```

Add `--preflight` to qualify cached inputs or `--smoke` for a separate 9-image
boundary run. Outputs remain external under `yolo-residual-cost-router-v1`,
with pinned source/map/code hashes, component inventories, per-word evidence,
TRAIN choices, complete per-image counts and CPU runtime records.

## Completed results

Full run: `/extra1/bmartelli/context-str-artifacts/yolo-residual-cost-router-v1/full/7e58d8eaf0c9`.
Plan SHA256: `73b7832ddf0ff8532974636c3b09c23fb47ece18e5d383987533577c754d47c3`.
Result SHA256: `62dd5b86a4d57a0da51785433107cc73045773fc0bd86c1c6168eaa89e6d75e2`.
No omitted samples, training, GPU inference or changes to original word outputs.

| Budget | TEST accepted | Previous base_proxy_f1 | Fixed primary risk_visual_consensus | TRAIN80-selected rule | Selected TEST F1 |
| --- | ---: | ---: | ---: | --- | ---: |
| 20% | 60 | 80.174% | 80.110% | risk_visual_mass | 79.752% |
| 30% | 90 | 79.569% | 79.835% | risk_visual_consensus | 79.835% |
| 50% | 150 | 76.675% | 77.791% | risk_visual_mass | 78.473% |

TRAIN125 selected-policy F1 at 20/30/50% is 77.848/76.964/75.382%, retaining
25/38/62 images. Selection itself uses TRAIN80; full TRAIN is diagnostic and
contains selection/calibration cases. TEST selection is never substituted.

At 50% bypass, the selected rule adds 41 TP and removes 36 FP versus
base_proxy_f1, giving +1.798 pp F1, paired 95% interval [+0.158,+3.625] pp.
However, the stronger pre-existing coverage control already gives 77.595%:
the new gain versus it is +0.877 pp, interval [-0.117,+1.900] pp, including zero.
Against all DeepSolo, 80.357%, the selected rule remains worse by 1.884 pp.
None of these exploratory uncorrected comparisons proves paper superiority.

At 30%, the primary also selected on TRAIN improves base_proxy_f1 by 0.265 pp,
interval [-0.394,+0.964] pp. Its 79.835% is only numerically above paper nominal
79.8. At 20%, neither primary nor TRAIN-selected policy improves base_proxy_f1.
The alternative risk_visual_count yields 80.183% at 20% but was not selected
on TRAIN and is not promoted as a deployment winner.

The selected 50% gate accepts 84/150 images reaching light image F1>=80%, versus
88/150 for base_proxy_f1, despite improving final word F1. This confirms that
good-image purity and final micro F1 are different objectives. At 30%, the new
gate accepts 56/90 good images versus 69/90 for the previous rule. It does not
certify an image-F1 threshold.

Posthoc explanatory controls, introduced after the main TEST result, give:

| Diagnostic TEST F1 | 20% | 30% | 50% |
| --- | ---: | ---: | ---: |
| Fewest predicted words only | 78.897% | 78.691% | 77.063% |
| Visual error cost without residual regions | 79.267% | 78.706% | 77.439% |
| Fixed primary with corroborated regions | 80.110% | 79.835% | 77.791% |

The residual signal therefore adds descriptive value relative to the same
word-error cost alone. These posthoc diagnostics do not choose new deployment
policies or establish corrected statistical significance. Predicted missing
regions occur in 212/300 TEST images; corroborated counts are nonzero in 207.

Independent full and smoke audits PASS: 425 full features, 7,168 full foreground
components and 264 routing points. SciPy labeling independently reconstructs
the components generated by OpenCV, coordinate projection, cross-head support,
mass scales, word evidence, every score, TRAIN choice and final denominator.
Only NPZ prediction members are accessed. Numerical cached replay with
PYTHONHASHSEED=1 and 37 gives the identical result SHA256.

```bash
OPENBLAS_NUM_THREADS=1 .venvs/context-str-yolo-py311/bin/python \
  -m scripts.analyze_yolo_residual_cost_router \
  --run /extra1/bmartelli/context-str-artifacts/yolo-residual-cost-router-v1/full/7e58d8eaf0c9
```

The run contains `report.md`, `comparison.csv`, `comparison.png/.pdf`,
`results.json`, `independent_audit.json`, and
`control_comparisons_and_ablations.json`. Paired intervals condition on fixed
gates, reuse explored TEST, and apply no multiplicity correction.

## CLIP sentence and word interpretation

This experiment reuses frozen word-level CLIP evidence, not a new full-sentence
CLIP run. Whole-image/transcription compatibility cannot certify the spelling
of every word or detect every missing word. The original [ARO study](https://arxiv.org/abs/2210.01936)
documents composition and word-order weaknesses; this is not itself an OCR
transcription benchmark. Inference: use global compatibility as secondary
evidence and require crop/word discrimination for local reading checks.

Scientific classifications: `YOLO-RESIDUAL-COST-001` in
`docs/PAPER_AMBIGUITIES.md`. Focused tests and audits cover this approximate
extension; no strict reproduction blocker is declared resolved.

Validation: **57 focused tests PASS**, including 10 residual-router tests.
