# DeepSolo parity thresholds for the common Total-Text / ICDAR2015 procedure

**Approximate reproduction / research extension, 2026-10-30.** Exhaustive scalar
threshold calibration and full TEST evaluation do not establish a useful
DeepSolo-parity policy for the frozen common router. Total-Text calibration
accepts no light images. ICDAR2015 accepts light images but loses about 0.55
F1 points on TEST. Preserve these negative results; TEST-derived cuts must not
be promoted as validation-selected thresholds.

## Protocol and boundaries

Source: the qualified [common budget pipeline](COMMON_BUDGET_PIPELINE.md), run
`common-budget-pipeline-20261029-v1/full`. Keep its selected models, correction
policies, score references and guards fixed. Paper Sections 3.3/3.5 and existing
implementation were inspected. These are trained utility scores; the paper's
MPNet/fuzzy tau 0.8 does not apply. Use one procedure with a separate threshold
for each differently trained dataset/expert model.

Use the reserved 63 Total-Text CAL images and 50 ICDAR2015 CAL images, retaining
1128/900 FIT and 62/50 model-selection images. Image IDs and RGB identities are
disjoint across roles. Upstream expert training/vocabulary dependencies and
historical TEST exploration remain explicit; this is not an untouched test.

Enumerate accept-none and every distinct CAL score. Accept eligible images at
`score >= threshold`, preserving whole ties and the selected guard (false in
all four source cases). Maximize accepted images subject to CAL micro F1>=CAL
all-DeepSolo, comparing integer TP and prediction/GT denominators exactly.
F1 is not assumed monotonic. Freeze every threshold before interpreting TEST
labels. No refit or TEST-based update follows. `route_observations` in
`recognition/common_parity_threshold.py` applies a cut to annotation-free
`CommonBudgetPipeline.score` outputs, without requiring DeepSolo predictions.

Metric: per-image exact multiset matching after Unicode casefold/isalnum, then
micro TP/FP/FN; no spatial matching or fuzzy threshold. Inherit native256
ignore/care handling. Keep all care words and empty outputs in denominators.
Empty normalized predictions remain FP and empty normalized care transcriptions
remain FN; zero-prediction/zero-GT aggregate F1 is 0. TEST includes 300 Total-Text
images/2204 care words and 500 ICDAR2015 images/2077 care words. This is not
Table 5 recognition accuracy or the official spatial E2E metric.

## CAL-selected thresholds applied unchanged to TEST

CAL DeepSolo-only F1: Total-Text 74.7429%, 63/63 (100.00%) fallback;
ICDAR2015 85.7143%, 50/50 (100.00%) fallback.

| Dataset | Expert | CAL threshold | CAL F1 % | CAL DeepSolo images/total (%) |
| --- | --- | ---: | ---: | --- |
| Total-Text | U-Net | accept none |74.7429|63/63 (100.00%)|
| Total-Text | YOLO | accept none |74.7429|63/63 (100.00%)|
| ICDAR2015 | U-Net |0.19578646220798304|85.9223|45/50 (90.00%)|
| ICDAR2015 | YOLO |-0.438117942311116|85.7143|35/50 (70.00%)|

TEST DeepSolo-only F1: Total-Text 82.6540%, 300/300 (100.00%) fallback;
ICDAR2015 82.7905%, 500/500 (100.00%) fallback.

| Dataset | Expert | TEST F1 % | Difference from DeepSolo, points | Light images | DeepSolo images/total (%) |
| --- | --- | ---: | ---: | ---: | --- |
| Total-Text | U-Net |82.6540|+0.0000|0|300/300 (100.00%)|
| Total-Text | YOLO |82.6540|+0.0000|0|300/300 (100.00%)|
| ICDAR2015 | U-Net |82.2171|-0.5734|56|444/500 (88.80%)|
| ICDAR2015 | YOLO |82.2482|-0.5422|141|359/500 (71.80%)|

Conditional paired-image 2000-bootstrap 95% intervals for the ICDAR2015 F1
differences: U-Net [-1.1958,-0.0465] points, 444/500 (88.80%) fallback;
YOLO [-1.1728,+0.0250], 359/500 (71.80%). These omit calibration/fitting
uncertainty and prior TEST exposure. Including zero does not establish
equivalence. No F1 degradation tolerance was authorized or substituted.

## TEST-label diagnostic: maximum of the fixed score ordering

After primary evaluation, exhaust TEST cuts only for diagnosis. These consume
TEST labels and are **not deployable validation-selected results**. They
optimize scalar cuts, not arbitrary image subsets or new models.

| Dataset | Expert | TEST-derived threshold | TEST F1 % | Light images | DeepSolo images/total (%) |
| --- | --- | ---: | ---: | ---: | --- |
| Total-Text | U-Net |0.47793356444968477|82.6818|7|293/300 (97.67%)|
| Total-Text | YOLO |0.84936198283751496|82.6705|2|298/300 (99.33%)|
| ICDAR2015 | U-Net |0.66486062662291201|82.7993|7|493/500 (98.60%)|
| ICDAR2015 | YOLO |0.56373529518873888|82.7993|5|495/500 (99.00%)|

Exhausting the frontier shows that threshold changes alone cannot retain many
light images at exact TEST parity with these fixed scores. Better expert
outputs or score ordering, or an explicitly authorized F1-loss tolerance,
would require a separate experiment. The requested useful parity target is unmet.

## Reproduction, qualification and artifacts

Profile: `configs/approximate/common_pipeline_parity_threshold_v1.json`.
Core: `recognition/common_parity_threshold.py`.
Runner: `scripts/calibrate_common_parity_threshold.py`.
Reporter: `scripts/report_common_parity_threshold.py`.
Tests: `tests/recognition/test_common_parity_threshold.py` and
`tests/recognition/test_common_budget_pipeline.py`.
Decision: `COMMON-PARITY-THRESHOLD-001` in [PAPER_AMBIGUITIES.md](PAPER_AMBIGUITIES.md).
Portable receipt: `provenance/COMMON_PARITY_THRESHOLD_COMPLETION_V1.json`.

Evidence root:
`/extra1/bmartelli/context-str-runs/common-pipeline-parity-threshold-20261030-v1`.
It saves 33 pinned input hashes, resolved configuration, executed source copies,
all cuts/counts, frozen thresholds, per-image TEST decisions, intervals and an
independent audit. `report_v2` contains CSV, Markdown, PNG/PDF curves and hash
manifest; `report_v1` preserves a failed plot export.

```bash
.venvs/context-str-main-py311/bin/python -m scripts.calibrate_common_parity_threshold \
  --mode approximate --config configs/approximate/common_pipeline_parity_threshold_v1.json \
  --stage calibrate
.venvs/context-str-main-py311/bin/python -m scripts.calibrate_common_parity_threshold \
  --mode approximate \
  --run-root /extra1/bmartelli/context-str-runs/common-pipeline-parity-threshold-20261030-v1 \
  --stage evaluate
.venvs/context-str-main-py311/bin/python -m scripts.calibrate_common_parity_threshold \
  --mode approximate \
  --run-root /extra1/bmartelli/context-str-runs/common-pipeline-parity-threshold-20261030-v1 \
  --stage audit
```

The first command exclusively creates its output root. For a new run, copy the
approximate profile and set an unused explicit output root; preserve evidence.
CPU replay uses existing main Python 3.11/NumPy 1.26.4. Plotting uses existing YOLO
Python/matplotlib 3.9.2. No package was installed. The host reports RTX 3090/24 GiB,
but this study makes zero new GPU, PARSeq or DeepSolo neural inference calls.
Fallback counts are logical cached decisions, not online latency or measured
memory savings. YOLO's inherited U-Net map dependency and Total-Text upstream
correctors remain part of expert cost.

All 23 focused regression tests pass. A synthetic two-case smoke verifies all
thresholds freeze without parsing TEST labels. Real-cache CAL/TEST runs and
the independent audit pass: 1826 per-image recounts and 1832 scalar-cut replays,
including maximal feasible cuts, ties and frozen TEST decisions. Upstream
model-fit and GT-free score qualifications remain linked and hash-checked.
The new annotation-free scalar routing adapter also replays all 1600 TEST
decisions from cached observable scores without GT or fallback predictions.

Two engineering failures were preserved and resolved: preflight initially
expected a newer `validation_policy` field absent from source-v1 (now verifies
explicit saved roles, with regression coverage); plotting initially used the
main environment without matplotlib (rerun in existing YOLO environment).
Neither changed expert predictions, selected thresholds or TEST metrics.

Classification: PAPER_SPECIFIED fallback motivation; REPOSITORY_SPECIFIED
qualified source experts/scores/protocol; ASSUMPTION exhaustive CAL parity and
scalar boundary; HARDWARE_DEVIATION CPU caches; UNRESOLVED generalization,
upstream dependencies and useful DeepSolo parity. The threshold study is
complete with a negative efficiency outcome.
