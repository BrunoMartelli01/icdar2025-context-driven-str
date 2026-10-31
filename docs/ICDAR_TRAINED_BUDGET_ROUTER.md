# Dataset-specific cheap YOLO/U-Net routing on ICDAR2013 and ICDAR2015

**Approximate reproduction / research extension.** User-authorized training of
four native-dataset lightweight gates with model and threshold selection on
validation. Paper Sections3.3/3.5 and Algorithm2 and existing essential-budget,
YOLO/PARSeq, corrected-U-Net and ICDAR2015 native implementations were inspected.
These trained routers do not reproduce the paper's training-free MPNet rule.

## Predeclared protocol

Use official TRAIN229/TEST233 for ICDAR2013 and TRAIN1000/TEST500 for ICDAR2015.
TRAIN13 transcriptions use the existing blank-separated Task2.2 character-group
parser because original Task1 TRAIN boxes are unavailable locally. This is an
ASSUMPTION; official Task1 TEST transcriptions remain unchanged. Exclude/report
exact TRAIN/TEST RGB copies from router fitting, never remove TEST images.

ICDAR2013 uses46 deterministically selected validation images, split into23
model-selection and23 calibration images; remaining183 fit the router. Eligible
U-Net candidates are the qualified TotalText Text+Kernel raw and corrected
pipelines, both without ICDAR image gradient fitting. The historical IC13 U-Net
trained on all229 images is not eligible for independent validation selection;
the two-epoch M9 checkpoint lacks qualified present bytes. "Best" means best
eligible checked pipeline on selection images, not a global or paper optimum.

ICDAR2015 retains the existing900 expert-fitting /100 expert-heldout TRAIN split.
Split heldout100 into50 selection and50 calibration. Candidate U-Nets include
the same two TotalText transfer pipelines and all three completed native
source768/long768/adapt1280 checkpoints with their previously validation-frozen
OCR/crop policies. OHEM retained an identical selected state; ongoing TotalText
fine-tuning runs are excluded from this frozen qualified-checkpoint comparison.
The separate `tt263-native1280-v1` run completed while this task was resumed;
it is not part of the three-candidate upstream selection contract frozen here.
This finite comparison does not establish the best result over every workspace
checkpoint, unqualified new branch or architecture.
Only selection images choose the U-Net; calibration images do not choose a head.
Historical upstream checkpoint selection used VAL100 and remains a limitation.

Keep YOLO direct640, genuine PARSeq and native ViTAEv2-S/TextOCR DeepSolo with
its frozen0.35/0.925 policy. YOLO test outputs and completed fallback exports may
be hash-qualified and reused. Export every missing TRAIN fallback prediction in
the isolated existing legacy environment; distinguish this actual offline work
from logical final fallback counts. No package installation or expert training.

Train small CPU Ridge/boosted-tree heads using exact paired GT/DeepSolo utility:
`(2*DeltaTP-F_reference*DeltaN)/(observed light rows+1)`, with fitting-image
weight`rows+1`. Reference F1 and standardization use fitting IDs only. Test
labels never train or select a head, expert or scalar threshold.

Use18 OCR summaries, optional8 token summaries and actual native map descriptors
for U-Net. Each genuine output channel contributes six coverage/missing-support
features. Single-head rectangular IC15 maps have six descriptors and their
qualified aspect-letterbox coordinates; no invented kernel channel or silent
square resize is introduced. YOLO never runs U-Net or another encoder for its
gate. All inference features are annotation/fallback blind.

Select the best head by selection-set hybrid lexical F1 at70% light acceptance
for ICDAR2013 (30% DeepSolo) and50% for ICDAR2015 (50% DeepSolo), following the
user's revised budget before full inference or fitting. Within0.25 points prefer
fewer stored scalars, then bytes and F1. Fix65/70/75% IC13 and45/50/55% IC15
acceptance cuts on the separate calibration images using the unchanged model.
Tie keys use RGB SHA256, avoiding TRAIN/TEST filename-prefix artifacts. Never
refit after calibration. Report transferred cuts and the separately predeclared
exact-budget batch rules, conditional GT oracles and all-light/all-fallback controls.
Scalar cuts can transfer to a different TEST fallback percentage; report that
percentage rather than promising an exact budget. The batch rank rule enforces
the predeclared quota without TEST labels.

Primary metric is per-image lexical-multiset micro F1, Unicode casefold then
`isalnum`, no localization IoU. Retain all care GT and empty/symbol-only output
denominators. Predictions wholly in ignored regions at source native256 are
excluded consistently for both experts; care regions override ignored polygons.
This harmonized router metric differs from earlier native IC15 spatial scores
that counted ignored-region predictions as FP. Do not claim official polygon
Hmean, Table5 accuracy or strict reproduction.

## Scientific boundaries

- PAPER_SPECIFIED: broad segmentation/crop/fallback mechanism only.
- REPOSITORY_SPECIFIED: hash-qualified experts, OCR, existing source policies,
  coordinate transforms and small-head implementations.
- ASSUMPTION: native dataset splits, character-group TRAIN13 labels, finite
  candidate selection, utility/weights, descriptors, lexical/native-ignore metric
  and budget calibration; cross-dataset candidate comparison.
- HARDWARE_DEVIATION: observed RTX3090 FP32, batch1 segmentation, batch32 PARSeq,
  staged expert processes and existing isolated legacy fallback environment;
  CPU lightweight fitting. Save allocated/reserved peaks and explicit failures.
- UNRESOLVED: limited validation, historical upstream/TEST exposure, expert-seen
  fitting examples for native IC15, pretraining-corpus overlap and online latency.

Implementation/configuration: `scripts/train_icdar_budget_router.py`,
`scripts/icdar_budget_stages.py`, `scripts/icdar_budget_data.py`,
`recognition/icdar_budget_router.py`, and explicit `configs/approximate/`
profiles. No result is claimed by this predeclaration. Completion is appended
only after full TEST evaluation and independent audits. The earlier completed
engineering smoke and unused prepared full manifest had50% budgets for both
datasets; preserve them under `icdar-native-budget-router-20261028-v1`.
Execute the revised full experiment under the distinct
`icdar-native-budget-router-20261028-budget-v2/full` namespace.

## Verification and deployment outputs

`scripts/audit_icdar_budget_router.py` independently verifies fit-only scaling,
weighted Ridge normal equations, boosted-tree refits, expert/head selection,
calibration quantiles, actual archived feature reconstruction, RGB partition
separation, complete TEST denominators and exact-cardinality oracle certificates.
`scripts/report_icdar_budget_router.py` exports machine-readable thresholds,
small selected heads and conditional2000-draw image-bootstrap intervals. These
intervals hold models/rules fixed and omit fitting/selection uncertainty. The
single-image interface in `recognition/icdar_budget_deployment.py` accepts only
OCR observations, genuine selected map outputs, RGB tie key and frozen policies.
It receives neither annotations nor actual fallback predictions. Export fixes
the older helper's descriptive `mask`/`maps` cost-label alias; weights and feature
calculations remain unchanged. Replay every TEST single-image gate decision and
compare scores against the vectorized evaluation. CPU gate observations exclude
expert inference, disk I/O and RGB hashing; they are not end-to-end benchmarks.

## Completed full TEST evaluation

**Approximate reproduction / research extension.** All233 official IC13 TEST images /1095 care words and500 IC15 TEST images /2077 care words are retained. Fit183/900 images respectively; model selection23/50 and separate calibration23/50. No TEST labels select the expert, head or scalar cutoff. Sixty candidate gates were fitted; no refit followed calibration. The selected U-Nets are qualified TotalText Text+Kernel corrected for IC13 and native `long768` for IC15. This is the best of the eligible finite validation comparison, not an exhaustive workspace optimum.

Logical routing uses cached expert outputs. The full run generated1229 new TRAIN DeepSolo inferences and reused733 existing TEST exports. No online pipeline latency or realized speedup was measured. Source/per-image peaks and explicit input/model metadata are retained; no GPU inference failure or OOM occurred. Main and legacy export workers overlapped on the RTX3090; process allocator peaks do not measure total simultaneous host GPU use. Main inference used deterministic FP32 math attention; legacy DeepSolo retained its qualified cuDNN enabled/benchmarkFalse/deterministicFalse policy. These are HARDWARE_DEVIATION choices, not an8GB RTX3070 qualification. The earlier smoke used40 TRAIN fallback inferences under the superseded50%-both engineering policy. An initial smoke launch with a relative interpreter path exited127 before inference; the absolute-path rerun succeeded.

### Primary fixed-budget results

Ranks use only frozen model scores and the predeclared budget, without TEST labels. Their decision cut depends on the batch, so they must not be called validation-fixed scalar thresholds. The IC13 nearest-integer budget is70/233 (30.04%); IC15 is250/500 (50.00%). Conditional oracles know GT and retain the same eligibility guard and fallback count as the router.

| Dataset | Light expert | Router F1 % | Conditional GT oracle F1 % | Oracle gap pp | Accepted light | DeepSolo images/total (%) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| icdar13 | unet (tt_corrected) | 82.9715 | 87.5414 | 4.5699 | 163 | 70/233 (30.04%) |
| icdar13 | yolo (yolo) | 86.2281 | 90.1395 | 3.9114 | 163 | 70/233 (30.04%) |
| icdar2015 | unet (long768) | 65.9547 | 74.7570 | 8.8024 | 250 | 250/500 (50.00%) |
| icdar2015 | yolo (yolo) | 69.3919 | 75.5886 | 6.1967 | 250 | 250/500 (50.00%) |

### Baselines with the identical metric and ignore policy

| Dataset | Expert alone | F1 % | DeepSolo images/total (%) |
| --- | --- | ---: | --- |
| icdar13 | unet (tt_corrected) | 73.3237 | 0/233 (0.00%) |
| icdar13 | yolo (yolo) | 82.3254 | 0/233 (0.00%) |
| icdar13 | DeepSolo ViTAE/TextOCR, frozen native policy | 88.4956 | 233/233 (100.00%) |
| icdar2015 | unet (long768) | 59.7031 | 0/500 (0.00%) |
| icdar2015 | yolo (yolo) | 65.8134 | 0/500 (0.00%) |
| icdar2015 | DeepSolo ViTAE/TextOCR, frozen native policy | 70.3665 | 500/500 (100.00%) |

### Scalar thresholds selected only on separate validation calibration

The IC13 nominal30% fallback cut does not transfer its intended quota: YOLO sends103/233 (44.21%) and U-Net109/233 (46.78%) to DeepSolo. IC15 transfers more closely: YOLO254/500 (50.80%) and U-Net263/500 (52.60%). The small23-image IC13 calibration population and score-distribution transfer remain limitations; no TEST-fitted replacement scalar is promoted.

| Dataset | Light expert | CAL target light acceptance % | Cutoff | TEST F1 % | Predicted care words | Accepted light | DeepSolo images/total (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| icdar13 | unet | 65 | -0.7661868225 | 85.1675 | 995 | 121 | 112/233 (48.07%) |
| icdar13 | unet | 70 | -0.8046180969 | 84.8804 | 995 | 124 | 109/233 (46.78%) |
| icdar13 | unet | 75 | -0.8678354092 | 84.9761 | 995 | 127 | 106/233 (45.49%) |
| icdar13 | yolo | 65 | -0.3238543176 | 87.5059 | 1026 | 118 | 115/233 (49.36%) |
| icdar13 | yolo | 70 | -0.3817747395 | 87.3226 | 1019 | 130 | 103/233 (44.21%) |
| icdar13 | yolo | 75 | -0.4478576440 | 87.2280 | 1019 | 136 | 97/233 (41.63%) |
| icdar2015 | unet | 45 | 0.3431066052 | 66.8744 | 1775 | 198 | 302/500 (60.40%) |
| icdar2015 | unet | 50 | 0.2758701963 | 65.9014 | 1756 | 237 | 263/500 (52.60%) |
| icdar2015 | unet | 55 | 0.2056138704 | 65.4279 | 1744 | 285 | 215/500 (43.00%) |
| icdar2015 | yolo | 45 | -0.3066845950 | 70.4183 | 1868 | 184 | 316/500 (63.20%) |
| icdar2015 | yolo | 50 | -0.4319664860 | 69.3960 | 1880 | 246 | 254/500 (50.80%) |
| icdar2015 | yolo | 55 | -0.6046712588 | 68.5227 | 1910 | 308 | 192/500 (38.40%) |

### Interpretation and head cost

YOLO is stronger than the selected U-Net at both primary budgets. Neither learned router reaches its conditional oracle or improves on the all-DeepSolo endpoint at those budgets. In IC13 the U-Net oracle itself is87.5414%, DeepSolo70/233 (30.04%), below DeepSolo-only88.4956%,233/233 (100.00%): expert quality imposes a ceiling even with perfect selection under this guard/budget. YOLO still has3.9114 points of selection gap in IC13,70/233 (30.04%), and6.1967 in IC15,250/500 (50.00%). The U-Net IC15 gap is8.8023 points,250/500 (50.00%). These are finite-population diagnostic bounds, not deployable performance.

All heads require zero additional encoder families. Both YOLO heads select OCR18 with30-iteration small boosted trees. IC13 U-Net selects OCR18+tokens8+maps12 Ridge alpha100; IC15 U-Net selects OCR18+maps6 Ridge alpha10. Ridge tensor-plus-normalization storage is920/584 bytes respectively; serialized deployment below includes policies and provenance. CPU timing excludes expert inference, disk I/O and RGB hashing.

| Dataset | Expert | Feature dimension | Serialized selected deployment bytes | CPU gate median ms | CPU gate p95 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| icdar13 | unet | 38 | 7187 | 1.586 | 5.382 |
| icdar13 | yolo | 18 | 43074 | 0.710 | 0.759 |
| icdar2015 | unet | 24 | 22777 | 1.583 | 1.626 |
| icdar2015 | yolo | 18 | 77613 | 0.712 | 0.765 |

### Evidence and completion checks

External full run: `/extra1/bmartelli/context-str-runs/icdar-native-budget-router-20261028-budget-v2/full`. `results.json` holds all routes and baselines; `frozen_choices.json` binds validation cuts to unchanged weights. `report/thresholds.csv` includes F1, TN=0 accuracy, TP/FP/FN, word counts, accepted images and DeepSolo images/total/percentage for every control. `report/thresholds.png` and `.svg` show budget and transferred-cut curves. `report/deployments/` contains selected small heads and portable Ridge JSON.

Independent audit PASS:704 immutable source files,41 pinned inputs,36 weighted-Ridge normal-equation checks,24 boosted-tree refits,60 candidate selection routes,12 calibration cuts,3924 rebuilt observations,12 complete TEST route checks and12 exact oracle certificates. Single-image qualification PASS:1466 calls /4398 decisions; maximum vectorized-versus-single score error1.56e-15. Twenty-eight focused unit tests pass. Preserve the original engineering smoke and unused prepared manifest. Source versions for postprocessing are retained under `report/source_at_generation/`; a label-offset-only plotting revision is separately bound by `report/plot_receipt.json`.

`provenance/ICDAR_TRAINED_BUDGET_ROUTER.json` preserves lightweight hashes, source identities and numerical summaries; external per-sample evidence remains authoritative. Image-bootstrap intervals use2000 fixed-policy draws and exclude fitting/selection uncertainty. These lexical F1 results are not official polygon Hmean and do not establish correspondence with paper Table5 recognition accuracy.
