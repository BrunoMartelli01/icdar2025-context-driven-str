# Independent inventories and bounded routing corrections

Completed on PAPER, 2026-10-23. **Approximate reproduction; exploratory router
experiments**, not strict reproduction or an official Total-Text paper result.

## What is new in this repository

The original paper, section3.5 / Algorithm2, compares MPNet/fuzzy text-context
scores and sends an uncertain image to DeepSolo. These experiments keep the
whole-image fallback boundary but change the evidence. They do not change a
single final YOLO/PARSeq or DeepSolo word.

Existing experiments already measured U-Net residuals, Florence OCR inventories,
context VQA, crop/resolution agreement, permutation sensitivity, glyph checks,
rendered text and learned branch-benefit prediction. CRAFT regions had been
evaluated as an alternative detector, but not as the following routing policies:

| New variant | Observable evidence | Important distinction |
| --- | --- | --- |
| Geometry audit | CRAFT region area unexplained by the union of original YOLO boxes, and YOLO area unsupported by CRAFT | A dedicated character-region detector, replacing neither the light branch nor its words; different from the existing Florence inventory and U-Net residuals |
| Active region audit | At most3 CRAFT regions selected **before** their OCR, prioritizing uncovered area and weak nearby YOLO readings; original and rectified PARSeq crops | Checks potentially omitted words instead of checking only previously detected words |
| Capture-recapture risk | Counts of YOLO and CRAFT regions and their one-to-one geometric matches | Chapman-inspired estimate of unseen inventory; an explicitly uncalibrated heuristic, not a completeness certificate |
| Complete reading agreement | Multiset overlap between original YOLO readings and a separate CRAFT/PARSeq inventory | More expensive control; shared PARSeq weights still correlate recognition errors |
| Bounded correction | Keep old decisions outside a narrow acceptance-rank band; use an inventory signal only within that band | Changes at most the declared number of accepted images; preserves the exact overall bypass quota |

The character-region motivation is supported by the original
[CRAFT paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Baek_Character_Region_Awareness_for_Text_Detection_CVPR_2019_paper.html).
Routing rules, active probes, risk formulas and bounded corrections here are our
`ASSUMPTION` experiments; that paper does not establish their routing utility.
No claim of global literature novelty is made.

## Frozen protocol

- TRAIN125 is the historical segmentation holdout, **not all1255 official TRAIN
  images**. It is partitioned into fit80 / VAL45, with unchanged prior IDs.
- TEST300 /2,204 care words. Every image and failed/empty reading remains in the
  evaluation. No file-identical TRAIN/TEST images occur in this population.
- Final light branch: frozen YOLO640 + qualified genuine PARSeq. Fallback: cached
  audited DeepSolo ViTAEv2-S/TextOCR. DeepSolo is not rerun or requalified here.
  This PARSeq artifact is not established as the paper's PARSeq-S checkpoint.
- Exact Unicode casefold multiset matching, punctuation retained, no spatial
  matching for word correctness; inherited native256 evaluation-only ignore
  handling. F1=2TP/(2TP+FP+FN), TN=0 accuracy=TP/(TP+FP+FN).
- A whole image is ineligible for light acceptance if the original word list is
  empty or contains an empty reading. TEST291 images are eligible.
- Fixed bypass quotas20/30/50% mean60/90/150 TEST images. Other images use the
  original complete fallback outputs. Scalar VAL cuts are also saved for the
  unrestricted experiment; they are separate from quota selection.
- Existing TEST was repeatedly explored. These are not independent benchmark
  confirmation results. Paper Table5 accuracy remains insufficiently specified
  for a direct numerical win claim.

## Implementation and selection

Science settings:
`configs/approximate/shadow_inventory_router_v1.json`, paths:
`configs/approximate/shadow_inventory_router_local_paths_v1.json`.

`recognition/shadow_inventory_router.py` implements exact rectangle-union
coverage, maximum-cardinality then maximum-IoU one-to-one assignment atIoU.25,
pre-OCR suspect selection, four fixed rules and16 extra scalar features.
Probe budget3, uncertainty weight.25 and confidence definitions are explicit
configurations/implementation assumptions, not paper parameters.

CRAFT uses the already qualified EasyOCR1.7.2 checkpoint SHA256
`4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17`,
canvas1280 / magnification1.5 / text.7 / low.4 / affinity.4. Strict loading,
original stored RGB orientation and native preprocessing are preserved. All
425 images receive fresh FP32 GPU detector and recognition inference, so TRAIN
and TEST do not mix earlier CPU and current GPU predictions. Rejected regions
are explicit; none was silently converted into a crop.

The confidence proxy for a nonempty reading is the geometric mean of its lowest
character probability and geometric mean character probability, retaining
punctuation. Original/rectified view ties favor original pixels. This is not a
calibrated probability of correct spelling. Primary rule `probe` at50% was
declared before the new TEST run. Its score uses only the selected regions'
readings; the complete agreement control and learned feature variants use all
independent readings. The suite computes all readings to compare these arms,
not to claim an optimized production cascade.

Four additional matched controls fit ridge/ExtraTrees regressors with core46
versus core+inventory62 inputs. Targets are deltaTP and deltaPred between the
frozen branches; scores are2*predictedDeltaTP-F1_DS_fit*predictedDeltaPred.
All scaler/target/reference fitting uses fit80 only, with RGB-grouped5-fold
inner validation. VAL45 chooses families/cuts, frozen before fresh TEST
inference. No neural model updates, package installations or downloads occurred.

The bounded follow-up was proposed **after** inspecting unrestricted TEST
summaries. Its inputs are already exposed TEST signals; no fresh-validation
claim is made. It searches half-band widths0/2.5/5/10% and audit weights
.25/.5/1 on fit80, then VAL45 chooses a signal or the unchanged baseline,
favoring baseline on ties. Reference controls are base_proxy_f1 at20%,
risk_visual_consensus at30%, risk_visual_mass at50%. This is batch routing
with an exact quota, not an online scalar threshold. Configuration:
`configs/approximate/boundary_inventory_router_v1.json` and its separate paths
profile `configs/approximate/boundary_inventory_router_local_paths_v1.json`.
All trials, policies and complete accepted IDs are preserved.
Implementation: `recognition/boundary_inventory_router.py` and
`scripts/run_boundary_inventory_router.py`.

## Results

Final TEST micro word F1, frozen complete-image branches:

| Router | Bypass20% | Bypass30% | Bypass50% |
| --- | ---: | ---: | ---: |
| Historical strongest established control per quota |80.174%|79.835%|78.473%|
| Geometry audit |80.037%|79.662%|77.036%|
| Active region audit, primary |80.202%|79.310%|78.034%|
| Capture-recapture risk |79.735%|79.270%|77.633%|
| Full inventory reading agreement |79.982%|77.849%|75.256%|
| Ridge with inventory |79.778%|79.425%|77.920%|
| ExtraTrees with inventory |80.110%|79.497%|77.770%|
| Unrestricted VAL-selected family |79.835%|79.497%|78.387%|
| Bounded VAL-selected policy |80.119%|79.835%|78.473%|

Unrestricted VAL chooses coverage_baseline / trees_shadow / trees_core across
the three quotas. TRAIN125 F1 for those selections is77.060% /76.554% /75.319%;
these include fitting/calibration images and are not generalization scores.
At50%, TRAIN F1 geometry73.646%, probe74.360%, recapture73.954%,
agreement73.153%, ridge_shadow74.934%, trees_shadow74.814%.

At50%, the primary probe rule produces TP1691 / FP439 / FN513:2,130 care
predictions,2,193 raw outputs and63 ignored, F1 **78.034%**, TN=0 accuracy
**63.980%**. The established mass control gives TP1706 / FP438 / FN498,
F1 **78.473%**, TN=0 accuracy **64.572%**. The primary does not improve it.

The probe rule's20% advantage is only+.028points, paired95% interval
[-.425,+.490]. Its50% difference is-.439points [-1.324,+.455]. A bounded
agreement variant reaches78.519%, only+.046points [-.144,+.292], and VAL does
**not** select it. All intervals use10,000 fixed-decision image bootstrap
replicates / seed20261023, uncorrected for multiple comparisons; no refitting or
training uncertainty is included. The earlier exploratory PARSeq permutation
candidate78.823% at50% also remains above these new candidates.

No new method is promoted. Full DeepSolo is80.357%; the unchanged-branch
eligibility-constrained GT oracle at50% is82.320%. Neither is a deployable
new routing result. Independent inventories provide information but do not
establish reliable separation on this small, repeatedly reused validation set.

## Runtime and verification

Actual GPU: RTX3090 /24GiB. Sequential FP32 CRAFT batch1 then PARSeq batch32;
noAMP/TF32. Before/after complete model-state hashes are identical; no optimizer
steps and no runtime/OOM failures. Stage totals include loading and state hashing:

| Stage | TRAIN125 seconds | TEST300 seconds | Maximum allocated / reserved bytes |
| --- | ---: | ---: | ---: |
| CRAFT |12.72|23.70|1,399,306,752 /3,955,228,672|
| PARSeq |8.85|17.79|301,876,224 /348,127,232|

TRAIN1,225 and TEST2,538 independent regions;2,450 /5,076 recognition crops.
Active-only deployment requires at most6 extra crops/image: selected357 TRAIN
and843 TEST regions. Extra detector cost is paid on every audited image.
Recorded times omit frozen upstream YOLO/CLIP/caption features and fallback
inference, so avoiding DeepSolo on half the images is **not** a demonstrated
speedup or an8GiB qualification.

39 focused tests PASS. Smoke20 and full425 both complete and pass independent
audits. Full audit verifies7,526 original RGB crop checksums, independent pixel
union coverage,47 independently refitted estimators,1,920 held-out predictions
and312 aggregate routing points. Bounded audit separately verifies204 finite
policy/point checks and protection/quota invariants. An initial indentation error
was fixed before smoke inference; a report-only schema-key error was corrected
after inference. Neither changed scientific outputs or required inference reuse.

Focused tests: `tests/recognition/test_shadow_inventory_router.py`,
`tests/recognition/test_boundary_inventory_router.py`, and the reused
`tests/recognition/test_trainable_image_router.py`.

## Reproduce and inspect

From the repository on PAPER, with the existing detector environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venvs/context-str-text-detectors-py311/bin/python -m scripts.run_shadow_inventory_router \
  --mode approximate \
  --config configs/approximate/shadow_inventory_router_v1.json \
  --paths configs/approximate/shadow_inventory_router_local_paths_v1.json
```

Append `--smoke` for the20-image qualification; `--preflight` checks identities
without model inference. Immutable outputs may be reused only with the same
resolved identity. No environment installation occurs.

Completed unrestricted source:
`/extra1/bmartelli/context-str-artifacts/shadow-inventory-router-v1/full/3f625f3bb35b`.
Its plan/results/audit SHA256:

```text
43e2afacd1dc02a9f9851ba0152d42da8b548ac80af903686596b263e52c6ada
c407101dc578f1066a194d5d71bccaeba0f3366e9b1fbfb004af79f97dd0cd80
cca587ed79c3ddf00d8ce8307a2ad004b2f3d1512d570e50bc6dcc6551f1cca0
```

Bounded source:
`/extra1/bmartelli/context-str-artifacts/boundary-inventory-router-v1/40032f2286f5`,
result SHA256 `a3baa5f9fb27e37767dafa9e4edfa7bf7be56ef6f89f3e9fa3ed54fbe04ac24c`.
Each contains `results.json`, `resolved_configuration.json`, `frozen_choices.json`,
code snapshots and `independent_audit.json`. The unrestricted source also has
`comparison.md`, `all_router_points.csv`, `analysis_summary.json`, raw per-image
regions/readings/features, receipts and every model/CV score.

Auditors: `scripts/audit_shadow_inventory_router.py` and
`scripts/audit_boundary_inventory_router.py`, each with explicit `--root`.
Reporting: `scripts/report_shadow_inventory_router.py`, with explicit `--root`,
`--boundary-root`, `--seed` and `--bootstrap-replicates`.
