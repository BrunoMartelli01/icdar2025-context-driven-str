# Corrected U-Net branch with minimum DeepSolo image fallback

User-authorized approximate reproduction / exploratory routing experiment.
The starting branch is the exact62.895928% spatial-word-F1 endpoint, with frozen
U-Net, native PARSeq, existing GNN, RGB verifier and text-only edit generator.
This is not the separate YOLO router. Authoritative paper Section3.5, Figure4,
Algorithm2 and Table5, existing grounded recovery and qualified DeepSolo cached
interfaces were inspected before implementation.

## Objective and protocol

Primary nominal target is strictly above79.8%, the paper's largest Total-Text
Table5 value;78.4% and80.0% are explicit secondary targets. Table5 calls these
recognition accuracy and does not establish an operational evaluator. Treat
crossing the number as a nominal comparison, not verified paper superiority.
The paper's Total-Text fallback image count is unavailable.

Preserve micro maximum one-to-one original bboxIoU>=0.5 plus exact Unicode
casefold, punctuation retained, all care GT and every output including empty,
duplicate and invalid-geometry predictions. No ignore-based prediction
neutralization is added. A routed image takes the complete light output or the
complete DeepSolo output; there is no GT-assisted word union or transcription
normalization change. Every TEST image and all2204 care words remain counted.

The exact image-wise minimum at target t has an additive diagnostic certificate:
`2*TP - t*(Npred+Ngt)>0`. Sorting heavy-minus-light increments in this expression
gives the minimum number of fallback images knowing GT. This is an oracle lower
bound, never a deployable gate.

The usable router predicts heavy-minus-light TP and output-count differences
from observable light-branch evidence and optional RGB-only frozen CLIP/PARSeq
features. Other OCR word/text arrays and actual DeepSolo outputs are excluded
from inference features. Targets alone use paired TRAIN outcomes. Six matched
tabular/visual Ridge or ExtraTrees recipes share grouped five-fold OOF fitting.
Portable model predictions must replay sklearn exactly within declared tolerance.

Reuse1030 FIT minus TRAIN81/212 pixel-identical TEST copies,100 calibration and
125 internal evaluation. The historical light experts already trained on those
copies and use reused calibration; router exclusion does not erase that exposure.
Four predeclared heavy confidence profiles use complete cached ViTAEv2-S/TextOCR
outputs at the available0.4 detection floor. No missing low-floor TRAIN predictions
are fabricated from the separate improved native/fusion study.

For each target and margin0/0.005, select the smallest fallback quota on a fixed
1%-step grid whose OOF and calibration F1 both exceed the threshold. Fit on FIT,
then freeze model, heavy profile and quota before held-out routed evaluation.
Report an unqualified selection explicitly if no development policy succeeds.
Also save every prefix of each frozen TEST ranking, including its descriptive
minimum crossing. That TEST-dependent point is not a development-selected policy.
Conditional image bootstrap omits model/selection variability.

All fallback counts are logical routing on cached outputs; actual new online
DeepSolo inference is zero. Cold deployment would invoke DeepSolo for the routed
image count. No end-to-end latency or compute saving is inferred from that alone.

## Classification and execution

`ASSUMPTION`: current branch pairing, target interpretation, no-ignore spatial
metric, learned count utility, descriptors, recipes, grouped folds, quota grid,
development constraints and descriptive TEST minimum. `HARDWARE_DEVIATION`:
actual RTX3090 FP32 frozen-head replay on cached features, CPU fitting/evaluation,
determinism, batches and memory recorded. `UNRESOLVED`: paper metric equivalence,
strict artifacts, independent validation, historical exposure and training variance.
STRICT configurations and original artifacts remain unchanged. No installs.

Configuration: `configs/approximate/unet_minimum_fallback_v1.json`.
Implementation: `recognition/unet_minimum_fallback.py`,
`scripts/run_unet_minimum_fallback.py`.
Tests: `tests/recognition/test_unet_minimum_fallback.py`.
Use an immutable source snapshot and explicit mode/config/fresh output:
`python -m scripts.run_unet_minimum_fallback --mode approximate --config PATH
--output PATH`; `--smoke` qualifies engineering, not efficacy.

External root: `/extra1/bmartelli/context-str-runs/unet-minimum-fallback-20261026-v1`.
Completed evidence and detailed analysis: `final_analysis/report.md` under that
external root. Primary `full/`, valid `transfer2/`, `cold_light1/` and
`cold_grid1/` retain separate manifests and selection status.

## Completed outcome

The primary nominal79.8 target has **no successful development-selected TEST
policy**. A fully retrospective frozen-ranking cut reaches79.8095%, using
294/300 DeepSolo images (98%); it saves only6 images and its margin over the
displayed target is0.0095 points. That cut was chosen on TEST and cannot be
reported as independent router qualification or paper superiority.

| TEST endpoint | F1 % | DeepSolo images / total (%) | Selection status |
| --- | ---: | ---: | --- |
| Corrected U-Net / PARSeq |62.8959|0/300 (0%)|Frozen reference|
| Initial79.8 policy |74.0650|114/300 (38%)|OOF/calibration selected; fails TEST target|
| Secondary78.4 policy |79.3354|255/300 (85%)|New OOF selected; passes78.4 target|
| Best new OOF policy for79.8 |79.3911|286/300 (95.33%)|Unqualified even on OOF|
| All heavy, score0.4 / char confidence0.925 |79.7333|300/300 (100%)|Fixed profile control|
| Minimum observed79.8 crossing |79.8095|294/300 (98%)|TEST-selected exploratory cut|
| Minimum observed78.4 crossing |78.4780|175/300 (58.33%)|TEST-selected exploratory cut|

All rows are logical routing on cached full-image DeepSolo predictions; actual
new online DeepSolo inference in this task is0 images. No online latency saving
is measured. The paper's Total-Text fallback count is unavailable.

The initial quota underestimates TEST requirements: the light branch gives
76.1051% on1028FIT images,70.9632% on100calibration,61.6921% on125internal and
62.8959% on300TEST, each with0/N (0%) DeepSolo. Router OOF does not remove light
expert training exposure. The selected38% quota is adequate on the stronger
development branch but fails on TEST. A later explicit research adaptation uses
the125 images excluded from light gradient updates as router development,
adding native OCR confidence and uncovered predicted text/kernel components and
mass. Those images are no longer independent evaluation evidence. Five-fold
OOF selects the255-image policy, recovering540 and losing96 words versus light.
For79.8, the best new OOF policy reaches only79.1983% with119/125 DeepSolo images
(95.2%), and79.3911% with286/300 (95.33%) on TEST. No success is claimed for it.

The complete retrospective12-model/6-profile grid finds no better79.8 crossing
than294/300 (98%) from the original models. That point has1676TP/320FP/528FN,
533 recovered/108 lost versus light, and conditional95% interval
[77.3800%,82.1036%], which additionally omits TEST model/profile/cut selection.
The255/300 (85%) policy has1695TP/374FP/509FN and conditional interval
[77.0380%,81.6808%]. These are not statistical paper-superiority claims.

An exact GT-assisted minimum among the six fixed global fallback profiles needs
96/300 images (32%) to reach79.8124%. This privileged lower bound proves that
better selection is mathematically possible for these frozen outputs; it is
not a usable router or a promised saving. The0.925 profile needs107/300 (35.67%)
under the same privileged diagnostic.

Additional isolated approximate configurations:
`configs/approximate/unet_minimum_fallback_transfer_v1.json` and
`configs/approximate/unet_cold_light_router_v1.json`. Implementations:
`scripts/analyze_unet_fallback_transfer.py`,
`recognition/unet_fallback_coverage.py`, `scripts/run_unet_cold_light_router.py`.
Tests: `tests/recognition/test_unet_fallback_transfer.py` and
`tests/recognition/test_unet_fallback_coverage.py`.

38 distinct unit/regression tests pass; final coverage implementation also
passes its4 dedicated tests. Smoke2 passes260 branch counts/108 routes. Primary
independent audit checks7765 branch counts,2550 routes,4659 input shards and1553
qualified cached DeepSolo responses. Extended audit checks1200 selected TEST
image/policy outcomes and all valid manifests. Frozen-head GPU replay reproduces
425 prior held-out outputs exactly, with39,429,632/41,943,040 bytes allocated/
reserved and no actual OOM. This measures the cached-head stage only.

Preserved failures: smoke1 missing newly required module in the source snapshot,
corrected by pinned dependency; read-only nonexistent filename lookup; invalid
`transfer1/` sample/array ordering, corrected with explicit population alignment
and regression tests before valid `transfer2/`. Invalid results are excluded.
No packages, weights, original datasets or strict defaults changed. Qualification:
`provenance/unet_minimum_fallback_v1_qualification.json`.
