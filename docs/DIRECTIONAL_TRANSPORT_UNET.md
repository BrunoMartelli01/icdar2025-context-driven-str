# Directional transport and RGB detail decoder study

Status: v1/v2/v4 completed on TEST300 and independently audited; v3/v5 failed
CAL-only qualification and were stopped before TEST. The 70% objective has not been reached. See
`docs/DIRECTIONAL_TRANSPORT_RESULTS.md` for measured outcomes and controls.

This is an **approximate reproduction / new research extension**, motivated by
the user's requirement for symbol-free word F1 of at least 70%, without DeepSolo
and with localization parameters comparable to the qualified U-Net. The strict
paper implementation and its unresolved provenance remain unchanged.

## Architecture and controls

The qualified two-head U-Net contains 58,155,146 total parameters and 32,598,114
trainable parameters. All source tensors are copied with exact key/shape/dtype
checks and strict loading. The ResNet50 encoder and its running statistics remain
frozen. Three arms receive identical FIT image order, paired augmentations and
update budgets:

1. `unet_continuation`: unchanged U-Net, native256 output, further training.
2. `global_detail`: full-resolution RGB detail residual and four low-rank
   directional transport blocks, with spatially averaged gates.
3. `local_transport`: the same active tensors and ownership supervision, with
   location-dependent gates for image-specific word boundaries.

The new modules act inside the dense decoder, before components are extracted.
They are not the previously negative postprocessing graphs over fragments or
contour points. At each decoder scale, projected features exchange information
with four axial neighbors at radii 1 and 4. Training-only owner labels supervise
same-word foreground pairs positively, and different-word/background endpoint
pairs negatively. Invalid and background/background pairs are excluded. Neither
owner labels nor annotations enter the model forward or crop generation.

The RGB branch restores a native512 residual from the original normalized image
and the last decoder feature. Its output layer and transport gains initialize to
zero, preserving the source prediction at initialization through nearest
upsampling. Native512 targets come directly from original owner/mask images with
the existing contact guard; native256 targets are not enlarged. The global and
local arms have identical active parameter counts; the predeclared upper bound
is 110% of the original total.

Directional affinity learning has established prior art in
[Spatial Propagation Networks](https://proceedings.neurips.cc/paper/2017/hash/c22abfa379f38b5b0411bc11fa9bf92f-Abstract.html).
The complementary semantic/detail-stream idea also has precedent in
[Gated-SCNN](https://openaccess.thecvf.com/content_ICCV_2019/papers/Takikawa_Gated-SCNN_Gated_Shape_CNNs_for_Semantic_Segmentation_ICCV_2019_paper.pdf).
These are inspirations, not claims to reproduce those networks or establish
worldwide novelty. Local novelty is the proposed decoder intervention relative
to this repository's already tested fragment/contour graphs and word slots.

## Protocol

The predeclared pilot trains each arm for 12 epochs with checkpoints eligible at
epochs 0, 3, 6 and 12. Original decoder learning rate is 1e-5; new modules use
2e-4. AdamW weight decay is 1e-4, FP32 batch size 4, seed 20261026. The common
CAL100 grid contains 54 policies: text threshold 0.5/0.65/0.8, padding equivalent
to 0/1 native256 pixels, and nine native PARSeq sequence-confidence thresholds.
Kernel threshold is 0.5, true component minimum area is four native256 pixels,
and at most 40 components are admitted before recognition. Area and padding
scale with output resolution. These are explicit research assumptions, not the
paper's strict maximum of ten blocks. Original RGB crops use the same pinned
native PARSeq AR/refine1 runtime in every arm. The recognizer is shared and is
counted separately from localization parameters.

FIT has 1,028 images after excluding TRAIN81/212, exact RGB duplicates of two
TEST images. CAL100 selects both checkpoint and policy. All selections are sealed
before internal125 and TEST300 evaluation. The source checkpoint previously saw
those duplicate TRAIN images; their removal does not undo source pretraining.
The warm-start source was trained on the 100 images now used as CAL; they are
not pristine validation images. CAL and TEST have also been explored historically.
Consequently, this is an
exploratory paired comparison, not fresh held-out proof of generalization.

Primary metric: micro word F1, Unicode casefold followed by retaining `isalnum`
characters; maximum one-to-one matching requires exact normalized text and
unpadded original-coordinate bbox IoU >= 0.5. All care words and all emitted
predictions count. Empty normalized outputs cannot match; duplicates still count
as predictions. Predictions in ignore regions are not neutralized. Lexical F1
without geometry is a separate secondary diagnostic. Pixel F1 is not the target.
DeepSolo is never invoked: 0/300 images (0%) in the full TEST experiment.

The previous corrected endpoint scores 64.9573% on 300 images / 2,204 care words
under its existing padded-box convention. This is a historical reference, not a
matched architectural baseline: the new study must compare its common unpadded
box protocol and common admission policies across all arms before attributing
any change to architecture or normalization.

## Evidence and execution

Implementation: `models/directional_transport_unet.py`,
`segmentation/directional_transport.py`,
`recognition/directional_transport_metrics.py`,
`scripts/run_directional_transport.py`.
Configuration: `configs/approximate/directional_transport_v1.json`.
Tests: `tests/models/test_directional_transport.py`.

External evidence root:
`/extra1/bmartelli/context-str-runs/directional-transport-unet-20261026-v1`.
Source inputs, code snapshot, resolved configuration, split IDs and hardware
are saved with hashes. Checkpoints preserve optimizer and RNG state. CAL and
evaluation preserve probabilities, proposals, raw OCR/cache hashes, selected
policies and per-sample matches. Allocated/reserved GPU peaks and OOM failures
are recorded; batch size is never silently reduced.

The first smoke attempt (`smoke-01`) failed before GPU execution because three
previously restored TRAIN image paths already used the qualified root. The
resolver now explicitly accepts the historical root or the configured qualified
root and rejects any other location. The failed receipt is preserved. The next
attempt uses a fresh directory; this is an engineering input-path correction,
not a model substitution or scientific-output change.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venvs/context-str-main-py311/bin/python \
  -m scripts.run_directional_transport \
  --config configs/approximate/directional_transport_v1.json \
  --mode approximate \
  --output /explicit/external/fresh-run
```

Scientific classifications and unresolved questions are recorded in
`docs/PAPER_AMBIGUITIES.md` under `DIRECTIONAL-TRANSPORT-001`.


## CAL-only diagnostic and snapshot qualification

A privileged CAL100 diagnostic passes the original annotated axis-aligned word
boxes, with zero padding, to the unchanged PARSeq. It keeps all802 care-word
predictions without confidence filtering. This yields734 correct words, or
91.5212% symbol-free word F1. The initial pipeline yields565 correct words and
72.3895% F1 under its CAL-selected admission policy. GT crops recover173 words
missed by that pipeline, including140 words without an overlapping selected proposal,
and lose4 previously correct words. This is evidence of recoverable localization
and crop quality failures, not a deployable method or mathematical upper bound.
Only CAL was inspected; these outputs never enter model training or TEST selection.
The labeled receipt is `full/calibration_gt_crops/classified_report.json`, with
source-report hash and per-sample predictions retained.

The second smoke attempt stopped at the free-memory precondition while other
GPU work was active; it did not cause an OOM. Smoke03 completed all three arms
with8 FIT /2 CAL /2 internal /2 TEST images. Twenty-five focused tests passed.
The new models have58,226,736 total /32,669,704 trainable parameters, an increase
of71,590 total parameters (+0.1231%). Measured new-arm smoke training peaks are
about5.20GB allocated /6.53GB reserved (decimalGB), FP32 batch4,3x512x512 input.
These smoke scores do not support full-dataset performance claims.

A resume against the live checkout correctly rejected two unrelated concurrent
script changes. Reproducible resumes use the archived `source` tree and copied
`source/configs/approximate/directional_transport_v1.json`. Three loaded pipeline
dependencies outside the runner's five initial source roots are preserved in
`supplemental_source_manifest.json`; their hashes match the pre-task workspace
capture. A CPU-only frozen-snapshot identity preflight passed. This preflight
mocked GPU setup and is not a GPU-training resume qualification.

The pre-existing environment has no pip module. Package inventory therefore uses
standard-library importlib.metadata; one unnamed distribution is retained with
a null name. No packages were installed or upgraded. The failed pip invocation
and initial unnamed-package sorting attempt are recorded with the final inventory.


The initial CAL error decomposition separates generation from confidence admission:
838 raw proposals become759 selected predictions.135 GT words have no raw box
at IoU>=0.5, whereas170 have no selected box after admission. Among237 unmatched
GT words,135 have no raw localization,95 have overlapping raw proposals but no
correct reading, and7 have a correct reading rejected by confidence. Therefore
170 must not be interpreted as network-only detection failures. The raw counts
are saved in `full/calibration_initial_error_causes.json`.


## Frozen-source follow-up (v2)

The completed v1 source-continuation arm declines from72.3895% CAL word F1 at
initialization to71.2021% /71.5415% /69.0349% at epochs3/6/12. The global-detail
control reaches72.2112% /70.6806% /70.5731%, also below initialization. These are
CAL observations, not TEST conclusions. They motivate a separate hypothesis:
updating the already useful source decoder and BatchNorm statistics may damage
its prior while new paths are still learning. This is not yet proof of causality.

`configs/approximate/directional_transport_frozen_v2.json` freezes every source
weight and every source BatchNorm statistic, and optimizes only71,590 new
parameters. It uses a zero-epoch `unet_reference`, matched global/local six-epoch
budgets and CAL selection at epochs0/3/6. Total deployed parameters and all
postprocessing/recognition rules remain the same. A focused training-step test
verifies exact preservation of every source tensor and statistic while the new
residual output learns. All27 focused tests pass. V2 is a separate approximate
research experiment, not an overwrite or relabeling of v1.

The archived v1 process stopped partway through global-detail epoch11 without a
failure receipt. A detached launch resumes the completed epoch10 checkpoint
from its frozen source. Rounded batch50/100 losses agree with the interrupted
partial epoch11 log; this is not a bitwise trajectory certificate. A detached
queue waits for successful v1 completion, then runs a v2 smoke test and the full
v2 pilot sequentially, checking hardware/free memory before each GPU run and
propagating any failure. V2 source/config hashes are saved before execution.

A documentation-only correction before any v2 execution replaced inherited
v1 twelve-epoch prose with the resolved six-epoch/zero-reference protocol.
Initial source/config manifests are preserved; no numerical protocol changed.
No claim of70% TEST F1 is made from these CAL or smoke observations.


## Controlled normalization diagnostic

A separate CAL100 intervention keeps the global-detail epoch12 trained weights
and fixes the original source policy (text0.5,padding1,confidence0.4), replacing
only source BatchNorm running buffers. Word F1 falls from70.5731% to33.0639%
(-37.5092points); localized care words fall613to289, correct words548to225.
This directly rejects post-hoc source-statistic restoration as a repair. It does
not test freezing the whole source from initialization. The input checkpoint,
restored-buffer names, fixed policy, model outputs and allocated/reserved memory
are retained in `full/global_batchnorm_diagnostic`.

A checkpoint-only CPU analysis separately measures source parameter drift,
excluding running buffers. Bottleneck relative L2 change is33.083% for unchanged
U-Net continuation and33.864% for global-detail continuation; ResNet encoder
change is exactly0%. Later source groups also change (e.g.global up3:16.335%).
The largest running-mean difference is5.290 trained running standard deviations
in one up0 channel. These observations support substantial decoder/statistic
co-adaptation; they do not establish that drift alone caused the original F1
loss or that frozen v2 will improve it. Learned transport coefficients alone
are not interpreted as feature-change percentages. The reproducible CPU script
and qualified receipt are `analyze_source_drift.py` and
`full/global_batchnorm_diagnostic/source_drift_qualified.json` externally.

V2 retains all eligible evaluation-epoch model checkpoints in addition to
best/last. This retention choice was sealed before its first execution and
changes neither training arithmetic nor the predeclared selection grid.
