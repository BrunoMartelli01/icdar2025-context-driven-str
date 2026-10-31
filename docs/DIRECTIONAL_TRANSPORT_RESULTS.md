# Directional decoder word-F1 results and failure analysis

Approximate reproduction / exploratory research extension. The 70% symbol-free
word-F1 objective has **not** been achieved. Completed v1/v2/v4 results are audited;
v3/v5 failed CAL-only qualification and were stopped before TEST.

## Completed comparison

All rows evaluate Total-Text TEST300 /2,204 care words. Primary matching requires
exact Unicode casefold-plus-isalnum text and unpadded original bbox IoU>=0.5,
with maximum one-to-one assignment. Every emitted output and care word counts;
empty normalized predictions cannot match and ignored-region predictions are
not neutralized. Native PARSeq is fixed. DeepSolo: **0/300 images (0%)** in every
full TEST row. The source decoder has58,155,146 total parameters; each custom
decoder has58,226,736 (+71,590; +0.1231%).

| Experiment | CAL-selected epoch | Correct words | FP | FN | TEST word F1 | Change versus source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| U-Net reference | 0 | 1400 | 924 | 804 | 61.8375% | 0.0000pp |
| v1 global/detail, source trainable | 0 | 1400 | 926 | 804 | 61.8102% | -0.0273pp |
| v1 local transport, source trainable | 0 | 1400 | 926 | 804 | 61.8102% | -0.0273pp |
| v2 global/detail, entire source frozen | 6 | 1367 | 732 | 837 | 63.5371% | +1.6996pp |
| v2 local transport, entire source frozen | 3 | 1361 | 754 | 843 | 63.0238% | +1.1864pp |
| v4 global/detail, continued to epoch12 | 6 (retained) | 1367 | 732 | 837 | 63.5371% | +1.6996pp |

V1 selects epoch0 because trained checkpoints worsen CAL word F1. Its tiny
TEST difference is nearest-upsampled mask/BFS discretization, not learned
architecture improvement. The v2 global model gains17 previously wrong words
and loses50 relative to the independently calibrated source, net33 fewer correct
words. F1 rises because it emits192 fewer false positives. The local model gains14
and loses53, with170 fewer false positives. Thus the observed improvement is
primarily precision, not recovery of more words.

## Architecture versus operating policy

Both policies below were selected on CAL before the parent TEST evaluation.
These factorial controls introduce no TEST-selected thresholds:

| Architecture | Source policy: text0.5 /OCR0.4 | New policy: text0.65 /OCR0.6 |
| --- | ---: | ---: |
| Source U-Net | 61.8375% | 62.9940% |
| Frozen global/detail | 61.7153% | 63.5371% |
| Architecture change at fixed policy | -0.1222pp | +0.5431pp |

Padding is one native256-equivalent pixel in every cell. Of the original
1.6996-point selected-system gain,1.1565points are also obtained by changing the
source policy alone. At the newer common policy, the architecture gains42 words
and loses30, net12 additional correct words; FP decreases743to732. The paired
5,000-draw bootstrap95% interval for its F1 change is[-0.2447,+1.3478]points.
This is a small, policy-dependent point estimate, not robust proof of a superior
architecture. The independently selected-system interval [+0.854,+2.547]points
must not be substituted for that architectural interval.

At the newer common policy, all net gains are in the relative-height>=8% group:
640to652 correct. Medium-height words remain610/956 and small-height words
105/257; gains and losses cancel in those groups. These are recall counts,
not category F1: false positives have no unique GT category. The proposed
small-word recovery mechanism is not supported by these results.

The previous corrected project endpoint is64.9573% under its existing padded-box
and selection convention. The custom decoder has not established a new project
best; that historical endpoint is not a matched architecture-only comparator.

## Why the tested paths fail

**Continuing to train the source damages its prior.** Source continuation drops
from72.3895% CAL word F1 to69.0349% after12epochs. Global/detail ends70.5731% and
local transport69.8539%. Source bottleneck relative parameter L2 shifts are
33.083% and33.864% for source/global continuation, despite learning rate1e-5;
the frozen encoder changes0%. This is substantial decoder adaptation, not merely
training the new71,590 parameters. It does not by itself prove causality for F1
loss. Freezing the full source produces the better v2 results, but the gain
remains small after controlling policy.

**Restoring old normalization afterward is incompatible with updated weights.**
Holding global epoch12 weights and policy fixed, replacing source BatchNorm
running buffers drops CAL word F1 from70.5731% to33.0639%; localized words fall
613to289. Some running-mean shifts exceed five trained running standard
deviations. This directly rejects post-hoc source-statistic restoration. It does
not reject freezing both weights and statistics from the start.

**Local relations do not improve the winning global control.** V2 local transport
is0.5132points below global/detail with identical parameter counts. Its local
edge predictions may support the auxiliary objective without improving final
word masks/crops. The evidence does not support pursuing this spatial branch.

**Ownership closedness is not a reliable admission signal yet.** A new readout
subtracts outgoing learned same-word affinity from internal affinity, using
image features only. It is designed to reject partial-word components without
adding parameters. On CAL, tuning its threshold appears to improve72.8000% to
73.6495%. In five image-group folds, however, it reaches72.4208%, versus72.2078%
for confidence-only and below the fixed72.8000% baseline. It fails the predeclared
minimum0.5-point improvement requirement. The pilot is stopped without another
TEST evaluation. The apparent fitted-CAL gain is not accepted as evidence.

**The remaining gap includes both localization and reading.** On initial CAL100,
135 unmatched words have no raw proposal at IoU>=0.5;95 have a raw overlapping
proposal but no correct reading, and7 have a correct reading rejected by
confidence. After admission,170 words have no overlapping selected proposal;
those170 must not all be called network-only misses. Privileged GT-box crops
yield734/802 correct (91.5212%), recover173 source errors and lose4 correct
outputs. This demonstrates recoverable geometry/crop quality, but is neither a
deployable method nor a mathematical performance upper bound.

## Continuation and graph-decoding outcomes

V2 global CAL F1 rises72.3895% ->72.4618% ->72.8000% at epochs0/3/6, with its best
checkpoint at the budget boundary. `configs/approximate/directional_transport_continuation_v4.json`
therefore continues only that model, preserving optimizer/RNG state, architecture,
inputs, targets, augmentation and learning rates. A smoke test verifies real GPU
continuation, and full CAL replay at epoch6 reproduces parent counts/policy.

Training was planned in six-epoch blocks, at most24total epochs. Another block
required a CAL improvement of at least0.1F1points and an advancing selected epoch.
At epoch12 CAL falls to71.8291%, versus72.8000% at epoch6. The controller stops;
epochs18/24 are not run. Final selection retains epoch6 and TEST remains63.5371%.
Both TEST prediction and per-sample files are byte-identical to v2. This tests
additional training in the same regime, not every possible optimization schedule.
All452 original source tensors, including normalization buffers, are unchanged
in v2 global/local best checkpoints and v4 epoch12. The source-freezing intervention
is verified; further source drift cannot explain this continuation failure.

V5 tests learned ownership affinities as a conservative instance splitter, using
mutual axial/diagonal graph edges, at least two sufficiently large cores and
deterministic fine-grid reassignment. It preserves all foreground pixels, never
merges original instances and adds no parameters. Each tested threshold splits
zero or one CAL instance. Its fitted-CAL best rises72.8000% to73.0180% (two more
correct words), but five-fold held-out F1 is72.2078%, identical to confidence-only
selection and0.5922points below the fixed parent baseline. It fails the same
predeclared minimum0.5-point qualification criterion. No TEST evaluation is run.
The gain depends on a single selected example; transferable word separation is
not established. Weak affinity fields and restrictive safeguards are possible
limits, but their individual effects have not been isolated experimentally.

V3 and v5 use DeepSolo0/100 CAL images (0%); no TEST result is available for either.

The custom design is locally new in this repository: dense decoder transport
plus RGB residual detail, rather than another post-hoc contour/fragment graph.
Spatial propagation and semantic/detail streams have established prior art.
There is no worldwide novelty claim. The strongest current variant uses globally
averaged gates; neither spatial ownership transport nor its closedness readout
has shown a dependable benefit; v5 also fails its independent-fold gate. Results are single-seed and historically
exploratory; the warm-start source already saw CAL100 and two TRAIN/TEST duplicate
images. Strict paper-reproduction blockers remain unchanged.

## Evidence

Machine-readable projection: `provenance/directional_transport_results.json`.
Implementation and protocol: `docs/DIRECTIONAL_TRANSPORT_UNET.md`.
Source files include `models/directional_transport_unet.py`,
`segmentation/directional_transport.py`, `recognition/directional_transport_metrics.py`,
`localize/directional_closedness.py`, `scripts/run_directional_transport.py`,
`localize/directional_affinity_split.py`, `scripts/run_directional_closedness.py`,
and `scripts/run_directional_affinity_split.py`.
Configurations: `configs/approximate/directional_transport_v1.json`,
`configs/approximate/directional_transport_frozen_v2.json`,
`configs/approximate/directional_closedness_v3.json`,
`configs/approximate/directional_transport_continuation_v4.json`,
`configs/approximate/directional_affinity_split_v5.json`.
Tests: `tests/models/test_directional_transport.py`,
`tests/localize/test_directional_closedness.py`,
`tests/localize/test_directional_affinity_split.py`,
`tests/segmentation/test_totaltext_word_kernel.py`.

The external root is
`/extra1/bmartelli/context-str-runs/directional-transport-unet-20261026-v1`.
`full/independent_audit.json`, `frozen-v2-full/independent_audit.json`, and
`continuation-v4-epoch12/independent_audit.json` pass
standalone IoU/normalization/Hungarian checks on every TEST prediction and verify
declared code snapshots, input/checkpoint hashes and5,600 finite probability maps.
Per-word gains/losses and fixed-policy intervals are retained in
`frozen-v2-full/cross_policy_controls/paired_analysis.json`.
The completed focused test suite has37passing tests; GPU OOM is not observed in
completed model runs. Hardware, model sets, batch/shape and allocated/reserved
peaks are recorded in each run; cached OCR stage timing is not online latency.
