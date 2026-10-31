# Fragment graphs and occlusion-aware U-Net teacher/student

**Approximate reproduction / exploratory research extension.** The user requested
implementation and GPU tests after rejecting the classical detector route.
This suite implements the first two proposed techniques with matched controls.
Replacing the U-Net decoder is explicitly deferred until graph evidence supports
that separate architecture change. STRICT/default models are unchanged.

## Scientific basis and actual new factors

Before implementation, inspect Das et al., *Soft set-based MSER end-to-end system
for occluded scene text detection, recognition and prediction*, KBS305(2024)112593,
pp5/7/10, and the original local paper `papers/2503.15639v2.pdf`, section3.2–3.3.
Das describes visual proposals, a contour-point GCN, and EMA pseudo-box training;
its complete teacher filtering/schedule is unresolved. Neither paper establishes
the following experimental adaptations. Related primary references:
[TextBPN](https://arxiv.org/abs/2107.12664),
[SemiETS](https://arxiv.org/abs/2504.09966).

Existing deterministic graph cut, local MALIS-inspired affinity heads, candidate
MLP/set filtering, three-epoch photometric EMA consistency and clean-to-occluded
PARSeq token distillation were inspected. These are different controls; previous
negative results are not relabeled as successful graph experiments.

1. **Fragment relation graph:** freeze the qualified M/K U-Net and PARSeq. Split
   existing predicted instances with confident predicted kernel cores and
   deterministic geodesic assignment; retain unseeded remainders. Each atom gets
   GT-blind geometry/RGB/probability features plus mean/std of the frozen64-channel
   visual feature map. A four-layer residual graph predicts text nodes and
   same-word links; union-find builds word boxes. The self-only control has
   identical tensors, initialization, features, edges, supervision, order and
   update count, replacing neighbor aggregation with self features. This is a
   **region graph adaptation**, not Das's contour-deformation GCN. The graph
   cannot create missing visual proposals; caps/drop counts and TRAIN positive
   edge counts are recorded. Incorrect transitive word chaining is a real risk.
2. **Clean-to-occluded U-Net:** compare GT-only occlusion training, GT+pixel
   distillation, and GT+pixel+instance-structure distillation. All receive identical
   images, flips, brightness/contrast and per-word vertical RGB strip occlusions,
   optimizer, seed and twelve-epoch budget. Original clean labels remain targets
   under synthetic occlusion, an explicit amodal-target assumption. A frozen
   qualified teacher sees the clean view; only confident teacher probabilities
   agreeing with TRAIN GT enter distillation. The third arm matches word-wise
   row/column occupancy profiles and background separation bands. It is **not**
   graph-link distillation, a connectivity guarantee or real hidden-pixel GT.
   Keeping the teacher frozen rather than EMA is an assumption intended to avoid
   reinforcing the degradation seen in the previous continuation pilot.

## Fixed populations, controls and metric

Reuse checksum-qualified1030 fit,100 calibration,125 internal evaluation and300
historically explored TEST images. Source segmentation already saw calibration
in historical training. Known duplicate TRAIN/TEST RGBs remain explicit. These
one-seed experiments do not establish independent generalization.

Primary metric: micro maximum one-to-one **original unpadded bbox IoU>=0.5 AND
exact Unicode casefold transcription**, preserving punctuation. All care GT and
all predictions, including duplicates/empty readings, remain in denominators;
no prediction neutralization using ignore annotations. TEST has2204 care words.
The frozen pointwise filter is replayed against its prior60.700588% TEST result.
That improvement came from fewer false positives while losing68 correct words;
it is not evidence of additional word recovery.

Both graph arms use a predeclared final epoch30; only node/link thresholds are
selected on calibration word F1. For U-Net, checkpoint selection uses calibration
word F1 every3epochs, retaining epoch0 on ties; final-epoch results are reported
separately. The fixed pre-OCR ranking admits at most20 unique crops per image
**before** calling PARSeq; the same frozen post-OCR filter applies to all arms.
Calibration tries several policies and has additional cumulative OCR cost.
All evaluation policies freeze before evaluation/TEST inference. TEST GT never
enters training, node proposals, filtering or checkpoint/policy selection.

## Artifacts and runtime

Entry points: `scripts/run_graph_occlusion_research.py` and
`scripts/run_graph_occlusion_battery.sh`; explicit config:
`configs/approximate/graph_occlusion_research_v1.json`. Decisions and sources are
classified in that config and `GRAPH-OCCLUSION-RESEARCH-001`.

Run from an immutable source snapshot. Original RGBs, annotation manifests,
weights, source reports, filter weights and critical implementations are
hash-checked. Exported graph maps must replay frozen source probabilities
exactly. Preserve atom rasters, pooled features, probabilities, edges, scores,
group membership, crops/readings, selections, losses and per-image outcomes.
An independent SciPy bipartite matcher audits every final image/arm count.
Comparisons save paired word recovery/damage, per-image F1 differences and
conditional image-bootstrap intervals; these intervals omit training variance.

The first smoke stopped on a genuine RGB checksum mismatch. An audit found
three TRAIN JPEGs40/149/336 changed at their historical paths. Exact pinned
bytes exist in the local acquisition mirror. An isolated full dataset copy uses
those original bytes; original user data remains unchanged. Configuration pins
`input_restoration.json` and explicitly rebases TRAIN image/annotation paths.
No checksum is weakened or updated to accept changed scientific inputs.

Use the existing main environment; install nothing. Verify actual GPU before
each stage; FP32/no AMP/no TF32, deterministic algorithms and explicit batches.
The ResNet50 encoder stays frozen in all U-Net arms; decoder/heads train. Record
allocated/reserved VRAM, model coexistence, shape and batch; OOM/errors write
FAILED status and propagate. Actual RTX3090 qualification does not qualify8GB
hardware. A completed smoke does not establish an F1 improvement.

External root:
`/extra1/bmartelli/context-str-runs/graph-occlusion-research-20261023-v1`.
Completion and numerical outcomes must be read from `full/results.json`,
`full/comparison.md`, and `full/independent_audit.json` when present.


## Qualification and launch

Focused pytest:69passed. Complete GPU smoke_v2:20images across fit/calibration/
internal-evaluation/TEST,72 independent image/arm matching checks pass. All three
trained U-Net endpoints change132decoder/head tensors; teacher and encoder states
remain identical. Peak allocated4,516,516,864bytes/reserved5,679,087,616bytes;
no smoke OOM. Smoke F1 is not an efficacy result.

The first full admission attempt correctly stopped before execution because an
unrelated workload left less than8GiB free VRAM; its failure remains in
`full_attempt1_vram`. The next full attempt is running from `source_snapshot_v4`,
whose Python bytes equal the passing smoke snapshot. It was observed processing
all1555images on the RTX3090. `launch.json`, `full/status.json` and
`full_attempt2.log` record live state. The source snapshot, all failures and input
restoration receipts are preserved. Completion and word-F1 gains remain pending.
