# Four controlled graph/occlusion follow-ups

**Approximate reproduction / exploratory research extension.** User-authorized
implementation of recalibration, controlled continuation, richer fragment-graph
training and contour-point GCN refinement. STRICT/default components, prior
weights and reports remain unchanged. These are not official paper results.

## Evidence and controls

The authoritative original `papers/2503.15639v2.pdf`, sections3.2–3.3, and Das et
al., KBS305(2024)112593 pp7/10, were inspected with existing implementations.
Das motivates iterative four-neighbor contour graph deformation, cyclic point
matching and teacher/student training. Exact recipes here are assumptions.
Existing heuristic word-guided contours, graph cut, local affinity/MALIS, old
teacher/student, graph and crop-refinement experiments remain distinct evidence.

The completed source is `graph-occlusion-research-20261023-v1/full`. Its independent
matching audit passes3825 image/arm checks. Baseline TEST F1 is60.700588%; the
epoch6 structural model is60.514372%, with13 more correct words and42 more word
false positives. Epoch6 calibration F1 exceeds epoch12 while training loss falls.
Only291/18563 eligible graph edges are positive same-word training links.

1. **Threshold recalibration:** freeze both networks and the existing pointwise
   filter. Select its threshold on calibration word F1 using the same explicit
   grid for baseline and epoch6 structural model. Compare retuned structure with
   retuned baseline, preserving the original60.70% reference too. This is policy
   calibration using cached recognition, not a new segmentation training gain.
2. **Controlled continuation:** load the actual epoch6 structural checkpoint.
   Compare fresh AdamW at3e-6 versus1e-6 for12 additional epochs, identical order,
   augmentations, losses, original frozen clean teacher, capacity and updates.
   The source checkpoint lacks optimizer/RNG state: both arms explicitly restart
   these identically; this is not exact interrupted-run recovery. Select every
   three epochs on calibration word F1, including unchanged epoch6; also report
   trained final endpoints. ResNet50 remains frozen.
3. **Richer fragment graphs:** cut predicted TRAIN boxes with a configured gray
   RGB strip. The same occluded RGB supplies neural and RGB features. Additionally
   zero that strip in the predicted proposal maps to generate controlled broken
   supports. Preserve raw maps, synthetic maps and cut masks. GT never constructs
   cuts/proposals: owner labels only supervise fit nodes/links and choose a richer
   TRAIN variant. Sample an enriched view at probability.5 when it adds positive
   links, retaining one update per eligible original image. Compare the full2x2:
   self/GCN and standard/enriched, with identical capacity, clean-fit normalization,
   optimizer and30epoch budget. Calibration every5epochs selects checkpoints;
   node.5/link.7 inference policy is fixed for every arm. Identity baseline is
   eligible and trained final endpoints are separate. Clean validation/TEST use
   the exact original proposal graphs. Synthetic map corruption is an assumption
   about useful training diversity, not evidence of real occlusion transfer.
4. **Contour-point GCN:** preserve the original complete candidate pool, IDs and
   split/merge alternatives. Each rectangle initializes32 clockwise pixel-center
   points. Four residual graph layers, four cyclic neighbors+self and three
   deformation steps predict coordinate offsets. The self-only control has the
   same tensors/features/updates and replaces aggregation with self features.
   TRAIN target is the largest external native v4 word-owner contour, uniformly
   resampled; cyclic normalized SmoothL1 handles start-point alignment. Ignored,
   unmatched, low-overlap or degenerate targets carry no invented label and are
   counted. Control-point geometry produces original-RGB boxes for unchanged
   PARSeq; exact zero offsets replay original boxes. This is a contour refinement
   extension, not the complete Das/TextBPN proposal network or U-Net replacement.

## Runtime and scientific limits

The feature field for contour refinement concatenates64 U-Net decoder features
and two probability maps, bilinearly reduced to128square and stored FP32 on CPU.
CPU bilinear grid sampling retains gradients to current contour coordinates
across device transfers, avoiding nondeterministic CUDA sampler backward. This
output-sensitive resolution/device choice is a recorded HARDWARE_DEVIATION, not
an exact paper setting. Original backbone and PARSeq remain frozen. Point-center
boundary conventions, disconnected owner targets and missing holes are limitations.

Both graph and contour variants include an identity checkpoint option and report
final trained endpoints to expose regressions. Candidate generation, inference,
ranking and filtering never receive GT. Calibration freezes all thresholds and
checkpoint choices before evaluation/TEST. No TEST-based selection or combination.

Reuse1030 fit/100 calibration/125 internal evaluation/300 TEST (2204 care words).
Primary metric is micro maximum one-to-one original unpadded bboxIoU>=.5 AND exact
Unicode casefold, retaining punctuation. All care GT, duplicate predictions and
empty readings remain in denominators; no ignore-based prediction neutralization.
The same pre-OCR ranking admits at most20 unique RGB crops before recognition.
Threshold tuning has additional cumulative calibration cost. Preserve one-seed,
historical TEST reuse, source calibration exposure and known duplicate RGB limits.
The isolated exact-byte restoration of three historical TRAIN JPEGs is inherited;
the user's changed source files are not modified.

## Implementation and execution

- `localize/graph_occlusion_followups.py`: GT-blind cuts, TRAIN sampling, contour
  sampling, graph refinement, cyclic loss and coordinate mapping.
- `scripts/run_graph_occlusion_followups.py`: explicit stages, source/artifact
  verification, matched training, calibration, recognition and reporting.
- `scripts/run_graph_occlusion_followups.sh`: sequential existing-environment GPU
  stages; no installation/download or hidden fallback.
- `configs/approximate/graph_occlusion_followups_v2.json`: fully explicit paths,
  hashes, seed, model/training settings and scientific classifications.
- `tests/localize/test_graph_occlusion_followups.py`: focused scientific tests.

Use an immutable source snapshot; check actual GPU before each stage. FP32,
deterministic algorithms, no AMP/TF32 and explicit batches. The prior identical
U-Net/teacher model set reserved5.68GB in GPU smoke. This run requires6GiB free
after CUDA context initialization, plus an additional configured.75GiB admission
margin before starting each stage. If the GPU is busy, the launcher waits without
interrupting other jobs. Actual smoke must qualify the new model sets.
Record model set, placement, shape, batch, peak allocated/reserved memory and OOM.
Failure propagation preserves failed attempts rather than replacing outputs.

Final reports include selected/final endpoints, delta word F1 versus baseline and
matched controls, paired recovered/damaged GT indices, per-image F1 differences
and conditional image-bootstrap intervals. An independent SciPy maximum matcher
audits every final image/arm. It does not measure training-seed variance.

External root:
`/extra1/bmartelli/context-str-runs/graph-occlusion-followups-20261023-v2`.
Read actual status from `full/status.json`; `full/comparison.md`, final results
and audit become authoritative only after successful completion. Implementation
or smoke success never implies a positive word-F1 gain.


## Qualification and launch status

All81focused tests pass; complete smoke_v1 passes160independent image/arm checks
on20images. Both continuation arms change132student tensors while encoder and
teacher tensors remain identical. On8fit smoke images, positive links increase
from2to19; this is a generator qualification, not an F1 improvement. Contour GCN
and self control each have37826parameters; fragment arms each have51266.
Maximum smoke GPU allocation4,508,058,112bytes/reservation5,649,727,488bytes; no
OOM or failed stage. CPU grid sampling backpropagates through CUDA point tensors.

The full battery is running from the same immutable `source_snapshot_v2` used by
smoke. Threshold recalibration is already complete: calibration selects.55for
baseline and.30for structural model, giving TEST60.646831% and60.043405% respectively.
The unchanged baseline is60.700588%; recalibration has not improved TEST word F1.
These completed threshold results do not establish training outcomes. Continuation,
enriched graph and contour results remain pending; read the live status/results.


## Completed results and failure investigation

The full battery completed successfully. The independent matcher passes8500
image/arm checks and the final manifest contains13633artifacts. Every graph and
contour branch selects the original identity pipeline on calibration; both
continuations retain source epoch6. Trained final endpoints expose regressions:

| Endpoint | TEST word F1 % | Delta from60.700588% baseline |
| --- | ---: | ---: |
| Retuned structural model |60.043|-0.657|
| Continuation original LR, final |60.041|-0.660|
| Continuation reduced LR, final |60.459|-0.241|
| Standard GNN, final |60.243|-0.457|
| Enriched GNN, final |56.925|-3.775|
| Contour self control, final |59.693|-1.007|
| Contour GNN, final |58.686|-2.015|

Post-hoc CPU investigation lives in the external run's `failure_analysis_v1/`:
`report.md`, stage attribution, native-owner graph/contour diagnostics, edge PR
curves, actual crop examples, and certified compatible-pool oracle calculations.
These diagnostics never alter trained models, archived predictions or policies.

Enrichment raises positive TRAIN links291to3186but mixed care-word groups on
internal evaluation increase10to44. Of160baseline-correct TEST words damaged by
the enriched GNN,143lack a valid geometry proposal before OCR. Contour point loss
on internal evaluation improves.10616to.07275while native-owner bboxIoU and word
F1 worsen. Of84damaged TEST words,55have admitted IoU>=.5crops with wrong text.
Spot checks confirm clipped suffix letters in PARLIAMENT/RESTAURANT examples.

A GT-assisted integer program finds1303correct words inside the already admitted
baseline pool versus1187selected. Keeping every baseline output, a restricted
compatible add-only oracle finds71additional TEST words (27internally). This
measures candidate availability; it is not a deployed gain or an expected F1.
The proposed untested follow-up is learning selective additions from natural
TRAIN candidates, comparing matched MLP versus hypothesis-graph selection while
preserving original outputs/crops and the20unique-crop recognition budget.

The investigation also found inherited old crop hashes in archived pre-OCR
contour traces. All19258checked FINAL crop hashes match actual RGB pixels, so
primary metrics are unaffected. The future trace writer now binds the changed
box to actual crop bytes before serializing; a regression test brings focused
verification to82passed. Original immutable snapshots/results remain unchanged.
This is a REPOSITORY_SPECIFIED logging-correctness fix, not a scientific method
change. Historical TEST reuse, source calibration exposure and one seed still
limit generalization claims.
