# V3/447 word graph cut

**Approximate reproduction / exploratory post-processing experiment.** This
implements the user's requested graph-cut test with fixed U-Net and PARSeq.
It does not implement the other proposed MSER, super-resolution, random-walker
or global candidate-selection methods. No OCR gain follows from implementation
or smoke qualification alone.

## Matched source

V3 epoch447 is frozen from the previous TRAIN bbox Word-F1 selection, SHA256
`5b44de7b88b5b1d636e8f53c586565f6491aabaae3ccea361e96e32d540c1faa`.
The source run is `guarded-v3-training-20261017`. Its executable snapshot supplies
the existing model, preprocessing, genuine pinned PARSeq and evaluation helpers;
new graph-cut modules/config/tests are added in a separate immutable snapshot.
No weights are duplicated or updated. No installation is performed.

RGB256 stretch preprocessing, native128 probability threshold.5, native pad1,
area0, top10 and historical padded-bbox ordering reproduce the existing source.
The baseline is compared byte-for-byte against all300 historical native binary
mask hashes and selected boxes; exact OCR word IDs must replay the795 result.

The primary metric is micro maximum one-to-one original **unpadded** boxIoU>=.5
and exact Unicode-casefold text; punctuation stays. All2204 TEST care words and
all predictions remain in denominators, including duplicate/empty readings.
This is not official Table5 accuracy or a context/DeepSolo pipeline evaluation.
The legacy bbox-area/ranking semantics are retained for comparability; they do
not become the strict paper component-area implementation.

## New partitioning method

Only the original ten selected parent components are eligible. Each can emit at
most two word instances; after replacement, the original ranking and cap10 are
reapplied. This isolates separation and cannot recover absent foreground or
parents already excluded by the source cap.

Threshold/erosion cores provide two shared marker regions. These marker methods
are reused existing mechanisms, not the scientific novelty. Bounded recipes and
geometry gates are explicit in `guarded_v3_graphcut_v1.json`; two largest eligible
cores are used, with stable tie-breaking. This is not guaranteed to identify two
words rather than two parts of a single word.

Inside each parent's RGB ROI, nearest support/seed scaling gives a2x native
graph grid; original-RGB pixels are resampled to it and probabilities use linear
interpolation. This adds RGB boundary evidence, not new neural segmentation
detail. Outside the original foreground support the zero/background label is
fixed. The first version optimizes **two word labels**, not three free classes,
and never erases foreground to manufacture separated components.

The energy sums normalized Euclidean seed-distance unary costs and8-neighbor
Potts discontinuity costs. Affinity is proportional to squared minimum endpoint
probability, an explicit positive floor, RGB similarity and inverse step length.
The pairwise/unary balance has two predeclared values. Integer capacities use
scale256; seed penalties exceed the total finite capacity. Overflow raises an
error. SciPy Dinic returns an exact cut for this quantized binary energy; the
integer cut energy must equal maximum flow. No unquantized/global multiclass
optimality is claimed. Small synthetic energies are independently enumerated.

Each proposed child must satisfy connectedness, minimum area/short side, mass
balance and bbox-overlap gates. All support pixels remain assigned to exactly
one label. Native coordinates can be half pixels; original lower/upper edges
use floor/ceil and clamp. Keep label maps per instance; unioning them and running
connected components again would lose the separation.

## Controls and calibration

The matched control is the existing deterministic minimax geodesic flood,
using exactly the same marker pairs, RGB ROI/grid, support, geometry gates and
cut-quality calculations. Its different assignment algorithm is explicit.
The graph cut adds seed-distance unary and global discontinuity optimization;
the control is not claimed to optimize the same energy.

There are19 predeclared policies: baseline/off, six geodesic gate combinations
and twelve graph-cut combinations. Candidate choice within a parent uses cut
affinity/probability and area balance, without OCR or annotations. Policies are
chosen on the source's125 TRAIN holdout images /983 care words; these images
have been explored repeatedly and are not an independent fresh validation set.

Promotion maximizes correctly read+localized words subject to nondecreasing OCR
F1 relative to baseline; ties prefer fewer changed images/operations and off.
Best active diagnostics are also recorded, even when off wins. Selection JSON
is hashed before TEST. TEST evaluates baseline, selected/promoted policies,
best-active policies, and the geodesic control with the graph-cut chosen gates.
No TEST results select parameters, models or promotion.

Every phase freezes all geometries and PARSeq readings before loading that
phase's word annotations. Existing exact-crop PARSeq readings can be reused by
RGB hash; new crops use the same pinned official AR/refine1 adapter. Crop RGB
never reaches disk. Exact FP32 probability NPZ and compressed numeric/RLE
partitions/proposals are retained for replay and independent audit.

## Runtime and qualification

Actual hardware is PAPER RTX3090/24GiB. CUDA FP32 segmentation batch1 is unloaded
before PARSeq batch32; graph cuts run on CPU using the existing SciPy. Explicit
determinism, no TF32/AMP and CUBLAS_WORKSPACE_CONFIG=:4096:8 are recorded. Runtime
peaks and OOM failures propagate; no model/precision/device fallback occurs.

35 targeted tests pass: exact min-cut vs exhaustive enumeration, support and
hard-marker preservation, diagonal connectivity, holes/weak bridges, deterministic
ties, invalid/overflow input, coordinate mapping, crop cap, policy promotion,
RLE and previous crop/V3 regressions. The first CUDA smoke reached genuine PARSeq
but failed in evaluation because a crop rank required by the existing metric
adapter was absent. Rank propagation was fixed with a regression test; the failed
root/source remains preserved. An initial packaging error was a missing test-only
directory and is separately recorded. No scientific error was suppressed.

Corrected CUDA smoke:4 TRAIN-validation images,36 partitions,53 unique OCR crops,
all19 policies audited, statusSUCCESS; peak allocated458178048B/reserved522190848B,
no OOM, zero individual crop export. This is a smoke result, not efficacy evidence.

The corrected executable snapshot has614 hashed files. Results are external under
`/extra1/bmartelli/context-str-runs/guarded-v3-graphcut-20261021-v2/`:

- `code_snapshot/`, `source_snapshot_manifest.json`, `launch.json`, `full.log`;
- `smoke/`: the successful separate smoke;
- `full/status.json`, `protocol.json`, `selection.json`, `comparison.json`;
- `full/validation/` and `full/test/`: exact probability maps, compressed
  partitions/proposals, new OCR JSONL, GT, per-policy results and independent
  NumPy/SciPy per-image audits.

All parameters/paths are explicit in the isolated approximate config. For a
fresh output directory, from the frozen snapshot:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 /explicit/main/python -u -m scripts.run_guarded_v3_graph_cut \
  --config configs/approximate/guarded_v3_graphcut_v1.json \
  --output /explicit/new/output
```

Scientific classifications, unresolved limits and source links are recorded in
`docs/PAPER_AMBIGUITIES.md`, GRAPH-CUT-WORD-001. This document records launch
qualification; the completed `comparison.json` and its analysis determine efficacy.


## Completed result (2026-10-21)

All125 validation and300 TEST images are complete; independent OCR audits pass.
Validation selects graphcut-e0-p0.65-a0.3:331 versus320 correct words, OCR F1
35.65% versus35.01%. This fixed policy gives775 versus795 correct TEST words,
50 recovered/70 lost, OCR F1 35.57% versus37.06%. The active geodesic control
has793 correct and F1 36.78%; its validation promotion remains off. Native TEST
mask and baseline box/word-ID replay passes300/300. Source614 files and frozen
selection remain exact. The review additionally audits4021 partitions across425
images, support preservation/connectivity, and2654 integer max-flow certificates.

Of70 baseline words lost:50 lose adequate split geometry,10 remain localized but
are read incorrectly,10 lose their original unsplit crop at cap10. This version
therefore stays experimental; the baseline remains the recommended reference.
No thresholds were retuned from TEST. Exact trial evidence and examples:
[review](../artifacts/diagnostics/guarded-v3-graphcut-20261021/README.md).

Full run peak allocated586276864B/reserved648019968B, no OOM, zero individual
crop files, about30MB numeric outputs. Total experiment time68.22s is not a
single-policy inference latency benchmark.
