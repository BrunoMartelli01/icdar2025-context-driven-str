# CLIP whole-image router: separate TRAIN and TEST

This is an **approximate reproduction**, an experimental extension of paper
section3.5/Algorithm2, and has no Table5 or strict reproduction claim.

The frozen source is Text+Kernel epoch61 (checkpoint SHA256
`e65691dda944478e9d8b4f022d45d0d3b21edc7c995274554b4c5ca31336766f`).
Its native word detection F1 on the held-out TRAIN125 is .621272. The original
seeded watershed recipe retains unseeded text components: text threshold .65,
kernel threshold .5, padding0, no area filtering or crop cap. Do not substitute
the epoch2 control, Hybrid384, or the seeded-only TEST detection recipe.

TRAIN here means the125 held-out TRAIN images already used in the CLIP study,
983 care words. They are exploratory validation data used for model selection,
not an independent test or a measurement over all1255 training images. TEST
contains all300 official images,2204 care words. Aggregates remain separate.

The router uses CLIP ViT-B/16, the minimum over **every** original PARSeq output
of the geometric mean of two TRAIN80 mid-CDF ranks: PARSeq balanced character
confidence and CLIP word-prompt ensemble minus background ensemble similarity.
Empty output or a blank transcription forces whole-image fallback. The chosen
threshold .146906 accepts62/125 TRAIN images. It comes from the preceding
TRAIN observable-score coverage sweep. Model/family selection used exploratory
TRAIN labels. Both reference distributions and the threshold are frozen before
TEST. No GT, ignore mask, TEST labels, or TEST quantile fits enter routing.

Accepted images emit all original PARSeq words; rejected images emit all
DeepSolo detections from the original full image at instance score>=.4, without
lexicon correction. The joined `selected_output` string is never evaluated as
a replacement for these detections. The isolated, previously qualified host
runtime is reused; Docker is unchanged. Main models are loaded sequentially.

Existing epoch61 CUDA batch1 TEST maps are checked against the frozen complete
run/checkpoint/input/source contracts and newly sealed with per-file digests.
They lack historical per-map checksums; that provenance limitation remains.
TEST DeepSolo responses were already computed on all300 images with threshold
.05. The pinned upstream `adet/modeling/text_spotter.py:269` computes network
outputs before applying an inclusive `scores>=INFERENCE_TH_TEST` mask. Thus a
second inclusive filter>=.4 selects the same retained network predictions.
No NMS or lexicon step modifies those raw responses. Original responses keep
their actual .05 threshold and individual runtime/memory records; conversion
receipts record the effective .4 threshold. This is a code-based equivalence,
not a new GPU repeatability experiment.

Evaluation is micro word aggregation, maximum-cardinality one-to-one matching
with original exclusive bbox IoU>=.5 and exact Unicode casefold text. Punctuation
is retained. Care GT only. Report both detected-word precision/accuracy
`TP/(TP+FP)` and user-requested `TN=0` accuracy `TP/(TP+FP+FN)`, plus recall/F1.
Missing words remain FN, empty predictions remain in the image denominator, and
failed images prevent a complete aggregate.

Evaluation-only ignore uses the exact U-Net native support. For DeepSolo it
uses the boundary polygon: invert OpenCV EXIF orientation to stored RGB, round
vertices with NumPy nearest-even, rasterize in original RGB with OpenCV,
letterbox and resize nearest to256. Ignore only nonempty native support with
zero valid pixels. Partial overlap keeps the entire detection. A tiny polygon
with no native raster pixels remains care; a finite polygon with an empty
clipped envelope remains an unmatchable FP. Malformed/nonfinite polygons fail.
This is an explicit approximate metric assumption, not official DetEval. The
validity preprocessing replay matched all125 original TRAIN masks byte-for-byte.

Configuration and paths are isolated in
`configs/approximate/clip_router_e2e_v1.json` and
`configs/approximate/clip_router_e2e_local_paths_v1.json`. Fully resolved plans,
input hashes, code hashes, all raw PARSeq probabilities, CLIP evidence, routing,
DeepSolo responses and per-image evaluation are preserved outside Git. The
main device is explicit; FP32 PARSeq batch8/CLIP batch16 and isolated single-image
DeepSolo CUDA execution are hardware deviations. Cross-device/batch numerical
equivalence is unproven. Allocator peaks and OOM/error receipts are retained.

```bash
# Read-only input/runtime provenance checks.
.venvs/context-str-main-py311/bin/python -m scripts.run_clip_router_e2e --mode approximate --check

# Prepare both splits with CPU PARSeq/CLIP and existing segmentation maps.
.venvs/context-str-main-py311/bin/python -m scripts.run_clip_router_e2e --mode approximate --device cpu --stage prepare --split both

# Evaluate TEST using its qualified cached full-image DeepSolo responses.
.venvs/context-str-main-py311/bin/python -m scripts.run_clip_router_e2e --mode approximate --device cpu --stage evaluate --split test

# On the GPU host: resume prepared outputs, execute missing TRAIN DeepSolo,
# and save separate TRAIN/TEST reports plus results.json with separate entries.
.venvs/context-str-main-py311/bin/python -m scripts.run_clip_router_e2e --mode approximate --device cpu --stage complete --split both

# Equivalent resumable host launcher.
bash scripts/run_clip_router_e2e_host.sh
```

Keeping `--device cpu` on the host resumes existing CPU PARSeq/CLIP outputs;
DeepSolo independently uses its qualified CUDA subprocess. Choosing CUDA for
the main models creates a separately identified run and records new numerical
outputs. `--limit` creates a separately identified smoke result, never a full
split claim. `--stage evaluate` never executes a missing DeepSolo prediction.

Initial validation: all425 input images and candidate/source artifacts passed
preflight; frozen TRAIN score replay accepts exactly62/125. The new evaluator,
router boundary, polygon/EXIF geometry, word localization and DeepSolo contract
suite passed97 tests. A first broader pytest command named a nonexistent
`test_host_adapter.py` and exited4 before collecting tests; the corrected suite
uses `test_host_setup.py` and passes. A real CPU PARSeq/CLIP plus cached CUDA
DeepSolo two-image TEST smoke completed with all intermediate/raw outputs at
`/tmp/clip-router-e2e-smoke-v1/4b1bea8f8c24/test/results.json`. This smoke is not
the complete TEST metric. New scientific files were reviewed against the
initial workspace snapshot; unrelated existing changes remain intact.

## Complete TEST300 result (2026-10-12)

No TEST fitting. B16 threshold .146906 accepts149/300 images (49.6667%), with151
whole-image DeepSolo routes. Every300 image and2204 care GT word is evaluated,
zero failures omitted. TRAIN remains separate:62 accept/63 fallback on125
held-out images; its combined result awaits those63 GPU predictions.

| TEST300 model | TP | FP | FN | Accuracy on detected words | Accuracy TN0 | Recall | Word F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Text+Kernel/PARSeq |1215|1294|989|48.4257%|34.7341%|55.1270%|51.5595%|
| Text+Kernel/PARSeq → CLIP router → DeepSolo |1459|610|745|70.5172%|51.8479%|66.1978%|68.2893%|
| DeepSolo on all300 (same frozen raw cache, independent control) |1588|420|616|79.0837%|60.5183%|72.0508%|75.4036%|

The final router output contains2128 total detections,59 ignored,2069 care.
PARSeq alone has2756 total,247 ignored,2509 care; DeepSolo alone2040 total,32
ignored,2008 care. Zero retained detector envelopes are empty after clipping.
For the149 accepted images, PARSeq contributes609 TP/369 FP/368 FN (978 care
outputs,977 GT). For the151 rejected images, DeepSolo contributes850 TP/241 FP/
377 FN (1091 care outputs,1227 GT). Thus the half-coverage gate improves both
metrics over PARSeq alone, while losing8.6704 percentage points of TN0 accuracy
against invoking DeepSolo on all300. The experiment measures invocation count;
it does not establish a wall-time or FLOP saving including CLIP.

Machine-readable outputs:

- `/tmp/clip-router-e2e-v1/000cf54b998f/resolved_configuration.json`
- `/tmp/clip-router-e2e-v1/000cf54b998f/test/results.json`, SHA256
  `03050c4c1dac1687c4154c2c51913efdabd74f6ecf633250a9ef66a339618123`
- `/tmp/clip-router-e2e-v1/000cf54b998f/test_independent_audit.json`
- independent audit source `/tmp/audit_clip_router_e2e.py`

The independent audit passes all300 samples using SciPy binary assignment,
separate CDF calculations, separately written inverse EXIF/polygon geometry and
PIL letterbox/nearest ignore. It exactly reproduces all router scores (maximum
difference0), original low-threshold source IDs, final per-image TP/FP/FN,
all aggregate counts and both ratios. It also evaluates the all300 DeepSolo
control without changing routing or selecting another TEST threshold.

Main TEST inference was CPU FP32, with four CPU threads: PARSeq batch8
`3x32x128` crops71.6312s, CLIP batch16 `3x224x224` crops411.4248s summed over
per-image inference-stage receipts. CPU allocator peaks are explicitly null.
The151 reused DeepSolo responses have historical peak allocated/reserved CUDA
memory up to911353344/1358954496 bytes (RTX3090, isolated FP32 batch1). They keep
their original .05 inference metadata; the evaluator applies .4 after inference.
No new GPU repeatability test or OOM stress test was claimed. Resuming with
`bash scripts/run_clip_router_e2e_host.sh` computes the missing63 TRAIN responses
and writes the separate TRAIN report and root split-indexed `results.json`.

Final whitespace/diff checks pass. An initial diff check flagged a new trailing
blank line in the ambiguity ledger; it was removed. The concurrent separated-
word target qualification entry already present in that ledger was preserved.
