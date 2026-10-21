# Independent word-slot diagnostic on Total-Text

This is an **approximate reproduction**, exploratory Total-Text TEST diagnostic.
It is outside the paper's Algorithm 1. The paper uses DeepSolo as a fallback,
but does not use an independent detector to group PARSeq crop hypotheses.

## Purpose and frozen inputs

The frozen crop selector emits 16,678 TEST predictions, including 1,745 locally
correct words (2,204 annotated words). This experiment asks whether a separately
run word detector can supply approximately one spatial slot per real word while
retaining the correct PARSeq crop readings. It does not use DeepSolo's text in
the final prediction. The DeepSolo ResNet-50 Total-Text model-zoo candidate is
already installed in a qualified isolated host runtime. It is a useful locally
available detector control, **not** evidence that a detector with no Total-Text
training would transfer. CRAFT/DBNet would require an additional environment and
checkpoint, neither of which is already qualified here.

The inference threshold 0.05 is a predeclared recall-oriented assumption; 0.4
is the detector's existing config threshold, evaluated as a post-hoc score
subset of the 0.05 run. That subset is not asserted to be byte-identical to
a separate 0.4 inference because detector post-processing can depend on the
inference threshold.
The crop-to-slot bbox IoU controls 0.1, 0.25 and 0.5 are a predeclared sweep,
not fitted to TEST. DeepSolo polygon envelopes define slots. Each frozen PARSeq
crop can be eligible for multiple slots; maximum-weight assignment emits at most
one crop per slot and uses each original crop once. The pair weight is the
geometric mean of existing crop score and box IoU. The original crop bbox and
normalized PARSeq reading are used for the local metric.

## What the report measures

For each predeclared setting, `results.json` reports:

- `slots`: detected word slots, before crop assignment;
- `covered_correct`: how many original correct PARSeq outputs remain recoverable
  when crops with no matching slot are removed;
- `outputs` and `correct`: one-per-slot assignment performance;
- `detector_gt_box_matches`: one-to-one GT coverage by detector boxes alone.

All 300 inference shards and all selection shards are frozen before reading GT.
The metric uses Unicode casefold, removal of Unicode punctuation and symbols,
and original-image axis-aligned bbox IoU at least 0.5 with one-to-one matching.
This is not official Total-Text polygon Hmean. The experiment logs checkpoint,
image, config and code hashes, raw detector output, per-image assignments,
latency and peak allocated/reserved VRAM. Results are outside Git.

## Run

From the project root on the GPU host:

```bash
bash scripts/test_totaltext_detector_slots.sh
```

To validate all local paths and hashes without inference:

```bash
bash scripts/test_totaltext_detector_slots.sh --check
```

The completed analysis report is
`/tmp/totaltext-independent-detector-slots-v3/full/results.json`.
To reproduce its CPU-only analysis from the 300 frozen v2 DeepSolo shards:

```bash
bash scripts/finish_totaltext_detector_slots.sh
```

The analysis verifies each source shard against the original detector plan and
records its SHA256 in the new plan. It reads raw detector shards from v2 without
copying or changing them. Selection and evaluation shards are written to v3.
Any changed inference configuration, checkpoint, manifest, worker dependency,
or shard causes an explicit failure. The original GPU launcher remains available
for a fresh inference run with a separate output root.

The first GPU attempt stopped before saving a detector shard: the suite resolved
the qualified virtualenv `bin/python` symlink to `/usr/bin/python3.8`, so the
worker could not import the already installed `cv2`. The suite now preserves the
configured interpreter entry point and checks imports for `cv2`, `detectron2`
and `torch` before inference. The failed v1 plan remains untouched; the corrected
run writes to v2.

The v2 GPU run completed all 300 detector shards, then stopped after 180
selection shards on image `total_text:651`. That JPEG has EXIF orientation 6:
OpenCV rotates it when DeepSolo reads it, whereas the frozen PIL crop boxes and
annotations use the stored raster. The v3 analysis maps DeepSolo polygon points
back through the inverse EXIF transform before clamping to the PIL image. The
mapping follows PIL's pixel-index convention and is classified as `ASSUMPTION`;
the script records the orientation and mapping for every image. A finite polygon
with no positive in-image envelope is explicitly logged and excluded; malformed
geometry remains a fatal error. In the completed run, one image was reoriented,
all 12,469 detections yielded valid slots, and none was excluded.

Inverse coordinate mapping preserves the original model outputs. DeepSolo saw
the EXIF-oriented raster for image 651; the effect of instead inferring on its
stored raster has not been measured. The correction establishes a common
coordinate frame for assignment, without qualifying those two inference inputs
as equivalent.

The local path profile is
`configs/approximate/totaltext_independent_detector_slots_local_paths_v1.json`.
Set `TOTALTEXT_SLOTS_PATHS` to a different explicit profile if paths differ.
The scientific controls live separately in
`configs/approximate/totaltext_independent_detector_slots_v1.json`.

## Completed approximate TEST diagnostic

The v3 analysis contains all 300 selection shards and all 300 evaluation
shards. The denominator is 2,204 GT words. The original frozen selector emits
16,678 predictions with 1,745 correct; the most compact predeclared arm
(`detector score >=0.4`, crop/slot bbox IoU >=0.5) emits 1,886 predictions
with 1,499 correct, for local precision/recall/Hmean proxies
0.795/0.680/0.733. At detector score >=0.05 and crop/slot IoU >=0.1,
1,724 correct readings remain among 6,579 outputs. These metrics use exact
normalized reading and original-image bbox IoU >=0.5. They are **not** the
official Total-Text polygon Hmean or a strict paper reproduction.

The detector execution recorded peak allocated/reserved VRAM of
911,353,344/1,358,954,496 bytes across the 300 single-image subprocesses.
The qualified runtime reports an NVIDIA RTX 3090. The sandbox used for the CPU
analysis exposes no NVIDIA device.

Validation passed 64 focused geometry, cache-contract, assignment and metric
tests, a one-image smoke, and a complete repeated 300-image analysis. Independent
checks reconciled all per-image counts with the aggregate report and confirmed
that all 481 original v2 artifacts and all 180 completed v2 selection arms remain
unchanged. The result SHA256 is
`c976960bd10ffe74eadcda80a4535bbe2fcd638bfd86149f99505e62ef5d6f4e`;
the machine-readable audit is `/tmp/detector-slots-fix-audit.json`.

The earlier sandbox CPU inference failure at `ms_deform_attn_forward`
(`RuntimeError: Not implemented on the CPU`) remains a real historical failure.
The completed analysis reuses the user's genuine GPU outputs and performs no
replacement CPU model inference.
