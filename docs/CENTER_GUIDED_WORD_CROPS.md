# Joint base-mask and center-offset crop experiment

This opt-in extension is **approximate reproduction**, an exploratory diagnostic
on the reused Total-Text TRAIN holdout. It is not a paper component or official
TEST evaluation. Evidence and classifications are in WCO-003 of
`docs/PAPER_AMBIGUITIES.md`. The existing scientific segmentation/localization
paths and both selected network checkpoints are unchanged.

## Inference

`localize/center_guided_word_crops.py` accepts the base U-Net probability map,
the auxiliary U-Net's **tanh-normalized x/y offsets**, a boolean RGB letterbox
canvas, and an explicit approximate refinement configuration. No owner map,
annotation, validity mask, OCR or desired word count enters inference. For
fresh network outputs, both networks must receive the same qualified RGB
preprocessing, input 512 and native 256 geometry; raw auxiliary logits require
`tanh()` before this interface. Config values are loaded explicitly, never
inherited by STRICT mode.

Only high-confidence base pixels (`p>=0.65`) vote for endpoints. Global bilinear
voting followed by Gaussian smoothing (sigma 3 native pixels), peak density>=0.5,
Chebyshev NMS distance 8 and at least 8 agreeing core pixels produces center
hypotheses. Endpoint agreement uses radius 12 native pixels. These are
predeclared experimental assumptions, not learned/calibrated word counts.

Dense agreement seeds and deterministic eight-connected geodesic assignment
partition fused foreground. Disconnected fragments voting for the same center
share one identity and RGB crop. Unseeded regions fall back to components.
All high-confidence pixels survive; “cutting” means partitioning identities,
not deleting possible characters. A marker is projected to foreground even
when the center lies in a gap or outside a concave word.

The optional growth branch admits `p>=0.35` pixels only if their endpoint agrees
with a retained center, they are reachable from that identity, lie on the RGB
canvas, and are at most 3 Euclidean native pixels from the original core.
Three synchronous eight-neighbor steps bound growth. Identities cannot claim
each other's pixels. Background hallucination or mistaken grouping remains
possible when predictions are wrong.

One RGB crop is extracted from each identity's original-image bounding box,
including separated fragments. Actual foreground area determines sorting.
This matched diagnostic uses pad 0, no area cutoff and no maximum count for
all methods; those are explicit deviations from paper section 3.3's padding,
component filtering and top 10 localization, not new STRICT defaults.

## Frozen comparison and scoring

Four branches use identical images, maps and crop rules: `baseline` external
contours; `local_centers` the previous component-local voting; `global_centers`
the new common identities; and `global_centers_expand` bounded growth.

Sources are base epoch 56 SHA 256
`db9db69f77f1e62b03b82d073e8a37c603ff6ff8feceafc3623644980625e964` and auxiliary
epoch 82 SHA 256
`4dcf6e14ce66a72e0870e73bdfbc31f2503c6b0f313b49b31fb2d13e9132d8f6`.
The runner verifies both checkpoint files, reports, per-sample export hashes,
RGB hashes, preprocessing, canvas/owner/validity alignment and frozen125 sample
identity/order before scoring. No threshold search or checkpoint reselection is
performed. Per-sample inference recipes and RGB crops are frozen before GT is
read for evaluation.

All 125 held-out TRAIN images and 983 care words remain in the denominator.
Only predicted identities fully outside native validity are ignored after
inference; partially overlapping boxes remain whole. Raw inference counts are
also saved. Count MAE is mean absolute care-proposal minus care-GT count per
image, including empty predictions. Geometry precision/recall/F1 use maximum
cardinality one-to-one original-image exclusive-box IoU>=0.5. Native diagnostics
reuse `instance_geometry`: bbox matching at0.25/0.5/0.75 and any shared-pixel
merge/split counts. These are exploratory box metrics, not official polygon
evaluation. Empty metric denominators yield 0.

Real pinned PARSeq recognition uses AR=True/refine1, FP32, batch8 and exact
casefold text plus IoU>=0.5 matching, with all 983 words counted. Symbol-normalized
results are secondary. Text-constrained maximum matching is not fixed-pair OCR
accuracy. DeepSolo/context routing is outside this crop diagnostic.

Both network predictions were generated during their completed RTX3090 FP32
validation runs. Postprocessing reuses those unchanged exports on CPU: no
segmentation inference is repeated, and no simultaneous two-network GPU fit is
claimed. Optional PARSeq runs alone on the explicitly selected device. Runtime
metadata and allocated/reserved GPU peaks are recorded when applicable;
synthetic error tests cannot qualify physical RTX3070 behavior. Legacy OCR reuse
requires identical pinned source/checkpoint and exact RGB hash, plus replay of
the first 16 reused crops on the current device; errors propagate.

## Run

Use the existing main environment; no installation or downloads are required:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
.venvs/context-str-main-py311/bin/python -m scripts.evaluate_center_guided_word_crops \
  --mode approximate \
  --base-run /extra1/bmartelli/context-str-runs/totaltext-word-multitask-v1/text \
  --offset-run artifacts/training/totaltext-word-center-offsets-v1/train \
  --target-root artifacts/totaltext-curved-word-boxes-v4 \
  --output-root artifacts/center-guided-word-crops-v1 \
  --action geometry --device cpu --cpu-threads 2
```

For real OCR use `--action all` and supply `--parseq-source`,
`--parseq-source-archive`, `--parseq-checkpoint` with the qualified local artifacts.
Optional repeated `--reuse-ocr` paths need qualified provider `protocol.json`.
Paths remain runtime arguments, not scientific defaults. Select a new output
directory after changing the protocol or source. Matching completed outputs are
verified before reuse. Failures stop execution and are recorded.

Results include `geometry_protocol.json`, `geometry_results.json`, an offline
`index.html`, and per-image `overview.png`, `centers.json`, `fusion.npz`, frozen
crop recipes and actual RGB crops. OCR adds `recognition_protocol.json`,
`recognition_results.json` and hashed per-crop records. Independent TEST behavior,
calibrated counts, rotation/curve-aware crop quality and official recognition
results remain unresolved.

## Fresh inference on any image

`scripts/predict_center_guided_word_crops.py` loads both selected checkpoints
strictly, checks architecture/preprocessing/source provenance, and applies them
to the same RGB tensor. Models execute sequentially in FP32/batch1 with per-model
memory reporting. This is an explicit `HARDWARE_DEVIATION` relative to the saved
batch8 RTX3090 validation outputs; small CPU/GPU numeric differences are possible.
No annotation files or desired count are used. The output contains predicted
center locations, raw word hypotheses, both mask variants and actual RGB crops.
Counts include every hypothesis, including possible false positives.

```bash
.venvs/context-str-main-py311/bin/python -m scripts.predict_center_guided_word_crops \
  --mode approximate \
  --base-run /extra1/bmartelli/context-str-runs/totaltext-word-multitask-v1/text \
  --offset-run artifacts/training/totaltext-word-center-offsets-v1/train \
  --image /path/to/image.jpg \
  --output-root artifacts/my-center-crops \
  --device cpu --cpu-threads 2
```

Choose a new output directory each time. `--device cuda:0` explicitly requests
the GPU without an automatic CPU/precision fallback; physical GPU qualification
is still required. The first real fresh-image CPU smoke uses Total-Text image 14
and compares the map geometry/counts with archived validation predictions.

## Completed v1 diagnostic

All 125 held-out TRAIN images and 983 care words were scored with no failures.
Original-box geometry F1 is 59.069% for the base, 35.529% for previous local
centers, 46.101% for global centers, and 46.840% with bounded growth. Count MAE
is respectively 2.248, 8.616, 3.560, 3.544 words/image. Real PARSeq exact-casefold
correct words with IoU>=0.5 are 425, 339, 334, 335 out of 983; corresponding text+crop
F1 is 45.479%, 22.898%, 32.761%, 32.827%. These results **do not support replacing
the baseline with this trained auxiliary checkpoint and frozen fusion settings**.
Global growth improves the number of geometric matches on 26 images, worsens 57
and ties 42. Image 490 recovers 12/12 geometric matches versus 6/12 for the base;
image 265 drops from 10/12 to 3/12. Neither selected example overrides the full
population result.

PARSeq ran on CPU/FP32: 3618 unique RGB crops, 3482 new reads, 136 hash-qualified
reuses, and 16/16 replay checks passed. Both networks' full-study outputs remain
the archived, paired RTX3090 FP32 predictions, with CPU postprocessing.
The fresh-image 14 CPU/batch1 smoke took 30.565 seconds for the two models and
crop export; all four hypothesis counts match the archived comparison. Maximum
probability difference is 0.010950 and normalized-offset difference 0.0005594;
the local and expanded masks differ at 1 pixel each. Baseline/global preserved
labels match exactly. Device/batch differences are measured, not assumed absent.
This does not qualify simultaneous GPU residency or independent TEST results.

75 focused/related tests pass, plus complete paired geometry/OCR and fresh-model
CPU smokes. The initial test failure used `native_bbox` instead of the existing
`bbox_native` key and was corrected before the complete rerun. Diff review and
whitespace checks pass; unrelated existing changes were preserved.
The common global-branch audit describes both preserved and expanded masks;
`added_pixels` refers only to the latter, not to preserved foreground.

Machine-readable results, hashes, per-image outputs and report are at
`artifacts/center-guided-word-crops-v1/{geometry_results,recognition_results,summary}.json`
and `REPORT.md`; the complete visual comparison is `index.html`. Fresh-image
outputs and the measured cross-device comparison are under
`artifacts/center-guided-word-crops-live14-v1/`. No thresholds were adjusted after
scoring this validation set, and neither checkpoint was modified.
