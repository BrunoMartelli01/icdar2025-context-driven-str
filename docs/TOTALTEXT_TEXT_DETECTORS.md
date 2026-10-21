# CRAFT, DBNet and FAST regions with PARSeq

## Run the complete comparison

From the project directory on the GPU host:

```bash
bash scripts/test_totaltext_text_detectors.sh
```

This installs pinned packages in `.venvs/context-str-text-detectors-py311`,
downloads the three versioned detector checkpoints, verifies their upstream
digests, freezes their full SHA256, runs focused tests, performs a real
one-image smoke test for each detector and then runs all **300 Total-Text TEST
images / 2,204 care words**. The main environment stays read-only. Weights and
results stay outside Git under the configured `/tmp/totaltext-text-detectors-v1`.

Explicit CPU execution is available, without automatic device fallback:

```bash
bash scripts/test_totaltext_text_detectors.sh --device cpu
```

A single-model smoke test:

```bash
bash scripts/test_totaltext_text_detectors.sh --models craft --limit 1
```

Paths and thresholds are configuration fields, not hardcoded implementation
values. Override `--paths` and `--config` as needed; scientific configuration
must remain in `configs/approximate/`. Cached shards resume only with identical
configuration, code, input checksums, weights, package source and device.
Changing one of these requires a new `output_root` profile, not editing caches.

## Models and scope

These are **text detectors**, not word recognizers. All three use the same
genuine, already qualified official PARSeq decoder for recognition:

| Detector | Implementation / checkpoint | Size reference |
|---|---|---|
| CRAFT | EasyOCR 1.7.2 / `craft_mlt_25k.pth` | **20,770,466**, counted in the real CPU smoke run |
| DBNet ResNet-50 | docTR 0.11.0 / `db_resnet50-79bd7d70.pt` | **25,412,162**, counted in the real CPU smoke run |
| FAST-Tiny | docTR 0.11.0 / `fast_tiny-1acac421.pt` | **8,521,920** after inference reparameterization, counted locally |
| Project practical U-Net | `PracticalTotalTextHighResAGUNet` | **58,155,081**, independently counted locally |

The separate approximate AG-U-Net definition has48,454,733 parameters. It is
not the practical U-Net reference used in these Total-Text experiments.
Parameter count is not a FLOP or latency measurement. Exact detector parameter
counts before/after reparameterization, weight bytes, input shapes, timing and
CUDA peaks are recorded by this suite.

CRAFT models character regions and character affinity. Its upstream component
formation/dilation remains intact; the adapter uses the general-purpose model,
without the optional sentence link refiner. Native maximum character heatmap
scores are retained. For joint admission only, they are explicitly clamped to
[0,1], since CRAFT regresses Gaussian targets rather than calibrated class
probabilities. DBNet predicts word text maps and uses differentiable
binarization during training. FAST uses text kernels; this experiment tests
docTR's implementation and pretrained weights, including its modified
postprocessing, not an exact reproduction of the original FAST paper.

Both docTR models use only `detection_predictor`. No docTR/EasyOCR recognizer,
orientation network, page rectification, paragraph merge or language correction
is loaded. Model and backbone factories use `pretrained=False`, followed by
explicit **strict** checkpoint loading, so unexpected implicit downloads or
missing parameters are not accepted.

## Geometry and crop controls

Each detector is run once per image, in FP32 / batch1. CRAFT uses its versioned
preprocessing with canvas1280/magnification1.5 and upstream thresholds
text0.7/link0.4/low0.4. DBNet/FAST use their versioned RGB normalization and
aspect-preserving symmetric letterboxing to1024. Inputs and confidence
definitions differ: their numeric score thresholds are not interchangeable.

Each accepted region yields two zero-added-padding recognition views:

1. **Axis aligned:** its clipped exclusive integer envelope, with original RGB
   pixels untouched. This is the comparison closest to direct YOLO box crops.
2. **Rectified:** the minimum-area rectangle is mapped to a horizontal crop
   using bicubic perspective warping. Its long side becomes horizontal. This
   does not straighten curved text or solve180-degree orientation ambiguity.

Nonfinite/malformed predictions fail explicitly. Degenerate or wholly outside
regions are retained in detector diagnostics as rejected records, never passed
as empty crops to PARSeq. They are reported separately, not counted as produced
word outputs. Rectification can contain black pixels where a raw detector
rectangle extends beyond the image; this behavior is explicit.

We evaluate each view separately and a **best OCR** view: choose the one with
higher balanced PARSeq confidence, without reading GT. Ties retain the first
view (original pixels). This returns one prediction per detected region, not
both crops in a single word list.

For every view the report contains raw detector-score controls, explicit
nonempty filtering, geometric NMS and the same predeclared rule used with YOLO:
`joint_score=sqrt(detector_score * balanced_PARSeq_confidence) >=0.6`, optionally
with same-normalized-reading NMS atIoU0.5. No weights/thresholds are fitted to
Total-Text TEST. All controls are reported; selecting a TEST winner would be
exploratory and would require independent validation.

## Metric and artifacts

Word correctness uses Unicode casefold and removes Unicode punctuation/symbol
categories, preserving whitespace. A prediction must match both the normalized
word and an original detector envelope with IoU>=0.5, using maximum-cardinality
one-to-one matching. Rectification never changes the evaluation box.

Micro precision=`correct/outputs`, recall=`correct/2,204`, and
Hmean=`2*correct/(outputs+2,204)`. Raw arms keep empty readings as false
positives; explicitly named nonempty/joint arms remove them. Every care word
and image remains in the denominator. The343 ignored annotations are excluded
from GT; predictions overlapping them still count as false positives, matching
the earlier local YOLO protocol. This is a **local bbox spotting Hmean proxy**,
not the official Total-Text polygon evaluation.

Complete GPU reports:

```text
/tmp/totaltext-text-detectors-v1/evaluation/cuda-0/full/results.json
/tmp/totaltext-text-detectors-v1/evaluation/cuda-0/full/comparison.md
/tmp/totaltext-text-detectors-v1/evaluation/cuda-0/full/comparison.csv
```

Each detector subdirectory contains resolved configuration, source/checkpoint
hashes, per-image regions, rejected regions, original crop hashes, raw PARSeq
readings and character probabilities, selected prediction IDs, GT/matching
outcomes and stage VRAM peaks. Detector and recognition loading are sequential.
Failures record stage, traceback, precision, device, input shape and OOM status.
First-image loading is included in recorded time; it is not warmed latency.
Incomplete package metadata directories (e.g. `UNKNOWN.egg-info` without
Name/Version) are recorded under `package_metadata_anomalies`. They do not
enter the sorted package/version inventory; required runtime package versions
are still checked explicitly before inference.
The pinned docTR0.11 rotated-box output has four normalized vertices followed
by `(0, score)`. Only the four vertices are scaled to image pixels; the final
row supplies confidence and does not belong to the polygon.

## Qualification and limits

The initial local installation attempt failed because PyPI DNS was unavailable.
After the host installed the packages and downloaded the weights, those assets
became visible locally. Real **CPU FP32 one-image smoke tests now complete for
all three pretrained detectors and genuine PARSeq**; the same command also
completes a cached rerun. CUDA remains unavailable to the restricted process;
the full TEST comparison and GPU memory qualification still require the host.
Smoke results are not a dataset accuracy claim.

### Complete CPU TEST comparison after the runtime fixes

The complete experiment now finished for each detector: **300 images /2,204
care words**, CPU FP32, four Torch threads, detector batch1, PARSeq batch32,
strictly loaded qualified checkpoints. This is an **approximate reproduction,
exploratory checkpoint comparison**, not a GPU qualification or official
polygon spotting score. No settings were fitted to these TEST results.

The predeclared common selector chooses the higher-confidence crop, admits
joint score>=0.6 and applies same-reading NMS atIoU0.5:

| Detector | Correct words | Final outputs | Precision | Recall | Local Hmean |
|---|---:|---:|---:|---:|---:|
| CRAFT |1,612|2,443|65.98%|73.14%|69.38%|
| DBNet ResNet-50 |1,134|1,603|70.74%|51.45%|59.57%|
| FAST-Tiny |1,347|3,874|34.77%|61.12%|44.32%|

For context, the earlier YOLO640 single-crop experiment with the same joint.6 /
same-reading-NMS.5 rule produced1,597 correct /2,205 outputs, precision72.43%,
recall72.46%, Hmean72.44%. It used a different detector/crop pool; its standalone
report remains `/tmp/yolo-failure-improvements/v1/cpu/full/results.json`, arm
`base640/joint_text_nms_0.6`. This is an exploratory local comparison.

| Detector | Generated regions | Rejected regions | PARSeq crops (two views) | Detector seconds | PARSeq seconds |
|---|---:|---:|---:|---:|---:|
| CRAFT |2,538|0|5,076|613.5|118.5|
| DBNet ResNet-50 |3,906|0|7,812|399.5|176.8|
| FAST-Tiny |4,950|1|9,898|183.3|225.4|

Times include loading and preprocessing; input sizes differ and no GPU/FLOP
comparison is inferred. All60 controls remain in the full report. An independent
SciPy bipartite matcher verified every control over all300 images, all2,204 GT
denominators and every11,393 original RGB crop checksum; all checks passed.

Completed results and audit:

```text
/tmp/totaltext-text-detectors-v1/metadata-fix-qualification-v3/cpu/full/results.json
/tmp/totaltext-text-detectors-v1/metadata-fix-qualification-v3/cpu/full/comparison.md
/tmp/totaltext-text-detectors-v1/metadata-fix-qualification-v3/cpu/full/independent_audit.json
```

The separate path profile at
`/tmp/totaltext-text-detectors-v1/qualification/metadata_fix_paths_v3.json`
preserves the initial failed smoke tests and their provenance. The regular short
host command continues to use its default output root and CUDA device.

Locally qualified: geometry conversion, RGB-preserving crops (including3x3x3),
rectification, malformed/outside/degenerate handling, empty outputs, GT-blind
one-per-region crop selection, scientific mode boundaries and artifact digest
checks.27 focused tests pass, including the existing YOLO crop/admission tests
and the installed DBNet/FAST postprocessor score-row contracts. One dependency
deprecation warning from `defusedxml.cElementTree` remains; no test failed.
A genuine PARSeq CPU/FP32 smoke test processed four real image crops through
the new view builder, using previously qualified YOLO regions. Its report is
`/tmp/totaltext-text-detectors-v1/qualification/parseq_crop_smoke.json`; it tests
the crop/recognizer boundary independently of the new detector smoke tests.
The complete detector smoke and cached-rerun qualification is recorded at
`/tmp/totaltext-text-detectors-v1/metadata-fix-qualification-v3/cpu/smoke-1/results.json`.
These use a separate output root and preserve earlier failed diagnostic runs.
Annotation audit verified all300 images: all2,204 care rows are identical
to previous frozen evaluations;343 ignored records and20 clipped polygons are
reported explicitly. CPU execution and sequential loading are documented
hardware deviations, not strict reproduction.

Training-set overlap for the docTR checkpoints is **UNRESOLVED**. These are
exploratory checkpoint comparisons, not evidence of unseen-dataset
generalization. The existing PARSeq artifact also remains unqualified as the
paper's exact PARSeq-S. See ambiguity entryETD-001.

## Primary sources

- [Official CRAFT implementation and pretrained model descriptions](https://github.com/clovaai/CRAFT-pytorch)
- [EasyOCR1.7.2 checkpoint URLs and MD5 checksums](https://github.com/JaidedAI/EasyOCR/blob/v1.7.2/easyocr/config.py)
- [docTR0.11 model benchmark and detection interface](https://mindee.github.io/doctr/v0.11.0/using_doctr/using_models.html)
- [Versioned FAST architecture/checkpoint metadata](https://github.com/mindee/doctr/blob/v0.11.0/doctr/models/detection/fast/pytorch.py)
- [Versioned docTR strict weight loading and SHA256 protocol](https://github.com/mindee/doctr/blob/v0.11.0/doctr/models/utils/pytorch.py)
