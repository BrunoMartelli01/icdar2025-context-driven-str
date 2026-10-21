# CLIP whole-image router on Text+Kernel epoch61

Result class: **approximate reproduction**, exploratory TRAIN holdout routing.
This user-requested CLIP extension does not modify Algorithm2 or STRICT defaults.
Paper section3.5/Algorithm2 motivates calling DeepSolo on the original image
when recognition is uncertain; the paper uses MPNet/fuzzy scores, not CLIP.

The frozen source is `artifacts/word-multitask-crop-refinement-v1/`:
`text_kernel/seeded`, checkpoint epoch61 SHA256
`e65691dda944478e9d8b4f022d45d0d3b21edc7c995274554b4c5ca31336766f`.
Its native geometry F1 is62.13%; original-coordinate bbox F1 is61.33%.
All125 images contain1109 predictions. The unchanged evaluation ignores80
wholly invalid components, leaving1029 outputs and983 care GT words.
The baseline has480 exact-casefold, bboxIoU>=.50 one-to-one matches.

Every prediction, including ignored ones, enters the routing score. There is
no word correction, native confidence prefilter, crop rejection, top-k or GT
ignore mask in inference. A whole image passes only if its **minimum word
score** reaches the threshold. Empty output or whitespace-only reading forces
fallback. Accepted images retain all original readings and boxes; rejected
images are exported for DeepSolo on the complete original image.

Signals include PARSeq character mean/minimum/balanced confidence, crop CLIP
written-word prompts, prompt ensembles, glyph-image similarity, text/background
contrasts and their existing rank fusions. Full-image CLIP/word comparisons
provide scene-only controls. Glyph references use the pinned three fonts and
two polarities from the earlier verifier; crop square canvases preserve all
pixels. Cosine scores and empirical ranks are **not probabilities**.

Before scoring, SHA256(seed:sample_id) fixes80 calibration and45 validation
images. Empirical mid-CDF references use only calibration words, without
labels. Features and image aggregates are sealed before threshold calibration.
For each signal family, search the distinct calibration-score breakpoints,
requiring>=99% correct/detected words and **zero** unsafe accepted calibration
images. Maximize accepted images, then accepted words, then lower threshold;
family selection uses the same calibration counts with lexicographic final
ties. Validation uses these fixed thresholds and is never used to retune them.
An impossible safe calibration results in explicit all-image abstention with
undefined accuracy. Additional all125 scans are labeled **post-hoc** and are
not deployment threshold selection. The125 TRAIN images have already been used
to select the checkpoint and other experiments; this internal split is not
an independently acquired evaluation dataset.

Requested accuracy on detected words is `TP/(TP+FP)`. Also report TN=0
accuracy including missing words, `TP/(TP+FP+FN)`, whereFN concerns only GT
in accepted images. Casefold exact retains punctuation. Matching maximizes
one-to-one matches using original exclusive xyxy boxes and IoU>=.50.
Ignored outputs are excluded only by the unchanged source evaluation rule.
Report all-correct emitted-text images separately from completely recognized
scenes: an accepted image may still miss words. Fallback image counts, already
correct images/words routed to DeepSolo and per-image counts remain auditable.
When all accepted images are safe, `.05**(1/n)` is the exact one-sided95%
binomial lower bound under independent identically distributed images. The
small repeatedly used development sample cannot establish99% future safety.

Run an artifact-only preflight, then inference:

```bash
.venvs/context-str-main-py311/bin/python -m scripts.run_clip_image_router \
  --mode approximate --model vit_l14 --check
.venvs/context-str-main-py311/bin/python -m scripts.run_clip_image_router \
  --mode approximate --model vit_l14 --device cuda:0
```

Use `--device cpu` explicitly for CPU; there is no automatic device fallback.
`--limit 1` is a smoke only. Models and packages
are never downloaded or installed. B32/B16/L14 use the existing pinned snapshots.
Resolved config, model/code/input hashes, features, calibration references,
threshold curves, per-image metrics and `routing_requests.json` go under the
configured external output root. Model-key mismatches, invalid inputs and OOM
fail explicitly. FP32, batch16, four CPU threads, deterministic execution and
TF32-disabled settings are `HARDWARE_DEVIATION` choices. GPU allocated/reserved
peaks are recorded on CUDA, null on CPU. DeepSolo inference is a separate
isolated-runtime phase; a routing manifest alone is not an end-to-end result.

Scientific choices: [CLIP-IMAGE-ROUTER-V1](PAPER_AMBIGUITIES.md#clip-image-router-v1).

The three-model GPU launcher is `bash scripts/run_clip_image_router_host.sh`;
each model runs sequentially and records its own immutable output directory.
An actual CPU comparison was launched with three independent processes;
overlapping CPU elapsed times do not measure isolated model performance.

To execute the saved whole-image fallback on the GPU host, pass the router
output directory (the parent of the printed `results.json`):

```bash
.venvs/context-str-main-py311/bin/python -m scripts.run_clip_image_router_fallback \
  --mode approximate --router-run /path/to/router/model/run \
  --output-root /tmp/clip-image-router-deepsolo-v1 --check
.venvs/context-str-main-py311/bin/python -m scripts.run_clip_image_router_fallback \
  --mode approximate --router-run /path/to/router/model/run \
  --output-root /tmp/clip-image-router-deepsolo-v1
```

The executor uses the pinned qualified Python3.8 host runtime/candidate,
score.4 without a lexicon (`ASSUMPTION`), preserves all raw detections and
worker memory/error records, and propagates every failure. `--family NAME`
can select another saved **calibration-derived** family without using
validation labels. Reading-order joined strings are interface metadata;
they do not replace the raw word detections. Combined end-to-end accuracy
stays null until a matching polygon-frame/ignore evaluator is qualified.
In particular, the earlier TEST detector-slot result is a different dataset,
different source and different inference threshold, and cannot supply this
TRAIN routing fallback result.

## Fixed word/background rule

The complete B32 battery suggests `native_word_advantage`: for every word,
calculate CLIP crop similarity to the three written-word prompt ensemble,
minus similarity to the ensemble of three backgrounds without readable text.
Convert this contrast and PARSeq balanced confidence to calibration mid-CDF
ranks, take their geometric mean, then take the minimum over **all** words
in the image. This family was selected after inspecting B32 validation;
per-model thresholds still use only the80 calibration images. The resulting
method comparison is exploratory, with explicit post-hoc family selection.

`scripts/run_clip_image_router_word_advantage.py` computes that exact signal
without running unused glyph or scene features. `--reuse-score-run` permits
explicit reuse of completed or partial battery shards only after model,
device, scientific config, all source-code hashes, payload digest and exact
observable OCR/crop rows match. Score provenance distinguishes reuse from
new inference. The new config is
`configs/approximate/clip_image_router_word_advantage_v1.json`.

```bash
.venvs/context-str-main-py311/bin/python -m scripts.run_clip_image_router_word_advantage \
  --mode approximate --model vit_l14 --device cuda:0
```

At B32 threshold `0.6214129671585633`, calibration accepts15/15 words in4
images, and validation accepts10/10 words in4 images. Together, **25/25
correct detected words,8 accepted images and117 fallback images**. All8
accepted scenes also have no missing care GT words under the local metric.
Five of the13 initially correct images are unnecessarily routed to DeepSolo.
The accuracy includingFN/TN=0 is also100% for this accepted subset. These
small counts do not establish99% future safety: the exact zero-error
one-sided95% image precision lower bound with8 images is only68.77%.

For comparison, the automatic calibration coverage winner in the complete
B32 battery, `native_glyph_textness` threshold `0.50096375283738`, has22/22
calibration words but16/22 validation words (72.73%). It fails the requested
validation safety target. Its higher coverage must not be presented as a
safe operating point. Calibration-only native confidence accepts just1
image/3 words at `0.9996414299140931`; the all125 post-hoc native scan can
accept2 images/5 words at `0.999541910486888` with no errors.

Full B32 report:
`/tmp/clip-image-router-v1/vit_b32/ef241dc192bd/results.json`, SHA256
`ac4e958e897403fb6cbcce28dc27fd69d39624fc43679642bcf76d4ae208348b`.
Fixed-rule B32 report:
`/tmp/clip-image-router-word-advantage-v1/vit_b32/66c0dff1880c/results.json`.
No DeepSolo fallback was executed; qualified runtime/source/checkpoint and
the117-image fixed-rule routing manifest passed the actual artifact preflight.

Fixed-rule B16 also completes125 images/1109 predictions: calibration12/12
correct words in3 images, validation10/10 in4 images, threshold
`0.6192174372838609`; together22/22 words,7 accepted/118 fallback images and6
initially correct images unnecessarily routed. No missing care words in the
accepted scenes. The complete B16 battery and efficient fixed-rule threshold,
curves, validation and aggregate decisions match exactly; common separately
inferred contrasts differ by at most float32 roundoff (below1e-7).
Report: `/tmp/clip-image-router-word-advantage-v1/vit_b16/34952fc322b7/results.json`.

Focused routing/CLIP/geometry/runtime tests pass115 cases. Independent audits
recompute counts with manual bboxIoU and SciPy maximum binary assignment,
check125 images/1109 OCR record identities, image/annotation/code hashes,
CDF references, every image score, threshold optimality on calibration and
all exported routing actions. The first standalone audit failed on an empty
prediction image because the auditor called `min([])`; the actual routers
already correctly force fallback. The auditor was corrected, the failure
receipt preserved at `/tmp/clip-image-router-audit-first-failures.json`, and
both completed model audits pass. Experiment outputs were unchanged.

## Completed three-model fixed-rule comparison

All3 fixed-rule runs now complete125 images and1109 original predictions.
L14 uses threshold `0.6801702981222083`:3/3 calibration words in1 image,
2/2 validation words in1 image,5/5 overall words in2 accepted images,
123 fallback images and11 initially correct images unnecessarily routed.
Both requested correct/detected accuracy and the accepted-scene TN0 metric
are100%; the accepted scenes also have no missing care GT words.

| Model | Calibration words | Validation words | All accepted words | Accepted images | DeepSolo requests |
| --- | ---: | ---: | ---: | ---: | ---: |
| B32 | 15/15 | 10/10 | 25/25 | 8/125 | 117 |
| B16 | 12/12 | 10/10 | 22/22 | 7/125 | 118 |
| L14 | 3/3 | 2/2 | 5/5 | 2/125 | 123 |

B32 is the most useful candidate for this fixed rule and sample, though the
method was chosen after inspecting B32 validation. L14's stronger performance
in the earlier word-selection experiment does not transfer to greater safe
image coverage here. An independent dataset is still needed to validate the
chosen model/rule/threshold. These are development results, not a guarantee.

Machine-readable comparison:
`/tmp/clip-image-router-word-advantage-v1/comparison.json`, SHA256
`2236c469e7faa7b9864f5e23b3aa3c7860b489e764dccd2832517dae8e21973e`.
L14 report:
`/tmp/clip-image-router-word-advantage-v1/vit_l14/1cebaa1ed822/results.json`,
SHA256 `7810dc4a185cb0f66fb51d623728338d049d95ce81f2de063899cd03aed63833`.
All3 independent per-run audits pass. B32 reused125 full-battery shards,
B16 reused88, L14 reused28; remaining fixed-rule scores are real CPU model
inference. The optional all-signal L14 battery remains a separate in-progress
experiment and is not included in this completed comparison.

Run the fixed-rule GPU comparison without downloads:

```bash
bash scripts/run_clip_image_router_word_advantage_host.sh
```

Execute the117 B32 DeepSolo requests on the GPU host:

```bash
.venvs/context-str-main-py311/bin/python -m scripts.run_clip_image_router_fallback \
  --mode approximate \
  --router-run /tmp/clip-image-router-word-advantage-v1/vit_b32/66c0dff1880c \
  --output-root /tmp/clip-image-router-deepsolo-v1
```

Actual fallback inference and post-fallback accuracy have **not** been
measured in this session. The command invokes the qualified isolated runtime
only for routed images, preserves all detections, and resumes identical
successful shards; its raw outputs still require a matched frame/ignore
evaluator before a combined accuracy claim.
