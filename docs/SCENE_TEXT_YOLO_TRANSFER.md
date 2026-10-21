# Frozen YOLO/PARSeq transfer to other TEST datasets

This is an **approximate reproduction / external detector experiment**, not
the paper's segmentation/component/PARSeq-S/DeepSolo pipeline. Paper sections
3.3 and 3.5 and the existing Total-Text YOLO code were inspected before this
extension. Scientific choices are frozen in
`configs/approximate/scene_text_yolo_parseq_transfer_v1.json` (YTR-001).

From the repository, run:

```bash
bash scripts/test_scene_text_yolo.sh
```

The default device is **cuda:0**. The script uses the existing isolated
`.venvs/context-str-yolo-py311` environment and already downloaded weights;
it does not install packages or download a substitute. All requested split
identities and both checkpoints are verified before inference. First a real
one-image smoke test runs for each dataset, then the complete TEST splits:

| Dataset | Images | GT unit |
|---|---:|---|
| ICDAR13 | 233 | word (Task1; 1,095 care words) |
| ICDAR2015 | 500 | word quadrilateral (2,077 care words) |
| CTW1500 | 500 | text line (2,687 care lines) |

Override the device or datasets explicitly:

```bash
bash scripts/test_scene_text_yolo.sh --device cpu --datasets icdar13 icdar2015
```

Use `--paths` with a new approximate path profile for other storage locations;
`--output-root` chooses a fresh external result directory. The supplied local
profile resolves this machine's actual images and qualified PARSeq artifacts.
ICDAR2015's recognition manifest is generated once outside Git, then verified
against its prepared TEST raster manifest on each run. CTW uses its qualified
E2E recognition manifest and prepared, explicitly normalized rasters.

## Method and metrics

YOLOv8s TextOCR checkpoint -> original-image exclusive axis rectangle without
additional padding -> genuine qualified PARSeq -> frozen selection. YOLO receives
BGR NumPy input; PARSeq receives RGB. All candidate readings are retained.
Primary selection is `joint_text_nms_0.6`: geometric mean of detector score and
balanced character confidence, threshold 0.6, followed by same-normalized-text
NMS at bbox IoU >0.5. Raw detector floors 0.05/0.25/0.4, nonempty output, geometric
NMS, and joint-only selection are retained as fixed controls. No new TEST labels
are used in crop generation, recognition, filtering or parameter fitting.

Scoring uses Unicode casefold, removes punctuation/symbols and retains spaces.
A prediction is correct only with the exact normalized GT string and bbox IoU
>=0.5, under maximum one-to-one matching. Counts are micro-aggregated over all
images. P=C/outputs, R=C/care-GT, Hmean=2C/(outputs+care-GT). Empty predictions
are false positives; normalized-empty care GT stays in the denominator. Ignored
GT is excluded, but predictions over ignored regions remain false positives.
This is the existing **local bbox proxy**, not official polygon/ignore-aware
benchmark Hmean. No lexicon is applied.

**CTW1500 is a separate exact-line diagnostic.** This word detector is evaluated
against full line strings/boxes; no invented word GT boxes or GT-derived grouping
is introduced. Its result cannot be presented as word accuracy or compared
directly to ICDAR13/15 word Hmean. Improving CTW line outputs requires a separate
line grouping/recognition experiment.

## Output and restart

Default GPU aggregate report:
`/tmp/scene-text-yolo-transfer-v1/cuda-0/full/results.json`.
The adjacent `comparison.csv` contains every fixed control. Each dataset has
its own resolved configuration, immutable detector/OCR/evaluation shards and
`last_execution.json`. GPU allocation/reservation peaks are recorded separately
for sequential YOLO and PARSeq residency; precision is FP32, detector batch1,
PARSeq batch32. CPU peaks are null. Failure records preserve errors and OOMs.

Rerunning the same command reuses valid shards. Changed config, code, images,
annotations or model identities cause an explicit mismatch; choose a fresh
`--output-root` for a changed experiment. Smoke and full outputs are isolated.
Local CPU smoke qualification does not establish full dataset or GPU performance.

## Local qualification

18 focused tests pass. Real CPU FP32 smoke tests completed for all three
datasets (45 direct crops), followed by a cached rerun with zero new inference.
An independent SciPy maximum bipartite matcher verified all 21 smoke controls
and every original-RGB crop checksum. ICDAR2015 image1 contains only ignored
GT; an additional two-image test includes image2's ten care words and exercises
nonempty word matching (65 crops total). These are smoke/partial results, not
full TEST accuracy. The initial implementation had a CSV bracket syntax error;
it was fixed before successful qualification. Ultralytics emits its existing
`torch.load(weights_only=False)` FutureWarning for the identity-checked checkpoint.

Qualification artifacts are under
`/tmp/scene-text-yolo-transfer-qualification-v1/cpu/`; they are separate from
the default future GPU/full result directory.

## Complete GPU qualification

The user ran all1,233 TEST images on NVIDIA RTX3090, FP32, detector batch1 /
PARSeq batch32. No failed images or failure records occurred. Frozen joint0.60
results are ICDAR13:862 correct/1,038 outputs/Hmean80.83%; ICDAR2015:1,401/2,336/
63.49%; CTW1500 full lines:1,106/5,245/27.89%. A separate SciPy audit verified
all25,992 original RGB crop hashes,8,631 per-image controls and full denominators.
The largest recorded GPU peaks are288.52MiB allocated and390MiB reserved;
these measurements qualify this RTX3090 run, not RTX3070 hardware or strict
paper reproduction. Reports are at the default GPU/full path above.

See `docs/SCENE_TEXT_YOLO_THRESHOLDS.md` for the separate cached confidence
analysis, quantified CTW line/word incompatibility and TEST-selected controls.
