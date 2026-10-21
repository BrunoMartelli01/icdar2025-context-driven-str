# Word-guided contours and actual crop recognition

This is an **approximate reproduction / experimental crop study**, not a STRICT
paper result. The original `audit_repository_contours` and the original
multi-task models, watershed, training checkpoints and results are unchanged.
Scientific choices are recorded as WM-002 in `docs/PAPER_AMBIGUITIES.md`.

## Inspection and scope

The local paper `papers/2503.15639v2.pdf`, section 3.3 / Algorithm 1, specifies
component localization, component-area filtering, padding, component-area sorting
and a top10 cap. It does not specify word kernels, center supervision or merging.
The frozen repository localizer instead uses `RETR_EXTERNAL` / `CHAIN_APPROX_SIMPLE`
and bounding-box-area filtering/ranking. That distinction is already documented;
neither path is replaced or silently redefined here.

The existing multi-task evaluator already compares CC, kernel BFS and watershed.
It exports hash-verified native probability/instance maps and original RGB boxes
on the same 125 held-out TRAIN images / 983 care words. It has not measured PARSeq
on those new model checkpoints' contour crops. The original kernel experiment's
PARSeq results concern a different trained checkpoint and cannot supply those
measurements. Existing `geodesic_partition`, `word_instances`, contour auditing,
RGB crop hashes, strict PARSeq adapter, text normalization and matching are reused.

The new component is `localize/word_guided_contours.py`. It exposes three
configuration-selected localization methods:

- `findcontours`: the original external contours of the text map at its frozen
  threshold. Actual component membership is retained instead of filling contour
  polygons and inventing text inside holes. Nested islands omitted by the
  original `RETR_EXTERNAL` algorithm remain omitted.
- `seeded`: the existing kernel/center watershed, reusing the saved labels when
  available. This is the already completed head-guided separation control.
- `refined`: the same seeded separation plus evidence-based recomposition.
  Shared kernels spanning a text>=.50 bridge may join distinct original text
  components. A connected center>=.50 region with one retained peak may join
  kernel-supported false partitions and the partition containing its center
  rescue marker. Endpoints must be within Chebyshev foreground distance3.

Two retained center peaks veto a union, including transitive unions. Kernel
unions cannot reconnect partitions of the same original text component. Without
common auxiliary evidence no union occurs. The refiner only changes integer
instance IDs: every original high-threshold foreground pixel survives, and no
bridge/background pixel is added. A joined ID may have disconnected support.
Contours are extracted per ID, with one bbox for all its fragments; taking a
binary union and calling `findContours` again would undo this separation.

The defaults .50 and distance3 are predeclared **ASSUMPTION** choices, not paper
values or an optimized sweep. False auxiliary bridges/centers can still join
different words, and missing centers weaken the veto. The new variant must earn
its recommendation from measured results; it is not assumed to improve them.

## Frozen comparison and metrics

`configs/approximate/word_contour_refinement_v1.json` selects the three methods.
The evaluator uses all three previously selected checkpoints (epochs56/61/66)
and their saved maps, with **no segmentation inference or retraining**. Text has
only the `findcontours` reference because auxiliary methods require a kernel.
The two auxiliary models each have all three methods: seven crop pipelines.

All inherit the existing threshold .65, pad0, no area filtering and no top10 cap.
Those settings isolate the user-requested crop comparison and are not the
paper's strict Algorithm1 protocol. The generic API supports explicit filtering
by true component pixels, sorting by component pixels, padding and a crop cap.
Boxes are exclusive xyxy, inverse-letterboxed to original RGB. Recognition gets
unmodified original RGB rectangles; this change does not add rectification,
mask painting or a new RGB normalization.

Crop recipes are frozen before care annotations are read. Ignore masks enter
post-hoc scoring only: even wholly don't-care crops are generated and recognized.
Only wholly invalid instances are excluded from the metric proposal denominator;
partially invalid predictions keep their complete boxes. Every image and all983
care words remain in the denominator. GT is never an input to localization.

Results report native bbox matching and the existing any-pixel merge/split
diagnostics, plus original-coordinate maximum-cardinality one-to-one bbox IoU>=.50.
Actual PARSeq results use maximum-cardinality matching requiring both IoU>=.50
and exact casefold transcription; Precision/Recall/F1 are reported, with recall
over **all care GT words**. A secondary metric uses the existing Unicode
punctuation/symbol stripping. Empty text is never counted as a correct reading.
The ratio of exact matches to geometry matches is exposed with a warning: the
two maximum matchings can differ, so it is not fixed-pair recognition accuracy.

The recognizer is the existing pinned official PARSeq runtime/checkpoint pair,
AR=True/refine_iters1. The exact PARSeq-S paper artifact remains unresolved.
Exact RGB crop SHA256s deduplicate recognition across all seven pipelines.
Optional existing OCR caches must use those same pinned weights/runtime and
RGB hashes. Legacy missing decode fields are reconstructed from the pinned
runtime defaults (`ASSUMPTION`). Cache-provider protocols and source-record
hashes are preserved; conflicting same-crop transcriptions fail explicitly.
The first16 reused hashes in lexicographic order are replayed on the selected
device and must reproduce the cached strings. This is a limited cross-device
qualification, not a proof that all GPU/CPU outputs are identical. Memory,
precision, device and batch size are recorded; CPU execution/mixed cache devices
are declared `HARDWARE_DEVIATION`.

## Run

No dependency installation or download is needed. The repository's existing main
environment has already executed one real eight-crop PARSeq CPU batch. Provide
the existing local qualified PARSeq files explicitly:

```bash
PY=.venvs/context-str-main-py311/bin/python
RUN=/extra1/bmartelli/context-str-runs/totaltext-word-multitask-v1
PARSEQ=/extra1/bmartelli/context-str-artifacts/parseq
REV=1902db043c029a7e03a3818c616c06600af574be
OUT=artifacts/word-multitask-crop-refinement-v1

"$PY" -m scripts.evaluate_word_multitask_crops --mode approximate \
  --run-root "$RUN" --output-root "$OUT" --action preflight

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" -m scripts.evaluate_word_multitask_crops \
  --mode approximate --run-root "$RUN" --output-root "$OUT" --action all \
  --device cpu --cpu-threads 2 \
  --parseq-source "$PARSEQ/parseq-$REV" \
  --parseq-source-archive "$PARSEQ/parseq-$REV.tar.gz" \
  --parseq-checkpoint "$PARSEQ/parseq-bb5792a6.pt" \
  --reuse-ocr /extra1/bmartelli/context-str-runs/totaltext-word-kernel-100ep-v1/validation-pad0-parseq-v1/ocr \
  --reuse-ocr /extra1/bmartelli/context-str-runs/totaltext-v4-image-frequency-ablation-v1/validation-pad0-parseq-v1/ocr \
  --reuse-ocr /extra1/bmartelli/context-str-runs/totaltext-letterbox-word-f1-separator-parseq-val-v1/ocr
```

For a separate GPU qualification select a **new output directory**, export
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, and explicitly choose `--device cuda:0`.
`--action geometry` performs crop/geometry scoring without PARSeq; `recognize`
reuses a verified completed geometry phase. Completed phases and per-crop OCR
are checksum-checked before reuse. Changed config/code/artifacts, invalid cache
entries, missing samples and inference failures propagate explicitly. A file
lock prevents concurrent writers. Failure metadata is preserved in the new
output directory; historical training results are never overwritten.

`geometry_results.json` and `recognition_results.json` retain all per-image
outputs. `<variant>/<method>/samples/<sample>/` contains the refined instance PNG,
contours/boxes/crop recipes and map/box visualizations. The OCR directory retains
raw transcriptions and token diagnostics indexed by exact RGB crop hash.

## Qualification

Unit tests cover actual contour semantics and holes, kernel splitting and
fragment joins, distinct-center and transitive vetoes, true foreground distance,
single-center recomposition including rescue partitions, parameter validation,
foreground preservation, tiny words, true-area filtering, empty predictions,
one-to-one spatial/text matching, namespace isolation and tampered cache inputs.
The initial single-center fixture exposed a rescue partition without kernel
pixels; including only the partition containing the sole retained center fixed
that scientific edge case. This failure is recorded rather than suppressed.
All111 related tests passed, including25 new crop/localization tests, the
existing multi-task/provenance regressions, frozen contours, contour padding
and genuine PARSeq adapter tests. The complete real-data crop study and a
second invocation reusing both completed phases passed.

## Completed results (2026-10-09)

All seven pipelines processed every125 held-out TRAIN image and all983 care
words. Recognition reused258 exact-crop results (58 from the old kernel run,
113 from the image-frequency run, 87 from the separator run), and inferred
2,949 new unique RGB crops. Overall3,207 unique crops served all pipelines.
All16 selected legacy-cache replay transcriptions matched the current device.
PARSeq ran alone on CPU, FP32/batch8/threads2, AR=True/refine1; recognition phase
elapsed253.67 seconds, including crop I/O, cache qualification and scoring.
No OOM occurred. Allocated/reserved GPU-memory fields are null: this is CPU
qualification, not a new GPU-memory measurement.

The geometry below uses **original-coordinate GT boxes**, rather than the
native-v4-owner boxes used in the earlier training report. Exact readings require
both IoU>=.50 and casefold equality; the recall denominator is all983 care words.

| Model | Cropping | Care proposals | Original bbox F1 | Exact matched words | Exact recall/all GT | Exact end-to-end F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Text | findcontours | 886 | 59.07% | 425 | 43.23% | 45.48% |
| Text + Kernel | findcontours | 974 | 60.81% | 463 | 47.10% | 47.32% |
| Text + Kernel | existing seeded | 1029 | 61.33% | 480 | 48.83% | 47.71% |
| Text + Kernel | refined | 1023 | 61.62% | **481** | **48.93%** | **47.96%** |
| Text + Kernel + Center | findcontours | 975 | 57.51% | 428 | 43.54% | 43.72% |
| Text + Kernel + Center | existing seeded | 1109 | 58.99% | 465 | 47.30% | 44.46% |
| Text + Kernel + Center | refined | 1108 | 59.01% | 465 | 47.30% | 44.48% |

Original-coordinate localization P/R/F1 and exact-casefold P/R/F1 are all
retained in `recognition_results.json`; example primary values for the best
refined two-head model are localization60.41/62.87/61.62% and exact
47.02/48.93/47.96%. The symbol-normalized secondary scores are also retained;
the best model matches516/983 under that secondary normalization.

| Model | Cropping | Native bbox F1 | Merge rate | Split rate |
| --- | --- | ---: | ---: | ---: |
| Text | findcontours | 58.96% | 29.81% | 4.88% |
| Text + Kernel | findcontours | 61.73% | 19.94% | 7.12% |
| Text + Kernel | existing seeded | 62.13% | 17.40% | 9.66% |
| Text + Kernel | refined | 62.41% | 17.40% | 9.46% |
| Text + Kernel + Center | findcontours | 56.89% | 32.15% | 5.49% |
| Text + Kernel + Center | existing seeded | 58.41% | 26.65% | 10.89% |
| Text + Kernel + Center | refined | 58.44% | 26.65% | 10.89% |

The additional refiner performs7 shared-kernel unions for two heads and1 for
three heads, in6 images total. No center-region union activates on these maps.
These are mask-union operations, not counts of recovered GT words. For two
heads, raw care proposals fall by6 (one additional union concerns wholly ignored
predictions), split GT words fall95→93, merged GT words remain171, and original
IoU>=.50 matches rise617→618. One additional word becomes a valid exact reading:
in `total_text:1260`, the prior crops read `ELI` and `ZABETH`; their union reads
`ELIZABETH`. The two-head refiner gains1 exact GT match and loses0 versus existing
seeded cropping. Three-head refinement gains0/loses0 exact matches.

Against two-head simple contours, existing seeded cropping gains27 exact words
and loses10, net+17. The full refiner gains28/loses10, net+18. For three heads,
seeded and refined each gain56/lose19, net+37 over that model's simple contours.
These paired counts expose regressions instead of hiding them in net gains.
Most of the benefit was already supplied by the existing seeded separation;
the new conservative recomposition adds a **small** measured benefit. Center
does not establish a benefit over the two-head model with these checkpoints.

Independent verification recomputed geometry/foreground preservation and
original RGB crop hashes for875 pipeline/sample pairs, checked2,625 generated
exports and3,207 OCR record hashes, and independently recomputed every aggregate
from the per-image outputs. Completed-phase reuse returned the same artifacts
without fresh model inference. The lightweight provenance receipt is
`provenance/word_contour_refinement_v1_results.json`. Full reports remain in the
new `artifacts/word-multitask-crop-refinement-v1/` output directory; old runs and
model parameters are unchanged.

Recognition report SHA256:
`90740316514031ac4fba44dafe4e48f150ec3f806fb98abd3a9ae1b094bca392`.
Geometry report SHA256:
`cc1553af753492a095839f196130a196e58e882bedef16ad4d1e2589b435be78`.
Single-seed selection on the reused TRAIN holdout, mixed historical/current
cache devices, native bbox diagnostics and unmeasured official TEST behavior
remain limitations. No crop outcome was used to change thresholds or select
another checkpoint, and no STRICT/official paper accuracy is claimed.

The final ledger-preservation check initially reported a mismatch because
concurrent UCR/WCO entries had been appended by other work. Inspection confirmed
that the entire pre-task content remained intact; those unrelated entries were
preserved. Python compilation, JSON/HTML/archive checks and `git diff --check`
passed. The inspected task diff includes only the new crop implementation,
configuration, tests, documentation and lightweight provenance receipt.

## Direct inference API and visual review

The same optional localizer can be called by inference code receiving the
model's sigmoid maps and existing paired RGB canvas:

```python
from localize.word_guided_contours import word_contour_crops

proposals, instance_ids, audit = word_contour_crops(
    probabilities, canvas, (rgb.shape[1], rgb.shape[0]),
    postprocessing_config, refinement_config, method="refined",
)
crops = [rgb[y1:y2, x1:x2] for x1, y1, x2, y2 in
         (proposal["bbox"] for proposal in proposals)]
```

`refinement_config` is the new JSON's `refinement` section;
`postprocessing_config` is the existing model experiment's post-processing
section. No cached labels are needed in live inference. Selecting
`method="findcontours"` retains simple contours; `seeded` retains existing
head-guided separation. The new API does not modify the strict/original localizer.

The optional review archive `artifacts/word-guided-contours-review-v1.zip`
contains **all8 changed union groups**, not a GT-selected sample, their original
RGB fragment/union crops, actual OCR strings, context map/box images and the full
recognition JSON covering all125 images in each pipeline. Extract it and open
`word-guided-contours-review-v1/index.html` offline. The archive is5.4 MB; SHA256
`b6b171a53acf22db33a04fd89d5ce47c67c2cb32653611d38187332f271453b7`.

On the receiving computer:

```bash
scp bmartelli@paper:/extra1/bmartelli/projects/icdar2025-context-driven-str/artifacts/word-guided-contours-review-v1.zip .
unzip word-guided-contours-review-v1.zip
```

The original80 MB network-comparison ZIP is preserved. The new compact review
shows every additional union, including cases that do not improve the score.
