# Total-Text crop recovery suite

Run from the repository root:

```bash
bash scripts/test_totaltext_crops.sh
```

This runs the focused unit tests, all five crop experiments, qualified PARSeq,
and evaluation on the frozen 125-image held-out **TRAIN** population. This is an
**approximate reproduction / TRAIN diagnostic**, not a Table 5 reproduction or
an independent accuracy estimate. The full suite includes tiled segmentation;
it does not require CLIP, downloads, or additional package installation.

The runner selects CUDA when available and CPU otherwise. Scientific parameters
are in `configs/approximate/totaltext_crop_recovery_suite_train_v1.json`. Machine
paths are in the separate `totaltext_crop_recovery_local_paths_v1.json` profile;
relative paths resolve from the repository root. Every path can be overridden
by a corresponding CLI argument, or a different profile passed with `--paths`.
STRICT configuration is never consulted or modified.

For a real one-image smoke test of the same stages:

```bash
bash scripts/test_totaltext_crops.sh --limit 1
```

Useful explicit options:

```bash
bash scripts/test_totaltext_crops.sh --device cuda:0 --batch-size 16
bash scripts/test_totaltext_crops.sh --prepare-only
bash scripts/test_totaltext_crops.sh --without-tiles
```

`--prepare-only` measures candidate counts before committing to OCR. It still
performs the requested tiled segmentation. `--without-tiles` is an explicit
partial suite using existing masks; its resolved plan records the omission.
`--limit` outputs are always labeled `SMOKE_TEST`. Default execution processes
all 125 images and keeps every care GT word in the denominator.

## What the suite tests

All generators consume RGB and predicted probabilities only. Annotation files
are opened only after generation and OCR, for post-hoc evaluation. No recognizer
confidence, GT string, dictionary or dataset-specific word pattern selects a
crop. The original mask/crop remains available alongside the hypotheses.

| Arm | Procedure | Classification and limitation |
|---|---|---|
| `baseline` | Frozen threshold .5, RETR_EXTERNAL, zero padding, original RGB, inverse letterbox | `REPOSITORY_SPECIFIED`; source crop identities must match the frozen reference exactly |
| `threshold_tree` | Keep all components at eight thresholds; partition lower-threshold support using higher-threshold seeds; also keep unions of up to three spatially adjacent partitions | `ASSUMPTION`; threshold snapshots and seed descendants form hypotheses, without a persistence-based selection rule. Seeds can be individual letters |
| `erosion_reconstruction` | Erode components by radii derived from their distance-transform thickness; use separated cores as seeds for probability-guided geodesic flooding; crop original RGB around reconstructed regions | `ASSUMPTION`; foreground pixels are not erased from the OCR image. Erosion can also separate characters |
| `image_seams` | Combine probability and normalized RGB Sobel magnitude; find low-cost top-to-bottom paths and their transposed equivalents; try individual cuts and combined cuts with adjacent unions | `ASSUMPTION`; three corridor seeds per axis limit this family's search. Axial paths may miss strongly curved inter-word gaps |
| `tiled_segmentation` | Run the same checkpoint on four overlapping image windows, preserving letterbox preprocessing; invert window coordinates and retain components at three thresholds | `ASSUMPTION`; altered scale/context may change predictions. No TEST-dependent window selection |
| `crop_variants` | Independent inward/outward shifts of each box side, symmetric shifts, median-color artificial margins, and PCA-oriented crops of whole-image mask regions | `ASSUMPTION`; fractions use the smaller axial box extent. Orientation is straight-line, not curved rectification; tile proposals receive border/margin variants |

The geodesic flood is a deterministic marker-based minimax watershed on the
8-connected foreground graph. Path cost is the largest `1-probability` value
encountered; path length and label ID resolve ties. It cannot cross background.
The original support is preserved. Adjacent unions are ordered along the
principal axis of partition centers and do not guarantee correct word grouping.

There is no global area filter, top-ten cap, NMS, or one-choice-per-baseline-anchor
restriction. Exact recipe duplicates preserve all method memberships; identical
RGB crops share OCR by cryptographic crop hash. No near-duplicate is suppressed.
The seam budget and adjacent-union limit are explicit experimental search
budgets, not paper parameters. Crop counts can be large; these are recorded
instead of silently truncating the experiment.

## Metric

The primary comparison is **number of correct care words in the candidate pool**.
Both annotation and PARSeq string undergo Unicode `casefold`, followed by removal
of characters whose Unicode category starts with `P` or `S`. Accents, digits and
whitespace remain. There is no edit distance, substring credit or spelling repair.
Empty normalized strings never match.

A crop can recover a GT word only when normalized strings match exactly and
the crop's source bounding box has IoU >= .5 with the annotation. Maximum
one-to-one matching counts each GT occurrence once, regardless of how many crops
recognize it. It also prevents a single crop from counting as two repeated word
occurrences. All correct candidate IDs are saved as witnesses, independently
of the matching. This is GT-assisted **oracle pool recall**, not the accuracy of
a crop selector. Choosing a crop is deliberately deferred.

For rotated crops the spatial gate uses the enclosing, clipped source bbox of
the sampled quadrilateral. Artificial margins do not enlarge the source bbox.
This preserves a consistent bbox protocol; it is not polygon-IoU evaluation.
The IoU gate can reject a correctly read but geometrically unusual crop.

Reports retain both the original GT denominator (983 in the full frozen split)
and the count of nonempty normalized GT words. A symbol-only annotation can
never be recognized correctly under this protocol. Scores are not directly
comparable with earlier casefold-only 368-word or one-per-anchor CLIP tables.

Each method has a standalone pool, a `baseline_plus_*` pool, and cumulative
pools. `all_methods` measures the maximum recall of their union. Outputs include
correct/new/lost word IDs, geometrically matched words, all witness crops,
candidate recipe counts, and distinct RGB crop counts. Raw method pools can
lose baseline words; pools containing baseline cannot decrease maximum matching
cardinality, although the particular matched IDs can change in ambiguous cases.

## Artifacts, resume and hardware

The output path is printed at startup and completion. Its default is a directory
under the path profile's `output_root`, identified by the resolved plan hash.
Configuration, source hashes, model hashes, sample IDs, runtime options and paths
contribute to that hash. Changing them selects a different directory. Repeating
the same command resumes it. Explicit `--output-dir` refuses a conflicting plan.

- `resolved_configuration.json`: full scientific config, paths, source/model
  identity, software versions, seed, devices and backend settings.
- `tiles/`: original per-window probability arrays and checksum sidecars.
- `prepared/`: boxes, crop recipes, transformations, origins, memberships,
  pixel hashes and crop shapes, frozen before OCR and GT evaluation.
- `ocr_by_crop/`, `ocr/`: PARSeq strings, native diagnostics for new crops,
  crop hashes and per-image model-stage contracts.
- `evaluation/`: care annotations, exact matching IDs and all correct witnesses.
- `results.json`, `summary.md`: machine-readable and readable comparison tables.
- `segmentation_last_execution.json`, `parseq_last_execution.json`: model set,
  device, float32 precision, batch/input shapes, elapsed time and peak allocated
  and reserved CUDA memory. Creation-time model peaks are also saved in shards;
  a resumed execution may load no model and have zero new GPU allocation.
- `failure.json`: failed stage, exception, traceback and OOM flag. Errors are
  propagated, never replaced with another model or dummy output. A historical
  failure file is retained as an audit record after a successful resume.

Segmentation and PARSeq load sequentially. Segmentation batch one, PARSeq batch
16, CPU execution and sequential loading are recorded `HARDWARE_DEVIATION`
choices. Existing qualified OCR caches can have been produced with another
batch/device; their model/source/crop hashes are verified and upstream execution
metadata is retained. Cached/native and newly computed outputs form a frozen
diagnostic, not a claim of numerical equivalence across devices. Checkpoint
loading remains strict. CPU runs have null CUDA peaks, not a GPU qualification.

The authoritative motivation is paper sections 3.2, 3.3/Algorithm 1 and 3.5.
The new algorithms, budgets and oracle evaluation are user-authorized extensions
classified in `docs/PAPER_AMBIGUITIES.md`, entry `TT-CROP-RECOVERY-POOL-V1`.

## Implementation validation (2026-10-04)

- 23 focused generator/metric/cache/failure tests passed, plus 21 existing
  PARSeq/crop/threshold regression tests.
- Real CPU smoke execution on `total_text:14` ran four segmentation windows,
  generated 1,487 recipes (1,473 distinct RGB crops), and evaluated qualified
  PARSeq outputs. Baseline recovered 0/2 words; the union recovered 2/2.
  A repeated execution reused all tile/preparation/OCR shards, computed zero
  new OCR crops and reproduced the report. This one image is not an estimate
  of the method's full-population benefit.
- The new metric was replayed on the complete frozen baseline cache: every
  per-image matched GT ID agreed with the upstream normalized exact metric.
  Baseline is **400/983**, with **982** nonempty normalized annotations. The
  earlier 368-word casefold-only baseline uses a different normalization.
- Initial smoke attempts exposed a canonical-JSON versus file-byte hash
  mismatch and a lookup of annotation hashes absent from the frozen aggregate.
  Both were corrected: use the original canonical protocol identity and compare
  frozen GT content, then preserve the current annotation-file checksum in the
  evaluation shard. Dedicated cache and aggregate-schema tests now cover these
  boundaries. Failed attempts retain their `failure.json` records.
- CUDA was unavailable in the implementation session. GPU peak measurements
  and RTX 3070 execution are pending; CPU null peaks do not qualify GPU memory.
  The OOM propagation/reporting boundary was tested with an injected OOM
  exception. The complete new 125-image suite has not been run in this session.
