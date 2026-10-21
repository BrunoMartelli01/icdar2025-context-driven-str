# U-Net crop resegmentation experiment

This opt-in **approximate reproduction** implements the requested procedural
variant: full-image U-Net → `findContours` → RGB crops with context → the **same
U-Net** → new contours. It does not retrain/change the network or modify the
existing pipelines, recognition, captioning, DeepSolo or STRICT configurations.
The paper's sections 3.2–3.3 and Algorithm 1 have only the first segmentation
pass. Scientific choices and limitations are recorded in **UCR-001** in
[PAPER_AMBIGUITIES.md](PAPER_AMBIGUITIES.md).

## Behavior and coordinate units

- The standalone driver loads a checksum-verified trained checkpoint with
  `strict=True` through the existing segmentation checkpoint loader. Training
  config/run, architecture, input/output shapes, skip floor and geometry must
  match; recorded practical model/transform source hashes are verified.
- First pass and crop passes share one model, eval mode, float32, no gradients,
  fixed input shape, ImageNet normalization and the same resize geometry. Crop
  input is original RGB, never the mask, an overlay or an already resized crop.
- Native sigmoid probabilities are thresholded with `>=`, then the binary mask
  is projected with nearest interpolation to original RGB dimensions. For
  letterbox, project to the original square canvas and remove its padding.
  Contours, area thresholds and output padding all use **original RGB pixels**.
  This is an explicit new baseline, different from the previous native256
  contour/padding experiments; do not compare their counts as identical controls.
- Extract external contours, measure each associated 8-connected component's
  true foreground-pixel area (holes excluded), filter and sort by that area.
  Ties use bbox coordinates. Never use bounding-box area as component area.
- Expand each padded seed bbox by
  `ceil(fraction * bbox_width/height) + context_margin_pixels` on each side,
  clamp to image boundaries, and infer **once per retained seed**. No recursive
  refinement, morphology, GT-based selection or alternative recognizer occurs.
- Remap all child contour boxes by the crop's original-image offset. Keep a
  separate baseline arm and refined arm. Exact duplicate padded boxes are
  removed with provenance; nonidentical overlapping boxes remain. Rank final
  proposals by component area and apply the configured final cap.
- The profile uses margin fraction **0.5**, minimum context **4 pixels**, area
  threshold **0**, output padding **0**, and caps **10** at each stage. These
  are explicit, uncalibrated assumptions. Override them only in a separate
  `configs/approximate/` profile, with corresponding scientific-choice records.

The reusable API is `localize.unet_crop_refinement.refine_contour_crops`.
It accepts an RGB array, its initial binary mask, an existing callable segmenter
returning a binary mask at its input RGB resolution, and `CropRefinementConfig`.
It returns a JSON-compatible report plus separate mask/crop arrays. This also
works with the existing frozen author-protocol adapter **before its release**.
The driver supports the author 256→128 and practical highres 512→256 single-head
architectures; other architectures require an explicitly qualified adapter.
For author runs, pass the run's exact saved `resolved_config.yaml` as the
training config. For practical runs, pass the config hashed by `run_contract`.

## Running the existing Total-Text letterbox checkpoint

The new profile matches the existing `letterbox` practical checkpoint. All input
and output paths are explicit arguments. Run from the repository root; replace
these example locations for your machine, and use a fresh output directory:

```bash
DATA_ROOT=/extra1/bmartelli/datasets/context-str/scene-segmentation-v1
TRAINING_RUN=/extra1/bmartelli/context-str-runs/totaltext-v4-image-frequency-ablation-v1/letterbox
.venvs/context-str-main-py311/bin/python -u -m scripts.diagnose_unet_crop_refinement \
  --config configs/approximate/unet_crop_refinement_v1.json \
  --training-config configs/approximate/totaltext_v4_image_frequency_ablation_v1.json \
  --training-run-manifest "$TRAINING_RUN/run_manifest.json" \
  --checkpoint "$TRAINING_RUN/best.pt" \
  --images "$DATA_ROOT/datasets/total_text/train/images/11.jpg" \
           "$DATA_ROOT/datasets/total_text/train/images/147.jpg" \
  --device cuda:0 \
  --output-dir /tmp/unet-crop-refinement-review
```

Use `--device cpu` if CUDA is unavailable. CPU smoke tests below used
`OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`; actual thread count is saved. No packages
or environments are installed. Existing output directories are rejected rather
than overwritten. An input/provenance/config mismatch or unavailable CUDA
fails before model allocation/output creation. Incompatible checkpoint tensors
fail during strict loading. Runtime failures write
`failure.json` with the current image/stage and requested/completed counts, then
propagate the error; a failed image is never silently dropped.

## Reviewing outputs

Each image directory contains `baseline-boxes.png`, `refined-boxes.png`, baseline
and final RGB crops, each expanded RGB seed crop, its first/second-pass masks,
its original component-support mask, native masks/probabilities and a hashed
artifact inventory in `report.json`. `resolved-config.json` records complete
configuration, choices, checkpoints, training provenance, input hashes,
software/device metadata and source hashes. `summary.json` records completion,
backend settings and memory qualification. CUDA runs record peak allocated and
reserved bytes for the single-model run. CPU runs report those measurements as
unavailable, not zero.

Review `seed_associated_after_components`: it counts predicted 8-connected
components sharing at least one foreground pixel with the original seed.
Also review the original seed's retained foreground fraction, ROI mask IoU,
added/removed pixels and foreground touching each crop edge. Total ROI component
counts include neighboring words; they are deliberately reported separately
from seed-associated counts. These are **diagnostic signals, not correctness
labels**. A lower component count can mean recovery, fusion or missed text.
Border contact can be legitimate at an image boundary. A foreground model
does not guarantee word-instance boundaries.

An empty second pass remains empty, is recorded, and produces no substitute
crop. A missed full-image component gets no seed/crop. Context margin cannot
recover text outside the expanded ROI. Neighboring words and background can
be admitted; overlapping crops can produce overlapping hypotheses. The union
mask is raw ROI evidence with zeros outside examined ROIs, **not a replacement
full-image prediction**, and is never relocalized to generate final proposals.

## Qualification and preliminary observation

Real CPU smoke: official TRAIN images **11 and 147**, same letterbox checkpoint
**epoch 71**, SHA256
`5fca2ffc15b9689abbd5d75e80710ddd38d6419d9610fed840b2c54bb01f3991`.
No annotations or OCR were read. Image 11 keeps 10 final proposals from 10
seeds; image 147 changes from 3 to 7 proposals from 3 seeds. Visual inspection
shows **more fragmentation of “MOTORS”** in image 147. This smoke does not
establish improved word segmentation or recognition accuracy. CPU inference is
a hardware deviation for this engineering smoke, not a paper/GPU result.

Final qualification: **103 tests passed**, including existing localization,
frozen segmentation and frequency-gate regression tests. Final real artifacts:
`/tmp/unet-crop-refinement-real-smoke-v4/`. All 122 artifact hashes are audited;
repeat CPU inference produces identical paired masks, probabilities and crops.

Focused tests cover actual component area, holes, coordinate mapping, exact
threshold boundaries, unchanged qualified letterbox preprocessing, fusion/split
and empty predictions, deterministic limits, config isolation, provenance,
strict checkpoint compatibility, artifact hashes and propagated simulated OOM.
An initial test-collection failure from duplicate file basenames was corrected
by renaming the driver test. CUDA execution, actual peak VRAM and physical GPU
OOM behavior remain unqualified in this session; synthetic OOM tests qualify
the error contract only. Do not infer RTX 3070 fit from CPU success.

Additional variants to consider separately: repeat suspicious crops at a second
context margin and compare seed-associated masks, or form a joint crop around
nearby fragments **before** resegmentation. Both need separately declared
approximate profiles and evaluation: simply joining components can also join
different words. No such merge/selection rule is silently enabled here.
