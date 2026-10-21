# AdaSpot Mask-Align v1

**Approximate reproduction / staged research extension.** Authorized after the
positive full1255 Gap results. Original paper sections3.2–3.3 specify rectangular
component localization and OCR; they do not establish Mask-Align. The supplied
author proposal section4.3 describes skeleton→cubic centerline→distance-transform
height→normal feature sampling8x32, with rotated fallback for branching skeletons.
Section4.6 requests16x64 for a later deep head. No implementation was present:
the existing Phase0 PCA/Bézier paths rectify RGB, rather than shared U-Net features.
See `ADASpot-MASK-ALIGN-001` in `PAPER_AMBIGUITIES.md` for classifications.

The new isolated `models/adaspot_mask_align.py` exposes mask-only geometry and
FP32 differentiable bilinear sampling of actual feature tensors. It has no GT,
OCR text, confidence or routing input. Geometry remains discrete/detached; gradients
flow through F. No old model, default, trained tensor or strict configuration is
changed. Light/deep OCR heads, feature-to-token training, calibrated gate and joint
training remain separate future interventions. In particular PARSeq cannot consume
the existing F64 directly without a new trained connector/head.

Predeclared assumptions:

- Reuse deterministic repository Zhang–Suen thinning. In the8-neighbor graph,
  remove a diagonal edge when an axial intermediate pixel connects its endpoints,
  avoiding triangle shortcuts in digital staircases. Genuine branches fall back.
- Fit a cubic transverse coordinate over the dominant rotated-rectangle axis,
  oriented left→right when mostly horizontal and top→bottom otherwise. Solve
  scaled least squares with explicit rank checking. Require a single chain with
  at least4nodes and4distinct projected positions; reverse steps beyond .5native
  pixel, cycles, disconnected masks/skeletons and degeneracy fall back.
- Test64points on the fitted skeleton extent; more than25% outside the mask
  falls back. No geometry option is fitted from TEST recognition outcomes.
- Sample uniform principal-axis coordinates over the full word-pixel extent,
  including half-pixel support. This is not uniform arc length and excludes
  folded/non-graph curves. EDT half-height interpolates skeleton radii, with
  constant endpoint extension and .5nativepixel floor. This preserves endcaps
  better than ending at the shrunken skeleton. No extra padding is introduced.
- Rotated fallback covers projected square pixel support and retains the same
  instance ID. No instance is dropped because fitting fails. Per-instance method
  and reason are saved. Empty masks are errors, never dummy samples.
- Grids use pixel-center xy coordinates, bin centers, bilinear zero padding and
  `align_corners=False`. Native→RGB inversion uses the existing square letterbox,
  integer floor left/top padding and explicit half-pixel mapping.
- The frozen original Text/Kernel F64x256x256 is used, not Gap-decoder features.
  Existing RGB512→native256 differs from proposed stride4. Both8x32 and16x64
  currently sample the same F; a separate higher-resolution deep feature is absent.

CUDA inference executes on the actual GPU. PyTorch documents nondeterministic
CUDA `grid_sample` backward; when deterministic training is requested, the
explicit option transfers F/grid to CPU for sampling and returns the result to
the original device with autograd intact. The real-F smoke verifies nonzero,
finite gradients on CUDA. This is a `HARDWARE_DEVIATION`, with CPU/transfer costs;
it is not a silent GPU-training fallback. See [PyTorch grid_sample documentation](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.grid_sample.html).
No package installation, precision change or segmentation optimization occurs.

The paired diagnostic consumes the audited full1255-trained Gap's **fixed epoch1**
labels (.9/preserve). All300 TEST images,2204care words and2624care predictions
are preserved. Actual source M/K probabilities must replay within absolute1e-6
(FP32 arithmetic tolerance; not a segmentation threshold), strict source loading
and unchanged model/BN state are verified. All8x32/16x64 aligned feature tensors
are exported FP32 with IDs/checksums outside the repository.

Recognition is a separate **RGB geometry proxy**: original integer-axis crop,
matched bilinear axis, rotated rectangle and mask geometry, all three warped
views at32x128 before the same frozen qualified PARSeq AR1. Its native RGB
re-encoding does not test actual feature OCR accuracy, AdaSpot head capacity,
H2 or shared-backbone speed. Original scoring boxes remain unchanged; no GT
or OCR-driven choice of view, fallback, thresholds or source checkpoint occurs.
Matching is maximum one-to-one bboxIoU>=.5 plus exact Unicode casefold, retaining
punctuation. Empty strings, duplicates, localization misses and allcare GT remain
in the reported denominators. Care/ignore admission is frozen from the source.

Primary300 contains known identical TRAIN/TEST RGB578 and656. Report the same
secondary298/2187 cohort excluding exactly these two, without filtering training
or silently replacing primary counts. TEST was historically explored; this
is not an untouched benchmark, official polygon score, Table5 or strict result.
The new independent audit checks per-image Hungarian counts, frozen boxes/IDs,
source hashes, all actual feature shapes/dtypes, crop hashes, fixed denominators,
baseline strings and recovered/damaged word IDs.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 \
  .venvs/context-str-main-py311/bin/python -m scripts.diagnose_adaspot_mask_align \
  --mode approximate --config configs/approximate/adaspot_mask_align_v1.json \
  --source-run /explicit/audited/fulltrain-run \
  --output-root /explicit/new/external/mask-align-run --device cuda:0
.venvs/context-str-main-py311/bin/python -m scripts.audit_adaspot_mask_align \
  --output-root /explicit/completed/mask-align-run
```

GPU memory receipts include model set, FP32, tensor shapes, batch size, actual
allocated/reserved peak and explicit OOM failures; segmentation and PARSeq load
sequentially. Actual RTX3090 qualification is not RTX3070/8GB qualification.
Focused tests cover coordinate values, gradients, identity/zero padding,
translation, odd letterbox padding, curved/straight masks and branch/cycle/
disconnected/singleton fallbacks. No training improvement is assumed.

## Completed qualification and diagnostic,2026-10-21

Run: `/extra1/bmartelli/context-str-runs/adaspot-mask-align-v1-20261021`.
All300images and2624care instances processed on actual RTX3090, FP32.
All source M/K probabilities replay **exactly** (maximum absolute error0),
complete source weights/BN digest unchanged. Actual F64 grids8x32 and16x64
saved for every instance, including fallbacks. Real-feature backward gives
28681nonzero finite CUDA feature gradients, L1=1.0019102; sampling backward
explicitly runs on CPU with autograd transfers. No optimizer step or new
segmentation/OCR training is claimed in this stage.

| Frozen PARSeq RGB view | Exact words /2204 | Hmean | Recovered / damaged vs original |
| --- | ---: | ---: | ---: |
| Original integer-axis | 1311 | 54.3082% | 0 / 0 |
| Matched bilinear-axis32x128 | 1300 | 53.8525% | 15 / 26 |
| Rotated rectangle32x128 | 1307 | 54.1425% | 22 / 26 |
| Mask geometry32x128 | 1170 | 48.4673% | 20 / 161 |

All views retain2624proposals,2204careGT and1612geometry matches. The measure
requires a word to be both localized (bboxIoU>=.5) and read exactly after casefold;
it is not the number of masks alone. Matched bilinear-axis controls sampling/
resolution changes: mask geometry is still130words worse,27recoveries/157losses.
This experiment does **not** support adopting mask-guided RGB crops. Keep the
preselected Gap1 and original axis crops as reference, without TEST-based
parameter changes. Gap1's prior gain1311vs1293M/K remains a separate result.

Secondary298/2187 (exact known RGBduplicates removed for allviews): original1296,
bilinear1285, rotated1292, mask1156. The negative geometry result persists.
These are exploratory approximate counts on historically explored TEST.

Of2624instances,2006use cubic centerlines (76.45%),618rotated fallback:
257short skeleton,174cycle/degenerate,134branch,41fold,9outside-mask fit,
3rank-deficient. All618fallback RGB crops are hash-identical to the rotated arm.
Differences between rotated and mask views therefore arise in cubic instances.
Posthoc native-center footprint rasterization covers96.72% of predicted cubic
support pixels (median instance99.12%); rotated fallback covers100%. This is
an approximate geometric diagnostic on **predicted masks**, not true character
coverage. It cannot establish that clipping alone causes the OCR regressions.
Variable EDT height, fitted normals and interpolation can distort characters;
these are hypotheses, not identified causes. A fixed-first-eight preview shows
the original/rotated/mask RGB crops, mask+grid and actual aligned feature summary,
without selecting examples by OCR outcome.

Feature/geometry stage took76.6seconds including CPU fits, disk compression and
RGB proxy generation, not a model latency measurement. Peak CUDA allocated
497704448bytes/reserved901775360bytes for frozen UNet, Mask-Align and RGB warp;
PARSeq FP32batch8 peak148423168/176160768bytes. Source input1x3x512x512,
F1x64x256x256, each feature grid1x64x8x32 /1x64x16x64, RGB warps1x3x32x128;
all original RGB tensor shapes are recorded in `geometry_inspection.json`.
No OOM. CPU bilinear backward is explicitly documented above; no8GB hardware
qualification or complete joint-training qualification follows.

Independent audit **PASS** checks2380source/input hashes,1200primary and1192
secondary image/arm cases,2624actual feature instances and8794RGB crop hashes.
Original1311baseline per-image counts/raw strings replay unchanged. ResultSHA256:
`cf07517084d7cf7b4f3dcffb1106081b4450897252ec8615e52a912c89d18fbe`.
49focused tests pass. The first expanded test invocation failed collection
because models/scripts test files shared a basename; rename the new runner
test to `test_adaspot_mask_align_runner.py` fixed it. No test/runtime/OOM failure
was suppressed. Earlier17and38focused checks also passed.

Actual feature-space OCR benefit remains unresolved: this negative frozen-RGB
proxy cannot establish performance of a head trained on aligned F. The next
isolated scientific comparison would train the **same light head** on matched
axis-vs-Mask-Align feature grids using TRAIN-only training/calibration, frozen
source and equal budgets. Keep gate/deep/joint training for later and require
independent evaluation rather than optimizing radius/thresholds on this TEST.
