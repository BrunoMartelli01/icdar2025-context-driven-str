# Centerline and radius: a new experiment on the frozen U-Net

**Approximate reproduction / experimental extension.** This experiment tests
whether a longitudinal line and local thickness help reconstruct complete words.
It neither replaces STRICT components nor reproduces TextSnake. The
[configuration](../configs/approximate/word_centerline_v1.json) records choices,
hashes, and classifications; scientific ambiguity:
[WCL-001](PAPER_AMBIGUITIES.md#wcl-001--word-centerline-and-radius-extension).

## Motivation

The [agent audit](AGENT_EXPERIMENT_REVIEW_2026_10_11.md) shows Text+Kernel kernels
to be a stronger reference than pixel links. New recrops gain/lose many identities
and increase false positives; greater mask coverage does not guarantee higher
OCR precision. Previous centers are points/offsets toward word centers, not
longitudinal lines. Those training runs are complete and are not relaunched here.

A line alone does not determine crop height. **TextSnake** represents words as
disks along their axis, with radius/orientation, and reconstructs the region from
those disks. This proposal uses the same geometric intuition with a private head
in the existing U-Net.
[Primary source, ECCV 2018, sections 3.1–3.5](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Shangbang_Long_TextSnake_A_Flexible_ECCV_2018_paper.pdf).

## Protocol fixed before training

- Total-Text TRAIN1255: 1.130 training images, 125 held out, 983 care words;
  seed 202503 and identical lists/hashes to earlier comparisons. Reused holdout,
  not independent verification on official TEST.
- Exact copy of Text+Kernel control V1, best epoch 2, with verified checksums.
  All source weights/buffers, including BatchNorm, remain frozen. The control
  is not retrained. This measures what a head can learn from existing features;
  failure does not rule out benefits from a future fully retrained U-Net with
  the new supervision.
- Two private3x3 convolutions, 128→32→32, followed by 1x1 with two channels.
  Input: decoder and first-encoder-level features. Output: line probability and
  positive local radius (`softplus + 0.5` native pixels). Initial line logit−2,
  radius 2 pixels, zero final-layer weights.
- Letterboxed RGB512×512, maps 256×256, identical preprocessing/augmentation to
  the qualified source. Adam1e−4, weight decay0.01, batch 8,20 epochs, FP32,
  TF32 disabled. Every value is recorded in the configuration.
- Checkpoint selected by native geometric F1 of `line_radius` at IoU≥0.5;
  ties: fewer merged GT words, recall, fgIoU, earliest epoch. Includes epoch 0.
  OCR outcomes from the three methods do not select the checkpoint.

Targets are additional; original GT stays unchanged. Each **filled word region**
is deterministically thinned after the same augmentation/resizing. Radius is
Euclidean distance to the boundary, with zero padding. Line supervision is a
band of half-width radius/5, clipped to the same owner. Band radius comes from
the nearest skeleton pixel. Fragments, small pixels, and branches are retained
and explicitly audited; a component erased by thinning keeps its interior pixel
farthest from the boundary, with row-major ties.

This is an **ASSUMPTION**: the Zhang–Suen raster skeleton need not coincide with
TextSnake's centerline from paired polygon sides. No orientation sin/cos is
predicted, endpoints are not shortened, and its striding algorithm is not used.
Loss is line BCE+Dice plus log-radius SmoothL1; radius weighting is equal per
owner within each image, then averaged per image. This is not TextSnake's
original loss. The first experiment does not also add clDice, avoiding two
simultaneous changes.

## Three crops on the same images and checkpoint

| Method | Rule | Experimental question |
| --- | --- | --- |
| `baseline` | Identical current Text+Kernel watershed | Preserved internal reference |
| `line_markers` | Predicted-line components seed growth inside frozen foreground; explicit fallback for components without seeds | Does the line separate fused words better? |
| `line_radius` | Predicted-line skeleton and predicted-radius disks, including outside frozen foreground | Does the representation recover incomplete crops? |

Lines use threshold 0.5 and 8-neighbor connectivity. Competing disks assign each
pixel to the smallest distance/radius ratio; ties retain row-major component
order. Clipping uses the RGB canvas, not GT. Radius is capped at128 pixels,
counting every cap. No area filter, crop limit, or padding beyond padding0:
these are approximate diagnostics, separate from paper Algorithm1. Final crops
are original-RGB bboxes; curved rectification is not introduced here.

The qualified public PARSeq checkpoint loads in a subsequent process. It is the
same approximate candidate as earlier comparisons; the paper's exact PARSeq-S
identity remains [UNRESOLVED, R-001](PAPER_AMBIGUITIES.md#r-001--exact-parseq-s-artifact-is-unidentified).
The primary metric requires original bbox IoU≥0.5 and exact Unicode-casefold
text under maximum one-to-one matching, retaining punctuation. All983 words
remain in the denominator. Only wholly ignored predictions are excluded, after
GT-free crop construction. Precision, recall, Hmean, and baseline-relative GT
identity gains/losses are saved. Symbol-free matching is secondary. Line F1 and
radius MAE over the GT band help distinguish learning from decoder problems.

## Local qualification and limitations

CPU qualification is in the
[target/oracle JSON](../artifacts/word-centerline-v1-qualified/target_qualification.json)
and [real smoke](../artifacts/word-centerline-v1-qualified/cpu-smoke/smoke.json).
**50 focused tests and 132 related regressions** pass; shell syntax and CLI are
checked. The [small qualification record](../provenance/word_centerline_v1_qualification.json)
records hashes, completed checks, and pending work. Over125 images, all 983
owners survive, each with a connected skeleton; no band component joins distinct
owners.26 components have a one-pixel skeleton.389 owners have pixels with more
than two8-neighbors: this includes discrete junction geometry and does not prove389
actual longitudinal branches.

With **GT line and radius**, reconstruction produces983 proposals and 968 native
IoU≥0.5 matches: precision/recall/F1 **98,47%**, macro fgIoU85,02%, macro fgF1
91,83%. The decoder still produces two instances containing pixels from multiple
owners and two words with multiple overlapping instances. Discrete disks do not
perfectly reconstruct every bbox. This is a **target oracle**, not network
performance, OCR, or an absolute recognition ceiling. An ideal line is a useful
representation; its prediction remains unverified.

The CPU smoke uses two512×512 images, batch 2/workers0, real forward/backward,
and every decoder: the head changes while source tensors remain identical.
Reduced batch and CPU are recorded as hardware deviations. CUDA/NVIDIA driver
are unavailable in this environment: no full training, PARSeq evaluation, or
new-suite VRAM measurements have been performed.

During preparation, the first preflight failed an overly restrictive check:
the historical reference contains resized owners not yet clipped to target/
validity, while the dataset clips them. The check now compares the same clipped
maps, keeping all expected IDs in the denominator; a dedicated test covers this
case. The original failure is preserved in
`artifacts/word-centerline-v1-qualification/failure.json`.

## Command on the GPU machine

From the repository root:

```bash
bash scripts/run_word_centerline_host.sh \
  artifacts/totaltext-curved-word-boxes-v4 \
  /extra1/bmartelli/context-str-runs/unet-word-research-v1/control \
  /extra1/bmartelli/context-str-runs/word-centerline-v1 \
  /extra1/bmartelli/context-str-artifacts/parseq
```

The command checks artifacts, runs the target oracle, a **batch 8** GPU smoke,
20 head-only epochs, all crops, then PARSeq. It neither installs dependencies,
downloads weights, nor relaunches old branches. `DEVICE=cuda:0` and `PYTHON` are
optional explicit runtime selections. It uses the existing main virtualenv.
Suite and single-GPU locks serialize this launcher; they do not control older
processes started without that lock.

Rerunning the same command resumes `last.pt` with optimizer/RNG states, or
verifies/reuses completed results. Changed config, source, hardware/device,
preprocessing, checkpoint, or split requires a new output. `best.pt` and
`last.pt` are retained, without per-epoch checkpoints. An incomplete run cannot
perform final evaluation. OOM never changes batch, precision, or model
automatically; the failure records model, input, batch, and allocated/reserved peaks.

Result to inspect:
`/extra1/bmartelli/context-str-runs/word-centerline-v1/centerline_ocr_comparison.md`
and its JSON. Maps and crop recipes explain gained/lost words. The new method's
benefit is **UNRESOLVED** until that comparison is complete.

Engineering tests:

```bash
.venvs/context-str-main-py311/bin/python -m pytest -q \
  tests/segmentation/test_word_centerline.py \
  tests/localize/test_word_centerline_decode.py \
  tests/scripts/test_word_centerline_runner.py \
  tests/scripts/test_word_centerline_ocr.py
```

## Other new directions to test separately

**TextField** learns a directed field pointing inward from the nearest boundary
to group irregular words. It is an alternative to lines/radii and differs from
previous global-center offsets; it is not implemented in this suite. Its primary
source describes text/non-text magnitude and directions for separating instances:
[TextField](https://arxiv.org/abs/1812.01393).

**Soft-clDice** can be a later loss ablation to reduce line breaks. It was studied
for tubular structures such as vessels, neurons, and roads; alone it does not
guarantee separation of neighboring words. Measure merges and OCR too:
[clDice, CVPR2021](https://arxiv.org/abs/2003.07311).

Before changing the backbone, compare line loss/recall and radius quality: if
frozen features cannot learn them, a reasonable next step is a **new** decoder-
retraining ablation with an equal-budget control, retaining this baseline.
Its usefulness is a hypothesis, not an available result.

Rotated/Bezier rectification, distance markers, watershed/kernel, embeddings,
pixel links, and adaptive thresholds already exist here. They are not proposed
as novel; [existing Bezier rectification](../diagnostics/curve_rectification_train.py)
can be reused in a later comparison on new crops.
