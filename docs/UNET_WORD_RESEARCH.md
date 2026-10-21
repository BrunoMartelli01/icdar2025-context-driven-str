# U-Net: original glyph masks and modified word regions

This is an **approximate reproduction / experimental extension**. It does not
change the original models, Algorithm1, recognition stack, existing experiments
or STRICT configuration. Scientific decisions are recorded in UWR-001.

## What the two ground truths mean

The qualified original Total-Text TRAIN manifest contains distributed pixel
foreground: character strokes with gaps between letters. It is a binary mask,
not individual character IDs or character-class labels. The modified v4 target
fills a smooth region for each word and retains a separate integer word-owner
map. They are complementary supervision, not interchangeable masks.

`GlyphWordDataset` joins these manifests by sample ID and RGB SHA256. It reorders
the original records to the word dataset's order and applies the identical indexed
flip/rotation/letterbox to both targets. RGB tensors must agree exactly. Glyph
loss validity is the intersection of both ignore masks. Every image/target/owner
artifact is checksum-verified. Original masks and modified masks are not rewritten.
The preflight reports original and surviving native word counts explicitly;
small words lost at native256 remain in the evaluation denominator.
The completed preflight finds9285 source words and9284 native words: one small
TRAIN word disappears at this resolution. The held-out125 still has all983 words.

## Controlled branches

All branches initialize **every** original tensor from the qualified Text + Kernel
checkpoint selected at epoch61, SHA256
`e65691dda944478e9d8b4f022d45d0d3b21edc7c995274554b4c5ca31336766f`.
Added heads have separate declared-seed initialization. The state is transferred
strictly; missing/unexpected/different-shape tensors fail. No encoder download or
dependency installation occurs.

| Branch | Intervention | Inference |
| --- | --- | --- |
| `control` | Text + Kernel continuation | Existing seeded watershed |
| `glyph_word` | One additional glyph-pixel head, BCE+Dice | Existing word IDs plus unambiguous associated glyph support |
| `embedding` | Four-vector pixel head, within-word pull / between-core push | PAN-inspired4-neighbor growth from predicted kernels |
| `adaptive_threshold` | Learned threshold, border L1 and soft binary BCE+Dice | Pixel-specific probability threshold plus kernel watershed |
| `quality` | Frozen segmentation parameters **and BN**, small ROI quality head | Same masks; bbox-quality/coverage/purity scores and separate top10 ranking |

The glyph variant shares one decoder rather than loading two full U-Nets.
`glyph_word_instances` also accepts masks predicted by separate networks, provided
they have already been aligned to the same coordinates. It attaches a whole glyph
component only if it overlaps exactly one word ID by the declared fraction. It
never joins distinct word IDs. Ambiguous/orphan glyphs remain diagnostics; the
original word support survives. This does not claim character recognition.
Evaluation reports both this variant's word-head-only geometry and the combined
geometry, separating the training effect from the glyph association effect.

Embedding growth cannot cross missing foreground, and a merged kernel can still
seed a merged word. Unassigned foreground is retained in explicit fallback IDs,
with a counter. Embedding outputs are unbounded vectors, not sigmoid scores.
This is an adaptation of PAN, not a reproduction of its backbone or benchmark.

Adaptive threshold training is inspired by DBNet but keeps this repository's
full-word probability target. Its native border target and loss weights are an
explicit adaptation; no DBNet accuracy is assumed. Initial threshold is.65 to
align with the control. Each pixel uses its predicted threshold at inference.

Quality head receives context-cropped decoder features and the predicted mask,
never owner/GT labels. TRAIN annotations then supply best-word bbox IoU,
coverage and purity regression targets. Averaging is equal instance means within
each supervised image, then image means. Fully ignored/no-proposal images have
no quality objective; entirely unsupervised batches skip even weight decay.
The learned ranking score is the product of the three sigmoid predictions.
It compares area10 and quality10 **on the same predicted candidate set**; no
automatic confidence rejection changes the unconstrained masks. Training quality
on predictions of an already TRAIN-trained segmenter can overfit its in-sample
errors. An out-of-fold study and independent calibration remain future work.

## Frozen protocol and outputs

Configuration: `configs/approximate/unet_word_research_v1.json`. Additional50 epochs,
fresh Adam LR1e-5, FP32/batch8, seed202503, frozen ResNet50 encoder/BN, paired
TRAIN1130/held-out125, RGB512/native256. Text/kernel BCE+Dice is common to the
first four branches. Every branch includes epoch0 in checkpoint selection.
Select native word F1, fewer merges, recall, macro pixel IoU, earliest tie;
quality instead selects minimum validation quality MSE, earliest tie.

Masks are localized without GT. The native word metric is the existing maximum
one-to-one bbox matching at IoU>=.50, with all983 expected care words. Ignore
only wholly invalid predictions; partially invalid boxes remain complete.
Merge/split use any shared native foreground pixel and divide by all expected
words. Pixel macro metrics use a perfect score if prediction/target are empty.
Main geometry has pad0/no area filtering/no top10; the separate ranking experiment
has top10. This diagnostic protocol is not STRICT Algorithm1 or official polygon
evaluation. The reused TRAIN holdout and one seed do not establish generalization.

Each branch saves resolved config/runtime/code/dataset/split hashes, strict
checkpoint and optimizer/RNG state, epoch logs, selected best, memory/environment,
native probabilities, integer labels, auxiliary outputs, paired targets and
original RGB proposal receipts. Failures propagate and preserve failure metadata.
CPU and GPU, batch/worker overrides, TF32 flags and deterministic settings are
explicitly recorded. No automatic OOM fallback changes precision/model/batch.
Checkpoints committed before logs are recovered without repeating an epoch.
Partial budgets are IN_PROGRESS; completed comparisons require all50 epochs.

PARSeq runs in a separate process after all segmentation models are unloaded,
on original RGB rectangles. Qualified checkpoint/archive hashes, strict loading,
AR=True/refine_iters1 and exact RGB crop hashes govern the resumable cache. Raw
outputs remain per sample. OCR uses original-coordinate one-to-one bbox IoU>=.50
plus exact casefold matching; a secondary rate removes Unicode punctuation and
symbols. Wholly ignored predictions are recognized but excluded from scoring;
partial predictions retain their full boxes. All983 care words are denominators.
Gains and losses versus control are both reported. The exact paper PARSeq-S
artifact remains unresolved; this is the existing qualified approximate pairing.

Run from the repository on the GPU host (all paths are configurable arguments):

```bash
bash scripts/run_unet_word_research_host.sh \
  artifacts/totaltext-curved-word-boxes-v4 \
  /extra1/bmartelli/datasets/context-str/scene-segmentation-v1 \
  /extra1/bmartelli/context-str-runs/totaltext-word-multitask-v1/text_kernel \
  /extra1/bmartelli/context-str-runs/unet-word-research-v1 \
  /extra1/bmartelli/context-str-artifacts/parseq
```

The wrapper runs paired-target preflight, cached-map stability, GPU smoke,
training/export for each branch sequentially, PARSeq and the comparison. It is
foreground, locked and restartable. Read `comparison.md`, `comparison.json`,
`ocr_comparison.json` and per-branch `validation.json`. No model is automatically
promoted. Preserve an older output and choose a new directory if configuration,
source or artifact hashes change.

For a CPU smoke use the Python runner with `--mode approximate --action smoke
--device cpu --batch-size2 --num-workers0` and the explicit word/glyph/kernel/output
arguments. Use a different output directory from full GPU experiments.

## First measured diagnostic, without retraining

`scripts.diagnose_unet_mask_stability` recomputes the frozen word instances at
thresholds.55/.65/.75 from the qualified saved maps and confirms exact baseline
label equality. For each predicted instance, risk is one minus the minimum best
support IoU across thresholds. GT enters only post-hoc scoring. “Bad bbox” means
best individual care-word bbox IoU<.50; this is not one-to-one correctness, mask
pixel IoU, word completeness or OCR correctness.

On125 images,1029 care proposals,404 have bad bbox (39.26%). Risk AUC is.8361;
the206 highest-risk candidates have bad-bbox precision87.86%. Low mean pixel
confidence has AUC.8699, so instability does not outperform that simpler score
in this frozen diagnostic. Both must remain controls for a learned selector.
The risk score identifies candidates for inspection; it has not corrected masks.

Evidence: `/tmp/unet-word-research-stability-v1/stability.json` (initial diagnostic).
Five real FP32/batch2 CPU forward/backward/validation smokes and full1255-image
paired-target preflight passed. The test suite additionally checks source hashes,
strict warm transfer, spatial pairing, GT-free grouping, ignored/empty targets,
auxiliary gradients, top10 ranking, checkpoint/RNG resume, interruption recovery,
OCR normalization/matching and error propagation. Final counts/artifact paths
are reported by the completion receipt. CUDA is inaccessible in this session:
no physical GPU memory qualification or full new training benefit is claimed.
The first real OCR smoke failed because batch diagnostics omit decoder provenance;
the new cache now records the verified recognizer settings explicitly and the
three-crop real CPU replay passes. The qualified CUDA allocator initializer is
used before peak resets/model loading; ordering tests are mocks, not GPU evidence.

Sources: [PAN](https://arxiv.org/abs/1908.05900),
[Mask Scoring R-CNN](https://arxiv.org/abs/1903.00241),
[DBNet](https://arxiv.org/abs/1911.08947).
