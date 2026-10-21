# Total-Text v4 image/frequency ablation (practical)

This experiment is a `PRACTICAL_APPROXIMATION`, not an exact reproduction of
the paper. It does not alter Algorithm 1, the qualified v4 target, the three
completed high-resolution models, PARSeq, or TEST data.

The frozen comparator is `highres_separator_residual`. Four independent arms
change one intervention each: moderate RGB-only photometric augmentation,
paired aspect-preserving letterbox, TRAIN-donor Fourier low-frequency amplitude
mixing, and a fixed three-band Haar-detail adapter at the first U-Net skip.
The Haar branch adds 192 trainable parameters and starts at zero contribution.
All four retain the same 1130/125 TRAIN split, flip/rotation, 500-epoch optimizer,
v4 mask, separation weight 3, attention skip floor 0.25 and native 256×256
output. Their exact parameters and classification are frozen in
`configs/approximate/totaltext_v4_image_frequency_ablation_v1.json`.

Model selection uses the same held-out TRAIN macro foreground IoU as the
frozen baseline. Post-selection validation reports native per-word proposal
IoU≥0.25/0.50/0.75, one-to-one precision/recall/F1 at 0.50, raw component
count, split words, components touching multiple words, and words touched by
those fused components. The separate pad-0 PARSeq validation uses the same 125
TRAIN images and original RGB crops, without component-area or top-ten filters.
Both casefold exact and casefold-plus-Unicode-punctuation/symbol removal are
reported with spatial IoU≥0.50 one-to-one matching. It does not use GT to form
crops or choose a threshold. The baseline is re-evaluated in the identical
validation OCR pass and its historical topology counts must match exactly.

The saved training geometry uses validation batches of eight, while cached
pad-0 OCR masks use inference batch size one. Binary contours can differ at
threshold-adjacent pixels even when the RGB input tensor is identical. Any
per-image geometry discrepancy triggers an exact batch-eight replay of the
frozen checkpoint and an immutable `geometry_batching_audits/<variant>.json`.
The report is accepted only when this replay reproduces the frozen per-image
training geometry; cached batch-one masks and PARSeq transcriptions are never
silently replaced. If the replay fails, validation remains blocked.
After all four training runs are complete, an interrupted OCR validation can
be resumed without repeating baseline validation or training using the host
script's `--validation-only` argument. It verifies/reuses the existing 125
mask shards per variant and the OCR records by their frozen hashes.

On PAPER, with the qualified Python/PARSeq environment and GPU available:

```bash
cd /extra1/bmartelli/projects/icdar2025-context-driven-str && bash scripts/run_totaltext_v4_image_frequency_ablation_host.sh /extra1/bmartelli
```

The command runs in the foreground and prints preflight, epoch and OCR
progress. Completed runs and crop OCR are verified/reused on restart; no
completed result is overwritten. Training checkpoints and validation artifacts
are stored under
`/extra1/bmartelli/context-str-runs/totaltext-v4-image-frequency-ablation-v1/`.
The aggregate OCR report is `validation-pad0-parseq-v1/results.json` there.
The original ablation configuration and checkpoint contracts specify 500 epochs.
Following the user-requested compute reduction, the host launcher now passes
`--stop-after-epoch 100` for unfinished and future arms. This is an explicit
`ASSUMPTION` and a **shortened practical experiment**, not a completed
500-epoch comparison: the run manifest records configured, requested and
completed epochs. The already completed photometric 500-epoch run is reused
unchanged; the interrupted letterbox run resumes from its verified checkpoint.
Because budgets differ, cross-arm differences are diagnostic rather than a
controlled comparison of image interventions. The remaining 500-epoch frozen
baseline is likewise a reference, not a matched-budget control.

Only after examining this TRAIN validation report should one arm be frozen and
evaluated once on official TEST. Do not pick an arm from TEST results.

## Frozen letterbox TEST-300 follow-up

Letterbox was selected on the completed TRAIN-125 report (SHA256
`c42e05c02369bcf32f96109b91b60adc1e451e2d79a9c4a4b42a1a4efee2b9ca`),
before this follow-up reads TEST labels. Its selected checkpoint SHA256 is
`5fca2ffc15b9689abbd5d75e80710ddd38d6419d9610fed840b2c54bb01f3991`.
The isolated TEST script uses the same batch-one letterbox transform,
native-256 threshold, inverse letterbox box mapping, zero padding, no area
filter, no top-ten cap, PARSeq checkpoint, and one-to-one word matching as
the TRAIN validation diagnostic. The official TEST-300 denominator is 2,204
care words. It saves 300 native masks, image-level proposals/GT, per-crop OCR,
and per-image matching before reporting aggregates; interrupted work resumes
from verified hashes. It does not modify the paper-faithful Algorithm 1 path.

On PAPER, run once in the foreground:

```bash
cd /extra1/bmartelli/projects/icdar2025-context-driven-str && bash scripts/run_totaltext_letterbox_pad0_test_host.sh /extra1/bmartelli
```

Output: `/extra1/bmartelli/context-str-runs/totaltext-letterbox-pad0-parseq-test-v1/results.json`.
This is `PRACTICAL_APPROXIMATION`, not a Table 5 reproduction. Do not use
TEST results to alter the selected checkpoint, preprocessing, or localizer.

## Post-TEST threshold sensitivity diagnostic (TRAIN only)

The TEST result above remains frozen at sigmoid threshold 0.5. To answer
whether other native-mask cutoffs produce different word crops, an isolated
TRAIN-125 sensitivity diagnostic predeclares thresholds
`{0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65}`. It runs the frozen letterbox
checkpoint once per TRAIN validation image, persists float32 native probability
maps, derives independent contour crops with pad 0 at each cutoff, and shares
PARSeq outputs by RGB-crop SHA256. It refuses to proceed unless every 0.50
native mask and every per-word 0.50 metric reproduces the previous frozen
TRAIN-125 validation result. The report includes exact casefold and
symbol-stripped word-ID intersection matrices, plus words gained/lost versus
0.50. GT is used only for post-hoc evaluation. No TEST images are loaded and
no new threshold is automatically selected or promoted. This is an
`ASSUMPTION`-based `PRACTICAL_APPROXIMATION` diagnostic, not paper Table 5.

## Two concurrent 500-epoch letterbox continuations

The completed original letterbox run has 100 epoch records but retained only
its best checkpoint from epoch 71 (including Adam state and RNG). Its epoch-100
state cannot be resumed exactly. Two isolated, matched branches therefore
restore that same epoch-71 checkpoint and run epochs 72–500: a weight-3
separator-loss control and one predeclared weight-5 separator-loss intervention.
The latter changes only the false-positive cost on valid inter-word separator
pixels; it does not alter architecture, input, GT mask, split, or inference
threshold. Both retain only their best checkpoint once complete and report
validation foreground IoU plus word-geometry/fusion metrics. The original
100-epoch run and its TEST result remain untouched. This is an `ASSUMPTION`
and `PRACTICAL_APPROXIMATION`, not a paper training protocol. Parallel use of
one GPU is recorded as `HARDWARE_DEVIATION`; the launcher checks for at least
16 GiB free before starting the two foreground child processes.

### Fixed-checkpoint word audit

The two continuation runs completed, but the weight-5 branch retained its
source epoch-71 checkpoint. A separate TRAIN-125 word-level audit first
verifies all 452 model tensors are byte-identical and reuses the frozen source
PARSeq results for that branch. It runs only the selected weight-3 epoch-263
checkpoint at batch one, with native threshold 0.5, pad 0, no area or top-ten
filter, and qualified PARSeq. Per-word source/control IoU and exact casefold or
symbol-stripped matches, including recovered and damaged GT identities, are
saved separately. TEST remains untouched and no checkpoint/threshold is
selected from OCR. This is a `PRACTICAL_APPROXIMATION` diagnostic.
