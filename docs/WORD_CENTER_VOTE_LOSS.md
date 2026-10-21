# Coherent word-center vote loss: isolated v2 experiment

This **approximate reproduction / experimental extension** implements the first
proposed intervention: change the auxiliary U-Net's training objective. The
architecture, two-channel output, targets, transforms and localization remain
the same. No reliability head, Gaussian-center model, embedding branch, encoder
change or postprocessing change is added. Decisions are recorded in WCO-004 of
`docs/PAPER_AMBIGUITIES.md`.

The matched profiles are in
`configs/approximate/word_center_votes_totaltext_train_v2.json`. The historical v1
source files and selected checkpoints remain byte-for-byte unchanged, so their
qualification and existing inference continue to work.

## Objective

For pixel p belonging to word k, the network produces the same normalized
offsets `tanh(raw_x), tanh(raw_y)`. For the two new terms define its endpoint
`u_p = (p_xy + [W,H] * tanh(raw_p)) / S`, with `S=max(H,W)`.
The true center `c_k` is the valid visible owner-pixel centroid divided by S.
Coordinates remain isotropic for rectangular grids. Targets are constructed
after all paired augmentation, padding and validity operations, as in v1.

The loss is the weighted sum of:

1. **Regression:** the exact existing word-balanced SmoothL1 of normalized
   offsets, beta 0.01, weight 1. It anchors votes to the correct centroid.
2. **Consensus:** SmoothL1 between the endpoints of a word and their mean,
   beta 0.01, weight 0.5. The mean is differentiable. It penalizes dispersed votes;
   consensus alone could collapse all votes to a wrong center, so regression
   remains mandatory.
3. **Neighbor separation:** let `g_k` be the nearest eligible true-center
   distance, and j the closest eligible rival center to each predicted vote.
   Penalize `relu(||u_p-c_k||² - ||u_p-c_j||² + m*g_k²) / g_k`, with m=0.1 and
   weight 0.5. A correct vote has zero penalty. A vote approaching/crossing a
   rival receives a gradient toward its own center, including beyond the rival.
   Using a difference of unsquared distances would have a flat gradient there.

Each term averages pixels inside each word, then equally weights words inside
each supervised image, then supervised images. Background and ignored pixels
have no gradient. Empty images remain in sample/word counts; wholly empty
batches skip optimizer updates, including weight decay. No background
confidence is learned by this loss-only intervention.

Coincident true centers within 0.001 native pixels are an explicit ambiguity:
xy coordinates cannot distinguish such word identities. Exclude those rival
pairs and record affected-word counts, without shifting GT or omitting words
from the regression denominator. One-word/no-eligible-rival cases have zero
neighbor loss and zero wrong-neighbor diagnostic.

Target endpoints must agree with their paired owner centroid within 0.001 native
pixels; misaligned geometry fails explicitly. The full target preflight checks
that exact target votes yield total loss<=1e-6, a numerical smoke tolerance.
These weights, margin and numerical tolerances are predeclared assumptions,
not fitted to validation performance.

Rival selection processes 2048 pixels per chunk without retaining candidate
distance matrices in autograd. The distance to the selected rival remains
differentiable. Tests compare objective values and gradients across chunk sizes.
`neighbor_weight` is deliberately distinct from the existing segmentation
configuration's `separation_weight`.

## Matched experiment and metrics

`control` has consensus/neighbor weights0 and preserves v1's exact loss and
optimizer update. `coherent` has both weights 0.5. Both use the same supplied
single-output text checkpoint, strictly loaded, with a fresh zero vector head.
They do not continue the already trained auxiliary epoch82 checkpoint.

Both profiles retain 100 epochs, Adam FP32 / batch8, seed202503, frozen encoder and
BN, RGB512/native256 and the qualified Total-Text v4 TRAIN 1130 / validation 125
split. Dataset manifest, split, source-code hashes, full resolved configuration,
checkpoint SHA, optimizer/RNG state, epoch order and memory are recorded. The
runner uses separate `control/{smoke,train}` and `coherent/{smoke,train}` folders
and rejects incompatible existing provenance. Partial budgets remain
IN_PROGRESS; atomic checkpoint-before-log interruptions recover without
repeating an epoch.

To isolate training loss, **both branches keep the historical checkpoint
selection: minimum original validation offset SmoothL1, earliest exact tie**.
Composite loss, consensus, neighbor penalty, endpoint error, endpoint RMS scatter,
wrong-neighbor vote rate and competing/coincident word counts are also logged.
Direct center-count/crop-based selection is a separate experiment, not silently
mixed into this comparison.

Endpoint error is mean native-pixel Euclidean error per word, words per image,
then supervised images. Scatter is per-word RMS distance of endpoints from
their own predicted mean, aggregated the same way. Wrong-neighbor rate is the
fraction of word pixels whose endpoint is strictly nearer an eligible rival
than its own true center, equally averaged by word/image; no-competitor words
contribute 0 and exact distance ties do not count as wrong. Empty-image metrics
are null. Every expected word remains in survival/loss reporting.
These are GT-assisted validation diagnostics, not GT-assisted localization or
predicted word counts. Actual center detection, crop F1 and PARSeq benefit
require the completed paired experiment and frozen two-U-Net crop evaluation.
Independent TEST behavior and calibrated confidence remain unresolved.

Validation exports retain the existing `offsets.npz` fields and shapes:
normalized_offsets, target_offsets, owners, validity. Completed v2 run folders
can therefore be supplied directly as `--offset-run` to the existing paired
crop evaluator and fresh-image inference, with unchanged base checkpoint.
No non-strict checkpoint loading or automatic alternative model is used.

## Run

Use the existing environment; no installation or downloads are required.
For the coherent branch on the physical GPU host:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
.venvs/context-str-main-py311/bin/python -m scripts.train_word_center_votes \
  --mode approximate --variant coherent \
  --target-root artifacts/totaltext-curved-word-boxes-v4 \
  --base-checkpoint /extra1/bmartelli/context-str-runs/totaltext-word-multitask-v1/text/best.pt \
  --base-sha256 db9db69f77f1e62b03b82d073e8a37c603ff6ff8feceafc3623644980625e964 \
  --output-root artifacts/training/totaltext-word-center-votes-v2 \
  --device cuda:0 --action train
```

Repeat with `--variant control` for the matched baseline. Both reuse the same
output root but store independent branch contracts/checkpoints. Each full run
trains 100 epochs; the old completed v1 checkpoint is historical evidence, not
the new paired control. Resume by repeating the same command and configuration.

For a one-batch real-data smoke use `--action smoke --device cpu --batch-size 2
--num-workers 0`, with a separate output root. `--action preflight` qualifies all
1255 targets without loading a model. CPU, batch and worker overrides are
explicit recorded hardware deviations. CUDA allocated/reserved peaks and OOM
are logged on actual GPU runs; CPU tests cannot establish physical RTX3070 fit.
There is no automatic precision, batch-size or device fallback.

## Implementation qualification

104 new/relevant tests pass, including numerical gradients, restoring gradients
at rival centers, v1 control loss/gradient/optimizer equivalence, chunk-size
invariance, word/image balancing, empty/ignore behavior, centroid alignment,
strict configuration, export compatibility, checkpoint interruption/resume,
earliest ties, changed-loss rejection and synthetic OOM propagation.

All 1255 real targets pass centroid and perfect-vote qualification. Their 9285
source words yield 9284 native words: the one lost TRAIN word is the existing
resolution limitation. All 125 validation images retain all 983 words. Maximum
perfect-vote loss is 1.11e-12; no coincident-center words occur in this unaugmented
qualification. Those results do not assume augmentation cannot create them.

Both real-model CPU smokes pass with FP32/batch2, RGB512/native256 and workers0.
They process the same first training batch, with 15 visible words, the same RGB
order and original regression loss 0.0231734663. The coherent total is
0.0361144021, including consensus 0.0231734682 and neighbor 0.0027084029; the
control total is exactly its regression. Both perform one successful backward
and optimizer step. These are initialization/execution checks, not trained
performance comparisons.

Initial checks found two issues, both corrected before the full rerun: unsquared
distance contrast had a flat gradient beyond rivals, and a vote-weight name
collided with the existing segmentation weight. Tests now verify the restoring
gradient and separate fields explicitly. Diff/whitespace review passes and
historical v1 files were checked by hash. Lightweight qualification/provenance
is in `provenance/word_center_vote_loss_v2_qualification.json`; detailed local
smoke/preflight outputs are under `/tmp/word-center-votes-v2-{smoke,preflight}`.

No full 100-epoch training or new physical GPU execution has been run here.
CUDA is unavailable and `nvidia-smi` cannot communicate with the driver.
Allocated/reserved GPU peaks remain null, and synthetic OOM tests establish
error propagation only. New center/count/crop/OCR benefit remains unresolved.
