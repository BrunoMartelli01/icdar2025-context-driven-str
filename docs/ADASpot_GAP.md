# AdaSpot: Text + Kernel + Gap, first controlled experiment

**Approximate reproduction / experimental extension.** This is the first stage
selected by the user: word separation, with unchanged PARSeq as the OCR control.
Feature Mask-Align, two recognition decoders, and per-word gating are later
stages. The STRICT path and its components remain separate.

Inspected sources: local paper `papers/2503.15639v2.pdf`, sections 3.2–3.3 and
Algorithm 1, and the user-provided AdaSpot proposal, sections 4.1–4.2 and 6.
Proposal SHA256: `7e2a5a663f5ada85c607cb5200a3e17578617e9e387ab682c10e032f25bfb059`.
The proposal documents a new hypothesis; it does not retroactively identify
missing weights, decoders, or protocols in the original paper. Choices/evidence:
[ADASpot-GAP-001](PAPER_AMBIGUITIES.md#adaspot-gap-001--textkernelgap-controlled-extension)
and [approximate configuration](../configs/approximate/adaspot_gap_v1.json).

## Protocol and reuse

- Total-Text TRAIN1255: training1130 and holdout125, 983 care words, seed 202503,
  SHA256-verified manifests and splits. The holdout has already been used in many
  comparisons; results are exploratory, not independent validation. No TEST
  image trains or selects this experiment.
- Exact warm start from Text+Kernel, best epoch 61 of the multitask comparison,
  checkpoint SHA256 `e65691dda944478e9d8b4f022d45d0d3b21edc7c995274554b4c5ca31336766f`.
  All common tensors, including BN buffers, are copied. Incompatible keys,
  shapes, or dtypes abort startup. Final loading uses `strict=True`.
- Two separate training runs: `gap` and `control`, each with 50 additional epochs,
  fresh Adam, shared decoder/heads LR1e-5 and weight decay0.01. The new Gap head
  uses LR1e-4. Encoder and its BatchNorm stay frozen; the decoder adapts.
- Letterboxed RGB512, maps 256, FP32/batch 8, TF32 disabled, deterministic
  algorithms, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Both branches use identical
  sample order and sample/epoch-indexed augmentations.
- The new1x1 head, 64→1, adds 65 parameters. Zero weights and bias−2 preserve
  text/kernel and disable barriers at epoch 0. Old suites are not retrained;
  historical targets, checkpoints, and runs are not overwritten.

## Targets, loss, and decoder

The Gap target is built **after** the same word-owner augmentation/resizing:
valid background pixels within Euclidean distance≤3 native pixels of at least
two distinct owners. The neighborhood disk handles any orientation; fragments
of the same owner do not become different words. No M/K pixel is rewritten.
M pixels retained by interpolation without a nearest owner are also excluded.
A word already lost in preprocessing is counted, not invented. Radius3 and
this definition are `ASSUMPTION`, not paper parameters. Wide inter-word spaces
may produce no target; false barriers may cut letters or kernels. Improvement
is not assumed.

Loss: sum of masked BCE+soft Dice for M/K/G, weights1, smooth1. M BCE retains
the previous separator weighting3. Each loss is averaged over valid pixels per
image, then over supervised images. Don't Care has zero gradient. Entirely
invalid batches neither update Adam nor apply weight decay. The control branch
uses exactly the same M/K losses, without G loss.

Both branches use simultaneous8-neighbor BFS with deterministic ties, M≥0.65
and K≥0.5. The new branch excludes G>0.5 **before** kernel components and growth:
a barrier can separate fused kernels too. G=0.5 remains traversable. Every
foreground component without a kernel has an explicit fallback. The decoder
receives only probabilities and the RGB canvas, never owners, GT validity, or
transcriptions. Instance IDs stay distinct; contours are not extracted from
the binary union.

Padding0, no area filter, and no limit10 are diagnostic `ASSUMPTION` choices,
separate from Algorithm1. This control uses BFS for the proposal's requested
comparison; earlier checkpoint selection used watershed. That difference is
explicit. The same new map is decoded with Gap enabled/disabled, without another
forward, to isolate the barrier effect.

## Metrics, checkpoints, and results

Geometry uses exclusive native bboxes, IoU≥0.50, and maximum one-to-one matching.
Precision/recall/F1 are micro over all 983 care words/proposals. Merges/splits
use any shared native pixel; denominator983. A prediction is ignored only when
entirely outside GT validity, **after** inference; a partly ignored box is kept
whole. Words without predicted support or lost in targets stay in the denominator.
Pixel IoU/F1 are per-image macro over valid pixels of final M foreground without
barriers; both empty score1. Gap-off metrics on the same maps are separate.

Checkpoint selection: word F1, then fewer merged words, recall, pixel IoU, and
earliest epoch. Epoch0 is a candidate: harmful training may select the warm
start. `best.pt`/`last.pt`, optimizer, all RNG states, and epoch logs are atomic/
resumable. Changed config/source/input/device/backend requires a new output.
Final evaluation is forbidden until both branches complete their full budget.

After training, PARSeq loads in a subsequent process with its already-qualified
checkpoint, archive/revision, preprocessing32x128, AR, and refine1. RGB crop
hashes, raw texts, probabilities, and provenance are retained. Original-RGB
matching uses IoU≥0.50 and exact Unicode-casefold text, retaining punctuation;
the symbol-free variant is secondary. P=TP/proposals, R=TP/983,
H=2TP/(proposals+983). GT identity gains/losses are saved, including Gap-off.
These are local bbox proxies, not official polygon Hmean or Table3/5. The PARSeq
candidate is not identified as the paper's exact PARSeq-S artifact.

## GPU launch and monitoring

Use existing environments, datasets, and weights; the launcher does not install
or download dependencies or substitutes. Required paths:

```bash
bash scripts/run_adaspot_gap_host.sh \
  /path/to/qualified/totaltext-v4 \
  /path/to/totaltext-word-multitask-v1/text_kernel \
  /path/to/new/external/adaspot-gap-run \
  /path/to/qualified/parseq
```

`ADASPOT_PYTHON` and `ADASPOT_DEVICE` are optional explicit overrides; training
must use CUDA. The command runs preflight1255, a real batch 8 smoke,50 Gap epochs,
export, smoke/control50 epochs/export, then PARSeq and
`comparison.md`/`ocr_comparison.json`. Networks load sequentially. The lock
prevents another launcher on the same output; it does not interrupt other user
GPU processes. Logs show device, step, loss, and progress.

Errors stop the suite and save cause/traceback; OOM records model, shape, batch,
and allocated/reserved peaks. Batch, precision, and device never change
automatically. Peaks are per-process, not global GPU memory. Actual hardware
and reductions/offload must remain documented as `HARDWARE_DEVIATION`. The
host's RTX3090 does not qualify8GB RTX3070 operation. Effectiveness and
generalization remain `UNRESOLVED`.

## Implementation qualification

36 new tests and 64 relevant regressions pass: pairing, rotations, fragmented
owners, ignore/gradients, invalid batches, barrier/threshold, fused kernels,
deterministic decoder, real network shapes, strict transfer, RNG/Adam resume,
errors, denominators, and file/OCR boundary caching/tampering. The first pytest
attempt failed collection because three files shared a basename; the two
ambiguous names were distinguished without skipping tests. GPU smoke, full
preflight, and physical launch are recorded separately in external results and
the small qualification receipt. Fixture unit tests are not benchmarks or
training results for the new network.

Physical launch verified on 15 October2026, RTX3090: successful FP32/batch 8/RGB512
smoke, Gap head updated, no OOM. Smoke peaks:3.70GiB allocated,3.98GiB reserved.
Full training is active on `cuda:0`; the first two Gap epochs are recorded,
all 1130 images/142 steps per epoch, followed by planned control50 epochs and
PARSeq. First-two-checkpoint peaks:3.95GiB allocated and 5.44GiB reserved; these
do not qualify RTX3070. Attributing benefit to Gap still requires the matched
control and OCR.

[Qualification receipt](../provenance/adaspot_gap_v1_qualification.json) and
[run log](/extra1/bmartelli/context-str-runs/adaspot-gap-v1-20261015T125256Z/training.log)
record code, config, environment, device, and launch. Launcher PID1407798
continues in the background independently of the terminal. Results are
approximate; launch/smoke/early epochs do not mean training is complete.

## Prior-work audit, 15 October 2026

**The Gap principle already exists in this repository.** The generator
`segmentation/totaltext_word_regions.py` already includes background reached by
at least two dilated owners, combined with owner contacts, independently of
orientation. The old `PracticalTotalTextSeparatorHeadAGUNet` predicts M+separator,
adds 65 parameters, and subtracts the separator before connected components.
This suite did not introduce that idea.

Both complete historical100-epoch runs, identical TRAIN ordering in all epochs,
best-checkpoint checksums, and saved PARSeq results on the same holdout125/983
were checked. Results under their own protocol:

| Historical variant | Best epoch | Native word F1 | Merged words | Split words | Exact OCR /983 | OCR Hmean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M control |68|52,07%|199|77|384|39,75%|
| M + separator |94|48,87%|162|99|362|36,47%|

That experiment reduces merges but worsens the overall result. Old training is
not relaunched. The new experiment measures adding G to **M+K**, with warm
start, BCE+Dice, targets rebuilt after transforms at the declared native radius,
and kernel BFS. It is a combination ablation, not proof of Gap novelty.
Historical results are not causal controls for this new run: loss, threshold,
extraction, initialization, and ignored-proposal handling differ. New controls
remain matched.

External primary precedents consulted:

- [WordFence, ICIP2017](https://arxiv.org/abs/1705.05483): predicts borders around
  individual words to separate them. A precedent for the principle, without
  establishing equivalence to the full proposed M/K/G architecture.
- [Border Learning, ICCV2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wu_Self-Organized_Text_Detection_ICCV_2017_paper.pdf): text, border, and background classes;
  [author repository](https://gitlab.com/rex-yue-wu/ISI-PPT-Text-Detector) with FCN
  and border support. README available; the old Keras/Theano/Python2 stack was
  neither installed nor requalified. Direct CVF PDF access returned HTTP403;
  indexed content was consulted.
- [Repulsive Text Border, CVPR Workshops2020](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w34/Liu_An_Accurate_Segmentation-Based_Scene_Text_Detector_With_Context_Attention_and_CVPRW_2020_paper.pdf): repulsive links near distinct
  instance borders. Related objective, with different representation/loss.
  This PDF also returns403 directly; indexed text is available.
- [CM-Net, TIP2022](https://crabwq.github.io/pdf/2022%20CM-Net%20Concentric%20Mask%20based%20Arbitrary-Shaped.pdf): a head named Gap Mask, but G is the **internal** band between
  a word contour and kernel, auxiliary during training. It does not equal the
  external inter-word corridor used here. Original PDF consulted.

This search does not prove absence of prior work on the exact M/K/G combination.
This audit establishes neither first introduction nor superiority. The small
receipt `provenance/adaspot_gap_prior_audit_20261015.json` records artifact
identities/counts; originals remain unchanged.
