# Text-mask recall: loss and input resolution

Result class: **approximate reproduction**, exploratory frozen TRAIN holdout.
This experiment improves the **foreground support** supplied to the separately
trained center-offset network. It does not add an instance-separation head or
change that auxiliary network, the original model, or STRICT configuration.

The local paper `papers/2503.15639v2.pdf` sections3.2/3.3 and4.1 were inspected.
ResNet50/attention are paper components; Tversky,768 inputs and the coverage
protocol below are explicit assumptions, not the paper's BCE256/128 protocol.
Decisions are recorded in `docs/PAPER_AMBIGUITIES.md`, entry TRR-001.

## Existing work and the new comparison

The completed multi-task experiment has a qualified single-text model, full
original-resolution v4 targets and immutable validation maps. Existing code also
contains word-balanced BCE, word zoom, deep supervision and partial encoder
experiments. Those interventions are not duplicated or silently combined here.
In this first comparison **only Dice is replaced by Tversky**; the existing
per-word balanced BCE would introduce an additional factor and is deferred.

All branches strictly copy the single-text epoch56 checkpoint, SHA256
`db9db69f77f1e62b03b82d073e8a37c603ff6ff8feceafc3623644980625e964`:

| Branch | Loss | Original RGB resized to | Native output/target |
| --- | --- | --- | --- |
| control512 | BCE + Dice | 512x512 | 256x256 |
| tversky512 | BCE + Tversky | 512x512 | 256x256 |
| control768 | BCE + Dice | 768x768 | 384x384 |
| tversky768 | BCE + Tversky | 768x768 | 384x384 |

The unchanged architecture has **58,155,081 parameters in every branch**.
Higher resolution resizes original RGB and original-resolution qualified
mask/validity/separator/owner artifacts directly; it never enlarges saved256
targets. The isolated transform subclass retains paired flip/rotation/letterbox,
normalization and interpolation. Original transform size validation stays intact.

Each branch has50 additional epochs, seed202503, fresh Adam, LR1e-5, WD.01,
FP32/batch4, frozen encoder and encoder BatchNorm. Sample order and per-sample
augmentation are matched; the augmentation schedule restarts at epoch0 for
all branches. Batch4 and sequential execution are `HARDWARE_DEVIATION` choices
for the target8GB GPU. GPU fit must be measured, not inferred from CPU success.
There is no automatic batch reduction, mixed precision or OOM fallback.
The new control is necessary: historical100-epoch results used a different
batch, optimizer history and checkpoint-selection objective. Saved historical
maps are reused as a reference without rerunning their model/training.

Masked BCE retains the original separator weight3 and weight1. Dice/Tversky
have weight1, smooth1. Tversky is
`1 - (TP+1)/(TP + .3*FP + .7*FN + 1)`, using soft sigmoid counts per image,
then averaging images containing valid pixels. This is inspired by the
[Tversky paper](https://arxiv.org/abs/1706.05721), whose medical results do not
establish a benefit for scene text. Ignore pixels have zero loss gradient;
fully ignored batches skip optimizer updates, including weight decay.

## Evaluation and checkpoint selection

The population remains official TRAIN1130/training and125/validation,983 care
words, with the existing checksum-frozen split. No TEST or OCR is read.
Every prediction is sampled onto the **same canonical256 grid**, using CPU
bilinear interpolation of sigmoid probabilities with align_corners=False.
All branches use the unchanged canonical v4 GT owners, validity and canvas,
verified exactly against the current qualified512 dataset. This keeps input
resolution from changing the scoring raster, word population or denominators.
These metrics describe the qualified v4 regions, not official polygon Hmean.

Coverage is predicted foreground inside a word's valid GT region divided by
its valid GT pixels. Average each word equally across all983 expected words.
Missing raster words count as zero coverage and are reported. A complete word
has coverage>=.90. Pixel precision/recall/F1/F2/IoU aggregate global valid-pixel
counts; macro IoU averages images. Ignore and padded canvas pixels are excluded.
An empty prediction with nonempty GT has precision0; both empty yields perfect
pixel scores. No-word coverage is0. Empty/failed samples are never dropped.

Predeclared validation thresholds are .35/.50/.65/.80/.90. Among points with
pixel precision>=.90 and at least one TP, choose highest complete-word rate,
then mean word coverage, pixelF2, microIoU and lower threshold. Select the best
epoch with the same key and earliest tie; epoch0 is eligible. When no threshold
meets the precision floor, the best attainable precision point is explicitly
flagged unqualified. An unqualified result does not establish improvement at
the required precision. All thresholds and per-word rows are retained.

Foreground component boxes, merge/split and geometry F1 remain **secondary**
diagnostics. They do not select a checkpoint: a fused foreground region is
allowed to be separated later by the offset model. The comparison reports
loss-only, resolution-only and combined effects, plus every branch's fixed
threshold measurements. Different selected thresholds remain visible.

## Run and resume

From the repository root, after other GPU training completes:

```bash
bash scripts/run_text_recall_resolution_host.sh \
  artifacts/totaltext-curved-word-boxes-v4 \
  /extra1/bmartelli/context-str-runs/totaltext-word-multitask-v1/text \
  /extra1/bmartelli/context-str-runs/totaltext-text-recall-resolution-v1
```

The three paths are arguments, not embedded dataset/checkpoint paths. Use a
new output root. The wrapper performs CPU target preflight, four actual GPU
smokes, sequential training, selected-checkpoint exports and comparison.
It declares CUBLAS_WORKSPACE_CONFIG=:4096:8. No download/installation is needed.

The shared command is `python -m scripts.run_text_recall_resolution`, with
`--mode approximate --target-root ... --base-run ... --output-root ...`.
Actions: `preflight`, `smoke`, `train`, `evaluate`, `compare`; select an individual
branch with `--branch`. Device is explicit. `--num-workers` and `--batch-size`
overrides are recorded for every branch. `--stop-after-epoch N` retains the full
configured budget and allows an epoch-boundary stop/resume.
Reissue the identical wrapper command to resume interrupted training. Config,
code, dataset, split, checkpoint or runtime-device mismatch fails explicitly.
CPU smoke directories are separate from training and never become checkpoints.

Each branch saves best/last model, optimizer and RNG, JSONL epoch records,
initial epoch0 results, resolved run contract, and the complete validation grid.
Interruption between atomic checkpoint saving and log append is recovered;
inconsistent histories fail. Completed branches reuse their results.
GPU allocated/reserved peaks, precision, model set, shape and batch are logged.
Failures are propagated and saved separately with OOM information.

Final outputs include `comparison.json`, `historical_reference.json`, per-word
coverage, canonical and native probability maps, mask/GT visualizations and
original-coordinate component boxes compatible with PARSeq crop callers.
Canonical maps also align with the existing256-grid offset predictions;
coupled offset/PARSeq performance still requires a separate evaluation.

## Qualification

157 focused/related tests pass, including21 new tests for asymmetric loss,
ignore gradients, unchanged512 preprocessing,768 owner replay, precision-floor
selection, lost-word denominators, configuration isolation, strict checkpoint
resume, interruption recovery, deterministic exports and propagated synthetic OOM.
Full preflight checks both resolutions on all1255 images:512 retains9284/9285
words, losing one training word upstream;768 retains9285/9285. Canonical
validation still contains all983 words. This target survival is not a detection
improvement claim.

Real CPU smoke passes for all four branches: FP32/batch4, unchanged parameter
count, same four TRAIN samples/order, finite forward/backward and a nonzero head
update, followed by one real validation image per branch. Smoke losses are
not final performance results. Artifacts:
`/tmp/text-recall-resolution-preflight-v1/` and
`/tmp/text-recall-resolution-smoke-v1/`.
CUDA is unavailable to this Codex session: no50-epoch GPU result or RTX3070
memory qualification is claimed. The host wrapper performs those checks.

## CUDA allocator initialization fix

The first host GPU smoke failed with `RuntimeError: Invalid device argument`
at `reset_peak_memory_stats`, before model transfer or any training update.
This was a launcher bug: checking CUDA availability and seeding CUDA generators
did not create allocator statistics. The runner now reuses the existing
`segmentation.evaluate._prepare_cuda_memory_tracking`: select device, allocate
and release one tensor, synchronize, empty the cache, then reset peaks using
the integer device index. The allocator setup does not sample random values or
change the model, optimizer, precision, batch, loss or evaluation.

Reissue the **same wrapper command and output root**. An audited compatibility
path accepts only the exact original runner and this exact correction, with
unchanged scientific sources/configuration and the recorded unsuccessful first
CUDA smoke. It rejects weights, epoch histories, successful smokes, other errors
or unexpected files. The original preflight and contracts are archived byte for
byte in `cuda_initialization_archive/`; the failure and warm-start records stay
untouched. `cuda_initialization_transition.json` records their hashes and the
source transition. The preflight targets are reused, and the GPU smoke runs anew.
The root contract is committed last so interrupted metadata commits can resume.

Fix qualification:185 related tests pass, including allocator call order,
integer device index, error propagation, narrow provenance repair, rejection of
changed settings/artifacts and interruption recovery. The actual failed host
directory was copied read-only to `/tmp/text-recall-cuda-failed-run-replay-v1/`:
the corrected runner repairs its metadata and returns the cached preflight PASS.
The external host directory is unchanged until the user reruns the command.
GPU execution and physical VRAM/OOM behavior still need host qualification;
mocked CUDA tests do not establish GPU execution success.
