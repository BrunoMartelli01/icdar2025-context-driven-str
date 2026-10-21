# Auxiliary word-center attraction U-Net

The classic U-Net supplies the text shape. `WordCenterOffsetAGUNet` supplies
**where to look** inside that shape: every text pixel predicts a vector toward
its word's center. Converging vectors create a location signal even when the
classic mask connects adjacent words. This is an experimental extension labeled
**approximate reproduction**, not a STRICT paper result.

This differs from the repository's completed text/kernel/Gaussian-center
experiment: there is no kernel target and no Gaussian heatmap regression.
Two dense vector channels are trained with word-balanced SmoothL1; a vote map
is constructed from their predicted endpoints at inference. Gaussian smoothing
of that accumulator is postprocessing, not the previous learned center target.

Evidence inspected: local `papers/2503.15639v2.pdf`, sections 3.2/3.3 and Fig.3;
`models/unet_adv.py`, `models/word_multitask.py`, their target/loss/localization
implementations, and `docs/WORD_MULTITASK.md`. The paper specifies text masks
and connected components, not this auxiliary signal. Decisions and alternatives
are recorded as WCO-001 in `docs/PAPER_AMBIGUITIES.md` and in the isolated
`configs/approximate/word_center_offsets_v1.json` profile.

The auxiliary model reuses the practical frozen ResNet50 attention decoder and
its encoder BatchNorm policy. It runs independently of the existing text model,
whose class, output, checkpoint schema and parameters are unchanged. Native
output is `B x 2 x H/2 x W/2` in **x, y** order, with no sigmoid. Apply `tanh`
to obtain displacements normalized by **native width, native height**. Its
58,155,146 parameters include 65 more than the single-output practical model.
This is a full separate network, not a claim of negligible combined VRAM.

The new head starts at zero displacement. `initialize_from_text_model` strictly
copies a loaded single-output practical model with the same skip floor and
matching dtypes, resetting only `head.weight` and `head.bias` to zero. It returns
an explicit reset receipt and never mutates the source. This is warm-start
initialization; a trained auxiliary checkpoint must instead use strict loading
with architecture identity `experimental-word-center-offset-agunet-v1` and its
own provenance. A kernel checkpoint has different semantics even if its two
head channels have the same shapes.

`WordCenterOffsetDataset` wraps the existing checksum-qualified segmentation
dataset with WORD owner annotations. It replays the existing paired flip,
rotation, letterbox and validity clipping before constructing vectors. For each
surviving owner, `(dx, dy) = ((cx-x)/W, (cy-y)/H)`, where `(cx, cy)` is its visible
valid raster centroid. A concave word's center may lie outside the word; it
remains a cue, not a contour. One-pixel words have a zero vector. Lost owners
are not reconstructed. No separable distance kernel is required, so touching
words can still receive different vector targets. Line IDs are not word labels.

The vector loss ignores background and invalid pixels, averages the x/y
SmoothL1 terms per word, averages words within each image, then averages images
with valid word supervision. Empty/fully ignored images contribute no vector
objective. Completely empty batches return differentiable zero and explicitly
skip optimizer updates, including weight decay; counts retain every sample.
`WordCenterOffsetTrainingConfig` records the actual loss name and explicit beta
with the existing checkpoint and optimizer helpers. The standalone epoch helper
trains only the auxiliary model. No full training run or offset checkpoint has
been produced by this implementation.

The executable Total-Text launcher is now `scripts.train_word_center_offsets.py`.
From the repository root, the following command uses the locally verified
single-output text checkpoint and the already-qualified v4 word-owner dataset:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
.venvs/context-str-main-py311/bin/python -m scripts.train_word_center_offsets \
  --mode approximate \
  --target-root artifacts/totaltext-curved-word-boxes-v4 \
  --base-checkpoint /extra1/bmartelli/context-str-runs/totaltext-word-multitask-v1/text/best.pt \
  --base-sha256 db9db69f77f1e62b03b82d073e8a37c603ff6ff8feceafc3623644980625e964 \
  --output-root artifacts/training/totaltext-word-center-offsets-v1 \
  --device cuda:0
```

Paths are explicit command arguments and can be changed for another machine.
No dataset/checkpoint/cache path is embedded in the launcher. Its default
approximate training profile is
`configs/approximate/word_center_offsets_totaltext_train_v1.json`: 100 epochs,
FP32/Adam/batch8, qualified letterbox512/native256, seed202503, existing
TRAIN1130/validation125 split. Warm start copies the SHA-bound baseline on CPU
and releases it before the auxiliary model moves to the execution device.
No download, new environment, classic-mask retraining or OCR is involved.

Add `--action preflight --device cpu --num-workers 0` instead of the GPU device
for complete checksum/target qualification, or add `--action smoke` for a real
training-batch forward/backward. Smoke artifacts are separate from the actual
training branch. `--epochs`, `--batch-size` and `--num-workers` overrides are
explicitly recorded; reduced batch/worker choices carry hardware-deviation
metadata. No batch, precision or device is silently changed after OOM.

The training branch saves `train/best.pt`, `train/last.pt`, `train/epochs.jsonl`,
resolved configuration, dataset/split/code hashes and a warm-start receipt.
Rerunning the identical command resumes model, optimizer, RNG and epoch state.
An interruption after atomic checkpoint saving but before log append is
recovered without repeating the epoch. Incompatible provenance fails instead
of overwriting previous results. `--stop-after-epoch N` stops at an epoch
boundary with status IN_PROGRESS, retaining the configured full budget.

Select the minimum held-out word-balanced offset SmoothL1 with earliest exact
ties. Native-pixel endpoint error is an additional diagnostic, aggregated by
equal word means within each supervised image, then image means. Empty images
remain in sample counts with null objectives; expected and lost word counts are
reported. After completion, the selected checkpoint is revalidated and its
per-sample vectors, targets, owners and validity are exported. These metrics
do not establish a recognition/segmentation improvement or official paper result.
Launcher decisions are recorded in ambiguity WCO-002.

Launcher qualification: 74 focused/related tests pass, full target preflight
passes on all1255 images, and a real warm-started CPU batch2 at512x512 completes
forward/backward with a nonzero head update. Artifacts are under
`/tmp/word-center-offset-totaltext-launcher-preflight-v1/` and
`/tmp/word-center-offset-totaltext-launcher-smoke-v1/`. A profile-generation
utility initially passed a string to a Path-only checksum helper; it failed
before writing the profile and was corrected. GPU peaks are null because CUDA
is unavailable; no full100-epoch or physical-GPU result is claimed.

Example integration with an already loaded classic model and qualified dataset:

```python
from models.word_center_offsets import WordCenterOffsetAGUNet
from segmentation.word_center_offsets import (
    load_center_offset_config, WordCenterOffsetDataset,
    train_word_center_offset_epoch,
)

cfg = load_center_offset_config(
    "configs/approximate/word_center_offsets_v1.json", mode="approximate"
)
auxiliary = WordCenterOffsetAGUNet(
    skip_floor=cfg["skip_floor"], encoder_pretrained=False
)
warm_start_receipt = auxiliary.initialize_from_text_model(base_model)
auxiliary.to(device)
dataset = WordCenterOffsetDataset(qualified_word_dataset)
# Build a seeded loader and optimizer with your recorded experiment protocol.
metrics = train_word_center_offset_epoch(
    auxiliary, loader, optimizer, device, beta=cfg["loss"]["beta"]
)
```

Save the resolved profile, training configuration, reset receipt, base checkpoint
hash/revision, dataset/split hashes, seed, backend settings and runtime placement
in the experiment run contract. Use the existing `save_checkpoint`/
`load_checkpoint` helpers with `WordCenterOffsetTrainingConfig`; never identify
these weights or this loss as a segmentation/Gaussian-center checkpoint.

At inference, preserve the exact same RGB preprocessing and native coordinate
grid for both networks. `canvas` is the boolean RGB letterbox support, never
a GT validity or owner mask. Mismatched map shapes fail explicitly.

```python
import torch
from localize.word_center_offsets import word_center_look_signal

base_model.eval()
auxiliary.eval()
with torch.inference_mode():
    shape = base_model(image).sigmoid()[0, 0].cpu().numpy()
    offsets = auxiliary(image).tanh()[0].cpu().numpy()
signal = word_center_look_signal(shape, offsets, canvas, cfg["voting"])
# signal["vote_density"]: where the text pixels point
# signal["centers"] / signal["markers"]: candidate word reference points
# signal["ownership_seeds"]: pixels agreeing with each candidate center
```

Only predicted text pixels vote. Probability-weighted bilinear accumulation
preserves subpixel locations; invalid endpoints are rejected and counted,
without clamping onto image borders. Each classic foreground component votes
independently, then smoothing/NMS finds its cues. The minimum density is in
weighted votes per native pixel, not a predicted probability. Centers project
into their own text support with deterministic row-major ties. Colliding
projected markers retain the stronger cue and are counted. Dense endpoints
within `assignment_radius` of a center seed its identity; distance ties follow
deterministic marker order. Returned maps/coordinates are in the native grid.

Optional `split_text_with_center_offsets` uses those agreement seeds with the
existing minimax flood, using only base text probabilities for its cost and
support. The union of resulting instance IDs is **exactly the base binary
shape**: no pixels are added, erased or replaced by vote-map contours. Regions
without cues survive as intact fallback components. Downstream
`parseq_word_proposals` can convert IDs to crops with explicit area filtering,
padding and limits. The location signal itself produces no crop list.

All thresholds/radii are explicit approximate profile parameters. They are
exploratory assumptions, not calibrated improvements or STRICT defaults.
False peaks can split a word; NMS/weak votes can merge close or small words;
incorrect base masks cannot recover missing text. Component-local voting does
not reconnect separate fragments into one word. Dense seeds may lock incorrect
predictions, so their radius needs held-out calibration. Independent validation
of merge/split rates and downstream recognition remains unresolved.

The profile command exercises a real forward/backward/update on synthetic data
with no downloads or dependency installation:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
.venvs/context-str-main-py311/bin/python -m scripts.profile_word_center_offsets \
  --mode approximate --config configs/approximate/word_center_offsets_v1.json \
  --device cpu --batch-size 2 --height 64 --width 96 --seed 37 \
  --output /tmp/word-center-offset-cpu-smoke.json
```

For physical GPU qualification, supply `--device cuda:0` and the intended input
shape/batch. The JSON report records resolved configuration, model set, FP32,
seed, runtime, input/output shapes, loss/update and peak allocated/reserved VRAM.
Failures, including OOM, are recorded and re-raised without alternate models or
precision retries. The current sandbox has no CUDA/NVML access; RTX3070 VRAM,
real OOM behavior and coexistence with the classic network remain unqualified.
Sequential inference with saved aligned maps is an explicit HARDWARE_DEVIATION
and must be recorded when used. Smoke data and random encoder weights support
only a smoke-test result.

Qualification history: the first test collection failed because two new test
modules shared a basename; the localization module was renamed. The first
fused-word test then exposed incorrect point-only flood boundaries on a flat
mask; the implementation now includes dense vote-agreement seeds, rather than
weakening that test. Corrected target/loss/voting/checkpoint tests and relevant
existing regressions pass. There is no measured real-dataset improvement.
