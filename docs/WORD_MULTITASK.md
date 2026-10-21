# Word-kernel and Gaussian-center Attention U-Net experiment

This is an **experimental extension / approximate reproduction**, never a STRICT
paper result. The existing models, qualified targets, checkpoints, recognition
pipeline and prior results are preserved. Scientific decisions are recorded in
`configs/approximate/word_multitask_v1.json` and ambiguity WM-001.

All three 100-epoch v1 runs and their validation exports have now completed on
RTX3090. Text + Kernel gives the best measured word F1 (.6213 versus the matched
Text control's .5896); the center variant gives .5841. See
[the completed results and offline visual archive](WORD_MULTITASK_RESULTS_V1.md)
and `provenance/word_multitask_v1_completion_audit.json`. Earlier pending statements
below describe implementation-time qualification; the completed result record
supersedes them for this run. Independent TEST, PARSeq benefit and RTX3070
qualification remain unresolved.

## Audit before implementation (2026-10-08)

The local paper `papers/2503.15639v2.pdf`, sections 3.2/3.3, Fig.3 and
Implementation Details specify a pretrained ResNet50 attention U-Net, binary
foreground, BCE, and connected-component crops. They do not specify kernel or
center supervision, Dice, watershed, or word-instance evaluation.

The repository is substantially beyond its initial scaffold audit. Existing
reusable components include frozen-encoder attention models, checksum-aware
manifests, v4 polygon-derived owner maps, ignore masks, paired augmentation and
letterboxing, Adam training, optimizer/RNG checkpoints, native word geometry,
PARSeq crop adapters and visualization helpers. In particular:

- `models/totaltext_word_kernel.py` already adds a 65-parameter kernel channel.
- `segmentation/totaltext_word_kernel.py` already generates connected distance
  cores, replays paired owner transforms, expands nuclei, preserves unseeded
  text and extracts per-instance boxes.
- `localize/crop_recovery.py::geodesic_partition` already implements a deterministic
  marker-controlled minimax watershed. It is reused here.
- Prior separator, pixel-link, word-balance, zoom and deep-supervision experiments
  have completed. No rerun of these experiments is requested by this implementation.

Completed historical native-validation results (125 held-out TRAIN images,
983 expected care words):

| Run | Epoch selected | Precision | Recall | F1 | Merge rate | Split rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `totaltext-letterbox-word-f1-separator-v1/word_f1_control` | 68 | .5300 | .5117 | .5207 | 199/983=.2024 | 77/983=.0783 |
| `totaltext-word-kernel-100ep-v1/word_kernel_100ep` | 47 | .5121 | .5158 | .5139 | 230/983=.2340 | 79/983=.0804 |

The old control uses text threshold .50; the kernel run uses .65. Their losses
also differ (separator-weighted text BCE versus text BCE plus balanced kernel
BCE). These are historical references, **not a matched causal comparison**.
The kernel run's original-coordinate PARSeq diagnostic matched 501/983 words at
bbox IoU >=.50, with 394 exact casefold transcriptions; these are different
metrics from native validation and are not paper Table5 accuracy.

Report provenance, relative to an explicitly supplied existing run root:

- Control manifest: `totaltext-letterbox-word-f1-separator-v1/word_f1_control/run_manifest.json`,
  SHA256 `919875a19c31dc5fc704cc706b7bd081dd0dd05125e2760f7e4a88c733cf38b4`.
- Kernel manifest: `totaltext-word-kernel-100ep-v1/word_kernel_100ep/run_manifest.json`,
  SHA256 `2d5384f3007d22162a39286f1a6aee7138ce15ad291f93fe786ec6459d832157`.
- Kernel PARSeq report: `totaltext-word-kernel-100ep-v1/validation-pad0-parseq-v1/results.json`,
  SHA256 `4e1e7ce71a4c0618d1b9da49fb1f665f9f68e77bce57f08d98d1ef98c68655d5`.

## Minimal new implementation

The only new model is `WordMultitaskAGUNet`, extending the existing two-output
model to three independent 1x1 outputs from the same 64-channel decoder feature.
Output ordering is **text, kernel, center**, all logits. The encoder, bottleneck,
four decoder stages, frozen ResNet50/BN and skip floor are unchanged. Under a
common seed, all shared tensors and existing output channels initialize identically.

| Config variant | Channels | Parameters | Added versus existing single-output model |
| --- | ---: | ---: | ---: |
| `text` | 1 | 58,155,081 | 0 |
| `text_kernel` | 2 | 58,155,146 | 65 |
| `text_kernel_center` | 3 | 58,155,211 | 130 |

This preserves the existing backbone's size; it does not establish that a
58-million-parameter backbone is small in absolute terms.

### Targets (`ASSUMPTION`)

The matched runner retains the qualified **v4 polygon-derived filled targets**,
validity and separator maps. These are already smoothed/rasterized annotations,
not newly substituted raw polygons. Owner IDs replay exactly the same spatial
augmentation and letterbox, nearest resize, valid-text clipping as the existing
kernel experiment. Native words lost before auxiliary construction are counted,
not invented. Connected kernels reuse half-maximum interior distance and
largest-connected-core selection, including its fail-closed touching-seed check.

For each surviving owner, find the raster centroid and project it to the nearest
kernel pixel (row-major ties). This keeps the center inside curved/concave words.
The Gaussian covariance is the full owner's population covariance multiplied by
`sigma_scale^2`, with eigenvalues floored at `min_sigma^2`. Defaults are scale .5
and minimum sigma 1 native pixel. The peak is exactly 1; the map is clipped to
its own valid owner. One-pixel words retain a peak. No OCR, transcription or
TEST target guides construction.

`targets_from_polygons` is an additional entry point for word-granularity
`SceneTextInstance` annotations. It reuses `make_word_regions` and its existing
overlap ownership, separators and ignore handling; it rejects lost care polygons
and line annotations. Empty/all-ignored images are supported. This helper does
not silently change the matched runner's v4 target source. Other dataset
experiments require independently qualified word-owner artifacts and a new
protocol; CTW1500 line labels are not word ground truth.

### Loss (`ASSUMPTION`)

`lambda_text * (BCE + Dice) + lambda_kernel * (BCE + Dice) + lambda_center * MSE`.
Only tasks selected by the variant participate. Task weights, BCE/Dice weights,
Dice smoothing and center loss are configurable. Text BCE preserves the prior
separator weight 3; kernel BCE is ordinary masked BCE. Soft Dice is per image:
`1 - (2*sum(p*y)+smooth)/(sum(p)+sum(y)+smooth)` over valid pixels. BCE uses
the per-image weighted valid mean; non-ignored image terms are averaged.

Center regression uses sigmoid outputs and soft Gaussian values. The default
`balanced_mse` averages positive Gaussian-support and zero-target background
means equally within each image, then averages images. Plain valid-pixel `mse`
is selectable. Ignore pixels have exactly zero gradient in every task. Completely
invalid images contribute differentiable zero; valid empty text remains a
negative training example. Checkpoints explicitly name the multi-task loss.

### Inference (`ASSUMPTION`)

All methods operate exclusively on predicted probabilities and the RGB canvas:

- `connected_components`: foreground-only control, auxiliary maps unused.
- `kernel_bfs`: reuse simultaneous 8-neighbor expansion for two maps. For three
  maps, construct labeled markers and use the existing flood with constant cost
  (shortest-path expansion, label-ID ties).
- `watershed`: the existing minimax flood on a confidence landscape weighted
  text/kernel/center .50/.25/.25, renormalized over the available channels.

Center maxima use plateau collapse and deterministic Chebyshev NMS (radius 3,
threshold .50). Multiple peaks inside a connected kernel split its markers with
the existing watershed. Peaks outside kernels create point rescue markers when
farther than the NMS radius from existing markers. Kernels without peaks remain
markers. Foreground components with no marker survive as single fallback
instances. There are no background shortcuts or watershed boundary deletions.
Text threshold .65 reuses earlier TRAIN-only evidence; auxiliary thresholds .50
and all other values are declared assumptions, not new threshold searches.

Each label ID yields its own box, even when neighboring labels touch. Default
pad0/no area filter/no top10 are existing practical choices and differ from
paper Algorithm1. `parseq_word_proposals` additionally supports explicit padding,
component-area filtering and limits; boxes use exclusive `xyxy` original RGB
coordinates, compatible with `rgb[y1:y2, x1:x2]` and the existing PARSeq adapter.
This extension does not replace PARSeq or DeepSolo in recognition experiments.

### Controlled experiment and metrics (`ASSUMPTION`)

All three new branches share v4 TRAIN1130/validation125, seed202503,
flip/rotation replay, RGB512/native256 letterboxing, pretrained V2 frozen encoder,
FP32, batch8, fresh Adam LR1e-4/decay.01 and exactly100 epochs. The fresh `text`
BCE+Dice control is necessary to isolate auxiliary heads from the requested loss
change. Historical BCE runs are reused separately, never mislabeled as this
control. Final checkpoint selection is word F1, fewer merged care words, recall,
macro foreground IoU, earliest exact tie. No TEST or recognition score selects it.

Word precision/recall/F1 use dataset-global maximum-cardinality one-to-one native
box IoU>=.50 matching, reusing the existing implementation. Diagnostics at .25/.75
remain available. This is **not official polygon-IoU detection evaluation**.
Merge rate is care words touched by at least one multi-word predicted instance
divided by all expected care words. Split rate is care words touched by at least
two instances divided by all expected care words; touching means any native
shared pixel. A separate merged-proposal rate uses the proposal denominator.
Lost GT words remain in denominators. Instances entirely inside invalid regions
are ignored for geometry; partially invalid instances keep their complete boxes.
This explicit policy differs from some historical diagnostics.

Foreground IoU/F1 reuse per-image macro and global micro metrics with validity,
and perfect-if-both-empty pixel policy. Zero word denominators yield zero word
ratios. Every validation image is retained. Repeated earlier use of this holdout
makes conclusions exploratory. Center false peaks can cause splits; NMS can
suppress very close words. The completed v1 comparison is reported in
`docs/WORD_MULTITASK_RESULTS_V1.md`; independent separation generalization and
recognition benefit remain unresolved.

## Run and reuse

No installation, download or new environment is required. Provide the local
ImageNet V2 checkpoint and its full SHA256 explicitly. The qualified local
checkpoint has SHA256
`11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`.
All weight loading is strict. Dataset/config/source/encoder hashes, split,
optimizer, resolved parameters and runtime paths are recorded with results.
Approximate config lookup cannot fall back to demo or strict namespaces.

From the repository, set paths for your machine and a new output directory:

```bash
PY=.venvs/context-str-main-py311/bin/python
TARGET=artifacts/totaltext-curved-word-boxes-v4
OUT=/path/to/new-word-multitask-run
ENCODER=/path/to/resnet50-11ad3fa6.pth
ENCODER_SHA=11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca
OLD_RUNS=/path/to/existing/context-str-runs

bash scripts/run_word_multitask_host.sh "$TARGET" "$OUT" "$ENCODER" "$ENCODER_SHA" \
  "$OLD_RUNS/totaltext-letterbox-word-f1-separator-v1/word_f1_control/run_manifest.json" \
  "$OLD_RUNS/totaltext-word-kernel-100ep-v1/word_kernel_100ep/run_manifest.json"
```

The launcher runs CPU target preflight, real-batch CUDA smoke, sequential
training of the three **new** branches, best-checkpoint validation/visual
exports, and comparison. Historical references are read only.

Individual actions and variants can be selected:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PY" -m scripts.run_word_multitask --mode approximate --action preflight --device cpu \
  --target-root "$TARGET" --output-root "$OUT"
"$PY" -m scripts.run_word_multitask --mode approximate --action train --device cuda:0 \
  --variant text_kernel_center --target-root "$TARGET" --output-root "$OUT" \
  --encoder-checkpoint "$ENCODER" --encoder-sha256 "$ENCODER_SHA"
"$PY" -m scripts.run_word_multitask --mode approximate --action evaluate --device cuda:0 \
  --variant all --target-root "$TARGET" --output-root "$OUT" \
  --encoder-checkpoint "$ENCODER" --encoder-sha256 "$ENCODER_SHA"
"$PY" -m scripts.compare_word_multitask --run-root "$OUT" --output "$OUT/comparison.json"
```

Use `--action smoke` for one real training batch per chosen variant. In restricted
sandboxes use **explicit** `--num-workers 0` consistently for that run; the
override is recorded as `HARDWARE_DEVIATION`. There is no automatic fallback.
The default two-worker DataLoader has a 60-second timeout instead of hanging
indefinitely on worker IPC failures. Single-GPU sequential execution is recorded
as `HARDWARE_DEVIATION`; smoke/train/evaluate record peak allocated/reserved VRAM,
precision, model set, shape and batch. OOM propagates and is saved in failure.json.
GPU qualification is not established by CPU success.

Same-contract training resumes optimizer/model/RNG and a recoverable interrupted
epoch log, or reuses a verified completed branch. Cached completed validation and
comparison are checksum-verified and reused. Changed code/config/artifacts require
a new output root. For target or threshold changes, copy the config to another
file under `configs/approximate/` and use `--config`; keep the whole matched
comparison consistent. The implemented matched geometry protocol requires
pad0/no filters. Strict mode fails before experiment execution.

### Repair of the initial launcher contract encoding

The initial runner compared an in-memory Adam beta tuple `(0.9, 0.999)` with
the JSON array `[0.9, 0.999]` read from disk. This rejected the training stage
after a successful smoke despite identical scientific parameters. Contracts
now use their canonical JSON representation before persistence/comparison.
PyTorch checkpoint training metadata retains the existing dataclass contract.

The exact initial runner SHA256 is
`298b9023dfb7fe2fb6d9e8cf4234da515fffe953c7595341c63197f453bff38c`.
Only its completed **smoke-only** branch directories can be repaired in place.
Every config, split, dataset, encoder and scientific-module hash must match.
Any training checkpoint/log/result, unexpected file, different runner revision
or scientific change prevents this repair. The old contract is preserved
byte-for-byte as `smoke_run_contract.json`; `smoke.json` stays untouched and
retains its original code provenance. `contract_transition.json` records both
versions and artifact checksums, then `run_contract.json` describes the repaired
runner. Archival/receipt commits can be resumed if interrupted before the
active contract commit. This is an explicit `ASSUMPTION` about compatibility of
the launcher-only fix, not a bypass of scientific-source/checkpoint validation.

Rerun the same host command and output root. The preflight and three verified
smoke results are reused, then fresh training starts. The repair never transfers
one-step smoke model weights into a training run. Regression tests cover JSON
round trips, all three real launcher stages using a tiny CPU fixture, repeated
stage reuse, rejected changes/training artifacts, tampered receipts and
interrupted contract commit.

The user's preserved smoke artifacts under
`/extra1/bmartelli/context-str-runs/totaltext-word-multitask-v1/` report one
successful FP32 CUDA training batch per variant, shape8x3x512x512:

| Variant | Peak allocated bytes | Peak reserved bytes | OOM |
| --- | ---: | ---: | --- |
| text | 3,967,820,800 | 5,349,834,752 | false |
| text_kernel | 3,969,918,464 | 5,349,834,752 | false |
| text_kernel_center | 3,972,017,152 | 5,349,834,752 | false |

The recorded physical GPU is **RTX3090**, a `HARDWARE_DEVIATION` from the intended
RTX3070 development target and paper environment. These measurements qualify
that single-batch execution only; they do not establish RTX3070 execution,
full-training peak memory or improved validation performance. The tuple/list
failure occurred before any training checkpoint or epoch log was produced.

Outputs per branch include `run_contract.json`, `epochs.jsonl`, `best.pt`,
`last.pt`, `run_manifest.json`, and `validation.json`. Final evaluation compares
CC/BFS/watershed on identical best-checkpoint maps without retraining or selecting
another checkpoint. `validation_exports/<sample>/` contains lossless probability,
target, validity, owner and instance arrays (`maps.npz`), three-map visualizations
and original-coordinate `proposals.json`. No duplicate RGB crops are saved.
`comparison.json` separates matched branches, deltas and historical references.

Tests:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" -m pytest tests/segmentation/test_word_multitask.py -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" -m pytest tests -q
```

## Verification record (2026-10-08)

- All 1,255 TRAIN images passed center-target qualification:9,284 surviving
  native words, each with a peak1; one pre-existing native word loss. The final
  local audit is `/tmp/context-str-word-multitask-preflight-final/target_qualification.json`.
- Complete pretrained three-head CPU training smoke passed, batch8/input512,
  native256/FP32/one Adam step, explicit workers0. Task losses were text1.720867,
  kernel1.672405, center.136271, total3.529543. This verifies execution, not
  training improvement. Report:
  `/tmp/context-str-word-multitask-smoke-final/text_kernel_center/smoke.json`.
  The final smoke also records actual NumPy/OpenCV/Pillow/SciPy/torchvision
  versions alongside PyTorch and backend settings.
- 22 new focused tests passed; the final related regression selection passed
  103 tests, including resume after checkpoint commit before epoch-log write
  and propagation/metadata of a simulated OOM. The initial target-transform fixture called an obsolete API;
  it was corrected to the existing paired-supervision API. A transient metric
  metadata spelling mismatch was corrected before the successful smoke.
- Full `pytest tests -q`: **1,325 passed, 3 failed**. All three failures reproduce
  independently without collecting the new tests, and their implementations and
  tests are byte-identical to the pre-task workspace. They are intentionally
  reported, not suppressed or modified:
  - `tests/data/test_segmentation_transforms.py::test_invalid_probabilities_sizes_and_ranges_fail`
    expects the old `Paper-specified` error phrase; current code says
    `Transform sizes must match the explicit classification`.
  - `tests/deepsolo/test_candidates_and_reduction.py::test_host_source_revision_verification_rejects_wrong_revision`
    expects `source revision mismatch`; current code says
    `Declared DeepSolo source revision does not match Git HEAD`.
  - `tests/test_main_environment.py::test_main_cuda_smoke_validation_passes`
    requires an environment named `context-str-main` and an RTX3070. This session
    uses `context-str-main-py311` and has no CUDA access.
- The first real-batch attempt with workers2 was interrupted after PyTorch
  worker IPC raised `PermissionError` for sandbox sockets. The explicit recorded
  workers0 smoke then passed. Initial broad root discovery was interrupted;
  explicit discovery under `tests/` completed with the counts above.
- GPU forward/backward, VRAM peaks, real OOM behavior, new100-epoch training,
  held-out performance gains, independent TEST and recognition are unqualified.
  No dependencies, historical checkpoints/results or strict defaults were changed.
- Python compilation, host-launcher Bash syntax and `git diff --check` passed.
  The complete task diff was inspected; existing model/data/training/recognition
  implementations are unchanged. Unrelated concurrent YOLO work was preserved.
