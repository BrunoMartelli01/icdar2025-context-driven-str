# Local word instance extensions

**Approximate reproduction / exploratory research extension.** These are three
independent experiments requested after the completed word-evidence suite. They
do not change the original U-Net, existing experiments, recognition adapter,
context routing, DeepSolo, or STRICT configuration. No improvement is claimed
from implementation or smoke tests.

## Novelty check before implementation

The machine-readable [audit](WORD_INSTANCE_NOVELTY_AUDIT.json) records inspected
sources, their SHA256, prior results, search scope and decisions. It distinguishes
local experimental novelty from literature novelty; no global novelty claim is
made. `research groups/find`, the audit index, Python implementations, configs,
tests, related documentation and the listed external reports were inspected.

| Proposal | Closest existing experiment | Actual new factor |
| --- | --- | --- |
| Kernel-conditioned single-word mask | `crop_word`/`crop_links`, seeded extraction, glyph association | The selected **predicted kernel enters the network**, and training asks for that word only. Neighboring words are valid negative mask labels. Same-capacity no-prompt control gets identical targets/windows. |
| Maximin connectivity loss | PixelLink-inspired local BCE, embedding pull/push, Gap BCE/Dice | Train on bottleneck edges determining **pixel-pair connectivity along paths**, in addition to matched local edge BCE. Same four-edge CNN, inputs and graph decoder in the control. |
| OCR-guided continuous crop | Bbox proxy losses, forced PARSeq candidate scores, OCR-based checkpoint selection | Backpropagate frozen PARSeq causal transcript CE **into continuous box coordinates**. Matched geometry control uses the same network and geometry objective without the OCR term. |

Relevant negative results are retained: crop_word524/control521 has more proposals
and worse OCR Hmean; global links490 versus520; word-evidence association520 and
completeness502 versus520. The later OCR-selected Gap result526/control524 is
also real but selects fixed candidates; it does not train boxes through OCR.
Those controls/protocols are different and are not pooled into one leaderboard.
Centerline/clDice discussion is not evidence of an executed maximin objective.

## Fixed source and controls

The explicit `--evidence-root` is the **results** directory of the completed
corrected word-evidence run. Final results, contract and both map receipts are
pinned in `configs/approximate/word_instance_extensions_v1.json`. Every1130TRAIN
and125evaluation map and original RGB is checked before inference; maps/RGB are
checked again when loaded. Original source postprocessing receipt is also pinned.
The v4 dataset manifest and its1-based care-owner order supply training targets.
No new glyph IDs or character labels are invented. The original masks are read-only.

The first-pass Text+Kernel checkpoint is not loaded or modified: inference uses
its already qualified canonical maps. Each new local model is a **fresh** small
RGB/Text/prompt CNN: three3x3 layers,24channels, ReLU, no BatchNorm/dropout,
zero-initialized output. This is not a pretrained second U-Net or full backbone
fine-tuning. Within each pair, seeds, capacity, sampled windows and optimizer
schedule match. A no-prompt input is a zero channel, not a smaller architecture.

Predicted native boxes choose context windows before any training annotation is
read. Original RGB is sampled from the corresponding square-letterbox extent,
then resized to96x96. Text interpolation is bilinear; prompt/canvas/owners/
validity use nearest. A prompt is that instance's actual predicted Kernel pixels,
not its full Text support or a GT kernel. Padded RGB uses explicit black fill.

## Three isolated objectives

`prompt_instance` and `prompt_control` use BCE+softDice on the selected word.
The qualified full word-owner map selects the majority care owner **only during
training**. Tied owners, absent care owner, ignore-only prompts and selected words
under95% visible in the predicted context have zero mask/geometry supervision.
They are counted, not relabeled as negative examples or removed from evaluation.

`affinity_control` and `affinity_maximin` predict four undirected8-neighbor edge
logits. Both use class-balanced local BCE, excluding background/background edges.
The new term is a local [MALIS-inspired maximin objective](https://proceedings.neurips.cc/paper_files/paper/2009/hash/68d30a9594728bc39aa24be94b319d21-Abstract.html):
Kruskal's maximum spanning forest identifies bottleneck edges; at most16
deterministic row-major-spaced anchors per care word carry total mass1 per word.
Positive/negative pair contributions receive separate mean BCE and weight.25.
Valid background may carry paths but has no endpoint labels; ignored pixels carry
neither paths nor labels. Detached graph selection retains gradients through the
selected current logits. Stable channel/y/x ordering resolves equal-logit ties.
This is **not** original all-pairs squared Rand loss, a full-image topology loss,
or a constrained two-pass MALIS implementation. With a strong wrong merge edge,
negative-pair paths can penalize internal-word bottlenecks too; tests preserve
this limitation rather than assume every gradient acts on a GT boundary.

`geometry_control` and `geometry_ocr` pool the local CNN features and source box
coordinates and predict four side deltas. `tanh` bounds each correction to25%
of its source width/height; corrections clip to the context. Both use normalized
SmoothL1 against the selected word's native owner box. The new arm adds.1 times
qualified PARSeq causal teacher-forcing CE over exact supported GT text plusEOS,
with equal mean per word. PARSeq stays eval and all of its parameters are frozen;
autograd to image sampling and box coordinates stays enabled. Unsupported/long/
empty OCR labels are explicitly counted without changing evaluation denominators.
The inference adapter itself is not changed. End-to-end supervision has prior
art such as [FOTS](https://openaccess.thecvf.com/content_cvpr_2018/html/Liu_FOTS_Fast_Oriented_CVPR_2018_paper.html);
this particular frozen-PARSeq box adaptation is an assumption.

Continuous bilinear `grid_sample` runs **on CPU**, retaining differentiable device
transfers, to avoid nondeterministic CUDA sampler backward. Training uses the
resampled96square context and32x128 bilinear crops; scored OCR uses the genuine
adapter's original-RGB PIL bicubic transform. The domain/transform mismatch is
explicit and unvalidated. A zero geometry residual replays the original box
exactly, avoiding floating-point round-out at initialization.

## Training, calibration and evaluation

The fixed original1130TRAIN IDs split deterministically into1030fit/100head-
calibration IDs, seed20261020. All125evaluation IDs remain disjoint. The original
segmentation already saw the100calibration images; this is held out only from
new local-model optimization. The125-image holdout is repeatedly explored.

Full runs use20epochs, one deterministically sampled predicted window per fit
image per epoch, batch1 FP32, AdamW1e-4/decay1e-4, no augmentation. All-invalid
objectives skip optimizer updates including decay; no-proposal/unsupervised images
remain in population audits. Every arm must have actual supervised updates.
Epoch0/5/10/20 and the off/baseline candidate are compared on100calibration images
using actual exact read+localized word count, then OCR Hmean, then geometry Hmean.
Ties retain off/earlier candidates. No evaluation label selects a model or a
threshold. Training resumes from the last completed epoch with strict model and
optimizer state, fixed per-epoch order/window sampling and a hash-bound contract.
Changing config, code, device or inputs requires a new run root.

At inference masks/graphs choose the component overlapping the supplied predicted
kernel most. Only frozen predicted Text foreground enters the affinity graph.
Each original instance emits at most one refined box. Missing predicted kernels
or empty anchored components retain their original boxes with explicit counters.
There are no new orphan words, new seeds, box deduplication, area filter, padding
or cap. This limits recovery outside source windows or where no kernel exists.
The mask branch can expand beyond the original Text support; the affinity branch
specifically studies repartitioning within that support. Other source instances
remain independent; duplicate kernels and overlapping boxes can persist.

All geometries are frozen and written before transcript scoring. Ignore flags
come from the original source instance identity after geometry inference; wholly
invalid source instances alone are excluded, partially ignored ones stay whole.
Maximum one-to-one original RGB exclusive bbox IoU>=.5 gives localization; adding
exact Unicode casefold text gives OCR matches, punctuation retained. All983care
words remain in the full evaluation. Empty/duplicate care readings remain in
prediction counts. Independent geometry and OCR matchings are not a disjoint
causal error taxonomy. Every arm is compared against a re-evaluated common source
baseline under the chosen device/batch1, not an assumed historical520 count.

Outputs retain resolved config, software/hardware and all source hashes, splits,
per-epoch per-image supervision/gradient audits, last/best model states, optimizer,
calibration candidates/choices, predicted local logits/supports/windows, original
boxes, RGB crop hashes, raw PARSeq tokens/provenance, per-image and aggregate
counts and gained/lost GT identities. Failures/OOM propagate with explicit stage;
there is no model, precision, batch-size or CPU inference fallback.

## Running

Use the existing main environment on PAPER; no installation is needed. Every
scientific root and PARSeq artifact is explicit. Example from the repository:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venvs/context-str-main-py311/bin/python -u -m scripts.run_word_instance_extensions \
  --mode approximate --config configs/approximate/word_instance_extensions_v1.json \
  --evidence-root /explicit/completed-word-evidence-run/results \
  --word-root /explicit/totaltext-v4 \
  --parseq-source /explicit/qualified-parseq-source \
  --parseq-checkpoint /explicit/parseq-bb5792a6.pt \
  --parseq-archive /explicit/qualified-parseq-archive.tar.gz \
  --output-root /explicit/new/external/run --device cuda:0 --action all
```

`--action preflight` verifies the scientific inputs without training/inference.
`--action smoke --smoke-images 8` performs one real training epoch over8fit images,
one calibration image and one fit-image evaluation in every arm. Smoke outputs
are labeled **Smoke-test result** and are not983-word performance estimates.

For a durable run, copy the Python source/config packages into an external
snapshot and invoke the same module/config there with the existing venv Python;
the run contract records the executable files. Do not edit the snapshot while
training. Preserve failed roots before retrying a changed implementation.

## Qualification status

Implementation and focused tests are qualified separately from experiments.
The first gradient fixture put a wrong inter-word edge above the internal edges
and incorrectly expected all internal gradients to be attractive. The independent
widest-path calculation confirmed the implementation: maximin paths can penalize
internal bottlenecks. The boundary-gradient fixture was corrected, and a separate
test now explicitly checks that limitation. No error or loss was suppressed.
Real smoke and regression receipts are recorded in the external qualification
directory; full20epoch efficacy remains unmeasured until an actual full run.
The first CUDA smoke passed1255-image input verification and genuine PARSeq
inference, then failed before training because NumPy integer annotation boxes
could not be serialized. The GT-reader boundary now converts coordinates to
Python integers, with a regression test. Failed output and source are preserved
under `cuda-smoke-v1`; a changed implementation uses a new qualification root.

Final qualification is in
`/extra1/bmartelli/context-str-artifacts/word-instance-extensions-20261020/`:

- **134 focused and related regression tests pass**, including the old crop,
  evidence, PARSeq and research-navigation suites.
- Final CUDA smoke executes from a739-file hashed executable source snapshot,
  `qualified_source_v3/`, using actual RTX3090 FP32. Each of sixarms performs8
  real updates with finite/nonzero gradients; all8OCR-arm updates include genuine
  frozen-PARSeq CE. Paired trained states differ for all three interventions.
- Source1255image/map verification passes; complete PARSeq state remains exact.
  Own-process peak allocated191413760bytes/reserved228589568bytes, batch1 local
  input1x5x96x96 plus frozen PARSeq. No OOM occurred. Injected-OOM unit coverage
  checks explicit failure propagation; no physical memory-exhaustion test or
  RTX3070/full-pipeline qualification is claimed.
- Completed-run resume preserves every selected choice, aggregate count and
  checkpoint SHA, performs no extra updates, and retains the original peak memory.
  This verifies idempotent completed resume, not an interrupted-epoch CUDA replay.
- `independent_qualification_audit.json` independently verifies20image/candidate
  cases and219proposal cases with SciPy Hungarian matching, all crop/qualified
  recognizer hashes, the split, actual update/gradient counts and resume receipts.
  Its accompanying `audit_qualification.py` is reproducible with explicit
  `--run` and `--output` paths. Frozen-reader equality is measured by the original
  runner; the independent audit verifies that receipt and artifact identities.

All final smoke selections retain the off/baseline candidate. With one epoch on
8fit images, this is **not an efficacy comparison**. Full20epoch training and
the983-word evaluation are implemented; their separate full run is tracked below.
No new model has been promoted. The failed first attempt and earlier successful
smoke are retained.

## Authorized full run — 21 October 2026

The user explicitly requested implementing and starting the experiments. The
complete six-arm suite was launched on PAPER under
`/extra1/bmartelli/context-str-runs/word-instance-extensions-20261021-v1/`.
It executes the739-file qualified source snapshot with the unchanged approximate
config:20epochs per arm,1030fit/100calibration/125evaluation images, all983care
evaluation words, seed20261020. Each new method has its matched trained control;
the original Text+Kernel reference is evaluated separately on the same runtime.
All six arms run sequentially on the verified RTX3090/24GiB, FP32/batch1.

Immediately before launch26focused tests pass, source snapshot hashes match the
qualified copy, GPU is idle and355GiB disk space is free. No dependencies or
weights are installed/downloaded. The durable supervisor runs independently of
the chat, preserves the runner exit code and starts a separate read-only audit
only after successful full training/evaluation. The independent auditor recomputes
all3375planned calibration/evaluation image-candidate cases with SciPy Hungarian
matching, validates split/denominators/selection/checkpoint/source hashes and
produces paired-control comparisons. Its matching self-test passes before launch.
An audit failure remains a failure, with raw training results retained.

Operational files at the run root:

- `launch_receipt.json`, `source_snapshot.json`, `command.json`: exact command,
  PID, source hashes, hardware and environment; no guessed scientific paths.
- `pipeline_status.json`: durable supervisor state and runner PID.
- `suite.log`, `results/progress.json`, `results/<arm>/training.json`: live
  experiment stage and completed epochs. Initial reference preparation may
  precede the first epoch log line.
- `results/results.json`: final machine-readable predictions and metrics.
- `independent_audit.json`, `audit.log`, `comparison.md`: automatic independent
  checks and readable comparison, created after successful completion.

At launch the suite is **RUNNING**, not a completed performance result. No gain
or generalization claim follows from launch. An interrupted full runner can be
resumed with its exact `command.json`, original snapshot and existing output
only after confirming its PID/lock is inactive. Preserve failed roots; do not
start a duplicate suite on the same output or edit the executable snapshot.
