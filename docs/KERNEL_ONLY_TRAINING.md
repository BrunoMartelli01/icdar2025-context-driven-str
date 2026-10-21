# Supervised Kernel-only fine-tuning

Explicitly requested by the user on16 October2026 after the completed
single-output extraction study. This is **approximate reproduction / a research
extension**, not strict paper reproduction or a claim of measured improvement.
The paper sections3.2–3.3 / Algorithm1 and existing model, kernel target,
training, checkpoint and padding components were inspected before modification.

The experiment starts from the checksum-qualified `kernel_only.pt` extracted
from the jointly trained Text+Kernel control. Its one-output architecture is
unchanged. **The new50-epoch phase uses exclusively Kernel supervision**:
masked BCE+soft Dice against K and validity. No Text, Separator or Gap loss
is calculated. M/owner annotations are used only to construct the same qualified
Kernel target after augmentation and native resizing, not as extra losses.
Historical pretraining remains joint M/K; this is not training from scratch.

Configuration: `configs/approximate/kernel_only_training_v1.json`. Reuse the
same1130/125 Total-Text TRAIN split,983 care validation words, RGB512 letterbox
and native256 maps, threshold0.5, seed202503 and paired flip/rotation. Fresh
Adam1e-5, betas0.9/0.999, epsilon1e-8, decay0.01, FP32 batch8. Encoder/BN remain
frozen; decoder and one-output head are trainable. ASSUMPTION: continuing this
warm start is an efficient direct test of the user's hypothesis, but no matched
simultaneous M+K continuation is run; any benefit could include further training.

Checkpoint selection is declared before training: highest per-image macro
Kernel IoU on valid pixels, then Kernel F1 and earliest epoch, including epoch0.
Both-empty foreground scores1. Loss averages per image over valid pixels and
then over supervised images. Invalid pixels have zero gradient; an entirely
invalid batch skips Adam/weight decay. Every epoch is run even if epoch0 remains
best. Training never selects checkpoints using OCR, padding or TEST outcomes.

Best/latest atomic checkpoints preserve optimizer and all RNG state. Resume
validates source/configuration/input/backend/device hashes and contiguous epoch
history; recovery from a checkpoint saved just before an epoch-log interruption
is tested. Actual loss metadata explicitly names Kernel-only BCE+Dice. Smoke
performs a real CUDA update, verifies head movement and resets model/optimizer/
RNG before the training phase, so the smoke is not an extra training step.

After all50 epochs: export probabilities/masks of the selected checkpoint,
unload segmentation, load qualified PARSeq on CUDA and evaluate the fixed
padding0/2/4/6/8/12 native256 grid. Count padded-box localization IoU>=0.5 and
exact-casefold OCR separately, with original unpadded localization also retained.
Use maximum one-to-one RGB bbox matching, punctuation retained, every983 care
word, and the same post-decoding wholly-invalid prediction rule as the previous
study. Empty predictions retain the GT denominator. No GT enters prediction or
crop selection. Results compare every padding with the existing extracted-K
and M+K results; they do not silently replace those results or strict defaults.

HARDWARE_DEVIATION: available RTX3090 rather than3070 target; one FP32 batch8
segmentation model at a time, TF32off and deterministic algorithms, sequential
PARSeq. Stage peaks, inputs/model set/device and OOM/error traces are recorded;
no CPU fallback, model substitute, package installation or checkpoint download.
Independent evaluation, multiple seeds and from-scratch K-only training remain
UNRESOLVED; the reused holdout is exploratory.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venvs/context-str-main-py311/bin/python \
  -u -m scripts.train_kernel_only --mode approximate --device cuda:0 \
  --reference-run /path/to/completed-kernel-extraction-study \
  --word-root /path/to/totaltext-v4 --parseq-artifacts /path/to/parseq \
  --output-root /path/to/new-external-output
```

Outputs: smoke receipt, full resolved contract, epoch0/epochs/progress,
best/last checkpoints, training manifest, validation maps, frozen crop hashes,
qualified OCR cache, `results.json`, `comparison.md`, and suite success/failure.
The `--stop-after-epoch` option is for explicit interrupted-run tests; default
execution completes the configured50 epochs and final OCR automatically.

## GPU launch qualification

Launched16 October2026 in a detached process, PID1450301, run root
`/extra1/bmartelli/context-str-runs/kernel-only-training-v1-20261016/`.
Actual RTX3090 CUDA FP32/batch8 smoke completed with a real Kernel-only update
(head Linf change1.001e-5), peak3.683GiB allocated /3.949GiB reserved, no OOM.
The epoch0 checkpoint was verified tensor-for-tensor equal to the qualified
warm start after resetting the smoke. Initial Kernel macro IoU0.635231,
F1=0.765516 across all125 validation images. The full training passed75 CUDA
optimizer steps before launch qualification; this is not training completion.
17 relevant tests pass, including exact resume/RNG and final padding/OCR-cache
boundaries. Source hashes remain unchanged. The detached process continues
independently and performs final OCR only after all50 epochs.

Progress/log and actual results remain external. Qualification is recorded in
`provenance/kernel_only_training_v1_qualification.json` and the run's
`qualification_at_launch.json`; consult `progress.json` and `training.log`
for subsequent state, not this historical launch record.

## Completed result and independent audit

The run completed all50 epochs,7,100 CUDA optimizer steps with exactly Kernel
loss active,1,130 samples/epoch and no OOM. Final six-padding PARSeq evaluation
also completed. The best checkpoint is epoch4, SHA256
`2a40805b828275b6e021c99cc89173bf5dec7087c38e1dfd73936a8f9326907b`.
Kernel macro IoU increases from63.5231% to63.9072%; epoch50 falls to62.9563%,
despite training loss decreasing from0.159387 to0.108069. A slight Kernel pixel
metric gain does not establish a word-localization or recognition gain.

| Native padding | Trained K localized | Trained K exact OCR | Extracted K localized | Extracted K exact OCR | M+K localized | M+K exact OCR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|0|413|257|400|249|648|524|
|2|623|478|622|483|605|476|
|4|536|421|529|422|408|313|
|6|379|298|386|303|257|195|
|8|288|226|298|231|150|102|
|12|147|103|147|107|64|42|

All counts use983 GT. Descriptive best grid results: trained K pad2 gives
623 localized /478 exact-OCR, OCR Hmean48.2341%,999 proposals; extracted K pad2
gives622/483,49.3109%,976 proposals; M+K pad0 gives648/524,51.8555%,1038 proposals.
The new training adds one localized word but loses five exact-OCR words relative
to the extracted model at the same padding. More proposals and slightly improved
Kernel IoU do not provide an overall gain in this exploratory experiment.

Completion is checked against the50 contiguous epoch records, CUDA device and
active-loss metadata for every epoch, checkpoint/result SHA256, unchanged source
hashes, and full125-image/983-word populations for every padding. Independent
NumPy bbox IoU and SciPy Hungarian matching reproduce all750 per-image/padding
localization and exact-OCR counts.17 relevant tests had passed before launch;
there are no outstanding failures. A stopped PID and idle GPU now indicate
normal completion, not an interrupted training.

Machine-readable audit: external run `completion_qualification.json` and
`provenance/kernel_only_training_v1_completion.json`. Full protocol, per-image
predictions and six-padding results remain in external `results.json`;
`comparison.md` is the compact report. Independent generalization and causal
isolation from continued optimization remain UNRESOLVED.
