# Auxiliary recognition supervision of shared U-Net features

**Approximate reproduction / exploratory research extension.** User authorized
implementation and launch on 21 October 2026. Decision `UOCR-001` in
`PAPER_AMBIGUITIES.md`. The local paper sections3.2–3.3, qualified Text+Kernel,
frozen-reader crop experiments and completed word-instance extensions were
inspected first. No original model, checkpoint, output, STRICT path or recognition
adapter is modified. This is not the paper's training-free context mechanism.

The new factor is whether a word-transcription loss changes the **shared U-Net
decoder**. Previous geometry_ocr trained a local box refiner with the original
U-Net frozen. Original glyph heads used binary pixel labels. Neither tested this
specific gradient intervention. Shared detection/recognition features have prior
art in [FOTS](https://arxiv.org/abs/1801.01671); this auxiliary-only adaptation is
an ASSUMPTION, not a new literature contribution or reproduction of FOTS.

Both arms start with strictly loaded research-v1/control epoch2, SHA256
`47a941e17472d26c4fff259d0a81c061e23040059c49d97cf2a89a9430045cd8`.
The original ResNet50 parameters and encoder BN stay frozen; the **same** original
decoder and Text/Kernel heads train, without a copied extra decoder.

| Component | Detached control | Connected treatment |
| --- | --- | --- |
| Text/Kernel supervision | Existing masked BCE+Dice, separator weight3 | Same |
| Training-only OCR head | Two64channel3x3 convs/ReLU, height mean, BiGRU64, classifier | Same |
| Feature crops | Owner-box axis bilinear8x96 | Same |
| Auxiliary CTC | Weight.1; reader learns on detached features | Weight.1; gradient reaches shared decoder |
| Final OCR | Qualified genuine PARSeq AR1 on original RGB crops | Same |
| Inference network | Original Text+Kernel only | Same |

Three common reader-only warmup epochs use the frozen/eval source, with a complete
state/BN digest check. Both arms copy the same resulting reader state and reset
AdamW. Each executes20full epochs: decoder LR1e-5, reader1e-3, decay.01, FP32batch4,
separate reader/decoder clipping5. No dropout, no mixed precision or auto fallback.
Seed202503 and identical per-epoch sample order are saved. The reader is discarded
at inference; actual original and training-only parameter counts are recorded.

For this first comparison **both arms disable augmentation**, including the old
horizontal flip. A mirrored word must not receive an ordinary left-to-right label
without explicit geometry correction. A future paired augmentation experiment
would be a new factor. The original image normalization, letterbox512/native256,
mask/owner/validity pairing and kernel generation are reused unchanged.

Original1130TRAIN IDs split into1030fit/100calibration using seed20261020; original
125evaluation/all983care words remain outside optimization. Source segmentation
already saw the100calibration images. The125evaluation was repeatedly explored;
no independent-generalization claim follows. No official TEST is accessed.

CTC vocabulary is sorted Unicode casefold characters from **fit transcripts only**,
blank0, punctuation retained. Owners keep their1-based original care-annotation
order; a missing native owner cannot shift later transcripts. GT chooses feature
boxes **only for training**. At most16eligible words/image rotate deterministically
by epoch, preventing permanent omission of later words in dense images. Empty,
unknown, absent-native and CTC-infeasible/over48character labels are counted;
they never remove evaluation GT. Repeated characters require extra CTC time steps.
CTC is character-normalized per word, then averaged equally over words in each
supervised image and supervised images in a batch. No zero_infinity error hiding.
Greedy reader training counts are diagnostic, separate from final PARSeq scores.

**HARDWARE_DEVIATION:** both arms explicitly perform bilinear feature sampling on
CPU, with differentiable transfers from/to CUDA; CPU CTC similarly transfers only
log probabilities. This avoids nondeterministic sampler/CTC backward without
giving control/treatment different forward kernels. CNN/GRU, segmentation and
optimizers execute on actual CUDA. Record FP32/model sets/input/batch, allocated
and reserved VRAM, actual GPU, finite reader gradients and zero/nonzero OCR-to-
feature gradients. OOM/runtime errors propagate with failure receipts; there is
no batch, precision, device or recognition-model fallback.

After both trainings, evaluate deployable epochs0/5/10/20 on100calibration images
with genuine qualified PARSeq. Choose exact localized+read count, then OCRHmean,
then earliest epoch. Epoch0/off is eligible; all20epochs execute regardless.
Freeze both choices before evaluation inference/transcripts. Frozen source and
selected detached/connected arms use identical actual batch1 inference and
watershed, original-RGB inverse-letterbox boxes, padding0/noarea/nocap. Wholly
invalid predictions are ignored after geometry; partial instances stay whole.
All care GT, misses, empty and duplicate care readings count. BboxIoU>=.5 maximum
one-to-one matching and exact casefold retains punctuation; this is not official
polygon scoring, fixed-pair recognition accuracy, strict Table5 or PARSeq-S proof.

Outputs include immutable resolved protocol/source hashes, dataset checksums,
fit/calibration/evaluation IDs, all epoch IDs/order/word exclusions/losses/gradients,
last optimizer+RNG states, deployable segmentation-only checkpoints, calibration
candidates/choices, probabilities/GT/native labels, original-RGB crop hashes and
raw qualified readings, per-image matches, merge/split/pixel counts and paired
gains/losses. Fixed-first-eight RGB overlays are saved without choosing by outcome.
Resume requires unchanged code/config/artifacts/device and a free runner lock;
interrupted epochs replay from the last completed atomic checkpoint. A changed
implementation uses a new run root. No automatic model promotion follows.

The numerical auditor independently recomputes bbox/text matching using SciPy
Hungarian assignment, validates calibration choices/populations/order, zero versus
nonzero feature gradients, checkpoint/source/RGB hashes and native mask metrics.
Unit tests and real CUDA smoke are qualification only, not evidence of efficacy.

Run with the existing main environment and explicit paths (no installations):

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venvs/context-str-main-py311/bin/python -u \
  -m scripts.run_unet_auxiliary_ocr --mode approximate \
  --config configs/approximate/unet_auxiliary_ocr_v1.json \
  --source-run /explicit/qualified/research-v1/control \
  --word-root /explicit/qualified/totaltext-v4 \
  --parseq-source /explicit/qualified/parseq-source \
  --parseq-checkpoint /explicit/qualified/parseq-bb5792a6.pt \
  --parseq-archive /explicit/qualified/parseq-source.tar.gz \
  --output-root /explicit/new/external/run --device cuda:0 --action run
```

`--action preflight` qualifies all1255targets and fit label alignment without
training. `--action smoke` uses8fit/2calibration/2evaluation, one warmup/main epoch,
and real PARSeq, explicitly labeled **smoke-test result**; it never claims983word
performance. Do not share its output with a full run. A durable full run should
execute a hashed external source snapshot, preserve runner exit status and run
the independent auditor only after successful training/evaluation.
