# New U-Net trained on Kernel GT

User instruction: delete the previous fine-tuned Kernel run, create a new
U-Net and train it using Kernel ground truth on GPU. The completed fine-tuning
run and its checkpoints were deleted at the user's explicit request; its small
historical audit remains in provenance. Dataset, original extracted-K/M+K
references and code remain intact. Deletion scope and authorization are in
`provenance/kernel_only_training_v1_deletion.json`.

This is **approximate reproduction / a research extension**. Existing model,
target/training/padding implementations and paper sections3.2–3.3 were inspected
before modification. The same attention U-Net/ResNet50 family is retained.
The new architecture ID is `experimental-fresh-kernel-only-agunet-v1`.

## Initialization and training

Construct a new one-output U-Net with a **random decoder, bottleneck and head**.
Initialize only its encoder from the existing local checksum-qualified
ImageNet ResNet50 V2 file, SHA256
`11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`.
Full encoder state loading is strict, including keys/shapes/dtypes. No previous
Text, Kernel or Gap segmentation weights are loaded. This is fresh segmentation
training with ImageNet encoder pretraining, not an all-weights-random encoder.
The encoder/BN are frozen; decoder and head are trainable.

Save the newly generated initial state and a hash of all non-encoder tensors.
The real CUDA smoke performs a Kernel-only update, then restores that **own
fresh initial state** and resets optimizer/RNG. Resume uses only this new run's
own best/last checkpoints; it cannot switch to previous segmentation weights.

Configuration: `configs/approximate/kernel_only_fresh_v1.json`.
100 epochs; Adam1e-4, betas0.9/0.999, epsilon1e-8, weight decay0.01; seed202503;
FP32 batch8, workers2, TF32off, deterministic algorithms. Loss is only masked
BCE+Dice(K), equal weights and smooth1. M/owners are used to construct the GT
Kernel, never as additional Text/Separator/Gap losses. All-invalid batches skip
Adam and weight decay. Kernel construction, validity, augmentations and
letterbox remain qualified: RGB512, native256, frozen Total-Text TRAIN1130 /
validation125 with983 care words. No TEST data is used.

Select checkpoints by per-image macro Kernel IoU, then F1 and earliest epoch,
including epoch0, at fixed K>=0.5. Both-empty scores1 on valid pixels. Run all100
epochs even if the best is early. No OCR/padding-based training selection.
Best/last optimizer and RNG resume and log recovery use the tested existing
Kernel trainer, with a separate configuration and architecture identity.

After100 epochs, export the best predicted Kernel masks, unload segmentation
and evaluate the existing six native256 paddings0/2/4/6/8/12 using qualified
PARSeq AR/refine1 on GPU. Preserve original and padded-box localization, exact
casefold OCR with punctuation, maximum one-to-one RGB bbox IoU>=0.5, every983
GT, per-image outputs and wholly-invalid ignore semantics from the previous
study. The original extracted-K/M+K run is read **only for metric comparison
and qualified OCR cache**, never for initialization. Final reports explicitly
name fresh training rather than fine-tuning.

HARDWARE_DEVIATION: RTX3090 rather than3070, one FP32 segmentation model then
PARSeq, measured allocator peaks/input shapes/device and explicit OOM failures;
no CPU fallback, downloads or package changes. Initialization/budget/selection
are ASSUMPTION; qualified GT construction is REPOSITORY_SPECIFIED. Independent
validation, seed replication and strict paper reproduction remain UNRESOLVED.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venvs/context-str-main-py311/bin/python \
  -u -m scripts.train_kernel_only_fresh --mode approximate --device cuda:0 \
  --encoder-checkpoint /path/to/resnet50-11ad3fa6.pth \
  --reference-run /path/to/original-kernel-extraction-study \
  --word-root /path/to/totaltext-v4 --parseq-artifacts /path/to/parseq \
  --output-root /path/to/new-external-output
```

Tests cover real one-output shapes/frozen encoder, strict ImageNet-only loading
without touching decoder/head, rejection of segmentation warm-start config,
configuration isolation and CPU rejection. Existing Kernel loss, optimizer/
RNG resume and final padding/OCR boundary regressions are rerun.

## Actual launch qualification

Launched16 October2026, detached PID1457380, under
`/extra1/bmartelli/context-str-runs/kernel-only-fresh-v1-20261016/`.
22 focused/regression tests pass. Real CUDA FP32/batch8 smoke passed and
updated the fresh head by Linf1.00002e-4, with3.683GiB peak allocated /
3.949GiB reserved and no OOM. The full run has passed25 CUDA optimizer steps.
This is an actual training start, not100-epoch completion.

Every encoder tensor in `initial_state.pt` was compared exactly to the pinned
local ImageNet state. The newly generated decoder/head fingerprint is recorded
in the contract. The epoch0 checkpoint matches every tensor of that new initial
state, confirming the smoke update was reset completely. Code hashes are
unchanged after launch. The deleted fine-tuning directory is absent; it is not
a source dependency of this run. Qualification is in
`provenance/kernel_only_fresh_v1_qualification.json` and external
`qualification_at_launch.json`. Live state is in `training.log`/`progress.json`;
the detached process automatically evaluates six paddings after100 epochs.
