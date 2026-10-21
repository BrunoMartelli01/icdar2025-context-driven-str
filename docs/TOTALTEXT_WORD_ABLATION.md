# Total-Text: four word-training ablations

Classification: **PRACTICAL_APPROXIMATION / approximate reproduction**.
The author-protocol model, Algorithm1 and all existing experimental defaults
are unchanged. These experiments do not establish strict Table5 accuracy.

## Frozen experiment

Five independent branches: `control`, `pixel_link`, `word_balanced`,
`word_zoom`, `deep_supervision`. Each forks the same qualified best
`totaltext-partial-encoder-50ep-v1/frozen_control/best.pt`, additional epoch23,
SHA256 `a047ecf75840f3ed30d1fbd59b2a3bc9db3c60681a7b1936ec702a74d0fdc65f`.
The source validation OCR reference is446/983 casefold exact,478/983 after
Unicode punctuation/symbol removal,567/983 original-coordinate IoU>=.5.

- Official TRAIN only: frozen1130 training/125 validation images,983 validation
  care words. No TEST is opened; no OCR guides training/checkpoint selection.
- Each branch trains50 additional epochs with fresh Adam, LR1e-5, weight decay.01,
  FP32, batch8, frozen ResNet50 parameters and BN. Same seed202503, sample order,
  flip/rotation replay and full letterbox validation. New heads use the declared
  seed; every base checkpoint tensor is transferred strictly.
- RGB512/native256, v4 targets, validity/Don't Care and separator weighting3.
  Foreground threshold.65; no area filter/top10/padding.
- Checkpoint selection: native validation proposal F1@IoU.5, fewer fused care
  words, higher recall, macro foreground IoU, earliest exact tie. Epoch0 included.
- Validation reports macro/micro foreground IoU/F1 and native instance geometry.
  Final PARSeq diagnostic uses original-coordinate bbox one-to-one matching,
  casefold and Unicode P/S-stripped exact rates across all983 GT words.
  Native checkpoint selection and final original-coordinate OCR are distinct.

## Interventions

1. **Pixel-link:** eight sigmoid links from the64-channel final decoder feature.
   Owner maps label foreground-source adjacency within the same instance;
   different-instance/background neighbors are negative. Invalid endpoints and
   out-of-image edges are ignored. Balanced positive/negative link BCE, weight.25.
   Foreground BCE is unchanged. A link starts at bias2 with zero weights, so
   initial connectivity is the baseline. Inference joins8-neighbor foreground
   pixels only when both reciprocal probabilities>=.5; no GT input. Final boxes
   enclose original linked pixels. Added520 parameters; no nucleus expansion.
   Inspired by https://arxiv.org/abs/1801.01315, not a full PixelLink reproduction.
2. **Word-balanced loss:** each care word receives the same total positive weight
   within an image. Average with valid hard-negative background BCE, up to3
   background pixels per positive pixel; separator weight3 retained. With no
   foreground, use all valid background. No extra parameters/inference cost.
   Hard negatives have precedent in https://arxiv.org/abs/1911.08947; the
   particular per-word normalization is an explicit ASSUMPTION.
3. **Word zoom:**50% of TRAIN presentations use a crop anchored on a complete
   owner instance, nominal side.5/.75 of the larger image dimension, enlarged
   where needed to retain the anchor. Other words cut by the crop lose pixel
   validity. Replay the same crop/flip/rotation/letterbox for RGB, target,
   validity, separation and owner labels. Full-image validation/inference.
   TRAIN GT is allowed for augmentation; it is not consumed in inference.
4. **Deep supervision:** one-channel1x1 heads at128/64 decoder resolutions,
   weights.15/.10. Upsample logits bilinearly to the unchanged native256 target
   before masked separator-weighted BCE; do not downsample GT and merge instances.
   Separable interpolation matrices implement align_corners=False bilinear resize
   through GEMM, avoiding nondeterministic CUDA interpolation backward. The
   wrapper declares CUBLAS_WORKSPACE_CONFIG=:4096:8; CPU forward/gradient checks
   compare against PyTorch bilinear interpolation within numerical tolerance.
   Added386 parameters used only during training; eval omits auxiliary forwards.
   Inspired by https://arxiv.org/abs/1807.10165, not a full UNet++ implementation.

All new scientific values are declared in
`configs/approximate/totaltext_word_ablation_50ep_v1.json` before any GPU run.
Their effectiveness is UNRESOLVED. ResNet50/attention follow paper section3.2,
but the high-resolution v4 protocol and new methods are ASSUMPTION, and the
sequential lifecycle/batch differences are HARDWARE_DEVIATION.

## Run, persistence and interruption recovery

`bash scripts/run_totaltext_word_ablation_50ep_host.sh`

The wrapper derives host paths from the repository and requires no installations
or long path arguments. It verifies source/PARSeq hashes and CPU preflights all
five branches, then runs them sequentially on cuda:0, with foreground batch/epoch
progress and visible errors. Sequential execution follows the observed single-GPU
contention; models are released before loading PARSeq. No CUDA was run by Codex.

Persistent output: `context-str-runs/totaltext-word-ablation-50ep-v1/`.
Each branch stores `best.pt`, `last.pt` while training, atomic `epochs.json`,
run/config/code contracts and a final `run_manifest.json`. Only best remains
after successful completion; historical runs are not pruned. Interrupted runs
resume strict model/optimizer/RNG state. An epoch checkpoint committed before
its epoch record is recovered; inconsistent provenance fails explicitly.

PARSeq resumes per-image shards and reuses exact RGB crop hashes with qualified
checkpoint/archive hashes. Cache compact native masks and, for pixel-link,
compressed integer labels; do not save duplicate crop RGB files. Comparison
includes all five branches and per-GT gains/losses versus the matched control.
No variant is automatically promoted or run on TEST.
