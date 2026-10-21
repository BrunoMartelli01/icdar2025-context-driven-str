# Total-Text partial encoder: controlled 50-epoch continuation

PRACTICAL_APPROXIMATION / approximate reproduction. Not the paper or frozen
author protocol. Existing architectures, weights, datasets and defaults are unchanged.

## Frozen protocol

- Fork the qualified letterbox best checkpoint **epoch71**, not epoch100.
- Two independent 50-additional-epoch branches: `frozen_control`, `partial_encoder`.
- Same strict model-state initialization, fresh Adam on both, source RNG and
  augmentation epoch72..121. No source optimizer moments are transferred because
  the parameter groups differ. Decoder LR1e-5; encoder LR1e-6; unchanged weight decay.
- Only layer3/4 non-BN parameters unfreeze. Encoder BN affine and running statistics
  remain frozen/eval, including while training. Early encoder stays under no_grad;
  layer3/4 forwards explicitly regain autograd. Decoder BN behavior stays unchanged.
- No new parameters; inference architecture/operations and state keys unchanged.
- Same official TRAIN1130/held-out125 split, v4 targets/Don't Care, input512,
  native256, letterbox, skip floor0.25, separation-weight3, FP32/batch8.
- Threshold0.65 frozen from prior TRAIN diagnostic; pad0, no area/top10 filters.
- Epoch selection: native validation proposal F1@0.5, fewer fused GT words,
  more recall, macro valid-pixel foreground IoU, earliest exact tie. Include
  unchanged source as epoch0 so continuation cannot displace it merely by finishing.
- No OCR in training/checkpoint selection. Final PARSeq on 125 validation images
  only; batch1 segmentation, batch32 OCR; original-coordinate one-to-one matching.
  Native training geometry and final original-coordinate OCR geometry are distinct.
- Existing threshold-sweep baseline is437/983 exact,469/983 without symbols.

## Execution and recovery

`bash scripts/run_totaltext_partial_encoder_50ep_host.sh`

Derives paths from the repo, no arguments/installations. Two branches sequentially
on one GPU to avoid contention. Foreground progress/errors, exclusive flock,
CPU preflights and source/PARSeq/checksum checks before GPU training.

Persistent output outside repo: `context-str-runs/totaltext-partial-encoder-50ep-v1/`.
Best and latest during training; validated latest removed after success. Atomic
writes, strict checkpoint/optimizer/RNG resume, interrupted checkpoint/log recovery.
OCR reuses exact RGB crop hashes plus pinned PARSeq checkpoint/source archive.
`comparison.json` records both reports and gained/lost GT words.

## Evidence

Paper section3.2/Fig3 and implementation details: pretrained ResNet50, attention
skip connections, BCE/Adam; do not establish this fine-tuning schedule. The current
author-protocol class explicitly requires frozen encoder and no_grad throughout.
Fine-tuning, small learning rates and validation selection are **ASSUMPTION**;
sequential model loading and batch choices are **HARDWARE_DEVIATION** relative to
the paper. Expected benefit is **UNRESOLVED**, not a claimed improvement.
