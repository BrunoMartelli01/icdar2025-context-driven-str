# ICDAR2015: explicit transfer of trained Total-Text weights

This is **approximate reproduction** under
`configs/approximate/icdar2015_totaltext_weights_v1.json`. The original ICDAR
implementation reused Total-Text architecture and masks but initialized only
from ImageNet; it did not reuse trained Total-Text decoder weights. The new
experiment explicitly tests this missing transfer. The local paper sections
3.2/Figure3 and4.1 and both source implementations were inspected first.

Two pre-existing, complete Total-Text runs supply candidates: the letterbox
control selected at epoch263 and the partial-encoder arm selected at epoch23
after source epoch71. Source checkpoint/report hashes and training manifest are
pinned in the isolated approximate profile. Both have 452 identical state keys
and 58,155,081 parameters. Complete strict loading into the target architecture
passed, with bitwise-identical CPU evaluation on a 64x96 engineering input. No
missing keys, head reset, new layers or non-strict loading is permitted. This
small compatibility check is not a quality result. Preserve original source
architecture identities and historical source hashes in the transfer report;
the target's identity records its explicit partial-encoder gradient regime.

Before training, compare both unchanged models on all fixed ICDAR VAL100/430
care words at RGB736x1280, threshold0.5, axis0 crops. Save masks, all components,
OCR hashes and both raw/emitted metrics. Verify that the complete recorded
Total-Text manifest has no exact image-checksum overlap with ICDAR TRAIN; this
does not establish absence of visually similar scenes. No TEST inference or
TEST-based source choice is involved.

Then fine-tune both sources for100 epochs on the same FIT900, using fresh seeded
Adam. Decoder LR1e-4 is PAPER_SPECIFIED; transfer, fresh optimizer, duration,
layer3/4 LR1e-5, cosine to0.1x, fixed encoder BN, clipping1, ignored-negative
weight3 and raw-word checkpoint selection are ASSUMPTION choices. Retain the
existing BCE/separator supervision and augmentation. Source epoch zero remains
eligible. Physical4/accumulation2 and larger images are HARDWARE_DEVIATION choices
relative to the paper; measure complete backward peaks on the actual RTX3090.

Evaluate qualified CPU PARSeq at epoch0, every5 epochs, new geometry bests and
the final epoch. Select by raw symbol-free spatial word F1, then emitted F1,
geometry F1 and earliest epoch. Keep all original care words and all emitted
predictions, including ignored regions. Output-policy calibration and any later
branch selection remain exclusive to VAL100. This experiment does not establish
a 70% TEST result. See ambiguity `ICDAR2015-TOTALTEXT-WEIGHTS-001`.

## Initial validation and execution readiness

Seventeen focused transfer/encoder/training-runtime tests passed. Complete
recorded manifests contain 1255 Total-Text and1000 ICDAR TRAIN images, with zero
exact image-checksum overlap. Both initial CUDA FP32 validation runs completed
all100 images/430 care words with zero failures and no OOM:

| Source before ICDAR adaptation | Raw word F1 | Emitted word F1 | Raw correct words / predictions |
| --- | ---: | ---: | ---: |
| Total-Text epoch263 |23.9680% |44.9315% |180/1072 |
| Total-Text partial-encoder epoch23 |33.8284% |50.7079% |205/782 |

These are validation measurements, not TEST results. Their lower scores do not
establish whether the learned weights help after domain adaptation. Both
predeclared100-epoch fine-tunings remain planned. Each inference run used GPU
U-Net batch1 and CPU PARSeq batch32, FP32; U-Net allocated903,720,448 /
reserved1,077,936,128 bytes. Another U-Net training process shared the RTX3090.
An earlier CPU segmentation diagnostic was stopped after two images because
CUDA had already completed all100 maps; its partial artifacts are retained and
excluded from quality claims. CPU PARSeq remains unchanged.

A detached, locked runner waits for successful completion of the preceding
OHEM100-epoch suite, then runs full-size backward smokes for both sources before
either new full training arm. Any failed smoke or changed frozen source stops
the queue. A second runner schedules the unchanged20-case VAL100 operating grid
only after each complete training report and selected checkpoint are verified.
No training success or memory qualification is claimed for these queued steps.
The readiness receipt is
`provenance/ICDAR2015_TOTALTEXT_WEIGHTS_READINESS_V1.json`.

The prerequisite OHEM suite has now finished. Both new full-size FP32 backward
smokes passed: physical4/accumulation2, RGB736x1280/native368x640, two logical
updates on16 fitting images and four-image validation. Each peaked
at9,994,451,456 allocated /12,771,655,680 reserved bytes on RTX3090, without OOM.
These small smoke scores are not quality evidence. The detached runner has
completed both100-epoch arms; each full epoch-zero validation exactly
replayed its qualified source. Historical
readiness flags retain their creation-time scope; see the separate
`provenance/ICDAR2015_TOTALTEXT_WEIGHTS_BACKWARD_QUALIFICATION_V1.json`.

Both full20-case operating grids also completed. The `tt263` arm retained
fine-tuning epoch10: raw25.8278% and emitted34.0491% word F1. The `tt_partial23`
arm retained epoch0: raw33.8284%, emitted50.7079%, or57.5130% after its operating
grid (threshold0.65, oriented0, mean character confidence>=0.95, length>=3).
Neither beats prior61.7608% validation emitted F1, so neither is selected for
TEST. Complete100-epoch histories, all900 fitting samples per epoch and all20
candidates are preserved. See
`provenance/ICDAR2015_TOTALTEXT_WEIGHTS_COMPLETION_V1.json` and the resulting
[matched encoder-decay experiment](ICDAR2015_ENCODER_DECAY.md).

## Completed larger-inference validation probe

The isolated `icdar2015_totaltext_resolution_probe_v1.json` uses the same frozen
sources at1152x2048 RGB/native576x1024. Training fields are unused, explicitly
labeled, and the queued736x1280 fine-tunings remain unchanged. This tests scale
mismatch using VAL100 only; it does not select anything from TEST.

| Source at2048 width | Raw word F1 | Emitted word F1 | Raw correct words / predictions |
| --- | ---: | ---: | ---: |
| Total-Text epoch263 |23.9766% |48.8064% |205/1280 |
| Total-Text partial-encoder epoch23 |25.7585% |50.6596% |208/1185 |

Both runs completed100 images/430 care words with zero failures and no OOM.
Each FP32 batch1 U-Net run peaked at1,897,278,976 allocated /
2,384,461,824 reserved bytes on the shared RTX3090; OCR stayed CPU FP32/batch32.
Fourteen relevant tests and an actual2048-coordinate/singleton smoke passed.
The first source recovers25 raw correct words but adds208 predictions; the
second recovers only3 while adding403. Enlarging input alone therefore does not
resolve the recall/precision problem. It does not justify enlarging the queued
training shapes. See
`provenance/ICDAR2015_TOTALTEXT_RESOLUTION_PROBE_V1.json` and ambiguity
`ICDAR2015-TOTALTEXT-RESOLUTION-001`. These are validation diagnostics, not
achieved TEST F1 or evidence that domain adaptation cannot help.
