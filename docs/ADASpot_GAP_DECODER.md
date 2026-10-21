# Trainable Gap decoder — approximate reproduction

User-authorized training improvement, ADASpot-GAP-DECODER-001. The original
paper3.2–3.3/Algorithm1 and supplied AdaSpot proposal4.1–4.2 were inspected with
the existing implementation. The proposal motivates Gap-aware separation but
does not specify this separate decoder or the numerical training choices.

The normalized linear/nonlinear frozen-feature experiments both completed20
epochs and selected epoch0/off:524/983 exact localized words, unchanged baseline.
Normalization and threshold selection alone did not fix the failure. This does
not prove a unique cause. The next test adapts the spatial representation itself.

`ASSUMPTION`: deep-copy the qualified bottleneck and four attention upsampling
stages and train them plus a new zero-weight Gap head/bias-2. Freeze the original
ResNet50 and the entire original Text/Kernel decoder, including BN statistics.
Source/checkpoint tensors load strictly. During training teacher predictions
define foreground strata only; they receive no gradients. Gap decoder BN trains.
The copied branch shares no parameter or buffer storage with its source.
No GT is available to prediction. This adds a decoder and cost: it is an isolated
feature-adaptation experiment, not the final shared-feature AdaSpot architecture.

`REPOSITORY_SPECIFIED`: same paired TRAIN image/mask/owner augmentations as the
qualified backbone, RGB512/native256, radius3 native-pixel inter-word Gap targets.
Original Text/Kernel inference and all legacy scientific source files remain
unchanged. Predictions use existing kernel assembly and the explicitly documented
cut/preserve variants. PARSeq AR1 is the qualified recognizer, not proven exact
paper PARSeq-S; no caption/context/fallback/strict result is claimed here.

`ASSUMPTION`:1030fit/100calibration/125evaluation IDs are reused unchanged from
the normalized trial. Calibration was seen in source M/K pretraining, but not
in new Gap optimization. Repeated evaluation is exploratory, not independent.
Optimize every fit image once per epoch with original paired augmentations and
new epoch seeds, AdamW lr1e-4/decay1e-4,10full epochs. The loss is.5balanced BCE
+ .5masked soft Dice + .25negative BCE on valid GT Text pixels. Four teacher-M
foreground/Gap strata contribute equal class means, fixed denominator4; an absent
class contributes0. Dice sums batch pixels, smoothing1. All-invalid batches skip
optimizer updates and weight decay. No reduction of the evaluation denominator.
These coupled changes do not measure each contribution independently.

`ASSUMPTION`: checkpoint candidates0/1/3/5/10 and thresholds.5/.75/.9/.95/.99,
cut/preserve plus baseline/off; select native one-to-one bbox wordF1, fewer fused
words, earlier epoch, higher threshold and cut tie, using ONLY100calibration
images. All10epochs run regardless of selection. Keep epoch0 eligible: forcing a
trained epoch would conceal failure. Evaluation images/labels load afterward.
Report native geometry separately from original RGB localization and exact OCR.
Native and RGB rescaling can change IoU matches at threshold.5. All983care words,
one-to-one matching, punctuation-preserving Unicode casefold, duplicate/empty
care predictions in denominator remain as in the qualified reference.

`HARDWARE_DEVIATION`: actual PAPER GPU verified before run, single GPU FP32
batch4, sequential segmentation/OCR loading, allocated/reserved VRAM and OOM
recorded, no CPU or precision fallback. This does not qualify8GB execution or
paper latency. Check CUDA updates, nonzero decoder gradients, exact teacher
outputs and unchanged source weights/BN after training.

`UNRESOLVED`: optimal target/schedule, independent transfer and variance,
calibration exposure, full shared AdaSpot architecture and exact paper results.
Configuration and all source/input hashes, training epochs, candidate metrics,
Gap probability maps, instance labels, RGB crops and PARSeq readings are saved
outside the repo. No scientific component is silently promoted into STRICT.

The initial new architecture test referenced a nonexistent `UpBlock.conv` field;
corrected the test to inspect the actual decoder parameters. The production
model and its exact teacher-forward assertion were not relaxed.

Completed run:
`/extra1/bmartelli/context-str-runs/adaspot-gap-decoder-v1-20261019`.
All10epochs,2580CUDAupdates,1030images/epoch, no invalid-batch skips, FP32 on
actual RTX3090. Loss .82273→.47205; selection remains epoch1/.9/cut by100image
calibration (667→671matches, F1 .8321896444→.8345771144). Full source state,
including BN, unchanged; exact teacher forward and nonzero new decoder/head
updates passed. Original frozen parameters58,155,146; additional trainable
Gap parameters32,598,049. This diagnostic substantially increases model cost.
Training epochs totaled565.5seconds. Peak allocated3,762,007,040bytes and
reserved4,588,568,576bytes, batch4×RGB512, no OOM.

On all125evaluation images/983care words:

| Measure | Baseline | Epoch1 /.9/cut |
| --- | ---: | ---: |
| Native bbox matches | 650 | 651 |
| Native wordF1 | .643245918 | .643280632 |
| RGB bbox matches | 648 | 649 |
| Exact read+localized words | 524 | 523 |
| Proposals | 1038 | 1041 |
| End-to-end Hmean | 51.8555% | 51.6798% |
| Recovered / damaged exact words | — | 2 / 3 |
| GT words involved in fusions | 202 | 195 |

Do not promote this as an OCR improvement. A nonzero selected epoch alone is
insufficient. Independent audit passed250native/250OCR and100selected-calibration
image cases, source/crop/checkpoint hashes and all candidate selection checks.
Result SHA256:
`443a51ce6b7e165bf5930966ad2164a08300305480f8403bfd2a9d5885a07338`.

Post-hoc pixel diagnostic (`selected_gap_pixel_diagnostic.json`):349/3745hard
Gap pixels detected at threshold.9,717valid GT whole-word-region pixels cut,
1177total predicted-M pixels cut. AllvalidGTGap19278pixels,5513true positives,
11362Gap predictions. Whole-word regions are not glyph ink. These explain the
limited separator effect, not a proven isolated cause of every OCR failure.
The subsequent explicitly post-hoc selection experiment optimizes actual
read-word count on calibration, rather than native geometry alone; see
`ADASpot_GAP_OCR_SELECTION.md`. All negative results remain preserved.
