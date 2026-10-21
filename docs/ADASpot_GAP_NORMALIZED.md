# Better Gap training: fit, decision calibration and full Text support

**Approximate reproduction / exploratory extension.** User-authorized response
to repeated epoch0 selections. Classification: ADASpot-GAP-NORMALIZED-001 in
PAPER_AMBIGUITIES.md. Original paper/proposal and current code were inspected.

The earlier weighted pointwise/spatial heads learned separators but removed
large parts of target whole-word regions. A fixed-threshold cut also applies
cost-sensitive scores as if they were calibrated probabilities. The new method
keeps the previous backbone and targets, and tests:

- TRAIN-only feature normalization, matched spatial linear/hidden32 ReLU heads;
- 1030 images for head fitting, 100 disjoint TRAIN images for checkpoint/decision
  calibration, and the existing 125-image evaluation cohort untouched by choices;
- fixed 20 epochs and epochs0/5/10/15/20 as candidates; calibrated thresholds
  .5/.75/.9/.95/.99 and off, using native word F1 and declared conservative ties;
- cut and explicit support-preserving decoders as matched calibration candidates.

Normalization and class weights use sampled fit pixels only. Both heads see the
same examples and permutations. No image augmentation or backbone update is
introduced. The100calibration images were previously seen by frozen M/K training;
they are held out only from new Gap-head fitting. Decision threshold calibration
is not calibrated-probability fitting or temperature scaling. All assumptions
and exact parameters are saved in the resolved approximate configuration.

Preserve decoding grows assigned instance IDs back into the original predicted
Text mask using the existing deterministic geodesic flood. Every existing ID
stays fixed; missing unseeded components get an explicit fallback. No foreground
is invented outside predicted M and no GT determines predictions. False seeds
and false bridges can remain; cut is retained as a matched candidate.

Every checkpoint/threshold/decoder is chosen on the100images. The125evaluation
features and recognition labels are loaded only after both20epoch trainings and
choices finish. Selection includes epoch0/off. Do not suppress this candidate
merely to force a positive training result.

Evaluation keeps all983care words, one-to-one original RGB bboxIoU>=.5 and exact
Unicode casefold with punctuation, empty/duplicate care predictions in proposal
counts. Genuine qualified PARSeq AR1 is loaded only after head tensors unload;
it is not established as the exact paper PARSeq-S. Results remain exploratory
because this evaluation cohort has been repeatedly examined; no TEST or official
polygon/strict reproduction claim.

Pixel patches, fit indices, fit-only mean/std, calibration maps/features, all
calibration candidate metrics, best/last heads, masks, crop pixels, OCR confidence
and per-word outputs remain auditable outside the repository with hashes. Actual
GPU, FP32 shape, memory and OOM are recorded. No installation, automatic fallback
or checkpoint overwrite. Native-convolution/patch normalization equivalence and
support/ID conservation have focused tests.

Completed on PAPER RTX3090 at
`/extra1/bmartelli/context-str-runs/adaspot-gap-normalized-v1-20261019`.
Both heads completed20epochs/640CUDAupdates on521892fit pixels; both selected
0/off on802calibration words, nativeF1 .8321896444. Each returns524/983exact,
648RGB-localized/650native-localized words,1038proposals, Hmean51.8555%,
no recovered or damaged words. Do not claim a gain or promote a trained head.

Independent audit `independent_audit.json` passes375native and375RGB/OCR
image-arm cases,200selected-calibration cases, all521892fit pixels for exact
fit/calibration isolation and independent mean/std reconstruction, all candidate
selection/population checks and input hashes. Both heads used identical fit
permutations. Result SHA256:
`c72756a0ad42a29d2da070774896b781473fdf434e3d4c9c1787cab41a6e1e4a`.
The new training direction adapts a complete spatial decoder;
see `ADASpot_GAP_DECODER.md`. This is a new hypothesis, not an attribution of the
failure to a conclusively isolated cause. Preserve these negative results.
