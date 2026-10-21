# Gap with local context: matched 1x1 and 3x3 tests

**Approximate reproduction / research extension.** User-authorized comparison,
following the completed Phase0 and balanced pointwise Gap probe. The original
paper and proposal were inspected; classification and alternatives are recorded
in `ADASpot-GAP-CONTEXT-001` in `PAPER_AMBIGUITIES.md`.

The completed baseline has 524 exact localized-and-read words out of 983;
replacing G with its target gave 600 as an oracle diagnostic. The old balanced
frozen-feature head remained at 524. These motivate the new controlled test,
without claiming that the oracle result is attainable by a learned model.

Two fresh linear heads predict G: 1x1 (65 parameters) and 3x3 (577 parameters).
Both start from zero weights and bias -2; the source Text+Kernel AG-UNet stays
in eval with every parameter and BN state frozen. The 3x3 head extends the
proposal's pointwise head explicitly; it is not a strict paper component.

Four TRAIN-only strata receive equal total BCE weight: Gap positives inside
predicted text, remaining Gap positives, non-Gap predicted-text negatives and
remaining valid background negatives. All positives are retained; each negative
stratum samples without replacement up to max(64, image positive count). Hard
positives are a proxy for bridges, not a topological bridge annotation. Sampling
and weighting alter priors and can overcut letters; the source thresholds stay
fixed. No new target, threshold grid or OCR-dependent checkpoint selection.

Each arm runs 20 epochs, Adam1e-3, zero decay, pixel batch16384, seed202503,
identical examples and epoch permutations, no augmentation. Epoch0 is eligible;
selection uses native word F1, then fewer fused words, earliest epoch. Both full
trainings finish before fresh PARSeq AR1 recognition. Output retains confidences,
raw strings, crop pixels/hashes, G probabilities, instance labels, per-image
metrics and per-GT recoveries/regressions.

Training uses one frozen-backbone extraction on 1130 TRAIN images and sampled
FP32 3x3 feature patches; the qualified old full-feature validation cache is
hash-verified and reused for the same 125 images. Only head optimization and
native head prediction run during the 20 epochs, on CUDA. Zero-padded patch
indexing and native convolution equivalence are tested at borders. This is a
frozen-feature probe, not augmented end-to-end U-Net retraining.

Primary OCR metric: original RGB bbox maximum one-to-one IoU>=.5 and exact
Unicode casefold, punctuation retained, all983 care words. Duplicate and empty
care predictions stay in proposal counts. Ignore handling excludes wholly
invalid instances only after GT-free decoding, retaining full partially invalid
boxes. Native geometry (baseline650 matches) differs from original RGB geometry
(baseline648). Matching identities can differ between geometry and exact OCR.

The holdout is repeatedly explored; results are not independent validation or
official polygon Hmean/Table5. No Mask-Align, dual recognition heads or calibrated
gate is implemented here. GPU model, precision, shapes, memory peaks and failures
are saved; no automatic CPU, batch or precision substitution. Existing runs and
checkpoints remain unchanged. No package installation/download or git commit.

Run `scripts/run_adaspot_gap_context.py` as a module with explicit APPROXIMATE
config and explicit source-run, previous-probe, phase0-run, word-root,
output-root, PARSeq source/archive/checkpoint and CUDA device arguments. The
output must be a new external directory; a duplicate run fails without writing
into an existing output. Concurrent GPU processes are preserved; this experiment
makes no speed or target-8GB qualification claim.

## Fixed-final-epoch follow-up diagnostic

Both primary arms completed their full 20 epochs and selected epoch0 by the
predeclared native geometric criterion. Thus primary selected-checkpoint OCR
alone cannot describe OCR of the trained heads. A separate **post-hoc fixed
final-epoch diagnostic** evaluates each epoch20 checkpoint once, without any
checkpoint search, promotion, new training or primary-result mutation.

`ASSUMPTION`: the fixed endpoint is appropriate to check geometry/OCR selection
mismatch, not a newly selected optimum. The existing `HARDWARE_DEVIATION` for
CUDA FP32 cached features and sequential PARSeq remains explicit. Configuration:
`configs/approximate/adaspot_gap_context_final_diagnostic_v1.json`; implementation:
`scripts/evaluate_adaspot_gap_context_final.py`. Outputs use a separate external
root, preserve full raw OCR/confidences and distinguish this post-hoc analysis
from the primary matched experiment.

## Completion: 19 October 2026

Primary run:
`/extra1/bmartelli/context-str-runs/adaspot-gap-context-v1-20261019`.
Both arms completed 20/20 CUDA epochs, 720 optimizer steps each, on RTX3090,
FP32. Their 580,938 examples and all epoch permutations match exactly; all
central feature vectors also match the previous frozen-feature probe bitwise.
TRAIN strata: 23,636 hard positives, 155,374 remaining positives, and 200,964
negatives in each of the two negative strata. No training or evaluation sample
was silently omitted.

Both arms selected epoch0 by the predeclared native geometry criterion. Selected
checkpoint OCR therefore remains baseline524/983, Hmean51.8555%. This is a
rejection of promotion under this protocol, not a claim that the trained heads
produce identical maps or OCR.

Separate fixed-final-epoch diagnostic (post-hoc, no promotion):
`/extra1/bmartelli/context-str-runs/adaspot-gap-context-final-diagnostic-v1-20261019-retry1`.

| Endpoint | Exact OCR /983 | Original RGB matches | Native word F1 | Fused-word involvement | Recovered / damaged |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original Text+Kernel | 524 | 648 | 64.3246% | 202 | reference |
| Trained 1x1, epoch20 | 496 | 624 | 62.4071% | 176 | 8 / 36 |
| Trained 3x3, epoch20 | 476 | 617 | 61.6222% | 151 | 15 / 63 |

At epoch20, 1x1 detects2197/3745 hard Gap pixels and removes93048 valid target-
Text pixels from predicted foreground; 3x3 detects2339/3745 and removes54515.
Target-Text denotes whole-word regions, not just character ink. RGB itself is
not erased: changes in instance support alter boxes/crops. Fewer fused words
therefore do not imply better localized-and-read word accuracy. These data do
not uniquely identify whether prior shift, calibration, head capacity or
optimization causes the damage; unchanged LR has different dynamics at the
larger head fan-in. No calibration grid or further training was added.

The qualified Text+Kernel reference is retained. No actual feature Mask-Align,
shared recognition heads or gate was added. Repeated-holdout results remain
exploratory and cannot establish independent transfer or official benchmark
performance.

Verification: 67 focused unit/regression tests pass; real CUDA smoke/update and
patch/native equivalence pass for both heads. Primary independent audit checks
250 native and250 OCR image cases,580938 center-feature cases, complete budgets,
cache/input hashes, identical sample order and unchanged common-crop OCR. The
separate final diagnostic audit checks another250 OCR image cases and verifies
all final native metrics/audits match the actual training endpoints.

Feature extraction peaks2.33GiB allocated/3.54GiB reserved; head training peaks
about1.36GiB allocated/1.38GiB reserved, with cached GPU patches included. No OOM.
Other GPU processes were preserved; no latency comparison or8GB qualification.

Resolved execution failures: the initial read-only preflight used system Python
without SciPy; corrected to the existing main venv. The first post-hoc diagnostic
failed before inference because CUDA memory-stat reset preceded CUDA context
initialization. Explicit initialization fixed it; the corrected diagnostic ran
to completion in a new output root. The failed root, traceback, log and original
source snapshot are preserved. Primary training/results were not rerun or
modified. No failures remain outstanding.
