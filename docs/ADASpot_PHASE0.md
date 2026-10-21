# AdaSpot Phase 0: paired crop and decoding diagnostics

**Approximate reproduction / exploratory research extension.** User-authorized
continuation of the AdaSpot roadmap. Read original paper sections3.2–3.3 and
proposal sections4.1–4.8, hypotheses and Phase0 before implementation. Choices
are classified in `configs/approximate/adaspot_phase0_v1.json` and
`ADASpot-PHASE0-001` in `PAPER_AMBIGUITIES.md`.

Reuse the completed Text+Kernel matched control, frozen RGB512/native256 maps,
125 held-out TRAIN images, all983 care words, and strictly loaded qualified
PARSeq. No segmentation training, target edits or TEST selection are introduced.
The holdout has been repeatedly explored; no independent generalization claim.

The new script freezes axis/PCA/Bézier RGB crops before reading recognition GT.
It reuses the existing rectification helpers; all predicted views retain their
original bbox for scoring. Failed fits preserve axis pixels with an explicit
reason. RGB rectification is not feature-space Mask-Align.

Two separate GT diagnostics are explicit: original care-word axis boxes, and
replacing G with its saved target while leaving predicted M/K and all thresholds
fixed. GT axis boxes are not guaranteed to be optimal recognition crops. The
oracle-G arm measures the potential of the *current* radius3 target, not every
possible word separator. Neither is a deployment method. Saved targets never
enter predicted-crop geometry or confidence routing.

NAR0/NAR1/AR1 use identical strictly loaded PARSeq weights and explicitly change
the upstream `decode_ar`/`refine_iters` flags only inside the new diagnostic.
No production adapter/default/checkpoint is changed. These modes are proxies:
they are not the proposed one-layer/four-layer recognition heads and cannot
establish the true shared-feature accuracy/compute gap. The qualified checkpoint
is not established as the paper's exact PARSeq-S.

Compare NAR1→AR1 confidence escalation with32 seeded random subsets at the same
integer crop budgets. Confidence is minimum character probability, empty strings
first, deterministic ties; no threshold fitting, rejection or calibrated MLP.
The GT-priority policy is a diagnostic, not an exact combinatorial oracle bound.
Matching remains maximum one-to-one bbox IoU>=.5 plus Unicode casefold, retaining
punctuation, all983 words and duplicate/empty care outputs. Gains and losses
retain sample/word IDs. Baseline per-word successes must replay unchanged.

GPU FP32/batch8 timing uses three warmups and synchronization, includes transform,
forward, transfer and CPU token decoding, excludes PNG disk loading. Report full
batch composition, actual GPU, PyTorch/CUDA, allocated/reserved VRAM and OOM.
All modes process identical unique crops in identical batches. These timings
are not end-to-end shared-backbone latency or FLOPs and include diagnostic crops.
No installation or download is performed.

All input maps, manifests, RGB, checkpoint/runtime archive and relevant sources
are hash-bound; checks repeat after inference. Generated crops, labels, readings,
fully resolved protocol, per-image/per-word metrics and report live outside the
repository. Interrupted inference can be replayed with identical configuration;
conflicting protocol or completed output fails explicitly.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venvs/context-str-main-py311/bin/python -m scripts.diagnose_adaspot_phase0 \
  --mode approximate --config configs/approximate/adaspot_phase0_v1.json \
  --source-run /explicit/completed/adaspot-gap-run \
  --word-root /explicit/totaltext-v4 \
  --output-root /explicit/new/external/phase0-run \
  --parseq-source /explicit/qualified/parseq-source \
  --parseq-archive /explicit/qualified/parseq-archive.tar.gz \
  --parseq-checkpoint /explicit/qualified/parseq-bb5792a6.pt --device cuda:0
```

## Completed diagnostic and next controlled intervention

PAPER run `/extra1/bmartelli/context-str-runs/adaspot-phase0-v1-20261017` completed
all15 arms on RTX3090, FP32/batch8,125images/all983carewords. Native baseline650
matches is distinct from originalRGB bbox648. Exact OCR: baseline524; GTaxis838
(322 recovered,8 damaged); oracleGap600 (78 recovered,2 damaged); oriented527
(13/10); Bézier523 (16/17). GTGap reduces native fused-word involvement202→34,
with splits89→84. These oracle counts are not achievable-model claims.

Same-weight NAR1 reads518, AR1 reads524, only6net/983=.61percentage points.
Confidence escalating259/1038crops reaches524; equal-budget random mean520
(32fixedseeds). The .61pp margin is over all983words, including localization
failures: it must not be compared directly with the proposal's3pp criterion on
hard-word crops. Neither that criterion nor the true AdaSpot-head margin has
been tested by substituting these modes.
Warm synchronized mean seconds/crop on identical4227unique RGB crops/batches:
NAR0 .001560, NAR1 .001843, AR1 .004248. Timings include GT/oracle crops;
recomputing AR also repeats the PARSeq image encoder, unlike shared AdaSpot.
Peak allocated .138GiB, reserved .164GiB; no OOM.

Independent `scripts/audit_adaspot_phase0.py` verifies1875image/arm cases,
15708proposal cases,192random-routing cases, identical mode batch membership,
all983denominators and521source/input hashes. Baseline raw strings and successes
replay unchanged.35focused tests pass. An initial test command referenced a
nonexistent `tests/localize/test_adaspot_gap.py`, exited4/no tests run; corrected
actual test paths pass. No failure was suppressed.

The favorable oracleGap motivates the separate `adaspot_gap_probe_v1.json`
experiment: train only a new65parameter Gap head with sampled, class-balanced
pixels and unchanged frozen Text/Kernel features. This is a20epoch CUDA
**linear probe**, without augmentation, rather than a full segmentation retrain.
All positives are retained; negative strata distinguish predicted foreground
from background. Include epoch0 in geometry-only selection. Calibration and
false cuts remain risks; no learned gain is assumed. Configuration choices and
cache/selection protocol are recorded in `ADASpot-GAP-PROBE-001`.

## Completed balanced Gap probe

PAPER run `/extra1/bmartelli/context-str-runs/adaspot-gap-probe-v1-20261017`
completed20/20CUDAepochs,720optimizersteps,FP32. Only65newheadparameters are
trained; all179010positive training pixels retained,200964sampled negatives in
predicted-text and200964background negatives. Selected epoch7 by native geometry
with epoch0 eligible. Original M/K checkpoint/exports are unchanged.

- Exact OCR remains524/983, Hmean51.8555%,648originalRGB matches,1038proposals.
  One recovery (`total_text:1475`,GT1), one regression (`total_text:1022`,GT3).
- Selected head detects260/3745(6.94%) target-Gap pixels incorrectly included in
  the *control's* predicted text; it blocks6192foreground pixels in total. These
  counts are not interchangeable with the earlier Gap-branch4256pixel audit.
- Native fused-word involvement202→199, split89→88; native matches stay650.
  At epoch20, bridge hits697/3745(18.61%), fused-word involvement194, matches649.
  Training loss .71559→.24460 does not imply improved localization or OCR.
- Feature extraction peak2.33GiB allocated/3.54GiB reserved; head training
  .243/.254GiB; no OOM. Full validation features and sampled train vectors are
  cached FP32. This probe is not a repeated augmented U-Net training schedule.
-55focused tests pass, real first CUDA update is saved; independent native and
  RGB Hungarian audits pass all125images, all983words,20epochs, selection/splits,
  input/cache hashes. Diagnostic sources and earlier run hashes remain intact.

Interpretation: oracle separation has substantial potential; this balanced
frozen-feature1x1probe does not realize it. Lack of information in frozen features,
limited head capacity, target mismatch, optimization and calibration remain
alternative hypotheses; this run does not identify a unique cause. Keep the
qualified Text+Kernel reference. Actual shared-feature Mask-Align and AdaSpot
recognition heads/gate remain separate future components, not completed here.

Per-word geometry and OCR use separate maximum matchings: tied assignments can
choose different GT identities. Do not read the saved geometry/OCR flags as an
exclusive causal taxonomy. Aggregate counts and exact OCR gain/loss IDs follow
the declared respective matching protocols.
