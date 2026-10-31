# ICDAR2013 Total-Text transfer and training duration

**Approximate reproduction / research extension.** This study tests frozen
Total-Text Text+Kernel transfer, ICDAR2013-only fine-tuning, and mixed
ICDAR2013/Total-Text fine-tuning with the identical source checkpoint,
PARSeq, crop and emission policy. The user asks whether training beyond 30
epochs improves ICDAR2013 word F1. Paper sections 3.2, 3.3, and 4.1 and the
existing matched-loss, mixed-replay and output evaluation code were inspected.
The extra kernel, Dice loss and this metric are not paper-specified.

## Predeclared comparison

The original 183 ICDAR2013 fitting images alone contribute gradients. Pool the
existing disjoint 23-image selection and 23-image calibration cohorts into a
46-image validation set for checkpoint choice. This uses more validation
images, but both groups have historical upstream exposure, and no untouched
calibration cohort remains. The complete 233-image TEST set is also historically
explored. Neither TEST labels nor TEST scores choose a checkpoint or policy.

Both arms start from the same strictly loaded Text+Kernel Total-Text model;
same seed, frozen ImageNet ResNet50 encoder/BN, Adam 1e-5, weight decay .01,
batch8, data transforms, 768 RGB/384 map and source BCE+Dice. The ICDAR-only
arm uses all183 fitting images once per epoch. Mixed adds one equal-sized
Total-Text batch at each ICDAR optimizer step; its objective is half ICDAR and
half Total-Text. Extra donor examples and compute are confounders. Input and
target hashes, train/test RGB-disjointness, donor overlap and source identities
are checked before execution. Epochs 0,10,20,30,40,50,60 are evaluated on the
same pooled validation population; each arm's best through30 and through60 is
frozen using F1, earliest epoch for exact ties. Predeclared epochs0/30/60 and
validation-selected checkpoints are evaluated on TEST only after both arms are
sealed. A higher training duration is not assumed to improve F1.

Every method uses the same frozen component/crop/sequence policy and genuine
PARSeq. The policy permits short alphanumeric words, although its sequence
confidence cutoff remains an assumption selected in previous work; it is not
retuned on TEST. Metric: casefold and alphanumeric per-image exact lexical
multisets, micro F1 without IoU; all1095 care words and233 images retained.
No DeepSolo inference or routing is involved. Prior test and policy exposure,
small validation and single-seed comparisons mean results are exploratory, not
official polygon Hmean or strict paper results.

## Implementation and execution

`configs/approximate/icdar13_transfer_duration_v1.json`,
`scripts/run_icdar13_transfer_duration.py`, and
`tests/scripts/test_icdar13_transfer_duration.py` define the isolated study.
The explicit source plan contains all data, checkpoint and cache locations;
no model default supplies them. The run saves configuration, original code
snapshot, hashes, validation/test per-image masks/readings/counts, checkpoint
identity and per-epoch allocated/reserved GPU peaks. Execution is FP32 on the
actual shared RTX3090 with a resident PARSeq and sequential mixed-domain
backward: `HARDWARE_DEVIATION`, not an 8GB GPU qualification. Failures and OOM
are recorded and propagated. The 46-image validation split and unchanged
reused policy, mixed-domain weight and requested budgets are `ASSUMPTION`;
strict model/metric equivalence and historical exposure remain `UNRESOLVED`.

The first smoke failed at its full-test denominator guard because it applied
the full population requirement to a four-image smoke. The fix preserves the
failure and the second real GPU smoke completed both one-step arms and the
post-freeze evaluation. This is not a quality result. External full run:
`/extra1/bmartelli/context-str-runs/icdar13-transfer-duration-20261029-v1/full`.
The original process stopped without a failure record after the ICDAR-only arm
completed60 epochs and the mixed arm completed29; neither arm reached TEST.
The saved mixed weights did not include optimizer state, so loading them as a
training continuation would change the optimization. The recovery uses
hardlinked immutable copies of the complete ICDAR-only evidence and repeats the
mixed arm from the same initialization, seed and optimizer. Every already-seen
mixed epoch must reproduce its fitting loss, sample order and step counts; the
epoch10 and20 state tensors must match exactly. The new recovery root is
`/extra1/bmartelli/context-str-runs/icdar13-transfer-duration-20261029-recovery-v2/full`.
A managed host process preserved it across chat interruptions and ran the
independent CSV audit after training and full evaluation succeeded.

## Completed results

**Approximate reproduction.** All233 TEST images and1095 care words are
retained in each outcome. DeepSolo was not run: every row is0/233 (0.00%).
The pooled validation has46 images and159 care words. The frozen policy has
text/kernel thresholds .5/.5, padding4 RGB pixels, maximum50 crops and PARSeq
sequence probability .9; alphanumeric words of any length may be emitted.

| Expert | Epoch | Validation F1 % | TEST F1 % | TEST TP / FP / FN | DeepSolo images/total (%) |
| --- | ---: | ---: | ---: | --- | --- |
| Fixed Total-Text transfer | 0 | 73.4568 | 72.6420 | 774 / 262 / 321 | 0/233 (0.00%) |
| IC13-only, best validation through30 | 10 | 80.3738 | 74.5779 | 795 / 242 / 300 | 0/233 (0.00%) |
| IC13-only, fixed | 30 | 79.7428 | 73.8756 | 772 / 223 / 323 | 0/233 (0.00%) |
| IC13-only, best validation through60 | 60 | 84.2444 | 72.4350 | 766 / 254 / 329 | 0/233 (0.00%) |
| Mixed, best validation through30 | 30 | 79.4953 | 74.5265 | 787 / 230 / 308 | 0/233 (0.00%) |
| Mixed, best validation through60 | 60 | 80.5031 | 73.7237 | 787 / 253 / 308 | 0/233 (0.00%) |

Epoch60 falls by1.4406 TEST F1 points versus30 for IC-only and by0.8029 for
mixed, despite improved validation in both arms. Paired2000-image bootstrap
intervals for those fixed-checkpoint differences are [-3.1543,+0.1934] and
[-2.3384,+0.7185] points respectively. They include zero, condition on the
trained models and exclude model-selection/training variance and historical
exposure. These data do not establish that longer training generally harms F1;
they do show no observed TEST gain in these two runs. The observed mismatch
between pooled validation and TEST is consistent with selection instability or
distribution shift, without distinguishing their causal contributions. Epoch10
was selected by validation among checkpoints up to30, not selected using TEST.

All29 interrupted mixed epochs reproduce original loss, batch order and step
counts exactly; the epoch10 and20 model tensors match exactly. Both arms verify
that the encoder state remains frozen. RTX3090 FP32 batch8 training allocated/
reserved peaks are8,998,063,616/12,079,595,520 bytes IC-only and
9,211,232,768/12,041,846,784 bytes mixed. The hardware was shared, the peaks are
per-process allocator values rather than total GPU use, and neither arm
qualifies 8GB execution. No OOM was recorded.

The independent audit replays1631 TEST image/checkpoint outputs, parses original
Task1 CSV transcriptions, verifies every mask checksum, confidence/empty-word
emission and TP/FP/FN, validates frozen epoch selection and both60-epoch
histories, and reproduces aggregate scores. It also computes paired bootstrap
differences. Receipt: `full/independent_audit.json`; numerical and per-image
evidence: `full/results.json` and `full/{icdar13_only,icdar13_totaltext_mixed}/test`.
Focused regression tests pass. The scope does not re-run neural inference or
remove the earlier TEST exposure.
