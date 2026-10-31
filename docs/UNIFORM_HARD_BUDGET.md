# Uniform pipeline with hard fallback budgets

Approximate reproduction / research extension. Resumed by explicit user request
on2026-10-31. Larger fallback controls cannot satisfy this objective.

| Dataset | Hard DeepSolo limit: images/total (%) | Lexical micro F1 target | Retained v6b full TEST F1 |
| --- | --- | --- | --- |
| ICDAR2013 | 61/233 (26.18%) | >89.7% | 84.6116% |
| ICDAR2015 | 228/500 (45.60%) | >78.2% | 79.9257% |
| Total-Text | 150/300 (50.00%) | >79.8% | 80.7919% |

These are authenticated cached expert results, with zero new DeepSolo calls
for metric replay. The metric is the existing per-image casefold/alphanumeric
lexical multiset micro F1, retaining all care words and native256 ignore
handling. It does not establish equivalence to paper Table5 accuracy.

## Budget enforcement

`configs/approximate/uniform_pipeline_hard_budget_v9.json` uses the unchanged
v6b U-Net/PARSeq/router/fusion assets and adds an explicit image-budget contract.
An excessive requested reference policy fails before inference resources load.
For another batch population, floor(reference quota * batch size / reference
population) preserves the upper fallback fraction. The exact full benchmark
counts remain61/233 (26.18%),228/500 (45.60%),150/300 (50.00%). Route partitions,
counts and percentages are checked before the isolated DeepSolo worker starts.
The older profiles and their historical ceil rounding remain available.

This is a hard operational constraint in APPROXIMATE mode, independent of
the repository's scientific STRICT reproduction requirements.

## First isolated comparison

`configs/approximate/uniform_pooled_gate_v9.json` compares four pooled two-count
HGB image heads with the retained native heads. Keep each dataset's v6b primary
word admission, vocabulary, encoder PCA and post-fallback fusion fixed. Each
pooled head uses the same feature schema and one shared pair of regressors
across the three datasets, with equal total fitting weight per dataset.
Targets are light-minus-native-DeepSolo TP and prediction-count differences,
normalized by raw crop count plus one. FIT observations come from the qualified
conditional decoder-heldout banks; validation uses the full-FIT frontend.

The predeclared grid is100 boosting iterations,10 minimum leaf samples,
3/7 leaves and10/100 L2 regularization. Choose a pooled head only if the paired
domain-macro bootstrap95% lower gain is positive and no dataset loses more
than0.1 validation percentage points. Among eligible heads choose highest mean
validation gain; exact ties retain grid order. Identity remains eligible.
All validation quotas use the new floor rule. No new TEST prediction or label
is loaded during fitting/selection. No neural inference or model is added.

The scripts preserve source snapshots, source-map hashes, prepared feature
hashes, all four trials, route populations and an independent refit/replay
audit. If no head qualifies, retain the current model and report the negative
result; do not claim progress from higher fallback fractions.

Classifications and remaining limitations are recorded in
`docs/PAPER_AMBIGUITIES.md`, UNIFORM-HARD-BUDGET-009. Historical TEST exposure,
reused small validation, nonnested stacking and upstream pretrained exposure
remain material. The hard-budget objective remains active until all three
targets are actually achieved or a documented external blocker applies.

## Completed first milestone

The hard budget boundary and unchanged pipeline pass22 focused tests. Fresh
RGB-only twelve-image probes pass all three independent engineering audits:
DeepSolo3/12 (25.00%) IC13,5/12 (41.67%) IC15,6/12 (50.00%) TT. These are real
selected-only calls, with no reused fallback outputs; they are smoke tests,
not new full-dataset quality results. The initial smoke omitted the `deepsolo`
package from its source snapshot and failed before fallback model loading.
Its frontend outputs and failure are retained; a separate complete snapshot
and new output directory pass all three probes. No OOM occurred on RTX3090.

The pooled study fits2211 conditional decoder-heldout images and evaluates271
TRAIN-heldout validation images. Independent refits reproduce all four models
and every validation decision. None passes the predeclared gain criterion;
the native gate is retained and no new TEST trial is run. For example the
best IC13 validation F1 is94.0810%,12/46 (26.09%) logical fallback, versus
93.2476% at the same12/46 (26.09%) for the retained gate. That isolated domain
gain does not establish a shared improvement; the paired domain-macro lower
gains are negative. IC15 validation uses45/100 (45.00%) under the hard floor
rule, TT62/125 (49.60%).

A separately labeled GT diagnostic recounts the original full IC13 metric and
computes exact-cardinality routing bounds with the current fixed word outputs.
At61/233 (26.18%), even privileged routing reaches only88.0976% with native
fallback or87.7005% with the current fixed fusion. A rational target-residual
certificate also excludes attaining89.7% anywhere within that budget. The
diagnostic opens cached fallback outputs for233/233 images (100.00%) and uses
TEST annotations, with zero new neural calls; its choices are not deployable
and do not select an operational policy. This result concerns the fixed current
branches, not all possible word policies or improved U-Nets. Further work must
improve the light word branch before expecting image routing alone to meet
the target. Portable evidence: `provenance/UNIFORM_HARD_BUDGET_V9.json`.

## Shared word admission and matched U-Net continuation

The v10 word-only comparison fits16122 nonignored FIT readings with equal
total domain weight. Four shared HGB recipes use the existing18 observable
features, with the native sequence pre-cut removed uniformly and validation
probability thresholds selected from0/.3/.5/.7/.9. None meets the predeclared
shared improvement criterion. Independent refits and all validation counts
pass; no TEST trial or neural inference is added. All light-only validation
results have DeepSolo0/46 (0.00%) IC13,0/100 (0.00%) IC15,0/125 (0.00%) TT.
The deployed word policy is retained. Portable receipt:
`provenance/UNIFORM_WORD_ADMISSION_V10.json`.

The next isolated intervention is a matched U-Net continuation, configured in
`configs/approximate/uniform_photometric_continuation_v11.json`. All three
datasets have prepared source-pinned plans, but actual fitting starts with
IC13 only. Compare ten additional epochs with fresh AdamW1e-5 from the same
native selected checkpoint, with and without the qualified RGB-only
photometric recipe. Loss, encoder freeze, mask geometry, physical4/accum2,
validation and inference remain matched. Epoch0 must reproduce the original
validation counts exactly. Targets are unchanged; unit tests verify that
augmentation changes RGB only, is deterministic per sample/epoch, and is absent
from validation. No additional online model or forward pass is introduced.

The controller qualifies one8-FIT/4-VAL backward/optimizer smoke and its strict
checkpoint/epoch/count audit before each183-FIT/46-VAL ten-epoch arm. It records
disk/GPU checks, allocated/reserved peaks, failures and actual child PIDs, and
holds an exclusive process lock. The matched pilot is complete: both ten-epoch
arms and both one-epoch smokes pass their independent audits and select epoch0.
All selected neural tensors exactly match the original model, so the original
deployment is retained. The two full arms each evaluate46 validation images
with DeepSolo0/46 (0.00%) and retain84.6395% fixed-policy raw lexical F1.
This is a light-only validation metric, not the budgeted full TEST metric.
The experiment covers3676 training image-epochs including smoke, and introduces
no additional online parameters or passes. No other domain is trained with
this unhelpful continuation. The immutable launch receipt remains historical;
the completed evidence is `provenance/UNIFORM_PHOTOMETRIC_PILOT_V11.json`.
Source, assumptions and limitations are recorded under
UNIFORM-PHOTOMETRIC-CONTINUATION-011 in `docs/PAPER_AMBIGUITIES.md`.

## Owner-balanced objective and full-pipeline checkpoint comparison

The v12 IC13 pilot changes only the text loss to the existing owner-balanced
BCE/local-Dice objective; the kernel loss and all other scientific inputs match
the completed v11 control. Eleven relevant loss/geometry tests pass. Real
one-epoch smoke and ten-epoch fitting both pass independent state/order/count
audits, selecting epoch0 at84.6395% raw validation F1, DeepSolo0/46 (0.00%).
All452 selected model tensors exactly match the original. The native target
audit covers all229 TRAIN images and849 word owners, with zero foreground
pixels lacking an owner. The proposed objective adds no inference parameters
or passes. Evidence: `provenance/UNIFORM_INSTANCE_OBJECTIVE_V12.json`.

To check the actual goal metric before discarding the retained final weights,
v13 separately evaluates the entire frozen word/image/fusion pipeline on VAL46.
It includes only retained endpoint weights; epoch5 is excluded. All routes are
frozen before fallback and annotation access. Selection uses highest validation
F1 within0.1 point, baseline first; bootstrap uncertainty is reported separately.
The baseline exactly reproduces original frozen counts and routing.

| Endpoint | Complete-pipeline validation lexical F1 | DeepSolo images/total (%) |
| --- | --- | --- |
| Original selected model | 93.2476% | 12/46 (26.09%) |
| Plain ten-epoch continuation | 93.0380% | 12/46 (26.09%) |
| Owner-balanced ten-epoch continuation | 92.0128% | 12/46 (26.09%) |
| Photometric ten-epoch continuation | 92.9487% | 12/46 (26.09%) |

Independent score sorting and Counter recount pass all184 candidate-image
replays, including predictions, ignore flags, care denominators, uncertainty
and selection. These are cached VAL results with zero new neural calls, not
TEST or three-domain quality results. The original model remains selected.
Evidence: `provenance/UNIFORM_CHECKPOINT_VALIDATION_V13.json`.

Disk maintenance reclaims0.9716GiB from optimizer state in four completed,
non-improving v11 checkpoints. Every model tensor, RNG field, history and other
metadata is verified identical after serialization, and selected checkpoints
are untouched. Exact optimizer continuation is unavailable for those completed
files. All four original frozen-source audits pass again. One audit launcher
initially resolved the venv symlink to base Python without Torch; this failed
before auditing/inference and was corrected using the actual venv path. The
failure record is preserved. Evidence: `provenance/UNIFORM_OPTIMIZER_STORAGE_V12.json`.

## Resolution controls and next matched training arm

Two isolated inference probes retain the original trained weights, crop
thresholds, original-RGB padding and50-crop cap. Actual encoder context is
captured at each resolution and passed through the transferred768-trained
PCA/word/image/fusion heads; matching refits are not claimed.

| Input / native logits | Complete-pipeline validation F1 | DeepSolo images/total (%) |
| --- | --- | --- |
| Qualified768 /384 baseline | 93.2476% | 12/46 (26.09%) |
| 1024 /512 | 92.3567% | 12/46 (26.09%) |
| 512 /256 | 90.0958% | 12/46 (26.09%) |

Both candidate exports cover all46 validation images, with no OOM. Each first
passes exact two-image native768 map/context/transcript/crop-hash parity. The
independent score-ranking and Counter audits pass. The1024 frontend performs
254 actual PARSeq crops in20.808 seconds, peaking at1168994816 allocated /
1314914304 reserved bytes;512 performs179 crops in7.666 seconds, peaking at
723288576 /809500672 bytes. These are single frontend runs including I/O on
RTX3090 FP32, U-Net batch1/PARSeq up to32. Native768 has only a two-image parity
probe, so no full native latency comparison is established. Neural parameter
counts are unchanged. Full-pipeline metrics replay cached DeepSolo, with zero
new fallback neural calls. The original resolution remains selected.

A separate deliberately capped process verifies OOM propagation and failure
reporting during model initialization: no neural forward or completion artifact
is produced. Its0.001 allocation fraction is an engineering test condition,
not a failure of the successful1024 run or a production memory setting.
Evidence: `provenance/UNIFORM_SCALED_FRONTEND_V14_V15.json`.

Post-hoc TRAIN-only diagnostics find that duplicate positives from the word
label presence proxy are negligible for the IC13 bottleneck: one excess FIT
positive among579, and zero excess among137 validation positives. No extra
word-label model is added on that basis. The already-qualified encoder transfer
also has exact source feature parity and matching normalization/stride; no
normalization correction is introduced.

The new v17 arm instead retrains from the original common initialization for
30 epochs, restoring the geometric augmentation types named in paper4.2.
Flip probability.5, rotation probability.5 and range[-15,+15] are explicit
assumptions. Only the declared training augmentation fields change scientifically;
loss, optimizer, seed, initialization, partitions, validation and inference match
the completed original control. A3GiB storage reserve is an engineering-only
change. Augmentation is paired across RGB and segmentation targets; validation
and PARSeq crops remain unaugmented.

The IC13 pilot completes all30 epochs and both physical smoke/full independent
training audits. Its earliest best raw-word checkpoint is epoch20,84.4156%
validation F1, DeepSolo0/46 (0.00%). The original raw-word reference is84.6395%,
DeepSolo0/46 (0.00%). Complete frozen pipeline validation gives93.2476% for
both original and geometric-best weights, versus90.4918% for the final weights;
every candidate routes12/46 (26.09%) images logically to cached DeepSolo, with
zero new fallback calls. The independent138-image replay audit passes. The
predeclared baseline-first tie rule retains the original model. Other datasets
are not trained with this recipe, and no new TEST is opened. Evidence:
`provenance/UNIFORM_GEOMETRIC_TRAINING_COMPLETED_V17.json`.

The next v18 controlled screen continues the original selected U-Net for ten
epochs and trains upper ResNet50 layer3/layer4 convolutions as well as the
decoder. Lower stages and every encoder BatchNorm affine/running value remain
frozen. The strict donor initializer freezes all parameters, so upper-stage
gradients are explicitly re-enabled after initialization. Decoder1e-5 and
encoder1e-6 are assumptions; the paper does not establish this freezing policy.
Loss, image order, seed, partitions and no-augmentation geometry match the
completed plain v11 continuation. The extra3GiB storage guard is engineering
only. Actual updated encoder context is saved with each validation map; old
frozen context cannot represent the adapted encoder. Any initial replay
transfers the existing PCA/heads and is only a validation screen, with matching
folds/refits still required before a final candidate. Profile:
`configs/approximate/uniform_upper_encoder_v18.json`.

The v18 smoke and all ten full epochs complete with independent audits passing.
The best raw-word validation checkpoint remains epoch0,84.6395%, versus83.3333%
at epoch10, both DeepSolo0/46 (0.00%). Full pipeline validation retains93.2476%
for the original and identical epoch0 model; the adapted final model gives
92.4051%. Each uses12/46 (26.09%) logical cached DeepSolo decisions, with zero
new fallback calls. The138-image independent replay passes, as does a separate
46-image exact legacy-regression replay after adding explicit adapted context.
No new TEST is opened or model promoted. Training uses54,618,210 trainable
parameters versus32,598,114 for the frozen encoder, with58,155,146 total neural
parameters unchanged. Full fitting peaks at6,155,339,776 allocated and
7,228,882,944 reserved bytes on RTX3090 FP32, physical4/accum2, without OOM.
Evidence: `provenance/UNIFORM_ENCODER_ADAPTATION_COMPLETED_V18.json`.

Storage maintenance also reclaims1.2145GiB from optimizer states in five
completed, nonselected original final checkpoints. Every other serialized
field is roundtrip-equal, selected checkpoints are untouched, and all five
original-source audits pass again. Two preflight audit launches failed before
auditing: the first found compaction still running; the second used an early
smoke snapshot without the subsequently added auditor. Waiting for completion
and using the hash-identical original full auditor resolves both; failure
records are retained. No training/inference failed in these launches. Evidence:
`provenance/UNIFORM_COMPLETED_STORAGE_V18.json`.

The hard61/233 (26.18%) IC13 goal remains unmet. The optional architecture
preference is pending; work continues with U-Net-only localization.

## Completed shared multiscale validation screens

The fixed v19 probability mean improves the initial IC13 VAL screen, but the
subsequent predeclared three-domain rule rejects it. Retain a common candidate
only if mean domain F1 gain exceeds0.1point and no domain loses more than
0.1point. The IC13 pilot was already exposed before that shared rule; the two
other domains were scored afterwards. Reused validation remains a limitation.

| VAL dataset | Original F1 | Equal mask mean F1 | DeepSolo images/total (%) |
| --- | --- | --- | --- |
| ICDAR2013 |93.2476%|93.6306%|12/46 (26.09%)|
| ICDAR2015 |83.9196%|83.9599%|45/100 (45.00%)|
| Total-Text |78.9071%|78.0624%|62/125 (49.60%)|

Every native/scaled RGB view uses identical selected weights within its
dataset. Original native768 parity checks exact maps, actual encoder context,
transcripts and crop hashes on two images per domain. Actual1024 exports cover
all100 IC15 and125 TT validation images; the earlier IC13 export covers46.
Mask averaging adds234 actual PARSeq crop readings on IC13 and fresh readings
on the other domains, reusing the real U-Net outputs. The independent audits
recompute source hashes, aligned probability arithmetic, native context,
ranking and lexical Counter counts. No new DeepSolo inference occurs.

Two engineering failures are preserved. The first IC13 score omitted the
dependency linking the already qualified native frontend; an explicit
hash-checked reference to the original genuine neural qualification resolves
it without inventing a fresh neural check. The first IC15 comparison assumed
the old validation quota equalled the hard quota. The old policy used46/100
(46.00%); hard floor permits45/100 (45.00%). The repair exactly reproduces
the original counts and routes at46/100 (46.00%), then compares both current
branches at45/100 (45.00%). No assertion or care denominator is discarded.
The completed IC15 RGB/crop exports are reused byte-identically in a separate
versioned scoring run. Shared mask mean is rejected, macro delta-0.1405points.
Evidence: `provenance/UNIFORM_MULTISCALE_SHARED_VALIDATION_V20.json`.

The v21 follow-up preserves native masks/context and every native raw word.
It adds only second-view alphanumeric readings with sequence probability>=.9
and bboxIoU<.3 against every retained candidate, with deterministic ordering
and a joint50-candidate cap. The same fixed rule and shared selection criterion
are declared before its scores; there is no threshold grid or dataset-specific
fusion algorithm. Both actual view outputs are authenticated existing caches.

| VAL dataset | Original F1 | Conservative word additions F1 | DeepSolo images/total (%) |
| --- | --- | --- | --- |
| ICDAR2013 |93.2476%|93.5897%|12/46 (26.09%)|
| ICDAR2015 |83.9196%|84.8259%|45/100 (45.00%)|
| Total-Text |78.9071%|78.8337%|62/125 (49.60%)|

Mean gain is0.3917points and maximum domain loss0.0734points, so the common
word-addition candidate passes this screen. All542 candidate-image ranking
and Counter replays pass, as do independent candidate-admission and exact
native-map/context checks. Raw additions are11/45/102 words respectively;
these are candidate counts, not extra DeepSolo images or guaranteed correct
words. This stage performs zero new neural calls and opens no TEST. Earlier
negative outcomes and exposed reused validation are retained; this is not an
unbiased generalization estimate.

The candidate transfers the original native-trained PCA/word/image/fusion
heads. Matching decoder-heldout observations/head qualification and full RGB
runtime integration remain required before a final candidate claim. Online
deployment would need two U-Net forwards and both PARSeq crop passes per
image, with unchanged neural parameter count; cached replay is not measured
online latency. The current deployed models and full TEST table above remain
unchanged. Profile: `configs/approximate/uniform_multiscale_words_shared_v21.json`.

Four historical completed generator checkpoints are exact copied artifacts
under two preserved paths. Their reuse receipts, successful completion,
independent audit, byte hashes, permissions and mtime are verified before
replacing duplicate storage with hardlinks. This reclaims4.4821GiB without
removing any checkpoint bytes, metadata or paths. The existing stage guard
verifies qualified receipts and skips checkpoint rewriting. Evidence:
`provenance/COMPLETED_IDENTICAL_CHECKPOINT_STORAGE_V20.json`.
