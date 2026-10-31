# Lightweight ICDAR2013 90% fixed-budget research

**Approximate reproduction / active research extension.** The user requires
lexical micro F1 at least90% on all233 ICDAR2013 TEST images and1095 care words,
with exactly61/233 DeepSolo decisions (26.18%). Keep Total-Text above78.2%
with150/300 (50.00%) fallback and avoid material ICDAR2015 degradation from
U-Net79.0943% and YOLO80.8403%, each228/500 (45.60%). Prefer lower inference cost.
The preceding common procedure and its original outputs remain preserved.
A later audited conditional-fusion result reaches 90.0094% on ICDAR2013; see the final section for the cross-dataset tradeoff and limitations.

Paper3.3/3.5 and existing lexical/word-risk/admission/fallback implementations
were inspected. This work extends the shared supervised procedure, rather than
claiming Algorithm2's training-free score. The unchanged primary metric is
per-image lexical multiset matching after Unicode casefold and alphanumeric
normalization, then micro TP/FP/FN. Native256 care/ignore, empty outputs and all
care targets remain counted. It is not official polygon Hmean or paper accuracy.

## Declared first experiment

Use real cached low-floor YOLO/PARSeq candidates, the existing TT U-Net expert
and the newly qualified text-encoder word U-Net. Compare YOLO alone, the latter
U-Net alone, and two explicitly named combined expert inputs. A combined branch
requires both light networks; its cost is not attributed to a free router.

For each raw reading, keep its literal text and any distinct existing qualified
distance<=1 dictionary alternative. Train a small word admission classifier from
FIT183 only, with the identical features/model grid for all source arms. Features
use token uncertainty, detector confidence, lexical evidence, geometry and
cross-source agreement; no image ID, annotation or fallback output is a feature.
Labels are normalized lexical set-membership proxies; the final metric retains
one-to-one multiplicities. Ignored candidates do not fit the classifier.

Pool the previous selection23 and CAL23 into46 validation images. Select one
model, probability cut and duplicate-overlap rule using light validation F1;
within0.1 F1 points prefer smaller model bytes. Preserve a complete trial table.
Construct five RGB-group FIT folds after freezing the admission recipe. Both the
vocabulary and admission model exclude each heldout fold's labels/images. The
newly fitted U-Net itself saw FIT183, so this does not establish end-to-end OOF.
No TEST label or trial is opened during this stage. The new shared procedure
has not yet proved cross-dataset nonregression or the90% target.

The next separately frozen stage will test reconciliation after routing:
preserve DeepSolo words and append only confidently admitted, spatially distinct
light words on images already sent to DeepSolo. The image gate may use only
pre-fallback observations. Every eventual result must report its logical
DeepSolo count, total and percentage; cached replay is not online latency.

Source: `recognition/lightweight_word_selection.py`,
`scripts/run_lightweight_word_selection.py`,
`tests/recognition/test_lightweight_word_selection.py` and
`configs/approximate/lightweight_word_selection_ic13_v1.json`.
External root:
`/extra1/bmartelli/context-str-runs/lightweight-ic13-90-20261029-v1`.
Existing environments only; no installation or new neural inference in stage1.

Classifications: PAPER_SPECIFIED broad localization/fallback motivation;
REPOSITORY_SPECIFIED authenticated experts, lexicon matcher and metric boundaries;
ASSUMPTION candidate alternatives, features, learned admission, finite selection,
FIT-only dictionary folds and validation pooling; HARDWARE_DEVIATION CPU cached
processing; UNRESOLVED upstream fitting/TEST exposure, selection uncertainty,
cross-dataset retention, generalization and attaining90%.

## First completed admission/router experiment

All four admission arms and28 image-head bundles per arm are fitted; each
admission arm retains six full-FIT candidates and five selected-recipe OOF
models. The full independent audit passes24 selected word refits,504 admission
trial recounts,732 OOF-image replays,1,832 branch recounts,352 image-router
trials, three selected image-head refits and184 annotation-free validation
deployment replays. Twenty focused admission/common/oracle tests pass. The first
smoke failure and corrected successful smoke remain preserved.

Validation selects the single text-encoder U-Net with an HGB word head, Ridge
image head and confidence>=.9 light complement after fallback, IoU<.1. It scores
90.6250% on46 validation images, logical DeepSolo12/46 (26.09%). The choice is
frozen before TEST. Complete TEST then scores TP892/FP121/FN203, **84.6300% F1
with61/233 (26.18%) DeepSolo decisions**. Only those61 cached fallback envelopes
are opened after the pre-fallback plan is saved. Light alone scores80.7992%,
DeepSolo0/233 (0.00%). This does not meet90%, and does not replace the earlier
qualified87.7275% mixed pipeline at61/233 (26.18%). The CPU admission/router
weights total126,880bytes, excluding the unchanged neural experts.

The existing TT/IC15 qualified configuration, model and audit hashes still
match; no new admission method is applied to those protected profiles in this
IC13 research test. That preservation is not evidence that the new admission
stage generalizes to all three datasets. The goal remains active.

The large validation-to-TEST gap supports the need to investigate upstream
expert generalization and selection uncertainty. Word/dictionary cross-fitting
alone did not remove the U-Net's FIT exposure. This is not an isolated causal
experiment attributing the failure to one mechanism.

## Declared fallback rereading follow-up

Before any new TEST procedure, generate real PARSeq readings on native DeepSolo
boundary envelopes in all229 TRAIN images only, at original-pixel padding0/2/4.
The raw cached detector profile remains ViTAEv2S+TextOCR1024/max1892, score>=.4.
No detector is newly run. PARSeq uses the qualified immutable checkpoint/source,
FP32, batches up to32 and original RGB crops. Record every crop hash and both
allocated/reserved CUDA peaks on the actual RTX3090; do not inherit3070 claims.
No GT or annotation is used in this inference preparation.

Subsequent policy selection will compare no change with confidence-qualified
PARSeq transcript replacement while preserving every DeepSolo instance/box.
At deployment rereading may run only after an image is routed to DeepSolo, so
the requested61-image budget remains explicit. This reuses PARSeq but adds crop
inference on those fallback images. Its benefit, exact common policy and actual
online latency remain unproven. Source: `recognition/fallback_reread.py`,
`scripts/reread_deepsolo_boxes.py`, explicit
`configs/approximate/fallback_parseq_reread_ic13_v1.json`.

The completed TRAIN preparation reads2,610 real PARSeq crops on229 images,
zero new DeepSolo calls. Peak allocated/reserved CUDA memory is180,920,832 /
276,824,064bytes on RTX3090; no OOM. The first launcher had a syntax error
before execution; its log is preserved and the corrected smoke/full run pass.
On all46 validation images, a fixed padding0/native-sequence>=.9 replacement
changes seven words and gains2 TP: standalone fallback F1 rises86.5031% to
87.7301%, each46/46 (100.00%) logical fallback. Those are standalone validation
controls, not the requested61-image TEST result. The finite61-policy grid and
independent counts are saved in `reread_policy_v1`.

Refitting the same image-head grid with this stronger fallback selects a
probability-only image route: validation91.3043%,12/46 (26.09%) fallback. No
additional TEST result is yet available for this follow-up. The original
84.6300% TEST trial remains preserved and is not overwritten.

## Lightweight detector control and supervised adaptation

An explicitly separate FAST-Tiny control uses the already installed docTR
model and qualified PARSeq. It is not called U-Net or strict reproduction.
The model has8,521,920 inference parameters after reparameterization, versus
13,535,296 before. All229 TRAIN images are inferred without annotations, first
detector1024/batch1 then PARSeq32x128/batch32, FP32. The2,864 real crop readings
include axis and rectified views. Detector memory peaks450,118,144 allocated /
603,979,776reserved bytes; reader peaks180,920,832 /272,629,760bytes. No OOM.
External pretrained dataset overlap remains unqualified.

The same admission grid gives77.3723% validation F1 for axis crops and77.3973%
for the confidence-selected two-view variant, each0/46 (0.00%) fallback.
Neither is promoted to TEST. This control does not support simply replacing
the light backbone with a smaller pretrained model for this benchmark.

The next controlled intervention keeps the existing YOLOv8s architecture and
640 input, so online model size does not increase. Fine-tune on FIT183 with
46 heldout TRAIN validation images, comparing head-only (modules0..9 including
buffers remain exactly frozen) with full-network optimization. Both run30
fixed epochs, FP32 physical/nominal batch8, AdamW lr.0003, text-safe augmentations
without reflection. Paint only ignored TRAIN pixels to128 with care override;
all validation RGB remains untouched. Word boxes use exclusive coordinates.
No TEST RGB or labels are used for training or checkpoint/policy selection.

The complete preloaded module bypasses the upstream intersection-based weight
transfer; explicit strict initialization verifies all355 state tensors with no
missing/unexpected/mismatched keys. The first training smoke exposed a retained
CPU-bound serialized loss object. Recreate that non-network loss after device
placement; network initialization remains strict and unchanged. Preserve the
failed run. Corrected one-epoch smoke passes both arms, including162 unchanged
head-only backbone tensors, with3,127,110,144 /3,904,897,024bytes peak
allocated/reserved in full training and1,324,253,184 /1,568,669,696 in head-only.

Full fitting completed under `yolo_word_adaptation_v1/full`. Preserve candidate
epochs5/10/20/30 for each arm and the unmodified source checkpoint. Detection
mAP is a training diagnostic; subsequent fixed PARSeq/crop/lexical validation
must choose the expert before TEST. `icdar13_yolo_word_selection_v1.json`
declares the source+eight candidates, padding0/2/4 and finite emission grid.
A trained61-image gate is still required after choosing a stronger expert.
The90% target and a coherent retained method on all three datasets remain open.

Both arms complete30 epochs. An independent state audit checks all eight saved
candidates against the original355-tensor architecture, verifies genuine learned
weight/bias updates and the unchanged head-only backbone. Full-network training
peaks3,292,592,128 allocated /3,976,200,192reserved bytes; head-only peaks
1,441,069,568 /1,793,064,960bytes. No OOM.

The fixed validation OCR grid chooses:

| Expert | Selected epoch | Validation lexical F1 % | DeepSolo images/total (%) |
| --- | ---: | ---: | --- |
| Original YOLO |0|87.6543|0/46 (0.00%)|
| Head-only adaptation |30|91.9614|0/46 (0.00%)|
| Full-network adaptation |20|88.9632|0/46 (0.00%)|

These are each arm's validation-selected operating points, not an isolated
fixed-threshold causal estimate or TEST result. The winner keeps original
backbone buffers and uses padding0, native sequence>=.9, joint score>=.8,
same-wordNMS.5, with no dictionary editing. Its counts are TP143/FP9/FN16 on159
care words. This is a promising stronger light expert, not the requested trained
61/233 (26.18%) full-TEST pipeline.

The next preparation is active under `yolo_oof_v1`: five RGB-grouped folds of
the original183 fitting images. Each starts from the original TextOCR checkpoint,
fits146/147 images for the already chosen30 head-only epochs, and validates only
the common46. Its own heldout36/37 images enter neither fitting nor checkpoint
validation. Export their genuine raw/emitted YOLO/PARSeq predictions from original
RGB, with no annotation use. Preserve a separate full-fitting seen-image control.
The global checkpoint/emission recipe is frozen before those folds.

This adds offline training, while deployment retains one detector with the same
640 input and architecture. The chosen vocabulary-free policy does not need GT
or a dictionary to emit words. Raw source/pretraining overlap and fold-size shift
remain unresolved. The subsequent router must be fitted, frozen and evaluated
on all233 TEST images with exactly61 fallback decisions before any90% claim.

## Declared fixed-emission count gate

Keep the selected detector/crop/emission rule unchanged; word-risk fitting must
not force a second word-admission stage. The shared gate consumes raw/emitted
words, aligned prediction-only traces, geometry and optional real map summaries.
Missing maps are explicitly marked unavailable; the adapted YOLO needs no U-Net,
CLIP or BLIP encoder. Word correctness supervision describes the emitted string
when editing occurred, otherwise the raw counterfactual for rejected hypotheses.
It remains a set-membership training proxy; final evaluation retains multiplicity.

Fit the same five-fold word-risk and two-count utility families around each
qualified domain expert. Compare seen versus image-excluded adapted expert inputs
on TRAIN/validation only. Native fallback, fixed PARSeq fallback rereading and a
finite confidence-qualified novel-word complement are eligible alternatives.
Reference/head/feature/ensemble selection and the exact batch quota rule are shared;
budgets stay dataset-specific. Prefer lower additional inference cost and smaller
CPU models within the declared validation tolerance. No TEST candidate search.

The primary IC13 supervision will reuse the historical TRAIN Generic resource
(common base plus all229 TRAIN transcripts), keep-unmatched, to model supplied
benchmark vocabulary assistance consistently with the released TEST Generic
condition. Its vocabulary includes the46 validation transcripts: validation
scores are conditional on that declared benchmark side information and are
not a vocabulary-independent generalization estimate. The light expert itself
remains vocabulary-free. Never use the released TEST dictionary as a TRAIN
teacher. Preserve a FIT-only-vocabulary native teacher as a diagnostic control,
excluded from primary method selection. Coverage/distribution equivalence of
the constructed TRAIN and released TEST resources remains unresolved.

This is an explicit ASSUMPTION using an existing REPOSITORY_SPECIFIED resource,
not strict paper reproduction. It changes the supervision condition relative to
the first failed admission trial; a later gain cannot be attributed entirely to
the new detector without the matched controls. Source:
`recognition/fixed_expert_budget.py`, `scripts/prepare_fixed_expert_budget.py`.


## Completed fixed-emission trial v3 (negative TEST)

All five30-epoch image-excluded detector fits and their heldout exports complete.
The shared fixed-emission gate compares seen/OOF sources and native/reread teachers,
with the same word-risk and two-count families. Independent audit passes12 word
model refits,16 selected count-head refits,4,580 branch recounts,10,620 trial
replays,2,509 original-RGB crop checks,687 fallback replays, five detector exclusion
checks and46 annotation-free deployment replays. The first audit invocation had
an incorrect vocabulary argument and failed explicitly; the corrected smoke/full
audits pass. Eight fixed-emission/core deployment tests pass.

Validation selects seen-source Ridge(raw/post/lexical), fixed fallback rereading
and word-risk>=.95 spatially distinct complements: TP152/FP5/FN7,96.2025%,
DeepSolo12/46 (26.09%). Freeze precedes every TEST fallback/annotation read.
Real TEST frontend inference covers233 images/1,573 PARSeq crops; only frozen61
fallback images are reread at padding0 (279 real crops). Complete TEST is
**TP901/FP59/FN194,87.6886% F1, DeepSolo61/233 (26.18%)**. All-light is82.2073%,
DeepSolo0/233 (0.00%). The90% target remains unmet. Do not promote over the
previous87.7275% mixed result, also61/233 (26.18%). No new DeepSolo neural calls
were made; this is logical cache routing with real new frontend/reread inference,
not measured online latency. TEST GPU peaks: detector115,835,392/144,703,488,
reader205,710,336/251,658,240 allocated/reserved bytes; no OOM.

The source-policy risk is visible on TRAIN without tuning on TEST: all-light
FIT seen93.3133% versus expert-image-excluded88.4942%, both0/183 (0.00%) DeepSolo;
validation91.9614%,0/46 (0.00%). The next declared test selects the existing
finite emission grid on all183 genuine image-excluded TRAIN outputs, with the
explicit benchmark TRAIN Generic assistance condition. Retain padding0 and the
chosen30-epoch detector; no new TEST policy search. This isolates a more broadly
supported operating-point estimate and allowed lexical assistance before further
neural training. Historical/repeated TEST exposure remains a limitation.


## Expert-OOF operating point and retained common procedure

A predeclared18-policy grid at fixed padding0 was evaluated on183 genuine
image-excluded expert outputs. The chosen rule uses joint detector/OCR confidence
at least.6, no extra sequence-confidence cutoff, rejects unmatched vocabulary
words and retains same-text NMS.5. Its image-excluded TRAIN F1 is91.6120%,
DeepSolo0/183 (0.00%), versus88.4942%,0/183 (0.00%) for the previous rule.
This comparison depends on supplied TRAIN Generic vocabulary side information,
which includes all229 TRAIN transcripts; it is not a blind vocabulary estimate.

The same fixed-emission gate family, including a probability-derived image score,
is trained with seen and expert-OOF observations. The frozen OOF choice uses the
lowest admitted-word correctness probability and optional fallback PARSeq rereads;
it uses no added map encoder or count-head model. Validation is TP155/FP7/FN4,
96.5732%, DeepSolo12/46 (26.09%). Independent audit passes12 word refits,
10,620 validation trial replays,4,580 branch recounts,2,509 crop-identity
checks,687 teacher replays, five expert-exclusion checks and46 deployment replays.
Each of the five actually used detector states is independently checked against
all355 original tensors, with102 changed weight/bias tensors and162 frozen
backbone tensors. Global OOF policy selection remains conditional, not nested CV.

After choices were frozen, complete TEST is **TP935/FP66/FN160,89.2176% F1,
DeepSolo61/233 (26.18%)**. Light alone is85.4823%, DeepSolo0/233 (0.00%).
The frontend reuses real previously generated233-image/1,573-crop YOLO/PARSeq
predictions; this trial adds303 real PARSeq rereads over only the61 routed cached
DeepSolo images and zero new DeepSolo neural calls. No online latency was measured.
Repeated historical TEST exploration limits any claim of independent generalization;
all methods, parameters and the61 decisions were frozen using TRAIN/validation
before this particular TEST evaluation. It does not meet90%.

The same fixed-emission/risk/count/conditional-complement procedure has also
been fitted on both versions of the other datasets, pooling selection and
calibration populations consistently with IC13. Dataset-specific experts and
DeepSolo variants are retained. Full TEST results are:

| Dataset | Light version | F1 % | Previous F1 % | Difference, points | DeepSolo images/total (%) |
| --- | --- | ---: | ---: | ---: | --- |
| Total-Text | U-Net | 80.3657 | 79.7119 | +0.6539 | 150/300 (50.00%) |
| Total-Text | YOLO | 80.6187 | 80.9388 | -0.3201 | 150/300 (50.00%) |
| ICDAR2015 | U-Net | 79.3203 | 79.0943 | +0.2260 | 228/500 (45.60%) |
| ICDAR2015 | YOLO | 81.4119 | 80.8403 | +0.5716 | 228/500 (45.60%) |

All exceed the requested Total-Text78.2% floor and both IC15 versions improve
on their previous points. YOLO versions require an additional U-Net predicted
map for the selected router; this is real inference cost even though this
retrospective study uses cached outputs. Each of the four TRAIN audits passes
six word-model refits,2,124 gate trials and every source/branch/deployment
replay. A separate TEST audit replays1,600 image scores and counts and756
selected fallbacks. Outputs are logical cached routing, not measured online
latency or new DeepSolo inference.

A further fixed-epoch final refit is running on all229 IC13 TRAIN images, using
the same pretrained initialization and head-only30-epoch recipe. The validation
shadow overlaps training and does not select an epoch. The emission rule and
expert-OOF-trained gate remain frozen; a checkpoint state audit and complete
TEST evaluation are still required. The final-refit proposal followed earlier
exploratory TEST results, so historical/repeated TEST exposure remains material.


The IC15 TRAIN Weak vocabulary is constructed from all1,000 official TRAIN
transcriptions, including the100 images pooled for validation; IC15 validation
therefore also assumes supplied benchmark-style vocabulary side information.
The released TEST Weak resource is used only for TEST inference. This vocabulary
condition, prior expert fitting and historical TEST use mean the comparison is
an approximate study, not a blind vocabulary-independent evaluation.

For the frozen v4 light expert and **native** DeepSolo fallback, an exact
privileged oracle reaches92.2430% at61/233 (26.18%), while the frozen trained
route on those same native branches reaches89.4084% at61/233 (26.18%). Computing
the oracle examines ground truth and cached fallback predictions for all233/233
images (100%); it is a nondeployable upper bound for those fixed branches, not a
measured61-call router. The deployed v4 branch with its preselected PARSeq
rereads measures89.2176%,61/233 (26.18%). The oracle is diagnostic only and is
not used to select a TEST method or operating point.


The predeclared all229-TRAIN refit completed30 epochs and passed the strict
355-tensor/162-frozen-tensor audit. All46 validation-shadow images were included
in fitting; no validation metric selected the checkpoint. On the complete TEST
set it yields TP938/FP75/FN157,88.9943%, DeepSolo61/233 (26.18%), worse than
the existing89.2176% at61/233 (26.18%). Its light-only result is84.9574%,
DeepSolo0/233 (0.00%); an independent audit replays233 decisions,1,551 light
PARSeq crops and286 selected fallback crops. Peak training allocation/reservation
is1,451,178,496/1,633,681,408bytes; no OOM. The new frontend takes17.36 seconds
for the233-image offline YOLO/PARSeq stage including model loading and I/O on
the actual RTX3090, excluding DeepSolo and online routing; this is not end-to-end
latency. The229-image refit is not promoted as an improvement.

A subsequent TRAIN-only selector study fits the same finite two-count models
with five image-grouped FIT folds, selects pooled heldout-FIT plus validation
F1 at separately scaled quotas, and compares the existing probability controls.
The source expert and word-risk folds are conditionally stacked rather than fully
nested; OOF scores must not be described as unbiased. No TEST output chooses the
router. Apply and verify the same selection procedure on the other datasets if
this selector becomes the proposed common pipeline.


The five-head crossfit selector is independently qualified and scores88.0941%
on complete ICDAR2013 TEST, with DeepSolo61/233 (26.18%). It reuses the original
light expert, adds358 real PARSeq fallback crops and makes no new DeepSolo neural
calls. Its audited pooled TRAIN-crossfit plus validation estimate was93.8824%,
DeepSolo60/229 (26.20%); the large discrepancy shows the danger of selecting
many heads on conditionally dependent folds. The separate TEST recount checks
all233 decisions,1,573 original and358 fallback crop identities, and all61
selected DeepSolo outputs. This selector does not improve on89.2176%,61/233
(26.18%). Historical and repeated TEST exposure preclude interpreting these
sequential experiments as a pristine holdout study.

The next selection is an explicitly predeclared, cost-aware one-standard-error
rule over the same independently audited TRAIN/validation trial grid. Bootstrap
resamples images, and eligible candidates must include a fitted image-count
head or a fitted word-risk model. Prefer fewer auxiliary map models, fewer
fallback PARSeq reads and lower CPU model complexity within one estimated
standard error of the best pooled point. This bootstrap is conditional on the
existing candidate search and non-nested expert/word features; it is not a
nested generalization confidence interval. The IC13 TRAIN/validation choice is
a trained expected-word-error gate and native fallback with a high-confidence
primary-word complement,93.2544% pooled at60/229 (26.20%) and92.5926% validation
at12/46 (26.09%). This is not a TEST score.

## Conservative gate and conditional fallback fusion: full TEST results

The same finite TRAIN/validation procedure applies the existing fitted word-risk
model to the light expert, uses a conservative one-standard-error preference for
retaining light words, freezes the exact image budget, then matches existing
light words to already selected DeepSolo boxes. On a qualified match, the frozen
policy may replace the DeepSolo transcript; the existing high-confidence,
spatially distinct light-word complement follows. The image gate has no access
to TEST truth or fallback output. The 113 KiB ICDAR2013 risk head runs on CPU;
the model has no additional U-Net map or additional fallback PARSeq crops.
All 37 fusion policies and every validation deployment were independently
replayed in each case before the TEST route was frozen.

| Complete TEST | Light expert | Lexical micro F1 % | Previous common gate % | DeepSolo images/total (%) |
| --- | --- | ---: | ---: | --- |
| ICDAR2013 | YOLO/PARSeq | 90.0094 | 89.5382 (prior risk gate) | 61/233 (26.18%) |
| ICDAR2015 | U-Net/PARSeq | 78.3711 | 79.3203 | 228/500 (45.60%) |
| ICDAR2015 | YOLO/PARSeq | 79.7308 | 81.4119 | 228/500 (45.60%) |
| Total-Text | U-Net/PARSeq | 80.0385 | 80.3657 | 150/300 (50.00%) |
| Total-Text | YOLO/PARSeq | 79.8271 | 80.6187 | 150/300 (50.00%) |

The ICDAR2013 result is TP 955, FP 72, FN 140, with all 1,095 care words and
233 images included. The routed images are identical to the prior risk gate;
five transcript replacements on five images each correct a word, increasing
TP from 950 to 955 and reducing both FP and FN by five. This exceeds 90% by
only 0.0094 percentage points. One additional false replacement of a correct
word would cross below 90%; this point estimate should not be treated as a
robust margin or a generalization guarantee. On ICDAR2015 the previous common
gate remains 0.9492/1.6811 F1 points higher (U-Net/YOLO), so the loss is
explicit even though this version is lighter and Total-Text exceeds 78.2%.

An independent TEST audit replays all 233 ICDAR2013 routes, 1,573 original
PARSeq crops and 61 selected fallback envelopes; another replays 1,600 TT/IC15
image scores/counts and 756 selected fallbacks. Both pass. The current results
are **approximate reproduction**, logical routing over authentic cached model
outputs; zero new DeepSolo calls were made, and no online latency was measured.
The exact quota is enforced by ranking the complete image batch. The ICDAR2013
frontend/decoder uses benchmark Generic vocabulary assistance, and TT/IC15 use
their qualified dataset-specific fallback resources. The reported metric is
per-image Unicode-casefolded, alphanumeric-only lexical multiset micro F1 with
native 256 care/ignore handling, not polygon Hmean or the paper's accuracy.
Historical TEST exploration, the TEST-motivated conservative criterion,
non-nested source/expert/word folds, reused validation, benchmark vocabulary
side information and upstream exposure prevent a pristine holdout claim.

Machine evidence: `test_fixed_fusion_v2/results.json` and
`test_fixed_fusion_v2/independent_test_audit_v1.json`, plus
`fallback_word_fusion_v2/independent_test_audit_v1.json` under the external
run root above. The frozen approximate evaluation config is
`configs/approximate/fixed_expert_budget_ic13_evaluation_fusion_v2.json`.

## Qualified no-extra-neural common selector

A subsequent common selection procedure considers three independently qualified
families for every dataset/expert case: the prior fixed-expert gate and the v1
and v2 risk/fusion gates. It admits only candidates requiring neither an
additional neural map model nor an additional fallback PARSeq pass, then chooses
the highest lexical micro F1 on the same validation population and quota. Exact
ties use the declared candidate order. The selected mechanisms can differ by
case: ICDAR2015 U-Net uses the fixed count head, while ICDAR2013 uses the v2
word-risk/fusion gate. Thus this is one common selection procedure with fitted
case-specific choices, not the same learned parameters or an identical routing
formula on every dataset. A missing eligible candidate is an explicit error.

| Complete TEST | Light expert | Lexical micro F1 % | Previous common gate % | DeepSolo images/total (%) |
| --- | --- | ---: | ---: | --- |
| ICDAR2013 | YOLO/PARSeq | 90.0094 | 89.5382 (prior risk gate) | 61/233 (26.18%) |
| ICDAR2015 | U-Net/PARSeq | 79.3203 | 79.3203 | 228/500 (45.60%) |
| ICDAR2015 | YOLO/PARSeq | 81.0254 | 81.4119 | 228/500 (45.60%) |
| Total-Text | U-Net/PARSeq | 80.3657 | 80.3657 | 150/300 (50.00%) |
| Total-Text | YOLO/PARSeq | 80.2219 | 80.6187 | 150/300 (50.00%) |

ICDAR2015 now exceeds the original qualified references of 79.0943% U-Net
and 80.8403% YOLO by 0.2260 and 0.1851 points, respectively. Its YOLO score is
0.3865 points below the later previous common gate, which required an additional
U-Net map. Total-Text is above 78.2% in both versions; its YOLO score is 0.3968
points below the previous common gate, which also required the additional map.
No added neural map or extra fallback reader is used by any selected case;
small fitted CPU models remain. The selected ICDAR2013 word-risk model is
115,008 bytes, excluding YOLO, PARSeq and DeepSolo themselves. This is an
architectural cost description, not measured end-to-end latency.

The selector's independent audit recounts 1,488 candidate-validation images.
A separate complete TEST audit covers ICDAR2013's 233 decisions, 1,573 source
crops and 61 selected fallback envelopes; selected TT/IC15 source audits replay
all decisions and counts, including a further independent 800-image, 378-selected
fallback replay for the two YOLO cases. All pass. The combined machine-readable
report is `validated_budget_variant_no_extra_v1/test_summary.json` under the
external run root. The pinned selector and reporting configurations are
`configs/approximate/validated_budget_variant_no_extra_v1.json` and
`configs/approximate/validated_budget_variant_no_extra_report_v1.json`. Runtime
selection loading is `recognition/validated_budget_variant.py`.

The earlier one-standard-error selector was also independently audited, but it
selected the prior ICDAR2013 gate: its complete TEST was 89.2176% with 61/233
(26.18%) DeepSolo decisions. The no-extra-neural rule was proposed afterward,
with earlier TEST results already known. Although its code reads only validation
counts to choose a candidate, both the research path and the rule design are
adaptive to historical TEST feedback and reuse of validation. Therefore these
are exploratory approximate results, not an unbiased independent holdout or a
strict reproduction. Full-batch ranking is needed to enforce each exact quota;
outputs here use previously generated cached inference, not a fresh end-to-end
run or a measured online latency. The ICDAR2013 point is only 0.0094 percentage
points above 90%, and the supplied benchmark vocabularies, non-nested expert
folds and upstream model exposure remain material limitations.

## Fresh selected-image DeepSolo verification

The ICDAR2013 route above was frozen before this follow-up. The qualified
ViTAEv2-S/TextOCR DeepSolo checkpoint was then actually run on exactly its 61
selected images: one smoke inference and 60 more, with the smoke image reused.
The input plan includes only image identifiers, paths and hashes; no annotation
paths are present. Full TEST recount using the frozen cached YOLO/PARSeq frontend
and the fresh selected DeepSolo results reproduces **90.009425%** lexical micro
F1: TP 955, FP 72, FN 140, 1,095 care words across 233 images, with DeepSolo
61/233 (26.18%). Per-image TP/FP/FN and final emitted text/boxes match the
previous cached run for all 233 images. This does not assert bit-identical raw
neural scores, or a guarantee under future nondeterministic GPU runs.

Independent audit passes all 233 decisions, 1,573 frontend crop hashes, 61
selected fallback replays and fresh-inference receipt checks. The run used the
existing isolated legacy environment on an NVIDIA RTX 3090, FP32, batch size 1,
with peak allocated/reserved CUDA memory 1,998,766,592/6,956,253,184 bytes; no
OOM was observed. The per-image measurements total 13.38 seconds inside the
DeepSolo subprocess, excluding setup, frontend, routing and complete end-to-end
execution; this is not an online pipeline latency measurement. The native
preprocessing had variable resized and padded image shapes, saved per image.
The other two datasets' final results still use qualified cached predictions.

The frozen export/evaluation configs are
`configs/approximate/fresh_deepsolo_ic13_selected_v1.json` and
`configs/approximate/fixed_expert_budget_ic13_evaluation_fresh_fusion_v1.json`.
Under the external run root, the fresh complete result and independent audit are
`test_fixed_fusion_fresh_v1/results.json` and
`test_fixed_fusion_fresh_v1/independent_test_audit_v1.json`; the combined
machine-readable report is
`validated_budget_variant_no_extra_v1/test_summary_fresh_v1.json`. All earlier
limitations, including adaptive prior TEST exploration, the benchmark vocabulary
and the narrow 0.0094-point margin, still apply.
