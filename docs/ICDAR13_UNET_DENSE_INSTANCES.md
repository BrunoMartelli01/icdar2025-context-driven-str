# ICDAR2013: U-Net dense word instances toward the 61-image fallback target

**Approximate reproduction / research extension.** User target: punctuation-free
lexical word micro-F1 at least90%, complete TEST233/1095 care words, and at most
61/233 DeepSolo images (26.18%). No result is claimed before completed inference
and independent audit. STRICT remains unchanged.

## Evidence and new factor

Read local paper3.2/3.3/3.5, the qualified native refit, the completed matched
loss controls, the frozen Generic fallback and the existing word-slot decoder
before changing the scientific component. The existing global slot architecture
has already been trained on Total-Text, and its selected branch did not improve
the prior control. Do not call a repeat implementation a new architectural idea.

This experiment keeps the complete pretrained ResNet50 Attention-Gated U-Net
and adds a spatially aligned dense center/extent/offset decoder. It regresses a
complete word box from a predicted word center, avoiding the requirement that
all word pixels form the correct connected component. This representation is
inspired by [Objects as Points](https://arxiv.org/abs/1904.07850), rather than an
exact reproduction of its backbone, targets, loss, augmentation or schedule.
The closest local center-offset experiment predicted text-pixel votes, not a
complete word extent at each center; the closest slot experiment used global
queries and Hungarian assignment. Both negative histories remain intact.

The frozen source is the complete qualified TT Text+Kernel state
SHA25636cd330376b53b5df6a717025bc59be2320b4879250b46b275ec526822e030ca.
Its original RGB768/384 features are average-pooled to192. The new head receives
all64 feature channels and two text/kernel probability channels, in float32.
No YOLO, CLIP or BLIP model enters this light branch. The head alone is trained
first, so any gain concerns the output representation and this training recipe,
not a newly trained backbone. Joint U-Net adaptation is a subsequent experiment.

## Fixed training and evaluation boundary

Reuse IC13 FIT183/selection23/CAL23/TEST233 and the existing checksum-qualified
TT1130 fitting cohort. Original word-owner envelopes supply complete word boxes.
Every center is retained even if the word is smaller than one feature-grid cell;
a center collision is an explicit preparation failure. TT has8302 care-owner
instances, IC13 TRAIN229 has849. Exact RGB overlap with IC13 TEST is rejected.
No TEST transcript is used in training, feature extraction or policy selection.

Fresh96-channel spatial head, three dilated convolutions [1,2,4], GroupNorm,
center focal loss, log-size/offset regression and aligned GIoU. Average losses
per image; IC13 fitting images receive explicit loss weight4. Train40 complete
epochs, AdamW3e-4/decay1e-4, batch8, seed20261029. Head-only feature-cache
training has no geometric augmentation in this first control. Compare all
declared validation epochs using fixed peak.1/NMS.5/pad2/top50/PARSeq full
sequence probability.7, retaining the earliest exact F1 tie. Subsequent finite
selection policy tests must be frozen before CAL/TEST. Do not promote TEST-best
thresholds or silently replace the prior expert.

The primary metric is per-image lexical multiset matching, Unicode casefold
and alphanumeric filtering, then micro TP/FP/FN aggregation. All care GT and
all233 images remain, including empty or symbol-only GT. Native256 polygon
ignore/care override is unchanged. Geometry IoU, if reported, is diagnostic;
it does not replace this metric. Explicit no-alphanumeric predictions are
rejected by the light emission policy, with raw readings preserved.

After a successful expert experiment, check a privileged image oracle at
61/233 DeepSolo choices (26.18%) and then fit a GT-free router on TRAIN-derived
observations. Reuse only the fixed IC13 Generic fallback TEST outputs. Every
reported router value must include DeepSolo count, total and percentage.
Cached logical decisions do not measure online invocation count or speed.

## Provenance and runtime

Entry points: `models/unet_dense_instances.py`,
`segmentation/unet_dense_instances.py`,
`scripts/run_icdar13_dense_instances.py`.
Explicit profile: `configs/approximate/icdar13_unet_dense_instances_v1.json`.
External root: `/extra1/bmartelli/context-str-runs/icdar13-unet-dense-instances-20261029-v1`.

Source initialization copies all keys strictly; trained heads have their own
state schema. Feature caches bind RGB and source checkpoint hashes and contain
no target inputs. Target masks and box geometry enter the loss only. Preserve
raw per-image crops/readings/counts, complete fitting orders, failures, selected
weights, feature receipts and source snapshots. Save only small new-head
checkpoints; the source checkpoint is referenced by hash.

Observed hardware is an RTX3090/24GB, FP32, deterministic CUDA/no TF32,
batch1 feature extraction, batch8 head fitting and PARSeq32. Record allocated
and reserved peaks with shapes and model sets, and propagate OOM/errors.
The source features use lossless float32 compression; this is not 8GB or online
latency qualification. Disk preflight protects existing artifacts; no old run
or dataset is deleted. The initial read-only target probe computed the expected
counts and zero192-grid collisions but failed to serialize a NumPy integer in
its multi-grid report. The persistent preparation converts scalar geometry to
Python lists, checks every center again and preserves that failure explicitly.

Classification: PAPER_SPECIFIED architecture/fallback motivation;
REPOSITORY_SPECIFIED source state/targets/preprocessing and genuine OCR;
ASSUMPTION dense center/extent representation, losses, fit recipe and metric;
HARDWARE_DEVIATION cache/sequential FP32 execution on RTX3090; UNRESOLVED
efficacy, prior TEST exploration, small validation and strict-paper equivalence.

## Executed qualification and current run

Eighteen focused dense-target/gradient/frame/lexicon/budget tests pass. The real
32-image smoke verifies every177 owner center, performs three real head updates
and three genuine PARSeq crop readings. Its end-to-end policy smoke replays27
selection policies and20 per-image expert counts independently; every fitting
order, source/cache identity and the unchanged heldout label scope pass.
These are engineering smoke results, not an accuracy estimate.

The full1359-image training cache completed: TT1130/8302 owner words and
IC13TRAIN229/849, no lost/colliding192-grid centers or IC13TEST RGB overlap.
Lossless FP32 caches bind every RGB and complete source checkpoint; actual
extraction uses batch1 and the selected768/192 geometry. No DeepSolo neural
inference is added.

Full training initially stopped at epoch2 because the epoch-history JSON used
the immutable artifact writer. The complete epoch2 model/optimizer/history
checkpoint is preserved as `full/failed_epoch2_checkpoint.pt`, along with the
original code, traceback and failed supervisor state. The v2 executable changes
only atomic replacement of that mutable history and unique failure filenames;
model/data/loss/optimizer/schedule are unchanged. The exact diff and checkpoint
hash are in `writer_fix_v2.patch` and `resume_v2_receipt.json`. Training resumes
from epoch2 under `training_source_v2`; epoch3 and subsequent history updates
have completed. This is an operational writer repair, not a new scientific arm.

The separate policy profile `configs/approximate/icdar13_dense_policy_v1.json`
predeclares27 peak/sequence/lexicon choices before completed full head results.
TRAIN Generic is constructed from the common base plus **FIT-only** care
transcriptions. Selection/CAL/TEST labels are excluded from that construction;
TEST uses the fixed released Generic dictionary. Policy code and its full
source snapshot are separately frozen under `evaluation_source_v1`.

The full40-epoch frozen-feature neural control is complete. Its earliest best
fixed-policy epoch25 scores31.8966% on selection23, DeepSolo0/23 (0.00%).
The predeclared27-case lexical policy grid selects source rejection, peak.1 and
sequence.7, giving36.0000% (TP36/FP100/FN28), DeepSolo0/23 (0.00%). The qualified
corrected U-Net pipeline on the same selection cohort scores75.7576% without
additional Generic correction, DeepSolo0/23 (0.00%); the FIT-only Generic keep
and reject controls score71.2121% and67.2131%, both DeepSolo0/23 (0.00%).
These are pipeline comparisons, not a matched causal architecture effect.

The frozen dense head is a **negative validation control and is not promoted
to full TEST**. Its source cache, every40-epoch update/order, writer-only
checkpoint resume and207 per-image selection counts pass the independent audit.
Preserve `full/validation_rejection.json` and `full/training_independent_audit.json`.
The90% target with61/233 DeepSolo images (26.18%) remains unmet.

## Preparation of joint adaptation and a text-pretrained encoder

The frozen-feature experiment has localized many text fragments rather than
whole words in selection previews. At epoch5,48/64 selection GT words have a
raw bboxIoU>=.5 proposal, but only22 have a correctly read viable raw proposal,
with DeepSolo0/23 (0.00%). This is a privileged TRAIN-selection diagnostic, not
the primary lexical F1 or an exact causal partition. Preserve the complete
40-epoch control; do not change its loss/schedule using these observations.

The next prepared model wrapper, `models/joint_word_instance_unet.py`, propagates
dense-instance loss into the U-Net decoder while preserving a frozen encoder.
Its real CPU backward fixture confirms nonzero last-decoder gradients and no
encoder gradients. The joint training recipe/efficacy still requires its own
resolved plan, CUDA smoke, completed run and independent audit.

A second initialization is qualified separately: copy only the265 functional
ResNet50 tensors from the existing official DeepSolo R50 Total-Text checkpoint
ac2a01b3 into the U-Net encoder. No transformer, detection or recognition
decoder is transferred or run by the U-Net. Every functional key is required;
missing/unexpected/changed-shape keys fail. FrozenBN parameters/running values
map to eval-mode torchvision BN; unused counters are explicitly generated and
the unused classification layer is recorded separately. All state loading is
strict.

Source normalization, RGB ordering, stride location and stage dilation match.
Three FIT images at256 square produce15 cross-environment feature maps with
**zero measured absolute error**, using the existing isolated CPU environments.
Two mapping rejection/completeness tests pass. No package or artifact download
occurs. Original and converted states, full key map, inputs, source/code/config
hashes and receipts are under `encoder_transfer_qualification/`; profile:
`configs/approximate/icdar13_text_encoder_qualification_v1.json`.

The first donor-source check incorrectly assumed a Git checkout and failed.
The existing host is a qualified commit archive, so the corrected run explicitly
uses the canonical commit-marker/archive-SHA verifier. Both failed and corrected
logs/configurations are retained. This does not relax source identity checks.
The donor's known training includes IC13 TRAIN, including images reserved for
new-head validation. That exposure must be reported; the validation is not a
fresh encoder holdout. The source protocol does not train on official IC13 TEST.
The transfer is an ASSUMPTION research branch, not a strict-paper checkpoint
choice or evidence of improved U-Net accuracy.

## Predeclared matched word-U-Net adaptation

The next complete model comparison is configured in
`configs/approximate/icdar13_text_encoder_word_training_v1.json` and
`scripts/train_icdar13_text_encoder_words.py`. Compare the original qualified
ImageNet encoder with the exact qualified text-pretrained encoder, retaining
the same complete initial attention decoder and text/kernel prediction heads.
Both encoders stay frozen; the complete U-Net decoder and segmentation heads
train. This tests the stronger visual initialization directly on word masks;
the prepared joint dense-head wrapper is a separate later intervention.

Each arm uses **every TT1130 and IC13FIT183 image each epoch**, matched order and
seed, original full-word/kernel BCE+Dice, IC13 loss weight4, AdamW1e-4/decay1e-4,
30 epochs, no augmentation in this first initialization comparison. Actual
shared-GPU placement uses physical1/accumulation8, FP32/RGB768/native384,
PARSeq32; qualify backward/memory before complete training. Retain source
epoch0 and evaluate every5 epochs. Fixed selection policy: text.5/kernel.5,
padding4/top50/PARSeq full-sequence probability.9, followed by no-alphanumeric
emission; preserve all care GT. Each arm selects its earliest best validation
epoch. No new TEST prediction or90% claim is authorized by a smoke result.

This control changes the encoder initialization relative to its matched arm,
not to a historical model with different optimizer/data/training exposure.
The exact original ImageNet U-Net and negative frozen dense-head experiment
remain intact. Donor pretraining exposure to IC13 TRAIN is explicit, and the
complete official TEST233 remains excluded from all new fitting and choice.

Both physical1 smokes pass independent encoder-state/update/order/word-count
audits. The first word-U-Net smoke stopped because the reused inference helper
required a missing `hardware.model_set` field in the new plan; preserve
`text_encoder_word_training/smoke` and its traceback. The corrected
`smoke_v2` schema passes without changing data, loss or model behavior.

The explicit full-run hardware profile is
`configs/approximate/icdar13_text_encoder_word_training_b4_v2.json`:
physical4/accumulation2 retains effective8 and165 complete optimizer steps per
1313-image epoch. This changes decoder BatchNorm statistics relative to the
retained physical1 smoke, so the profiles are separate. Both physical4 real
backward smokes pass; independent checks confirm all320 encoder state tensors
unchanged, actual decoder updates and every count/order/choice. Maximum smoke
allocated/reserved memory is4,946,950,144/6,241,124,352bytes on RTX3090.
This does not qualify RTX3070/8GB or online latency.

The matched30-epoch full run is prepared under
`text_encoder_word_training/full_b4`, with immutable source/config/inputs.
The durable `supervisor_b4.py` runs text-encoder and ImageNet-control arms,
then their independent audit; check the actual PID and
`pipeline_status_b4.json` before resuming. This launch establishes no new
accuracy result, router result or attainment of90% with61/233 (26.18%) fallback.

## Completed text-encoder arm and frozen downstream evaluation

The text-encoder arm has completed all30 epochs. Under the original fixed
selection policy its earliest best epoch30 has TP56/FP7/FN8 over all64 care
words in23 selection images: lexical micro-F1 **88.188976%**, DeepSolo0/23
(0.00%). The matched ImageNet control is also complete: its best fixed-policy
epoch20 has TP56/FP12/FN8, F1 **84.848485%**, DeepSolo0/23 (0.00%). The observed
validation difference is3.340491 percentage points, from five fewer FP with
the same TP/FN. This is TRAIN validation, not the233-image TEST or a router
result, and not a statistical generalization guarantee.

The independent full training audit passes both30-epoch runs: every fitting
order/step, actual decoder updates, all320 unchanged encoder state tensors per
arm and322 per-image validation replays. Maximum allocated/reserved memory is
4,951,144,448/6,461,325,312bytes on RTX3090. The complete immutable source,
configuration, selected weights and audit remain under `full_b4`.

The original supervisor and recorded control process were both absent when
rechecked, after the control's completed epoch5; no terminal failure traceback
or cause was recorded. Preserve `interruption_b4_20261029.json`, the old status,
last-checkpoint digest and completed text-arm evidence. The existing trainer
resumes the same control model, optimizer and history with epoch-seeded order;
no scientific profile changes. An uninterrupted-trajectory equivalence claim
is not made. `supervisor_resume_b4_v2.py` is detached from the conversational
session and continues the control followed by its independent audit. Inspect
the actual process as well as the status before taking any recovery action.

The downstream profile is now frozen in
`configs/approximate/icdar13_text_word_policy_v1.json`. It compares each arm's
already selected checkpoint on selection23 only: text thresholds [.3,.5,.7],
RGB padding [2,4,8], native sequence cuts [.7,.9,.95], and lexical choices
[none,keep,reject], for162 declared points. Kernel threshold.5, native384 and
top50 are unchanged. Ties favor fewer normalized lexical edits, then declared
order. The full experiment's explicit validation promotion floor is80%; this
is an experimental guard above the previous pipeline's75.7576% on this cohort,
not a statistical guarantee. A failed guard stops before new full TEST inference.

TRAIN vocabulary uses the common base plus FIT183 transcripts only; selection,
CAL and TEST labels are excluded from dictionary construction. TEST uses the
fixed released Generic resources. Native sequence probability includes EOS;
lexical edits preserve original token uncertainty, boxes and candidate identity.
The deployment policy receives only predicted readings and the lexicon.

`scripts/evaluate_icdar13_text_encoder_words.py` freezes the winner before any
CAL/TEST scoring, runs the selected U-Net/PARSeq branch on all configured roles
from RGB only, and saves complete raw/emitted words and predicted maps. It also
saves192 shared decoder observations (64 channel means, standard deviations and
text-weighted means), without adding CLIP, BLIP or another vision encoder.
These are prepared router inputs, not a newly fitted router. FIT predictions
remain in-sample for this decoder and must not be relabeled OOF training data.

The complete TEST result is followed by an exact61/233 (26.18%) privileged
image oracle using the unchanged Generic DeepSolo outputs. The oracle includes
all233 images and1095 care words, independently certified with rational
arithmetic; it is an upper bound using GT, never a deployable gate. A separate
full fallback recount confirms TP977/FP95/FN118, F1 90.170743%, DeepSolo233/233
(100.00%). The oracle makes logical choices on cached outputs, not measured
online DeepSolo calls or latency.

`scripts/audit_icdar13_text_word_policy.py` independently reconstructs the
FIT-only dictionary, every declared selection count, RGB/map/crop hashes,
lexical decisions and full expert totals. Its full mode also requires exact
selected-checkpoint neural map replays on selection images, full fallback
evidence recounts and rational oracle certificates. The real reduced smoke
passes:12 policy points,179 RGB crop hashes,20 complete per-image expert counts,
four exact neural-map replays and four fallback count checks. Six focused tests
pass, including the three new native-sequence/lexical/selection boundary tests.
The first two new tests failed because their fixtures omitted genuine PARSeq
token log-probabilities; complete the fixtures and retain the production schema.
No model or evaluator error was suppressed. Smoke accuracy is not promoted.

The immutable full evaluation is under
`text_encoder_word_training/expert_policy_v1`. Its detached
`supervisor_expert_v1.py` waits for the verified training/audit completion, then
runs selection, independent selection audit, full inference, oracle and full
audit sequentially. A missing training supervisor without verified completion
is an explicit failure and does not trigger an automatic restart. The final
state distinguishes `router_fit_required` from `expert_improvement_required`;
neither establishes90% with a trained GT-free router. Check
`expert_pipeline_v1.json` and the actual PID before continuing.

## Fold-safe TRAIN teacher preparation

The future gate requires a teacher whose vocabulary construction respects its
heldout decoder folds. The historical IC13 teacher used the released TEST
Generic resource also on TRAIN; do not reuse those TRAIN counts for the new
fold-safe experiment. The fixed TEST fallback remains unchanged.

`configs/approximate/icdar13_fold_safe_teachers_v1.json` and
`scripts/prepare_icdar13_fold_teachers.py` now prepare the supervision on CPU
from authenticated existing raw envelopes. The complete model/resize/score
profile is retained: ViTAEv2-S+TextOCR,1024/max1892, detector.4, no character
pruning, keep unmatched. No new DeepSolo neural call or package is added.

There are six explicit vocabularies. The full-fitting vocabulary uses the
common base plus FIT183 labels and predicts all229 TRAIN images; each of five
RGB-grouped FIT folds uses the common base plus only the remaining146/147
fitting images and predicts its heldout36/37 images. Literal spaces in labels
are preserved in JSON pair mappings. Selection/CAL labels and TEST vocabulary
bytes/predictions never enter dictionary construction. Teacher GT is accessed
after prediction only for supervised counts, and every care word remains.
Known upstream teacher/encoder exposure to IC13 TRAIN remains explicit: these
are decoder-fold boundaries, not fresh pretrained-backbone holdouts.

All412 scope/image outcomes pass raw payload and protocol-binding checks,
complete crop-envelope/lexical replay and independent Counter recounts. The
three new tests pass for out-of-scope label rejection, literal-space mapping,
payload/image/protocol corruption, failed raw inference, inclusive score
boundaries, unmatched retention and duplicate detection identities. The full
FIT-only teacher differs from the old Generic teacher in47/229 per-image count
records; preserving the new teacher identity therefore matters. These are
supervision controls, not a measured routing gain. Cached standalone teacher
choices are229/229 (100.00%) in the full TRAIN scope,37/37 (100.00%) for three
folds and36/36 (100.00%) for two; actual new recognizer calls are zero.

Resolved scopes, dictionary bytes, raw identities and independent audit are
under `text_encoder_word_training/fold_safe_teachers_v1`. No OOF light decoder
or router has been trained by this preparation alone. Promote this supervision
only with matching fold-safe light outputs and a validation-frozen gate.
