# Shared-feature AdaSpot joint prototype

**Approximate reproduction / initial joint-training qualification.** User explicitly
requests both remaining OCR heads, gate and joint training in one intervention.
Original local paper3.2-3.3/recognition section and supplied author proposal4.1-4.8
inspected together with the qualified backbone, actual PARSeq model/decoder/tokenizer
and previous Mask-Align implementation. Decisions: `ADASpot-JOINT-001`.

This is a genuine shared-feature prototype: one trainable ResNet50/AG-UNet pass,
three Text/Kernel/Gap maps, 8x32 feature-grid light OCR with one two-stream transformer
layer plus exactly one NAR/cloze refinement, 16x64 feature-grid deep OCR with four
causal AR decoder layers, and a two-layer correctness gate. No crop is passed through
the PARSeq RGB image encoder. Deep inference runs only on escalated words; diagnostic
always-deep evaluation is kept separate and does not establish adaptive latency.

The proposal is an extension absent from the original paper; this path is entirely
isolated in `configs/approximate/adaspot_joint_v1.json`. Existing STRICT paths and
completed scientific source implementations/checkpoints are preserved. A new subclass
explicitly unfreezes the original encoder, including BN. A new shared Gap1x1 head
starts with zero weights/bias-2; the earlier separate32.6M Gap decoder is not part of
this model. Original Text/Kernel weights warm-start strictly from the hash-bound
source. Historical baseline remains the independently saved original model.

Preserve the qualified v4 RGB512/native256 segmentation supervision: M/K/G use the
same stride2 F64. Average-pool2 supplies stride4 F64 to light8x32, and deep16x64 uses
the higher-resolution original stride2 F64. This retains stride2 segmentation heads
instead of the proposal's stride4 heads, an explicit ASSUMPTION. Both feature grids
are projected64→384, fixed2D sine/cosine positions added and LayerNorm applied.
All geometry remains predicted-mask-only and detached; gradients flow through F.
The known negative RGB rectification proxy does not substitute for this experiment.

Qualified local official PARSeq runtime revision1902db043c029a7e03a3818c616c06600af574be,
archive SHA256f57289cb1fc7fe41f488a52ea48f8a7f114252ef405b0b64efc1d528c29b48ef and
checkpointbb5792a68e367476abca029cbf8699abc805f3d3dc7e57aae45c8ec4f7b7cd00 are verified
before import/load. Each head copies all26compatible decoder-layer0/norm, token
embedding, position-query and output-classifier tensors, with exact shape/dtype checks
and strict full-state loading after explicit merging. All fresh keys are listed.
Three new deep layers receive fresh independent initialization; these are not claimed
as pretrained AR4. Runtime dimensions384/12attentionheads/MLPratio4, dropout.1,
94character charset/max25labels are recorded. The source artifact remains qualified
PARSeq, not identified as the unresolved exact paper PARSeq-S.

Joint objective: Dice(M)+Dice(K)+Dice(G)+CE_light+CE_deep+BCE_gate. Light CE averages
first/refined predictions; deep CE uses teacher forcing with future-masked query and
content attention. Gate cues: minimum/mean refined character probabilities, first-
refined disagreement, mean predicted M/K confidence, normalized centerline curvature,
and log1p(character count / aspect ratio). All cues detach before the gate; the gate
cannot optimize OCR/backbone through its objective. Targets are actual casefold
correctness after one-to-one predicted/native-owner box matching (IoU>=.5), not
teacher-forced accuracy. Unmatched ROIs remain negative gate examples. GT never
forms inference masks, geometries, features, confidences or routes.

Training-only annotations map owner IDs to care transcription order from the original
target builder. Unsupported characters, empty/overlength labels are logged and excluded
only from CE. They are never silently stripped/truncated or omitted from evaluation GT.
Current orientation-independent radius3 GapGT and kernel distance fraction.5 are reused;
the proposal's same-line horizontal target remains unresolved.

The first complete fit is a **two-epoch qualification**, not converged scene pretraining:
deterministic hash split of officialTRAIN1255 into1155fit/100calibration, all1155fit
images once each epoch. The new components never optimize on these100calibration
images, although some were previously seen by the source M/K. No augmentation in this
initial controlled run. At most16predicted ROIs per training image are selected by
seeded sampling independent of GT, with every selected and unselected count logged.
Every evaluation ROI survives without area/cap filtering. All assumptions, limits and
scientific choice classifications are in the resolved approximate config.

Joint AdamW groups:1e-5backbone,1e-4newheads, decay1e-4, gradient normclip5; all loss
coefficients1 and Dice smoothing1. All are explicit ASSUMPTION, not original-paper
numerics. Verify nonzero finite CUDA gradients and actual parameter changes separately
for encoder, decoder, Text/Kernel, Gap, light, deep and gate. Save loss/ROI/CE-exclusion
coverage, full epoch IDs, optimizer/RNG checkpoint and strict checkpoint roundtrip.

Calibration reserves50images for scalar gate temperature and disjoint50for threshold
selection. Choose reject/keep/deep-accept thresholds from declared grid by bbox/casefold
Hmean, then fewer deep calls; final epoch fixed before TEST. Gate routing is keep if
p>=keep, reject ifp<reject, otherwise escalate; deep accepts if its minimum character
probability meets its calibrated threshold. Empty readings are not emitted. Explicitly
rejected proposals are removed from precision's emitted-prediction denominator, while
all original careGT/misses remain in recall and all per-ROI decisions remain exported.
Always-light/always-deep retain all care proposals including empty strings. This
denominator difference is reported, not hidden.

TEST evaluates all300images/2204careGT, retaining the two known duplicate RGBs in the
primary result and reporting the same298/2187secondary cohort separately. TEST is
historically explored. Matching is maximum one-to-one bboxIoU>=.5, exact Unicode
casefold with punctuation; not official polygons, Table5accuracy or strict reproduction.
Single-class gate AUROC is explicitly undefined, never fabricated. Gate AUROC/ECE,
rejection count and escalated fraction accompany Hmean. Independent audit recomputes
Hungarian counts, all calibration candidates, temperature NLL, full-fit populations,
seeded ROI choices, checkpoint/source hashes, routes and selective deep execution.

Actual RTX3090, FP32 imagebatch1, trainingROIcap16, evaluationROIchunks16. CUDA model
parameters/updates and inference; explicit CPU bilinear sampling backward with autograd
transfers for deterministic gradients. This is a HARDWARE_DEVIATION with CPU/transfer
costs, not fully GPU sampling backward. Record peak allocated/reservedVRAM, model set,
shape/batch/dtype and OOM. No mixed precision, installation, implicit CPU fallback,
8GBqualification or GFLOPs/latency superiority claim.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 \
  .venvs/context-str-main-py311/bin/python -m scripts.run_adaspot_joint \
  --mode approximate --stage smoke --config configs/approximate/adaspot_joint_v1.json \
  --source-run /explicit/audited/fulltrain-source --output-root /explicit/new/smoke-root --device cuda:0
# Same command with --stage full and a new output root completes both1155image epochs,
# held-out calibration and all300TESTimages. Smoke results are never full accuracy claims.
.venvs/context-str-main-py311/bin/python -m scripts.audit_adaspot_joint --output-root /explicit/completed/root
```

Technical sources: [official pinned PARSeq decoding implementation](https://github.com/baudm/parseq/blob/1902db043c029a7e03a3818c616c06600af574be/strhub/models/parseq/model.py),
[Guo et al., temperature scaling](https://arxiv.org/abs/1706.04599).
Original proposal's SynthText/TextOCR scene pretraining, multiple seeds, independent
benchmarks and robust accuracy gains remain unresolved. A working gradient/route test
does not establish recognition quality. All exported failure figure annotations must
be in English, preserving GT and OCR strings.

## Completed implementation, joint fit and negative accuracy result

Full run: `/extra1/bmartelli/context-str-runs/adaspot-joint-v1-20261021`.
Smoke: `/extra1/bmartelli/context-str-runs/adaspot-joint-smoke-v1-20261021`.
Smoke audit PASS precedes the full fit. All7tracked groups (encoder, decoder,
Text/Kernel, Gap, light, deep, gate) have nonzero finite CUDA gradients and actual
optimizer changes. The complete prototype has70212906trainable parameters.

Both1155image epochs completed:2310CUDAupdates,488.6s+428.0s. Initial per-image
loss15.17in the preceding smoke is distinct from full-fit meanloss7.5390→6.5670.
Mean lightCE3.1751→2.9364, deep teacher-forcedCE2.5811→2.2175. These losses do
not establish useful free-running OCR. Training selected12105/15160ROIs and
supervised4779/3553matched OCR word exposures in epochs1/2, respectively;
these are exposures, not unique words. No CE charset/length exclusion occurred
among these matched selectedROIs. Gate positive targets were0then1, out of
27265selected ROI exposures across bothepochs. New-head calibration remained
disjoint50temperature/50thresholdimages.

On all300TESTimages/2204careGT:

| New shared-feature mode | Exact localized+read words | Geometry matches | Emitted/care predictions | Hmean |
| --- | ---: | ---: | ---: | ---: |
| Always light NAR1+cloze | 0 | 1437 | 5053 | 0% |
| Always deep AR4 | 0 | 1437 | 5053 | 0% |
| Calibrated adaptive | 0 | 1437 | 5053 | 0% |

This is an **unsuccessful initial accuracy experiment**, despite a working
implementation and decreasing losses. Calibration had0correct words, so every
threshold candidate had0Hmean. The declared tie rule selected keep/reject/deep
thresholds0/0/0, emitting the light readings with0deep calls. Temperature is
.0497871; TEST has5053negative and0positive correctness labels. AUROC is explicitly
undefined. Tiny ECE/Brier in this degenerate all-wrong population is not useful
gate discrimination or successful adaptive OCR. Secondary298/2187also has0exact
words in everyarm. Do not present the0deep-call outcome as an accuracy/compute gain.

The earlier fixed Gap1 + frozen RGB PARSeq reference remains1311/2204correct,
1612geometry matches and2624care predictions, Hmean54.3082%. No checkpoint is
overwritten or replaced by the new prototype. The new model's additional proposals
and lower geometry matches also show that the joint fit degraded localization.
Neither that observation nor the OCR loss curve isolates a unique cause.

An independent fresh **actual calibrated selective** replay processes all300TEST
images with exactly300backbone calls and0deep calls, preserving all5053instances,
predicted maps/boxes, routes, strings and per-image/aggregate scores. Execution
receipt: `actual_selective_v1/independent_execution_audit.json`. Separately forced
keep/reject/escalate policies verify actual conditional deep invocation and raw
string replay, without selecting thresholds on TEST. Thus the failure is not a
discrepancy between cached scoring and the executable selective path.

After this fixed negative result, a separate **fit-only single-word memorization
sanity check** freezes the backbone/gate and caches real predicted feature grids.
Selection is the first supported matched predicted ROI in fixed fit-ID order:
`total_text:35`, instance1, owner2, transcription`ATHENS`, nativeboxIoU.663112.
Both heads learn this word by step10of100AdamW1e-3updates. At100steps, lightCE
.00001897/deepteacherCE.00002256, both greedy readings`ATHENS`. These updated
heads are not substituted into the full model or its scores. This verifies that
loss/token/decoder paths can memorize a known fit target; it does not establish
learned visual discrimination, correct annotation mapping across the dataset or
generalization. Diagnostic: `fit_word_sanity_v1/results.json`; configuration
`configs/approximate/adaspot_joint_overfit_check_v1.json`.

Two epochs on small scene data with fresh64→384adapters and three new deep layers
do not realize the proposal's SynthText/TextOCR pretraining. Missing pretraining,
insufficient recognition adaptation and joint-objective interference are plausible
hypotheses, not proven causal explanations. A next controlled study should first
qualify feature OCR on TRAIN with a frozen segmentation backbone and a genuine
held-out OCR set, establish both heads' free-running accuracy, then unfreeze for
joint fine-tuning and calibrate the gate only after positive/negative examples
exist. No TEST-driven optimizer/threshold changes were made in these runs.

Independent full audit PASS:2310training image/epoch cases,900TESTimage/arm cases,
all196threshold candidates and1905source/input hashes, actualCUDAupdates, strict
initialization and checkpoint roundtrip.65focused tests pass, including causal
teacher masks, detached gate gradients, charset handling, one-to-one matching,
mode/partition isolation, routing boundaries and controlled OOM propagation.
An initial unit test expected25copied tensors instead of26; the assertion was
corrected and all26pinned tensors checked exactly. No production check was relaxed.
The stale PTY handle encountered after user resume affected log retrieval only;
the completed artifacts and independent audit establish both epochs finished.

Actual RTX3090 FP32/imagebatch1/trainingROIs<=16: peak allocated2628188160bytes,
reserved2799697920bytes. All-head diagnostic inference peak928555520/1212153856bytes;
fresh actual selective replay601897472/832569344bytes. Light8x32 usesFstride4,
deep16x64usesFstride2,64channels each. No actual OOM or silent fallback. The
controlled unit-test OOM is synthetic error-reporting verification, not a run failure.
CPU deterministic sampling backward remains the explicit hardware deviation.
No8GB/generalized latency qualification follows from these RTX3090 measurements.

Full resultsSHA256:
`c99ad4f1f38fbca107e13c14e80025db4e2fe9ef3264c9e6f0e6926771094d7b`.
Joint checkpointSHA256:
`ec0cbd9c1949f86eebb77c85bd7ebfd33e48c858f0a39cc4fb0bd0e323393790`.

```bash
# Execute the learned selective network again on every source TEST image.
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 \
  .venvs/context-str-main-py311/bin/python -m scripts.evaluate_adaspot_joint_selective \
  --mode approximate --run-root /explicit/completed/joint-run \
  --output-root /explicit/new/selective-replay --device cuda:0
# Separate fit-only memorization sanity check; never substitute for benchmark scores.
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 \
  .venvs/context-str-main-py311/bin/python -m scripts.check_adaspot_joint_ocr_training \
  --mode approximate --config configs/approximate/adaspot_joint_overfit_check_v1.json \
  --run-root /explicit/completed/joint-run --output-root /explicit/new/fit-word-check --device cuda:0
```
