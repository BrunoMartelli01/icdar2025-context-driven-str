# Token image routing with the qualified corrected U-Net

This user-authorized experiment is an **approximate reproduction / research
extension**. It replaces the previous YOLO/PARSeq light expert with the strongest
completed corrected U-Net reference available in the inspected evidence. It
fits new token gates using the new words and RGB crops; it does not reuse YOLO
gate scores as U-Net scores. STRICT/default configurations remain unchanged.

## Frozen reference and controlled comparison

The starting expert is the [grounded recovery](GROUNDED_WORD_RECOVERY.md) text-edit
endpoint: qualified Text/Kernel U-Net, native PARSeq, existing GNN selection,
RGB validity verifier and a byte-level text editor. Its historical source result
is 62.895928% spatial/exact F1 on 300 TEST images, with DeepSolo 0/300 (0%). That
number retains punctuation and every output, unlike the current token metric.
It is a frozen historical reference, not a new TEST-selected winner or proof of
global optimality. No missing-letter reconstruction claim follows from the source.

Complete source observations for 1028 FIT, 100 calibration, 125 reused evaluation
and 300 TEST come from the audited minimum-fallback replay. The original verifier
prefix is reconstructed exactly as an uncorrected control. Each variant receives
its own newly trained token head with identical initialization per fold, optimizer,
budget and grouped folds. The corrected arm is primary in advance.

Native PARSeq confidence is inherited from the original crop after an edit; it
is not presented as confidence in the corrected transcript. The source RGB
verifier's joint score fills the former detector-score feature and is identified
as such. Source boxes and native crop pixels are preserved. The canonical source
crop hash includes shape and RGB bytes.

## Tokens, training and thresholds

The unchanged token architecture uses 980 CLIP patch vectors from the global
image/four tiles, 32 BLIP-2 visual queries, every current OCR text/crop embedding,
geometry and 18 scalar descriptors. Only image-derived global context is reused
after exact sample, RGB and qualified feature-hash checks. New CLIP text/crop
embeddings describe the actual U-Net variants. No YOLO words, transcript
likelihoods, dense-transport summaries, GT or actual DeepSolo outputs enter
inference features.

Five RGB-grouped folds cover 1253 usable TRAIN images; known exact TEST copies
TRAIN81/212 remain excluded from gate fitting. Within each variant, select among
epochs 5/10/20 using OOF final F1 at 50% bypass, then refit the chosen epoch on all
usable TRAIN. The corrected expert's pre-existing TRAIN/calibration exposure
remains; gate OOF does not make those experts unseen. No new TEST features or
outputs are used to choose a gate epoch.

The head predicts light-minus-DeepSolo TP and output-count differences. Its
score remains `2*predicted_deltaTP - fitting_DeepSolo_F1*predicted_deltaN`.
Report predeclared batch acceptance quotas 0/10/20/30/40/50%, plus scalar thresholds
derived from TRAIN OOF quantiles. Scalar cutoffs are not calibrated probabilities
or guarantees of unchanged coverage on TEST. Every row includes DeepSolo image
count, evaluated total and percentage.

## Metric and fallback boundary

Use the previous token comparison's exact per-image word-multiset micro F1:
Unicode casefold then `isalnum`, no spatial IoU, original native256 care/ignore
handling, all care words and unmatched/empty outputs retained. Empty normalized
strings never match. The complete TEST denominator is 300 images / 2204 care words.
This is not official polygon spotting, paper Table 5 accuracy or strict reproduction.

Gate targets use the original frozen DeepSolo profile available on full TRAIN.
TEST reports both that comparator and the separately TRAIN-selected native-only
optimized profile. The latter is a frozen fallback-transfer result; missing
low-floor TRAIN outputs are not fabricated. No new DeepSolo inference occurs.
Routing counts are logical choices over saved outputs, not measured online
latency or demonstrated compute savings.

## Completed results

The follow-up [threshold-transfer and marginal-loss analysis](UNET_CORRECTED_TOKEN_THRESHOLDS.md)
separates head-dependent score shifts, expert generalization and the exact
30-image contribution to the 40%-to-50% acceptance decline.

Full run `full_v2` completed, using immutable `source_snapshot_v4`. Both paired
variants selected **epoch 5** using TRAIN OOF results, then refitted on all usable
TRAIN before new TEST features were computed. The head has 220,035 parameters.
The table uses the previously TRAIN-selected `native_word` fallback: native
DeepSolo, instance threshold 0.35 and character-confidence threshold 0.925.
Its transfer into this gate remains explicit; gate targets used `original`.

| U-Net image acceptance quota | Corrected U-Net token F1 % | Earlier YOLO token F1 %, descriptive | DeepSolo images / total (%) for both |
| ---: | ---: | ---: | ---: |
| 0% | 82.6540 | 82.6540 | 300/300 (100.00%) |
| 10% | 82.2746 | 82.7684 | 270/300 (90.00%) |
| 20% | 81.9570 | 82.4615 | 240/300 (80.00%) |
| 30% | 81.2981 | 82.0440 | 210/300 (70.00%) |
| 40% | 80.0966 | 81.6076 | 180/300 (60.00%) |
| 50% | 78.4428 | 80.8248 | 150/300 (50.00%) |

The earlier YOLO curve is a descriptive reference: its original gate used
punctuation-retaining training targets. This is not an isolated architecture or
correction ablation. No TEST operating point is promoted as a new policy.

Actual scalar cutoffs below are fixed using the selected epoch's TRAIN OOF
scores. Accept U-Net when its score is greater than or equal to the threshold;
otherwise use DeepSolo. These are utility scores, not probabilities.

| TRAIN acceptance target | Fixed score threshold | TEST F1 % | DeepSolo images / total (%) |
| ---: | ---: | ---: | ---: |
| 10% | 1.888248 | 82.6818 | 299/300 (99.67%) |
| 20% | 1.386984 | 82.7406 | 295/300 (98.33%) |
| 30% | 0.995257 | 82.5012 | 285/300 (95.00%) |
| 40% | 0.599553 | 82.2746 | 273/300 (91.00%) |
| 50% | 0.222338 | 82.1173 | 246/300 (82.00%) |

These cutoffs transfer substantially less acceptance than their TRAIN targets.
For example, the nominal 20% cutoff accepts only 5/300 TEST images. Its F1 gain
over all-native DeepSolo is 0.0866 points, with DeepSolo 295/300 (98.33%), versus
82.6540% with DeepSolo 300/300 (100%). A 1000-draw paired RGB-cluster bootstrap
gives a conditional delta interval of [-0.1028, +0.3098] points. It does not
establish a robust accuracy improvement or useful online speedup. Intervals
condition on these trained models and routes and omit historical search and
training variation. The same routes and both variants with the original
DeepSolo fallback remain in the machine-readable completion record.

## What the corrections and router actually achieve

The corrected standalone expert has TP 1308, FP 438, FN 896, precision 74.9141%,
recall 59.3466% and lexical F1 **66.2278%**, DeepSolo **0/300 (0%)**. It emits 1774
observations, of which 28 are ignored under the inherited native256 rule. The
verified prefix before the editor has identical counts. All 47 edited TEST
strings preserve the normalized alphanumeric content, and normalized box/word
multisets agree on all 300 images. Therefore these corrections bring no direct
gain under the user's symbol-free comparison.

The two trained gates can still differ: their raw CLIP text embeddings and
training histories differ. Their TEST routing difference must not be described
as repaired alphanumeric recognition. At 50% acceptance, the corrected gate
scores 78.4428% versus 76.1693% before edits, with DeepSolo 150/300 (50%) for both.
This is a single paired run, not evidence of a general correction benefit.

There is useful expert complementarity that the learned gate misses. Relative
to the all-native DeepSolo F1, using U-Net has positive marginal utility on 47
TEST images, negative utility on 180 and zero utility on 73. A separate exact
GT-oracle diagnostic, restricted to the same whole-image outputs and eligibility,
reaches **84.3713%**, DeepSolo **256/300 (85.33%)**. At the predeclared 50% quota,
the oracle reaches **83.6555%**, DeepSolo **150/300 (50%)**, while the learned gate
reaches 78.4428% with the same DeepSolo 150/300 (50%). The exact ratio solver was
checked against exhaustive enumeration on eight seven-image fixtures at every
cardinality. This is a TEST-GT upper bound, never a deployable policy or training
selection criterion.

The current result supports improving the gate's prediction of expert advantage
and the U-Net's missed-word recall. Increasing this gate's acceptance quota
currently loses accuracy. A future specialist should target alphanumeric errors
and words missed by DeepSolo, using TRAIN-only paired supervision and a separate
calibration split; punctuation-only edits do not supply that specialization.
Such retraining has not been performed by this experiment.

## Execution and qualification

Implementation: `scripts/run_unet_corrected_token_router.py` and
`recognition/unet_corrected_token_interfaces.py`. Explicit configuration:
`configs/approximate/unet_corrected_token_router_v1.json`. Tests:
`tests/recognition/test_unet_corrected_token_interfaces.py`,
`tests/recognition/test_unet_corrected_token_audit.py`, plus token, fallback
and grounded-recovery regressions. All **67 focused tests pass**, including native
confidence inheritance, crop shape/pixel binding, normalized denominators,
ineligibility, routing counts and injected OOM propagation.

`smoke_v2` and resource-resume `smoke_v3` complete 16 TRAIN / 4 TEST images, two
folds, both variants and both fallback profiles. Their features, predicted
scores and learned checkpoint tensors are exactly equal. These qualify execution
only. The first smoke stopped
before training because raw-pixel SHA was compared with the upstream
shape-and-pixel SHA; the canonical hash fixes that boundary. Original inputs
remain unchanged and the failure receipt is preserved.

`full_v1` completed both TRAIN feature sets and stopped before fitting because
the shared GPU lacked the inherited 8 GiB free-memory margin. No CUDA OOM occurred.
After measurement, only that preflight margin changed to 2 GiB; precision, batch,
architecture, targets and optimizer stayed identical. All feature hashes were
verified before reuse in fresh `full_v2`. The earlier failure and source snapshots
remain. An injected-OOM test fixture initially lacked mandatory runtime fields;
the fixture was fixed before the 67-test pass. Figure export initially lacked
Matplotlib in the main environment; an existing plotting environment generated
the final figures, without installation or scientific rerun.

The immutable source snapshot is external. RTX3090 CUDA FP32, deterministic math
attention, sequential frozen CLIP extraction and tiny token-head fitting are
documented HARDWARE_DEVIATIONs. No packages/weights are installed. Each GPU stage
records models, shape/batch, precision, allocated/reserved peaks and OOM handling.
Cached U-Net/BLIP generation costs are not hidden as current online measurements.

Measured process peaks are 719,228,416 / 805,306,368 allocated/reserved bytes for
TRAIN CLIP extraction, 752,758,272 / 845,152,256 for TEST CLIP extraction, and
117,515,264 / 138,412,032 across the token-training stage and all saved fitting
checkpoints. The CLIP processor uses 224-pixel inputs; token fitting uses batch 8,
980x512 patches, 32x768 queries and variable word counts. All stages use FP32 on
RTX3090 and complete without OOM. Per-fit peaks and the outer stage receipt are
both retained because the fitter resets peak counters. These measurements do
not include regenerating the cached U-Net, PARSeq, editor or BLIP outputs.

Independent audit `scripts/audit_unet_corrected_token_router.py` passes: 30 pinned
sources, 3706 image/expert pairs, 3106 feature records, 5 disjoint RGB-grouped
folds, 32 checkpoint fitting scopes, 6 OOF epoch points and all 44 TEST routes.
It recomputes lexical counts and ignore masks, replays routing, checks fit-only
scaling/targets and strictly loads every learned checkpoint on CPU. This is an
independent implementation audit, not an independent held-out validation set.

External root:
`/extra1/bmartelli/context-str-runs/unet-corrected-token-router-20261026-v1`.
The audited full result is `full_v2/results.json`, SHA256
`06c1694e52cce3294c3e94989d1049cb8ace25bd244de34aad7d2c90fee26b60`.
Reviewed tables/PNG/PDF and the GT-only diagnostic are in `reviewed_results_v2`.
Portable counts, decisions, source references and qualification are saved in
[the completion record](../provenance/unet_corrected_token_router_v1_completion.json).
The early root `source_plan.json` describes the abandoned pixel/boundary draft;
the actual frozen plan is `full_v2/resolved_configuration.json`.
Original source data, weights, strict settings and unrelated work stay preserved.
Decision: `UNET-CORRECTED-TOKEN-001` in `PAPER_AMBIGUITIES.md`.
