# ICDAR2013 full-word U-Net refit and ICDAR2015 fallback feasibility

**Approximate reproduction / research extension.** User target: IC13 lexical
micro F1 >=90% with strictly less than30% logical DeepSolo fallback. Separately,
measure whether the already used IC15 long768 expert can reach78.2% by routing.
Neither target is a promised result or an identification of paper accuracy.
Paper sections3.2/3.3/3.5 and4.1 and existing mask/model/training implementations
were inspected before modifying scientific components.

## Mask finding and isolated correction

Historical native IC13 Task2.2 training uses per-character RGB BMP foreground.
The preceding router used a TotalText-trained Text+Kernel corrected expert,
without native IC13 fitting. It therefore did not train on those IC13 masks.
IC15 long768 already uses filled disjoint word polygons, not glyph silhouettes.

The new IC13 target candidate has229 official TRAIN images and849 care word
groups. Blank-separated Task2.2 character groups form filled convex word hulls;
care-over-ignore precedence, word owners and contact separators preserve one
connected region per word. This parser/supervision is ASSUMPTION, not original
Task1 TRAIN word polygons. Original raw masks remain untouched. Audit all words
again after the actual RGB768/native384 paired transform: zero lost and zero
fragmented owner IDs. Raw foreground contains5695 components, including ignored
text; this is a component count, not a count of care characters. Inspect
`mask_comparison.png` in the external targets directory.

The first preparation failed before GPU work because an explicit binarization
configuration field was missing. Preserve its incomplete v1 target folder.
The corrected v2 profile succeeds; no failed sample is removed. An initial
synthetic gradient test had an invalid all-positive kernel fixture, violating
the reused balanced-loss requirement for negative pixels. The fixture was
corrected; all26 relevant target/kernel/gradient tests pass.

## Predeclared training and selection

Retain the previous RGB-disjoint183 FIT /23 model-selection /23 calibration
partition of official IC13 TRAIN. Preserve all233 official TEST images and1095
care words. Fit segmentation only on183; choose epochs/branches/OCR policy and
cheap gate recipe on23 selection images; unchanged gate thresholds use the
separate23 calibration images. TEST labels never select weights or thresholds.
Historical TEST exposure and small validation remain limitations.

Transfer every qualified Text+Kernel source tensor with strict=True; no head
reset, implicit ImageNet-only warm start or missing-key relaxation. Reuse filled
word targets, word nuclei, weighted separator BCE, residual attention skips,
aspect-preserving letterbox and original-resolution PARSeq crops. Compare frozen
encoder and partial layer3/4 control because prior partial training improved
proposal F1 but lost7 exact words against its matched frozen continuation; it
must not be claimed as a proven recognition improvement. Encoder BN remains
frozen. Additional Gap/RGB/graph correctors are not automatically declared
compatible with new native384 observations.

Two100-epoch arms use FP32 RGB768/native384, physical4/accumulation2, Adam with
fresh moments, decoder LR1e-5, partial encoder LR1e-6, weight decay.01, cosine to
0.1x, clipping1, seed20261028. Full-word separation weight3 plus balanced kernel
BCE weight1 and distance fraction.5 reuse existing kernels. These transfer and
optimization choices are ASSUMPTION; native size/batch/staged GPU models on the
actual RTX3090 are HARDWARE_DEVIATION. Do not present this as8GB qualification.
GPU smoke passes both real backward/optimizer arms; allocated/reserved peaks
are4,909,002,752/6,209,667,072 bytes frozen and6,351,695,872/7,455,375,360 partial.
Another authorized IC15 job shares the GPU; preserve it and make no isolated
latency claim. Errors and OOM propagate explicitly with preserved tracebacks.

Evaluate genuine PARSeq every10 epochs and at epoch0/final. Epoch0 remains
eligible. Fixed checkpoint policy: text.65/kernel.5, maximum50 crops, pad2
original pixels, sequence probability.7. Select epoch by validation lexical F1,
earliest exact tie. After training, compare the finite validation-only text
threshold [.5,.65,.8], padding [0,2,4] and sequence cut [0,.4,.7,.9] grid and both
qualified arms. No TEST-driven winner or late threshold search is allowed.

Fit the previous small CPU DeepSolo-utility recipes using native OCR18, optional
tokens8 and actual text/kernel maps12. No extra encoder is needed. Strict
fallback count is ceil(max_fraction*N)-1: at30%, at most69/233 (29.61%) on TEST
and6/23 (26.09%) on validation. Scalar calibration is reported separately from
an annotation-free batch rank that enforces the hard budget. Conditional GT
oracles are diagnostics with the same count/eligibility, never deployed.

Primary metric keeps the previous casefold+Unicode isalnum per-image lexical
multiset micro F1, source native256 ignore policy, care/empty denominators and
no localization match. It is not official polygon Hmean or verified Table5
accuracy. Strict paper components/configuration remain unchanged.

Implementation: `segmentation/icdar13_word_refit.py`,
`scripts/prepare_icdar13_word_refit.py`, `scripts/train_icdar13_word_refit.py`.
Profiles: `configs/approximate/icdar13_word_refit_v1.json`,
`configs/approximate/icdar13_word_refit_local_paths_v1.json`.
Tests: `tests/segmentation/test_icdar13_word_refit.py`.
External targets: `/extra1/bmartelli/datasets/context-str/icdar13-word-regions-20261028-v2`.
External run: `/extra1/bmartelli/context-str-runs/icdar13-word-refit-20261028-v1`.

## Official word annotations recovered after the training contract froze

The official `Challenge2_Training_Task1_GT.zip` is now available locally. Its
MD5 `f3a425284a66cd67f455d389c972cce4` matches the primary
[MMOCR dataset registry](https://github.com/open-mmlab/mmocr/blob/main/dataset_zoo/icdar2013/textdet.py).
All229 annotation files pass ZIP CRC checks. Acquisition and transport failures
are retained under `/extra1/bmartelli/context-str-artifacts/icdar13-word-annotation-check-20261028/`.
Both official HTTPS aliases initially failed local certificate-chain verification;
the successful public annotation-only request disabled that check and verified
the archive against the registry checksum before using its text files. No
packages, system certificate settings or raw dataset files were changed. The
first comparison command used system Python without OpenCV and failed before
reading labels; rerunning in the existing main environment passed.

All849 care-word identities match the Task2.2 group parser after the declared
casefold/alphanumeric normalization. Only raw punctuation in images117/143
differs, with no effect on the primary lexical metric. The running frozen
configuration's statement that Task1 TRAIN annotations were unavailable is a
creation-time limitation resolved by this companion evidence. Keep its bytes
and experiment inputs unchanged. Filled group hull geometry remains an explicit
ASSUMPTION, rather than silently replacing it with Task1 rectangles mid-run.

The independent SciPy audit follows the **actual active loader** and verifies
all229/849 word identities and target hashes: zero native lost words, zero
fragmented words, zero fused binary components. Its active-training-plan hash
binds these new full-word PNGs to the running model, not just a visual demo.
`mask_independent_audit.json` and `acquisition_and_comparison.json` retain source
identities and per-image checks.

## Qualification and automatic completion

Both genuine GPU backward smokes and the complete smoke pipeline pass. The
smoke performs validation policy selection,20 small-head fits, independent Ridge
equations/tree replay, separate calibration, real held-out crop inference and
exact routing/oracle checks. It uses4 diagnostic TEST images and is not a quality
result. The single declared smoke budget sends1/4 images (25.00%) to DeepSolo
logically. All29 focused target/kernel/encoder/feasibility/emission tests pass.
The first emission fixture lacked native token log-probabilities; correcting the
fixture preserved the production input validation. No model error was suppressed.

At readiness, the full two-arm100-epoch training was running and no full-refit
TEST result was available. Both arms are now complete; see the completed recheck
below. The90% target was not achieved. `full/workflow_status.json` is the live status. A
detached supervised runner waits for both complete branches and verifies frozen
finish source/profile hashes before automatic validation-only policy/gate
selection, calibration and complete TEST evaluation. It uses an isolated copy of
the original training sources plus the qualified finisher. Worker disappearance
and stage failures produce explicit status/failure files. Original runner source
and launch evidence are preserved when adding process-liveness supervision.

`configs/approximate/icdar13_word_refit_finish_v1.json` explicitly rejects empty
OCR decodes as no-word emissions, while retaining raw outputs and all care GT.
This is an ASSUMPTION and a deliberate new emission policy, not denominator
cleanup after scoring. Optional strict fallback controls are20/25/30%; scalar
CAL cuts and annotation-free exact-budget ranks are kept separate. Final
`results.json`, `independent_audit.json`, `completion.json` and
`report/SUMMARY.md` are written automatically only after their checks pass.
The90% target remains aspirational. New DeepSolo inferences for this refit:0;
teacher/fallback outputs are reused with raw payload/image/profile verification.

Finisher: `scripts/finish_icdar13_word_refit.py`; word audit:
`scripts/audit_icdar13_word_targets.py`; explicit-emission test:
`tests/scripts/test_finish_icdar13_word_refit.py`.

## ICDAR2015: minimum fallback needed for target F1 78.2%

This calculation uses the exact `long768` U-Net/PARSeq outputs from the preceding
complete TEST500 router comparison, unchanged native ViTAE/TextOCR DeepSolo
policy,2077 care words and the identical lexical/ignore metric. It is not a
claim about newly running IC15 models, official polygon Hmean or paper accuracy.

| Fixed expert / routing rule | TEST F1 % | DeepSolo images/total (%) |
| --- | ---: | --- |
| U-Net long768 alone | 59.7031 | 0/500 (0.00%) |
| Trained U-Net router at half fallback | 65.9547 | 250/500 (50.00%) |
| Frozen validation scalar cut | 65.9014 | 263/500 (52.60%) |
| DeepSolo alone | 70.3665 | 500/500 (100.00%) |
| Global GT oracle with current eligibility guard | 74.9869 | 270/500 (54.00%) |
| Global GT oracle allowing every image to use either expert | 75.1112 | 259/500 (51.80%) |

**No fallback count reaches78.2% with these fixed image-level expert outputs.**
The unrestricted global maximum is75.1112%, even with GT access and freedom to
route every image. Its witness uses259/500 (51.80%) DeepSolo images; it is also
the minimum fallback among routes attaining that global maximum. The maximum
under the current guard is74.9869%,270/500 (54.00%). These are diagnostic
oracles, not trained or deployable rules. To reach78.2%, at least an expert's
outputs/emission policy or the image-only selection mechanism must change;
merely increasing the fallback fraction cannot do it.

The proof uses the exact target fraction391/500. Starting from all-DeepSolo,
let each image gain be `1000*DeltaTP -391*DeltaPredictions`. At any fixed light
acceptance count, selecting the largest gains maximizes the target residual.
The largest residual over **all501 cardinalities**, including no eligibility
restriction, is **-59011**. A negative value certifies that every image-level
mixture stays below the target. Fractional optimization separately certifies
the global F1 maximum with exact zero residual; no average-difficulty assumption
or linear interpolation of aggregate F1 is used. The trained router's entire
score-rank frontier also never reaches78.2%; its diagnostic maximum is the
all-DeepSolo endpoint70.3665%,500/500 (100.00%). No TEST-chosen threshold is
promoted to deployment.

`recognition/fallback_feasibility.py`,
`scripts/analyze_icdar_fallback_target.py` and
`tests/recognition/test_fallback_feasibility.py` implement/check the calculation.
Exhaustive enumeration of all subsets in20 small random populations independently
checks target minimum counts, global F1 and minimum fallback at tied optima.
Machine-readable full source hashes, integer certificates and trained frontiers:
`icdar2015_target782_feasibility.json` under the external refit root.

The completion runner also writes `report/deployment_qualified.joblib`: it fixes
the inherited descriptive `mask`/`maps` cost-label alias using actual selected
feature groups. Original deployment and frozen scientific code remain preserved;
regressor/scaler objects and numerical predictions are unchanged. The separate
`deployment_metadata_qualification.json` binds both artifacts. This follows the
same metadata-only qualification boundary as the earlier native-router export.

## Completed training and independent recheck

Both100-epoch branches finished with all183 fitting images in every epoch (36600 total fitting-image passes). Both selected epoch70 under the fixed checkpoint policy. Each strictly loaded replay agrees with its stored selection counts. The finite branch/crop/emission grid selects the frozen encoder, text.65/kernel.5, pad4 original pixels and native PARSeq sequence probability.9. The chosen cheap gate is OCR18 Ridge alpha100,55 coefficient/intercept/normalization scalars (440 numeric bytes). No additional encoder is used.

All233 TEST images /1095 care words are evaluated, with zero inference failures. All final routes are logical cached-output choices; new DeepSolo calls for this refit:0. The numerical run completed successfully, while its scientific90% F1 /strictly-below30% fallback target failed. This remains approximate reproduction / research extension, not paper Table5 accuracy or official polygon Hmean.

### TEST routes

| Rule | TEST lexical F1 % | Accepted light | Predicted care words | DeepSolo images/total (%) |
| --- | ---: | ---: | ---: | --- |
| all_light | 71.8511 | 233 | 1001 | 0/233 (0.00%) |
| all_deepsolo | 88.4956 | 0 | 1052 | 233/233 (100.00%) |
| strict_cap_0.2 | 77.6550 | 187 | 986 | 46/233 (19.74%) |
| strict_cap_0.25 | 79.8077 | 175 | 985 | 58/233 (24.89%) |
| strict_cap_0.3 | 80.1923 | 164 | 985 | 69/233 (29.61%) |
| conditional_gt_oracle_0.2 | 82.9736 | 187 | 990 | 46/233 (19.74%) |
| conditional_gt_oracle_0.25 | 84.8976 | 175 | 1004 | 58/233 (24.89%) |
| conditional_gt_oracle_0.3 | 86.4891 | 164 | 1007 | 69/233 (29.61%) |
| validation_cut_0.2 | 79.4415 | 179 | 982 | 54/233 (23.18%) |
| validation_cut_0.25 | 80.0000 | 167 | 985 | 66/233 (28.33%) |
| validation_cut_0.3 | 82.3923 | 123 | 995 | 110/233 (47.21%) |

The primary scalar validation cut transfers to110/233 (47.21%) fallback and therefore exceeds the requested cost cap. The predeclared25% calibration control gives80.0000% F1,66/233 (28.33%); this descriptive TEST control is not promoted to a TEST-selected primary. The exact-budget rank enforces69/233 (29.61%) and produces80.1923% F1. All weights and scalar cuts were frozen before full TEST scoring.

### The90% target is impossible at this budget for the current fixed outputs

With the current eligibility guard, an exact GT oracle at69/233 (29.61%) reaches86.4891% F1. Even allowing every image to choose either expert, including currently ineligible empty predictions, the exact maximum at that same69/233 (29.61%) is87.2813%. Therefore improving only the image router cannot reach90% with these outputs under the requested budget.

Exact target-residual analysis finds at least107/233 (45.92%) fallback images are necessary for90% in the unrestricted ideal case; its witness gives90.0328%. Under the current eligibility guard, the minimum is110/233 (47.21%), with a90.0000% witness. These use GT and are diagnostics only. The raw expert predictions and deployment policies are unchanged.

### What the recheck establishes

Independent recheck PASS:706 immutable training sources,708 frozen finish sources, all200 training epochs,229 exact training/inference RGB tensor comparisons,229 reconstructed training observations,20 head-fit checks,233 reconstructed TEST observations,6 independent routes and2555 original-RGB crop hashes. The23 selection probability maps are bitwise identical between the selected-checkpoint replay and final expert cache. No preprocessing mismatch was found.

Word supervision is correct at word granularity. All849 derived word envelopes also match official Task1 word rectangles exactly (one-to-one same-normalized-word envelope IoU1.0 for all849), in addition to the earlier zero-loss/fragmentation/fusion mask audit. Filled hull interiors remain the declared geometric approximation. This rules out still loading glyph-only targets in the completed refit.

### Generalization evidence

| Population | Images | Care words | Light expert F1 % | DeepSolo images/total (%) for that light-only result |
| --- | ---: | ---: | ---: | --- |
| fit | 183 | 690 | 95.3437 | 0/183 (0.00%) |
| selection | 23 | 64 | 88.7097 | 0/23 (0.00%) |
| calibration | 23 | 95 | 72.0430 | 0/23 (0.00%) |
| test | 233 | 1095 | 71.8511 | 0/233 (0.00%) |

The expert fits its seen images well but performs much worse on the independent calibration population and TEST. The selection population contains only23 images /64 words, compared with23 /95 words in calibration. Its88.7097% light F1,0/23 (0.00%) fallback, is optimistic relative to72.0430%,0/23 (0.00%), on separate calibration and71.8511%,0/233 (0.00%), on TEST. This is evidence of limited generalization and selection instability; the precise causal share of scene composition, overfitting and policy search is not isolated by this run.

The gate also trains on outputs from segmentation-fitting images. Its fit-domain light expert is better than DeepSolo, while the unseen expert is much worse; this observed change in relative expert quality makes its fitted utility targets optimistic. A cross-fitted expert/router study would be needed to isolate this effect. No cross-fitting claim is made here.

On the identical cached crops, all four predeclared sequence cuts were rechecked without promoting a TEST-selected policy. Cut.9 gives the highest observed TEST F1 in that finite control (.0:60.1855%,.4:67.5926%,.7:70.1975%,.9:71.8511%; all0/233,0.00% fallback). It retains753 of811 unfiltered correct readings and removes599 predictions. Simply relaxing that sequence cut is not supported by these controls. This does not test every other emission rule.

### Comparison with previous frozen pipelines at the same strict budget

| Pipeline | TEST F1 % | DeepSolo images/total (%) |
| --- | ---: | --- |
| Previous corrected TotalText U-Net/router | 82.9952 | 69/233 (29.61%) |
| Previous YOLO/router | 86.0927 | 69/233 (29.61%) |
| New native word refit/router | 80.1923 | 69/233 (29.61%) |

This refit does not improve the previous corrected U-Net pipeline at the same budget. They differ in expert training and correction stages; the previous expert includes qualified TotalText correction modules. These are pipeline comparisons, not a controlled glyph-versus-word-mask ablation. Do not attribute the decline solely to mask granularity.

### Completion evidence

External full run: `/extra1/bmartelli/context-str-runs/icdar13-word-refit-20261028-v1/full`. Completed source evidence includes `results.json`, `completion.json`, `independent_audit.json`, `completed_recheck_v1.json`, `official_word_geometry_recheck_v1.json`, `selection_map_equivalence_recheck_v1.json`, `fixed_crop_emission_diagnostic_v1.json` and `matched_previous_router_comparison_v1.json`. `report/verified_results.csv` preserves counts for all predeclared controls. Original readiness, smoke and failure records remain intact.

The first supplementary diagnostic command used system Python without SciPy and failed before analysis. The existing qualified main environment rerun passed; no environment was installed or changed. All scientific results are based on the qualified-environment commands.

The repeatable recheck entry point is `scripts/recheck_icdar13_word_refit.py`.
`completed_recheck_v2.json` adds explicit approximate-mode/result-class metadata
and a mode guard; all numerical checks and diagnostics match the preserved V1
exactly. Both source versions are archived under `recheck_sources/`. A separate
exact Counter comparison of all849 word envelopes and bitwise map comparisons
also passes (`geometry_and_map_exact_recheck_v1.json`), independently confirming
the IoU-based geometry check. Seven focused shape/gradient/budget/feasibility
checks passed during the completed-run recheck. No model, threshold or original
result was altered.

### Supervision masks versus predicted masks

The zero-fragmentation supervision audit does not assert perfect predicted word instances. The completed model was therefore checked separately on actual qualified TRAIN word owners, including the held-out calibration images. No TEST owner map was fabricated and no model inference or threshold tuning was added.

| Population | Care words | Predicted instances | Words touching multiple predicted instances | Words touched by fused instances | Native bbox matches IoU>=.5 | DeepSolo images/total (%) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fit | 690 | 736 | 2 | 14 | 678 | 0/183 (0.00%) |
| selection | 64 | 90 | 8 | 6 | 57 | 0/23 (0.00%) |
| calibration | 95 | 129 | 10 | 14 | 76 | 0/23 (0.00%) |

On CAL23 /95 words, the raw binary foreground has8 multi-component word contacts and16 words involved in fused regions; kernel expansion changes these to10 and14, with76 native bbox matches instead of74. These are all-component, any-pixel contact/native-box proxies before top50 and OCR rejection, not official detection F1 or a proof that every component is a character. No CAL word lacks every positive foreground pixel, but grouping/localization is still imperfect.

The two stored calibration examples show formatted numbers split into partial-word regions and separate words joined into a larger region. The targets are filled full-word regions; the predictions still make grouping errors. See the diagnostic image at `/extra1/bmartelli/context-str-runs/icdar13-word-refit-20261028-v1/full/report/predicted_mask_recheck.png`. Figure examples are explicitly selected for high split/fusion counts and are not a representative accuracy sample. All-population counts are in `predicted_word_topology_recheck_v1.json`; the plot recipe and hashes are preserved separately.
