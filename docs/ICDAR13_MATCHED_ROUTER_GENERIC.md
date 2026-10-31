# Newly trained ICDAR2013 U-Net/Generic-DeepSolo router at exactly61 fallback images

**Approximate reproduction / research extension.** User explicitly requests
NEW router training, then a full TEST result with exactly61 images sent to
DeepSolo. This is a trained gate, not just substitution of fallback outputs
behind the previous weights. The numerical experiment uses real authenticated
expert caches; logical fallback counts are distinct from physical new inference.

## Full TEST outcome

All233 official ICDAR2013 TEST images and1095 care words are retained. The
requested route accepts172 U-Net/PARSeq images and sends61 to DeepSolo:
**61/233 (26.18%)**. New router F1 is **78.7879%**, with832TP,185FP and263FN.

| Expert/rule | TP | FP | FN | Lexical micro F1 % | DeepSolo images/total (%) |
| --- | ---: | ---: | ---: | ---: | --- |
| Improved U-Net/PARSeq alone | 772 | 223 | 323 | 73.8756 | 0/233 (0.00%) |
| Previous frozen gate on the same new expert outputs, exact budget | 831 | 136 | 264 | 80.6014 | 61/233 (26.18%) |
| **Newly trained gate, user-requested exact budget** | **832** | **185** | **263** | **78.7879** | **61/233 (26.18%)** |
| Generic DeepSolo alone | 977 | 95 | 118 | 90.1707 | 233/233 (100.00%) |
| Conditional GT oracle, diagnostic only | 922 | 99 | 173 | 87.1456 | 61/233 (26.18%) |

The new training improves over U-Net alone by4.9123 percentage points, while
performing1.8135 points worse than the previous gate on the identical new expert
outputs and identical61-image budget. It adds only one TP but49 FP relative to
that gate. Do not promote it as an improvement over the previous gate based
only on its high selection F1.

The oracle row uses GT and the same nonempty-light eligibility guard. It is an
upper bound for that guarded image-routing problem, not a deployable rule. Exact
target-residual analysis under the same guard finds at least86/233 (36.91%)
DeepSolo images for an ideal90% witness (90.0376%,86/233,36.91%). This does not
promise that the trained observable gate can achieve that witness or establish
an unrestricted bound after changing expert emissions/eligibility.

## Actual fresh training

The frozen light expert is the preceding qualified BCE+Dice epoch30 U-Net at
RGB768/native384, text.5/kernel.5, original-pixel padding4, maximum50 crops,
PARSeq native sequence probability.9. Original text/crops/emission are preserved.
The router adapter adds only the qualified mean/min-token confidence descriptor
needed by the existing gate interface; it does not change recognition outputs.

The stronger teacher is the independently qualified ICDAR2013 Generic
keep-unmatched DeepSolo profile: ViTAEv2-S/TextOCR checkpoint54d3a454..., native
shortest1024/max1892, detector score.4, no character-confidence pruning, released
IC13 Generic87778-entry dictionary. Its complete TEST endpoint reproduces
90.1707%,DeepSolo233/233 (100.00%). No lexicon is applied to U-Net/PARSeq.

Use the existing RGB-disjoint183 FIT /23 selection /23 CAL split. Generate183
fresh light fitting observations; reuse the exact selected light outputs for
selection/CAL/TEST and all authenticated229 TRAIN/233 TEST Generic-teacher
outputs. Only183 FIT images enter gradients/fits. Standardization, reference F1
and weighted normalized paired-utility targets also use FIT only.

Train20 fresh candidates: Ridge alpha1/10/100 and two small HGB recipes, each
with OCR18, OCR18+tokens8, OCR18+maps12, or all38 descriptors. Select at the
rounded same-budget counterpart6/23 DeepSolo images (26.09%) on selection, then
within0.25 F1 points prefer fewer stored scalars/bytes. Calibrate unchanged
weights on separate CAL23, preserving all scalar controls. TEST labels never
choose weights, groups or thresholds.

The selected new model is HGB30 iterations, maximum7 leaves, L2=10,
minimum leaf40, with OCR+tokens26 input dimensions and329 stored numerical
scalars. Selection F1 is95.3846%,DeepSolo6/23 (26.09%). All20 fresh fits pass
independent Ridge normal equations or deterministic tree refitting. The selected
model SHA256 is `f9bee0f51441fb05b85a2f752c6a3fd8919d43eb35f6ff2ca7f4ce7ef884a3d0`.

Exact61 uses the new trained score order, with original RGB hashes breaking
ties, to accept172 images. It is an annotation-free batch budget, rather than
a claim that one fixed scalar threshold always produces61 calls. The separate
CAL scalar cutoff0.27901718568782713 transfers to105/233 images (45.06%) and
F1 80.3588%; that result is not substituted for the requested61-image outcome.

## Observed generalization limitation

| Population | U-Net F1 % | DeepSolo count for U-Net-only | Generic DeepSolo F1 % | DeepSolo-only count |
| --- | ---: | --- | ---: | --- |
| FIT183 | 93.1060 | 0/183 (0.00%) | 82.0476 | 183/183 (100.00%) |
| Selection23 | 90.3226 | 0/23 (0.00%) | 85.0746 | 23/23 (100.00%) |
| CAL23 | 72.7273 | 0/23 (0.00%) | 89.5833 | 23/23 (100.00%) |
| TEST233 | 73.8756 | 0/233 (0.00%) | 90.1707 | 233/233 (100.00%) |

The relative expert quality reverses between fitting/selection and independent
CAL/TEST. U-Net has fitted the183 FIT images, while its independent performance
is much weaker. This makes fitting-domain utility optimistic; a large selection
score is not evidence of corresponding TEST F1. The precise share due to
expert-seen fitting, small selection composition and head selection is not
isolated here. No out-of-fold expert training is claimed.

Primary metric: per-image lexical multiset counts after Unicode casefold and
alphanumeric filtering, aggregated micro F1; no localization IoU. Keep all care
GT, native256 care/ignore precedence and expert-specific declared emissions.
Empty predictions remain FN. Invalid DeepSolo envelopes remain unmatchable FP.
This is not official polygon Hmean or verified paper Table5 accuracy.

## Qualification, runtime and evidence

Eight focused interface/exact-budget/replay/feasibility tests pass. The genuine
8-FIT/four-image heldout smoke trains20 fresh heads, independently verifies all
20 fits and runs an exact1/4 DeepSolo budget (25.00%); its F1 is not a full-data
quality estimate. Full run passes462 expert count replays, all20 independent
head fits and233 single-image deployment-score replays with maximum error0.
An additional independent direct-CSV audit recounts466 TEST expert outputs,
checks five aggregate routes and exactly61 selected fallback IDs.

Actual hardware is a shared RTX3090. FP32 U-Net batch1/PARSeq32 supplies the183
missing fitting observations, with763,219,968 allocated /855,638,016 reserved
bytes at peak. Other authorized GPU jobs remain intact. Router fitting itself
runs on CPU. No installation, new DeepSolo neural inference, online latency
measurement or8GB whole-pipeline qualification is claimed. Physical new light
inferences:183; physical new DeepSolo inferences:0. Logical primary fallback:
61/233 (26.18%).

Decision classifications and limitations are in
`ICDAR13-MATCHED-ROUTER-GENERIC-001` in `docs/PAPER_AMBIGUITIES.md`.
Code: `recognition/icdar13_matched_router.py`,
`scripts/train_icdar13_matched_router_generic.py`,
`scripts/audit_icdar13_matched_router_generic.py`.
Config: `configs/approximate/icdar13_matched_router_generic_v1.json`.
Tests: `tests/recognition/test_icdar13_matched_router.py`.

External evidence:
`/extra1/bmartelli/context-str-runs/icdar13-matched-router-generic-20261028-v1/full/`.
`results.json` and `test_scores.json` preserve counts/decisions; `trials.json`
and `routers/` preserve all20 fresh fitted heads; `frozen_router.json` binds
selection/calibration; `deployment.joblib` exports the selected new model;
`exact61_deployment.json` binds the observable rank rule to the61-image budget;
`deepsolo_selected_images.csv` and `.json` identify exactly61 fallback images;
`independent_audit.json` and `independent_csv_audit.json` retain verification.
Source snapshots, resolved configuration, targets/weights and all input hashes
remain beside the evidence. No original result or strict configuration is changed.
