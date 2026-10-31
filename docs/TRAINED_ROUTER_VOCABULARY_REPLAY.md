# Frozen trained routers with ICDAR-specific DeepSolo vocabularies

**Approximate reproduction / research extension.** Full official TEST: ICDAR2013 233 images /1095 care words; ICDAR2015 500 images /2077 care words. Logical image routing on cached expert outputs; zero new neural inferences and no training in this replay.

Use the existing dataset-specific trained gates from `ICDAR_TRAINED_BUDGET_ROUTER.md`, not the earlier TotalText-only transferred essential head. Keep every selected weight, light-expert output, eligibility guard, RGB tie key, validation scalar cutoff and original batch quota unchanged. The ICDAR2013 U-Net is the selected TotalText corrected chain; ICDAR2015 U-Net is the selected native `long768` chain. YOLO uses its original direct640/PARSeq expert and OCR-only small boosted head. Only the fallback expert changes.

ICDAR2013 fallback: ViTAEv2-S+TextOCR TotalText checkpoint, RGB1024/max1892, Generic vocabulary, neural score0.40 / character0.0, keep-unmatched; this is the prior TRAIN-calibration-selected profile. ICDAR2015 fallback: ICDAR2015-finetuned ViTAEv2-S+TextOCR checkpoint, RGB1440/max4000, Weak vocabulary, score0.35 / character0.85, original edit-distance<1.5 rejection. Weak is released benchmark side information; this is not a no-vocabulary result. No lexicon is applied to U-Net/PARSeq or YOLO/PARSeq.

## Fixed predeclared budgets

Ranks use only unchanged model scores and RGB hashes. Batch-dependent decision boundaries are not validation-fixed scalar thresholds. All rows retain complete care GT and unchanged lexical microF1 normalization (Unicode casefold then alphanumeric characters), per-image multiset matching, no length filtering, no localization IoU and unmatched empty-normalized denominators.

| Dataset | Light expert | Previous F1 % | New F1 % | Gain pp | DeepSolo images/total (%) |
| --- | --- | ---: | ---: | ---: | --- |
| icdar13 | unet (tt_corrected) | 82.9715 | 84.0038 | 1.0323 | 70/233 (30.04%) |
| icdar13 | yolo (yolo) | 86.2281 | 87.3120 | 1.0839 | 70/233 (30.04%) |
| icdar2015 | unet (long768) | 65.9547 | 71.4805 | 5.5258 | 250/500 (50.00%) |
| icdar2015 | yolo (yolo) | 69.3919 | 75.8759 | 6.4840 | 250/500 (50.00%) |

## Original validation scalar cuts

No cut is recalibrated using the new fallback or TEST labels. The source calibration populations remain23 IC13 and50 IC15 images, separate from source head selection. Actual TEST fallback fractions differ from calibration budgets.

| Dataset | Expert | Target light acceptance % | Frozen cutoff | Previous F1 % | New F1 % | DeepSolo images/total (%) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| icdar13 | unet | 65 | -0.7661868225 | 85.1675 | 86.5842 | 112/233 (48.07%) |
| icdar13 | unet | 70 | -0.8046180969 | 84.8804 | 86.3398 | 109/233 (46.78%) |
| icdar13 | unet | 75 | -0.8678354092 | 84.9761 | 86.4350 | 106/233 (45.49%) |
| icdar13 | yolo | 65 | -0.3238543176 | 87.5059 | 88.6396 | 115/233 (49.36%) |
| icdar13 | yolo | 70 | -0.3817747395 | 87.3226 | 88.4201 | 103/233 (44.21%) |
| icdar13 | yolo | 75 | -0.4478576440 | 87.2280 | 88.3263 | 97/233 (41.63%) |
| icdar2015 | unet | 45 | 0.3431066052 | 66.8744 | 73.1518 | 302/500 (60.40%) |
| icdar2015 | unet | 50 | 0.2758701963 | 65.9014 | 71.7295 | 263/500 (52.60%) |
| icdar2015 | unet | 55 | 0.2056138704 | 65.4279 | 70.4688 | 215/500 (43.00%) |
| icdar2015 | yolo | 45 | -0.3066845950 | 70.4183 | 78.0856 | 316/500 (63.20%) |
| icdar2015 | yolo | 50 | -0.4319664860 | 69.3960 | 75.9709 | 254/500 (50.80%) |
| icdar2015 | yolo | 55 | -0.6046712588 | 68.5227 | 73.7733 | 192/500 (38.40%) |

## Full DeepSolo endpoint and complete rank curves

The original trained-router ignore implementation is kept for both before/after experts. IC13 has no changed ignore decision. IC15 differs from the previous standalone lexicon report on five predicted polygons: this router rebuilds native256 validity from clipped annotations and lets care polygons override ignored regions; the standalone study used the prepared validity PNG. A five-image diagnostic confirms that care-region override, rather than coordinate clipping, explains all five changed decisions (`ignore_policy_diagnostic.json`). All2077 care words and500 images are identical. Exactly five outputs move from ignored to care and add five FP without new TP: TP1614, FP208, FN463 gives82.7905%, DeepSolo500/500 (100.00%), versus the prior standalone82.8968%, DeepSolo500/500 (100.00%). This protocol detail is explicit rather than silently mixing denominators. The standalone lexicon result is preserved.

| Dataset | DeepSolo baseline | TP | FP | FN | F1 % | DeepSolo images/total (%) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| icdar13 | Generic calibrated | 977 | 95 | 118 | 90.1707 | 233/233 (100.00%) |
| icdar2015 | Weak source rule | 1614 | 208 | 463 | 82.7905 | 500/500 (100.00%) |

Every prefix of the frozen score ordering is also evaluated. Equal scores can be split only by the original RGB tie key. These TEST-descriptive rank extrema are not deployable validation choices and are not GT-oracle upper bounds.

| Dataset | Expert | Descriptive point | F1 % | DeepSolo images/total (%) |
| --- | --- | --- | ---: | --- |
| icdar13 | unet | test_descriptive_best | 90.2540 | 219/233 (93.99%) |
| icdar13 | unet | test_descriptive_minimum_deepsolo_no_loss | 90.2540 | 219/233 (93.99%) |
| icdar13 | yolo | test_descriptive_best | 90.2270 | 208/233 (89.27%) |
| icdar13 | yolo | test_descriptive_minimum_deepsolo_no_loss | 90.2179 | 203/233 (87.12%) |
| icdar2015 | unet | test_descriptive_best | 82.7905 | 500/500 (100.00%) |
| icdar2015 | unet | test_descriptive_minimum_deepsolo_no_loss | 82.7905 | 500/500 (100.00%) |
| icdar2015 | yolo | test_descriptive_best | 82.7905 | 500/500 (100.00%) |
| icdar2015 | yolo | test_descriptive_minimum_deepsolo_no_loss | 82.7905 | 500/500 (100.00%) |

## Interpretation and uncertainty

The stronger fallback improves all four original primary-budget hybrids, with YOLO remaining stronger than the selected U-Net. The unchanged heads were supervised against the old weaker fallback, so their utility targets are mismatched to the new experts. They do not maintain new DeepSolo-only performance at the original budgets. The complete frozen ordering has no hybrid gain over the new IC15 fallback; IC13 gains at its descriptive extrema are small, historically TEST-exposed, and require most images to run DeepSolo. No TEST-selected threshold is promoted.

| Dataset | Expert | Rule | F1 95% interval | Gain over new DeepSolo 95% interval pp | DeepSolo images/total (%) |
| --- | --- | --- | --- | --- | --- |
| icdar13 | unet | batch_0.7 | [80.935, 86.723] | [-8.585, -4.137] | 70/233 (30.04%) |
| icdar13 | unet | validation_0.7 | [83.480, 88.920] | [-5.763, -2.122] | 109/233 (46.78%) |
| icdar13 | yolo | batch_0.7 | [84.963, 89.589] | [-4.276, -1.491] | 70/233 (30.04%) |
| icdar13 | yolo | validation_0.7 | [86.162, 90.542] | [-2.890, -0.687] | 103/233 (44.21%) |
| icdar2015 | unet | batch_0.5 | [69.129, 73.734] | [-13.473, -9.231] | 250/500 (50.00%) |
| icdar2015 | unet | validation_0.5 | [69.395, 73.956] | [-13.220, -8.933] | 263/500 (52.60%) |
| icdar2015 | yolo | batch_0.5 | [74.032, 77.772] | [-8.290, -5.616] | 250/500 (50.00%) |
| icdar2015 | yolo | validation_0.5 | [74.112, 77.879] | [-8.201, -5.538] | 254/500 (50.80%) |

Intervals use2000 paired image-bootstrap draws, seed20261028, conditional on fixed experts/heads/decisions; fitting, selection, pretraining overlap and historical TEST exposure are omitted. These word lexical scores are not official polygon Hmean or verified equivalence with paper Table5 accuracy.

## Verification and evidence

External evidence: `/extra1/bmartelli/context-str-runs/trained-router-vocabulary-replay-20261028-v1`. `thresholds.csv` contains every original frozen cut/quota and before/after counts; `frontiers/` contains every ordered prefix; `per_image/` binds branches, unchanged scores and model hashes; `fallback_evaluation/` preserves corrected strings, geometry, GT and counts; `prediction_only/` contains733 GT-free fallback exports; `input_evidence.json` preserves raw hashes and all changed ignore decisions.

The independent audit recounts all fallback/light predictions, replays every deployed single-image gate and frozen decision, verifies maps/weights and every frontier point. The initial auditor incorrectly expected the DeepSolo envelope's `payload_sha256` field in plain lightweight-expert JSON and stopped with KeyError before producing an audit. Its executable and traceback are preserved in `failures/audit_schema_v1/`; the corrected auditor uses the actual original cache schema and independently reproduces scores/counts. The first plot attempt found no matplotlib in the main environment after writing its draft document. No package was installed: final plotting uses the existing YOLO environment, with the failed trace and draft document preserved under `failures/plot_environment_v1/`. No scientific predictions or reported metric changed during these repairs.

Source classifications: PAPER_SPECIFIED Section3.5/Algorithm2 motivation only; REPOSITORY_SPECIFIED existing qualified gate/expert/metric procedures; ASSUMPTION user-requested frozen-fallback substitution and harmonized original ignore protocol; HARDWARE_DEVIATION CPU cached replay, with historical isolated RTX3090 expert runs; UNRESOLVED metric equivalence, small validation, historical exposure, target mismatch and online latency. STRICT configurations are unchanged.

Config: `configs/approximate/trained_router_vocabulary_replay_v1.json`. Implementation: `recognition/router_fallback_replay.py`, `scripts/replay_trained_router_vocabularies.py`, `scripts/audit_trained_router_vocabulary_replay.py`, `scripts/report_trained_router_vocabulary_replay.py`. Tests: `tests/recognition/test_router_fallback_replay.py` plus the relevant original gate/deployment/lexicon/metric/frontier suites.

[Standalone PNG](/extra1/bmartelli/context-str-runs/trained-router-vocabulary-replay-20261028-v1/figures/frontiers.png), [SVG](/extra1/bmartelli/context-str-runs/trained-router-vocabulary-replay-20261028-v1/figures/frontiers.svg). Every plotted curve has explicit DeepSolo count axes, full denominators, percentage axes and machine-readable per-point counts.
