# Essential router: threshold selection on three datasets

Date: 2026-10-28. **Approximate reproduction / trained-router extension.** Full TEST populations: Total-Text 300 images, ICDAR2013 233, ICDAR2015 500. The primary metric is symbol-free lexical micro F1; it is not verified as the paper's recognition-accuracy protocol. Every image remains in the denominator.

The operational comparison uses the qualified essential corrected-U-Net head and a new matched YOLO head with the same 30 descriptors, Ridge alpha 1 and 91 FP64 numerical scalars (728 bytes). The YOLO head fits only the original Total-Text FIT 1028 and is frozen before its TEST scores are prepared. Its own validation cuts are then frozen before loading those TEST scores for frontier analysis. No new encoder or TEST fitting is introduced. The original U-Net-head-on-YOLO transfer and normalized-score follow-up remain separate, preserved controls. See [the inference and provenance guide](ESSENTIAL_ROUTER_ICDAR_TRANSFER.md).

## Method and interpretation

For each attainable scalar cutoff, accept the light expert when its score is at least that cutoff; identical scores move together. Empty/blank observations always defer. Enumerate the complete frontier, including unconditional DeepSolo. Select on validation the cutoff minimizing DeepSolo calls subject to `F1 >= F1_DeepSolo - delta`. The default zero-loss objective uses exact rational comparisons. Also measure the maximum-validation-F1 cutoff and minimum cost within a declared tolerance of that maximum. These objectives need not choose the same point.

An approximate simultaneous bootstrap lower band across the entire cutoff family is a more conservative validation control. It uses paired image resampling, 2000 draws, seed 20261028 and a maximum centered bootstrap statistic. It is conditional on frozen expert outputs and omits model-fitting/search uncertainty and historical TEST exposure. It is not a finite-sample population-F1 guarantee. With zero allowed loss, this tested conservative method chooses all DeepSolo in every dataset/expert case: 300/300 (100.00%), 233/233 (100.00%) and 500/500 (100.00%), respectively.

The exact improvement condition is `sum(2*deltaTP - F1_DeepSolo*delta_predictions) > 0` over images accepted by the light expert. The care-word denominator is unchanged. Actual paired counts can certify that inequality during evaluation; the router must estimate them from observable features during deployment. A cutoff cannot turn an incorrect ordering of estimated advantages into the privileged ordering.

The validation groups are Total-Text CAL 63, ICDAR2013 heldout 23 and ICDAR2015 heldout 100. ICDAR2013 validation uses the existing canonical Task2.2 blank-line word groups and glyph-color ignore parser; TEST uses Task1 word annotations. ICDAR2015 uses the existing quadrilateral/validity protocol. These source choices and small-cohort limitations are explicit. Validation is separate from fitting; it was historically used for upstream model selection.

## Matched half-fallback control

This is a fixed-quota ranking control rather than a claim that a fixed scalar cut transfers an exact budget. With 233 images, the implementation accepts 116 and sends 117 to DeepSolo. The GT oracle uses the same measured expert outputs and the same observable eligibility, but privileged image choices; it is not deployable.

| Dataset | Light branch / matching head | Measured half-budget F1 % | DeepSolo images/total (%) | Conditional GT-oracle F1 % at the same cost |
| --- | --- | ---: | ---: | ---: |
|Total-Text|unet|79.9904|150/300 (50.00%)|83.6555|
|Total-Text|yolo|80.8813|150/300 (50.00%)|84.2983|
|ICDAR2013|unet|85.4733|117/233 (50.21%)|89.9436|
|ICDAR2013|yolo|88.2739|117/233 (50.21%)|91.3753|
|ICDAR2015|unet|62.9394|250/500 (50.00%)|71.1170|
|ICDAR2015|yolo|69.5060|250/500 (50.00%)|75.7696|

The oracle demonstrates measured expert complementarity, including on ICDAR2015. The gap between an attainable score cutoff and privileged image selection is a ranking problem; moving one cutoff cannot repair that ranking. The oracle does not establish that the available observable features can learn its choices.

## Minimum fallback without an observed TEST F1 loss

**Post-hoc descriptive optima.** These values use TEST outcomes and are excluded from exported operational profiles. They answer what the fixed ranking can attain, rather than which cutoff has been validated for new images.

| Dataset | Head | DeepSolo reference F1 %, all images | Minimum-cost no-loss cut | Hybrid F1 % | Hybrid DeepSolo images/total (%) |
| --- | --- | --- | ---: | ---: | ---: |
|Total-Text|unet|82.6540; 300/300 (100.00%)|All DeepSolo|82.6540|300/300 (100.00%)|
|Total-Text|yolo|82.6540; 300/300 (100.00%)|-0.6452314953|82.6641|256/300 (85.33%)|
|ICDAR2013|unet|88.4956; 233/233 (100.00%)|0.5474004183|88.4956|227/233 (97.42%)|
|ICDAR2013|yolo|88.4956; 233/233 (100.00%)|-0.6693736297|88.5567|123/233 (52.79%)|
|ICDAR2015|unet|70.5669; 500/500 (100.00%)|All DeepSolo|70.5669|500/500 (100.00%)|
|ICDAR2015|yolo|70.5669; 500/500 (100.00%)|-0.7533351098|70.5669|452/500 (90.40%)|

## Can a cutoff exceed DeepSolo?

Numerically yes for the matched YOLO head, in these cached TEST populations. All gains remain small and their conditional paired intervals include zero. A TEST-best cutoff is not a validation-selected result or proof of general superiority. For the original U-Net absolute ranking there is no strict gain on these three TEST sets.

| Dataset | Head | Best descriptive TEST cut | Best F1 % | DeepSolo images/total (%) | Gain over its all-DeepSolo reference, points | Conditional pointwise 95% gain interval, points |
| --- | --- | ---: | ---: | ---: | ---: | --- |
|Total-Text|unet|All DeepSolo|82.6540|300/300 (100.00%)|+0.0000|[+0.0000, +0.0000]|
|Total-Text|yolo|-0.6189649136|82.6837|258/300 (86.00%)|+0.0296|[-0.3825, +0.4964]|
|ICDAR2013|unet|0.5474004183|88.4956|227/233 (97.42%)|+0.0000|[+0.0000, +0.0000]|
|ICDAR2013|yolo|-0.3320297639|88.8370|181/233 (77.68%)|+0.3414|[-0.5300, +1.3464]|
|ICDAR2015|unet|All DeepSolo|70.5669|500/500 (100.00%)|+0.0000|[+0.0000, +0.0000]|
|ICDAR2015|yolo|-0.5890931281|70.7008|480/500 (96.00%)|+0.1339|[-0.1242, +0.4538]|

## One common YOLO cutoff across the three datasets

A common cutoff **-0.6613352397** gives a strictly positive point estimate versus the router's DeepSolo profile in every TEST dataset. It minimizes total fallback among the searched common cuts satisfying that condition. Total logical fallback is **855/1033 (82.77%)**. It was found using TEST labels and remains a descriptive existence witness; its tiny gains do not establish a robust improvement.

| Dataset | Hybrid F1 % | DeepSolo reference F1 % | Hybrid DeepSolo images/total (%) |
| --- | ---: | ---: | ---: |
|Total-Text|82.6641|82.6540|256/300 (85.33%)|
|ICDAR2013|88.5567|88.4956|124/233 (53.22%)|
|ICDAR2015|70.6491|70.5669|475/500 (95.00%)|

The separate common cutoff selected using validation alone is **0.4184815511**. Its TEST routing is 298/300 (99.33%) on Total-Text, 227/233 (97.42%) on ICDAR2013 and 500/500 (100.00%) on ICDAR2015: 1025/1033 (99.23%) overall. Its TEST F1 values are 82.6705%, 88.5674% and 70.5669%, respectively, at those same costs. This observed non-loss saves only eight calls and is not a general population guarantee.

## Validation-selected single-image policies on TEST

Every cutoff below is chosen from the declared validation data using unchanged head weights. The empirical no-loss rule fails to preserve TEST F1 in some cases; the 23-image ICDAR2013 group is particularly unstable for the U-Net-trained head. Maximizing validation F1 also does not guarantee a TEST improvement. A common raw number is not a calibrated probability.

| Dataset | Head | Validation objective | Frozen cut | TEST F1 % | TEST DeepSolo images/total (%) |
| --- | --- | --- | ---: | ---: | ---: |
|Total-Text|unet|empirical_loss_0.0|0.5941322908|82.6427|298/300 (99.33%)|
|Total-Text|unet|maximum_validation_f1|0.5941322908|82.6427|298/300 (99.33%)|
|Total-Text|unet|shared_validation_no_loss|0.5883071818|82.6427|298/300 (99.33%)|
|Total-Text|unet|bootstrap_band_loss_0.0|All DeepSolo|82.6540|300/300 (100.00%)|
|Total-Text|yolo|empirical_loss_0.0|0.4184815511|82.6705|298/300 (99.33%)|
|Total-Text|yolo|maximum_validation_f1|0.4184815511|82.6705|298/300 (99.33%)|
|Total-Text|yolo|shared_validation_no_loss|0.4184815511|82.6705|298/300 (99.33%)|
|Total-Text|yolo|bootstrap_band_loss_0.0|All DeepSolo|82.6540|300/300 (100.00%)|
|ICDAR2013|unet|empirical_loss_0.0|-0.6910019990|80.0761|56/233 (24.03%)|
|ICDAR2013|unet|maximum_validation_f1|-0.2236659445|83.7275|100/233 (42.92%)|
|ICDAR2013|unet|shared_validation_no_loss|0.5883071818|88.4956|228/233 (97.85%)|
|ICDAR2013|unet|bootstrap_band_loss_0.0|All DeepSolo|88.4956|233/233 (100.00%)|
|ICDAR2013|yolo|empirical_loss_0.0|0.4533500040|88.5674|227/233 (97.42%)|
|ICDAR2013|yolo|maximum_validation_f1|0.4533500040|88.5674|227/233 (97.42%)|
|ICDAR2013|yolo|shared_validation_no_loss|0.4184815511|88.5674|227/233 (97.42%)|
|ICDAR2013|yolo|bootstrap_band_loss_0.0|All DeepSolo|88.4956|233/233 (100.00%)|
|ICDAR2015|unet|empirical_loss_0.0|0.0801037460|69.7017|440/500 (88.00%)|
|ICDAR2015|unet|maximum_validation_f1|0.1228329268|70.1718|477/500 (95.40%)|
|ICDAR2015|unet|shared_validation_no_loss|0.5883071818|70.4969|499/500 (99.80%)|
|ICDAR2015|unet|bootstrap_band_loss_0.0|All DeepSolo|70.5669|500/500 (100.00%)|
|ICDAR2015|yolo|empirical_loss_0.0|-1.4191147343|68.9328|220/500 (44.00%)|
|ICDAR2015|yolo|maximum_validation_f1|-1.0905184575|69.8746|347/500 (69.40%)|
|ICDAR2015|yolo|shared_validation_no_loss|0.4184815511|70.5669|500/500 (100.00%)|
|ICDAR2015|yolo|bootstrap_band_loss_0.0|All DeepSolo|70.5669|500/500 (100.00%)|

## Controls, uncertainty and deployment

The ordinary DeepSolo control uses the same checkpoint/RGB and its previously qualified native score floor 0.4 with no character-confidence pruning. The primary router fallback remains the previously TRAIN-selected native 0.35 / 0.925 profile. This exploratory control was added after observing cross-dataset endpoints; it does not retune or replace the primary fallback.

| Dataset | Ordinary DeepSolo F1 % | DeepSolo images/total (%) |
| --- | ---: | ---: |
|Total-Text|81.7308|300/300 (100.00%)|
|ICDAR2013|88.0480|233/233 (100.00%)|
|ICDAR2015|69.2562|500/500 (100.00%)|

Normalized utility divides the existing score by observed raw-word count plus one, recovering its training scale without new weights. This exploratory control retains only one Total-Text U-Net image at unchanged F1: 82.6540%, DeepSolo 299/300 (99.67%). Its ICDAR2013 YOLO-transfer best is 88.7442%, DeepSolo 230/233 (98.71%); it produces no ICDAR2015 gain over all DeepSolo. It does not recover a half-cost accuracy-preserving solution. It is not the matched YOLO head.

Exported `validation_deployments/{dataset}__{expert}.json` contain only validation-selected empirical, maximum-F1, shared-cut and conservative-band policies. TEST oracle cuts are excluded. `recognition/essential_threshold_deployment.py` accepts predicted rows, image dimensions and predicted maps only. YOLO still requires the shared frozen U-Net masks for these 30 descriptors; this is not a detector-only YOLO gate. Numerical head size is 728 bytes for either matched head, excluding common expert weights, activations and software overhead.

All routing metrics use cached expert outputs; no measured end-to-end speedup is claimed. Current-task actual offline DeepSolo inference covers all 733 ICDAR TEST images plus 123 validation images, shared across expert comparisons. Logical routing counts in each table are separate from those 856 physical cache-generation inferences.

## Evidence and verification

External root: `/extra1/bmartelli/context-str-runs/essential-router-icdar-transfer-20261028-v1`.

Primary U-Net and cross-expert-transfer analysis: `threshold_analysis_v1`. Normalized control: `threshold_analysis_normalized_v2`; its failed import-only predecessor is preserved. Matched YOLO: `yolo_head_v1` and `threshold_analysis_yolo_matched_v1`. Each analysis saves validation choices before TEST evaluation, complete score/count curves, CSV frontiers, bootstrap bands, machine-readable reports and completion hashes. `conditional_oracle_analysis.json` contains privileged bounds and exact residual certificates.

`independent_audit_final.json` passes all 733 new TEST image/annotation replays, 5496 selected crop-pixel checks, 1466 independent expert count/head replays, all 1028 matched-head training observations/targets and an independently solved weighted-Ridge system (maximum coefficient error 1.30e-13). It replays 885 validation cuts, 4936 TEST cuts, 150 validation objectives, 270 frozen-policy TEST cases and 30 complete bootstrap bands using an independent exact F1-gain identity.

`profile_oracle_audit.json` also passes all 2066 dataset/expert TEST profile replays, 8264 single-image policy decisions, 24 aggregate deployment metrics and 12 exact privileged-oracle optimality certificates. Maximum deployment score discrepancy is 3.56e-15. Shared-profile exports contain the identical independently audited head and cutoff for every dataset; no dataset identity changes the threshold. The final focused suite passes 48 tests.

Implementation: `recognition/essential_threshold_frontier.py`, `recognition/essential_threshold_deployment.py`, `scripts/analyze_essential_router_thresholds.py`, `scripts/fit_essential_yolo_head.py`, `scripts/analyze_essential_fallback_control.py`, `scripts/analyze_essential_oracle.py`, `scripts/audit_essential_thresholds.py`, `scripts/audit_essential_profiles_oracle.py`, `scripts/plot_essential_thresholds.py`.

Configurations: `configs/approximate/essential_router_threshold_analysis_v1.json`, `configs/approximate/essential_router_normalized_threshold_analysis_v1.json`, `configs/approximate/essential_yolo_matched_head_v1.json`.

Tests: `tests/recognition/test_essential_threshold_frontier.py`, `tests/recognition/test_essential_threshold_deployment.py`, `tests/recognition/test_essential_conditional_oracle.py`, `tests/recognition/test_essential_icdar_transfer.py`.

Standalone figures: [PNG](/extra1/bmartelli/context-str-runs/essential-router-icdar-transfer-20261028-v1/figures/essential_router_frontiers.png), [SVG](/extra1/bmartelli/context-str-runs/essential-router-icdar-transfer-20261028-v1/figures/essential_router_frontiers.svg). The counts and percentages underlying every curve are in each `test_frontiers.csv`.
