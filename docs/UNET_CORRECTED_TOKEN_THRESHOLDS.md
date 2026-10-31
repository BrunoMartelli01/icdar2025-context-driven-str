# Frozen TRAIN thresholds and the 40-to-50 percent acceptance loss

Approximate reproduction. Full TotalText TEST300 / 2204 care words, symbol-free lexical micro F1, original native256 ignore; no IoU. Current expert: qualified corrected U-Net/PARSeq. Primary fallback: frozen native_word DeepSolo. All routing counts refer to cached logical decisions; no online latency or new DeepSolo inference.

## Threshold meaning and transfer

A numeric threshold accepts U-Net when its predicted utility is at least the cutoff. Utility is 2 predicted deltaTP minus fitting-DeepSolo-F1 times predicted deltaN. It is not a correctness probability. The cuts are TRAIN OOF coverage quantiles, not thresholds optimized for a stated F1 loss or calibrated risk guarantee.

The original analysis pools held-out predictions from five fitted gates, selects epoch 5, derives the quantiles, and transfers the same numbers to a newly fitted all-TRAIN gate. No independent calibration of that final head was performed.

| TRAIN target acceptance | Score cutoff | TRAIN OOF DeepSolo images/total (%) | Final-gate TRAIN DeepSolo images/total (%), resubstitution | TEST DeepSolo images/total (%) | TEST F1 % |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10% | 1.888248 | 1128/1253 (90.02%) | 1236/1253 (98.64%) | 299/300 (99.67%) | 82.6818 |
| 20% | 1.386984 | 1002/1253 (79.97%) | 1179/1253 (94.09%) | 295/300 (98.33%) | 82.7406 |
| 30% | 0.995257 | 877/1253 (69.99%) | 1090/1253 (86.99%) | 285/300 (95.00%) | 82.5012 |
| 40% | 0.599553 | 752/1253 (60.02%) | 977/1253 (77.97%) | 273/300 (91.00%) | 82.2746 |
| 50% | 0.222338 | 627/1253 (50.04%) | 832/1253 (66.40%) | 246/300 (82.00%) | 82.1173 |

The final-gate TRAIN column uses images that fitted that gate; it diagnoses a model-dependent score shift and is not validation performance or a proposed recalibration.

| Score population | Eligible images | Mean score | Median score | 90th percentile |
| --- | ---: | ---: | ---: | ---: |
| train_oof | 1228 | -0.0975 | 0.2434 | 1.8933 |
| final_model_train_resubstitution | 1228 | -0.5265 | -0.1775 | 1.1285 |
| final_model_test | 295 | -1.0133 | -0.8989 | 0.5548 |

The median on eligible samples shifts from +0.2434 for pooled OOF TRAIN to -0.1775 for final-head TRAIN and -0.8989 for final-head TEST. The paired median final-minus-OOF shift on the same eligible TRAIN images is -0.5439. These comparisons diagnose distinct model and population shifts; they do not uniquely identify their causes.

| Frozen CV head | Held TRAIN median | Same-head TEST median | TEST DeepSolo images/total (%) at fixed cutoff 0.222338 |
| ---: | ---: | ---: | ---: |
| 0 | 0.4929 | 0.2538 | 152/300 (50.67%) |
| 1 | -0.3194 | -1.0001 | 223/300 (74.33%) |
| 2 | -0.5599 | -0.7613 | 218/300 (72.67%) |
| 3 | 0.0209 | -0.2952 | 195/300 (65.00%) |
| 4 | 1.6247 | 0.9424 | 89/300 (29.67%) |

Every row evaluates the same TEST images at the same numeric cutoff. This confirms substantial head-dependent calibration variability. It does not select or deploy any fold model or ensemble.

## Expert generalization boundary

| Expert and population | F1 % | DeepSolo images/total (%) |
| --- | ---: | ---: |
| U-Net TRAIN | 76.6294 | 0/1253 (0.00%) |
| Original DeepSolo TRAIN | 82.4384 | 1253/1253 (100.00%) |
| U-Net TEST | 66.2278 | 0/300 (0.00%) |
| Original DeepSolo TEST | 81.7308 | 300/300 (100.00%) |

TRAIN and TEST are different populations. The source U-Net, editor and calibration have historical TRAIN exposure; OOF training of the gate does not remove that exposure. This is consistent with an expert-specific generalization gap, not proof of a single causal mechanism.

## Exact loss from accepting more images

| U-Net acceptance transition | Additional correct words lost | Additional false positives | F1 delta, points | DeepSolo before | DeepSolo after |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0% to 10% | 15 | -2 | -0.3795 | 300/300 (100.00%) | 270/300 (90.00%) |
| 10% to 20% | 12 | -1 | -0.3175 | 270/300 (90.00%) | 240/300 (80.00%) |
| 20% to 30% | 26 | -4 | -0.6590 | 240/300 (80.00%) | 210/300 (70.00%) |
| 30% to 40% | 33 | +13 | -1.2015 | 210/300 (70.00%) | 180/300 (60.00%) |
| 40% to 50% | 46 | +16 | -1.6538 | 180/300 (60.00%) | 150/300 (50.00%) |

The quotas rank the same scores and force acceptance of a specified number of images. The 40% quota has score boundary -0.579141; 50% has boundary -0.928258. All 30 newly accepted images have negative scores. Only 77/300 TEST images have nonnegative scores. At 40%, 43 accepted images already have negative scores; at 50%, 73 do. A negative score predicts a disadvantage relative to the original training comparator; the quota overrides that sign.

| 40% to 50% subset: 30 images / 229 care words | TP | FP | FN | F1 % | DeepSolo images/subset (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| U-Net outputs on added subset | 151 | 41 | 78 | 71.7340 | 0/30 (0.00%) |
| Native DeepSolo on same subset | 197 | 25 | 32 | 87.3614 | 30/30 (100.00%) |

Those 30 images contain more care words than any preceding 30-image acceptance band. There are 17 harmful, 3 beneficial and 10 neutral replacements at the 40% system F1 reference. The full-population counts change from TP1658/FP278/FN546 to TP1612/FP294/FN592: 46 more missed words and 16 more false positives.

| Newly accepted image | Router score | Predicted deltaTP | Actual deltaTP | Contribution to full TEST F1 delta, points |
| --- | ---: | ---: | ---: | ---: |
| total_text:791 | -0.7826 | -0.818 | -12 | -0.2721 |
| total_text:579 | -0.8235 | -0.975 | -7 | -0.2627 |
| total_text:668 | -0.8625 | -1.493 | -5 | -0.1848 |
| total_text:1548 | -0.6640 | -0.360 | -4 | -0.1557 |
| total_text:999 | -0.6423 | -0.357 | -2 | -0.1168 |

Contributions describe the full transition from DeepSolo 180/300 (60%) to 150/300 (50%); they are not separate routing policies. The first three images explain 43.52% of the total net decline; the first five explain 59.99%. In image 791, U-Net emits 20 observations and obtains 19TP against 33 care words, while DeepSolo obtains 31TP.

For the newly accepted band, predicted deltaTP sums to -13.4721, versus actual -46 with native DeepSolo and -47 with the original training comparator. The mean predicted utility is -0.7787 per image; the native-comparator actual utility is about -2.24 at the original training reference. The router underestimates severe losses even where it predicts the correct negative sign. Mean per-image PARSeq confidence on the emitted U-Net words is 98.31%; that confidence does not measure missing-word coverage.

The exact contribution identity is deltaF1 = sum(2 deltaTP_i - F1_before deltaN_i)/(N_after + GT_total). Every per-image contribution was checked to sum to the saved transition under both fallback profiles. Replacing only the comparator with original DeepSolo still gives a -1.4371-point decline, from 80.0000% to 78.5629%, DeepSolo 180/300 (60%) to 150/300 (50%). The optimized fallback transfer increases the observed decline by 0.2167 points; it does not explain the main failure.

The 3000-draw paired RGB-cluster bootstrap interval for the 40-to-50% native transition is [-2.7220, -0.7190] F1 points, conditional on fixed models and routes. It excludes training and historical search uncertainty.

## What should change next

Freeze the deployed router first, then calibrate that exact head on disjoint unseen calibration data. Use the same DeepSolo profile to construct TRAIN supervision and TEST evaluation. Select acceptance against a declared accuracy-loss or benefit criterion rather than only a coverage quantile. Diagnose and supervise large TP losses and missing-word coverage; confidence in the words already read cannot certify complete transcription.

A nonnegative-utility guard would avoid forced negative substitutions, but it is insufficient by itself: the diagnostic score>=0 rule gives 81.6659% F1 with DeepSolo 223/300 (74.33%), below the all-native baseline 82.6540% with DeepSolo 300/300 (100%). No threshold, guard, new head or calibration policy has been promoted on the basis of this explored TEST.

## Validation

The original audit passes. All saved fixed-threshold counts replay exactly. Frozen fold/final head rescoring on CPU uses strict loading, no training and no DeepSolo calls. CPU/CUDA test score discrepancy is at most 0.000002552, with identical saved quota and threshold decisions. Model hashes, input hashes, raw per-image attribution and rescoring outputs remain beside this report.

Source experiment: [corrected U-Net token router](UNET_CORRECTED_TOKEN_ROUTER.md).

External diagnostic root: `/extra1/bmartelli/context-str-runs/unet-corrected-token-router-20261027-threshold-diagnosis-v1`.

Portable evidence: [threshold analysis record](../provenance/unet_corrected_token_thresholds_v1_analysis.json).

Scientific classifications: `UNET-CORRECTED-THRESHOLDS-001` in `PAPER_AMBIGUITIES.md`.
