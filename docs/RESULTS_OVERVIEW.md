# Saved model comparisons

Approximate reproduction — read-only reporting snapshot.

Each table retains its own dataset, matching, ignore policy and cost boundary.
No global winner or paper reproduction is inferred. These are existing saved results; no new inference.

All groups: `python3 -m research groups`. Usage and limits: [research guide](README.md).

## Native128 checkpoints, bbox-area150, padding0 native pixels

approximate reproduction

Total-Text; TEST official / post-hoc exploration. Unpadded original bbox IoU>=0.50 + exact casefold; ALL predictions count (no ignore neutralization). RETR_EXTERNAL, bbox-area150 at128, padded-bbox-area ordering, cap10; historical repository variant, not paper component-area filtering.

| variant | tp | predictions | gt | precision | recall | f1 | tn0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| compact_126-a150 | 365 | 797 | 2204 | 45.80% | 16.56% | 24.33% | 13.85% |
| compact_361-a150 | 419 | 864 | 2204 | 48.50% | 19.01% | 27.31% | 15.82% |
| compact_382-a150 | 366 | 730 | 2204 | 50.14% | 16.61% | 24.95% | 14.25% |
| compact_500-a150 | 316 | 659 | 2204 | 47.95% | 14.34% | 22.07% | 12.41% |
| curved-a150 | 355 | 904 | 2204 | 39.27% | 16.11% | 22.84% | 12.90% |
| pixel_ctw-a150 | 182 | 778 | 2204 | 23.39% | 8.26% | 12.21% | 6.50% |
| pixel_icdar13_a-a150 | 35 | 361 | 2204 | 9.70% | 1.59% | 2.73% | 1.38% |
| pixel_icdar13_b-a150 | 35 | 361 | 2204 | 9.70% | 1.59% | 2.73% | 1.38% |
| pixel_icdar15-a150 | 195 | 727 | 2204 | 26.82% | 8.85% | 13.31% | 7.13% |
| pixel_mix-a150 | 221 | 825 | 2204 | 26.79% | 10.03% | 14.59% | 7.87% |
| pixel_tt-a150 | 94 | 625 | 2204 | 15.04% | 4.26% | 6.65% | 3.44% |
| sep_447-a150 | 420 | 914 | 2204 | 45.95% | 19.06% | 26.94% | 15.57% |
| sep_485-a150 | 401 | 938 | 2204 | 42.75% | 18.19% | 25.53% | 14.63% |
| sep_500-a150 | 352 | 811 | 2204 | 43.40% | 15.97% | 23.35% | 13.22% |
| v4-a150 | 355 | 873 | 2204 | 40.66% | 16.11% | 23.07% | 13.04% |

P=TP/predictions; R=TP/GT; F1=2TP/(predictions+GT); TN0=TP/(TP+FP+FN).

Source: [saved report](</extra1/bmartelli/projects/icdar2025-context-driven-str/artifacts/diagnostics/native128-gt-review-20261016/full/results.json>); SHA256 `a3c08d4a29406f77bf46fbe4970ab2f2a00c845045c67fdd80768da985b28ef4`.

- Post-hoc repeated TEST exploration; best padding must not be reported as an untouched test selection.
- 15 files,14 tensor states: ICDAR13 A/B are identical weights, not independent replications.
- Uniform inference does not equal uniform training data, targets or checkpoint selection.
- No router, DeepSolo fallback or Table5 evaluation in this group.

## Text / Text+Kernel / Text+Kernel+Center: native geometry

approximate reproduction

Total-Text; TRAIN holdout. Native instance bbox IoU>=0.50, geometry only, source care/ignore handling. Not OCR and not pixel F1.

Performance scope: Reported original TRAIN peaks: segmentation only, sequential CUDA FP32, batch8, [8,3,512,512], RTX3090 per original audit; not inference latency or RTX3070 qualification.

| variant | tp | predictions | gt | precision | recall | f1 | tn0 | epoch | parameters | train_peak_allocated_bytes | train_peak_reserved_bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| text | 551 | 886 | 983 | 62.19% | 56.05% | 58.96% | 41.81% | 56 | 58155081 | 4233335296 | 5354029056 |
| text_kernel | 625 | 1029 | 983 | 60.74% | 63.58% | 62.13% | 45.06% | 61 | 58155146 | 4236481536 | 5354029056 |
| text_kernel_center | 611 | 1109 | 983 | 55.09% | 62.16% | 58.41% | 41.26% | 66 | 58155211 | 4237008384 | 5354029056 |

P=TP/predictions; R=TP/GT; F1=2TP/(predictions+GT); TN0=TP/(TP+FP+FN).

Source: [saved report](</extra1/bmartelli/projects/icdar2025-context-driven-str/provenance/word_multitask_v1_completion_audit.json>); SHA256 `74eb529408c7083349fcceb1f5513fb457721f8db486ed843dd5de484d39838c`.

- Single seed and repeatedly reused TRAIN holdout; not independent validation or official TEST.
- No paper-table or STRICT claim. Compare against the control from this exact source; equal denominators alone do not establish equal protocols.

## Refinements: selected thresholds

approximate reproduction

Total-Text; TRAIN holdout. Maximum-cardinality one-to-one original RGB exclusive bbox IoU>=0.50 AND exact Unicode casefold; punctuation retained; micro counts over all care words. Ignore policy remains source-specific. Ignore only wholly invalid post-localization instances; partial overlap remains care.

| variant | tp | predictions | gt | precision | recall | f1 | tn0 | threshold | geometry_hmean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid256 / baseline | 525 | 1143 | 983 | 45.93% | 53.41% | 49.39% | 32.79% | 0.9 | 0.6180620884289746 |
| hybrid256 / bounded_growth | 524 | 1173 | 983 | 44.67% | 53.31% | 48.61% | 32.11% | 0.9 | 0.6428571428571429 |
| hybrid256 / refined | 523 | 1127 | 983 | 46.41% | 53.20% | 49.57% | 32.96% | 0.9 | 0.6218009478672986 |
| hybrid256 / seeded_only | 514 | 960 | 983 | 53.54% | 52.29% | 52.91% | 35.97% | 0.9 | 0.6577457539886773 |
| hybrid256 / soft_fallback | 514 | 968 | 983 | 53.10% | 52.29% | 52.69% | 35.77% | 0.9 | 0.6560738083034341 |
| hybrid384 / baseline | 528 | 1213 | 983 | 43.53% | 53.71% | 48.09% | 31.65% | 0.9 | 0.6074681238615665 |
| hybrid384 / bounded_growth | 530 | 1234 | 983 | 42.95% | 53.92% | 47.81% | 31.42% | 0.9 | 0.6260712674785747 |
| hybrid384 / refined | 525 | 1196 | 983 | 43.90% | 53.41% | 48.19% | 31.74% | 0.9 | 0.6112895823772373 |
| hybrid384 / seeded_only | 516 | 979 | 983 | 52.71% | 52.49% | 52.60% | 35.68% | 0.9 | 0.6605504587155964 |
| hybrid384 / soft_fallback | 517 | 990 | 983 | 52.22% | 52.59% | 52.41% | 35.51% | 0.9 | 0.6588950836289914 |
| reference512 / baseline | 520 | 1037 | 983 | 50.14% | 52.90% | 51.49% | 34.67% | 0.65 | 0.6405940594059406 |
| reference512 / bounded_growth | 521 | 1044 | 983 | 49.90% | 53.00% | 51.41% | 34.59% | 0.65 | 0.6452886038480513 |
| reference512 / refined | 520 | 1035 | 983 | 50.24% | 52.90% | 51.54% | 34.71% | 0.65 | 0.6412289395441031 |
| reference512 / seeded_only | 520 | 1006 | 983 | 51.69% | 52.90% | 52.29% | 35.40% | 0.65 | 0.6505781799899447 |
| reference512 / soft_fallback | 520 | 1006 | 983 | 51.69% | 52.90% | 52.29% | 35.40% | 0.65 | 0.6505781799899447 |

P=TP/predictions; R=TP/GT; F1=2TP/(predictions+GT); TN0=TP/(TP+FP+FN).

Source: [saved report](</extra1/bmartelli/projects/icdar2025-context-driven-str/artifacts/text-recall-refinements-v1-qualified/summary.json>); SHA256 `c076c22b713d9854ac663acb4e5bb90a2319ce6324aeb2f71a8ab445924d0ddc`.

- Single seed and repeatedly reused TRAIN holdout; not independent validation or official TEST.
- No paper-table or STRICT claim. Compare against the control from this exact source; equal denominators alone do not establish equal protocols.

## Jointly trained kernel: extracted single output and padding

approximate reproduction

Total-Text; TRAIN holdout. Maximum-cardinality one-to-one original RGB exclusive bbox IoU>=0.50 AND exact Unicode casefold; punctuation retained; micro counts over all care words. Ignore policy remains source-specific.

| variant | tp | predictions | gt | precision | recall | f1 | tn0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| kernel_only/pad0 | 249 | 976 | 983 | 25.51% | 25.33% | 25.42% | 14.56% |
| kernel_only/pad12 | 107 | 976 | 983 | 10.96% | 10.89% | 10.92% | 5.78% |
| kernel_only/pad2 | 483 | 976 | 983 | 49.49% | 49.14% | 49.31% | 32.72% |
| kernel_only/pad4 | 422 | 976 | 983 | 43.24% | 42.93% | 43.08% | 27.46% |
| kernel_only/pad6 | 303 | 976 | 983 | 31.05% | 30.82% | 30.93% | 18.30% |
| kernel_only/pad8 | 231 | 976 | 983 | 23.67% | 23.50% | 23.58% | 13.37% |
| text_kernel/pad0 | 524 | 1038 | 983 | 50.48% | 53.31% | 51.86% | 35.00% |
| text_kernel/pad12 | 42 | 1038 | 983 | 4.05% | 4.27% | 4.16% | 2.12% |
| text_kernel/pad2 | 476 | 1038 | 983 | 45.86% | 48.42% | 47.11% | 30.81% |
| text_kernel/pad4 | 313 | 1038 | 983 | 30.15% | 31.84% | 30.97% | 18.33% |
| text_kernel/pad6 | 195 | 1038 | 983 | 18.79% | 19.84% | 19.30% | 10.68% |
| text_kernel/pad8 | 102 | 1038 | 983 | 9.83% | 10.38% | 10.09% | 5.32% |

P=TP/predictions; R=TP/GT; F1=2TP/(predictions+GT); TN0=TP/(TP+FP+FN).

Source: [saved report](</extra1/bmartelli/context-str-runs/kernel-only-padding-v1-20261016/results.json>); SHA256 `7919ab722c3cbc7156225e2e3aa297b20344ca166b88cf84ee6a4eef4962b557`.

- Single seed and repeatedly reused TRAIN holdout; not independent validation or official TEST.
- No paper-table or STRICT claim. Compare against the control from this exact source; equal denominators alone do not establish equal protocols.
- Padding is in native-mask pixels; extraction from joint training and a newly initialized network are separate experiments.

## New Kernel-only U-Net: padding

approximate reproduction

Total-Text; TRAIN holdout. Maximum-cardinality one-to-one original RGB exclusive bbox IoU>=0.50 AND exact Unicode casefold; punctuation retained; micro counts over all care words. Ignore policy remains source-specific.

| variant | tp | predictions | gt | precision | recall | f1 | tn0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pad0 | 262 | 1056 | 983 | 24.81% | 26.65% | 25.70% | 14.74% |
| pad12 | 93 | 1056 | 983 | 8.81% | 9.46% | 9.12% | 4.78% |
| pad2 | 463 | 1056 | 983 | 43.84% | 47.10% | 45.41% | 29.38% |
| pad4 | 404 | 1056 | 983 | 38.26% | 41.10% | 39.63% | 24.71% |
| pad6 | 280 | 1056 | 983 | 26.52% | 28.48% | 27.46% | 15.92% |
| pad8 | 204 | 1056 | 983 | 19.32% | 20.75% | 20.01% | 11.12% |

P=TP/predictions; R=TP/GT; F1=2TP/(predictions+GT); TN0=TP/(TP+FP+FN).

Source: [saved report](</extra1/bmartelli/context-str-runs/kernel-only-fresh-v1-20261016/results.json>); SHA256 `d11e6ed5c4c37bd15acccb6e4dd24d246616e9abed5e37364ee469b1279ff43e`.

- Single seed and repeatedly reused TRAIN holdout; not independent validation or official TEST.
- No paper-table or STRICT claim. Compare against the control from this exact source; equal denominators alone do not establish equal protocols.
- Padding is in native-mask pixels; extraction from joint training and a newly initialized network are separate experiments.

## YOLO/PARSeq router comparison: test / equal

approximate reproduction

Total-Text; TEST official / explored. Within-image one-to-one word multiset exact Unicode casefold; punctuation retained; NO spatial IoU; native256 evaluation-only ignore as recorded in the original report.

| variant | tp | predictions | gt | precision | recall | f1 | tn0 | accepted_images | fallback_images | gained_vs_yolo | lost_vs_yolo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yolo_only | 1496 | 2143 | 2204 | 69.81% | 67.88% | 68.83% | 52.47% | 300 | 0 | 0 | 0 |
| deepsolo_only | 1595 | 2008 | 2204 | 79.43% | 72.37% | 75.74% | 60.95% | 0 | 300 | 226 | 127 |
| clip/same_budget_150 | 1530 | 1962 | 2204 | 77.98% | 69.42% | 73.45% | 58.04% | 150 | 150 | 135 | 101 |
| paper_selected/same_budget_150 | 1556 | 2056 | 2204 | 75.68% | 70.60% | 73.05% | 57.54% | 150 | 150 | 133 | 73 |
| joined_t3/same_budget_150 | 1571 | 2076 | 2204 | 75.67% | 71.28% | 73.41% | 57.99% | 150 | 150 | 136 | 61 |
| native/same_budget_150 | 1522 | 1972 | 2204 | 77.18% | 69.06% | 72.89% | 57.35% | 150 | 150 | 126 | 100 |
| joint/same_budget_150 | 1519 | 1974 | 2204 | 76.95% | 68.92% | 72.71% | 57.13% | 150 | 150 | 112 | 89 |

P=TP/predictions; R=TP/GT; F1=2TP/(predictions+GT); TN0=TP/(TP+FP+FN).

Source: [saved report](</extra1/bmartelli/context-str-artifacts/yolo-paired-routers-v1/full/results.json>); SHA256 `4d452a6462c199d94ddd630db99d71c4230d2e4f63cc792085e9fd90a9681de1`.

- Standalone rows are reference controls with different cost.
- Equal-budget points are descriptive rank budgets, not transferred scalar thresholds.
- Not official polygon matching or Table5. Previously explored TEST and single-seed training do not establish generalization.

## Long-CLIP B/L and LLM2CLIP: fixed margin_scene_en / same_budget_descriptive

approximate reproduction

Total-Text; TEST official / explored. Within-image one-to-one word multiset exact Unicode casefold; punctuation retained; NO spatial IoU; native256 evaluation-only ignore as recorded in the original report.

Performance scope: Feature seconds are the original model feature-export duration; not end-to-end latency or a matched timing benchmark.

| variant | tp | predictions | gt | precision | recall | f1 | tn0 | accepted_images | fallback_images | feature_seconds_reported |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llm2clip_l14_336 | 1503 | 2123 | 2204 | 70.80% | 68.19% | 69.47% | 53.22% | 149 | 151 | 1443.6267311596312 |
| longclip_b | 1478 | 2112 | 2204 | 69.98% | 67.06% | 68.49% | 52.08% | 149 | 151 | 245.9825597642921 |
| longclip_l | 1491 | 2101 | 2204 | 70.97% | 67.65% | 69.27% | 52.99% | 149 | 151 | 402.66949955327436 |

P=TP/predictions; R=TP/GT; F1=2TP/(predictions+GT); TN0=TP/(TP+FP+FN).

Source: [saved report](</extra1/bmartelli/context-str-artifacts/clip-specialized-comparison-v1/final/comparison.json>); SHA256 `98abc31f96f043123d941f96a7e8545967b2507699c169de1c0415794bd32681`.

- Fixed prompt family, not an assertion that it is every encoder's TRAIN-selected winner.
- Descriptive TEST budgets and transferred TRAIN cuts remain separate; acceptance counts are displayed.
- Feature timing scopes differ from OCR/training. No Table5 or strict claim.
