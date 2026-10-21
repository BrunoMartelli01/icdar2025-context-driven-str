# YOLO/PARSeq router comparison: train / equal

approximate reproduction

Total-Text; TRAIN holdout. Within-image one-to-one word multiset exact Unicode casefold; punctuation retained; NO spatial IoU; native256 evaluation-only ignore as recorded in the original report.

| variant | tp | predictions | gt | precision | recall | f1 | tn0 | accepted_images | fallback_images | gained_vs_yolo | lost_vs_yolo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yolo_only | 626 | 966 | 983 | 64.80% | 63.68% | 64.24% | 47.32% | 125 | 0 | 0 | 0 |
| deepsolo_only | 734 | 875 | 983 | 83.89% | 74.67% | 79.01% | 65.30% | 0 | 125 | 138 | 30 |
| clip/same_budget_62 | 665 | 845 | 983 | 78.70% | 67.65% | 72.76% | 57.18% | 62 | 63 | 60 | 21 |
| paper_selected/same_budget_62 | 681 | 892 | 983 | 76.35% | 69.28% | 72.64% | 57.04% | 62 | 63 | 79 | 24 |
| joined_t3/same_budget_62 | 681 | 892 | 983 | 76.35% | 69.28% | 72.64% | 57.04% | 62 | 63 | 76 | 21 |
| native/same_budget_62 | 680 | 870 | 983 | 78.16% | 69.18% | 73.39% | 57.97% | 62 | 63 | 73 | 19 |
| joint/same_budget_62 | 672 | 860 | 983 | 78.14% | 68.36% | 72.92% | 57.39% | 62 | 63 | 67 | 21 |

P=TP/predictions; R=TP/GT; F1=2TP/(predictions+GT); TN0=TP/(TP+FP+FN).

Original evidence: `external_artifacts:yolo-paired-routers-v1/full/results.json`; SHA256 `4d452a6462c199d94ddd630db99d71c4230d2e4f63cc792085e9fd90a9681de1`.

- Standalone rows are reference controls with different cost.
- Equal-budget points are descriptive rank budgets, not transferred scalar thresholds.
- Not official polygon matching or Table5. Previously explored TEST and single-seed training do not establish generalization.

This is a saved numerical summary, not a new evaluation. Its checksum is verified locally; recheck original evidence with `python3 -m research compare yolo-train-equal`.
