# YOLO/PARSeq router comparison: test / equal

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

Original evidence: `external_artifacts:yolo-paired-routers-v1/full/results.json`; SHA256 `4d452a6462c199d94ddd630db99d71c4230d2e4f63cc792085e9fd90a9681de1`.

- Standalone rows are reference controls with different cost.
- Equal-budget points are descriptive rank budgets, not transferred scalar thresholds.
- Not official polygon matching or Table5. Previously explored TEST and single-seed training do not establish generalization.

This is a saved numerical summary, not a new evaluation. Its checksum is verified locally; recheck original evidence with `python3 -m research compare yolo-test-equal`.
