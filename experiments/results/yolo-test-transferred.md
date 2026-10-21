# YOLO/PARSeq router comparison: test / transferred

approximate reproduction

Total-Text; TEST official / explored. Within-image one-to-one word multiset exact Unicode casefold; punctuation retained; NO spatial IoU; native256 evaluation-only ignore as recorded in the original report.

| variant | tp | predictions | gt | precision | recall | f1 | tn0 | accepted_images | fallback_images | gained_vs_yolo | lost_vs_yolo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yolo_only | 1496 | 2143 | 2204 | 69.81% | 67.88% | 68.83% | 52.47% | 300 | 0 | 0 | 0 |
| deepsolo_only | 1595 | 2008 | 2204 | 79.43% | 72.37% | 75.74% | 60.95% | 0 | 300 | 226 | 127 |
| clip/train_coverage_0.5 | 1521 | 1969 | 2204 | 77.25% | 69.01% | 72.90% | 57.35% | 189 | 111 | 107 | 82 |
| paper_selected/train_coverage_0.5 | 1518 | 2077 | 2204 | 73.09% | 68.87% | 70.92% | 54.94% | 197 | 103 | 77 | 55 |
| joined_t3/train_coverage_0.5 | 1531 | 2081 | 2204 | 73.57% | 69.46% | 71.46% | 55.59% | 189 | 111 | 83 | 48 |
| native/train_coverage_0.5 | 1530 | 1999 | 2204 | 76.54% | 69.42% | 72.81% | 57.24% | 190 | 110 | 105 | 71 |
| joint/train_coverage_0.5 | 1511 | 1973 | 2204 | 76.58% | 68.56% | 72.35% | 56.68% | 164 | 136 | 101 | 86 |

P=TP/predictions; R=TP/GT; F1=2TP/(predictions+GT); TN0=TP/(TP+FP+FN).

Original evidence: `external_artifacts:yolo-paired-routers-v1/full/results.json`; SHA256 `4d452a6462c199d94ddd630db99d71c4230d2e4f63cc792085e9fd90a9681de1`.

- Standalone rows are reference controls with different cost.
- TRAIN-derived scalar cuts transfer to different TEST acceptance counts; costs are not equal.
- Not official polygon matching or Table5. Previously explored TEST and single-seed training do not establish generalization.

This is a saved numerical summary, not a new evaluation. Its checksum is verified locally; recheck original evidence with `python3 -m research compare yolo-test-transferred`.
