# Native128 checkpoints, bbox-area150, padding2 native pixels

approximate reproduction

Total-Text; TEST official / post-hoc exploration. Unpadded original bbox IoU>=0.50 + exact casefold; ALL predictions count (no ignore neutralization). RETR_EXTERNAL, bbox-area150 at128, padded-bbox-area ordering, cap10; historical repository variant, not paper component-area filtering.

| variant | tp | predictions | gt | precision | recall | f1 | tn0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| compact_126-a150 | 387 | 797 | 2204 | 48.56% | 17.56% | 25.79% | 14.80% |
| compact_361-a150 | 432 | 864 | 2204 | 50.00% | 19.60% | 28.16% | 16.39% |
| compact_382-a150 | 394 | 730 | 2204 | 53.97% | 17.88% | 26.86% | 15.51% |
| compact_500-a150 | 360 | 659 | 2204 | 54.63% | 16.33% | 25.15% | 14.38% |
| curved-a150 | 349 | 904 | 2204 | 38.61% | 15.83% | 22.46% | 12.65% |
| pixel_ctw-a150 | 181 | 778 | 2204 | 23.26% | 8.21% | 12.14% | 6.46% |
| pixel_icdar13_a-a150 | 41 | 361 | 2204 | 11.36% | 1.86% | 3.20% | 1.62% |
| pixel_icdar13_b-a150 | 41 | 361 | 2204 | 11.36% | 1.86% | 3.20% | 1.62% |
| pixel_icdar15-a150 | 207 | 727 | 2204 | 28.47% | 9.39% | 14.12% | 7.60% |
| pixel_mix-a150 | 230 | 825 | 2204 | 27.88% | 10.44% | 15.19% | 8.22% |
| pixel_tt-a150 | 98 | 625 | 2204 | 15.68% | 4.45% | 6.93% | 3.59% |
| sep_447-a150 | 421 | 914 | 2204 | 46.06% | 19.10% | 27.00% | 15.61% |
| sep_485-a150 | 395 | 938 | 2204 | 42.11% | 17.92% | 25.14% | 14.38% |
| sep_500-a150 | 356 | 811 | 2204 | 43.90% | 16.15% | 23.62% | 13.39% |
| v4-a150 | 354 | 873 | 2204 | 40.55% | 16.06% | 23.01% | 13.00% |

P=TP/predictions; R=TP/GT; F1=2TP/(predictions+GT); TN0=TP/(TP+FP+FN).

Original evidence: `repository:artifacts/diagnostics/native128-gt-review-20261016/full/results.json`; SHA256 `a3c08d4a29406f77bf46fbe4970ab2f2a00c845045c67fdd80768da985b28ef4`.

- Post-hoc repeated TEST exploration; best padding must not be reported as an untouched test selection.
- 15 files,14 tensor states: ICDAR13 A/B are identical weights, not independent replications.
- Uniform inference does not equal uniform training data, targets or checkpoint selection.
- No router, DeepSolo fallback or Table5 evaluation in this group.

This is a saved numerical summary, not a new evaluation. Its checksum is verified locally; recheck original evidence with `python3 -m research compare native128-pad2`.
