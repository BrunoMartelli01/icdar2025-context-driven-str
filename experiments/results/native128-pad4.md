# Native128 checkpoints, bbox-area150, padding4 native pixels

approximate reproduction

Total-Text; TEST official / post-hoc exploration. Unpadded original bbox IoU>=0.50 + exact casefold; ALL predictions count (no ignore neutralization). RETR_EXTERNAL, bbox-area150 at128, padded-bbox-area ordering, cap10; historical repository variant, not paper component-area filtering.

| variant | tp | predictions | gt | precision | recall | f1 | tn0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| compact_126-a150 | 373 | 797 | 2204 | 46.80% | 16.92% | 24.86% | 14.19% |
| compact_361-a150 | 425 | 864 | 2204 | 49.19% | 19.28% | 27.71% | 16.08% |
| compact_382-a150 | 381 | 730 | 2204 | 52.19% | 17.29% | 25.97% | 14.92% |
| compact_500-a150 | 345 | 659 | 2204 | 52.35% | 15.65% | 24.10% | 13.70% |
| curved-a150 | 344 | 904 | 2204 | 38.05% | 15.61% | 22.14% | 12.45% |
| pixel_ctw-a150 | 177 | 778 | 2204 | 22.75% | 8.03% | 11.87% | 6.31% |
| pixel_icdar13_a-a150 | 46 | 361 | 2204 | 12.74% | 2.09% | 3.59% | 1.83% |
| pixel_icdar13_b-a150 | 46 | 361 | 2204 | 12.74% | 2.09% | 3.59% | 1.83% |
| pixel_icdar15-a150 | 214 | 727 | 2204 | 29.44% | 9.71% | 14.60% | 7.88% |
| pixel_mix-a150 | 225 | 825 | 2204 | 27.27% | 10.21% | 14.86% | 8.02% |
| pixel_tt-a150 | 98 | 625 | 2204 | 15.68% | 4.45% | 6.93% | 3.59% |
| sep_447-a150 | 397 | 914 | 2204 | 43.44% | 18.01% | 25.47% | 14.59% |
| sep_485-a150 | 380 | 938 | 2204 | 40.51% | 17.24% | 24.19% | 13.76% |
| sep_500-a150 | 349 | 811 | 2204 | 43.03% | 15.83% | 23.15% | 13.09% |
| v4-a150 | 345 | 873 | 2204 | 39.52% | 15.65% | 22.42% | 12.63% |

P=TP/predictions; R=TP/GT; F1=2TP/(predictions+GT); TN0=TP/(TP+FP+FN).

Original evidence: `repository:artifacts/diagnostics/native128-gt-review-20261016/full/results.json`; SHA256 `a3c08d4a29406f77bf46fbe4970ab2f2a00c845045c67fdd80768da985b28ef4`.

- Post-hoc repeated TEST exploration; best padding must not be reported as an untouched test selection.
- 15 files,14 tensor states: ICDAR13 A/B are identical weights, not independent replications.
- Uniform inference does not equal uniform training data, targets or checkpoint selection.
- No router, DeepSolo fallback or Table5 evaluation in this group.

This is a saved numerical summary, not a new evaluation. Its checksum is verified locally; recheck original evidence with `python3 -m research compare native128-pad4`.
