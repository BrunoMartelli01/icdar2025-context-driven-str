# Native128 checkpoints, bbox-area150, padding0 native pixels

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

Original evidence: `repository:artifacts/diagnostics/native128-gt-review-20261016/full/results.json`; SHA256 `a3c08d4a29406f77bf46fbe4970ab2f2a00c845045c67fdd80768da985b28ef4`.

- Post-hoc repeated TEST exploration; best padding must not be reported as an untouched test selection.
- 15 files,14 tensor states: ICDAR13 A/B are identical weights, not independent replications.
- Uniform inference does not equal uniform training data, targets or checkpoint selection.
- No router, DeepSolo fallback or Table5 evaluation in this group.

This is a saved numerical summary, not a new evaluation. Its checksum is verified locally; recheck original evidence with `python3 -m research compare native128-pad0`.
