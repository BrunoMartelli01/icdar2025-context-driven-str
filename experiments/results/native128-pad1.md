# Native128 checkpoints, bbox-area150, padding1 native pixels

approximate reproduction

Total-Text; TEST official / post-hoc exploration. Unpadded original bbox IoU>=0.50 + exact casefold; ALL predictions count (no ignore neutralization). RETR_EXTERNAL, bbox-area150 at128, padded-bbox-area ordering, cap10; historical repository variant, not paper component-area filtering.

| variant | tp | predictions | gt | precision | recall | f1 | tn0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| compact_126-a150 | 388 | 797 | 2204 | 48.68% | 17.60% | 25.86% | 14.85% |
| compact_361-a150 | 437 | 864 | 2204 | 50.58% | 19.83% | 28.49% | 16.61% |
| compact_382-a150 | 392 | 730 | 2204 | 53.70% | 17.79% | 26.72% | 15.42% |
| compact_500-a150 | 355 | 659 | 2204 | 53.87% | 16.11% | 24.80% | 14.15% |
| curved-a150 | 349 | 904 | 2204 | 38.61% | 15.83% | 22.46% | 12.65% |
| pixel_ctw-a150 | 178 | 778 | 2204 | 22.88% | 8.08% | 11.94% | 6.35% |
| pixel_icdar13_a-a150 | 39 | 361 | 2204 | 10.80% | 1.77% | 3.04% | 1.54% |
| pixel_icdar13_b-a150 | 39 | 361 | 2204 | 10.80% | 1.77% | 3.04% | 1.54% |
| pixel_icdar15-a150 | 205 | 727 | 2204 | 28.20% | 9.30% | 13.99% | 7.52% |
| pixel_mix-a150 | 230 | 825 | 2204 | 27.88% | 10.44% | 15.19% | 8.22% |
| pixel_tt-a150 | 92 | 625 | 2204 | 14.72% | 4.17% | 6.50% | 3.36% |
| sep_447-a150 | 420 | 914 | 2204 | 45.95% | 19.06% | 26.94% | 15.57% |
| sep_485-a150 | 401 | 938 | 2204 | 42.75% | 18.19% | 25.53% | 14.63% |
| sep_500-a150 | 348 | 811 | 2204 | 42.91% | 15.79% | 23.08% | 13.05% |
| v4-a150 | 361 | 873 | 2204 | 41.35% | 16.38% | 23.46% | 13.29% |

P=TP/predictions; R=TP/GT; F1=2TP/(predictions+GT); TN0=TP/(TP+FP+FN).

Original evidence: `repository:artifacts/diagnostics/native128-gt-review-20261016/full/results.json`; SHA256 `a3c08d4a29406f77bf46fbe4970ab2f2a00c845045c67fdd80768da985b28ef4`.

- Post-hoc repeated TEST exploration; best padding must not be reported as an untouched test selection.
- 15 files,14 tensor states: ICDAR13 A/B are identical weights, not independent replications.
- Uniform inference does not equal uniform training data, targets or checkpoint selection.
- No router, DeepSolo fallback or Table5 evaluation in this group.

This is a saved numerical summary, not a new evaluation. Its checksum is verified locally; recheck original evidence with `python3 -m research compare native128-pad1`.
