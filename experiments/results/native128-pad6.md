# Native128 checkpoints, bbox-area150, padding6 native pixels

approximate reproduction

Total-Text; TEST official / post-hoc exploration. Unpadded original bbox IoU>=0.50 + exact casefold; ALL predictions count (no ignore neutralization). RETR_EXTERNAL, bbox-area150 at128, padded-bbox-area ordering, cap10; historical repository variant, not paper component-area filtering.

| variant | tp | predictions | gt | precision | recall | f1 | tn0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| compact_126-a150 | 344 | 797 | 2204 | 43.16% | 15.61% | 22.93% | 12.95% |
| compact_361-a150 | 380 | 864 | 2204 | 43.98% | 17.24% | 24.77% | 14.14% |
| compact_382-a150 | 361 | 730 | 2204 | 49.45% | 16.38% | 24.61% | 14.03% |
| compact_500-a150 | 331 | 659 | 2204 | 50.23% | 15.02% | 23.12% | 13.07% |
| curved-a150 | 304 | 904 | 2204 | 33.63% | 13.79% | 19.56% | 10.84% |
| pixel_ctw-a150 | 165 | 778 | 2204 | 21.21% | 7.49% | 11.07% | 5.86% |
| pixel_icdar13_a-a150 | 43 | 361 | 2204 | 11.91% | 1.95% | 3.35% | 1.70% |
| pixel_icdar13_b-a150 | 43 | 361 | 2204 | 11.91% | 1.95% | 3.35% | 1.70% |
| pixel_icdar15-a150 | 199 | 727 | 2204 | 27.37% | 9.03% | 13.58% | 7.28% |
| pixel_mix-a150 | 210 | 825 | 2204 | 25.45% | 9.53% | 13.87% | 7.45% |
| pixel_tt-a150 | 88 | 625 | 2204 | 14.08% | 3.99% | 6.22% | 3.21% |
| sep_447-a150 | 341 | 914 | 2204 | 37.31% | 15.47% | 21.87% | 12.28% |
| sep_485-a150 | 307 | 938 | 2204 | 32.73% | 13.93% | 19.54% | 10.83% |
| sep_500-a150 | 307 | 811 | 2204 | 37.85% | 13.93% | 20.36% | 11.34% |
| v4-a150 | 295 | 873 | 2204 | 33.79% | 13.38% | 19.17% | 10.60% |

P=TP/predictions; R=TP/GT; F1=2TP/(predictions+GT); TN0=TP/(TP+FP+FN).

Original evidence: `repository:artifacts/diagnostics/native128-gt-review-20261016/full/results.json`; SHA256 `a3c08d4a29406f77bf46fbe4970ab2f2a00c845045c67fdd80768da985b28ef4`.

- Post-hoc repeated TEST exploration; best padding must not be reported as an untouched test selection.
- 15 files,14 tensor states: ICDAR13 A/B are identical weights, not independent replications.
- Uniform inference does not equal uniform training data, targets or checkpoint selection.
- No router, DeepSolo fallback or Table5 evaluation in this group.

This is a saved numerical summary, not a new evaluation. Its checksum is verified locally; recheck original evidence with `python3 -m research compare native128-pad6`.
