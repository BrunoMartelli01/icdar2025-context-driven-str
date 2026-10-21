# Native128 checkpoints, bbox-area150, padding8 native pixels

approximate reproduction

Total-Text; TEST official / post-hoc exploration. Unpadded original bbox IoU>=0.50 + exact casefold; ALL predictions count (no ignore neutralization). RETR_EXTERNAL, bbox-area150 at128, padded-bbox-area ordering, cap10; historical repository variant, not paper component-area filtering.

| variant | tp | predictions | gt | precision | recall | f1 | tn0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| compact_126-a150 | 295 | 797 | 2204 | 37.01% | 13.38% | 19.66% | 10.90% |
| compact_361-a150 | 313 | 864 | 2204 | 36.23% | 14.20% | 20.40% | 11.36% |
| compact_382-a150 | 299 | 730 | 2204 | 40.96% | 13.57% | 20.38% | 11.35% |
| compact_500-a150 | 271 | 659 | 2204 | 41.12% | 12.30% | 18.93% | 10.46% |
| curved-a150 | 252 | 904 | 2204 | 27.88% | 11.43% | 16.22% | 8.82% |
| pixel_ctw-a150 | 152 | 778 | 2204 | 19.54% | 6.90% | 10.19% | 5.37% |
| pixel_icdar13_a-a150 | 35 | 361 | 2204 | 9.70% | 1.59% | 2.73% | 1.38% |
| pixel_icdar13_b-a150 | 35 | 361 | 2204 | 9.70% | 1.59% | 2.73% | 1.38% |
| pixel_icdar15-a150 | 169 | 727 | 2204 | 23.25% | 7.67% | 11.53% | 6.12% |
| pixel_mix-a150 | 183 | 825 | 2204 | 22.18% | 8.30% | 12.08% | 6.43% |
| pixel_tt-a150 | 66 | 625 | 2204 | 10.56% | 2.99% | 4.67% | 2.39% |
| sep_447-a150 | 282 | 914 | 2204 | 30.85% | 12.79% | 18.09% | 9.94% |
| sep_485-a150 | 252 | 938 | 2204 | 26.87% | 11.43% | 16.04% | 8.72% |
| sep_500-a150 | 252 | 811 | 2204 | 31.07% | 11.43% | 16.72% | 9.12% |
| v4-a150 | 244 | 873 | 2204 | 27.95% | 11.07% | 15.86% | 8.61% |

P=TP/predictions; R=TP/GT; F1=2TP/(predictions+GT); TN0=TP/(TP+FP+FN).

Original evidence: `repository:artifacts/diagnostics/native128-gt-review-20261016/full/results.json`; SHA256 `a3c08d4a29406f77bf46fbe4970ab2f2a00c845045c67fdd80768da985b28ef4`.

- Post-hoc repeated TEST exploration; best padding must not be reported as an untouched test selection.
- 15 files,14 tensor states: ICDAR13 A/B are identical weights, not independent replications.
- Uniform inference does not equal uniform training data, targets or checkpoint selection.
- No router, DeepSolo fallback or Table5 evaluation in this group.

This is a saved numerical summary, not a new evaluation. Its checksum is verified locally; recheck original evidence with `python3 -m research compare native128-pad8`.
