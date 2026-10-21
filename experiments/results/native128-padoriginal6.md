# Native128 checkpoints, bbox-area150, 6 original RGB pixels

approximate reproduction

Total-Text; TEST official / post-hoc exploration. Unpadded original bbox IoU>=0.50 + exact casefold; ALL predictions count (no ignore neutralization). RETR_EXTERNAL, bbox-area150 at128, padded-bbox-area ordering, cap10; historical repository variant, not paper component-area filtering.

| variant | tp | predictions | gt | precision | recall | f1 | tn0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| compact_126-a150 | 385 | 797 | 2204 | 48.31% | 17.47% | 25.66% | 14.72% |
| compact_361-a150 | 434 | 864 | 2204 | 50.23% | 19.69% | 28.29% | 16.48% |
| compact_382-a150 | 386 | 730 | 2204 | 52.88% | 17.51% | 26.31% | 15.15% |
| compact_500-a150 | 351 | 659 | 2204 | 53.26% | 15.93% | 24.52% | 13.97% |
| curved-a150 | 352 | 904 | 2204 | 38.94% | 15.97% | 22.65% | 12.77% |
| pixel_ctw-a150 | 179 | 778 | 2204 | 23.01% | 8.12% | 12.01% | 6.39% |
| pixel_icdar13_a-a150 | 37 | 361 | 2204 | 10.25% | 1.68% | 2.88% | 1.46% |
| pixel_icdar13_b-a150 | 37 | 361 | 2204 | 10.25% | 1.68% | 2.88% | 1.46% |
| pixel_icdar15-a150 | 206 | 727 | 2204 | 28.34% | 9.35% | 14.06% | 7.56% |
| pixel_mix-a150 | 225 | 825 | 2204 | 27.27% | 10.21% | 14.86% | 8.02% |
| pixel_tt-a150 | 94 | 625 | 2204 | 15.04% | 4.26% | 6.65% | 3.44% |
| sep_447-a150 | 414 | 914 | 2204 | 45.30% | 18.78% | 26.56% | 15.31% |
| sep_485-a150 | 394 | 938 | 2204 | 42.00% | 17.88% | 25.08% | 14.34% |
| sep_500-a150 | 354 | 811 | 2204 | 43.65% | 16.06% | 23.48% | 13.30% |
| v4-a150 | 361 | 873 | 2204 | 41.35% | 16.38% | 23.46% | 13.29% |

P=TP/predictions; R=TP/GT; F1=2TP/(predictions+GT); TN0=TP/(TP+FP+FN).

Original evidence: `repository:artifacts/diagnostics/native128-gt-review-20261016/full/results.json`; SHA256 `a3c08d4a29406f77bf46fbe4970ab2f2a00c845045c67fdd80768da985b28ef4`.

- Post-hoc repeated TEST exploration; best padding must not be reported as an untouched test selection.
- 15 files,14 tensor states: ICDAR13 A/B are identical weights, not independent replications.
- Uniform inference does not equal uniform training data, targets or checkpoint selection.
- No router, DeepSolo fallback or Table5 evaluation in this group.

This is a saved numerical summary, not a new evaluation. Its checksum is verified locally; recheck original evidence with `python3 -m research compare native128-padoriginal6`.
