# Learned crop word/kernel and pixel links

approximate reproduction

Total-Text; TRAIN holdout. Maximum-cardinality one-to-one original RGB exclusive bbox IoU>=0.50 AND exact Unicode casefold; punctuation retained; micro counts over all care words. Ignore policy remains source-specific.

Gained/lost baseline: `control`.

| variant | tp | predictions | gt | precision | recall | f1 | tn0 | epoch | gained | lost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | 521 | 1054 | 983 | 49.43% | 53.00% | 51.15% | 34.37% | 2 | 0 | 0 |
| crop_links | 476 | 18851 | 983 | 2.53% | 48.42% | 4.80% | 2.46% | 4 | 63 | 108 |
| crop_word | 524 | 1301 | 983 | 40.28% | 53.31% | 45.88% | 29.77% | 16 | 75 | 72 |
| global_links | 490 | 20298 | 983 | 2.41% | 49.85% | 4.61% | 2.36% | 33 | 20 | 51 |

P=TP/predictions; R=TP/GT; F1=2TP/(predictions+GT); TN0=TP/(TP+FP+FN).

Original evidence: `runs:unet-crop-links-v1/comparison.json`; SHA256 `e680d21bb67a8a15a6c9ee9b0d2d13900b282b21406794befc61a7b680f8f8f5`.

- Single seed and repeatedly reused TRAIN holdout; not independent validation or official TEST.
- No paper-table or STRICT claim. Compare against the control from this exact source; equal denominators alone do not establish equal protocols.

This is a saved numerical summary, not a new evaluation. Its checksum is verified locally; recheck original evidence with `python3 -m research compare crop-links`.
