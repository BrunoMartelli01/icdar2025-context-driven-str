# New Kernel-only U-Net: padding

approximate reproduction

Total-Text; TRAIN holdout. Maximum-cardinality one-to-one original RGB exclusive bbox IoU>=0.50 AND exact Unicode casefold; punctuation retained; micro counts over all care words. Ignore policy remains source-specific.

| variant | tp | predictions | gt | precision | recall | f1 | tn0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pad0 | 262 | 1056 | 983 | 24.81% | 26.65% | 25.70% | 14.74% |
| pad12 | 93 | 1056 | 983 | 8.81% | 9.46% | 9.12% | 4.78% |
| pad2 | 463 | 1056 | 983 | 43.84% | 47.10% | 45.41% | 29.38% |
| pad4 | 404 | 1056 | 983 | 38.26% | 41.10% | 39.63% | 24.71% |
| pad6 | 280 | 1056 | 983 | 26.52% | 28.48% | 27.46% | 15.92% |
| pad8 | 204 | 1056 | 983 | 19.32% | 20.75% | 20.01% | 11.12% |

P=TP/predictions; R=TP/GT; F1=2TP/(predictions+GT); TN0=TP/(TP+FP+FN).

Original evidence: `runs:kernel-only-fresh-v1-20261016/results.json`; SHA256 `d11e6ed5c4c37bd15acccb6e4dd24d246616e9abed5e37364ee469b1279ff43e`.

- Single seed and repeatedly reused TRAIN holdout; not independent validation or official TEST.
- No paper-table or STRICT claim. Compare against the control from this exact source; equal denominators alone do not establish equal protocols.
- Padding is in native-mask pixels; extraction from joint training and a newly initialized network are separate experiments.

This is a saved numerical summary, not a new evaluation. Its checksum is verified locally; recheck original evidence with `python3 -m research compare kernel-fresh`.
