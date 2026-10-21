# Word-center offsets: V1 and V2 crops

approximate reproduction

Total-Text; TRAIN holdout. Maximum-cardinality one-to-one original RGB exclusive bbox IoU>=0.50 AND exact Unicode casefold; punctuation retained; micro counts over all care words. Ignore policy remains source-specific.

| variant | tp | predictions | gt | precision | recall | f1 | tn0 | geometry_hmean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1 / baseline | 425 | 886 | 983 | 47.97% | 43.23% | 45.48% | 29.43% | 0.5906902086677368 |
| v1 / local_centers | 339 | 1978 | 983 | 17.14% | 34.49% | 22.90% | 12.93% | 0.3552853765619723 |
| v1 / global_centers | 334 | 1056 | 983 | 31.63% | 33.98% | 32.76% | 19.59% | 0.46101029916625796 |
| v1 / global_centers_expand | 335 | 1058 | 983 | 31.66% | 34.08% | 32.83% | 19.64% | 0.46839784419402253 |
| v2 / baseline | 425 | 886 | 983 | 47.97% | 43.23% | 45.48% | 29.43% | 0.5906902086677368 |
| v2 / local_centers | 309 | 2060 | 983 | 15.00% | 31.43% | 20.31% | 11.30% | 0.3181071311206047 |
| v2 / global_centers | 339 | 1063 | 983 | 31.89% | 34.49% | 33.14% | 19.86% | 0.46920821114369504 |
| v2 / global_centers_expand | 348 | 1065 | 983 | 32.68% | 35.40% | 33.98% | 20.47% | 0.4736328125 |

P=TP/predictions; R=TP/GT; F1=2TP/(predictions+GT); TN0=TP/(TP+FP+FN).

Original evidence: `repository:artifacts/center-guided-word-crops-v2/summary.json`; SHA256 `f08981d713688389f91c7acfbc1bc454e3e8df6c58eb87acf481c73bc6629e67`.

- Single seed and repeatedly reused TRAIN holdout; not independent validation or official TEST.
- No paper-table or STRICT claim. Compare against the control from this exact source; equal denominators alone do not establish equal protocols.

This is a saved numerical summary, not a new evaluation. Its checksum is verified locally; recheck original evidence with `python3 -m research compare center-votes`.
