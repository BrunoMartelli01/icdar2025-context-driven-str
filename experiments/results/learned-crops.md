# Learned Crops: selected thresholds

approximate reproduction

Total-Text; TRAIN holdout. Maximum-cardinality one-to-one original RGB exclusive bbox IoU>=0.50 AND exact Unicode casefold; punctuation retained; micro counts over all care words. Ignore policy remains source-specific. Ignore only wholly invalid post-localization instances; partial overlap remains care.

| variant | tp | predictions | gt | precision | recall | f1 | tn0 | threshold | geometry_hmean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control512 / kernel_watershed | 501 | 1070 | 983 | 46.82% | 50.97% | 48.81% | 32.28% | 0.5 | 0.6264003896736483 |
| control512 / offset_votes_v2 | 348 | 1074 | 983 | 32.40% | 35.40% | 33.84% | 20.36% | 0.5 | 0.4725328147788041 |
| control768 / kernel_watershed | 499 | 1126 | 983 | 44.32% | 50.76% | 47.32% | 30.99% | 0.65 | 0.6126126126126126 |
| control768 / offset_votes_v2 | 335 | 1083 | 983 | 30.93% | 34.08% | 32.43% | 19.35% | 0.65 | 0.46950629235237173 |
| tversky512 / kernel_watershed | 501 | 1070 | 983 | 46.82% | 50.97% | 48.81% | 32.28% | 0.5 | 0.6264003896736483 |
| tversky512 / offset_votes_v2 | 348 | 1074 | 983 | 32.40% | 35.40% | 33.84% | 20.36% | 0.5 | 0.4725328147788041 |
| tversky768 / kernel_watershed | 525 | 1143 | 983 | 45.93% | 53.41% | 49.39% | 32.79% | 0.9 | 0.6180620884289746 |
| tversky768 / offset_votes_v2 | 358 | 1161 | 983 | 30.84% | 36.42% | 33.40% | 20.04% | 0.9 | 0.46175373134328357 |

P=TP/predictions; R=TP/GT; F1=2TP/(predictions+GT); TN0=TP/(TP+FP+FN).

Original evidence: `repository:artifacts/text-recall-learned-crops-v1/summary.json`; SHA256 `c023bb79264181992a99e8f063514bd598ba68d61b806c32f03f5be36068eba3`.

- Single seed and repeatedly reused TRAIN holdout; not independent validation or official TEST.
- No paper-table or STRICT claim. Compare against the control from this exact source; equal denominators alone do not establish equal protocols.

This is a saved numerical summary, not a new evaluation. Its checksum is verified locally; recheck original evidence with `python3 -m research compare learned-crops`.
