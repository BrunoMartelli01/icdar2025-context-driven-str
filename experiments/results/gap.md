# Text+Kernel+Gap matched experiment

approximate reproduction

Total-Text; TRAIN holdout. Maximum-cardinality one-to-one original RGB exclusive bbox IoU>=0.50 AND exact Unicode casefold; punctuation retained; micro counts over all care words. Ignore policy remains source-specific.

| variant | tp | predictions | gt | precision | recall | f1 | tn0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| control | 524 | 1038 | 983 | 50.48% | 53.31% | 51.86% | 35.00% |
| gap | 521 | 1035 | 983 | 50.34% | 53.00% | 51.64% | 34.80% |
| gap_disabled | 521 | 1035 | 983 | 50.34% | 53.00% | 51.64% | 34.80% |

P=TP/predictions; R=TP/GT; F1=2TP/(predictions+GT); TN0=TP/(TP+FP+FN).

Original evidence: `runs:adaspot-gap-v1-20261015T125256Z/ocr_comparison.json`; SHA256 `b40d2fdc9f46d7f7c6bcd1a946db25d6a8bfeb355d0460ac5feedd43852fd20d`.

- Single seed and repeatedly reused TRAIN holdout; not independent validation or official TEST.
- No paper-table or STRICT claim. Compare against the control from this exact source; equal denominators alone do not establish equal protocols.

This is a saved numerical summary, not a new evaluation. Its checksum is verified locally; recheck original evidence with `python3 -m research compare gap`.
