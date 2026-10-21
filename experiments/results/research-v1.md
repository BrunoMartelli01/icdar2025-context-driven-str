# Word research V1: uncapped crop variants

approximate reproduction

Total-Text; TRAIN holdout. Maximum-cardinality one-to-one original RGB exclusive bbox IoU>=0.50 AND exact Unicode casefold; punctuation retained; micro counts over all care words. Ignore policy remains source-specific.

| variant | tp | predictions | gt | precision | recall | f1 | tn0 | gained_vs_control | lost_vs_control |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adaptive_threshold | 505 | 1052 | 983 | 48.00% | 51.37% | 49.63% | 33.01% | 19 | 34 |
| control | 520 | 1037 | 983 | 50.14% | 52.90% | 51.49% | 34.67% | 0 | 0 |
| embedding | 481 | 1033 | 983 | 46.56% | 48.93% | 47.72% | 31.34% | 27 | 66 |
| glyph_word | 517 | 1042 | 983 | 49.62% | 52.59% | 51.06% | 34.28% | 5 | 8 |
| quality | 480 | 1029 | 983 | 46.65% | 48.83% | 47.71% | 31.33% | 26 | 66 |

P=TP/predictions; R=TP/GT; F1=2TP/(predictions+GT); TN0=TP/(TP+FP+FN).

Original evidence: `runs:unet-word-research-v1/comparison.json`; SHA256 `c6704de0e313a9f419f8d3aca7060a69e3f8f669e39b0cc4f397881d81a9f88d`.

- Single seed and repeatedly reused TRAIN holdout; not independent validation or official TEST.
- No paper-table or STRICT claim. Compare against the control from this exact source; equal denominators alone do not establish equal protocols.

This is a saved numerical summary, not a new evaluation. Its checksum is verified locally; recheck original evidence with `python3 -m research compare research-v1`.
