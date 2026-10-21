# Word research V1: fixed top10 budget

approximate reproduction

Total-Text; TRAIN holdout. Maximum-cardinality one-to-one original RGB exclusive bbox IoU>=0.50 AND exact Unicode casefold; punctuation retained; micro counts over all care words. Ignore policy remains source-specific.

| variant | tp | predictions | gt | precision | recall | f1 | tn0 | gained_vs_control | lost_vs_control |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| quality_area10 | 429 | 816 | 983 | 52.57% | 43.64% | 47.69% | 31.31% | 22 | 113 |
| quality_rank10 | 438 | 813 | 983 | 53.87% | 44.56% | 48.78% | 32.25% | 21 | 103 |

P=TP/predictions; R=TP/GT; F1=2TP/(predictions+GT); TN0=TP/(TP+FP+FN).

Original evidence: `runs:unet-word-research-v1/comparison.json`; SHA256 `c6704de0e313a9f419f8d3aca7060a69e3f8f669e39b0cc4f397881d81a9f88d`.

- Single seed and repeatedly reused TRAIN holdout; not independent validation or official TEST.
- No paper-table or STRICT claim. Compare against the control from this exact source; equal denominators alone do not establish equal protocols.
- Top10 quality-area baseline only; do not compare its gain directly with uncapped520-word control.

This is a saved numerical summary, not a new evaluation. Its checksum is verified locally; recheck original evidence with `python3 -m research compare research-v1-top10`.
