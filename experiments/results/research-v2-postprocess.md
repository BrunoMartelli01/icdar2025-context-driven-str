# Corrective word research V2: postprocess

approximate reproduction

Total-Text; TRAIN holdout. Maximum-cardinality one-to-one original RGB exclusive bbox IoU>=0.50 AND exact Unicode casefold; punctuation retained; micro counts over all care words. Ignore policy remains source-specific.

Gained/lost baseline: `embedding50_hard`.

| variant | tp | predictions | gt | precision | recall | f1 | tn0 | epoch | gained | lost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| embedding50_hard | 498 | 1563 | 983 | 31.86% | 50.66% | 39.12% | 24.32% | 50 | 0 | 0 |
| embedding50_watershed | 500 | 1104 | 983 | 45.29% | 50.86% | 47.92% | 31.51% | 50 | 6 | 4 |
| embedding50_soft | 494 | 1104 | 983 | 44.75% | 50.25% | 47.34% | 31.01% | 50 | 6 | 10 |

P=TP/predictions; R=TP/GT; F1=2TP/(predictions+GT); TN0=TP/(TP+FP+FN).

Original evidence: `runs:unet-word-research-v2/comparison.json`; SHA256 `cc737b83301bf5d6b7b2f31a59ee5745dee77359c016c2393d68fcc021b0b010`.

- Single seed and repeatedly reused TRAIN holdout; not independent validation or official TEST.
- No paper-table or STRICT claim. Compare against the control from this exact source; equal denominators alone do not establish equal protocols.
- Identical embedding epoch50 maps; attribution is to extraction, not segmentation.

This is a saved numerical summary, not a new evaluation. Its checksum is verified locally; recheck original evidence with `python3 -m research compare research-v2-postprocess`.
