# Corrective word research V2: heads

approximate reproduction

Total-Text; TRAIN holdout. Maximum-cardinality one-to-one original RGB exclusive bbox IoU>=0.50 AND exact Unicode casefold; punctuation retained; micro counts over all care words. Ignore policy remains source-specific.

Gained/lost baseline: `reference_control`.

| variant | tp | predictions | gt | precision | recall | f1 | tn0 | epoch | gained | lost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference_control | 520 | 1037 | 983 | 50.14% | 52.90% | 51.49% | 34.67% | 2 | 0 | 0 |
| control | 520 | 1037 | 983 | 50.14% | 52.90% | 51.49% | 34.67% | 0 | 0 | 0 |
| db_core | 522 | 1050 | 983 | 49.71% | 53.10% | 51.35% | 34.55% | 14 | 5 | 3 |
| glyph_isolated | 520 | 1038 | 983 | 50.10% | 52.90% | 51.46% | 34.64% | 50 | 0 | 0 |
| glyph_staged | 520 | 1037 | 983 | 50.14% | 52.90% | 51.49% | 34.67% | 7 | 0 | 0 |
| links_isolated | 490 | 20200 | 983 | 2.43% | 49.85% | 4.63% | 2.37% | 33 | 20 | 50 |
| threshold_frozen | 520 | 1037 | 983 | 50.14% | 52.90% | 51.49% | 34.67% | 0 | 0 | 0 |

P=TP/predictions; R=TP/GT; F1=2TP/(predictions+GT); TN0=TP/(TP+FP+FN).

Original evidence: `runs:unet-word-research-v2/comparison.json`; SHA256 `cc737b83301bf5d6b7b2f31a59ee5745dee77359c016c2393d68fcc021b0b010`.

- Single seed and repeatedly reused TRAIN holdout; not independent validation or official TEST.
- No paper-table or STRICT claim. Compare against the control from this exact source; equal denominators alone do not establish equal protocols.
- Staged glyph also needs the continued control; no automatic model promotion.

This is a saved numerical summary, not a new evaluation. Its checksum is verified locally; recheck original evidence with `python3 -m research compare research-v2-heads`.
