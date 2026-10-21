# Refinements: selected thresholds

approximate reproduction

Total-Text; TRAIN holdout. Maximum-cardinality one-to-one original RGB exclusive bbox IoU>=0.50 AND exact Unicode casefold; punctuation retained; micro counts over all care words. Ignore policy remains source-specific. Ignore only wholly invalid post-localization instances; partial overlap remains care.

| variant | tp | predictions | gt | precision | recall | f1 | tn0 | threshold | geometry_hmean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid256 / baseline | 525 | 1143 | 983 | 45.93% | 53.41% | 49.39% | 32.79% | 0.9 | 0.6180620884289746 |
| hybrid256 / bounded_growth | 524 | 1173 | 983 | 44.67% | 53.31% | 48.61% | 32.11% | 0.9 | 0.6428571428571429 |
| hybrid256 / refined | 523 | 1127 | 983 | 46.41% | 53.20% | 49.57% | 32.96% | 0.9 | 0.6218009478672986 |
| hybrid256 / seeded_only | 514 | 960 | 983 | 53.54% | 52.29% | 52.91% | 35.97% | 0.9 | 0.6577457539886773 |
| hybrid256 / soft_fallback | 514 | 968 | 983 | 53.10% | 52.29% | 52.69% | 35.77% | 0.9 | 0.6560738083034341 |
| hybrid384 / baseline | 528 | 1213 | 983 | 43.53% | 53.71% | 48.09% | 31.65% | 0.9 | 0.6074681238615665 |
| hybrid384 / bounded_growth | 530 | 1234 | 983 | 42.95% | 53.92% | 47.81% | 31.42% | 0.9 | 0.6260712674785747 |
| hybrid384 / refined | 525 | 1196 | 983 | 43.90% | 53.41% | 48.19% | 31.74% | 0.9 | 0.6112895823772373 |
| hybrid384 / seeded_only | 516 | 979 | 983 | 52.71% | 52.49% | 52.60% | 35.68% | 0.9 | 0.6605504587155964 |
| hybrid384 / soft_fallback | 517 | 990 | 983 | 52.22% | 52.59% | 52.41% | 35.51% | 0.9 | 0.6588950836289914 |
| reference512 / baseline | 520 | 1037 | 983 | 50.14% | 52.90% | 51.49% | 34.67% | 0.65 | 0.6405940594059406 |
| reference512 / bounded_growth | 521 | 1044 | 983 | 49.90% | 53.00% | 51.41% | 34.59% | 0.65 | 0.6452886038480513 |
| reference512 / refined | 520 | 1035 | 983 | 50.24% | 52.90% | 51.54% | 34.71% | 0.65 | 0.6412289395441031 |
| reference512 / seeded_only | 520 | 1006 | 983 | 51.69% | 52.90% | 52.29% | 35.40% | 0.65 | 0.6505781799899447 |
| reference512 / soft_fallback | 520 | 1006 | 983 | 51.69% | 52.90% | 52.29% | 35.40% | 0.65 | 0.6505781799899447 |

P=TP/predictions; R=TP/GT; F1=2TP/(predictions+GT); TN0=TP/(TP+FP+FN).

Original evidence: `repository:artifacts/text-recall-refinements-v1-qualified/summary.json`; SHA256 `c076c22b713d9854ac663acb4e5bb90a2319ce6324aeb2f71a8ab445924d0ddc`.

- Single seed and repeatedly reused TRAIN holdout; not independent validation or official TEST.
- No paper-table or STRICT claim. Compare against the control from this exact source; equal denominators alone do not establish equal protocols.

This is a saved numerical summary, not a new evaluation. Its checksum is verified locally; recheck original evidence with `python3 -m research compare refinements`.
