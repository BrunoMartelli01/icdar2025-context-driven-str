# GT crop oracle: diagnostic only

approximate reproduction

Total-Text; TRAIN holdout. Maximum-cardinality one-to-one original RGB exclusive bbox IoU>=0.50 AND exact Unicode casefold; punctuation retained; micro counts over all care words. Ignore policy remains source-specific.

| variant | tp | predictions | gt | precision | recall | f1 | tn0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| oracle / gt_boxes | 838 | 983 | 983 | 85.25% | 85.25% | 85.25% | 74.29% |

P=TP/predictions; R=TP/GT; F1=2TP/(predictions+GT); TN0=TP/(TP+FP+FN).

Original evidence: `repository:artifacts/text-recall-refinements-v1-qualified/summary.json`; SHA256 `c076c22b713d9854ac663acb4e5bb90a2319ce6324aeb2f71a8ab445924d0ddc`.

- Single seed and repeatedly reused TRAIN holdout; not independent validation or official TEST.
- No paper-table or STRICT claim. Compare against the control from this exact source; equal denominators alone do not establish equal protocols.
- GT boxes supplied after the fact; this is not an inference method or fixed-identity recognizer accuracy. Fixed-identity diagnostic is separately linked in the original source.

This is a saved numerical summary, not a new evaluation. Its checksum is verified locally; recheck original evidence with `python3 -m research compare gt-crop-diagnostic`.
