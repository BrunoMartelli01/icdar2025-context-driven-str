# Jointly trained kernel: extracted single output and padding

approximate reproduction

Total-Text; TRAIN holdout. Maximum-cardinality one-to-one original RGB exclusive bbox IoU>=0.50 AND exact Unicode casefold; punctuation retained; micro counts over all care words. Ignore policy remains source-specific.

| variant | tp | predictions | gt | precision | recall | f1 | tn0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| kernel_only/pad0 | 249 | 976 | 983 | 25.51% | 25.33% | 25.42% | 14.56% |
| kernel_only/pad12 | 107 | 976 | 983 | 10.96% | 10.89% | 10.92% | 5.78% |
| kernel_only/pad2 | 483 | 976 | 983 | 49.49% | 49.14% | 49.31% | 32.72% |
| kernel_only/pad4 | 422 | 976 | 983 | 43.24% | 42.93% | 43.08% | 27.46% |
| kernel_only/pad6 | 303 | 976 | 983 | 31.05% | 30.82% | 30.93% | 18.30% |
| kernel_only/pad8 | 231 | 976 | 983 | 23.67% | 23.50% | 23.58% | 13.37% |
| text_kernel/pad0 | 524 | 1038 | 983 | 50.48% | 53.31% | 51.86% | 35.00% |
| text_kernel/pad12 | 42 | 1038 | 983 | 4.05% | 4.27% | 4.16% | 2.12% |
| text_kernel/pad2 | 476 | 1038 | 983 | 45.86% | 48.42% | 47.11% | 30.81% |
| text_kernel/pad4 | 313 | 1038 | 983 | 30.15% | 31.84% | 30.97% | 18.33% |
| text_kernel/pad6 | 195 | 1038 | 983 | 18.79% | 19.84% | 19.30% | 10.68% |
| text_kernel/pad8 | 102 | 1038 | 983 | 9.83% | 10.38% | 10.09% | 5.32% |

P=TP/predictions; R=TP/GT; F1=2TP/(predictions+GT); TN0=TP/(TP+FP+FN).

Original evidence: `runs:kernel-only-padding-v1-20261016/results.json`; SHA256 `7919ab722c3cbc7156225e2e3aa297b20344ca166b88cf84ee6a4eef4962b557`.

- Single seed and repeatedly reused TRAIN holdout; not independent validation or official TEST.
- No paper-table or STRICT claim. Compare against the control from this exact source; equal denominators alone do not establish equal protocols.
- Padding is in native-mask pixels; extraction from joint training and a newly initialized network are separate experiments.

This is a saved numerical summary, not a new evaluation. Its checksum is verified locally; recheck original evidence with `python3 -m research compare kernel-extracted`.
