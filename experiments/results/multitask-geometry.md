# Text / Text+Kernel / Text+Kernel+Center: native geometry

approximate reproduction

Total-Text; TRAIN holdout. Native instance bbox IoU>=0.50, geometry only, source care/ignore handling. Not OCR and not pixel F1.

Performance scope: Reported original TRAIN peaks: segmentation only, sequential CUDA FP32, batch8, [8,3,512,512], RTX3090 per original audit; not inference latency or RTX3070 qualification.

| variant | tp | predictions | gt | precision | recall | f1 | tn0 | epoch | parameters | train_peak_allocated_bytes | train_peak_reserved_bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| text | 551 | 886 | 983 | 62.19% | 56.05% | 58.96% | 41.81% | 56 | 58155081 | 4233335296 | 5354029056 |
| text_kernel | 625 | 1029 | 983 | 60.74% | 63.58% | 62.13% | 45.06% | 61 | 58155146 | 4236481536 | 5354029056 |
| text_kernel_center | 611 | 1109 | 983 | 55.09% | 62.16% | 58.41% | 41.26% | 66 | 58155211 | 4237008384 | 5354029056 |

P=TP/predictions; R=TP/GT; F1=2TP/(predictions+GT); TN0=TP/(TP+FP+FN).

Original evidence: `repository:provenance/word_multitask_v1_completion_audit.json`; SHA256 `74eb529408c7083349fcceb1f5513fb457721f8db486ed843dd5de484d39838c`.

- Single seed and repeatedly reused TRAIN holdout; not independent validation or official TEST.
- No paper-table or STRICT claim. Compare against the control from this exact source; equal denominators alone do not establish equal protocols.

This is a saved numerical summary, not a new evaluation. Its checksum is verified locally; recheck original evidence with `python3 -m research compare multitask-geometry`.
