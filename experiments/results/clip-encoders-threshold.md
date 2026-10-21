# Long-CLIP B/L and LLM2CLIP: fixed margin_scene_en / train_threshold

approximate reproduction

Total-Text; TEST official / explored. Within-image one-to-one word multiset exact Unicode casefold; punctuation retained; NO spatial IoU; native256 evaluation-only ignore as recorded in the original report.

Performance scope: Feature seconds are the original model feature-export duration; not end-to-end latency or a matched timing benchmark.

| variant | tp | predictions | gt | precision | recall | f1 | tn0 | accepted_images | fallback_images | feature_seconds_reported |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llm2clip_l14_336 | 1464 | 2158 | 2204 | 67.84% | 66.42% | 67.13% | 50.52% | 174 | 126 | 1443.6267311596312 |
| longclip_b | 1456 | 2155 | 2204 | 67.56% | 66.06% | 66.80% | 50.16% | 176 | 124 | 245.9825597642921 |
| longclip_l | 1451 | 2180 | 2204 | 66.56% | 65.83% | 66.20% | 49.47% | 186 | 114 | 402.66949955327436 |

P=TP/predictions; R=TP/GT; F1=2TP/(predictions+GT); TN0=TP/(TP+FP+FN).

Original evidence: `external_artifacts:clip-specialized-comparison-v1/final/comparison.json`; SHA256 `98abc31f96f043123d941f96a7e8545967b2507699c169de1c0415794bd32681`.

- Fixed prompt family, not an assertion that it is every encoder's TRAIN-selected winner.
- Descriptive TEST budgets and transferred TRAIN cuts remain separate; acceptance counts are displayed.
- Feature timing scopes differ from OCR/training. No Table5 or strict claim.

This is a saved numerical summary, not a new evaluation. Its checksum is verified locally; recheck original evidence with `python3 -m research compare clip-encoders-threshold`.
