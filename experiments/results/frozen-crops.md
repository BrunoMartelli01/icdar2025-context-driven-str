# Frozen Crops: selected thresholds

approximate reproduction

Total-Text; TRAIN holdout. Maximum-cardinality one-to-one original RGB exclusive bbox IoU>=0.50 AND exact Unicode casefold; punctuation retained; micro counts over all care words. Ignore policy remains source-specific. Ignore only wholly invalid post-localization instances; partial overlap remains care.

| variant | tp | predictions | gt | precision | recall | f1 | tn0 | threshold | geometry_hmean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control512 / connected_components | 418 | 892 | 983 | 46.86% | 42.52% | 44.59% | 28.69% | 0.5 | 0.5696 |
| control512 / findcontours | 418 | 891 | 983 | 46.91% | 42.52% | 44.61% | 28.71% | 0.5 | 0.5699039487726788 |
| control512 / distance_watershed | 321 | 1328 | 983 | 24.17% | 32.66% | 27.78% | 16.13% | 0.5 | 0.4110774556469061 |
| control512 / offset_watershed | 331 | 1071 | 983 | 30.91% | 33.67% | 32.23% | 19.21% | 0.5 | 0.46153846153846156 |
| tversky512 / connected_components | 418 | 892 | 983 | 46.86% | 42.52% | 44.59% | 28.69% | 0.5 | 0.5696 |
| tversky512 / findcontours | 418 | 891 | 983 | 46.91% | 42.52% | 44.61% | 28.71% | 0.5 | 0.5699039487726788 |
| tversky512 / distance_watershed | 321 | 1328 | 983 | 24.17% | 32.66% | 27.78% | 16.13% | 0.5 | 0.4110774556469061 |
| tversky512 / offset_watershed | 331 | 1071 | 983 | 30.91% | 33.67% | 32.23% | 19.21% | 0.5 | 0.46153846153846156 |
| control768 / connected_components | 414 | 946 | 983 | 43.76% | 42.12% | 42.92% | 27.33% | 0.65 | 0.5640228097459824 |
| control768 / findcontours | 414 | 946 | 983 | 43.76% | 42.12% | 42.92% | 27.33% | 0.65 | 0.5640228097459824 |
| control768 / distance_watershed | 335 | 1353 | 983 | 24.76% | 34.08% | 28.68% | 16.74% | 0.65 | 0.4400684931506849 |
| control768 / offset_watershed | 328 | 1059 | 983 | 30.97% | 33.37% | 32.13% | 19.14% | 0.65 | 0.4593535749265426 |
| tversky768 / connected_components | 443 | 975 | 983 | 45.44% | 45.07% | 45.25% | 29.24% | 0.9 | 0.567926455566905 |
| tversky768 / findcontours | 443 | 975 | 983 | 45.44% | 45.07% | 45.25% | 29.24% | 0.9 | 0.567926455566905 |
| tversky768 / distance_watershed | 345 | 1407 | 983 | 24.52% | 35.10% | 28.87% | 16.87% | 0.9 | 0.4276150627615063 |
| tversky768 / offset_watershed | 355 | 1155 | 983 | 30.74% | 36.11% | 33.21% | 19.91% | 0.9 | 0.4536950420954163 |

P=TP/predictions; R=TP/GT; F1=2TP/(predictions+GT); TN0=TP/(TP+FP+FN).

Original evidence: `repository:artifacts/text-recall-parseq-crops-v1/summary.json`; SHA256 `0fdef4221694c9f94f7a5794b990078e75c234814a607d2428db770b0f48184f`.

- Single seed and repeatedly reused TRAIN holdout; not independent validation or official TEST.
- No paper-table or STRICT claim. Compare against the control from this exact source; equal denominators alone do not establish equal protocols.

This is a saved numerical summary, not a new evaluation. Its checksum is verified locally; recheck original evidence with `python3 -m research compare frozen-crops`.
