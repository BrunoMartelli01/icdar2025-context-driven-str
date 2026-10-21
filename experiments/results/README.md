# Saved numerical results

Approximate reproduction. These are small projections of hash-verified saved reports, not new evaluations.
Each table keeps its own protocol, denominator, split, ignore policy, and cost limitations.
Original per-sample outputs/checkpoints remain external; no global ranking is valid.

| Comparison | Rows | Dataset / split | Metric |
| --- | ---: | --- | --- |
| [multitask-geometry](multitask-geometry.md) · [JSON](multitask-geometry.json) · [CSV](multitask-geometry.csv) | 3 | Total-Text / TRAIN holdout | word_localization_native_bbox |
| [research-v1](research-v1.md) · [JSON](research-v1.json) · [CSV](research-v1.csv) | 5 | Total-Text / TRAIN holdout | word_e2e_bbox_casefold |
| [research-v1-top10](research-v1-top10.md) · [JSON](research-v1-top10.json) · [CSV](research-v1-top10.csv) | 2 | Total-Text / TRAIN holdout | word_e2e_bbox_casefold |
| [research-v2-heads](research-v2-heads.md) · [JSON](research-v2-heads.json) · [CSV](research-v2-heads.csv) | 7 | Total-Text / TRAIN holdout | word_e2e_bbox_casefold |
| [research-v2-postprocess](research-v2-postprocess.md) · [JSON](research-v2-postprocess.json) · [CSV](research-v2-postprocess.csv) | 3 | Total-Text / TRAIN holdout | word_e2e_bbox_casefold |
| [crop-links](crop-links.md) · [JSON](crop-links.json) · [CSV](crop-links.csv) | 4 | Total-Text / TRAIN holdout | word_e2e_bbox_casefold |
| [frozen-crops](frozen-crops.md) · [JSON](frozen-crops.json) · [CSV](frozen-crops.csv) | 16 | Total-Text / TRAIN holdout | word_e2e_bbox_casefold |
| [learned-crops](learned-crops.md) · [JSON](learned-crops.json) · [CSV](learned-crops.csv) | 8 | Total-Text / TRAIN holdout | word_e2e_bbox_casefold |
| [refinements](refinements.md) · [JSON](refinements.json) · [CSV](refinements.csv) | 15 | Total-Text / TRAIN holdout | word_e2e_bbox_casefold |
| [gt-crop-diagnostic](gt-crop-diagnostic.md) · [JSON](gt-crop-diagnostic.json) · [CSV](gt-crop-diagnostic.csv) | 1 | Total-Text / TRAIN holdout | word_e2e_bbox_casefold |
| [center-votes](center-votes.md) · [JSON](center-votes.json) · [CSV](center-votes.csv) | 8 | Total-Text / TRAIN holdout | word_e2e_bbox_casefold |
| [gap](gap.md) · [JSON](gap.json) · [CSV](gap.csv) | 3 | Total-Text / TRAIN holdout | word_e2e_bbox_casefold |
| [kernel-extracted](kernel-extracted.md) · [JSON](kernel-extracted.json) · [CSV](kernel-extracted.csv) | 12 | Total-Text / TRAIN holdout | word_e2e_bbox_casefold |
| [kernel-fresh](kernel-fresh.md) · [JSON](kernel-fresh.json) · [CSV](kernel-fresh.csv) | 6 | Total-Text / TRAIN holdout | word_e2e_bbox_casefold |
| [native128-pad0](native128-pad0.md) · [JSON](native128-pad0.json) · [CSV](native128-pad0.csv) | 15 | Total-Text / TEST official / post-hoc exploration | word_e2e_bbox_casefold |
| [native128-pad1](native128-pad1.md) · [JSON](native128-pad1.json) · [CSV](native128-pad1.csv) | 15 | Total-Text / TEST official / post-hoc exploration | word_e2e_bbox_casefold |
| [native128-pad2](native128-pad2.md) · [JSON](native128-pad2.json) · [CSV](native128-pad2.csv) | 15 | Total-Text / TEST official / post-hoc exploration | word_e2e_bbox_casefold |
| [native128-pad4](native128-pad4.md) · [JSON](native128-pad4.json) · [CSV](native128-pad4.csv) | 15 | Total-Text / TEST official / post-hoc exploration | word_e2e_bbox_casefold |
| [native128-pad6](native128-pad6.md) · [JSON](native128-pad6.json) · [CSV](native128-pad6.csv) | 15 | Total-Text / TEST official / post-hoc exploration | word_e2e_bbox_casefold |
| [native128-pad8](native128-pad8.md) · [JSON](native128-pad8.json) · [CSV](native128-pad8.csv) | 15 | Total-Text / TEST official / post-hoc exploration | word_e2e_bbox_casefold |
| [native128-padoriginal6](native128-padoriginal6.md) · [JSON](native128-padoriginal6.json) · [CSV](native128-padoriginal6.csv) | 15 | Total-Text / TEST official / post-hoc exploration | word_e2e_bbox_casefold |
| [yolo-test-equal](yolo-test-equal.md) · [JSON](yolo-test-equal.json) · [CSV](yolo-test-equal.csv) | 7 | Total-Text / TEST official / explored | word_multiset_casefold |
| [yolo-test-transferred](yolo-test-transferred.md) · [JSON](yolo-test-transferred.json) · [CSV](yolo-test-transferred.csv) | 7 | Total-Text / TEST official / explored | word_multiset_casefold |
| [yolo-train-equal](yolo-train-equal.md) · [JSON](yolo-train-equal.json) · [CSV](yolo-train-equal.csv) | 7 | Total-Text / TRAIN holdout | word_multiset_casefold |
| [clip-encoders-budget](clip-encoders-budget.md) · [JSON](clip-encoders-budget.json) · [CSV](clip-encoders-budget.csv) | 3 | Total-Text / TEST official / explored | word_multiset_casefold |
| [clip-encoders-threshold](clip-encoders-threshold.md) · [JSON](clip-encoders-threshold.json) · [CSV](clip-encoders-threshold.csv) | 3 | Total-Text / TEST official / explored | word_multiset_casefold |

Read offline: `python3 -m research results GROUP`.
Recheck original sources: `python3 -m research compare GROUP` with the required external roots.
[Restore original evidence](../../docs/EXTERNAL_ASSETS.md#restore-saved-comparisons).
