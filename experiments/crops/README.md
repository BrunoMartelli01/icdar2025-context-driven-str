# Crop generation, selection, and refinement

[All families](../README.md) · [Code index](../CODE_INDEX.md) · [Saved results](../results/README.md)

Category membership is navigation, not a shared evaluation protocol or a completion claim.

## Frozen crop-group output postfilters

ID: `crop-group-postfilters`. [Guide](<../../docs/CROP_GROUP_POSTFILTERS.md>).

Code: [scripts/test_crop_group_postfilters.sh](<../../scripts/test_crop_group_postfilters.sh>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Crop-group refinement diagnostic

ID: `crop-group-refinements`. [Guide](<../../docs/CROP_GROUP_REFINEMENTS.md>).

Code: [scripts/evaluate_crop_group_refinements.py](<../../scripts/evaluate_crop_group_refinements.py>).

Configuration: [configs/approximate/crop_group_refinements_v1.json](<../../configs/approximate/crop_group_refinements_v1.json>).

Tests: [tests/recognition/test_crop_group_refinements.py](<../../tests/recognition/test_crop_group_refinements.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Selecting one reading from a crop group

ID: `crop-group-word-selection`. [Guide](<../../docs/CROP_GROUP_WORD_SELECTION.md>).

Code: [recognition/group_word_selection.py](<../../recognition/group_word_selection.py>), [scripts/test_crop_group_selectors.sh](<../../scripts/test_crop_group_selectors.sh>), [scripts/test_totaltext_test.sh](<../../scripts/test_totaltext_test.sh>).

Configuration: [configs/approximate/crop_group_word_selection_local_paths_v1.json](<../../configs/approximate/crop_group_word_selection_local_paths_v1.json>), [configs/approximate/crop_group_word_selection_v1.json](<../../configs/approximate/crop_group_word_selection_v1.json>), [configs/approximate/totaltext_crop_group_test_local_paths_v1.json](<../../configs/approximate/totaltext_crop_group_test_local_paths_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## V3/447 word graph cut

ID: `guarded-v3-graph-cut`. [Guide](<../../docs/GUARDED_V3_GRAPH_CUT.md>).

Code: [scripts/run_guarded_v3_graph_cut.py](<../../scripts/run_guarded_v3_graph_cut.py>).

Configuration: [configs/approximate/guarded_v3_graphcut_v1.json](<../../configs/approximate/guarded_v3_graphcut_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Guarded V3 training and PARSeq — approximate reproduction

ID: `guarded-v3-training-ocr`. [Guide](<../../docs/GUARDED_V3_TRAINING_OCR.md>).

Configuration: [configs/approximate/guarded_v3_training_ocr_v1.json](<../../configs/approximate/guarded_v3_training_ocr_v1.json>), [configs/approximate/guarded_v3_training_v1.json](<../../configs/approximate/guarded_v3_training_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Three tests: from V3 masks to readable crops

ID: `mask-crop-three-tests`. [Guide](<../../docs/MASK_CROP_THREE_TESTS.md>).

Code: [scripts/run_mask_crop_budget.py](<../../scripts/run_mask_crop_budget.py>), [scripts/run_mask_crop_learned_tests.py](<../../scripts/run_mask_crop_learned_tests.py>).

Configuration: [configs/approximate/mask_crop_three_tests_v1.json](<../../configs/approximate/mask_crop_three_tests_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Small RGB/mask/probability correction battery

ID: `mask-probability-refiner`. [Guide](<../../docs/MASK_PROBABILITY_REFINER.md>).

Code: [scripts/run_mask_probability_refiner.py](<../../scripts/run_mask_probability_refiner.py>).

Configuration: [configs/approximate/mask_probability_refiner_v1.json](<../../configs/approximate/mask_probability_refiner_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Opt-in repository contour localizer

ID: `repository-contours-variant`. [Guide](<../../docs/REPOSITORY_CONTOURS_VARIANT.md>).

Code: [diagnostics/compare_repository_localizer.py](<../../diagnostics/compare_repository_localizer.py>), [diagnostics/repository_contours_native_train.py](<../../diagnostics/repository_contours_native_train.py>), [diagnostics/repository_contours_train.py](<../../diagnostics/repository_contours_train.py>), [diagnostics/run_repository_contours_native_ocr_host.sh](<../../diagnostics/run_repository_contours_native_ocr_host.sh>), [diagnostics/run_repository_contours_train_host.sh](<../../diagnostics/run_repository_contours_train_host.sh>), [localize/mask_to_blocks.py](<../../localize/mask_to_blocks.py>), [localize/native_repository_contours.py](<../../localize/native_repository_contours.py>), [localize/repository_contours.py](<../../localize/repository_contours.py>).

Configuration: [configs/approximate/localization_repository_contours_v1.json](<../../configs/approximate/localization_repository_contours_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Frozen text-recall crops and actual PARSeq recognition

ID: `text-recall-crops`. [Guide](<../../docs/TEXT_RECALL_CROPS.md>).

Code: [scripts/evaluate_text_recall_crops.py](<../../scripts/evaluate_text_recall_crops.py>), [scripts/evaluate_text_recall_learned_crops.py](<../../scripts/evaluate_text_recall_learned_crops.py>).

Configuration: [configs/approximate/text_recall_crops_v1.json](<../../configs/approximate/text_recall_crops_v1.json>).

Numerical results: [frozen-crops](../results/frozen-crops.md).

## Frozen word-crop refinements

ID: `text-recall-refinements`. [Guide](<../../docs/TEXT_RECALL_REFINEMENTS.md>).

Code: [scripts/evaluate_text_recall_refinements.py](<../../scripts/evaluate_text_recall_refinements.py>), [scripts/package_text_recall_refinements_review.py](<../../scripts/package_text_recall_refinements_review.py>).

Configuration: [configs/approximate/text_recall_refinements_v1.json](<../../configs/approximate/text_recall_refinements_v1.json>).

Numerical results: [learned-crops](../results/learned-crops.md), [refinements](../results/refinements.md), [gt-crop-diagnostic](../results/gt-crop-diagnostic.md).

## Text-mask recall: loss and input resolution

ID: `text-recall-resolution`. [Guide](<../../docs/TEXT_RECALL_RESOLUTION.md>).

Code: [scripts/run_text_recall_resolution.py](<../../scripts/run_text_recall_resolution.py>), [scripts/run_text_recall_resolution_host.sh](<../../scripts/run_text_recall_resolution_host.sh>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Total-Text crop variation and CLIP selector battery

ID: `totaltext-crop-clip-battery`. [Guide](<../../docs/TOTALTEXT_CROP_CLIP_BATTERY.md>).

Code: [scripts/run_totaltext_crop_clip_battery_host.sh](<../../scripts/run_totaltext_crop_clip_battery_host.sh>), [scripts/run_totaltext_crop_clip_tests.sh](<../../scripts/run_totaltext_crop_clip_tests.sh>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Total-Text crop recovery suite

ID: `totaltext-crop-recovery-suite`. [Guide](<../../docs/TOTALTEXT_CROP_RECOVERY_SUITE.md>).

Code: [scripts/test_totaltext_crops.sh](<../../scripts/test_totaltext_crops.sh>).

Configuration: [configs/approximate/totaltext_crop_recovery_suite_train_v1.json](<../../configs/approximate/totaltext_crop_recovery_suite_train_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Grouping threshold and window crops

ID: `totaltext-crop-word-groups`. [Guide](<../../docs/TOTALTEXT_CROP_WORD_GROUPS.md>).

Code: [scripts/test_totaltext_groups.sh](<../../scripts/test_totaltext_groups.sh>).

Configuration: [configs/approximate/totaltext_crop_word_groups_train_v1.json](<../../configs/approximate/totaltext_crop_word_groups_train_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## TRAIN: localization errors, crop completeness, and OCR recovery

ID: `train-grouping-failures-and-recovery`. [Guide](<../../docs/TRAIN_GROUPING_FAILURES_AND_RECOVERY.md>).

Code: [diagnostics/run_recovered_train_crops_host.sh](<../../diagnostics/run_recovered_train_crops_host.sh>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## U-Net crop resegmentation experiment

ID: `unet-crop-refinement`. [Guide](<../../docs/UNET_CROP_REFINEMENT.md>).

Code: [scripts/diagnose_unet_crop_refinement.py](<../../scripts/diagnose_unet_crop_refinement.py>).

Configuration: [configs/approximate/totaltext_v4_image_frequency_ablation_v1.json](<../../configs/approximate/totaltext_v4_image_frequency_ablation_v1.json>), [configs/approximate/unet_crop_refinement_v1.json](<../../configs/approximate/unet_crop_refinement_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.
