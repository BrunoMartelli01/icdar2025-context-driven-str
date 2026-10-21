# Datasets, targets, and annotation preparation

[All families](../README.md) · [Code index](../CODE_INDEX.md) · [Saved results](../results/README.md)

Category membership is navigation, not a shared evaluation protocol or a completion claim.

## CTW1500: word recovery inside annotated text lines

ID: `ctw1500-word-recovery`. [Guide](<../../docs/CTW1500_WORD_RECOVERY.md>).

Code: [scripts/test_ctw_words.sh](<../../scripts/test_ctw_words.sh>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Guarded word GT V3 — approximate reproduction

ID: `guarded-word-gt-v3`. [Guide](<../../docs/GUARDED_WORD_GT_V3.md>).

Code: [scripts/qualify_guarded_word_dataloader.py](<../../scripts/qualify_guarded_word_dataloader.py>), [scripts/qualify_guarded_word_gt.py](<../../scripts/qualify_guarded_word_gt.py>).

Configuration: [configs/approximate/guarded_word_gt_v3.json](<../../configs/approximate/guarded_word_gt_v3.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Additional scene-text datasets

ID: `scene-text-datasets`. [Guide](<../../docs/SCENE_TEXT_DATASETS.md>).

Code: [scripts/prepare_ctw1500_recognition_host.sh](<../../scripts/prepare_ctw1500_recognition_host.sh>), [scripts/prepare_scene_text_dataset.py](<../../scripts/prepare_scene_text_dataset.py>), [scripts/prepare_scene_text_training_host.sh](<../../scripts/prepare_scene_text_training_host.sh>), [scripts/rescore_ctw1500_v1_parseq_host.sh](<../../scripts/rescore_ctw1500_v1_parseq_host.sh>), [scripts/train_scene_text_segmentation_host.sh](<../../scripts/train_scene_text_segmentation_host.sh>).

Configuration: [configs/approximate/datasets/scene_text_sources_v1.yaml](<../../configs/approximate/datasets/scene_text_sources_v1.yaml>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Separated full-word masks from downloaded datasets

ID: `separated-word-gt`. [Guide](<../../docs/SEPARATED_WORD_GT.md>).

Code: [scripts/prepare_ctw1500_separated_word_gt.py](<../../scripts/prepare_ctw1500_separated_word_gt.py>), [scripts/prepare_separated_word_gt.py](<../../scripts/prepare_separated_word_gt.py>), [scripts/run_totaltext_separated_words_train_and_parseq_host.sh](<../../scripts/run_totaltext_separated_words_train_and_parseq_host.sh>), [scripts/verify_separated_word_gt.py](<../../scripts/verify_separated_word_gt.py>).

Configuration: [configs/approximate/segmentation_total_text_separated_words_v1.yaml](<../../configs/approximate/segmentation_total_text_separated_words_v1.yaml>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Automatic acquisition for the detection experiment

ID: `synthtext-download`. [Guide](<../../docs/SYNTHTEXT_DOWNLOAD.md>).

Code: [scripts/run_unet_detection_methods_with_download.sh](<../../scripts/run_unet_detection_methods_with_download.sh>).

Tests: [tests/scripts/test_synthtext_download.py](<../../tests/scripts/test_synthtext_download.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Compact word GT v2 and isolated retraining

ID: `tighter-word-gt-v2`. [Guide](<../../docs/TIGHTER_WORD_GT_V2.md>).

Code: [scripts/prepare_tighter_word_gt.py](<../../scripts/prepare_tighter_word_gt.py>), [scripts/run_totaltext_tighter_words_train_and_parseq_host.sh](<../../scripts/run_totaltext_tighter_words_train_and_parseq_host.sh>), [scripts/verify_tighter_word_gt.py](<../../scripts/verify_tighter_word_gt.py>).

Configuration: [configs/approximate/segmentation_total_text_tighter_words_v2.yaml](<../../configs/approximate/segmentation_total_text_tighter_words_v2.yaml>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Total-Text curved word-region training (practical approximation)

ID: `totaltext-curved-word-regions`. [Guide](<../../docs/TOTALTEXT_CURVED_WORD_REGIONS.md>).

Code: [scripts/run_totaltext_curved_box_v4_train_and_test_host.sh](<../../scripts/run_totaltext_curved_box_v4_train_and_test_host.sh>), [scripts/run_totaltext_v1_findcontours_parseq_host.sh](<../../scripts/run_totaltext_v1_findcontours_parseq_host.sh>), [scripts/train_totaltext_curved_word_regions_host.sh](<../../scripts/train_totaltext_curved_word_regions_host.sh>).

Configuration: [configs/approximate/segmentation_total_text_adaptive_word_boxes_v3.yaml](<../../configs/approximate/segmentation_total_text_adaptive_word_boxes_v3.yaml>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Structural grouping of frozen Total-Text crop proposals

ID: `totaltext-structural-word-groups`. [Guide](<../../docs/TOTALTEXT_STRUCTURAL_WORD_GROUPS.md>).

Code: [scripts/test_totaltext_structural_groups.sh](<../../scripts/test_totaltext_structural_groups.sh>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.
