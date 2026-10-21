# Diagnostics, audits, ablations and research synthesis

[All families](../README.md) · [Code index](../CODE_INDEX.md) · [Saved results](../results/README.md)

Category membership is navigation, not a shared evaluation protocol or a completion claim.

## Agent experiment audit —11 October2026

ID: `agent-experiment-review-2026-10-11`. [Guide](<../../docs/AGENT_EXPERIMENT_REVIEW_2026_10_11.md>).

Code: [localize/unet_word_research_v2.py](<../../localize/unet_word_research_v2.py>), [segmentation/totaltext_curved_box_v4.py](<../../segmentation/totaltext_curved_box_v4.py>), [segmentation/unet_word_research_v2.py](<../../segmentation/unet_word_research_v2.py>).

Numerical results: [research-v2-heads](../results/research-v2-heads.md), [research-v2-postprocess](../results/research-v2-postprocess.md).

## Appearance-guided local re-map v1

ID: `appearance-remap-diagnostic`. [Guide](<../../docs/APPEARANCE_REMAP_DIAGNOSTIC.md>).

Code: [scripts/diagnose_appearance_remap_train.py](<../../scripts/diagnose_appearance_remap_train.py>).

Configuration: [configs/approximate/localization_appearance_guided_local_remap_v1.yaml](<../../configs/approximate/localization_appearance_guided_local_remap_v1.yaml>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Joint base-mask and center-offset crop experiment

ID: `center-guided-word-crops`. [Guide](<../../docs/CENTER_GUIDED_WORD_CROPS.md>).

Code: [localize/center_guided_word_crops.py](<../../localize/center_guided_word_crops.py>), [scripts/evaluate_center_guided_word_crops.py](<../../scripts/evaluate_center_guided_word_crops.py>), [scripts/predict_center_guided_word_crops.py](<../../scripts/predict_center_guided_word_crops.py>).

Numerical results: [center-votes](../results/center-votes.md).

## DeepSolo TextOCR comparison and IC13 adaptation

ID: `deepsolo-textocr-ic13-comparison`. [Guide](<../../docs/DEEPSOLO_TEXTOCR_IC13_COMPARISON.md>).

Code: [scripts/run_deepsolo_textocr_comparison.py](<../../scripts/run_deepsolo_textocr_comparison.py>).

Configuration: [configs/approximate/deepsolo_ic13_author_recipe_v1.json](<../../configs/approximate/deepsolo_ic13_author_recipe_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## ViTAEv2-S+TextOCR GPU evaluation — 2026-10-19

ID: `deepsolo-vitae-gpu-20261019`. [Guide](<../../docs/DEEPSOLO_VITAE_GPU_20261019.md>).

Configuration: [configs/approximate/deepsolo_textocr_gpu_20261019.json](<../../configs/approximate/deepsolo_textocr_gpu_20261019.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Experiment summary: V1, contextual routing and LLM correction

ID: `experiment-summary-v1-routing-llms`. [Guide](<../../docs/EXPERIMENT_SUMMARY_V1_ROUTING_LLMS.md>).

Code: [diagnostics/run_ocr_target_bound_battery_host.sh](<../../diagnostics/run_ocr_target_bound_battery_host.sh>), [localize/grouped_word_blocks.py](<../../localize/grouped_word_blocks.py>).

Configuration: [configs/approximate/pipeline_icdar13_practical_remap_v1.yaml](<../../configs/approximate/pipeline_icdar13_practical_remap_v1.yaml>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Non-oracle geometry/context OCR reconstruction

ID: `geometry-context-reconstruction-train-v1`. [Guide](<../../docs/GEOMETRY_CONTEXT_RECONSTRUCTION_TRAIN_V1.md>).

Code: [diagnostics/run_geometry_context_train_host.sh](<../../diagnostics/run_geometry_context_train_host.sh>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## One GT-word group per annotated word

ID: `gt-word-group-selector`. [Guide](<../../docs/GT_WORD_GROUP_SELECTOR.md>).

Code: [scripts/test_gt_word_group_selector.sh](<../../scripts/test_gt_word_group_selector.sh>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## ICDAR13 routing audit and persistent comparison

ID: `icdar13-routing-audit`. [Guide](<../../docs/ICDAR13_ROUTING_AUDIT.md>).

Code: [scripts/evaluate_icdar13_routing_suite.py](<../../scripts/evaluate_icdar13_routing_suite.py>), [scripts/run_icdar13_routing_suite_host.sh](<../../scripts/run_icdar13_routing_suite_host.sh>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Table5: source verification and meaning of accuracy

ID: `paper-table5-source-audit`. [Guide](<../../docs/PAPER_TABLE5_SOURCE_AUDIT.md>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Research status and directions: masks, instances, crops

ID: `research-synthesis-20261021`. [Guide](<../../docs/RESEARCH_SYNTHESIS_20261021.md>).

Configuration: [configs/approximate/research_synthesis_20261021_v1.json](<../../configs/approximate/research_synthesis_20261021_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Total-Text feature-wavelet gated-skip ablation

ID: `totaltext-frequency-gated-skip`. [Guide](<../../docs/TOTALTEXT_FREQUENCY_GATED_SKIP.md>).

Code: [scripts/run_totaltext_frequency_gated_skip_100ep_host.sh](<../../scripts/run_totaltext_frequency_gated_skip_100ep_host.sh>).

Configuration: [configs/approximate/totaltext_frequency_gated_skip_100ep_v1.json](<../../configs/approximate/totaltext_frequency_gated_skip_100ep_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Total-Text partial encoder: controlled 50-epoch continuation

ID: `totaltext-partial-encoder`. [Guide](<../../docs/TOTALTEXT_PARTIAL_ENCODER.md>).

Code: [scripts/run_totaltext_partial_encoder_50ep_host.sh](<../../scripts/run_totaltext_partial_encoder_50ep_host.sh>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Total-Text v4 image/frequency ablation (practical)

ID: `totaltext-v4-image-frequency-ablation`. [Guide](<../../docs/TOTALTEXT_V4_IMAGE_FREQUENCY_ABLATION.md>).

Code: [scripts/run_totaltext_letterbox_pad0_test_host.sh](<../../scripts/run_totaltext_letterbox_pad0_test_host.sh>), [scripts/run_totaltext_v4_image_frequency_ablation_host.sh](<../../scripts/run_totaltext_v4_image_frequency_ablation_host.sh>).

Configuration: [configs/approximate/totaltext_v4_image_frequency_ablation_v1.json](<../../configs/approximate/totaltext_v4_image_frequency_ablation_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Total-Text: four word-training ablations

ID: `totaltext-word-ablation`. [Guide](<../../docs/TOTALTEXT_WORD_ABLATION.md>).

Code: [scripts/run_totaltext_word_ablation_50ep_host.sh](<../../scripts/run_totaltext_word_ablation_50ep_host.sh>).

Configuration: [configs/approximate/totaltext_word_ablation_50ep_v1.json](<../../configs/approximate/totaltext_word_ablation_50ep_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Total-Text word-level U-Net ablation (practical)

ID: `totaltext-word-f1-separator-ablation`. [Guide](<../../docs/TOTALTEXT_WORD_F1_SEPARATOR_ABLATION.md>).

Code: [models/unet_adv.py](<../../models/unet_adv.py>), [scripts/run_totaltext_letterbox_threshold065_test_host.sh](<../../scripts/run_totaltext_letterbox_threshold065_test_host.sh>), [scripts/run_totaltext_word_f1_separator_suite_host.sh](<../../scripts/run_totaltext_word_f1_separator_suite_host.sh>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Total-Text word-kernel U-Net experiment

ID: `totaltext-word-kernel`. [Guide](<../../docs/TOTALTEXT_WORD_KERNEL.md>).

Code: [scripts/run_totaltext_word_kernel_100ep_host.sh](<../../scripts/run_totaltext_word_kernel_100ep_host.sh>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Restart and failed-run storage cleanup

ID: `unet-restart-storage`. [Guide](<../../docs/UNET_RESTART_STORAGE.md>).

Code: [scripts/cleanup_failed_unet_detection_run.py](<../../scripts/cleanup_failed_unet_detection_run.py>), [scripts/run_unet_detection_methods_host.sh](<../../scripts/run_unet_detection_methods_host.sh>).

Configuration: [configs/approximate/unet_detection_methods_synthtext_targets_v2.json](<../../configs/approximate/unet_detection_methods_synthtext_targets_v2.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Word-guided contours and actual crop recognition

ID: `word-guided-contours`. [Guide](<../../docs/WORD_GUIDED_CONTOURS.md>).

Code: [localize/word_guided_contours.py](<../../localize/word_guided_contours.py>), [scripts/evaluate_word_multitask_crops.py](<../../scripts/evaluate_word_multitask_crops.py>).

Configuration: [configs/approximate/word_contour_refinement_v1.json](<../../configs/approximate/word_contour_refinement_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.
