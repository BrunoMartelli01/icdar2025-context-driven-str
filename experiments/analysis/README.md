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

## Exact budget feasibility of existing ICDAR experts

ID: `budget-target-router`. [Guide](<../../docs/BUDGET_TARGET_ROUTER.md>).

Code: [scripts/prepare_ic13_complement.py](<../../scripts/prepare_ic13_complement.py>), [scripts/prepare_ic13_word_risk.py](<../../scripts/prepare_ic13_word_risk.py>), [scripts/run_lexical_budget_router.py](<../../scripts/run_lexical_budget_router.py>), [scripts/train_ic13_fulltrain_oof.py](<../../scripts/train_ic13_fulltrain_oof.py>).

Configuration: [configs/approximate/ic13_complement_fulltrain_no_wordrisk_v1.json](<../../configs/approximate/ic13_complement_fulltrain_no_wordrisk_v1.json>), [configs/approximate/ic13_complement_fulltrain_oof_v1.json](<../../configs/approximate/ic13_complement_fulltrain_oof_v1.json>), [configs/approximate/ic13_complement_v1.json](<../../configs/approximate/ic13_complement_v1.json>), [configs/approximate/ic13_word_risk_budget_v1.json](<../../configs/approximate/ic13_word_risk_budget_v1.json>), [configs/approximate/lexical_budget_target_router_v1.json](<../../configs/approximate/lexical_budget_target_router_v1.json>).

Tests: [tests/recognition/test_budget_target_oracle.py](<../../tests/recognition/test_budget_target_oracle.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Joint base-mask and center-offset crop experiment

ID: `center-guided-word-crops`. [Guide](<../../docs/CENTER_GUIDED_WORD_CROPS.md>).

Code: [localize/center_guided_word_crops.py](<../../localize/center_guided_word_crops.py>), [scripts/evaluate_center_guided_word_crops.py](<../../scripts/evaluate_center_guided_word_crops.py>), [scripts/predict_center_guided_word_crops.py](<../../scripts/predict_center_guided_word_crops.py>).

Numerical results: [center-votes](../results/center-votes.md).

## Shared budgeted OCR pipeline for Total-Text and ICDAR2015

ID: `common-budget-pipeline`. [Guide](<../../docs/COMMON_BUDGET_PIPELINE.md>).

Code: [recognition/common_budget_pipeline.py](<../../recognition/common_budget_pipeline.py>), [scripts/run_common_budget_pipeline.py](<../../scripts/run_common_budget_pipeline.py>).

Configuration: [configs/approximate/common_budget_pipeline_icdar13_v1.json](<../../configs/approximate/common_budget_pipeline_icdar13_v1.json>), [configs/approximate/common_budget_pipeline_pooled_validation_v2.json](<../../configs/approximate/common_budget_pipeline_pooled_validation_v2.json>), [configs/approximate/common_budget_pipeline_v1.json](<../../configs/approximate/common_budget_pipeline_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## DeepSolo parity thresholds for the common Total-Text / ICDAR2015 procedure

ID: `common-parity-threshold`. [Guide](<../../docs/COMMON_PARITY_THRESHOLD.md>).

Code: [recognition/common_parity_threshold.py](<../../recognition/common_parity_threshold.py>), [scripts/calibrate_common_parity_threshold.py](<../../scripts/calibrate_common_parity_threshold.py>), [scripts/report_common_parity_threshold.py](<../../scripts/report_common_parity_threshold.py>).

Configuration: [configs/approximate/common_pipeline_parity_threshold_v1.json](<../../configs/approximate/common_pipeline_parity_threshold_v1.json>).

Tests: [tests/recognition/test_common_budget_pipeline.py](<../../tests/recognition/test_common_budget_pipeline.py>), [tests/recognition/test_common_parity_threshold.py](<../../tests/recognition/test_common_parity_threshold.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Six context signals for whole-image OCR routing

ID: `creative-context-router`. [Guide](<../../docs/CREATIVE_CONTEXT_ROUTER.md>).

Code: [recognition/context_memory_router.py](<../../recognition/context_memory_router.py>), [recognition/creative_context_runtime.py](<../../recognition/creative_context_runtime.py>), [recognition/creative_context_signals.py](<../../recognition/creative_context_signals.py>), [scripts/audit_creative_context_router.py](<../../scripts/audit_creative_context_router.py>), [scripts/report_creative_context_router.py](<../../scripts/report_creative_context_router.py>), [scripts/run_creative_context_router.py](<../../scripts/run_creative_context_router.py>).

Configuration: [configs/approximate/creative_context_router_local_paths_v1.json](<../../configs/approximate/creative_context_router_local_paths_v1.json>), [configs/approximate/creative_context_router_v1.json](<../../configs/approximate/creative_context_router_v1.json>).

Tests: [tests/recognition/test_creative_context_router.py](<../../tests/recognition/test_creative_context_router.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## DeepSolo ICDAR2015: checkpoint and threshold gap analysis

ID: `deepsolo-icdar2015-gap`. [Guide](<../../docs/DEEPSOLO_ICDAR2015_GAP.md>).

Code: [deepsolo/worker.py](<../../deepsolo/worker.py>), [recognition/deepsolo_icdar2015_gap.py](<../../recognition/deepsolo_icdar2015_gap.py>), [scripts/analyze_deepsolo_icdar2015_gap.py](<../../scripts/analyze_deepsolo_icdar2015_gap.py>), [scripts/audit_deepsolo_icdar2015_gap.py](<../../scripts/audit_deepsolo_icdar2015_gap.py>), [scripts/export_essential_transfer_deepsolo.py](<../../scripts/export_essential_transfer_deepsolo.py>).

Configuration: [configs/approximate/deepsolo_icdar2015_gap_v1.json](<../../configs/approximate/deepsolo_icdar2015_gap_v1.json>), [configs/approximate/deepsolo_icdar2015_selected_native_word_v1.json](<../../configs/approximate/deepsolo_icdar2015_selected_native_word_v1.json>).

Tests: [tests/recognition/test_deepsolo_icdar2015_gap.py](<../../tests/recognition/test_deepsolo_icdar2015_gap.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## DeepSolo with released ICDAR lexicons

ID: `deepsolo-icdar-lexicons`. [Guide](<../../docs/DEEPSOLO_ICDAR_LEXICONS.md>).

Code: [recognition/deepsolo_lexicon.py](<../../recognition/deepsolo_lexicon.py>), [recognition/deepsolo_lexicon_deployment.py](<../../recognition/deepsolo_lexicon_deployment.py>), [recognition/deepsolo_lexicon_guard.py](<../../recognition/deepsolo_lexicon_guard.py>), [scripts/audit_deepsolo_lexicon_deployment.py](<../../scripts/audit_deepsolo_lexicon_deployment.py>), [scripts/audit_deepsolo_lexicons.py](<../../scripts/audit_deepsolo_lexicons.py>), [scripts/evaluate_deepsolo_ic13_lexicons.py](<../../scripts/evaluate_deepsolo_ic13_lexicons.py>), [scripts/evaluate_deepsolo_icdar_lexicon.py](<../../scripts/evaluate_deepsolo_icdar_lexicon.py>), [scripts/evaluate_deepsolo_lexicon_guard.py](<../../scripts/evaluate_deepsolo_lexicon_guard.py>).

Configuration: [configs/approximate/deepsolo_ic13_lexicons_v1.json](<../../configs/approximate/deepsolo_ic13_lexicons_v1.json>), [configs/approximate/deepsolo_icdar13_generic_calibrated_v1.json](<../../configs/approximate/deepsolo_icdar13_generic_calibrated_v1.json>), [configs/approximate/deepsolo_icdar13_generic_source_v1.json](<../../configs/approximate/deepsolo_icdar13_generic_source_v1.json>), [configs/approximate/deepsolo_icdar2015_generic_calibrated_v1.json](<../../configs/approximate/deepsolo_icdar2015_generic_calibrated_v1.json>), [configs/approximate/deepsolo_icdar2015_generic_source_v1.json](<../../configs/approximate/deepsolo_icdar2015_generic_source_v1.json>), [configs/approximate/deepsolo_icdar_lexicon_v1.json](<../../configs/approximate/deepsolo_icdar_lexicon_v1.json>).

Tests: [tests/recognition/test_deepsolo_lexicon.py](<../../tests/recognition/test_deepsolo_lexicon.py>), [tests/recognition/test_deepsolo_lexicon_deployment.py](<../../tests/recognition/test_deepsolo_lexicon_deployment.py>), [tests/recognition/test_deepsolo_lexicon_guard.py](<../../tests/recognition/test_deepsolo_lexicon_guard.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## DeepSolo minimum-three-character diagnostic

ID: `deepsolo-short-words`. [Guide](<../../docs/DEEPSOLO_SHORT_WORDS.md>).

Code: [scripts/evaluate_deepsolo_short_words.py](<../../scripts/evaluate_deepsolo_short_words.py>).

Configuration: [configs/approximate/deepsolo_short_words_v1.json](<../../configs/approximate/deepsolo_short_words_v1.json>).

Tests: [tests/scripts/test_deepsolo_short_words.py](<../../tests/scripts/test_deepsolo_short_words.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## DeepSolo TextOCR comparison and IC13 adaptation

ID: `deepsolo-textocr-ic13-comparison`. [Guide](<../../docs/DEEPSOLO_TEXTOCR_IC13_COMPARISON.md>).

Code: [scripts/run_deepsolo_textocr_comparison.py](<../../scripts/run_deepsolo_textocr_comparison.py>).

Configuration: [configs/approximate/deepsolo_ic13_author_recipe_v1.json](<../../configs/approximate/deepsolo_ic13_author_recipe_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## ViTAEv2-S+TextOCR GPU evaluation — 2026-10-19

ID: `deepsolo-vitae-gpu-20261019`. [Guide](<../../docs/DEEPSOLO_VITAE_GPU_20261019.md>).

Configuration: [configs/approximate/deepsolo_textocr_gpu_20261019.json](<../../configs/approximate/deepsolo_textocr_gpu_20261019.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## DeepSolo symbol-free word F1 study — 2026-10-25

ID: `deepsolo-word-f1-study`. [Guide](<../../docs/DEEPSOLO_WORD_F1_STUDY.md>).

Code: [recognition/deepsolo_word_f1.py](<../../recognition/deepsolo_word_f1.py>), [scripts/analyze_deepsolo_word_f1.py](<../../scripts/analyze_deepsolo_word_f1.py>), [scripts/audit_deepsolo_word_f1.py](<../../scripts/audit_deepsolo_word_f1.py>), [scripts/audit_deepsolo_word_f1_profiles.py](<../../scripts/audit_deepsolo_word_f1_profiles.py>), [scripts/evaluate_deepsolo_word_f1_fusion.py](<../../scripts/evaluate_deepsolo_word_f1_fusion.py>), [scripts/infer_deepsolo_word_f1.py](<../../scripts/infer_deepsolo_word_f1.py>), [scripts/report_deepsolo_word_f1_profiles.py](<../../scripts/report_deepsolo_word_f1_profiles.py>), [scripts/run_deepsolo_word_f1_study.py](<../../scripts/run_deepsolo_word_f1_study.py>).

Configuration: [configs/approximate/deepsolo_word_f1_fusion_v2.json](<../../configs/approximate/deepsolo_word_f1_fusion_v2.json>), [configs/approximate/deepsolo_word_f1_resolution_v2.json](<../../configs/approximate/deepsolo_word_f1_resolution_v2.json>), [configs/approximate/deepsolo_word_f1_selected_global_word_v1.json](<../../configs/approximate/deepsolo_word_f1_selected_global_word_v1.json>), [configs/approximate/deepsolo_word_f1_selected_native_word_v1.json](<../../configs/approximate/deepsolo_word_f1_selected_native_word_v1.json>), [configs/approximate/deepsolo_word_f1_v2.json](<../../configs/approximate/deepsolo_word_f1_v2.json>).

Tests: [tests/recognition/test_deepsolo_word_f1.py](<../../tests/recognition/test_deepsolo_word_f1.py>), [tests/scripts/test_audit_deepsolo_word_f1.py](<../../tests/scripts/test_audit_deepsolo_word_f1.py>), [tests/scripts/test_deepsolo_word_f1_fusion.py](<../../tests/scripts/test_deepsolo_word_f1_fusion.py>), [tests/scripts/test_deepsolo_word_f1_study.py](<../../tests/scripts/test_deepsolo_word_f1_study.py>), [tests/scripts/test_infer_deepsolo_word_f1.py](<../../tests/scripts/test_infer_deepsolo_word_f1.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Directional decoder word-F1 results and failure analysis

ID: `directional-transport-results`. [Guide](<../../docs/DIRECTIONAL_TRANSPORT_RESULTS.md>).

Code: [localize/directional_affinity_split.py](<../../localize/directional_affinity_split.py>), [localize/directional_closedness.py](<../../localize/directional_closedness.py>), [models/directional_transport_unet.py](<../../models/directional_transport_unet.py>), [recognition/directional_transport_metrics.py](<../../recognition/directional_transport_metrics.py>), [scripts/run_directional_affinity_split.py](<../../scripts/run_directional_affinity_split.py>), [scripts/run_directional_closedness.py](<../../scripts/run_directional_closedness.py>), [scripts/run_directional_transport.py](<../../scripts/run_directional_transport.py>), [segmentation/directional_transport.py](<../../segmentation/directional_transport.py>).

Configuration: [configs/approximate/directional_affinity_split_v5.json](<../../configs/approximate/directional_affinity_split_v5.json>), [configs/approximate/directional_closedness_v3.json](<../../configs/approximate/directional_closedness_v3.json>), [configs/approximate/directional_transport_continuation_v4.json](<../../configs/approximate/directional_transport_continuation_v4.json>), [configs/approximate/directional_transport_frozen_v2.json](<../../configs/approximate/directional_transport_frozen_v2.json>), [configs/approximate/directional_transport_v1.json](<../../configs/approximate/directional_transport_v1.json>).

Tests: [tests/localize/test_directional_affinity_split.py](<../../tests/localize/test_directional_affinity_split.py>), [tests/localize/test_directional_closedness.py](<../../tests/localize/test_directional_closedness.py>), [tests/models/test_directional_transport.py](<../../tests/models/test_directional_transport.py>), [tests/segmentation/test_totaltext_word_kernel.py](<../../tests/segmentation/test_totaltext_word_kernel.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Directional transport and RGB detail decoder study

ID: `directional-transport-unet`. [Guide](<../../docs/DIRECTIONAL_TRANSPORT_UNET.md>).

Code: [models/directional_transport_unet.py](<../../models/directional_transport_unet.py>), [recognition/directional_transport_metrics.py](<../../recognition/directional_transport_metrics.py>), [scripts/run_directional_transport.py](<../../scripts/run_directional_transport.py>), [segmentation/directional_transport.py](<../../segmentation/directional_transport.py>).

Configuration: [configs/approximate/directional_transport_frozen_v2.json](<../../configs/approximate/directional_transport_frozen_v2.json>), [configs/approximate/directional_transport_v1.json](<../../configs/approximate/directional_transport_v1.json>).

Tests: [tests/models/test_directional_transport.py](<../../tests/models/test_directional_transport.py>).

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

## Four controlled graph/occlusion follow-ups

ID: `graph-occlusion-followups`. [Guide](<../../docs/GRAPH_OCCLUSION_FOLLOWUPS.md>).

Code: [localize/graph_occlusion_followups.py](<../../localize/graph_occlusion_followups.py>), [scripts/run_graph_occlusion_followups.py](<../../scripts/run_graph_occlusion_followups.py>), [scripts/run_graph_occlusion_followups.sh](<../../scripts/run_graph_occlusion_followups.sh>).

Configuration: [configs/approximate/graph_occlusion_followups_v2.json](<../../configs/approximate/graph_occlusion_followups_v2.json>).

Tests: [tests/localize/test_graph_occlusion_followups.py](<../../tests/localize/test_graph_occlusion_followups.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Fragment graphs and occlusion-aware U-Net teacher/student

ID: `graph-occlusion-research`. [Guide](<../../docs/GRAPH_OCCLUSION_RESEARCH.md>).

Code: [scripts/run_graph_occlusion_battery.sh](<../../scripts/run_graph_occlusion_battery.sh>), [scripts/run_graph_occlusion_research.py](<../../scripts/run_graph_occlusion_research.py>).

Configuration: [configs/approximate/graph_occlusion_research_v1.json](<../../configs/approximate/graph_occlusion_research_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Grounded context inventory and local verification router

ID: `grounded-context-router`. [Guide](<../../docs/GROUNDED_CONTEXT_ROUTER.md>).

Code: [recognition/grounded_context_router.py](<../../recognition/grounded_context_router.py>), [recognition/grounded_context_runtime.py](<../../recognition/grounded_context_runtime.py>), [scripts/analyze_grounded_context_router.py](<../../scripts/analyze_grounded_context_router.py>), [scripts/report_grounded_context_router.py](<../../scripts/report_grounded_context_router.py>), [scripts/run_grounded_context_router.py](<../../scripts/run_grounded_context_router.py>).

Configuration: [configs/approximate/grounded_context_router_local_paths_v1.json](<../../configs/approximate/grounded_context_router_local_paths_v1.json>), [configs/approximate/grounded_context_router_v1.json](<../../configs/approximate/grounded_context_router_v1.json>).

Tests: [tests/recognition/test_grounded_context_router.py](<../../tests/recognition/test_grounded_context_router.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Grounded verification, new localization queries and visual byte edits

ID: `grounded-word-recovery`. [Guide](<../../docs/GROUNDED_WORD_RECOVERY.md>).

Code: [models/grounded_word_recovery.py](<../../models/grounded_word_recovery.py>), [recognition/grounded_word_recovery.py](<../../recognition/grounded_word_recovery.py>), [scripts/analyze_grounded_word_recovery.py](<../../scripts/analyze_grounded_word_recovery.py>), [scripts/grounded_word_editor.py](<../../scripts/grounded_word_editor.py>), [scripts/grounded_word_search.py](<../../scripts/grounded_word_search.py>), [scripts/run_grounded_word_recovery.py](<../../scripts/run_grounded_word_recovery.py>).

Configuration: [configs/approximate/grounded_word_recovery_v1.json](<../../configs/approximate/grounded_word_recovery_v1.json>).

Tests: [tests/recognition/test_grounded_word_recovery.py](<../../tests/recognition/test_grounded_word_recovery.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## One GT-word group per annotated word

ID: `gt-word-group-selector`. [Guide](<../../docs/GT_WORD_GROUP_SELECTOR.md>).

Code: [scripts/test_gt_word_group_selector.sh](<../../scripts/test_gt_word_group_selector.sh>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## ICDAR2013: vocabulary, crop, cross-fitting and mixed-data controls

ID: `icdar13-hypothesis-tests`. [Guide](<../../docs/ICDAR13_HYPOTHESIS_TESTS.md>).

Code: [recognition/icdar13_hypothesis_tests.py](<../../recognition/icdar13_hypothesis_tests.py>), [scripts/run_icdar13_hypothesis_tests.py](<../../scripts/run_icdar13_hypothesis_tests.py>).

Configuration: [configs/approximate/icdar13_hypothesis_fixed_epoch_control_v1.json](<../../configs/approximate/icdar13_hypothesis_fixed_epoch_control_v1.json>), [configs/approximate/icdar13_hypothesis_tests_v1.json](<../../configs/approximate/icdar13_hypothesis_tests_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Newly trained ICDAR2013 U-Net/Generic-DeepSolo router at exactly61 fallback images

ID: `icdar13-matched-router-generic`. [Guide](<../../docs/ICDAR13_MATCHED_ROUTER_GENERIC.md>).

Code: [recognition/icdar13_matched_router.py](<../../recognition/icdar13_matched_router.py>), [scripts/audit_icdar13_matched_router_generic.py](<../../scripts/audit_icdar13_matched_router_generic.py>), [scripts/train_icdar13_matched_router_generic.py](<../../scripts/train_icdar13_matched_router_generic.py>).

Configuration: [configs/approximate/icdar13_matched_router_generic_v1.json](<../../configs/approximate/icdar13_matched_router_generic_v1.json>).

Tests: [tests/recognition/test_icdar13_matched_router.py](<../../tests/recognition/test_icdar13_matched_router.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## ICDAR2013 symbol-free word F1: audit and matched Total-Text loss controls

ID: `icdar13-matched-word-training`. [Guide](<../../docs/ICDAR13_MATCHED_WORD_TRAINING.md>).

Code: [scripts/audit_icdar13_matched_word_training.py](<../../scripts/audit_icdar13_matched_word_training.py>), [scripts/report_icdar13_matched_word_training.py](<../../scripts/report_icdar13_matched_word_training.py>), [scripts/run_icdar13_matched_word_training.py](<../../scripts/run_icdar13_matched_word_training.py>), [segmentation/icdar13_matched_training.py](<../../segmentation/icdar13_matched_training.py>).

Configuration: [configs/approximate/icdar13_matched_word_training_local_paths_v1.json](<../../configs/approximate/icdar13_matched_word_training_local_paths_v1.json>), [configs/approximate/icdar13_matched_word_training_v1.json](<../../configs/approximate/icdar13_matched_word_training_v1.json>).

Tests: [tests/segmentation/test_icdar13_matched_training.py](<../../tests/segmentation/test_icdar13_matched_training.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## ICDAR13 routing audit and persistent comparison

ID: `icdar13-routing-audit`. [Guide](<../../docs/ICDAR13_ROUTING_AUDIT.md>).

Code: [scripts/evaluate_icdar13_routing_suite.py](<../../scripts/evaluate_icdar13_routing_suite.py>), [scripts/run_icdar13_routing_suite_host.sh](<../../scripts/run_icdar13_routing_suite_host.sh>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## ICDAR2013 Total-Text transfer and training duration

ID: `icdar13-transfer-duration`. [Guide](<../../docs/ICDAR13_TRANSFER_DURATION.md>).

Code: [scripts/run_icdar13_transfer_duration.py](<../../scripts/run_icdar13_transfer_duration.py>).

Configuration: [configs/approximate/icdar13_transfer_duration_v1.json](<../../configs/approximate/icdar13_transfer_duration_v1.json>).

Tests: [tests/scripts/test_icdar13_transfer_duration.py](<../../tests/scripts/test_icdar13_transfer_duration.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## ICDAR2013: U-Net dense word instances toward the 61-image fallback target

ID: `icdar13-unet-dense-instances`. [Guide](<../../docs/ICDAR13_UNET_DENSE_INSTANCES.md>).

Code: [models/joint_word_instance_unet.py](<../../models/joint_word_instance_unet.py>), [models/unet_dense_instances.py](<../../models/unet_dense_instances.py>), [scripts/audit_icdar13_text_word_policy.py](<../../scripts/audit_icdar13_text_word_policy.py>), [scripts/evaluate_icdar13_text_encoder_words.py](<../../scripts/evaluate_icdar13_text_encoder_words.py>), [scripts/prepare_icdar13_fold_teachers.py](<../../scripts/prepare_icdar13_fold_teachers.py>), [scripts/run_icdar13_dense_instances.py](<../../scripts/run_icdar13_dense_instances.py>), [scripts/train_icdar13_text_encoder_words.py](<../../scripts/train_icdar13_text_encoder_words.py>), [segmentation/unet_dense_instances.py](<../../segmentation/unet_dense_instances.py>).

Configuration: [configs/approximate/icdar13_dense_policy_v1.json](<../../configs/approximate/icdar13_dense_policy_v1.json>), [configs/approximate/icdar13_fold_safe_teachers_v1.json](<../../configs/approximate/icdar13_fold_safe_teachers_v1.json>), [configs/approximate/icdar13_text_encoder_qualification_v1.json](<../../configs/approximate/icdar13_text_encoder_qualification_v1.json>), [configs/approximate/icdar13_text_encoder_word_training_b4_v2.json](<../../configs/approximate/icdar13_text_encoder_word_training_b4_v2.json>), [configs/approximate/icdar13_text_encoder_word_training_v1.json](<../../configs/approximate/icdar13_text_encoder_word_training_v1.json>), [configs/approximate/icdar13_text_word_policy_v1.json](<../../configs/approximate/icdar13_text_word_policy_v1.json>), [configs/approximate/icdar13_unet_dense_instances_v1.json](<../../configs/approximate/icdar13_unet_dense_instances_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## ICDAR2013 U-Net: matched YOLO selection procedure

ID: `icdar13-unet-yolo-procedure`. [Guide](<../../docs/ICDAR13_UNET_YOLO_PROCEDURE.md>).

Code: [recognition/unet_expert_records.py](<../../recognition/unet_expert_records.py>), [scripts/audit_ic13_unet_expert.py](<../../scripts/audit_ic13_unet_expert.py>), [scripts/export_ic13_unet_expert.py](<../../scripts/export_ic13_unet_expert.py>), [scripts/prepare_ic13_unet_oof.py](<../../scripts/prepare_ic13_unet_oof.py>), [scripts/supervise_ic13_unet_oof.py](<../../scripts/supervise_ic13_unet_oof.py>).

Configuration: [configs/approximate/icdar13_unet_yolo_oof_v1.json](<../../configs/approximate/icdar13_unet_yolo_oof_v1.json>), [configs/approximate/icdar13_unet_yolo_supervisor_v1.json](<../../configs/approximate/icdar13_unet_yolo_supervisor_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## ICDAR2015: matched encoder regularization experiment

ID: `icdar2015-encoder-decay`. [Guide](<../../docs/ICDAR2015_ENCODER_DECAY.md>).

Configuration: [configs/approximate/icdar2015_encoder_decay_v1.json](<../../configs/approximate/icdar2015_encoder_decay_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## ICDAR2015: confidence seeds for a frozen single-output U-Net

ID: `icdar2015-seed-partition`. [Guide](<../../docs/ICDAR2015_SEED_PARTITION.md>).

Configuration: [configs/approximate/icdar2015_seed_partition_probe_v1.json](<../../configs/approximate/icdar2015_seed_partition_probe_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## ICDAR2015: existing SynthText data for broader text pretraining

ID: `icdar2015-synthtext-pretraining`. [Guide](<../../docs/ICDAR2015_SYNTHTEXT_PRETRAINING.md>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## ICDAR2015 U-Net: the 70% correct-word F1 objective

ID: `icdar2015-target70`. [Guide](<../../docs/ICDAR2015_TARGET70.md>).

Code: [scripts/audit_icdar2015_target70.py](<../../scripts/audit_icdar2015_target70.py>), [scripts/evaluate_icdar2015_target70.py](<../../scripts/evaluate_icdar2015_target70.py>), [scripts/qualify_icdar2015_target70_ocr_device.py](<../../scripts/qualify_icdar2015_target70_ocr_device.py>), [scripts/select_icdar2015_target70.py](<../../scripts/select_icdar2015_target70.py>), [scripts/train_icdar2015_target70.py](<../../scripts/train_icdar2015_target70.py>).

Configuration: [configs/approximate/icdar2015_target70_selection_v1.json](<../../configs/approximate/icdar2015_target70_selection_v1.json>), [configs/approximate/icdar2015_target70_v1.json](<../../configs/approximate/icdar2015_target70_v1.json>).

Tests: [tests/models/test_icdar2015_target70_encoder.py](<../../tests/models/test_icdar2015_target70_encoder.py>), [tests/scripts/test_icdar2015_target70_oracle_geometry.py](<../../tests/scripts/test_icdar2015_target70_oracle_geometry.py>), [tests/segmentation/test_icdar2015_operating_crops.py](<../../tests/segmentation/test_icdar2015_operating_crops.py>), [tests/segmentation/test_icdar2015_target70.py](<../../tests/segmentation/test_icdar2015_target70.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## ICDAR2015 target70: hard-negative loss and complete encoder adaptation

ID: `icdar2015-target70-ohem`. [Guide](<../../docs/ICDAR2015_TARGET70_OHEM.md>).

Code: [segmentation/totaltext_word_ablation.py](<../../segmentation/totaltext_word_ablation.py>).

Configuration: [configs/approximate/icdar2015_target70_ohem_v1.json](<../../configs/approximate/icdar2015_target70_ohem_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## ICDAR2015 U-Net and word-mask transfer from Total-Text

ID: `icdar2015-totaltext-transfer`. [Guide](<../../docs/ICDAR2015_TOTALTEXT_TRANSFER.md>).

Code: [models/icdar2015_unet.py](<../../models/icdar2015_unet.py>), [models/unet_adv.py](<../../models/unet_adv.py>), [scripts/prepare_icdar2015_word_regions.py](<../../scripts/prepare_icdar2015_word_regions.py>), [scripts/preview_icdar2015_word_transfer.py](<../../scripts/preview_icdar2015_word_transfer.py>), [scripts/train_icdar2015_totaltext_transfer.py](<../../scripts/train_icdar2015_totaltext_transfer.py>), [segmentation/scene_text_preparation.py](<../../segmentation/scene_text_preparation.py>).

Configuration: [configs/approximate/icdar2015_totaltext_transfer_v1.json](<../../configs/approximate/icdar2015_totaltext_transfer_v1.json>).

Tests: [tests/segmentation/test_icdar2015_totaltext_transfer.py](<../../tests/segmentation/test_icdar2015_totaltext_transfer.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## ICDAR2015: explicit transfer of trained Total-Text weights

ID: `icdar2015-totaltext-weights`. [Guide](<../../docs/ICDAR2015_TOTALTEXT_WEIGHTS.md>).

Configuration: [configs/approximate/icdar2015_totaltext_weights_v1.json](<../../configs/approximate/icdar2015_totaltext_weights_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Controlled ICDAR2015 U-Net improvement

ID: `icdar2015-unet-improvement`. [Guide](<../../docs/ICDAR2015_UNET_IMPROVEMENT.md>).

Code: [models/icdar2015_improvements.py](<../../models/icdar2015_improvements.py>), [scripts/audit_icdar2015_improvement_target_geometry.py](<../../scripts/audit_icdar2015_improvement_target_geometry.py>), [scripts/audit_icdar2015_improvement_word_f1.py](<../../scripts/audit_icdar2015_improvement_word_f1.py>), [scripts/audit_icdar2015_word_masks.py](<../../scripts/audit_icdar2015_word_masks.py>), [scripts/evaluate_icdar2015_improvement_test.py](<../../scripts/evaluate_icdar2015_improvement_test.py>), [scripts/evaluate_icdar2015_improvement_validation.py](<../../scripts/evaluate_icdar2015_improvement_validation.py>), [scripts/preview_icdar2015_improvement_predictions.py](<../../scripts/preview_icdar2015_improvement_predictions.py>), [scripts/run_icdar2015_improvement_evaluation.py](<../../scripts/run_icdar2015_improvement_evaluation.py>), [scripts/screen_icdar2015_unet_resolutions.py](<../../scripts/screen_icdar2015_unet_resolutions.py>), [scripts/select_icdar2015_improvement.py](<../../scripts/select_icdar2015_improvement.py>), [scripts/train_icdar2015_unet_improvement.py](<../../scripts/train_icdar2015_unet_improvement.py>), [segmentation/icdar2015_improvement_runtime.py](<../../segmentation/icdar2015_improvement_runtime.py>), [segmentation/icdar2015_improvements.py](<../../segmentation/icdar2015_improvements.py>).

Configuration: [configs/approximate/icdar2015_unet_improvement_v1.json](<../../configs/approximate/icdar2015_unet_improvement_v1.json>).

Tests: [tests/models/test_icdar2015_improvements.py](<../../tests/models/test_icdar2015_improvements.py>), [tests/scripts/test_icdar2015_improvement_evaluation.py](<../../tests/scripts/test_icdar2015_improvement_evaluation.py>), [tests/scripts/test_icdar2015_improvement_failures.py](<../../tests/scripts/test_icdar2015_improvement_failures.py>), [tests/scripts/test_icdar2015_improvement_selection.py](<../../tests/scripts/test_icdar2015_improvement_selection.py>), [tests/segmentation/test_icdar2015_improvement_targets.py](<../../tests/segmentation/test_icdar2015_improvement_targets.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## ICDAR2015 full training and symbol-free word F1

ID: `icdar2015-word-f1`. [Guide](<../../docs/ICDAR2015_WORD_F1.md>).

Code: [recognition/icdar2015_word_f1.py](<../../recognition/icdar2015_word_f1.py>), [scripts/audit_icdar2015_word_f1.py](<../../scripts/audit_icdar2015_word_f1.py>), [scripts/evaluate_icdar2015_totaltext_transfer_word_f1.py](<../../scripts/evaluate_icdar2015_totaltext_transfer_word_f1.py>), [scripts/train_icdar2015_totaltext_transfer.py](<../../scripts/train_icdar2015_totaltext_transfer.py>).

Configuration: [configs/approximate/icdar2015_totaltext_transfer_v1.json](<../../configs/approximate/icdar2015_totaltext_transfer_v1.json>), [configs/approximate/icdar2015_totaltext_transfer_word_f1_v1.json](<../../configs/approximate/icdar2015_totaltext_transfer_word_f1_v1.json>).

Tests: [tests/recognition/test_icdar2015_word_f1.py](<../../tests/recognition/test_icdar2015_word_f1.py>), [tests/scripts/test_audit_icdar2015_word_f1.py](<../../tests/scripts/test_audit_icdar2015_word_f1.py>), [tests/scripts/test_evaluate_icdar2015_transfer_word_f1.py](<../../tests/scripts/test_evaluate_icdar2015_transfer_word_f1.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Lightweight ICDAR2013 90% fixed-budget research

ID: `lightweight-icdar13-90`. [Guide](<../../docs/LIGHTWEIGHT_ICDAR13_90.md>).

Code: [recognition/fallback_reread.py](<../../recognition/fallback_reread.py>), [recognition/fixed_expert_budget.py](<../../recognition/fixed_expert_budget.py>), [recognition/lightweight_word_selection.py](<../../recognition/lightweight_word_selection.py>), [recognition/validated_budget_variant.py](<../../recognition/validated_budget_variant.py>), [scripts/prepare_fixed_expert_budget.py](<../../scripts/prepare_fixed_expert_budget.py>), [scripts/reread_deepsolo_boxes.py](<../../scripts/reread_deepsolo_boxes.py>), [scripts/run_lightweight_word_selection.py](<../../scripts/run_lightweight_word_selection.py>).

Configuration: [configs/approximate/fallback_parseq_reread_ic13_v1.json](<../../configs/approximate/fallback_parseq_reread_ic13_v1.json>), [configs/approximate/fixed_expert_budget_ic13_evaluation_fresh_fusion_v1.json](<../../configs/approximate/fixed_expert_budget_ic13_evaluation_fresh_fusion_v1.json>), [configs/approximate/fixed_expert_budget_ic13_evaluation_fusion_v2.json](<../../configs/approximate/fixed_expert_budget_ic13_evaluation_fusion_v2.json>), [configs/approximate/fresh_deepsolo_ic13_selected_v1.json](<../../configs/approximate/fresh_deepsolo_ic13_selected_v1.json>), [configs/approximate/lightweight_word_selection_ic13_v1.json](<../../configs/approximate/lightweight_word_selection_ic13_v1.json>), [configs/approximate/validated_budget_variant_no_extra_report_v1.json](<../../configs/approximate/validated_budget_variant_no_extra_report_v1.json>), [configs/approximate/validated_budget_variant_no_extra_v1.json](<../../configs/approximate/validated_budget_variant_no_extra_v1.json>).

Tests: [tests/recognition/test_lightweight_word_selection.py](<../../tests/recognition/test_lightweight_word_selection.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Lightweight corrected-U-Net router: retained signals and calibration

ID: `lightweight-unet-router`. [Guide](<../../docs/LIGHTWEIGHT_UNET_ROUTER.md>).

Code: [recognition/lightweight_unet_router.py](<../../recognition/lightweight_unet_router.py>), [scripts/audit_lightweight_unet_router.py](<../../scripts/audit_lightweight_unet_router.py>), [scripts/prune_lightweight_unet_router.py](<../../scripts/prune_lightweight_unet_router.py>), [scripts/run_lightweight_unet_router.py](<../../scripts/run_lightweight_unet_router.py>).

Configuration: [configs/approximate/lightweight_unet_router_v1.json](<../../configs/approximate/lightweight_unet_router_v1.json>).

Tests: [tests/recognition/test_lightweight_unet_router.py](<../../tests/recognition/test_lightweight_unet_router.py>), [tests/recognition/test_lightweight_validation_pruning.py](<../../tests/recognition/test_lightweight_validation_pruning.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Four occlusion-inspired component experiments

ID: `occlusion-four-tests`. [Guide](<../../docs/OCCLUSION_FOUR_TESTS.md>).

Code: [localize/occlusion_four_tests.py](<../../localize/occlusion_four_tests.py>), [scripts/run_occlusion_four_tests.py](<../../scripts/run_occlusion_four_tests.py>).

Configuration: [configs/approximate/occlusion_four_tests_v1.json](<../../configs/approximate/occlusion_four_tests_v1.json>).

Tests: [tests/localize/test_occlusion_four_tests.py](<../../tests/localize/test_occlusion_four_tests.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Table5: source verification and meaning of accuracy

ID: `paper-table5-source-audit`. [Guide](<../../docs/PAPER_TABLE5_SOURCE_AUDIT.md>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Research status and directions: masks, instances, crops

ID: `research-synthesis-20261021`. [Guide](<../../docs/RESEARCH_SYNTHESIS_20261021.md>).

Configuration: [configs/approximate/research_synthesis_20261021_v1.json](<../../configs/approximate/research_synthesis_20261021_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Shared pipeline: code tour for a research presentation

ID: `shared-pipeline-code-tour`. [Guide](<../../docs/SHARED_PIPELINE_CODE_TOUR.md>).

Code: [pipeline/uniform_word_pipeline.py](<../../pipeline/uniform_word_pipeline.py>), [recognition/hard_fallback_budget.py](<../../recognition/hard_fallback_budget.py>), [recognition/parseq.py](<../../recognition/parseq.py>), [recognition/uniform_encoder_context.py](<../../recognition/uniform_encoder_context.py>), [recognition/uniform_fallback.py](<../../recognition/uniform_fallback.py>), [recognition/uniform_multiscale_words.py](<../../recognition/uniform_multiscale_words.py>), [recognition/uniform_risk_fusion.py](<../../recognition/uniform_risk_fusion.py>), [recognition/uniform_utility_deployment.py](<../../recognition/uniform_utility_deployment.py>), [recognition/uniform_utility_fusion.py](<../../recognition/uniform_utility_fusion.py>), [recognition/uniform_word_union.py](<../../recognition/uniform_word_union.py>), [scripts/evaluate_uniform_utility_fusion.py](<../../scripts/evaluate_uniform_utility_fusion.py>), [scripts/infer_uniform_word_pipeline.py](<../../scripts/infer_uniform_word_pipeline.py>), [scripts/run_uniform_unet_training.py](<../../scripts/run_uniform_unet_training.py>), [scripts/train_icdar13_word_refit.py](<../../scripts/train_icdar13_word_refit.py>), [segmentation/icdar13_word_refit.py](<../../segmentation/icdar13_word_refit.py>), [segmentation/uniform_unet_fusion.py](<../../segmentation/uniform_unet_fusion.py>).

Configuration: [configs/approximate/uniform_multiscale_words_shared_v21.json](<../../configs/approximate/uniform_multiscale_words_shared_v21.json>), [configs/approximate/uniform_pipeline_deployment_v6b.json](<../../configs/approximate/uniform_pipeline_deployment_v6b.json>), [configs/approximate/uniform_pipeline_hard_budget_v9.json](<../../configs/approximate/uniform_pipeline_hard_budget_v9.json>).

Tests: [tests/pipeline/test_uniform_word_pipeline.py](<../../tests/pipeline/test_uniform_word_pipeline.py>), [tests/recognition/test_hard_fallback_budget.py](<../../tests/recognition/test_hard_fallback_budget.py>), [tests/recognition/test_uniform_encoder_context.py](<../../tests/recognition/test_uniform_encoder_context.py>), [tests/recognition/test_uniform_fallback.py](<../../tests/recognition/test_uniform_fallback.py>), [tests/recognition/test_uniform_utility_fusion.py](<../../tests/recognition/test_uniform_utility_fusion.py>), [tests/recognition/test_uniform_word_union.py](<../../tests/recognition/test_uniform_word_union.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Five teacher/student and selection pilots

ID: `teacher-student-tests`. [Guide](<../../docs/TEACHER_STUDENT_TESTS.md>).

Code: [localize/teacher_student_tests.py](<../../localize/teacher_student_tests.py>), [scripts/run_teacher_student_battery.sh](<../../scripts/run_teacher_student_battery.sh>), [scripts/run_teacher_student_tests.py](<../../scripts/run_teacher_student_tests.py>), [scripts/run_teacher_student_yolo.py](<../../scripts/run_teacher_student_yolo.py>).

Configuration: [configs/approximate/teacher_student_tests_v1.json](<../../configs/approximate/teacher_student_tests_v1.json>).

Tests: [tests/localize/test_teacher_student_tests.py](<../../tests/localize/test_teacher_student_tests.py>).

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

## Frozen trained routers with ICDAR-specific DeepSolo vocabularies

ID: `trained-router-vocabulary-replay`. [Guide](<../../docs/TRAINED_ROUTER_VOCABULARY_REPLAY.md>).

Code: [recognition/router_fallback_replay.py](<../../recognition/router_fallback_replay.py>), [scripts/audit_trained_router_vocabulary_replay.py](<../../scripts/audit_trained_router_vocabulary_replay.py>), [scripts/replay_trained_router_vocabularies.py](<../../scripts/replay_trained_router_vocabularies.py>), [scripts/report_trained_router_vocabulary_replay.py](<../../scripts/report_trained_router_vocabulary_replay.py>).

Configuration: [configs/approximate/trained_router_vocabulary_replay_v1.json](<../../configs/approximate/trained_router_vocabulary_replay_v1.json>).

Tests: [tests/recognition/test_router_fallback_replay.py](<../../tests/recognition/test_router_fallback_replay.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Token image routing with the qualified corrected U-Net

ID: `unet-corrected-token-router`. [Guide](<../../docs/UNET_CORRECTED_TOKEN_ROUTER.md>).

Code: [recognition/unet_corrected_token_interfaces.py](<../../recognition/unet_corrected_token_interfaces.py>), [scripts/audit_unet_corrected_token_router.py](<../../scripts/audit_unet_corrected_token_router.py>), [scripts/run_unet_corrected_token_router.py](<../../scripts/run_unet_corrected_token_router.py>).

Configuration: [configs/approximate/unet_corrected_token_router_v1.json](<../../configs/approximate/unet_corrected_token_router_v1.json>).

Tests: [tests/recognition/test_unet_corrected_token_audit.py](<../../tests/recognition/test_unet_corrected_token_audit.py>), [tests/recognition/test_unet_corrected_token_interfaces.py](<../../tests/recognition/test_unet_corrected_token_interfaces.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Frozen TRAIN thresholds and the 40-to-50 percent acceptance loss

ID: `unet-corrected-token-thresholds`. [Guide](<../../docs/UNET_CORRECTED_TOKEN_THRESHOLDS.md>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Corrected U-Net branch with minimum DeepSolo image fallback

ID: `unet-minimum-fallback`. [Guide](<../../docs/UNET_MINIMUM_FALLBACK.md>).

Code: [recognition/unet_fallback_coverage.py](<../../recognition/unet_fallback_coverage.py>), [recognition/unet_minimum_fallback.py](<../../recognition/unet_minimum_fallback.py>), [scripts/analyze_unet_fallback_transfer.py](<../../scripts/analyze_unet_fallback_transfer.py>), [scripts/run_unet_cold_light_router.py](<../../scripts/run_unet_cold_light_router.py>), [scripts/run_unet_minimum_fallback.py](<../../scripts/run_unet_minimum_fallback.py>).

Configuration: [configs/approximate/unet_cold_light_router_v1.json](<../../configs/approximate/unet_cold_light_router_v1.json>), [configs/approximate/unet_minimum_fallback_transfer_v1.json](<../../configs/approximate/unet_minimum_fallback_transfer_v1.json>), [configs/approximate/unet_minimum_fallback_v1.json](<../../configs/approximate/unet_minimum_fallback_v1.json>).

Tests: [tests/recognition/test_unet_fallback_coverage.py](<../../tests/recognition/test_unet_fallback_coverage.py>), [tests/recognition/test_unet_fallback_transfer.py](<../../tests/recognition/test_unet_fallback_transfer.py>), [tests/recognition/test_unet_minimum_fallback.py](<../../tests/recognition/test_unet_minimum_fallback.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Restart and failed-run storage cleanup

ID: `unet-restart-storage`. [Guide](<../../docs/UNET_RESTART_STORAGE.md>).

Code: [scripts/cleanup_failed_unet_detection_run.py](<../../scripts/cleanup_failed_unet_detection_run.py>), [scripts/run_unet_detection_methods_host.sh](<../../scripts/run_unet_detection_methods_host.sh>).

Configuration: [configs/approximate/unet_detection_methods_synthtext_targets_v2.json](<../../configs/approximate/unet_detection_methods_synthtext_targets_v2.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Uniform pipeline with hard fallback budgets

ID: `uniform-hard-budget`. [Guide](<../../docs/UNIFORM_HARD_BUDGET.md>).

Configuration: [configs/approximate/uniform_multiscale_words_shared_v21.json](<../../configs/approximate/uniform_multiscale_words_shared_v21.json>), [configs/approximate/uniform_photometric_continuation_v11.json](<../../configs/approximate/uniform_photometric_continuation_v11.json>), [configs/approximate/uniform_pipeline_hard_budget_v9.json](<../../configs/approximate/uniform_pipeline_hard_budget_v9.json>), [configs/approximate/uniform_pooled_gate_v9.json](<../../configs/approximate/uniform_pooled_gate_v9.json>), [configs/approximate/uniform_upper_encoder_v18.json](<../../configs/approximate/uniform_upper_encoder_v18.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Uniform pipeline: current verified result

ID: `uniform-pipeline-current-results`. [Guide](<../../docs/UNIFORM_PIPELINE_CURRENT_RESULTS.md>).

Code: [scripts/infer_uniform_word_pipeline.py](<../../scripts/infer_uniform_word_pipeline.py>).

Configuration: [configs/approximate/uniform_pipeline_deployment_v6b.json](<../../configs/approximate/uniform_pipeline_deployment_v6b.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Uniform U-Net, word-risk routing and conditional fusion

ID: `uniform-unet-fusion`. [Guide](<../../docs/UNIFORM_UNET_FUSION.md>).

Code: [pipeline/uniform_word_pipeline.py](<../../pipeline/uniform_word_pipeline.py>), [recognition/uniform_fallback.py](<../../recognition/uniform_fallback.py>), [recognition/uniform_lexical_risk.py](<../../recognition/uniform_lexical_risk.py>), [recognition/uniform_risk_fusion.py](<../../recognition/uniform_risk_fusion.py>), [recognition/uniform_utility_fusion.py](<../../recognition/uniform_utility_fusion.py>), [scripts/audit_uniform_unet_training.py](<../../scripts/audit_uniform_unet_training.py>), [scripts/infer_uniform_word_pipeline.py](<../../scripts/infer_uniform_word_pipeline.py>), [scripts/manage_uniform_decoder_oof.py](<../../scripts/manage_uniform_decoder_oof.py>), [scripts/profile_uniform_unet_compute.py](<../../scripts/profile_uniform_unet_compute.py>), [scripts/run_uniform_unet_training.py](<../../scripts/run_uniform_unet_training.py>), [scripts/select_uniform_lexical_policy.py](<../../scripts/select_uniform_lexical_policy.py>), [scripts/supervise_uniform_unet_training.py](<../../scripts/supervise_uniform_unet_training.py>), [scripts/supervise_uniform_utility_oof.py](<../../scripts/supervise_uniform_utility_oof.py>), [scripts/train_uniform_risk_fusion.py](<../../scripts/train_uniform_risk_fusion.py>), [scripts/train_uniform_utility_fusion.py](<../../scripts/train_uniform_utility_fusion.py>), [scripts/uniform_decoder_folds.py](<../../scripts/uniform_decoder_folds.py>), [segmentation/uniform_unet_fusion.py](<../../segmentation/uniform_unet_fusion.py>).

Configuration: [configs/approximate/uniform_decoder_oof_v3.json](<../../configs/approximate/uniform_decoder_oof_v3.json>), [configs/approximate/uniform_pipeline_deployment_v2.json](<../../configs/approximate/uniform_pipeline_deployment_v2.json>), [configs/approximate/uniform_unet_fusion_v1.json](<../../configs/approximate/uniform_unet_fusion_v1.json>), [configs/approximate/uniform_utility_oof_supervisor_v3.json](<../../configs/approximate/uniform_utility_oof_supervisor_v3.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Word-guided contours and actual crop recognition

ID: `word-guided-contours`. [Guide](<../../docs/WORD_GUIDED_CONTOURS.md>).

Code: [localize/word_guided_contours.py](<../../localize/word_guided_contours.py>), [scripts/evaluate_word_multitask_crops.py](<../../scripts/evaluate_word_multitask_crops.py>).

Configuration: [configs/approximate/word_contour_refinement_v1.json](<../../configs/approximate/word_contour_refinement_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Symbol-free comparison of word-slot U-Net, PARSeq and DeepSolo

ID: `word-slot-alphanumeric`. [Guide](<../../docs/WORD_SLOT_ALPHANUMERIC.md>).

Code: [recognition/word_slot_routing.py](<../../recognition/word_slot_routing.py>), [recognition/word_slot_text.py](<../../recognition/word_slot_text.py>), [scripts/reassess_word_slot_alphanumeric.py](<../../scripts/reassess_word_slot_alphanumeric.py>).

Configuration: [configs/approximate/word_slot_alphanumeric_v1.json](<../../configs/approximate/word_slot_alphanumeric_v1.json>).

Tests: [tests/recognition/test_word_slot_alphanumeric.py](<../../tests/recognition/test_word_slot_alphanumeric.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Multi-scale word slots and complementary DeepSolo routing

ID: `word-slot-methods`. [Guide](<../../docs/WORD_SLOT_METHODS.md>).

Code: [models/word_slot_unet.py](<../../models/word_slot_unet.py>), [pipeline/word_slot_specialist.py](<../../pipeline/word_slot_specialist.py>), [recognition/word_slot_routing.py](<../../recognition/word_slot_routing.py>), [recognition/word_slot_teachers.py](<../../recognition/word_slot_teachers.py>), [scripts/analyze_word_slot_routers.py](<../../scripts/analyze_word_slot_routers.py>), [scripts/evaluate_word_slot_methods.py](<../../scripts/evaluate_word_slot_methods.py>), [scripts/qualify_word_slot_mae_teacher.py](<../../scripts/qualify_word_slot_mae_teacher.py>), [scripts/report_word_slot_methods.py](<../../scripts/report_word_slot_methods.py>), [scripts/run_word_slot_methods.py](<../../scripts/run_word_slot_methods.py>), [scripts/run_word_slot_specialist.py](<../../scripts/run_word_slot_specialist.py>), [segmentation/word_slot_objectives.py](<../../segmentation/word_slot_objectives.py>).

Configuration: [configs/approximate/word_slot_analysis_v1.json](<../../configs/approximate/word_slot_analysis_v1.json>), [configs/approximate/word_slot_controls_v1.json](<../../configs/approximate/word_slot_controls_v1.json>), [configs/approximate/word_slot_mae_teacher_v2.json](<../../configs/approximate/word_slot_mae_teacher_v2.json>), [configs/approximate/word_slot_methods_v1.json](<../../configs/approximate/word_slot_methods_v1.json>), [configs/approximate/word_slot_router_v1.json](<../../configs/approximate/word_slot_router_v1.json>).

Tests: [tests/models/test_word_slot_unet.py](<../../tests/models/test_word_slot_unet.py>), [tests/recognition/test_word_slot_methods.py](<../../tests/recognition/test_word_slot_methods.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.
