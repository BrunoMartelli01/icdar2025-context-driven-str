# Context, captioning, CLIP and YOLO routing

[All families](../README.md) · [Code index](../CODE_INDEX.md) · [Saved results](../results/README.md)

Category membership is navigation, not a shared evaluation protocol or a completion claim.

## Frozen InstructBLIP paired routing: TRAIN and TEST

ID: `budgeted-paired-e2e`. [Guide](<../../docs/BUDGETED_PAIRED_E2E.md>).

Code: [diagnostics/run_budgeted_paired_e2e_host.sh](<../../diagnostics/run_budgeted_paired_e2e_host.sh>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Cheap DeepSolo-supervised router at half-image acceptance

ID: `budget-distilled-router`. [Guide](<../../docs/BUDGET_DISTILLED_ROUTER.md>).

Code: [recognition/budget_distilled_router.py](<../../recognition/budget_distilled_router.py>), [recognition/essential_portable_router.py](<../../recognition/essential_portable_router.py>), [scripts/audit_budget_distilled_router.py](<../../scripts/audit_budget_distilled_router.py>), [scripts/audit_essential_router_budget.py](<../../scripts/audit_essential_router_budget.py>), [scripts/export_essential_portable_router.py](<../../scripts/export_essential_portable_router.py>), [scripts/report_budget_distilled_router.py](<../../scripts/report_budget_distilled_router.py>), [scripts/run_budget_distilled_router.py](<../../scripts/run_budget_distilled_router.py>).

Configuration: [configs/approximate/budget_distilled_router_adapt_v2.json](<../../configs/approximate/budget_distilled_router_adapt_v2.json>), [configs/approximate/budget_distilled_router_local_paths_v1.json](<../../configs/approximate/budget_distilled_router_local_paths_v1.json>), [configs/approximate/budget_distilled_router_unseen_v3.json](<../../configs/approximate/budget_distilled_router_unseen_v3.json>), [configs/approximate/budget_distilled_router_v1.json](<../../configs/approximate/budget_distilled_router_v1.json>).

Tests: [tests/recognition/test_budget_distilled_router.py](<../../tests/recognition/test_budget_distilled_router.py>), [tests/recognition/test_essential_portable_router.py](<../../tests/recognition/test_essential_portable_router.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## CLIP whole-image router on Text+Kernel epoch61

ID: `clip-image-router`. [Guide](<../../docs/CLIP_IMAGE_ROUTER.md>).

Code: [scripts/run_clip_image_router.py](<../../scripts/run_clip_image_router.py>), [scripts/run_clip_image_router_fallback.py](<../../scripts/run_clip_image_router_fallback.py>), [scripts/run_clip_image_router_host.sh](<../../scripts/run_clip_image_router_host.sh>), [scripts/run_clip_image_router_word_advantage.py](<../../scripts/run_clip_image_router_word_advantage.py>), [scripts/run_clip_image_router_word_advantage_host.sh](<../../scripts/run_clip_image_router_word_advantage_host.sh>).

Configuration: [configs/approximate/clip_image_router_word_advantage_v1.json](<../../configs/approximate/clip_image_router_word_advantage_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## CLIP whole-image router: separate TRAIN and TEST

ID: `clip-router-e2e`. [Guide](<../../docs/CLIP_ROUTER_E2E.md>).

Code: [scripts/run_clip_router_e2e.py](<../../scripts/run_clip_router_e2e.py>), [scripts/run_clip_router_e2e_host.sh](<../../scripts/run_clip_router_e2e_host.sh>).

Configuration: [configs/approximate/clip_router_e2e_local_paths_v1.json](<../../configs/approximate/clip_router_e2e_local_paths_v1.json>), [configs/approximate/clip_router_e2e_v1.json](<../../configs/approximate/clip_router_e2e_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Literature-motivated CLIP router experiments (v3)

ID: `clip-router-literature-models`. [Guide](<../../docs/CLIP_ROUTER_LITERATURE_MODELS.md>).

Code: [recognition/clip_router_context_chunks.py](<../../recognition/clip_router_context_chunks.py>), [scripts/run_clip_router_literature_v3.py](<../../scripts/run_clip_router_literature_v3.py>).

Configuration: [configs/approximate/clip_router_literature_local_paths_v3.json](<../../configs/approximate/clip_router_literature_local_paths_v3.json>), [configs/approximate/clip_router_literature_v3.json](<../../configs/approximate/clip_router_literature_v3.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Updated CLIP-family encoders and transcript prompts

ID: `clip-router-prompt-ablation`. [Guide](<../../docs/CLIP_ROUTER_PROMPT_ABLATION.md>).

Code: [scripts/run_clip_router_prompt_ablation.py](<../../scripts/run_clip_router_prompt_ablation.py>), [scripts/summarize_clip_router_prompt_ablation.py](<../../scripts/summarize_clip_router_prompt_ablation.py>), [scripts/sweep_clip_router_e2e.py](<../../scripts/sweep_clip_router_e2e.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Frozen CLIP router threshold curves

ID: `clip-router-threshold-sweep`. [Guide](<../../docs/CLIP_ROUTER_THRESHOLD_SWEEP.md>).

Code: [scripts/sweep_clip_router_e2e.py](<../../scripts/sweep_clip_router_e2e.py>).

Configuration: [configs/approximate/clip_router_threshold_sweep_local_paths_v1.json](<../../configs/approximate/clip_router_threshold_sweep_local_paths_v1.json>), [configs/approximate/clip_router_threshold_sweep_v1.json](<../../configs/approximate/clip_router_threshold_sweep_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Native Long-CLIP / LLM2CLIP OCR router experiment

ID: `clip-specialized-routers`. [Guide](<../../docs/CLIP_SPECIALIZED_ROUTERS.md>).

Code: [scripts/audit_clip_specialized_routers.py](<../../scripts/audit_clip_specialized_routers.py>), [scripts/complete_clip_router_train_fallback.py](<../../scripts/complete_clip_router_train_fallback.py>), [scripts/run_clip_specialized_routers.py](<../../scripts/run_clip_specialized_routers.py>).

Numerical results: [clip-encoders-budget](../results/clip-encoders-budget.md), [clip-encoders-threshold](../results/clip-encoders-threshold.md).

## Improving scene-caption quality

ID: `detailed-scene-captions`. [Guide](<../../docs/DETAILED_SCENE_CAPTIONS.md>).

Code: [caption/detailed_scene.py](<../../caption/detailed_scene.py>), [caption/florence_scene.py](<../../caption/florence_scene.py>), [caption/qwen_scene.py](<../../caption/qwen_scene.py>), [scripts/audit_detailed_captions.py](<../../scripts/audit_detailed_captions.py>), [scripts/caption_scene.py](<../../scripts/caption_scene.py>), [scripts/probe_detailed_scene_captions.py](<../../scripts/probe_detailed_scene_captions.py>), [scripts/render_detailed_captions.py](<../../scripts/render_detailed_captions.py>), [scripts/run_detailed_caption_comparison.py](<../../scripts/run_detailed_caption_comparison.py>).

Configuration: [configs/approximate/detailed_caption_router_comparison_v13.json](<../../configs/approximate/detailed_caption_router_comparison_v13.json>), [configs/approximate/detailed_scene_captions_qwen_batch2_v12.json](<../../configs/approximate/detailed_scene_captions_qwen_batch2_v12.json>), [configs/approximate/detailed_scene_captions_qwen_offload15_v5.json](<../../configs/approximate/detailed_scene_captions_qwen_offload15_v5.json>), [configs/approximate/detailed_scene_captions_qwen_selected_v10.json](<../../configs/approximate/detailed_scene_captions_qwen_selected_v10.json>), [configs/approximate/detailed_scene_captions_qwen_v3.json](<../../configs/approximate/detailed_scene_captions_qwen_v3.json>), [configs/approximate/detailed_scene_captions_v1.json](<../../configs/approximate/detailed_scene_captions_v1.json>), [configs/approximate/detailed_scene_captions_v2.json](<../../configs/approximate/detailed_scene_captions_v2.json>).

Tests: [tests/caption/test_detailed_scene.py](<../../tests/caption/test_detailed_scene.py>), [tests/caption/test_qwen_scene.py](<../../tests/caption/test_qwen_scene.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Matched training of CLIP, BLIP and combined image routers

ID: `encoder-family-router`. [Guide](<../../docs/ENCODER_FAMILY_ROUTER.md>).

Code: [recognition/blip_visual_only.py](<../../recognition/blip_visual_only.py>), [recognition/encoder_family_inference.py](<../../recognition/encoder_family_inference.py>), [recognition/encoder_family_router.py](<../../recognition/encoder_family_router.py>), [scripts/analyze_encoder_family_signal.py](<../../scripts/analyze_encoder_family_signal.py>), [scripts/audit_encoder_family_router.py](<../../scripts/audit_encoder_family_router.py>), [scripts/qualify_blip_visual_only.py](<../../scripts/qualify_blip_visual_only.py>), [scripts/run_encoder_family_router.py](<../../scripts/run_encoder_family_router.py>).

Configuration: [configs/approximate/encoder_family_router_v1.json](<../../configs/approximate/encoder_family_router_v1.json>).

Tests: [tests/recognition/test_blip_visual_only.py](<../../tests/recognition/test_blip_visual_only.py>), [tests/recognition/test_encoder_family_inference.py](<../../tests/recognition/test_encoder_family_inference.py>), [tests/recognition/test_encoder_family_router.py](<../../tests/recognition/test_encoder_family_router.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Essential router weights and validation calibration near half fallback

ID: `essential-router-budget50`. [Guide](<../../docs/ESSENTIAL_ROUTER_BUDGET50.md>).

Code: [recognition/essential_budget_router.py](<../../recognition/essential_budget_router.py>), [scripts/audit_essential_router_budget.py](<../../scripts/audit_essential_router_budget.py>), [scripts/measure_router_weight_storage.py](<../../scripts/measure_router_weight_storage.py>), [scripts/run_essential_router_budget.py](<../../scripts/run_essential_router_budget.py>).

Configuration: [configs/approximate/essential_router_budget50_v1.json](<../../configs/approximate/essential_router_budget50_v1.json>).

Tests: [tests/recognition/test_essential_router_budget.py](<../../tests/recognition/test_essential_router_budget.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Essential router transfer to ICDAR2013 and ICDAR2015

ID: `essential-router-icdar-transfer`. [Guide](<../../docs/ESSENTIAL_ROUTER_ICDAR_TRANSFER.md>).

Code: [recognition/corrected_unet_transfer.py](<../../recognition/corrected_unet_transfer.py>), [recognition/essential_threshold_deployment.py](<../../recognition/essential_threshold_deployment.py>), [scripts/export_essential_transfer_deepsolo.py](<../../scripts/export_essential_transfer_deepsolo.py>), [scripts/qualify_corrected_unet_transfer.py](<../../scripts/qualify_corrected_unet_transfer.py>), [scripts/run_essential_icdar_transfer.py](<../../scripts/run_essential_icdar_transfer.py>).

Configuration: [configs/approximate/essential_router_icdar_transfer_v1.json](<../../configs/approximate/essential_router_icdar_transfer_v1.json>).

Tests: [tests/recognition/test_essential_icdar_transfer.py](<../../tests/recognition/test_essential_icdar_transfer.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Essential router: threshold selection on three datasets

ID: `essential-router-threshold-analysis`. [Guide](<../../docs/ESSENTIAL_ROUTER_THRESHOLD_ANALYSIS.md>).

Code: [recognition/essential_threshold_deployment.py](<../../recognition/essential_threshold_deployment.py>), [recognition/essential_threshold_frontier.py](<../../recognition/essential_threshold_frontier.py>), [scripts/analyze_essential_fallback_control.py](<../../scripts/analyze_essential_fallback_control.py>), [scripts/analyze_essential_oracle.py](<../../scripts/analyze_essential_oracle.py>), [scripts/analyze_essential_router_thresholds.py](<../../scripts/analyze_essential_router_thresholds.py>), [scripts/audit_essential_profiles_oracle.py](<../../scripts/audit_essential_profiles_oracle.py>), [scripts/audit_essential_thresholds.py](<../../scripts/audit_essential_thresholds.py>), [scripts/fit_essential_yolo_head.py](<../../scripts/fit_essential_yolo_head.py>), [scripts/plot_essential_thresholds.py](<../../scripts/plot_essential_thresholds.py>).

Configuration: [configs/approximate/essential_router_normalized_threshold_analysis_v1.json](<../../configs/approximate/essential_router_normalized_threshold_analysis_v1.json>), [configs/approximate/essential_router_threshold_analysis_v1.json](<../../configs/approximate/essential_router_threshold_analysis_v1.json>), [configs/approximate/essential_yolo_matched_head_v1.json](<../../configs/approximate/essential_yolo_matched_head_v1.json>).

Tests: [tests/recognition/test_essential_conditional_oracle.py](<../../tests/recognition/test_essential_conditional_oracle.py>), [tests/recognition/test_essential_icdar_transfer.py](<../../tests/recognition/test_essential_icdar_transfer.py>), [tests/recognition/test_essential_threshold_deployment.py](<../../tests/recognition/test_essential_threshold_deployment.py>), [tests/recognition/test_essential_threshold_frontier.py](<../../tests/recognition/test_essential_threshold_frontier.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Dataset-specific cheap YOLO/U-Net routing on ICDAR2013 and ICDAR2015

ID: `icdar-trained-budget-router`. [Guide](<../../docs/ICDAR_TRAINED_BUDGET_ROUTER.md>).

Code: [recognition/icdar_budget_deployment.py](<../../recognition/icdar_budget_deployment.py>), [recognition/icdar_budget_router.py](<../../recognition/icdar_budget_router.py>), [scripts/audit_icdar_budget_router.py](<../../scripts/audit_icdar_budget_router.py>), [scripts/icdar_budget_data.py](<../../scripts/icdar_budget_data.py>), [scripts/icdar_budget_stages.py](<../../scripts/icdar_budget_stages.py>), [scripts/report_icdar_budget_router.py](<../../scripts/report_icdar_budget_router.py>), [scripts/train_icdar_budget_router.py](<../../scripts/train_icdar_budget_router.py>).

Configuration: [configs/approximate/icdar_trained_budget_router_local_paths_v1.json](<../../configs/approximate/icdar_trained_budget_router_local_paths_v1.json>), [configs/approximate/icdar_trained_budget_router_v1.json](<../../configs/approximate/icdar_trained_budget_router_v1.json>).

Tests: [tests/recognition/test_icdar_budget_deployment.py](<../../tests/recognition/test_icdar_budget_deployment.py>), [tests/recognition/test_icdar_budget_router.py](<../../tests/recognition/test_icdar_budget_router.py>), [tests/scripts/test_audit_icdar_budget_router.py](<../../tests/scripts/test_audit_icdar_budget_router.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Caption router versus the qualified lightweight router

ID: `paper-caption-router-comparison`. [Guide](<../../docs/PAPER_CAPTION_ROUTER_COMPARISON.md>).

Code: [caption/paper_batch_blip2.py](<../../caption/paper_batch_blip2.py>), [recognition/paper_caption_comparison.py](<../../recognition/paper_caption_comparison.py>), [scripts/audit_paper_caption_comparison.py](<../../scripts/audit_paper_caption_comparison.py>), [scripts/run_paper_caption_comparison.py](<../../scripts/run_paper_caption_comparison.py>).

Configuration: [configs/approximate/paper_caption_comparison_batch8_v2.json](<../../configs/approximate/paper_caption_comparison_batch8_v2.json>).

Tests: [tests/caption/test_paper_batch_blip2.py](<../../tests/caption/test_paper_batch_blip2.py>), [tests/recognition/test_paper_caption_comparison.py](<../../tests/recognition/test_paper_caption_comparison.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Paper image router baseline — approximate reproduction

ID: `paper-image-router-baseline`. [Guide](<../../docs/PAPER_IMAGE_ROUTER_BASELINE.md>).

Code: [scripts/run_paper_image_router_baseline.py](<../../scripts/run_paper_image_router_baseline.py>).

Configuration: [configs/approximate/paper_image_router_baseline_local_paths_v1.json](<../../configs/approximate/paper_image_router_baseline_local_paths_v1.json>), [configs/approximate/paper_image_router_baseline_v1.json](<../../configs/approximate/paper_image_router_baseline_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Accuracy gaps and conditional U-Net requirements

ID: `router-accuracy-targets`. [Guide](<../../docs/ROUTER_ACCURACY_TARGETS.md>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Controlled tests of context-router failure causes

ID: `router-cause-tests`. [Guide](<../../docs/ROUTER_CAUSE_TESTS.md>).

Code: [provenance/router_cause_tests_20261026_completion.json](<../../provenance/router_cause_tests_20261026_completion.json>), [recognition/router_cause_tests.py](<../../recognition/router_cause_tests.py>), [scripts/analyze_router_cause_tests.py](<../../scripts/analyze_router_cause_tests.py>), [scripts/analyze_router_learning_steps.py](<../../scripts/analyze_router_learning_steps.py>), [scripts/router_cause_gpu.py](<../../scripts/router_cause_gpu.py>), [scripts/router_cause_observations.py](<../../scripts/router_cause_observations.py>), [scripts/router_cause_training.py](<../../scripts/router_cause_training.py>), [scripts/run_router_cause_tests.py](<../../scripts/run_router_cause_tests.py>).

Configuration: [configs/approximate/router_cause_analysis_v1.json](<../../configs/approximate/router_cause_analysis_v1.json>), [configs/approximate/router_cause_tests_local_paths_v1.json](<../../configs/approximate/router_cause_tests_local_paths_v1.json>), [configs/approximate/router_cause_tests_v1.json](<../../configs/approximate/router_cause_tests_v1.json>), [configs/approximate/router_learning_steps_v1.json](<../../configs/approximate/router_learning_steps_v1.json>).

Tests: [tests/recognition/test_router_cause_analysis.py](<../../tests/recognition/test_router_cause_analysis.py>), [tests/recognition/test_router_cause_tests.py](<../../tests/recognition/test_router_cause_tests.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Frozen routers with improved DeepSolo and two transcription metrics

ID: `router-improved-deepsolo-replay`. [Guide](<../../docs/ROUTER_IMPROVED_DEEPSOLO_REPLAY.md>).

Code: [provenance/router_improved_deepsolo_20261026_completion.json](<../../provenance/router_improved_deepsolo_20261026_completion.json>), [scripts/replay_router_improved_deepsolo.py](<../../scripts/replay_router_improved_deepsolo.py>).

Configuration: [configs/approximate/router_improved_deepsolo_replay_local_paths_v1.json](<../../configs/approximate/router_improved_deepsolo_replay_local_paths_v1.json>), [configs/approximate/router_improved_deepsolo_replay_v1.json](<../../configs/approximate/router_improved_deepsolo_replay_v1.json>), [configs/approximate/router_improved_deepsolo_small_quota_analysis_v1.json](<../../configs/approximate/router_improved_deepsolo_small_quota_analysis_v1.json>).

Tests: [tests/recognition/test_router_improved_deepsolo_replay.py](<../../tests/recognition/test_router_improved_deepsolo_replay.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## ICDAR router retraining with corrected DeepSolo teachers

ID: `router-lexicon-retraining`. [Guide](<../../docs/ROUTER_LEXICON_RETRAINING.md>).

Configuration: [configs/approximate/router_lexicon_retraining_v1.json](<../../configs/approximate/router_lexicon_retraining_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Router generalization and held-out validation calibration

ID: `router-validation-study`. [Guide](<../../docs/ROUTER_VALIDATION_STUDY.md>).

Code: [provenance/router_validation_20261027_completion.json](<../../provenance/router_validation_20261027_completion.json>), [recognition/router_validation.py](<../../recognition/router_validation.py>), [scripts/diagnose_router_generalization.py](<../../scripts/diagnose_router_generalization.py>), [scripts/report_router_validation.py](<../../scripts/report_router_validation.py>), [scripts/router_gt_crop_control.py](<../../scripts/router_gt_crop_control.py>), [scripts/router_localization_control.py](<../../scripts/router_localization_control.py>), [scripts/run_router_validation_study.py](<../../scripts/run_router_validation_study.py>).

Configuration: [configs/approximate/router_gt_crop_control_v1.json](<../../configs/approximate/router_gt_crop_control_v1.json>), [configs/approximate/router_localization_control_v1.json](<../../configs/approximate/router_localization_control_v1.json>), [configs/approximate/router_validation_local_paths_v1.json](<../../configs/approximate/router_validation_local_paths_v1.json>), [configs/approximate/router_validation_v1.json](<../../configs/approximate/router_validation_v1.json>).

Tests: [tests/recognition/test_router_localization_control.py](<../../tests/recognition/test_router_localization_control.py>), [tests/recognition/test_router_validation.py](<../../tests/recognition/test_router_validation.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Cached YOLO/PARSeq confidence controls and CTW1500 failures

ID: `scene-text-yolo-thresholds`. [Guide](<../../docs/SCENE_TEXT_YOLO_THRESHOLDS.md>).

Code: [scripts/test_scene_text_yolo_thresholds.sh](<../../scripts/test_scene_text_yolo_thresholds.sh>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Frozen YOLO/PARSeq transfer to other TEST datasets

ID: `scene-text-yolo-transfer`. [Guide](<../../docs/SCENE_TEXT_YOLO_TRANSFER.md>).

Code: [scripts/test_scene_text_yolo.sh](<../../scripts/test_scene_text_yolo.sh>).

Configuration: [configs/approximate/scene_text_yolo_parseq_transfer_v1.json](<../../configs/approximate/scene_text_yolo_parseq_transfer_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Independent inventories and bounded routing corrections

ID: `shadow-inventory-router`. [Guide](<../../docs/SHADOW_INVENTORY_ROUTER.md>).

Code: [recognition/boundary_inventory_router.py](<../../recognition/boundary_inventory_router.py>), [recognition/shadow_inventory_router.py](<../../recognition/shadow_inventory_router.py>), [scripts/audit_boundary_inventory_router.py](<../../scripts/audit_boundary_inventory_router.py>), [scripts/audit_shadow_inventory_router.py](<../../scripts/audit_shadow_inventory_router.py>), [scripts/report_shadow_inventory_router.py](<../../scripts/report_shadow_inventory_router.py>), [scripts/run_boundary_inventory_router.py](<../../scripts/run_boundary_inventory_router.py>), [scripts/run_shadow_inventory_router.py](<../../scripts/run_shadow_inventory_router.py>).

Configuration: [configs/approximate/boundary_inventory_router_local_paths_v1.json](<../../configs/approximate/boundary_inventory_router_local_paths_v1.json>), [configs/approximate/boundary_inventory_router_v1.json](<../../configs/approximate/boundary_inventory_router_v1.json>), [configs/approximate/shadow_inventory_router_local_paths_v1.json](<../../configs/approximate/shadow_inventory_router_local_paths_v1.json>), [configs/approximate/shadow_inventory_router_v1.json](<../../configs/approximate/shadow_inventory_router_v1.json>).

Tests: [tests/recognition/test_boundary_inventory_router.py](<../../tests/recognition/test_boundary_inventory_router.py>), [tests/recognition/test_shadow_inventory_router.py](<../../tests/recognition/test_shadow_inventory_router.py>), [tests/recognition/test_trainable_image_router.py](<../../tests/recognition/test_trainable_image_router.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Three context routers with full-TRAIN benefit supervision

ID: `token-context-router`. [Guide](<../../docs/TOKEN_CONTEXT_ROUTER.md>).

Code: [recognition/token_context_router.py](<../../recognition/token_context_router.py>), [recognition/token_context_runtime.py](<../../recognition/token_context_runtime.py>), [scripts/audit_token_context_router.py](<../../scripts/audit_token_context_router.py>), [scripts/run_token_context_router.py](<../../scripts/run_token_context_router.py>), [scripts/token_context_data.py](<../../scripts/token_context_data.py>), [scripts/token_context_training.py](<../../scripts/token_context_training.py>).

Configuration: [configs/approximate/token_context_router_local_paths_v1.json](<../../configs/approximate/token_context_router_local_paths_v1.json>), [configs/approximate/token_context_router_v1.json](<../../configs/approximate/token_context_router_v1.json>).

Tests: [tests/recognition/test_token_context_router.py](<../../tests/recognition/test_token_context_router.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Independent word-slot diagnostic on Total-Text

ID: `totaltext-independent-detector-slots`. [Guide](<../../docs/TOTALTEXT_INDEPENDENT_DETECTOR_SLOTS.md>).

Code: [scripts/finish_totaltext_detector_slots.sh](<../../scripts/finish_totaltext_detector_slots.sh>), [scripts/test_totaltext_detector_slots.sh](<../../scripts/test_totaltext_detector_slots.sh>).

Configuration: [configs/approximate/totaltext_independent_detector_slots_local_paths_v1.json](<../../configs/approximate/totaltext_independent_detector_slots_local_paths_v1.json>), [configs/approximate/totaltext_independent_detector_slots_v1.json](<../../configs/approximate/totaltext_independent_detector_slots_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## CRAFT, DBNet and FAST regions with PARSeq

ID: `totaltext-text-detectors`. [Guide](<../../docs/TOTALTEXT_TEXT_DETECTORS.md>).

Code: [scripts/test_totaltext_text_detectors.sh](<../../scripts/test_totaltext_text_detectors.sh>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Direct YOLO boxes → PARSeq on Total-Text

ID: `totaltext-yolo-direct-parseq`. [Guide](<../../docs/TOTALTEXT_YOLO_DIRECT_PARSEQ.md>).

Code: [scripts/test_totaltext_yolo_direct_parseq.sh](<../../scripts/test_totaltext_yolo_direct_parseq.sh>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Exploratory YOLO failure analysis and improvement battery

ID: `totaltext-yolo-improvements`. [Guide](<../../docs/TOTALTEXT_YOLO_IMPROVEMENTS.md>).

Code: [scripts/test_totaltext_yolo_improvements.sh](<../../scripts/test_totaltext_yolo_improvements.sh>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Lightweight word-object slots for frozen Total-Text crops

ID: `totaltext-yolo-word-slots`. [Guide](<../../docs/TOTALTEXT_YOLO_WORD_SLOTS.md>).

Code: [scripts/run_totaltext_yolo_word_slots.sh](<../../scripts/run_totaltext_yolo_word_slots.sh>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Matched token-router training on qualified U-Net and YOLO

ID: `unet-token-router`. [Guide](<../../docs/UNET_TOKEN_ROUTER.md>).

Code: [provenance/unet_token_router_20261027_completion.json](<../../provenance/unet_token_router_20261027_completion.json>), [recognition/unet_token_routing.py](<../../recognition/unet_token_routing.py>), [scripts/report_unet_token_router.py](<../../scripts/report_unet_token_router.py>), [scripts/run_unet_token_router.py](<../../scripts/run_unet_token_router.py>), [scripts/unet_token_router_stages.py](<../../scripts/unet_token_router_stages.py>).

Configuration: [configs/approximate/unet_token_router_local_paths_v1.json](<../../configs/approximate/unet_token_router_local_paths_v1.json>), [configs/approximate/unet_token_router_v1.json](<../../configs/approximate/unet_token_router_v1.json>).

Tests: [tests/recognition/test_unet_token_routing.py](<../../tests/recognition/test_unet_token_routing.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Training-free contextual image-quality routing

ID: `yolo-context-quality-router`. [Guide](<../../docs/YOLO_CONTEXT_QUALITY_ROUTER.md>).

Code: [scripts/run_yolo_context_quality.py](<../../scripts/run_yolo_context_quality.py>).

Configuration: [configs/approximate/yolo_context_quality_local_paths_v1.json](<../../configs/approximate/yolo_context_quality_local_paths_v1.json>), [configs/approximate/yolo_context_quality_v1.json](<../../configs/approximate/yolo_context_quality_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Frozen YOLO/PARSeq evaluation with don't-care neutralization

ID: `yolo-dont-care-evaluation`. [Guide](<../../docs/YOLO_DONT_CARE_EVALUATION.md>).

Code: [scripts/test_yolo_evaluation.sh](<../../scripts/test_yolo_evaluation.sh>).

Configuration: [configs/approximate/yolo_dont_care_word_evaluation_local_paths_v1.json](<../../configs/approximate/yolo_dont_care_word_evaluation_local_paths_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Per-image F1 quality targets with frozen DeepSolo fallback

ID: `yolo-f1-quality-study`. [Guide](<../../docs/YOLO_F1_QUALITY_STUDY.md>).

Code: [recognition/f1_quality_study.py](<../../recognition/f1_quality_study.py>), [scripts/analyze_yolo_f1_quality_study.py](<../../scripts/analyze_yolo_f1_quality_study.py>), [scripts/audit_yolo_f1_quality_study.py](<../../scripts/audit_yolo_f1_quality_study.py>), [scripts/run_yolo_f1_quality_study.py](<../../scripts/run_yolo_f1_quality_study.py>).

Configuration: [configs/approximate/yolo_f1_quality_study_local_paths_v1.json](<../../configs/approximate/yolo_f1_quality_study_local_paths_v1.json>), [configs/approximate/yolo_f1_quality_study_v1.json](<../../configs/approximate/yolo_f1_quality_study_v1.json>).

Tests: [tests/recognition/test_f1_quality_study.py](<../../tests/recognition/test_f1_quality_study.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## YOLO: paired CLIP / paper-router comparison

ID: `yolo-paired-router-comparison`. [Guide](<../../docs/YOLO_PAIRED_ROUTER_COMPARISON.md>).

Code: [scripts/audit_yolo_router_comparison.py](<../../scripts/audit_yolo_router_comparison.py>), [scripts/run_yolo_router_comparison.py](<../../scripts/run_yolo_router_comparison.py>).

Configuration: [configs/approximate/yolo_router_comparison_local_paths_v1.json](<../../configs/approximate/yolo_router_comparison_local_paths_v1.json>), [configs/approximate/yolo_router_comparison_v1.json](<../../configs/approximate/yolo_router_comparison_v1.json>).

Numerical results: [yolo-test-equal](../results/yolo-test-equal.md), [yolo-test-transferred](../results/yolo-test-transferred.md), [yolo-train-equal](../results/yolo-train-equal.md).

## YOLO/PARSeq: four TEST datasets and per-image compute estimates

ID: `yolo-parseq-four-dataset-comparison`. [Guide](<../../docs/YOLO_PARSEQ_FOUR_DATASET_COMPARISON.md>).

Code: [scripts/test_yolo_four_datasets.sh](<../../scripts/test_yolo_four_datasets.sh>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Residual-region and word-error-cost routing

ID: `yolo-residual-cost-router`. [Guide](<../../docs/YOLO_RESIDUAL_COST_ROUTER.md>).

Code: [recognition/residual_cost_router.py](<../../recognition/residual_cost_router.py>), [scripts/analyze_yolo_residual_cost_router.py](<../../scripts/analyze_yolo_residual_cost_router.py>), [scripts/run_yolo_residual_cost_router.py](<../../scripts/run_yolo_residual_cost_router.py>).

Configuration: [configs/approximate/yolo_residual_cost_router_local_paths_v1.json](<../../configs/approximate/yolo_residual_cost_router_local_paths_v1.json>), [configs/approximate/yolo_residual_cost_router_v1.json](<../../configs/approximate/yolo_residual_cost_router_v1.json>).

Tests: [tests/recognition/test_residual_cost_router.py](<../../tests/recognition/test_residual_cost_router.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Training-free visual residual-text test

ID: `yolo-residual-text-vqa`. [Guide](<../../docs/YOLO_RESIDUAL_TEXT_VQA.md>).

Code: [scripts/run_yolo_residual_vqa.py](<../../scripts/run_yolo_residual_vqa.py>).

Configuration: [configs/approximate/yolo_residual_vqa_local_paths_v1.json](<../../configs/approximate/yolo_residual_vqa_local_paths_v1.json>), [configs/approximate/yolo_residual_vqa_v1.json](<../../configs/approximate/yolo_residual_vqa_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## YOLO routing with inference-only text coverage and resolution controls

ID: `yolo-text-coverage-resolution`. [Guide](<../../docs/YOLO_TEXT_COVERAGE_RESOLUTION.md>).

Code: [scripts/analyze_yolo_coverage_resolution.py](<../../scripts/analyze_yolo_coverage_resolution.py>), [scripts/run_yolo_coverage_resolution.py](<../../scripts/run_yolo_coverage_resolution.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Supervised router targeting fixed-budget oracle separation

ID: `yolo-trainable-router`. [Guide](<../../docs/YOLO_TRAINABLE_ROUTER.md>).

Code: [recognition/trainable_image_router.py](<../../recognition/trainable_image_router.py>), [scripts/analyze_trainable_image_router.py](<../../scripts/analyze_trainable_image_router.py>), [scripts/run_trainable_image_router.py](<../../scripts/run_trainable_image_router.py>).

Configuration: [configs/approximate/trainable_image_router_local_paths_v1.json](<../../configs/approximate/trainable_image_router_local_paths_v1.json>), [configs/approximate/trainable_image_router_v1.json](<../../configs/approximate/trainable_image_router_v1.json>).

Tests: [tests/recognition/test_trainable_image_router.py](<../../tests/recognition/test_trainable_image_router.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Frozen-model scale/crop consistency router

ID: `yolo-training-free-consistency-router`. [Guide](<../../docs/YOLO_TRAINING_FREE_CONSISTENCY_ROUTER.md>).

Code: [recognition/training_free_consistency_router.py](<../../recognition/training_free_consistency_router.py>), [scripts/audit_yolo_frozen_consistency.py](<../../scripts/audit_yolo_frozen_consistency.py>), [scripts/run_yolo_frozen_consistency.py](<../../scripts/run_yolo_frozen_consistency.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.
