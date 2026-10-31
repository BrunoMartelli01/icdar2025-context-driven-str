# Word instances, kernels, links, centers, and Gap

[All families](../README.md) · [Code index](../CODE_INDEX.md) · [Saved results](../results/README.md)

Category membership is navigation, not a shared evaluation protocol or a completion claim.

## AdaSpot: Text + Kernel + Gap, first controlled experiment

ID: `adaspot-gap`. [Guide](<../../docs/ADASpot_GAP.md>).

Code: [scripts/run_adaspot_gap_host.sh](<../../scripts/run_adaspot_gap_host.sh>), [segmentation/totaltext_word_regions.py](<../../segmentation/totaltext_word_regions.py>).

Configuration: [configs/approximate/adaspot_gap_v1.json](<../../configs/approximate/adaspot_gap_v1.json>).

Numerical results: [gap](../results/gap.md).

## Gap with local context: matched 1x1 and 3x3 tests

ID: `adaspot-gap-context`. [Guide](<../../docs/ADASpot_GAP_CONTEXT.md>).

Code: [scripts/evaluate_adaspot_gap_context_final.py](<../../scripts/evaluate_adaspot_gap_context_final.py>), [scripts/run_adaspot_gap_context.py](<../../scripts/run_adaspot_gap_context.py>).

Configuration: [configs/approximate/adaspot_gap_context_final_diagnostic_v1.json](<../../configs/approximate/adaspot_gap_context_final_diagnostic_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Trainable Gap decoder — approximate reproduction

ID: `adaspot-gap-decoder`. [Guide](<../../docs/ADASpot_GAP_DECODER.md>).

Code: [models/adaspot_gap_decoder.py](<../../models/adaspot_gap_decoder.py>), [scripts/train_adaspot_gap_decoder.py](<../../scripts/train_adaspot_gap_decoder.py>).

Configuration: [configs/approximate/adaspot_gap_decoder_v1.json](<../../configs/approximate/adaspot_gap_decoder_v1.json>).

Tests: [tests/scripts/test_adaspot_gap_decoder.py](<../../tests/scripts/test_adaspot_gap_decoder.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Full official TRAIN1255 Gap refit — approximate reproduction

ID: `adaspot-gap-fulltrain`. [Guide](<../../docs/ADASpot_GAP_FULLTRAIN.md>).

Code: [scripts/train_adaspot_gap_fulltrain.py](<../../scripts/train_adaspot_gap_fulltrain.py>).

Configuration: [configs/approximate/adaspot_gap_fulltrain_v1.json](<../../configs/approximate/adaspot_gap_fulltrain_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Better Gap training: fit, decision calibration and full Text support

ID: `adaspot-gap-normalized`. [Guide](<../../docs/ADASpot_GAP_NORMALIZED.md>).

Code: [models/adaspot_gap_normalized.py](<../../models/adaspot_gap_normalized.py>), [scripts/train_adaspot_gap_normalized.py](<../../scripts/train_adaspot_gap_normalized.py>).

Configuration: [configs/approximate/adaspot_gap_normalized_v1.json](<../../configs/approximate/adaspot_gap_normalized_v1.json>).

Tests: [tests/scripts/test_adaspot_gap_normalized.py](<../../tests/scripts/test_adaspot_gap_normalized.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Select for actual OCR — approximate reproduction

ID: `adaspot-gap-ocr-selection`. [Guide](<../../docs/ADASpot_GAP_OCR_SELECTION.md>).

Code: [scripts/audit_adaspot_gap_ocr_selection.py](<../../scripts/audit_adaspot_gap_ocr_selection.py>).

Configuration: [configs/approximate/adaspot_gap_ocr_selection_v1.json](<../../configs/approximate/adaspot_gap_ocr_selection_v1.json>).

Tests: [tests/scripts/test_adaspot_gap_ocr_selection.py](<../../tests/scripts/test_adaspot_gap_ocr_selection.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Shared-feature AdaSpot joint prototype

ID: `adaspot-joint`. [Guide](<../../docs/ADASpot_JOINT.md>).

Code: [scripts/audit_adaspot_joint.py](<../../scripts/audit_adaspot_joint.py>), [scripts/check_adaspot_joint_ocr_training.py](<../../scripts/check_adaspot_joint_ocr_training.py>), [scripts/evaluate_adaspot_joint_selective.py](<../../scripts/evaluate_adaspot_joint_selective.py>), [scripts/run_adaspot_joint.py](<../../scripts/run_adaspot_joint.py>).

Configuration: [configs/approximate/adaspot_joint_overfit_check_v1.json](<../../configs/approximate/adaspot_joint_overfit_check_v1.json>), [configs/approximate/adaspot_joint_v1.json](<../../configs/approximate/adaspot_joint_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## AdaSpot Mask-Align v1

ID: `adaspot-mask-align`. [Guide](<../../docs/ADASpot_MASK_ALIGN.md>).

Code: [models/adaspot_mask_align.py](<../../models/adaspot_mask_align.py>), [scripts/audit_adaspot_mask_align.py](<../../scripts/audit_adaspot_mask_align.py>), [scripts/diagnose_adaspot_mask_align.py](<../../scripts/diagnose_adaspot_mask_align.py>).

Configuration: [configs/approximate/adaspot_mask_align_v1.json](<../../configs/approximate/adaspot_mask_align_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## AdaSpot Phase 0: paired crop and decoding diagnostics

ID: `adaspot-phase0`. [Guide](<../../docs/ADASpot_PHASE0.md>).

Code: [scripts/audit_adaspot_phase0.py](<../../scripts/audit_adaspot_phase0.py>), [scripts/diagnose_adaspot_phase0.py](<../../scripts/diagnose_adaspot_phase0.py>).

Configuration: [configs/approximate/adaspot_phase0_v1.json](<../../configs/approximate/adaspot_phase0_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## ICDAR2013 full-word U-Net refit and ICDAR2015 fallback feasibility

ID: `icdar13-word-refit`. [Guide](<../../docs/ICDAR13_WORD_REFIT.md>).

Code: [recognition/fallback_feasibility.py](<../../recognition/fallback_feasibility.py>), [scripts/analyze_icdar_fallback_target.py](<../../scripts/analyze_icdar_fallback_target.py>), [scripts/audit_icdar13_word_targets.py](<../../scripts/audit_icdar13_word_targets.py>), [scripts/finish_icdar13_word_refit.py](<../../scripts/finish_icdar13_word_refit.py>), [scripts/prepare_icdar13_word_refit.py](<../../scripts/prepare_icdar13_word_refit.py>), [scripts/recheck_icdar13_word_refit.py](<../../scripts/recheck_icdar13_word_refit.py>), [scripts/train_icdar13_word_refit.py](<../../scripts/train_icdar13_word_refit.py>), [segmentation/icdar13_word_refit.py](<../../segmentation/icdar13_word_refit.py>).

Configuration: [configs/approximate/icdar13_word_refit_finish_smoke_v1.json](<../../configs/approximate/icdar13_word_refit_finish_smoke_v1.json>), [configs/approximate/icdar13_word_refit_finish_v1.json](<../../configs/approximate/icdar13_word_refit_finish_v1.json>), [configs/approximate/icdar13_word_refit_local_paths_v1.json](<../../configs/approximate/icdar13_word_refit_local_paths_v1.json>), [configs/approximate/icdar13_word_refit_v1.json](<../../configs/approximate/icdar13_word_refit_v1.json>).

Tests: [tests/recognition/test_fallback_feasibility.py](<../../tests/recognition/test_fallback_feasibility.py>), [tests/scripts/test_finish_icdar13_word_refit.py](<../../tests/scripts/test_finish_icdar13_word_refit.py>), [tests/segmentation/test_icdar13_word_refit.py](<../../tests/segmentation/test_icdar13_word_refit.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## New U-Net trained on Kernel GT

ID: `kernel-only-fresh`. [Guide](<../../docs/KERNEL_ONLY_FRESH.md>).

Code: [scripts/train_kernel_only_fresh.py](<../../scripts/train_kernel_only_fresh.py>).

Configuration: [configs/approximate/kernel_only_fresh_v1.json](<../../configs/approximate/kernel_only_fresh_v1.json>).

Numerical results: [kernel-fresh](../results/kernel-fresh.md).

## Extracted Kernel-only output and padding

ID: `kernel-only-padding`. [Guide](<../../docs/KERNEL_ONLY_PADDING.md>).

Code: [scripts/evaluate_kernel_only_padding.py](<../../scripts/evaluate_kernel_only_padding.py>).

Numerical results: [kernel-extracted](../results/kernel-extracted.md).

## Supervised Kernel-only fine-tuning

ID: `kernel-only-training`. [Guide](<../../docs/KERNEL_ONLY_TRAINING.md>).

Code: [scripts/train_kernel_only.py](<../../scripts/train_kernel_only.py>).

Configuration: [configs/approximate/kernel_only_training_v1.json](<../../configs/approximate/kernel_only_training_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Auxiliary recognition supervision of shared U-Net features

ID: `unet-auxiliary-ocr`. [Guide](<../../docs/UNET_AUXILIARY_OCR.md>).

Code: [scripts/run_unet_auxiliary_ocr.py](<../../scripts/run_unet_auxiliary_ocr.py>).

Configuration: [configs/approximate/unet_auxiliary_ocr_v1.json](<../../configs/approximate/unet_auxiliary_ocr_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Learned crop refinement with word-instance links

ID: `unet-crop-links`. [Guide](<../../docs/UNET_CROP_LINKS.md>).

Code: [scripts/run_unet_crop_links_host.sh](<../../scripts/run_unet_crop_links_host.sh>).

Configuration: [configs/approximate/unet_crop_links_v1.json](<../../configs/approximate/unet_crop_links_v1.json>).

Numerical results: [crop-links](../results/crop-links.md).

## Text pretraining, scale fusion, progressive kernels and word rectification

ID: `unet-detection-methods`. [Guide](<../../docs/UNET_DETECTION_METHODS.md>).

Code: [scripts/run_unet_detection_methods_host.sh](<../../scripts/run_unet_detection_methods_host.sh>).

Configuration: [configs/approximate/unet_detection_methods_synthtext_targets_v2.json](<../../configs/approximate/unet_detection_methods_synthtext_targets_v2.json>).

Tests: [tests/localize/test_unet_detection_instances.py](<../../tests/localize/test_unet_detection_instances.py>), [tests/localize/test_word_crop_refinements.py](<../../tests/localize/test_word_crop_refinements.py>), [tests/localize/test_word_guided_contours.py](<../../tests/localize/test_word_guided_contours.py>), [tests/models/test_unet_detection_architecture.py](<../../tests/models/test_unet_detection_architecture.py>), [tests/scripts/test_evaluate_text_recall_refinements.py](<../../tests/scripts/test_evaluate_text_recall_refinements.py>), [tests/scripts/test_synthtext_download.py](<../../tests/scripts/test_synthtext_download.py>), [tests/scripts/test_unet_detection_runner.py](<../../tests/scripts/test_unet_detection_runner.py>), [tests/segmentation/test_totaltext_word_kernel.py](<../../tests/segmentation/test_totaltext_word_kernel.py>), [tests/segmentation/test_totaltext_word_regions.py](<../../tests/segmentation/test_totaltext_word_regions.py>), [tests/segmentation/test_unet_detection_targets.py](<../../tests/segmentation/test_unet_detection_targets.py>), [tests/segmentation/test_word_multitask.py](<../../tests/segmentation/test_word_multitask.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Nucleus-guided links and selective crop refinement

ID: `unet-seeded-selective`. [Guide](<../../docs/UNET_SEEDED_SELECTIVE.md>).

Code: [scripts/run_unet_seeded_selective_host.sh](<../../scripts/run_unet_seeded_selective_host.sh>).

Configuration: [configs/approximate/unet_seeded_selective_v1.json](<../../configs/approximate/unet_seeded_selective_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Text+Kernel and Hybrid384 word detection on TEST

ID: `unet-word-detection-test`. [Guide](<../../docs/UNET_WORD_DETECTION_TEST.md>).

Code: [scripts/test_unet_word_detection.sh](<../../scripts/test_unet_word_detection.sh>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## U-Net: original glyph masks and modified word regions

ID: `unet-word-research`. [Guide](<../../docs/UNET_WORD_RESEARCH.md>).

Code: [scripts/run_unet_word_research_host.sh](<../../scripts/run_unet_word_research_host.sh>).

Configuration: [configs/approximate/unet_word_research_v1.json](<../../configs/approximate/unet_word_research_v1.json>).

Numerical results: [research-v1](../results/research-v1.md), [research-v1-top10](../results/research-v1-top10.md).

## Corrective U-Net experiments v2

ID: `unet-word-research-v2`. [Guide](<../../docs/UNET_WORD_RESEARCH_V2.md>).

Code: [scripts/run_unet_word_research_v2_host.sh](<../../scripts/run_unet_word_research_v2_host.sh>).

Configuration: [configs/approximate/unet_word_research_v2.json](<../../configs/approximate/unet_word_research_v2.json>).

Numerical results: [research-v2-heads](../results/research-v2-heads.md), [research-v2-postprocess](../results/research-v2-postprocess.md).

## Centerline and radius: a new experiment on the frozen U-Net

ID: `word-centerline`. [Guide](<../../docs/WORD_CENTERLINE.md>).

Code: [diagnostics/curve_rectification_train.py](<../../diagnostics/curve_rectification_train.py>), [scripts/run_word_centerline_host.sh](<../../scripts/run_word_centerline_host.sh>).

Configuration: [configs/approximate/word_centerline_v1.json](<../../configs/approximate/word_centerline_v1.json>).

Tests: [tests/localize/test_word_centerline_decode.py](<../../tests/localize/test_word_centerline_decode.py>), [tests/scripts/test_word_centerline_ocr.py](<../../tests/scripts/test_word_centerline_ocr.py>), [tests/scripts/test_word_centerline_runner.py](<../../tests/scripts/test_word_centerline_runner.py>), [tests/segmentation/test_word_centerline.py](<../../tests/segmentation/test_word_centerline.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Auxiliary word-center attraction U-Net

ID: `word-center-offsets`. [Guide](<../../docs/WORD_CENTER_OFFSETS.md>).

Code: [models/unet_adv.py](<../../models/unet_adv.py>), [models/word_multitask.py](<../../models/word_multitask.py>), [scripts/profile_word_center_offsets.py](<../../scripts/profile_word_center_offsets.py>), [scripts/train_word_center_offsets.py](<../../scripts/train_word_center_offsets.py>).

Configuration: [configs/approximate/word_center_offsets_totaltext_train_v1.json](<../../configs/approximate/word_center_offsets_totaltext_train_v1.json>), [configs/approximate/word_center_offsets_v1.json](<../../configs/approximate/word_center_offsets_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Coherent word-center vote loss: isolated v2 experiment

ID: `word-center-vote-loss`. [Guide](<../../docs/WORD_CENTER_VOTE_LOSS.md>).

Code: [scripts/train_word_center_votes.py](<../../scripts/train_word_center_votes.py>).

Configuration: [configs/approximate/word_center_votes_totaltext_train_v2.json](<../../configs/approximate/word_center_votes_totaltext_train_v2.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Word evidence graphs: four controlled factors

ID: `word-evidence-graphs`. [Guide](<../../docs/WORD_EVIDENCE_GRAPHS.md>).

Code: [scripts/run_word_evidence_graphs.py](<../../scripts/run_word_evidence_graphs.py>).

Configuration: [configs/approximate/word_evidence_graphs_v1.json](<../../configs/approximate/word_evidence_graphs_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Text+Kernel: three controlled word-evidence tests

ID: `word-evidence-suite`. [Guide](<../../docs/WORD_EVIDENCE_SUITE.md>).

Code: [scripts/run_word_evidence_suite.py](<../../scripts/run_word_evidence_suite.py>).

Configuration: [configs/approximate/word_evidence_suite_v1.json](<../../configs/approximate/word_evidence_suite_v1.json>).

Tests: [tests/segmentation/test_word_evidence_suite.py](<../../tests/segmentation/test_word_evidence_suite.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Word hypotheses, complete crops, geometric decoders and regional distillation

ID: `word-graph-recovery`. [Guide](<../../docs/WORD_GRAPH_RECOVERY.md>).

Code: [localize/word_graph_recovery.py](<../../localize/word_graph_recovery.py>), [models/word_graph_decoder.py](<../../models/word_graph_decoder.py>), [scripts/qualify_word_graph_decoder.py](<../../scripts/qualify_word_graph_decoder.py>), [scripts/run_word_graph_recovery.py](<../../scripts/run_word_graph_recovery.py>), [scripts/run_word_graph_recovery.sh](<../../scripts/run_word_graph_recovery.sh>), [scripts/word_graph_geometry_stages.py](<../../scripts/word_graph_geometry_stages.py>), [scripts/word_graph_regional_stages.py](<../../scripts/word_graph_regional_stages.py>).

Configuration: [configs/approximate/word_graph_recovery_v1.json](<../../configs/approximate/word_graph_recovery_v1.json>).

Tests: [tests/localize/test_word_graph_recovery.py](<../../tests/localize/test_word_graph_recovery.py>), [tests/models/test_word_graph_decoder.py](<../../tests/models/test_word_graph_decoder.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Local word instance extensions

ID: `word-instance-extensions`. [Guide](<../../docs/WORD_INSTANCE_EXTENSIONS.md>).

Code: [scripts/run_word_instance_extensions.py](<../../scripts/run_word_instance_extensions.py>).

Configuration: [configs/approximate/word_instance_extensions_v1.json](<../../configs/approximate/word_instance_extensions_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Word-kernel and Gaussian-center Attention U-Net experiment

ID: `word-multitask`. [Guide](<../../docs/WORD_MULTITASK.md>).

Code: [localize/crop_recovery.py](<../../localize/crop_recovery.py>), [models/totaltext_word_kernel.py](<../../models/totaltext_word_kernel.py>), [scripts/compare_word_multitask.py](<../../scripts/compare_word_multitask.py>), [scripts/run_word_multitask.py](<../../scripts/run_word_multitask.py>), [scripts/run_word_multitask_host.sh](<../../scripts/run_word_multitask_host.sh>), [segmentation/totaltext_word_kernel.py](<../../segmentation/totaltext_word_kernel.py>).

Configuration: [configs/approximate/word_multitask_v1.json](<../../configs/approximate/word_multitask_v1.json>).

Tests: [tests/data/test_segmentation_transforms.py](<../../tests/data/test_segmentation_transforms.py>), [tests/deepsolo/test_candidates_and_reduction.py](<../../tests/deepsolo/test_candidates_and_reduction.py>), [tests/segmentation/test_word_multitask.py](<../../tests/segmentation/test_word_multitask.py>), [tests/test_main_environment.py](<../../tests/test_main_environment.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Completed word multi-task experiment v1

ID: `word-multitask-results-v1`. [Guide](<../../docs/WORD_MULTITASK_RESULTS_V1.md>).

Code: [models/word_multitask.py](<../../models/word_multitask.py>), [scripts/compare_word_multitask.py](<../../scripts/compare_word_multitask.py>), [scripts/run_word_multitask.py](<../../scripts/run_word_multitask.py>).

Configuration: [configs/approximate/word_multitask_v1.json](<../../configs/approximate/word_multitask_v1.json>).

Tests: [tests/segmentation/test_word_multitask.py](<../../tests/segmentation/test_word_multitask.py>), [tests/segmentation/test_word_multitask_provenance.py](<../../tests/segmentation/test_word_multitask_provenance.py>).

Numerical results: [multitask-geometry](../results/multitask-geometry.md).
