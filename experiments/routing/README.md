# Context, captioning, CLIP and YOLO routing

[All families](../README.md) · [Code index](../CODE_INDEX.md) · [Saved results](../results/README.md)

Category membership is navigation, not a shared evaluation protocol or a completion claim.

## Frozen InstructBLIP paired routing: TRAIN and TEST

ID: `budgeted-paired-e2e`. [Guide](<../../docs/BUDGETED_PAIRED_E2E.md>).

Code: [diagnostics/run_budgeted_paired_e2e_host.sh](<../../diagnostics/run_budgeted_paired_e2e_host.sh>).

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

## Paper image router baseline — approximate reproduction

ID: `paper-image-router-baseline`. [Guide](<../../docs/PAPER_IMAGE_ROUTER_BASELINE.md>).

Code: [scripts/run_paper_image_router_baseline.py](<../../scripts/run_paper_image_router_baseline.py>).

Configuration: [configs/approximate/paper_image_router_baseline_local_paths_v1.json](<../../configs/approximate/paper_image_router_baseline_local_paths_v1.json>), [configs/approximate/paper_image_router_baseline_v1.json](<../../configs/approximate/paper_image_router_baseline_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Accuracy gaps and conditional U-Net requirements

ID: `router-accuracy-targets`. [Guide](<../../docs/ROUTER_ACCURACY_TARGETS.md>).

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

## YOLO: paired CLIP / paper-router comparison

ID: `yolo-paired-router-comparison`. [Guide](<../../docs/YOLO_PAIRED_ROUTER_COMPARISON.md>).

Code: [scripts/audit_yolo_router_comparison.py](<../../scripts/audit_yolo_router_comparison.py>), [scripts/run_yolo_router_comparison.py](<../../scripts/run_yolo_router_comparison.py>).

Configuration: [configs/approximate/yolo_router_comparison_local_paths_v1.json](<../../configs/approximate/yolo_router_comparison_local_paths_v1.json>), [configs/approximate/yolo_router_comparison_v1.json](<../../configs/approximate/yolo_router_comparison_v1.json>).

Numerical results: [yolo-test-equal](../results/yolo-test-equal.md), [yolo-test-transferred](../results/yolo-test-transferred.md), [yolo-train-equal](../results/yolo-train-equal.md).

## YOLO/PARSeq: four TEST datasets and per-image compute estimates

ID: `yolo-parseq-four-dataset-comparison`. [Guide](<../../docs/YOLO_PARSEQ_FOUR_DATASET_COMPARISON.md>).

Code: [scripts/test_yolo_four_datasets.sh](<../../scripts/test_yolo_four_datasets.sh>).

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

## Frozen-model scale/crop consistency router

ID: `yolo-training-free-consistency-router`. [Guide](<../../docs/YOLO_TRAINING_FREE_CONSISTENCY_ROUTER.md>).

Code: [recognition/training_free_consistency_router.py](<../../recognition/training_free_consistency_router.py>), [scripts/audit_yolo_frozen_consistency.py](<../../scripts/audit_yolo_frozen_consistency.py>), [scripts/run_yolo_frozen_consistency.py](<../../scripts/run_yolo_frozen_consistency.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.
