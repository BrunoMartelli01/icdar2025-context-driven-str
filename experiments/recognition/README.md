# Recognition, language correction, and candidate verification

[All families](../README.md) · [Code index](../CODE_INDEX.md) · [Saved results](../results/README.md)

Category membership is navigation, not a shared evaluation protocol or a completion claim.

## CLIP and PARSeq on the high-recall pool

ID: `clip-high-recall-admission`. [Guide](<../../docs/CLIP_HIGH_RECALL_ADMISSION.md>).

Code: [scripts/run_clip_high_recall_admission_host.sh](<../../scripts/run_clip_high_recall_admission_host.sh>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## New CLIP uses and GPU comparison command

ID: `clip-innovative-verification`. [Guide](<../../docs/CLIP_INNOVATIVE_VERIFICATION.md>).

Code: [scripts/run_clip_verification_suite_host.sh](<../../scripts/run_clip_verification_suite_host.sh>).

Configuration: [configs/approximate/clip_innovative_verification_train_v2.json](<../../configs/approximate/clip_innovative_verification_train_v2.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Completed CLIP comparison: B/32, B/16, and L/14

ID: `clip-verification-results`. [Guide](<../../docs/CLIP_VERIFICATION_RESULTS.md>).

Configuration: [configs/approximate/clip_innovative_verification_train_v2.json](<../../configs/approximate/clip_innovative_verification_train_v2.json>), [configs/approximate/clip_word_verification_train_v1.json](<../../configs/approximate/clip_word_verification_train_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## CLIP verification of PARSeq transcriptions

ID: `clip-word-verification`. [Guide](<../../docs/CLIP_WORD_VERIFICATION.md>).

Code: [recognition/crop_clip_battery.py](<../../recognition/crop_clip_battery.py>), [recognition/group_word_selection_runtime.py](<../../recognition/group_word_selection_runtime.py>), [scripts/crop_clip_runtime.py](<../../scripts/crop_clip_runtime.py>), [scripts/run_clip_word_verification.py](<../../scripts/run_clip_word_verification.py>).

Configuration: [configs/approximate/clip_word_verification_local_paths_v1.json](<../../configs/approximate/clip_word_verification_local_paths_v1.json>), [configs/approximate/clip_word_verification_train_v1.json](<../../configs/approximate/clip_word_verification_train_v1.json>).

Tests: [tests/recognition/test_clip_word_verifier.py](<../../tests/recognition/test_clip_word_verifier.py>), [tests/recognition/test_group_word_selection.py](<../../tests/recognition/test_group_word_selection.py>), [tests/recognition/test_totaltext_crop_clip_battery.py](<../../tests/recognition/test_totaltext_crop_clip_battery.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Contextual selection of OCR word boundaries

ID: `contextual-boundary-selection`. [Guide](<../../docs/CONTEXTUAL_BOUNDARY_SELECTION.md>).

Code: [recognition/contextual_boundary_selection.py](<../../recognition/contextual_boundary_selection.py>), [scripts/analyze_contextual_boundary_selection.py](<../../scripts/analyze_contextual_boundary_selection.py>), [scripts/audit_contextual_boundary_selection.py](<../../scripts/audit_contextual_boundary_selection.py>), [scripts/calibrate_contextual_boundary_sensitivity.py](<../../scripts/calibrate_contextual_boundary_sensitivity.py>), [scripts/run_contextual_boundary_selection.py](<../../scripts/run_contextual_boundary_selection.py>).

Configuration: [configs/approximate/contextual_boundary_selection_v1.json](<../../configs/approximate/contextual_boundary_selection_v1.json>), [configs/approximate/contextual_boundary_threshold_sensitivity_v1.json](<../../configs/approximate/contextual_boundary_threshold_sensitivity_v1.json>).

Tests: [tests/recognition/test_contextual_boundary_selection.py](<../../tests/recognition/test_contextual_boundary_selection.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Contextual OCR correction: frozen TRAIN comparison v1

ID: `contextual-ocr-correction-train-v1`. [Guide](<../../docs/CONTEXTUAL_OCR_CORRECTION_TRAIN_V1.md>).

Code: [diagnostics/run_contextual_ocr_correction_host.sh](<../../diagnostics/run_contextual_ocr_correction_host.sh>).

Configuration: [configs/approximate/contextual_ocr_correction_train_v1.json](<../../configs/approximate/contextual_ocr_correction_train_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Image-held-out TRAIN few-shot OCR correction

ID: `contextual-ocr-fewshot-train-v1`. [Guide](<../../docs/CONTEXTUAL_OCR_FEWSHOT_TRAIN_V1.md>).

Code: [diagnostics/run_contextual_ocr_fewshot_train_host.sh](<../../diagnostics/run_contextual_ocr_fewshot_train_host.sh>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Full TRAIN plain-text OCR correction diagnostic

ID: `contextual-ocr-plaintext-train-v2`. [Guide](<../../docs/CONTEXTUAL_OCR_PLAINTEXT_TRAIN_V2.md>).

Code: [diagnostics/run_contextual_ocr_plaintext_train_host.sh](<../../diagnostics/run_contextual_ocr_plaintext_train_host.sh>).

Tests: [tests/recognition/test_contextual_ocr_plaintext_train.py](<../../tests/recognition/test_contextual_ocr_plaintext_train.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Image-wide OCR correction battery

ID: `image-context-correction`. [Guide](<../../docs/IMAGE_CONTEXT_CORRECTION.md>).

Code: [recognition/image_context_correction.py](<../../recognition/image_context_correction.py>), [recognition/image_context_joint_policy.py](<../../recognition/image_context_joint_policy.py>), [scripts/analyze_image_context_correction.py](<../../scripts/analyze_image_context_correction.py>), [scripts/audit_image_context_correction.py](<../../scripts/audit_image_context_correction.py>), [scripts/diagnose_image_context_failures.py](<../../scripts/diagnose_image_context_failures.py>), [scripts/reuse_image_context_generators.py](<../../scripts/reuse_image_context_generators.py>), [scripts/run_image_context_correction.py](<../../scripts/run_image_context_correction.py>), [scripts/run_image_context_joint_policy.py](<../../scripts/run_image_context_joint_policy.py>), [scripts/run_legacy_global_context.py](<../../scripts/run_legacy_global_context.py>).

Configuration: [configs/approximate/image_context_correction_v1.json](<../../configs/approximate/image_context_correction_v1.json>), [configs/approximate/image_context_joint_policy_v1.json](<../../configs/approximate/image_context_joint_policy_v1.json>), [configs/approximate/image_context_legacy_features_v1.json](<../../configs/approximate/image_context_legacy_features_v1.json>).

Tests: [tests/recognition/test_image_context_correction.py](<../../tests/recognition/test_image_context_correction.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Image-wide OCR correction: completed results

ID: `image-context-correction-results`. [Guide](<../../docs/IMAGE_CONTEXT_CORRECTION_RESULTS.md>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## OCR evidence generation battery — frozen TRAIN exploration

ID: `ocr-evidence-generation-battery-train-v1`. [Guide](<../../docs/OCR_EVIDENCE_GENERATION_BATTERY_TRAIN_V1.md>).

Code: [diagnostics/run_ocr_evidence_battery_host.sh](<../../diagnostics/run_ocr_evidence_battery_host.sh>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Frozen prompt comparison, TRAIN229

ID: `ocr-prompt-comparison-train-v1`. [Guide](<../../docs/OCR_PROMPT_COMPARISON_TRAIN_V1.md>).

Code: [diagnostics/run_ocr_prompt_comparison_train_host.sh](<../../diagnostics/run_ocr_prompt_comparison_train_host.sh>).

Configuration: [configs/approximate/ocr_prompt_comparison_train_v1.json](<../../configs/approximate/ocr_prompt_comparison_train_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## OCR battery: damage audit and 80 target-bound diagnostics

ID: `ocr-target-bound-battery-train-v2`. [Guide](<../../docs/OCR_TARGET_BOUND_BATTERY_TRAIN_V2.md>).

Code: [diagnostics/run_ocr_target_bound_battery_host.sh](<../../diagnostics/run_ocr_target_bound_battery_host.sh>).

Configuration: [configs/approximate/ocr_target_bound_battery_train_v2.json](<../../configs/approximate/ocr_target_bound_battery_train_v2.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## PARSeq visual-margin contextual gate — TRAIN offline diagnostic

ID: `parseq-visual-margin-context-gate-train-v1`. [Guide](<../../docs/PARSEQ_VISUAL_MARGIN_CONTEXT_GATE_TRAIN_V1.md>).

Code: [diagnostics/parseq_visual_margin_context_gate_train.py](<../../diagnostics/parseq_visual_margin_context_gate_train.py>).

Configuration: [configs/approximate/parseq_visual_margin_context_gate_train_v1.json](<../../configs/approximate/parseq_visual_margin_context_gate_train_v1.json>).

Tests: [tests/recognition/test_parseq_visual_margin_context_gate_train.py](<../../tests/recognition/test_parseq_visual_margin_context_gate_train.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Selective word repair and false-positive control

ID: `selective-word-repair-tests`. [Guide](<../../docs/SELECTIVE_WORD_REPAIR_TESTS.md>).

Code: [recognition/selective_word_repair.py](<../../recognition/selective_word_repair.py>), [scripts/analyze_selective_word_repair_tests.py](<../../scripts/analyze_selective_word_repair_tests.py>), [scripts/audit_selective_word_repair_tests.py](<../../scripts/audit_selective_word_repair_tests.py>), [scripts/run_selective_word_repair_tests.py](<../../scripts/run_selective_word_repair_tests.py>).

Configuration: [configs/approximate/selective_word_repair_tests_v1.json](<../../configs/approximate/selective_word_repair_tests_v1.json>).

Tests: [tests/recognition/test_selective_word_repair.py](<../../tests/recognition/test_selective_word_repair.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Frozen OCR: completion, structural selection, and byte denoising

ID: `text-reconstruction-tests`. [Guide](<../../docs/TEXT_RECONSTRUCTION_TESTS.md>).

Code: [recognition/text_reconstruction.py](<../../recognition/text_reconstruction.py>), [scripts/analyze_text_reconstruction_tests.py](<../../scripts/analyze_text_reconstruction_tests.py>), [scripts/audit_text_reconstruction_tests.py](<../../scripts/audit_text_reconstruction_tests.py>), [scripts/replay_qwen_insertion_controls.py](<../../scripts/replay_qwen_insertion_controls.py>), [scripts/run_text_reconstruction_tests.py](<../../scripts/run_text_reconstruction_tests.py>).

Configuration: [configs/approximate/qwen_insertion_replay_v1.json](<../../configs/approximate/qwen_insertion_replay_v1.json>), [configs/approximate/text_reconstruction_tests_v1.json](<../../configs/approximate/text_reconstruction_tests_v1.json>).

Tests: [tests/recognition/test_text_reconstruction.py](<../../tests/recognition/test_text_reconstruction.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## BERT after PARSeq: Total-Text letterbox

ID: `totaltext-bert-correction`. [Guide](<../../docs/TOTALTEXT_BERT_CORRECTION.md>).

Code: [models/unet_adv.py](<../../models/unet_adv.py>), [scripts/run_totaltext_bert_correction_host.sh](<../../scripts/run_totaltext_bert_correction_host.sh>).

Configuration: [configs/approximate/totaltext_v4_bert_confidence_ablation_v1.json](<../../configs/approximate/totaltext_v4_bert_confidence_ablation_v1.json>), [configs/approximate/totaltext_v4_bert_correction_v1.json](<../../configs/approximate/totaltext_v4_bert_correction_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## BERT selection of existing Total-Text crops

ID: `totaltext-bert-crop-selection-v1`. [Guide](<../../docs/TOTALTEXT_BERT_CROP_SELECTION_V1.md>).

Code: [recognition/bert_crop_selection.py](<../../recognition/bert_crop_selection.py>), [scripts/run_totaltext_bert_crop_selection_host.sh](<../../scripts/run_totaltext_bert_crop_selection_host.sh>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Total-Text BERT methods with local OCR and Florence scene context

ID: `totaltext-bert-methods-v1`. [Guide](<../../docs/TOTALTEXT_BERT_METHODS_V1.md>).

Code: [scripts/evaluate_totaltext_bert_methods.py](<../../scripts/evaluate_totaltext_bert_methods.py>), [scripts/run_totaltext_bert_methods_host.sh](<../../scripts/run_totaltext_bert_methods_host.sh>).

Configuration: [configs/approximate/totaltext_bert_methods_v1.json](<../../configs/approximate/totaltext_bert_methods_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Total-Text BERT visual correction — TRAIN development

ID: `totaltext-bert-visual-correction-v1`. [Guide](<../../docs/TOTALTEXT_BERT_VISUAL_CORRECTION_V1.md>).

Code: [scripts/evaluate_totaltext_bert_padded_evidence.py](<../../scripts/evaluate_totaltext_bert_padded_evidence.py>), [scripts/run_totaltext_bert_padded_evidence_host.sh](<../../scripts/run_totaltext_bert_padded_evidence_host.sh>), [scripts/run_totaltext_bert_visual_correction_host.sh](<../../scripts/run_totaltext_bert_visual_correction_host.sh>).

Configuration: [configs/approximate/totaltext_bert_padded_evidence_v1.json](<../../configs/approximate/totaltext_bert_padded_evidence_v1.json>), [configs/approximate/totaltext_bert_visual_correction_v1.json](<../../configs/approximate/totaltext_bert_visual_correction_v1.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Total-Text: frozen letterbox/PARSeq with spelling-trained BERT

ID: `totaltext-neuspell-correction`. [Guide](<../../docs/TOTALTEXT_NEUSPELL_CORRECTION.md>).

Code: [scripts/run_totaltext_neuspell_correction_host.sh](<../../scripts/run_totaltext_neuspell_correction_host.sh>).

Configuration: [configs/approximate/totaltext_letterbox_neuspell_correction_v2.json](<../../configs/approximate/totaltext_letterbox_neuspell_correction_v2.json>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.
