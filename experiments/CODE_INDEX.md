# Code index

Modules grouped by package. Summaries come from source docstrings; no module is imported.

[Experiment families](README.md) · [Repository responsibilities](../docs/REPOSITORY_MAP.md)

## caption

| Source | Purpose | Experiment guides |
| --- | --- | --- |
| [caption/__init__.py](../caption/__init__.py) | Public caption package API. | See source/tests |
| [caption/blip2_captioner.py](../caption/blip2_captioner.py) | Pinned APPROXIMATE BLIP-2 integration and auditable inference metadata. | See source/tests |
| [caption/cache.py](../caption/cache.py) | Content-addressed caption cache with all scientifically relevant key inputs. | See source/tests |
| [caption/detailed_scene.py](../caption/detailed_scene.py) | Explicit approximate long scene captions with raw native answers preserved. | detailed-scene-captions |
| [caption/florence_scene.py](../caption/florence_scene.py) | Pinned local Florence-2 native detailed-caption ablation; no OCR task. | detailed-scene-captions |
| [caption/init.py](../caption/init.py) | Public API for `caption` package. | See source/tests |
| [caption/paper_batch_blip2.py](../caption/paper_batch_blip2.py) | Native batched BLIP-2 decoding with explicit shared-batch memory accounting. | paper-caption-router-comparison |
| [caption/paper_style_blip2.py](../caption/paper_style_blip2.py) | Local-only BLIP-2 caption profiles motivated by paper Figure 4. | See source/tests |
| [caption/qwen_batch_scene.py](../caption/qwen_batch_scene.py) | Native Qwen caption batching with explicit padding and shared timing. | See source/tests |
| [caption/qwen_scene.py](../caption/qwen_scene.py) | Native pinned Qwen2-VL caption-quality ablation, separate from paper BLIP-2. | detailed-scene-captions |

## cli

| Source | Purpose | Experiment guides |
| --- | --- | --- |
| [cli/__init__.py](../cli/__init__.py) | Command-line interfaces for the repository. | See source/tests |
| [cli/main.py](../cli/main.py) | Command-line entrypoint for the training-free context-driven STR pipeline. | See source/tests |

## deepsolo

| Source | Purpose | Experiment guides |
| --- | --- | --- |
| [deepsolo/__init__.py](../deepsolo/__init__.py) | Versioned process boundary for the isolated APPROXIMATE DeepSolo runtime. | See source/tests |
| [deepsolo/adapter.py](../deepsolo/adapter.py) | Main-environment adapter that invokes DeepSolo only through a subprocess. | See source/tests |
| [deepsolo/candidates.py](../deepsolo/candidates.py) | Provenance-defined DeepSolo candidate identities; none is paper-selected. | See source/tests |
| [deepsolo/contracts.py](../deepsolo/contracts.py) | Dependency-free JSON contract shared by the main and DeepSolo environments. | See source/tests |
| [deepsolo/reduction.py](../deepsolo/reduction.py) | Explicit ASSUMPTION policies for DeepSolo's unresolved multi-output mapping. | See source/tests |
| [deepsolo/worker.py](../deepsolo/worker.py) | DeepSolo-environment worker. This module is kept parseable by Python 3.8. | deepsolo-icdar2015-gap |

## diagnostics

| Source | Purpose | Experiment guides |
| --- | --- | --- |
| [diagnostics/__init__.py](../diagnostics/__init__.py) | Isolated replay diagnostics; frozen pipeline implementations are unchanged. | See source/tests |
| [diagnostics/bilinear_mask_backprojection_train.py](../diagnostics/bilinear_mask_backprojection_train.py) | Only change binary native-mask resizing; CPU TRAIN geometry, frozen methods. | See source/tests |
| [diagnostics/bilinear_threshold_train.py](../diagnostics/bilinear_threshold_train.py) | CPU TRAIN-only post-bilinear binary-mask threshold diagnostic, no models. | See source/tests |
| [diagnostics/budgeted_paired_e2e.py](../diagnostics/budgeted_paired_e2e.py) | Frozen InstructBLIP paired routing: CPU preparation/evaluation, explicit PAPER export. | See source/tests |
| [diagnostics/budgeted_paired_train.py](../diagnostics/budgeted_paired_train.py) | Apply one GT-free, image-budget threshold to frozen InstructBLIP TRAIN scores. | See source/tests |
| [diagnostics/caption_routing_train.py](../diagnostics/caption_routing_train.py) | Persistent TRAIN caption/routing comparison; staged GPU export, CPU replay. | See source/tests |
| [diagnostics/compare_repository_localizer.py](../diagnostics/compare_repository_localizer.py) | CPU-only saved-mask comparison; no model inference or GT-dependent selection. | repository-contours-variant |
| [diagnostics/conservative_ocr_edit_gates_train.py](../diagnostics/conservative_ocr_edit_gates_train.py) | CPU-only offline edit gates; preserve every frozen experiment and model. | See source/tests |
| [diagnostics/contextual_ocr_conservative_crop_train.py](../diagnostics/contextual_ocr_conservative_crop_train.py) | TRAIN-only image-held-out few-shot OCR correction. No gradient updates. | See source/tests |
| [diagnostics/contextual_ocr_correction.py](../diagnostics/contextual_ocr_correction.py) | TRAIN-only frozen text correction study. GPU inference is an explicit stage. | See source/tests |
| [diagnostics/contextual_ocr_fewshot_train.py](../diagnostics/contextual_ocr_fewshot_train.py) | TRAIN-only image-held-out few-shot OCR correction. No gradient updates. | See source/tests |
| [diagnostics/contextual_ocr_plaintext_train.py](../diagnostics/contextual_ocr_plaintext_train.py) | Frozen TRAIN229 per-target plain-text diagnostic. No segmentation/OCR/caption inference. | See source/tests |
| [diagnostics/curve_rectification_train.py](../diagnostics/curve_rectification_train.py) | Separate fixed-group shape diagnostic. prepare is CPU; infer is manual PAPER only. | word-centerline |
| [diagnostics/directional_neighbors_train.py](../diagnostics/directional_neighbors_train.py) | CPU-only TRAIN audit: word orientation and frozen directional neighbours. | See source/tests |
| [diagnostics/export_grouping_probabilities.py](../diagnostics/export_grouping_probabilities.py) | PAPER-only CUDA export; resumable persistent probabilities, exact frozen checks. | See source/tests |
| [diagnostics/export_scene_text_selected_visuals.py](../diagnostics/export_scene_text_selected_visuals.py) | Export compact visual audits from frozen scene-text V1/PARSeq results. | See source/tests |
| [diagnostics/export_train_visual_bundle.py](../diagnostics/export_train_visual_bundle.py) | Export a portable, read-only visual audit of frozen TRAIN artifacts. | See source/tests |
| [diagnostics/finalize_lavis_captioner_review.py](../diagnostics/finalize_lavis_captioner_review.py) | Summarize frozen caption outputs and explicit five-image visual observations. | See source/tests |
| [diagnostics/fixed_group_curve_geometry.py](../diagnostics/fixed_group_curve_geometry.py) | GT-free rectification of a FIXED component group, not a word detector. | See source/tests |
| [diagnostics/fragment_reconstruction_train.py](../diagnostics/fragment_reconstruction_train.py) | Static TRAIN demonstrations; full-image reconstruction and GT-box word oracle. | See source/tests |
| [diagnostics/fragment_reconstruction_train_v2.py](../diagnostics/fragment_reconstruction_train_v2.py) | Static TRAIN demonstrations; full-image reconstruction and GT-box word oracle. | See source/tests |
| [diagnostics/fragment_reconstruction_train_v3.py](../diagnostics/fragment_reconstruction_train_v3.py) | Static TRAIN demonstrations; full-image reconstruction and GT-box word oracle. | See source/tests |
| [diagnostics/fragment_reconstruction_train_v4.py](../diagnostics/fragment_reconstruction_train_v4.py) | Static TRAIN demonstrations; full-image reconstruction and GT-box word oracle. | See source/tests |
| [diagnostics/geometry_context_reconstruction_train.py](../diagnostics/geometry_context_reconstruction_train.py) | Frozen, non-oracle ALL_WORDS geometry/caption ablation. GPU is manual only. | See source/tests |
| [diagnostics/geometry_structural_oracle_train.py](../diagnostics/geometry_structural_oracle_train.py) | DIAGNOSTIC_ONLY_ORACLE: structural upper bound on frozen TRAIN229 V1 OCR crops. | See source/tests |
| [diagnostics/group_coherence_probability_train.py](../diagnostics/group_coherence_probability_train.py) | CPU-only TRAIN geometry evaluation; probabilities must be verified cached data. | See source/tests |
| [diagnostics/gt_mask_localization_oracle.py](../diagnostics/gt_mask_localization_oracle.py) | TRAIN-only CPU geometry counterfactual; never instantiate a learned model. | See source/tests |
| [diagnostics/gt_word_group_selector.py](../diagnostics/gt_word_group_selector.py) | Measure crop choice with one oracle-localized group per annotated word. | See source/tests |
| [diagnostics/joined_t3.py](../diagnostics/joined_t3.py) | CPU replay of T1/T2/T3 routing with all retained crop texts joined into T3. | See source/tests |
| [diagnostics/lavis_captioner_candidates.py](../diagnostics/lavis_captioner_candidates.py) | Five-image caption diagnostics; isolated from all scientific pipeline defaults. | See source/tests |
| [diagnostics/ocr_context_evidence_next_train.py](../diagnostics/ocr_context_evidence_next_train.py) | Isolated CPU context controls and small user-run Qwen edit-interface pilot. | See source/tests |
| [diagnostics/ocr_evidence_generation_battery_train.py](../diagnostics/ocr_evidence_generation_battery_train.py) | Frozen exploratory TRAIN generation battery. CPU prepare/evaluate; manual GPU only. | See source/tests |
| [diagnostics/ocr_prompt_comparison_train.py](../diagnostics/ocr_prompt_comparison_train.py) | Predeclared TRAIN prompt comparison; no changes to models or crop grouping. | See source/tests |
| [diagnostics/ocr_target_bound_battery_train.py](../diagnostics/ocr_target_bound_battery_train.py) | 80 predeclared TRAIN diagnostics; CPU preparation/evaluation, user-run GPU only. | See source/tests |
| [diagnostics/parseq_candidate_coverage_train.py](../diagnostics/parseq_candidate_coverage_train.py) | TRAIN-only PARSeq candidate-coverage diagnostic. | See source/tests |
| [diagnostics/parseq_context_reranker_train.py](../diagnostics/parseq_context_reranker_train.py) | Training-free PARSeq N-best contextual reranking diagnostic. | See source/tests |
| [diagnostics/parseq_mask_region_audit.py](../diagnostics/parseq_mask_region_audit.py) | CPU-only, post-hoc TEST audit: native mask quality versus frozen v1 OCR errors. | See source/tests |
| [diagnostics/parseq_visual_margin_context_gate_train.py](../diagnostics/parseq_visual_margin_context_gate_train.py) | Offline TRAIN margin-gated reranking; no model imports or inference. | parseq-visual-margin-context-gate-train-v1 |
| [diagnostics/project_storage_cleanup.py](../diagnostics/project_storage_cleanup.py) | Audited removal of regenerable project caches; no model/results deletion. | See source/tests |
| [diagnostics/qwen_structural_caption_ablation_train.py](../diagnostics/qwen_structural_caption_ablation_train.py) | No module docstring; see linked source and tests | See source/tests |
| [diagnostics/qwen_structural_selector_train.py](../diagnostics/qwen_structural_selector_train.py) | No module docstring; see linked source and tests | See source/tests |
| [diagnostics/recognize_recovered_train_crops.py](../diagnostics/recognize_recovered_train_crops.py) | PARSeq-only diagnostic on frozen80 TRAIN geometric recoveries and GT_BOX controls. | See source/tests |
| [diagnostics/render_train_visual_sheets.py](../diagnostics/render_train_visual_sheets.py) | Render one self-contained visual sheet per frozen ICDAR13 TRAIN image. | See source/tests |
| [diagnostics/repair_decord_wheel.py](../diagnostics/repair_decord_wheel.py) | Repair the upstream Linux wheel's internal tag; preserve all runtime bytes. | See source/tests |
| [diagnostics/repository_contours_native_train.py](../diagnostics/repository_contours_native_train.py) | CPU native128 geometry from verified TRAIN cache; optional PAPER-only OCR. | repository-contours-variant |
| [diagnostics/repository_contours_train.py](../diagnostics/repository_contours_train.py) | Frozen TRAIN comparison of current Algorithm1, upstream contours and v1. | repository-contours-variant |
| [diagnostics/routing_threshold_train.py](../diagnostics/routing_threshold_train.py) | Post-hoc TRAIN-only C separability audit, without changing routing or models. | See source/tests |
| [diagnostics/show_joined_t3.py](../diagnostics/show_joined_t3.py) | Read-only, standard-library viewer for the joined-T3 replay report. | See source/tests |
| [diagnostics/train_crop_completeness.py](../diagnostics/train_crop_completeness.py) | CPU TRAIN-only post-hoc crop completeness/contamination; no new inference. | See source/tests |
| [diagnostics/train_v1_crop_glyph_audit.py](../diagnostics/train_v1_crop_glyph_audit.py) | Post-hoc TRAIN crop clipping, using original GT glyph colors; no inference. | See source/tests |
| [diagnostics/train_v1_ocr_failure_audit.py](../diagnostics/train_v1_ocr_failure_audit.py) | CPU-only post-hoc TRAIN V1 OCR/mask/geometry failure audit; no inference. | See source/tests |
| [diagnostics/two_track_common.py](../diagnostics/two_track_common.py) | Persistent provenance and fixed-spatial evaluation shared by isolated diagnostics. | See source/tests |

## localize

| Source | Purpose | Experiment guides |
| --- | --- | --- |
| [localize/__init__.py](../localize/__init__.py) | Public localization package API. | See source/tests |
| [localize/adaspot_gap.py](../localize/adaspot_gap.py) | GT-blind kernel BFS with an explicit, auditable predicted-gap barrier. | See source/tests |
| [localize/adaspot_gap_preserve.py](../localize/adaspot_gap_preserve.py) | Explicit experimental Gap partition with optional full Text support recovery. | See source/tests |
| [localize/appearance_remap_config.py](../localize/appearance_remap_config.py) | Explicit approximate-only configuration for the appearance remap prototype. | See source/tests |
| [localize/appearance_word_remap.py](../localize/appearance_word_remap.py) | CPU-only, one-hop appearance rescue of immutable practical-remap-v1 groups. | See source/tests |
| [localize/center_guided_word_crops.py](../localize/center_guided_word_crops.py) | Opt-in approximate shape/center fusion; no annotations enter this module. | center-guided-word-crops |
| [localize/classic_word_regions.py](../localize/classic_word_regions.py) | Classical mask-to-word diagnostics kept separate from all pipeline modes. | See source/tests |
| [localize/coherent_probability_groups.py](../localize/coherent_probability_groups.py) | Two independent mask-only grouping diagnostics; no GT/OCR/model inputs. | See source/tests |
| [localize/contiguous_span_candidates.py](../localize/contiguous_span_candidates.py) | Inference-safe contiguous raw-component span enumeration. | See source/tests |
| [localize/crop_recovery.py](../localize/crop_recovery.py) | GT-blind crop hypotheses from probability topology and RGB evidence. | word-multitask |
| [localize/crop_word_groups.py](../localize/crop_word_groups.py) | Group original crop hypotheses without consuming OCR or annotations. | See source/tests |
| [localize/curved_component_crop.py](../localize/curved_component_crop.py) | Isolate a native-mask component's curved RGB support on a white canvas. | See source/tests |
| [localize/directional_affinity_split.py](../localize/directional_affinity_split.py) | Conservative graph splitting from learned ownership, preserving every pixel. | directional-transport-results |
| [localize/directional_closedness.py](../localize/directional_closedness.py) | Image-only word completeness readout from learned ownership affinities. | directional-transport-results |
| [localize/directional_component_neighbors.py](../localize/directional_component_neighbors.py) | Opt-in geometric TRAIN diagnostic; frozen v1 and Algorithm1 stay unchanged. | See source/tests |
| [localize/external_text_detectors.py](../localize/external_text_detectors.py) | Detector-only adapters; pretrained models never substitute for PARSeq decoding. | See source/tests |
| [localize/gap_word_regions.py](../localize/gap_word_regions.py) | TRAIN-diagnostic classical gap grouping on native binary masks. | See source/tests |
| [localize/graph_occlusion_followups.py](../localize/graph_occlusion_followups.py) | Isolated APPROXIMATE fragmentation augmentation and contour-point GCN. | graph-occlusion-followups |
| [localize/graph_occlusion_research.py](../localize/graph_occlusion_research.py) | Exploratory component graphs and clean-to-occluded instance distillation. | See source/tests |
| [localize/grouped_word_blocks.py](../localize/grouped_word_blocks.py) | Practical graph-based remapping of glyph-like segmentation components. | experiment-summary-v1-routing-llms |
| [localize/independent_word_slots.py](../localize/independent_word_slots.py) | Approximate, GT-blind assignment of frozen OCR crops to independent text slots. | See source/tests |
| [localize/line_word_blocks.py](../localize/line_word_blocks.py) | Two-stage geometric word grouping for ``practical-remap-v2``. | See source/tests |
| [localize/mask_crop_hypotheses.py](../localize/mask_crop_hypotheses.py) | APPROXIMATE local split/merge hypotheses for stretch-resized native128 masks. | See source/tests |
| [localize/mask_to_blocks.py](../localize/mask_to_blocks.py) | Paper-specified mechanics of Algorithm 1 with explicit approximate parameters. | repository-contours-variant, reproduction-plan |
| [localize/multihypothesis_word_proposals.py](../localize/multihypothesis_word_proposals.py) | Inference-safe multi-hypothesis word proposals from native probabilities. | See source/tests |
| [localize/multiscale_component_persistence.py](../localize/multiscale_component_persistence.py) | TRAIN-diagnostic multiscale persistence grouping of native-mask components. | See source/tests |
| [localize/native_algorithm1.py](../localize/native_algorithm1.py) | Algorithm 1 localization in the native 128x128 segmentation frame. | See source/tests |
| [localize/native_repository_contours.py](../localize/native_repository_contours.py) | Opt-in native128 coordinate-frame variant of frozen repository contours. | repository-contours-variant |
| [localize/nn_word_grouping.py](../localize/nn_word_grouping.py) | Train-diagnostic nearest-neighbour word grouping from raw components only. | See source/tests |
| [localize/occlusion_four_tests.py](../localize/occlusion_four_tests.py) | GT-blind APPROXIMATE proposals and policies for the four occlusion tests. | occlusion-four-tests |
| [localize/practical_candidate_selection.py](../localize/practical_candidate_selection.py) | Training-free selection of coherent word proposals from frozen hypotheses. | See source/tests |
| [localize/practical_candidate_selection_v2.py](../localize/practical_candidate_selection_v2.py) | Anchor-complete local word-proposal selection. | See source/tests |
| [localize/practical_remap_failure_analysis.py](../localize/practical_remap_failure_analysis.py) | Train-only failure analysis for frozen practical-remap-v1. | See source/tests |
| [localize/practical_remap_protocol.py](../localize/practical_remap_protocol.py) | Train-only calibration and frozen-test evaluation for practical-remap-v1. | See source/tests |
| [localize/practical_remap_v2_protocol.py](../localize/practical_remap_v2_protocol.py) | Train-only calibration and frozen-test geometry for practical-remap-v2. | See source/tests |
| [localize/probability_threshold_word_connectivity.py](../localize/probability_threshold_word_connectivity.py) | Native-CC implementation of frozen paper-like Algorithm 1 post-processing. | See source/tests |
| [localize/repository_contours.py](../localize/repository_contours.py) | Frozen author-repository localization, NOT the paper-component-area path. | repository-contours-variant |
| [localize/rlsa_gap_postsplit.py](../localize/rlsa_gap_postsplit.py) | Deterministic original-gap post-split for the frozen TRAIN RLSA diagnostic. | See source/tests |
| [localize/structural_word_groups.py](../localize/structural_word_groups.py) | GT-blind structural ablations on frozen threshold/window crop proposals. | See source/tests |
| [localize/teacher_student_tests.py](../localize/teacher_student_tests.py) | Explicit APPROXIMATE teacher/student losses and set selection primitives. | teacher-student-tests |
| [localize/text_recall_crops.py](../localize/text_recall_crops.py) | GT-blind crop alternatives for frozen text-recall experiments (APPROXIMATE). | See source/tests |
| [localize/unet_crop_links.py](../localize/unet_crop_links.py) | Inference-only crop geometry and reciprocal pixel-link components. | See source/tests |
| [localize/unet_crop_refinement.py](../localize/unet_crop_refinement.py) | Opt-in APPROXIMATE RGB crop -> same segmenter -> contours experiment. | See source/tests |
| [localize/unet_detection_methods.py](../localize/unet_detection_methods.py) | GT-blind progressive expansion and original-RGB word rectification. | See source/tests |
| [localize/unet_seeded_selective.py](../localize/unet_seeded_selective.py) | GT-free, nucleus-preserving link growth and conservative crop replacement. | See source/tests |
| [localize/unet_word_research.py](../localize/unet_word_research.py) | GT-blind mask stability, learned grouping and glyph/word collaboration. | See source/tests |
| [localize/unet_word_research_v2.py](../localize/unet_word_research_v2.py) | GT-blind soft embedding growth and pixel links; no silent pixel rejection. | agent-experiment-review-2026-10-11 |
| [localize/v1_local_rescue.py](../localize/v1_local_rescue.py) | Local one-hop proposal rescue layered over frozen practical-remap-v1. | See source/tests |
| [localize/word_center_offsets.py](../localize/word_center_offsets.py) | Location cues from dense center voting, constrained by the classic U-Net mask. | See source/tests |
| [localize/word_centerline.py](../localize/word_centerline.py) | GT-blind line markers and line/radius instance reconstruction. | See source/tests |
| [localize/word_crop_refinements.py](../localize/word_crop_refinements.py) | Small GT-blind ablations on frozen word/kernel probabilities (approximate). | See source/tests |
| [localize/word_detector_training.py](../localize/word_detector_training.py) | Explicit word-box and strict-state boundaries for approximate YOLO fitting. | See source/tests |
| [localize/word_evidence.py](../localize/word_evidence.py) | GT-free fragment association and crop proposals on frozen Text+Kernel maps. | See source/tests |
| [localize/word_evidence_graphs.py](../localize/word_evidence_graphs.py) | APPROXIMATE word hypotheses with typed relations and visual ownership. | See source/tests |
| [localize/word_graph_cut.py](../localize/word_graph_cut.py) | APPROXIMATE, support-preserving two-word graph cut; fixed background. | See source/tests |
| [localize/word_graph_recovery.py](../localize/word_graph_recovery.py) | APPROXIMATE word hypotheses, non-destructive admission and crop supervision. | word-graph-recovery |
| [localize/word_guided_contours.py](../localize/word_guided_contours.py) | Opt-in approximate contours guided by existing word-kernel/center maps. | word-guided-contours |
| [localize/word_instance_extensions.py](../localize/word_instance_extensions.py) | GT-free prompt selection and graph partitions for local word experiments. | See source/tests |
| [localize/word_multitask.py](../localize/word_multitask.py) | GT-blind instance separation using existing kernels and minimax watershed. | See source/tests |

## models

| Source | Purpose | Experiment guides |
| --- | --- | --- |
| [models/__init__.py](../models/__init__.py) | Public model package API. | See source/tests |
| [models/adaspot_gap.py](../models/adaspot_gap.py) | Isolated AdaSpot text/kernel/gap experiment on the qualified word backbone. | See source/tests |
| [models/adaspot_gap_context.py](../models/adaspot_gap_context.py) | Isolated frozen-feature Gap heads and matched TRAIN-only pixel strata. | See source/tests |
| [models/adaspot_gap_decoder.py](../models/adaspot_gap_decoder.py) | Approximate Gap-specific decoder; qualified Text/Kernel path stays frozen. | adaspot-gap-decoder |
| [models/adaspot_gap_normalized.py](../models/adaspot_gap_normalized.py) | TRAIN-normalized linear/nonlinear spatial Gap heads, isolated from M/K. | adaspot-gap-normalized |
| [models/adaspot_joint.py](../models/adaspot_joint.py) | Isolated APPROXIMATE AdaSpot: shared AG-UNet, two real OCR heads and gate. | See source/tests |
| [models/adaspot_mask_align.py](../models/adaspot_mask_align.py) | APPROXIMATE AdaSpot Mask-Align: mask-only geometry, bilinear feature sampling. | adaspot-mask-align |
| [models/ag_unet_approx.py](../models/ag_unet_approx.py) | Figure-derived APPROXIMATE AG-UNet reconstruction. | See source/tests |
| [models/directional_transport_unet.py](../models/directional_transport_unet.py) | APPROXIMATE ownership-conditioned feature transport with full-resolution detail. | directional-transport-results, directional-transport-unet |
| [models/fallback_deepsolo.py](../models/fallback_deepsolo.py) | Retired in-process fallback hook. | See source/tests |
| [models/grounded_word_recovery.py](../models/grounded_word_recovery.py) | Small RGB/text verification and variable-length edit models, APPROXIMATE. | grounded-word-recovery |
| [models/icdar2015_improvements.py](../models/icdar2015_improvements.py) | Approximate U-Net norm ablations; keep historical architectures byte-identical. | icdar2015-unet-improvement |
| [models/icdar2015_target70.py](../models/icdar2015_target70.py) | Approximate ICDAR U-Net continuation and explicit partial-encoder gradients. | See source/tests |
| [models/icdar2015_target70_ohem.py](../models/icdar2015_target70_ohem.py) | Approximate continuation: explicit full feature-encoder gradients, frozen BN. | See source/tests |
| [models/icdar2015_unet.py](../models/icdar2015_unet.py) | Isolated approximate ICDAR2015 transfer of the qualified Total-Text U-Net. | icdar2015-totaltext-transfer |
| [models/init.py](../models/init.py) | Public API for `models` package. | See source/tests |
| [models/joint_word_instance_unet.py](../models/joint_word_instance_unet.py) | Approximate jointly trainable attention U-Net and dense word instances. | icdar13-unet-dense-instances |
| [models/kernel_only.py](../models/kernel_only.py) | One-output inference extraction; source training remains joint Text+Kernel. | See source/tests |
| [models/kernel_only_fresh.py](../models/kernel_only_fresh.py) | New Kernel U-Net: fresh decoder/head, local ImageNet encoder weights only. | See source/tests |
| [models/mask_probability_refiner.py](../models/mask_probability_refiner.py) | Small identity-initialized Text/Kernel correctors (research extensions). | See source/tests |
| [models/recognizers.py](../models/recognizers.py) | models/recognizers.py | See source/tests |
| [models/segmenter.py](../models/segmenter.py) | models/segmenter.py | reproduction-spec |
| [models/text_pretrained_encoder.py](../models/text_pretrained_encoder.py) | Strict functional ResNet50 mapping from a pinned text-pretrained checkpoint. | See source/tests |
| [models/totaltext_frequency_gated_skip.py](../models/totaltext_frequency_gated_skip.py) | Practical feature-wavelet skip ablation; the paper-faithful model is untouched. | See source/tests |
| [models/totaltext_partial_encoder.py](../models/totaltext_partial_encoder.py) | Isolated encoder fine-tuning ablation, never used by the author protocol. | See source/tests |
| [models/totaltext_wavelet.py](../models/totaltext_wavelet.py) | Tiny fixed-Haar detail adapter for the practical native-256 Total-Text model. | See source/tests |
| [models/totaltext_word_ablation.py](../models/totaltext_word_ablation.py) | Isolated practical word-training ablations; author models are unchanged. | See source/tests |
| [models/totaltext_word_kernel.py](../models/totaltext_word_kernel.py) | Isolated practical full-word/kernel head; the existing models are unchanged. | word-multitask |
| [models/unet_adv.py](../models/unet_adv.py) | AdvancedUNet: historical repository Attention U-Net plus the clarified reproduction model. | author-clarifications-and-reproduction-update, icdar2015-totaltext-transfer, reproduction-plan, totaltext-bert-correction, totaltext-word-f1-separator-ablation, word-center-offsets |
| [models/unet_auxiliary_ocr.py](../models/unet_auxiliary_ocr.py) | Training-only sequence reader on shared native U-Net features (APPROXIMATE). | See source/tests |
| [models/unet_crop_links.py](../models/unet_crop_links.py) | Independent crop-trained copies of the qualified approximate word U-Net. | See source/tests |
| [models/unet_dense_instances.py](../models/unet_dense_instances.py) | Approximate center/extent instance decoder on qualified frozen U-Net features. | icdar13-unet-dense-instances |
| [models/unet_detection_methods.py](../models/unet_detection_methods.py) | Isolated APPROXIMATE text adaptation, scale fusion and progressive kernels. | See source/tests |
| [models/unet_word_box_field.py](../models/unet_word_box_field.py) | Research word-box field on the complete attention U-Net decoder. | See source/tests |
| [models/unet_word_research.py](../models/unet_word_research.py) | Isolated APPROXIMATE heads; existing word and strict models are unchanged. | See source/tests |
| [models/unet_word_research_v2.py](../models/unet_word_research_v2.py) | Isolated APPROXIMATE corrective experiments, with protected word features. | See source/tests |
| [models/word_center_offsets.py](../models/word_center_offsets.py) | Experimental auxiliary U-Net: each text pixel points toward its word center. | See source/tests |
| [models/word_centerline.py](../models/word_centerline.py) | An isolated APPROXIMATE centerline/radius head on frozen word features. | See source/tests |
| [models/word_graph_decoder.py](../models/word_graph_decoder.py) | Isolated APPROXIMATE ResNet50 geometric proposal decoder. | word-graph-recovery |
| [models/word_instance_extensions.py](../models/word_instance_extensions.py) | Opt-in local word models; no changes to the qualified Text+Kernel model. | See source/tests |
| [models/word_multitask.py](../models/word_multitask.py) | Small experimental heads on the existing Attention U-Net shared decoder. | word-center-offsets, word-multitask-results-v1 |
| [models/word_slot_unet.py](../models/word_slot_unet.py) | APPROXIMATE word-slot U-Net: instance feedback at every decoder scale. | word-slot-methods |

## pipeline

| Source | Purpose | Experiment guides |
| --- | --- | --- |
| [pipeline/__init__.py](../pipeline/__init__.py) | Pipeline utilities that do not instantiate scientific models. | See source/tests |
| [pipeline/approximate_e2e.py](../pipeline/approximate_e2e.py) | One-shot APPROXIMATE M6 integration through conditional DeepSolo fallback. | See source/tests |
| [pipeline/config.py](../pipeline/config.py) | Small, mode-isolated OmegaConf loader with provenance validation. | See source/tests |
| [pipeline/e2e_comparison_diagnostic.py](../pipeline/e2e_comparison_diagnostic.py) | Pure helpers for the APPROXIMATE faithful-vs-practical E2E diagnostic. | See source/tests |
| [pipeline/frozen_segmentation.py](../pipeline/frozen_segmentation.py) | Frozen author-protocol segmentation adapter for downstream recognition. | See source/tests |
| [pipeline/init.py](../pipeline/init.py) | Pipeline utilities: seeding and device helpers. | See source/tests |
| [pipeline/io_utils.py](../pipeline/io_utils.py) | I/O helpers for images and artifacts; kept dependency-light and explicit. | See source/tests |
| [pipeline/lightweight_context.py](../pipeline/lightweight_context.py) | APPROXIMATE M5 pipeline that deliberately stops before DeepSolo fallback. | See source/tests |
| [pipeline/runner.py](../pipeline/runner.py) | End-to-end pipeline runner: | reproduction-spec |
| [pipeline/sequential_lifecycle.py](../pipeline/sequential_lifecycle.py) | Engineering-only sequential GPU lifecycle tracing for the 8 GB target host. | See source/tests |
| [pipeline/uniform_scaled_frontend.py](../pipeline/uniform_scaled_frontend.py) | Isolated resolution experiment; original qualified frontend remains unchanged. | See source/tests |
| [pipeline/uniform_word_pipeline.py](../pipeline/uniform_word_pipeline.py) | RGB-only common U-Net/PARSeq, CPU utility gate and isolated DeepSolo pipeline. | shared-pipeline-code-tour, uniform-unet-fusion |
| [pipeline/word_slot_specialist.py](../pipeline/word_slot_specialist.py) | Ground-truth-free live U-Net/PARSeq specialist with the trained seed protocol. | word-slot-methods |

## recognition

| Source | Purpose | Experiment guides |
| --- | --- | --- |
| [recognition/__init__.py](../recognition/__init__.py) | Scientifically qualified recognition adapters. | See source/tests |
| [recognition/adapted_word_policy.py](../recognition/adapted_word_policy.py) | Prediction-only admission of adapted detector/PARSeq word hypotheses. | See source/tests |
| [recognition/adaspot_gate.py](../recognition/adaspot_gate.py) | TRAIN-only temperature and routing calibration for APPROXIMATE AdaSpot. | See source/tests |
| [recognition/bert_correction.py](../recognition/bert_correction.py) | Bounded BERT post-OCR correction, exclusively for approximate experiments. | See source/tests |
| [recognition/bert_crop_selection.py](../recognition/bert_crop_selection.py) | GT-blind BERT selection of existing crops, for APPROXIMATE experiments only. | totaltext-bert-crop-selection-v1 |
| [recognition/bert_method_selection.py](../recognition/bert_method_selection.py) | Auditable, training-free BERT method controls; inference never consumes GT. | See source/tests |
| [recognition/bert_visual_correction.py](../recognition/bert_visual_correction.py) | Approximate visual/contextual correction. No ground truth enters this module. | See source/tests |
| [recognition/blip_visual_only.py](../recognition/blip_visual_only.py) | Explicit BLIP2 visual/QFormer subset; the language model is never constructed. | encoder-family-router |
| [recognition/boundary_inventory_router.py](../recognition/boundary_inventory_router.py) | Budget-preserving, bounded exchanges around a frozen router's acceptance rank. | shadow-inventory-router |
| [recognition/budget_deployment_factory.py](../recognition/budget_deployment_factory.py) | Instantiate the explicit approximate budget-gate deployment contract. | See source/tests |
| [recognition/budget_distilled_router.py](../recognition/budget_distilled_router.py) | Approximate CPU gate from existing OCR, map layout and shared U-Net context. | budget-distilled-router |
| [recognition/budget_target_oracle.py](../recognition/budget_target_oracle.py) | Exact finite-population target feasibility; every decision here is privileged. | See source/tests |
| [recognition/clip_crop_admission.py](../recognition/clip_crop_admission.py) | APPROXIMATE filtering of frozen high-recall PARSeq outputs with CLIP. | See source/tests |
| [recognition/clip_image_router.py](../recognition/clip_image_router.py) | Selective whole-image routing of frozen PARSeq readings (APPROXIMATE). | See source/tests |
| [recognition/clip_innovative_verifier.py](../recognition/clip_innovative_verifier.py) | APPROXIMATE CLIP hypotheses: letter removal, glyph retrieval and stretching. | See source/tests |
| [recognition/clip_literature_runtime.py](../recognition/clip_literature_runtime.py) | Explicit preprocessing for the isolated, literature-motivated CLIP experiment. | See source/tests |
| [recognition/clip_router_context_chunks.py](../recognition/clip_router_context_chunks.py) | Validated text chunks for the approximate CLIP literature experiment. | clip-router-literature-models |
| [recognition/clip_router_evaluation.py](../recognition/clip_router_evaluation.py) | Auditable word metrics for the APPROXIMATE whole-image CLIP/DeepSolo router. | See source/tests |
| [recognition/clip_router_prompt_ablation.py](../recognition/clip_router_prompt_ablation.py) | GT-free prompt features and TRAIN-only calibration for experimental routers. | See source/tests |
| [recognition/clip_router_threshold_sweep.py](../recognition/clip_router_threshold_sweep.py) | Whole-image router threshold curves from frozen, per-engine word metrics. | See source/tests |
| [recognition/clip_specialized_runtime.py](../recognition/clip_specialized_runtime.py) | Native Long-CLIP/LLM2CLIP inference and a versioned cross-environment bridge. | See source/tests |
| [recognition/clip_word_verifier.py](../recognition/clip_word_verifier.py) | Experimental frozen CLIP verification of PARSeq words on a fixed image. | See source/tests |
| [recognition/common_budget_deployment.py](../recognition/common_budget_deployment.py) | One GT-free deployment interface for every shared-pipeline dataset/expert. | See source/tests |
| [recognition/common_budget_pipeline.py](../recognition/common_budget_pipeline.py) | Shared APPROXIMATE OCR/correction/risk/count-utility pipeline. | common-budget-pipeline |
| [recognition/common_parity_threshold.py](../recognition/common_parity_threshold.py) | Validation-only F1 parity calibration for a frozen image utility score. | common-parity-threshold |
| [recognition/context_memory_router.py](../recognition/context_memory_router.py) | Supervised TRAIN-only retrieval of frozen branch errors; no neural updates. | creative-context-router |
| [recognition/contextual_boundary_selection.py](../recognition/contextual_boundary_selection.py) | GT-blind boundary alternatives and matched contextual selection features. | contextual-boundary-selection |
| [recognition/contextual_quality_features.py](../recognition/contextual_quality_features.py) | APPROXIMATE literal-caption evidence and label-free contrastive text probes. | See source/tests |
| [recognition/contextual_quality_router.py](../recognition/contextual_quality_router.py) | APPROXIMATE, frozen contextual probes for whole-image OCR routing. | See source/tests |
| [recognition/corrected_unet_transfer.py](../recognition/corrected_unet_transfer.py) | Prediction-only deployment of the frozen TotalText corrected U-Net expert. | essential-router-icdar-transfer |
| [recognition/creative_context_runtime.py](../recognition/creative_context_runtime.py) | Frozen PARSeq context-order probes and unchanged-pixel BLIP2 context swaps. | creative-context-router |
| [recognition/creative_context_signals.py](../recognition/creative_context_signals.py) | Approximate GT-free context interventions, repeated glyphs and structural checks. | creative-context-router |
| [recognition/crop_clip_battery.py](../recognition/crop_clip_battery.py) | GT-blind crop variants and score selection for frozen Total-Text TRAIN masks. | clip-word-verification |
| [recognition/crop_group_refinements.py](../recognition/crop_group_refinements.py) | GT-blind refinements for approximate crop-group diagnostics. | See source/tests |
| [recognition/crop_recovery_metric.py](../recognition/crop_recovery_metric.py) | Post-hoc oracle word recall of an unrestricted crop hypothesis pool. | See source/tests |
| [recognition/crop_word_group_metric.py](../recognition/crop_word_group_metric.py) | Post-hoc oracle evaluation with one output per group AND per original crop. | See source/tests |
| [recognition/deepsolo_icdar2015_gap.py](../recognition/deepsolo_icdar2015_gap.py) | Validation-only DeepSolo filtering and paired, symbol-free ICDAR15 diagnostics. | deepsolo-icdar2015-gap |
| [recognition/deepsolo_lexicon.py](../recognition/deepsolo_lexicon.py) | Pinned-source DeepSolo lexicon correction; never receives evaluation targets. | deepsolo-icdar-lexicons |
| [recognition/deepsolo_lexicon_deployment.py](../recognition/deepsolo_lexicon_deployment.py) | GT-blind DeepSolo vocabulary postprocessor with explicit artifact identities. | deepsolo-icdar-lexicons |
| [recognition/deepsolo_lexicon_guard.py](../recognition/deepsolo_lexicon_guard.py) | Observable confidence/ambiguity guards for general-vocabulary DeepSolo correction. | deepsolo-icdar-lexicons |
| [recognition/deepsolo_word_f1.py](../recognition/deepsolo_word_f1.py) | APPROXIMATE symbol-free word counts and lexicon-free DeepSolo experiments. | deepsolo-word-f1-study |
| [recognition/directional_transport_metrics.py](../recognition/directional_transport_metrics.py) | Symbol-free localized word metrics; empty/duplicate predictions remain counted. | directional-transport-results, directional-transport-unet |
| [recognition/dont_care_evaluation.py](../recognition/dont_care_evaluation.py) | Neutralize final predictions for metrics only, retaining all original rows. | See source/tests |
| [recognition/encoder_family_inference.py](../recognition/encoder_family_inference.py) | Prediction-only single-image boundary for a frozen encoder-family checkpoint. | encoder-family-router |
| [recognition/encoder_family_router.py](../recognition/encoder_family_router.py) | Approximate matched encoder ablations; inputs contain predictions only. | encoder-family-router |
| [recognition/essential_budget_router.py](../recognition/essential_budget_router.py) | Single-image interface for the frozen, validation-budgeted essential gate. | essential-router-budget50 |
| [recognition/essential_portable_router.py](../recognition/essential_portable_router.py) | Portable NumPy deployment of the qualified corrected-U-Net budget gate. | budget-distilled-router |
| [recognition/essential_threshold_deployment.py](../recognition/essential_threshold_deployment.py) | GT-blind single-image interface for exported validation-selected gate profiles. | essential-router-icdar-transfer, essential-router-threshold-analysis |
| [recognition/essential_threshold_frontier.py](../recognition/essential_threshold_frontier.py) | Exact cached-expert threshold frontiers and explicitly conditional uncertainty. | essential-router-threshold-analysis |
| [recognition/essential_word_heuristics.py](../recognition/essential_word_heuristics.py) | Small GT-free routing controls using the already executed word U-Net. | See source/tests |
| [recognition/f1_quality_study.py](../recognition/f1_quality_study.py) | Training-free quality proxies and TRAIN-only cuts for approximate OCR routing. | yolo-f1-quality-study |
| [recognition/fallback_feasibility.py](../recognition/fallback_feasibility.py) | Exact target-F1 feasibility for two fixed image-level experts; GT diagnostics only. | icdar13-word-refit |
| [recognition/fallback_reread.py](../recognition/fallback_reread.py) | GT-free replacement of fallback transcripts from optional PARSeq rereads. | lightweight-icdar13-90 |
| [recognition/fallback_word_fusion.py](../recognition/fallback_word_fusion.py) | GT-free, one-to-one transcript replacement on already routed fallback words. | See source/tests |
| [recognition/fixed_expert_budget.py](../recognition/fixed_expert_budget.py) | Shared count-utility gate around fixed observable expert word emissions. | lightweight-icdar13-90 |
| [recognition/fixed_expert_deployment.py](../recognition/fixed_expert_deployment.py) | Annotation-free inference for a qualified fixed-emission budget gate. | See source/tests |
| [recognition/fold_safe_teacher.py](../recognition/fold_safe_teacher.py) | CPU-only teacher boundaries; no validation/test dictionary is accepted. | See source/tests |
| [recognition/fusion_deployment_factory.py](../recognition/fusion_deployment_factory.py) | Resolve frozen approximate fusion and budget-gate inference contracts. | See source/tests |
| [recognition/grounded_context_router.py](../recognition/grounded_context_router.py) | GT-blind region inventory, local evidence and frozen visual router features. | grounded-context-router |
| [recognition/grounded_context_runtime.py](../recognition/grounded_context_runtime.py) | Offline frozen Florence inventory and local BLIP2 visual evidence runtimes. | grounded-context-router |
| [recognition/grounded_word_recovery.py](../recognition/grounded_word_recovery.py) | GT-free decisions and explicit byte edits for APPROXIMATE word recovery. | grounded-word-recovery |
| [recognition/group_output_filter.py](../recognition/group_output_filter.py) | Inference-only postfilters for frozen crop-group predictions. | See source/tests |
| [recognition/group_word_selection.py](../recognition/group_word_selection.py) | Dataset-independent, training-free selection of observed crop readings. | crop-group-word-selection |
| [recognition/group_word_selection_runtime.py](../recognition/group_word_selection_runtime.py) | Frozen PARSeq likelihood and CLIP scoring for arbitrary crop groups. | clip-word-verification |
| [recognition/hard_fallback_budget.py](../recognition/hard_fallback_budget.py) | Explicit image-budget constraints, separate from scientific STRICT mode. | shared-pipeline-code-tour |
| [recognition/icdar13_annotations.py](../recognition/icdar13_annotations.py) | Verified ICDAR13 Task 1 recognition annotations for APPROXIMATE evaluation work. | See source/tests |
| [recognition/icdar13_hypothesis_tests.py](../recognition/icdar13_hypothesis_tests.py) | Isolated APPROXIMATE controls for IC13 vocabulary and out-of-fold routing. | icdar13-hypothesis-tests |
| [recognition/icdar13_matched_router.py](../recognition/icdar13_matched_router.py) | GT-blind OCR observations and exact image budgets for the native IC13 gate. | icdar13-matched-router-generic |
| [recognition/icdar13_trace_diagnostics.py](../recognition/icdar13_trace_diagnostics.py) | Explicit non-Table-5 diagnostics over frozen ICDAR13 APPROXIMATE traces. | See source/tests |
| [recognition/icdar2015_word_f1.py](../recognition/icdar2015_word_f1.py) | Auditable bbox word F1 for approximate ICDAR2015 U-Net/PARSeq evaluation. | icdar2015-word-f1 |
| [recognition/icdar_budget_deployment.py](../recognition/icdar_budget_deployment.py) | Cheap single-image ICDAR gate; inputs exclude GT and fallback predictions. | icdar-trained-budget-router |
| [recognition/icdar_budget_router.py](../recognition/icdar_budget_router.py) | Approximate native-dataset low-cost routing, with explicit map frames. | icdar-trained-budget-router |
| [recognition/image_context_correction.py](../recognition/image_context_correction.py) | Experimental image-wide OCR correction; inference functions never read GT. | image-context-correction |
| [recognition/image_context_joint_policy.py](../recognition/image_context_joint_policy.py) | GT-blind composition of frozen correction and false-positive policies. | image-context-correction |
| [recognition/lexical_budget_router.py](../recognition/lexical_budget_router.py) | GT-free conservative lexicon editing and cached-observation budget heads. | See source/tests |
| [recognition/lightweight_unet_router.py](../recognition/lightweight_unet_router.py) | Approximate small image gates with explicit feature cost and frozen calibration. | lightweight-unet-router |
| [recognition/lightweight_word_deployment.py](../recognition/lightweight_word_deployment.py) | Annotation-free deployment of the trained admission and fixed-budget router. | See source/tests |
| [recognition/lightweight_word_router.py](../recognition/lightweight_word_router.py) | Small pre-fallback features for a word-admission image router. | See source/tests |
| [recognition/lightweight_word_selection.py](../recognition/lightweight_word_selection.py) | Observable word admission and conditional fallback reconciliation. | lightweight-icdar13-90 |
| [recognition/line_token_evaluation.py](../recognition/line_token_evaluation.py) | Approximate word recovery from line transcriptions, without invented word boxes. | See source/tests |
| [recognition/neuspell_correction.py](../recognition/neuspell_correction.py) | Frozen NeuSpell BERT word-head decoding for approximate post-OCR studies. | See source/tests |
| [recognition/paper_caption_comparison.py](../recognition/paper_caption_comparison.py) | Matched-expert APPROXIMATE replay of paper section 3.5 / Algorithm 2. | paper-caption-router-comparison |
| [recognition/paper_image_router_evaluation.py](../recognition/paper_image_router_evaluation.py) | APPROXIMATE image-word evaluation for unlocalized T1 and frozen T3/DeepSolo. | See source/tests |
| [recognition/parseq.py](../recognition/parseq.py) | Provenance-locked APPROXIMATE PARSeq sequence recognition. | shared-pipeline-code-tour |
| [recognition/parseq_candidate_ranking.py](../recognition/parseq_candidate_ranking.py) | Post-hoc geometry/OCR ranking metrics for frozen TRAIN proposals. | See source/tests |
| [recognition/parseq_crop_objective.py](../recognition/parseq_crop_objective.py) | Differentiable local boxes and frozen PARSeq teacher forcing, opt-in only. | See source/tests |
| [recognition/parseq_crop_sensitivity.py](../recognition/parseq_crop_sensitivity.py) | GT-bridged PARSeq crop sensitivity diagnostics for APPROXIMATE traces. | See source/tests |
| [recognition/parseq_deepsolo_alignment.py](../recognition/parseq_deepsolo_alignment.py) | Spatial/text diagnostics between PARSeq crop outputs and DeepSolo detections. | See source/tests |
| [recognition/replacement_fallback_counts.py](../recognition/replacement_fallback_counts.py) | Explicit hash-bound teacher counts for approximate router retraining. | See source/tests |
| [recognition/residual_cost_router.py](../recognition/residual_cost_router.py) | Approximate, training-free routing costs from uncovered predicted text regions. | yolo-residual-cost-router |
| [recognition/residual_text_vqa.py](../recognition/residual_text_vqa.py) | Frozen BLIP2 residual-text verification, an APPROXIMATE routing experiment. | See source/tests |
| [recognition/risk_admission_deployment.py](../recognition/risk_admission_deployment.py) | GT-free risk admission and image routing using one trained word model. | See source/tests |
| [recognition/risk_fusion_deployment.py](../recognition/risk_fusion_deployment.py) | Add a frozen post-fallback fusion policy to a trained risk-admission gate. | See source/tests |
| [recognition/router_cause_tests.py](../recognition/router_cause_tests.py) | Predeclared interventions for approximate context-router failure tests. | router-cause-tests |
| [recognition/router_fallback_replay.py](../recognition/router_fallback_replay.py) | Frozen image decisions with a replacement fallback and complete count vectors. | trained-router-vocabulary-replay |
| [recognition/router_validation.py](../recognition/router_validation.py) | Held-out validation identities and cuts for an unchanged deployed router. | router-validation-study |
| [recognition/scene_text_datasets.py](../recognition/scene_text_datasets.py) | Fail-closed annotation parsers for additional scene-text benchmarks. | See source/tests |
| [recognition/scene_text_manifest.py](../recognition/scene_text_manifest.py) | Deterministic manifests for ICDAR2015, Total-Text, and CTW1500. | See source/tests |
| [recognition/scene_text_v1_evaluation.py](../recognition/scene_text_v1_evaluation.py) | Evaluation primitives for frozen practical-remap-v1 scene-text proposals. | See source/tests |
| [recognition/segmentation_target_component_topology.py](../recognition/segmentation_target_component_topology.py) | Native target/prediction component topology diagnostics. | See source/tests |
| [recognition/segmentation_word_topology.py](../recognition/segmentation_word_topology.py) | GT-bridged native-mask topology and OCR diagnostics for first-ten traces. | See source/tests |
| [recognition/selected_deepsolo_evidence.py](../recognition/selected_deepsolo_evidence.py) | Verify fresh, selected-only DeepSolo export receipts and resource evidence. | See source/tests |
| [recognition/selective_word_repair.py](../recognition/selective_word_repair.py) | Pure APPROXIMATE selective word-repair primitives; inference never takes GT. | selective-word-repair-tests |
| [recognition/shadow_inventory_router.py](../recognition/shadow_inventory_router.py) | Approximate whole-image routing from an independent text-region inventory. | shadow-inventory-router |
| [recognition/text_coverage_router.py](../recognition/text_coverage_router.py) | APPROXIMATE image routing: reading confidence plus inference-only text coverage. | See source/tests |
| [recognition/text_reconstruction.py](../recognition/text_reconstruction.py) | Text-only APPROXIMATE reconstruction of frozen OCR observations. | text-reconstruction-tests |
| [recognition/token_context_router.py](../recognition/token_context_router.py) | Approximate benefit routing from frozen visual tokens and dense word support. | token-context-router |
| [recognition/token_context_runtime.py](../recognition/token_context_runtime.py) | Frozen local CLIP dense tokens and BLIP2 transcript likelihood, FP32 only. | token-context-router |
| [recognition/trainable_image_router.py](../recognition/trainable_image_router.py) | Approximate supervised tabular routing; frozen visual branches and GT-free inputs. | yolo-trainable-router |
| [recognition/training_free_consistency_router.py](../recognition/training_free_consistency_router.py) | APPROXIMATE frozen-model image gates; labels enter calibration/evaluation only. | yolo-training-free-consistency-router |
| [recognition/unet_corrected_token_interfaces.py](../recognition/unet_corrected_token_interfaces.py) | Explicit approximate corrected-expert counting and token routing boundaries. | unet-corrected-token-router |
| [recognition/unet_expert_records.py](../recognition/unet_expert_records.py) | Prediction-only U-Net/PARSeq rows in the common fixed-expert schema. | icdar13-unet-yolo-procedure |
| [recognition/unet_fallback_coverage.py](../recognition/unet_fallback_coverage.py) | GT-free missing-text and native OCR confidence evidence for image fallback. | unet-minimum-fallback |
| [recognition/unet_minimum_fallback.py](../recognition/unet_minimum_fallback.py) | APPROXIMATE fixed-ranking fallback budgets and observable U-Net evidence. | unet-minimum-fallback |
| [recognition/unet_token_routing.py](../recognition/unet_token_routing.py) | Explicit approximate U-Net crop and TRAIN-threshold routing boundaries. | unet-token-router |
| [recognition/uniform_encoder_context.py](../recognition/uniform_encoder_context.py) | Qualified linear projection of already-computed frozen encoder features. | shared-pipeline-code-tour |
| [recognition/uniform_fallback.py](../recognition/uniform_fallback.py) | Shared annotation-free DeepSolo word filtering and optional benchmark lexicon. | shared-pipeline-code-tour, uniform-unet-fusion |
| [recognition/uniform_lexical_risk.py](../recognition/uniform_lexical_risk.py) | Shared optional benchmark-lexicon preparation for uniform OCR/risk models. | uniform-unet-fusion |
| [recognition/uniform_multiscale_words.py](../recognition/uniform_multiscale_words.py) | Conservative observable word additions from an unchanged second-scale expert. | shared-pipeline-code-tour |
| [recognition/uniform_observation_bank.py](../recognition/uniform_observation_bank.py) | Fail-closed selection of qualified native or decoder-heldout FIT observations. | See source/tests |
| [recognition/uniform_pooled_gate.py](../recognition/uniform_pooled_gate.py) | Small shared image-count router, fitted with equal total domain weights. | See source/tests |
| [recognition/uniform_risk_fusion.py](../recognition/uniform_risk_fusion.py) | One learned word-risk, expected-error gate and conditional fusion for all domains. | shared-pipeline-code-tour, uniform-unet-fusion |
| [recognition/uniform_utility_deployment.py](../recognition/uniform_utility_deployment.py) | Qualified annotation-free common utility/fusion deployment boundary. | shared-pipeline-code-tour |
| [recognition/uniform_utility_fusion.py](../recognition/uniform_utility_fusion.py) | One lexical/word-risk/count-utility model family for all three domains. | shared-pipeline-code-tour, uniform-unet-fusion |
| [recognition/uniform_word_admission.py](../recognition/uniform_word_admission.py) | Explicit probability filtering and validation selection for light OCR words. | See source/tests |
| [recognition/uniform_word_union.py](../recognition/uniform_word_union.py) | Shared prediction-only word union, learned correctness and spatial conflicts. | shared-pipeline-code-tour |
| [recognition/validated_budget_variant.py](../recognition/validated_budget_variant.py) | Load one independently qualified, validation-selected approximate pipeline. | lightweight-icdar13-90 |
| [recognition/word_evidence_alignment.py](../recognition/word_evidence_alignment.py) | Frozen PARSeq cross-attention alignment for APPROXIMATE evidence experiments. | See source/tests |
| [recognition/word_slot_evaluation.py](../recognition/word_slot_evaluation.py) | Harmonized bbox/text word counts and deployable, calibration-only routers. | See source/tests |
| [recognition/word_slot_routing.py](../recognition/word_slot_routing.py) | Deploy calibrated word-slot selectors without ground truth or sklearn objects. | word-slot-alphanumeric, word-slot-methods |
| [recognition/word_slot_teachers.py](../recognition/word_slot_teachers.py) | Pinned external crop teachers; no guessed checkpoint architecture mapping. | word-slot-methods |
| [recognition/word_slot_text.py](../recognition/word_slot_text.py) | Explicit comparison-only text policies; original OCR/GT strings stay intact. | word-slot-alphanumeric |
| [recognition/word_unet_policy.py](../recognition/word_unet_policy.py) | GT-free emission policy for the matched two-map word U-Net extension. | See source/tests |
| [recognition/yolo_router_comparison.py](../recognition/yolo_router_comparison.py) | Paired whole-image router evaluation; GT appears only in these diagnostics. | See source/tests |
| [recognition/yolo_scale_agreement.py](../recognition/yolo_scale_agreement.py) | Match word readings across detector resolutions without labels. | See source/tests |
| [recognition/yolo_word_refinement.py](../recognition/yolo_word_refinement.py) | GT-blind object-box crop variants and OCR-supported admission, no dataset rules. | See source/tests |

## research

| Source | Purpose | Experiment guides |
| --- | --- | --- |
| [research/__init__.py](../research/__init__.py) | Read-only navigation and comparison of saved research evidence (stdlib only). | See source/tests |
| [research/__main__.py](../research/__main__.py) | Usage: python3 -m research {groups,compare,check,overview,find,section,catalog}. | See source/tests |
| [research/comparison.py](../research/comparison.py) | Explicit projections of pinned reports; never load models or infer protocols. | See source/tests |
| [research/documents.py](../research/documents.py) | Bounded audit discovery and section reads, without walking sample caches. | See source/tests |
| [research/library.py](../research/library.py) | Portable experiment navigation and verified saved-result summaries. | See source/tests |

## scoring

| Source | Purpose | Experiment guides |
| --- | --- | --- |
| [scoring/__init__.py](../scoring/__init__.py) | Public context-scoring package API. | See source/tests |
| [scoring/budgeted_paired.py](../scoring/budgeted_paired.py) | Opt-in approximate paired routing; faithful scoring defaults stay unchanged. | See source/tests |
| [scoring/context_candidate_ranking.py](../scoring/context_candidate_ranking.py) | GT-blind context scores and post-hoc ranking diagnostics for word proposals. | See source/tests |
| [scoring/context_score.py](../scoring/context_score.py) | scoring/context_score.py | See source/tests |
| [scoring/feasibility.py](../scoring/feasibility.py) | Diagnostic for literal whole-string fuzzy matching at the paper threshold. | See source/tests |
| [scoring/init.py](../scoring/init.py) | Public API for `scoring` package. | See source/tests |
| [scoring/lexical.py](../scoring/lexical.py) | Explicit lexical preprocessing modes around the repository's literal ratio path. | See source/tests |
| [scoring/local_context.py](../scoring/local_context.py) | Training-free local caption matching for a separate practical approximation. | See source/tests |
| [scoring/paper_image_router.py](../scoring/paper_image_router.py) | APPROXIMATE image-level interpretation of paper section 3.5/Algorithm 2. | See source/tests |
| [scoring/semantic.py](../scoring/semantic.py) | Pinned APPROXIMATE MPNet sentence embedding integration. | See source/tests |
| [scoring/strategies.py](../scoring/strategies.py) | Named interpretations of the paper/repository context-score conflict. | See source/tests |

## scripts

| Source | Purpose | Experiment guides |
| --- | --- | --- |
| [scripts/analyze_budget_target_router.py](../scripts/analyze_budget_target_router.py) | Reproducible exact61/228 oracle feasibility and independent certificates. | See source/tests |
| [scripts/analyze_contextual_boundary_selection.py](../scripts/analyze_contextual_boundary_selection.py) | Post-hoc boundary failure analysis; never changes fitted models or policies. | contextual-boundary-selection |
| [scripts/analyze_deepsolo_icdar2015_gap.py](../scripts/analyze_deepsolo_icdar2015_gap.py) | Freeze ICDAR15 calibration policies, then evaluate authenticated full TEST outputs. | deepsolo-icdar2015-gap |
| [scripts/analyze_deepsolo_word_f1.py](../scripts/analyze_deepsolo_word_f1.py) | Replay immutable DeepSolo reports; select symbol-free F1 policies on TRAIN only. | deepsolo-word-f1-study |
| [scripts/analyze_encoder_family_signal.py](../scripts/analyze_encoder_family_signal.py) | Fixed-policy paired evidence for keeping one frozen encoder or both. | encoder-family-router |
| [scripts/analyze_essential_fallback_control.py](../scripts/analyze_essential_fallback_control.py) | Exploratory ordinary-DeepSolo control; no expert or router threshold search. | essential-router-threshold-analysis |
| [scripts/analyze_essential_oracle.py](../scripts/analyze_essential_oracle.py) | Privileged GT-routing bounds; these are not deployable threshold methods. | essential-router-threshold-analysis |
| [scripts/analyze_essential_router_thresholds.py](../scripts/analyze_essential_router_thresholds.py) | Validation-frozen and explicitly post-hoc threshold frontiers on three datasets. | essential-router-threshold-analysis |
| [scripts/analyze_grounded_context_router.py](../scripts/analyze_grounded_context_router.py) | Audit grounded probes, independent model refits and full-word routing metrics. | grounded-context-router |
| [scripts/analyze_grounded_word_recovery.py](../scripts/analyze_grounded_word_recovery.py) | Independent matching, paired word attribution, costs and artifact hashes. | grounded-word-recovery |
| [scripts/analyze_icdar_fallback_target.py](../scripts/analyze_icdar_fallback_target.py) | Certify target F1 feasibility for the frozen ICDAR budget-router experts. | icdar13-word-refit |
| [scripts/analyze_image_context_correction.py](../scripts/analyze_image_context_correction.py) | Post-hoc failure analysis of frozen image-context correction experiments. | image-context-correction |
| [scripts/analyze_mask_to_crop_failures.py](../scripts/analyze_mask_to_crop_failures.py) | CPU-only failure diagnosis of a TRAIN-selected U-Net's frozen TEST crops. | See source/tests |
| [scripts/analyze_native128_failures.py](../scripts/analyze_native128_failures.py) | CPU-only APPROXIMATE frozen OCR failure and native TRAIN target analysis. | See source/tests |
| [scripts/analyze_practical_remap_train_failures.py](../scripts/analyze_practical_remap_train_failures.py) | Analyze frozen practical-remap-v1 failures on ICDAR13 train-229 only. | See source/tests |
| [scripts/analyze_router_cause_tests.py](../scripts/analyze_router_cause_tests.py) | Offline paired analysis of completed approximate router cause tests. | router-cause-tests |
| [scripts/analyze_router_learning_steps.py](../scripts/analyze_router_learning_steps.py) | OOF learning-curve controls with equal or nearly equal optimizer updates. | router-cause-tests |
| [scripts/analyze_scene_text_yolo_thresholds.py](../scripts/analyze_scene_text_yolo_thresholds.py) | Cached TEST threshold controls and CTW full-line failure diagnostics; no inference. | See source/tests |
| [scripts/analyze_selective_word_repair_tests.py](../scripts/analyze_selective_word_repair_tests.py) | Post-hoc diagnostics of immutable selective-repair results; no policy changes. | selective-word-repair-tests |
| [scripts/analyze_text_reconstruction_tests.py](../scripts/analyze_text_reconstruction_tests.py) | Post-hoc error analysis of completed, immutable frozen-OCR experiments. | text-reconstruction-tests |
| [scripts/analyze_totaltext_crop_word_groups.py](../scripts/analyze_totaltext_crop_word_groups.py) | CPU-only grouping battery reusing frozen Total-Text crop/OCR shards. | See source/tests |
| [scripts/analyze_totaltext_structural_groups.py](../scripts/analyze_totaltext_structural_groups.py) | Four GT-blind structural grouping ablations on frozen Total-Text TEST crops. | See source/tests |
| [scripts/analyze_trainable_image_router.py](../scripts/analyze_trainable_image_router.py) | Audit supervised fit scopes/portable predictions and report oracle separation. | yolo-trainable-router |
| [scripts/analyze_unet_fallback_transfer.py](../scripts/analyze_unet_fallback_transfer.py) | Post-hoc approximate frozen U-Net router grid; explicitly TEST-dependent cuts. | unet-minimum-fallback |
| [scripts/analyze_word_evidence_graphs.py](../scripts/analyze_word_evidence_graphs.py) | Independent word-F1 audit and failure accounting for evidence graph experiments. | See source/tests |
| [scripts/analyze_word_slot_routers.py](../scripts/analyze_word_slot_routers.py) | Fit deployable image/word selectors on TRAIN; measure untouched TEST outcomes. | word-slot-methods |
| [scripts/analyze_yolo_coverage_resolution.py](../scripts/analyze_yolo_coverage_resolution.py) | Audit a completed approximate coverage/resolution experiment without model inference. | yolo-text-coverage-resolution |
| [scripts/analyze_yolo_f1_quality_study.py](../scripts/analyze_yolo_f1_quality_study.py) | Descriptive score diagnostics and publication-style plots of an audited study. | yolo-f1-quality-study |
| [scripts/analyze_yolo_residual_cost_router.py](../scripts/analyze_yolo_residual_cost_router.py) | Independently audit residual components/counts and export approximate results. | yolo-residual-cost-router |
| [scripts/audit_adaspot_gap_adaptation.py](../scripts/audit_adaspot_gap_adaptation.py) | Independent held-out selection, native/RGB matching and artifact audit. | See source/tests |
| [scripts/audit_adaspot_gap_context.py](../scripts/audit_adaspot_gap_context.py) | Independent counts and matched-training audit for Gap context experiments. | See source/tests |
| [scripts/audit_adaspot_gap_context_final.py](../scripts/audit_adaspot_gap_context_final.py) | Independent audit of the explicitly post-hoc fixed final-epoch diagnostic. | See source/tests |
| [scripts/audit_adaspot_gap_fulltrain.py](../scripts/audit_adaspot_gap_fulltrain.py) | Independent full-population, fixed-selection and TEST overlap/count audit. | See source/tests |
| [scripts/audit_adaspot_gap_ocr_selection.py](../scripts/audit_adaspot_gap_ocr_selection.py) | Independent calibration-only OCR selection and evaluation count audit. | adaspot-gap-ocr-selection |
| [scripts/audit_adaspot_gap_probe.py](../scripts/audit_adaspot_gap_probe.py) | CPU count/cache/selection audit for the completed frozen-feature Gap probe. | See source/tests |
| [scripts/audit_adaspot_joint.py](../scripts/audit_adaspot_joint.py) | Independent population, checkpoint, Hungarian metric and routing audit. | adaspot-joint |
| [scripts/audit_adaspot_mask_align.py](../scripts/audit_adaspot_mask_align.py) | Independent fixed-population Hungarian audit; no production scorer imports. | adaspot-mask-align |
| [scripts/audit_adaspot_phase0.py](../scripts/audit_adaspot_phase0.py) | Independent Hungarian count audit of frozen AdaSpot Phase-0 outputs (CPU). | adaspot-phase0 |
| [scripts/audit_boundary_inventory_router.py](../scripts/audit_boundary_inventory_router.py) | Independent finite-policy search and word-count replay for bounded routing. | shadow-inventory-router |
| [scripts/audit_budget_distilled_router.py](../scripts/audit_budget_distilled_router.py) | Independent replay of cheap-gate fitting, augmentation labels and deployment. | budget-distilled-router |
| [scripts/audit_clip_specialized_routers.py](../scripts/audit_clip_specialized_routers.py) | Replay and audit native specialized-router results without model inference. | clip-specialized-routers |
| [scripts/audit_common_budget_pipeline.py](../scripts/audit_common_budget_pipeline.py) | Independent fitting, selection, count and deployment audit of the shared gate. | See source/tests |
| [scripts/audit_contextual_boundary_selection.py](../scripts/audit_contextual_boundary_selection.py) | Independent lexical, copying, candidate, fold and calibration audit. | contextual-boundary-selection |
| [scripts/audit_creative_context_router.py](../scripts/audit_creative_context_router.py) | Replay six context signals, independent regressors/retrieval and routing counts. | creative-context-router |
| [scripts/audit_crossfit_fixed_gate.py](../scripts/audit_crossfit_fixed_gate.py) | Independently refit every candidate fold and replay crossfit gate selection. | See source/tests |
| [scripts/audit_deepsolo_icdar2015_gap.py](../scripts/audit_deepsolo_icdar2015_gap.py) | Independent lexical/count, calibration optimum and new checkpoint/export audit. | deepsolo-icdar2015-gap |
| [scripts/audit_deepsolo_lexicon_deployment.py](../scripts/audit_deepsolo_lexicon_deployment.py) | Replay promoted prediction-only dictionary experts on every cached TEST image. | deepsolo-icdar-lexicons |
| [scripts/audit_deepsolo_lexicons.py](../scripts/audit_deepsolo_lexicons.py) | Independent order-sensitive dictionary lookup, policy and full word-count audit. | deepsolo-icdar-lexicons |
| [scripts/audit_deepsolo_word_f1.py](../scripts/audit_deepsolo_word_f1.py) | Independent Hungarian recount and decision replay for the DeepSolo F1 study. | deepsolo-word-f1-study |
| [scripts/audit_deepsolo_word_f1_profiles.py](../scripts/audit_deepsolo_word_f1_profiles.py) | Independently recount deployable profiles and compare real GT-free GPU smokes. | deepsolo-word-f1-study |
| [scripts/audit_detailed_caption_comparison.py](../scripts/audit_detailed_caption_comparison.py) | Independent native-token/RGB/T1-reuse audit for the Qwen router ablation. | See source/tests |
| [scripts/audit_detailed_captions.py](../scripts/audit_detailed_captions.py) | Verify native Qwen caption tokens, images, artifacts, prompts and provenance. | detailed-scene-captions |
| [scripts/audit_encoder_family_router.py](../scripts/audit_encoder_family_router.py) | Independent split, normalization, checkpoint selection and routing-count audit. | encoder-family-router |
| [scripts/audit_english_refactor.py](../scripts/audit_english_refactor.py) | Verify English presentation against archived source and pinned original reports. | See source/tests |
| [scripts/audit_essential_profiles_oracle.py](../scripts/audit_essential_profiles_oracle.py) | Replay every exported validation profile and certify privileged routing bounds. | essential-router-threshold-analysis |
| [scripts/audit_essential_router_budget.py](../scripts/audit_essential_router_budget.py) | Independent budget-head selection/count replay and single-image deployment audit. | budget-distilled-router, essential-router-budget50 |
| [scripts/audit_essential_thresholds.py](../scripts/audit_essential_thresholds.py) | Independent image matching, gate replay and rational threshold optimality audit. | essential-router-threshold-analysis |
| [scripts/audit_fallback_word_fusion.py](../scripts/audit_fallback_word_fusion.py) | Independently recount and replay every frozen fallback fusion policy. | See source/tests |
| [scripts/audit_fixed_expert_budget.py](../scripts/audit_fixed_expert_budget.py) | Independent TRAIN recounts, model refits and deployment replay before TEST. | See source/tests |
| [scripts/audit_fixed_expert_retention.py](../scripts/audit_fixed_expert_retention.py) | Independently audit fixed-gate training on previously qualified TT/IC15 data. | See source/tests |
| [scripts/audit_fixed_expert_test.py](../scripts/audit_fixed_expert_test.py) | Independently replay a frozen IC13 TEST route, words, crops and counts. | See source/tests |
| [scripts/audit_fixed_retention_test.py](../scripts/audit_fixed_retention_test.py) | Recompute all fixed-budget retention TEST decisions and lexical totals. | See source/tests |
| [scripts/audit_ic13_complement.py](../scripts/audit_ic13_complement.py) | Replay the selection-only complement and every staged light image. | See source/tests |
| [scripts/audit_ic13_fulltrain_oof.py](../scripts/audit_ic13_fulltrain_oof.py) | Check image-group OOF isolation, model choice, and exact TEST route. | See source/tests |
| [scripts/audit_ic13_split_generic.py](../scripts/audit_ic13_split_generic.py) | Recount TRAIN-only Generic DeepSolo preparation from authenticated raw envelopes. | See source/tests |
| [scripts/audit_ic13_unet_expert.py](../scripts/audit_ic13_unet_expert.py) | Audit U-Net/PARSeq exported maps, crops and decoder-heldout boundaries. | icdar13-unet-yolo-procedure |
| [scripts/audit_ic13_word_risk.py](../scripts/audit_ic13_word_risk.py) | Audit image-disjoint OOF word risk and its archived router descriptors. | See source/tests |
| [scripts/audit_icdar13_dense_instances.py](../scripts/audit_icdar13_dense_instances.py) | Independent geometry, cache, fit-scope and lexical count audit. | See source/tests |
| [scripts/audit_icdar13_dense_policy.py](../scripts/audit_icdar13_dense_policy.py) | Reconstruct FIT-only vocabulary and every frozen dense-expert policy count. | See source/tests |
| [scripts/audit_icdar13_matched_router_generic.py](../scripts/audit_icdar13_matched_router_generic.py) | Independent official-CSV recount and exact61-image decision audit. | icdar13-matched-router-generic |
| [scripts/audit_icdar13_matched_word_training.py](../scripts/audit_icdar13_matched_word_training.py) | Independently replay ICDAR13 TEST lexical counts from CSV labels and emissions. | icdar13-matched-word-training |
| [scripts/audit_icdar13_text_encoder_words.py](../scripts/audit_icdar13_text_encoder_words.py) | Check source identities, frozen encoders, fitting scopes and word counts. | See source/tests |
| [scripts/audit_icdar13_text_word_policy.py](../scripts/audit_icdar13_text_word_policy.py) | Independently recount finite selection, RGB crops and the full frozen expert. | icdar13-unet-dense-instances |
| [scripts/audit_icdar13_transfer_duration.py](../scripts/audit_icdar13_transfer_duration.py) | Independent CSV word-count replay of a completed duration comparison. | See source/tests |
| [scripts/audit_icdar13_word_targets.py](../scripts/audit_icdar13_word_targets.py) | Independent full-word topology and official Task1 identity audit for IC13 refits. | icdar13-word-refit |
| [scripts/audit_icdar2015_improvement_target_geometry.py](../scripts/audit_icdar2015_improvement_target_geometry.py) | Replay perfect native GT targets against original care boxes on validation. | icdar2015-unet-improvement |
| [scripts/audit_icdar2015_improvement_word_f1.py](../scripts/audit_icdar2015_improvement_word_f1.py) | Independent CSV/SciPy audit of dynamic-resolution validation/TEST word F1. | icdar2015-unet-improvement |
| [scripts/audit_icdar2015_target70.py](../scripts/audit_icdar2015_target70.py) | Independent CSV/SciPy word metrics and complete raw/emitted TEST populations. | icdar2015-target70 |
| [scripts/audit_icdar2015_word_f1.py](../scripts/audit_icdar2015_word_f1.py) | Independent SciPy/CSV replay of the frozen approximate ICDAR2015 word F1. | icdar2015-word-f1 |
| [scripts/audit_icdar2015_word_masks.py](../scripts/audit_icdar2015_word_masks.py) | Full TRAIN word-separation audit of ICDAR2015 and qualified Total-Text masks. | icdar2015-unet-improvement |
| [scripts/audit_icdar_budget_router.py](../scripts/audit_icdar_budget_router.py) | Independent fitting, validation-budget and complete ICDAR TEST route audit. | icdar-trained-budget-router |
| [scripts/audit_image_context_correction.py](../scripts/audit_image_context_correction.py) | Independent count, frozen-policy, context-coverage and OOF provenance audit. | image-context-correction |
| [scripts/audit_lexical_budget_router.py](../scripts/audit_lexical_budget_router.py) | Independent word, partition, head-fit and exact-budget replay checks. | See source/tests |
| [scripts/audit_lightweight_unet_router.py](../scripts/audit_lightweight_unet_router.py) | Independent small-gate fitting-scope, selection, calibration and count replay. | lightweight-unet-router |
| [scripts/audit_lightweight_word_pipeline.py](../scripts/audit_lightweight_word_pipeline.py) | Recount admission/route trials and refit selected lightweight CPU models. | See source/tests |
| [scripts/audit_native128_crop_limit.py](../scripts/audit_native128_crop_limit.py) | Independent SciPy matching and frozen-OCR audit of the cap-only diagnostic. | See source/tests |
| [scripts/audit_native128_review.py](../scripts/audit_native128_review.py) | Independent numerical audit of matched native128 OCR review. | See source/tests |
| [scripts/audit_native128_target_topology.py](../scripts/audit_native128_target_topology.py) | Check actual native128 survival and fragmentation, not full-resolution GT only. | See source/tests |
| [scripts/audit_one_se_fixed_gate.py](../scripts/audit_one_se_fixed_gate.py) | Recompute the cost-aware one-standard-error rule and GT-free deployment. | See source/tests |
| [scripts/audit_paper_caption_comparison.py](../scripts/audit_paper_caption_comparison.py) | Independent count, routing, cache and semantic audit for the caption study. | paper-caption-router-comparison |
| [scripts/audit_risk_admission_gate.py](../scripts/audit_risk_admission_gate.py) | Refit word models and independently replay risk admission, quota and selection. | See source/tests |
| [scripts/audit_selective_word_repair_tests.py](../scripts/audit_selective_word_repair_tests.py) | Independent SciPy replay of selective repair counts, pixels and invariants. | selective-word-repair-tests |
| [scripts/audit_shadow_inventory_router.py](../scripts/audit_shadow_inventory_router.py) | Raw-pixel replay, independent estimator refits and count audit for shadow routing. | shadow-inventory-router |
| [scripts/audit_text_reconstruction_tests.py](../scripts/audit_text_reconstruction_tests.py) | Independently recount saved reconstruction results and validate provenance. | text-reconstruction-tests |
| [scripts/audit_token_context_router.py](../scripts/audit_token_context_router.py) | Independent population, leakage, likelihood and rational-count replay audit. | token-context-router |
| [scripts/audit_totaltext_compact_v2_selected_ocr.py](../scripts/audit_totaltext_compact_v2_selected_ocr.py) | Independent NumPy/SciPy matching and numeric-area audit of selected-v2 OCR. | See source/tests |
| [scripts/audit_trained_router_vocabulary_replay.py](../scripts/audit_trained_router_vocabulary_replay.py) | Independently recount frozen routes and replay every deployed TEST gate. | trained-router-vocabulary-replay |
| [scripts/audit_unet_auxiliary_ocr.py](../scripts/audit_unet_auxiliary_ocr.py) | Read-only numerical/source audit of shared-feature auxiliary OCR experiments. | See source/tests |
| [scripts/audit_unet_corrected_token_router.py](../scripts/audit_unet_corrected_token_router.py) | Independent counts, fold scopes and threshold replay for corrected-U-Net gates. | unet-corrected-token-router |
| [scripts/audit_uniform_checkpoint_validation.py](../scripts/audit_uniform_checkpoint_validation.py) | Independently rank and recount retained-checkpoint validation comparisons. | See source/tests |
| [scripts/audit_uniform_evaluation.py](../scripts/audit_uniform_evaluation.py) | Independently replay every full-TEST uniform gate and word-count result. | See source/tests |
| [scripts/audit_uniform_fallback.py](../scripts/audit_uniform_fallback.py) | Verify prediction-only fallback processing against every qualified TEST cache. | See source/tests |
| [scripts/audit_uniform_fallback_rereads.py](../scripts/audit_uniform_fallback_rereads.py) | Independently verify all real fallback reread crop pixels and input coverage. | See source/tests |
| [scripts/audit_uniform_risk_fusion.py](../scripts/audit_uniform_risk_fusion.py) | Refit word-risk models and recount every saved validation routing trial. | See source/tests |
| [scripts/audit_uniform_unet_training.py](../scripts/audit_uniform_unet_training.py) | Independently check learned states, complete epochs and validation word counts. | uniform-unet-fusion |
| [scripts/audit_uniform_utility_evaluation.py](../scripts/audit_uniform_utility_evaluation.py) | Recount full TEST utility/fusion results and independently rank all images. | See source/tests |
| [scripts/audit_uniform_utility_fusion.py](../scripts/audit_uniform_utility_fusion.py) | Refit compact models and independently recount the common utility study. | See source/tests |
| [scripts/audit_uniform_word_targets.py](../scripts/audit_uniform_word_targets.py) | Audit native word survival without dropping any evaluation care annotation. | See source/tests |
| [scripts/audit_validated_budget_variant.py](../scripts/audit_validated_budget_variant.py) | Independently recount and qualify a validation-only variant selection. | See source/tests |
| [scripts/audit_yolo_context_quality.py](../scripts/audit_yolo_context_quality.py) | Independent arithmetic/source audit of frozen contextual routing artifacts. | See source/tests |
| [scripts/audit_yolo_f1_quality_study.py](../scripts/audit_yolo_f1_quality_study.py) | Independent rational-count audit of the offline F1-target study. | yolo-f1-quality-study |
| [scripts/audit_yolo_frozen_consistency.py](../scripts/audit_yolo_frozen_consistency.py) | Independent count/geometry/threshold audit of completed frozen consistency routing. | yolo-training-free-consistency-router |
| [scripts/audit_yolo_residual_vqa.py](../scripts/audit_yolo_residual_vqa.py) | Independent mask, probability, calibration and routing audit for frozen VQA. | See source/tests |
| [scripts/audit_yolo_router_comparison.py](../scripts/audit_yolo_router_comparison.py) | Independent count/selection replay and readable report, without neural inference. | yolo-paired-router-comparison |
| [scripts/build_experiment_library.py](../scripts/build_experiment_library.py) | Build the English experiment library; optionally refresh verified result summaries. | See source/tests |
| [scripts/build_uniform_multiscale_word_banks.py](../scripts/build_uniform_multiscale_word_banks.py) | Generate/audit matching two-view conditional decoder-heldout observations. | See source/tests |
| [scripts/calibrate_common_parity_threshold.py](../scripts/calibrate_common_parity_threshold.py) | Calibrate common-v1 score cuts on unused CAL, then evaluate frozen cuts on TEST. | common-parity-threshold |
| [scripts/calibrate_contextual_boundary_sensitivity.py](../scripts/calibrate_contextual_boundary_sensitivity.py) | Predeclared CAL-only threshold/objective sensitivity for fixed selectors. | contextual-boundary-selection |
| [scripts/calibrate_icdar2015_target70_operating_policy.py](../scripts/calibrate_icdar2015_target70_operating_policy.py) | Calibrate a predeclared U-Net operating grid on complete fixed TRAIN VAL100. | See source/tests |
| [scripts/calibrate_icdar2015_word_size.py](../scripts/calibrate_icdar2015_word_size.py) | Cached GT-blind component-size calibration on fixed ICDAR2015 VAL100 only. | See source/tests |
| [scripts/calibrate_practical_remap_train.py](../scripts/calibrate_practical_remap_train.py) | Calibrate practical-remap-v1 geometry exclusively on ICDAR13 train-229. | See source/tests |
| [scripts/calibrate_practical_remap_v2_train.py](../scripts/calibrate_practical_remap_v2_train.py) | Calibrate practical-remap-v2 only on the frozen ICDAR13 train-229 split. | See source/tests |
| [scripts/caption_scene.py](../scripts/caption_scene.py) | Generate a caption with an explicitly selected, qualified approximate profile. | detailed-scene-captions |
| [scripts/check_adaspot_joint_ocr_training.py](../scripts/check_adaspot_joint_ocr_training.py) | Fit-only single-word OCR memorization diagnostic after full joint training. | adaspot-joint |
| [scripts/check_deepsolo_environment.py](../scripts/check_deepsolo_environment.py) | Validate the frozen isolated runtime without importing it into the main process. | See source/tests |
| [scripts/check_icdar13_recognition_manifest.py](../scripts/check_icdar13_recognition_manifest.py) | Verify the frozen ICDAR13 recognition population without defining a metric. | See source/tests |
| [scripts/check_main_environment.py](../scripts/check_main_environment.py) | Validate the installed ``context-str-main`` runtime without model downloads. | See source/tests |
| [scripts/check_repository_language.py](../scripts/check_repository_language.py) | Detect known Italian prose/label leftovers; preserve scientific payload languages. | See source/tests |
| [scripts/check_system.py](../scripts/check_system.py) | Report physical-host prerequisites for the reproduction environments. | environments |
| [scripts/check_table5_assets.py](../scripts/check_table5_assets.py) | Offline identity audit for artifacts used by the APPROXIMATE Table 5 smoke. | See source/tests |
| [scripts/cleanup_failed_unet_detection_run.py](../scripts/cleanup_failed_unet_detection_run.py) | Remove only failed epoch-zero copies; keep a verified manifest backup. | unet-restart-storage |
| [scripts/clip_router_sweep_visuals.py](../scripts/clip_router_sweep_visuals.py) | Dependency-free SVG export and offline interactive threshold explorer. | See source/tests |
| [scripts/clip_specialized_worker.py](../scripts/clip_specialized_worker.py) | Standalone native encoder worker; no evaluator/dependency-stack imports. | See source/tests |
| [scripts/collaboration.py](../scripts/collaboration.py) | Check or export a source-only collaboration tree (Python 3.8+, Git). | See source/tests |
| [scripts/compact_completed_uniform_optimizers.py](../scripts/compact_completed_uniform_optimizers.py) | Drop optimizer state only from completed non-improving uniform pilot cases. | See source/tests |
| [scripts/compact_uniform_completed_storage.py](../scripts/compact_uniform_completed_storage.py) | Explicitly compact completed, unselected optimizer states without losing models. | See source/tests |
| [scripts/compare_frozen_quality_routers.py](../scripts/compare_frozen_quality_routers.py) | Post-evaluation paired intervals for every frozen family, without refitting. | See source/tests |
| [scripts/compare_word_multitask.py](../scripts/compare_word_multitask.py) | Compare completed matched branches and reuse historical reports, without training. | word-multitask, word-multitask-results-v1 |
| [scripts/compare_yolo_parseq_four_datasets.py](../scripts/compare_yolo_parseq_four_datasets.py) | Add frozen Total-Text controls and profile YOLO/PARSeq dominant arithmetic. | See source/tests |
| [scripts/complete_clip_router_train_fallback.py](../scripts/complete_clip_router_train_fallback.py) | Complete missing frozen TRAIN DeepSolo branches without refitting any gate. | clip-specialized-routers |
| [scripts/context_str_m6_deepsolo_json_worker.py](../scripts/context_str_m6_deepsolo_json_worker.py) | Path-verifying bootstrap for the staged, repository-owned M6 JSON worker. | See source/tests |
| [scripts/contextual_lm_verifier_runtime.py](../scripts/contextual_lm_verifier_runtime.py) | Frozen, local FLAN-T5 caption/OCR plausibility probes (approximate only). | See source/tests |
| [scripts/crop_clip_runtime.py](../scripts/crop_clip_runtime.py) | Shared hash-checked crop, PARSeq/CLIP runtime utilities. | clip-word-verification |
| [scripts/deduplicate_completed_generator_copies.py](../scripts/deduplicate_completed_generator_copies.py) | Hardlink byte-identical completed historical copies, preserving every path/hash. | See source/tests |
| [scripts/diagnose_adaspot_mask_align.py](../scripts/diagnose_adaspot_mask_align.py) | Frozen full-TEST feature Mask-Align qualification and matched RGB proxy. | adaspot-mask-align |
| [scripts/diagnose_adaspot_phase0.py](../scripts/diagnose_adaspot_phase0.py) | Paired frozen-instance diagnostics; explicitly approximate AdaSpot Phase-0 proxies. | adaspot-phase0 |
| [scripts/diagnose_appearance_remap_train.py](../scripts/diagnose_appearance_remap_train.py) | One configuration, CPU-only TRAIN diagnostics from verified native-mask cache. | appearance-remap-diagnostic |
| [scripts/diagnose_balanced_local_context_train.py](../scripts/diagnose_balanced_local_context_train.py) | TRAIN-only local-context gate on frozen v1 and BALANCED proposal sets. | See source/tests |
| [scripts/diagnose_blip_context_candidate_ranking_train.py](../scripts/diagnose_blip_context_candidate_ranking_train.py) | CPU-only context ranking of frozen TRAIN word-proposal OCR candidates. | See source/tests |
| [scripts/diagnose_contiguous_span_oracle_train.py](../scripts/diagnose_contiguous_span_oracle_train.py) | CPU-only TRAIN feasibility oracle for inference-safe contiguous CC spans. | See source/tests |
| [scripts/diagnose_e2e_parseq_vs_deepsolo.py](../scripts/diagnose_e2e_parseq_vs_deepsolo.py) | Compare frozen E2E PARSeq proposals with raw and reduced DeepSolo outputs. | See source/tests |
| [scripts/diagnose_fixed_expert_oracle.py](../scripts/diagnose_fixed_expert_oracle.py) | Privileged exact-budget diagnostic over a frozen TEST run and cached native fallback. | See source/tests |
| [scripts/diagnose_icdar13_text_word_boxes.py](../scripts/diagnose_icdar13_text_word_boxes.py) | Privileged error diagnosis; never changes the frozen expert or router. | See source/tests |
| [scripts/diagnose_icdar13_traces.py](../scripts/diagnose_icdar13_traces.py) | Validate ICDAR13 traces and emit explicit non-Table-5 diagnostics. | See source/tests |
| [scripts/diagnose_icdar2015_target70_oracle.py](../scripts/diagnose_icdar2015_target70_oracle.py) | Perfect-localization OCR ceiling on fixed TRAIN validation, never TEST. | See source/tests |
| [scripts/diagnose_image_context_failures.py](../scripts/diagnose_image_context_failures.py) | Post-hoc context-specificity, FP-removal and named-case diagnostics; no policy fitting. | image-context-correction |
| [scripts/diagnose_local_context_matching_train.py](../scripts/diagnose_local_context_matching_train.py) | TRAIN-only diagnostic for practical local caption matching. | See source/tests |
| [scripts/diagnose_multihypothesis_word_proposal_oracle_train.py](../scripts/diagnose_multihypothesis_word_proposal_oracle_train.py) | CPU-only TRAIN oracle for multi-hypothesis mask-driven word proposals. | See source/tests |
| [scripts/diagnose_multiscale_component_persistence_train.py](../scripts/diagnose_multiscale_component_persistence_train.py) | CPU-only TRAIN diagnostic for MULTISCALE_COMPONENT_PERSISTENCE_V1. | See source/tests |
| [scripts/diagnose_native_cc_visual.py](../scripts/diagnose_native_cc_visual.py) | First-ten visual and CURRENT-vs-NATIVE_CC diagnostic; no parameter selection. | See source/tests |
| [scripts/diagnose_native_padding_coordinates.py](../scripts/diagnose_native_padding_coordinates.py) | Compare original-coordinate padding with padding at native 128, first10 only. | See source/tests |
| [scripts/diagnose_nn_word_grouping_train.py](../scripts/diagnose_nn_word_grouping_train.py) | Train-only calibration diagnostic for nn-word-grouping-v1. | See source/tests |
| [scripts/diagnose_paperlike_gap_grouping_train.py](../scripts/diagnose_paperlike_gap_grouping_train.py) | TRAIN-only RLSA/Otsu/2-means gap-grouping diagnostic from cached masks. | See source/tests |
| [scripts/diagnose_paperlike_localization_classic_train.py](../scripts/diagnose_paperlike_localization_classic_train.py) | TRAIN-only comparison of classical glyph-mask to word-region diagnostics. | See source/tests |
| [scripts/diagnose_parseq_candidate_ranking_train.py](../scripts/diagnose_parseq_candidate_ranking_train.py) | GPU PARSeq ranking diagnostic for frozen TRAIN multi-hypothesis proposals. | See source/tests |
| [scripts/diagnose_parseq_crop_sensitivity.py](../scripts/diagnose_parseq_crop_sensitivity.py) | Run a non-accuracy PARSeq crop/GT sensitivity grid from frozen trace masks. | See source/tests |
| [scripts/diagnose_parseq_deepsolo_alignment.py](../scripts/diagnose_parseq_deepsolo_alignment.py) | Emit explicit non-accuracy alignment diagnostics from existing trace outputs. | See source/tests |
| [scripts/diagnose_practical_candidate_selection_train.py](../scripts/diagnose_practical_candidate_selection_train.py) | TRAIN-only evaluation of frozen practical word-candidate selectors. | See source/tests |
| [scripts/diagnose_probability_threshold_word_connectivity_train.py](../scripts/diagnose_probability_threshold_word_connectivity_train.py) | TRAIN-only native probability-threshold connectivity diagnostic. | See source/tests |
| [scripts/diagnose_rlsa_original_gap_postsplit_train.py](../scripts/diagnose_rlsa_original_gap_postsplit_train.py) | TRAIN-only original-gap post-split sweep for frozen RLSA run/height=0.25. | See source/tests |
| [scripts/diagnose_router_generalization.py](../scripts/diagnose_router_generalization.py) | Read-only exposure, difficulty, OCR-filter and calibration controls. | router-validation-study |
| [scripts/diagnose_segmentation_target_component_topology.py](../scripts/diagnose_segmentation_target_component_topology.py) | Write the fixed first-ten target/prediction component topology diagnostic. | See source/tests |
| [scripts/diagnose_segmentation_word_topology.py](../scripts/diagnose_segmentation_word_topology.py) | Run first-ten native segmentation topology and conditional OCR diagnostics. | See source/tests |
| [scripts/diagnose_totaltext_original_gt_components_parseq.py](../scripts/diagnose_totaltext_original_gt_components_parseq.py) | Oracle diagnostic: PARSeq on every full-resolution Total-Text GT contour. | See source/tests |
| [scripts/diagnose_totaltext_v4_perfect_mask_localization_train.py](../scripts/diagnose_totaltext_v4_perfect_mask_localization_train.py) | CPU-only TRAIN ceiling for the frozen v4 native-128 contour localizer. | See source/tests |
| [scripts/diagnose_train_gtbox_vs_v1_parseq.py](../scripts/diagnose_train_gtbox_vs_v1_parseq.py) | Full TRAIN GT_BOX-vs-v1 PARSeq ceiling diagnostic; never a Table-5 result. | See source/tests |
| [scripts/diagnose_unet_crop_refinement.py](../scripts/diagnose_unet_crop_refinement.py) | Paired, GT-free APPROXIMATE U-Net crop resegmentation with saved evidence. | unet-crop-refinement |
| [scripts/diagnose_unet_mask_stability.py](../scripts/diagnose_unet_mask_stability.py) | GT-blind uncertainty ranking from saved TRAIN-holdout word probability maps. | See source/tests |
| [scripts/diagnose_uniform_hard_budget_bound.py](../scripts/diagnose_uniform_hard_budget_bound.py) | Privileged fixed-output routing bound; never a deployable image selector. | See source/tests |
| [scripts/diagnose_uniform_reread_validation.py](../scripts/diagnose_uniform_reread_validation.py) | Compare a shared reread grid on fixed native validation routes only. | See source/tests |
| [scripts/diagnose_v1_local_rescue_train.py](../scripts/diagnose_v1_local_rescue_train.py) | Calibrate the v1-local-rescue diagnostic on author-protocol TRAIN only. | See source/tests |
| [scripts/dont_care_polygon_worker.py](../scripts/dont_care_polygon_worker.py) | CPU-only Polygon3 ignore-region intersections; no images or model imports. | See source/tests |
| [scripts/download_synthtext.py](../scripts/download_synthtext.py) | Acquire the author-linked original SynthText archive without package installs. | See source/tests |
| [scripts/evaluate_adaspot_gap_context_final.py](../scripts/evaluate_adaspot_gap_context_final.py) | Post-hoc fixed-final-epoch OCR diagnostic, preserving primary selection/results. | adaspot-gap-context |
| [scripts/evaluate_adaspot_gap_ocr.py](../scripts/evaluate_adaspot_gap_ocr.py) | Unchanged qualified PARSeq on frozen Gap/control and gap-disabled word crops. | See source/tests |
| [scripts/evaluate_adaspot_joint_selective.py](../scripts/evaluate_adaspot_joint_selective.py) | Replay the actual calibrated selective network, independently of OCR caches. | adaspot-joint |
| [scripts/evaluate_cached_totaltext_padding.py](../scripts/evaluate_cached_totaltext_padding.py) | CPU-only, provenance-checked padding diagnostic on frozen Total-Text masks. | See source/tests |
| [scripts/evaluate_center_guided_word_crops.py](../scripts/evaluate_center_guided_word_crops.py) | Paired, frozen two-U-Net masks/vectors -> crops -> optional real PARSeq study. | center-guided-word-crops |
| [scripts/evaluate_crop_group_refinements.py](../scripts/evaluate_crop_group_refinements.py) | Evaluate GT-blind group/reference refinements on frozen crop artifacts. | crop-group-refinements |
| [scripts/evaluate_ctw1500_word_recovery.py](../scripts/evaluate_ctw1500_word_recovery.py) | Rescore frozen CTW TEST outputs as line-supported word occurrences on CPU. | See source/tests |
| [scripts/evaluate_deepsolo_ic13_lexicons.py](../scripts/evaluate_deepsolo_ic13_lexicons.py) | Correct IC13-specific released lexicons; fixed DeepSolo, full233-image word F1. | deepsolo-icdar-lexicons |
| [scripts/evaluate_deepsolo_icdar_lexicon.py](../scripts/evaluate_deepsolo_icdar_lexicon.py) | Calibrate and evaluate fixed DeepSolo outputs with explicit official vocabularies. | deepsolo-icdar-lexicons |
| [scripts/evaluate_deepsolo_lexicon_guard.py](../scripts/evaluate_deepsolo_lexicon_guard.py) | Calibration-frozen generic-lexicon guards on unchanged ICDAR word F1. | deepsolo-icdar-lexicons |
| [scripts/evaluate_deepsolo_short_words.py](../scripts/evaluate_deepsolo_short_words.py) | Reassess frozen DeepSolo profiles after an explicit symmetric length filter. | deepsolo-short-words |
| [scripts/evaluate_deepsolo_word_f1_fusion.py](../scripts/evaluate_deepsolo_word_f1_fusion.py) | TRAIN-selected DeepSolo resolution/fusion policies under two explicit metrics. | deepsolo-word-f1-study |
| [scripts/evaluate_fixed_expert_budget.py](../scripts/evaluate_fixed_expert_budget.py) | Freeze annotation-free TEST routing, then evaluate only selected fallbacks. | See source/tests |
| [scripts/evaluate_fixed_expert_retention.py](../scripts/evaluate_fixed_expert_retention.py) | Route without TEST labels, then recount qualified TT/IC15 fallback outputs. | See source/tests |
| [scripts/evaluate_icdar13_dense_instances.py](../scripts/evaluate_icdar13_dense_instances.py) | Freeze a validation-only lexical policy, then replay the complete U-Net expert. | See source/tests |
| [scripts/evaluate_icdar13_e2e_faithful_vs_balanced.py](../scripts/evaluate_icdar13_e2e_faithful_vs_balanced.py) | Run a resumable ICDAR13 TEST E2E diagnostic for Algorithm1 and balanced mean. | See source/tests |
| [scripts/evaluate_icdar13_hypothesis_fixed_epoch.py](../scripts/evaluate_icdar13_hypothesis_fixed_epoch.py) | Predeclared equal-epoch mixed-data control; never chooses a TEST winner. | See source/tests |
| [scripts/evaluate_icdar13_routing_suite.py](../scripts/evaluate_icdar13_routing_suite.py) | Persistent TEST-233 comparison of frozen localization/context alternatives. | icdar13-routing-audit |
| [scripts/evaluate_icdar13_text_encoder_words.py](../scripts/evaluate_icdar13_text_encoder_words.py) | Select a finite TRAIN-only word-U-Net policy, infer and certify feasibility. | icdar13-unet-dense-instances |
| [scripts/evaluate_icdar2015_improvement_test.py](../scripts/evaluate_icdar2015_improvement_test.py) | TEST500 word F1 for the branch frozen by TRAIN validation100. | icdar2015-unet-improvement |
| [scripts/evaluate_icdar2015_improvement_validation.py](../scripts/evaluate_icdar2015_improvement_validation.py) | Symbol-free spatial word F1 on frozen validation100 segmentation artifacts. | icdar2015-unet-improvement |
| [scripts/evaluate_icdar2015_seed_partition_words.py](../scripts/evaluate_icdar2015_seed_partition_words.py) | Exploratory fixed-VAL word evaluation of all confidence-partition cases. | See source/tests |
| [scripts/evaluate_icdar2015_target70.py](../scripts/evaluate_icdar2015_target70.py) | Frozen approximate ICDAR TEST500 evaluation; all care and emitted outputs count. | icdar2015-target70 |
| [scripts/evaluate_icdar2015_totaltext_transfer_word_f1.py](../scripts/evaluate_icdar2015_totaltext_transfer_word_f1.py) | Frozen TEST500 approximate U-Net/PARSeq word F1, ignoring text symbols. | icdar2015-word-f1 |
| [scripts/evaluate_icdar2015_totaltext_weights_val.py](../scripts/evaluate_icdar2015_totaltext_weights_val.py) | Compare pinned Total-Text weights on complete fixed ICDAR TRAIN VAL100. | See source/tests |
| [scripts/evaluate_kernel_only_padding.py](../scripts/evaluate_kernel_only_padding.py) | Extract a single Kernel output, verify CUDA predictions and compare padding. | kernel-only-padding |
| [scripts/evaluate_lightweight_word_pipeline.py](../scripts/evaluate_lightweight_word_pipeline.py) | Evaluate one validation-frozen lightweight pipeline with exactly61 fallbacks. | See source/tests |
| [scripts/evaluate_native128_crop_limit.py](../scripts/evaluate_native128_crop_limit.py) | APPROXIMATE cap-only OCR ablation from complete frozen native128 contours. | See source/tests |
| [scripts/evaluate_practical_candidate_selection_v2_train.py](../scripts/evaluate_practical_candidate_selection_v2_train.py) | Evaluate anchor-complete proposal selection on frozen ICDAR13 TRAIN artifacts. | See source/tests |
| [scripts/evaluate_practical_remap_test.py](../scripts/evaluate_practical_remap_test.py) | Evaluate train-frozen practical-remap-v1 on test geometry and first-ten OCR. | See source/tests |
| [scripts/evaluate_practical_remap_v2_test.py](../scripts/evaluate_practical_remap_v2_test.py) | Frozen test geometry and new-v2-only PARSeq crops for practical-remap-v2. | See source/tests |
| [scripts/evaluate_probability_threshold_crops_parseq_train.py](../scripts/evaluate_probability_threshold_crops_parseq_train.py) | GPU PARSeq evaluation of frozen TRAIN threshold-sweep crop artifacts. | See source/tests |
| [scripts/evaluate_scene_text_algorithm1_parseq.py](../scripts/evaluate_scene_text_algorithm1_parseq.py) | Recognize frozen native Algorithm 1 crops and score official TEST geometry/OCR. | See source/tests |
| [scripts/evaluate_scene_text_v1_parseq.py](../scripts/evaluate_scene_text_v1_parseq.py) | Run frozen practical-remap-v1 and qualified PARSeq on one official TEST split. | See source/tests |
| [scripts/evaluate_scene_text_yolo_parseq.py](../scripts/evaluate_scene_text_yolo_parseq.py) | Transfer frozen YOLO -> direct PARSeq -> fixed confidence filtering to TEST splits. | See source/tests |
| [scripts/evaluate_supervised_word_boundary_low_thresholds.py](../scripts/evaluate_supervised_word_boundary_low_thresholds.py) | TRAIN-only low-threshold sweep from reconstructed OOF boundary probabilities. | See source/tests |
| [scripts/evaluate_supervised_word_boundary_train.py](../scripts/evaluate_supervised_word_boundary_train.py) | TRAIN-only OOF diagnostic for ``supervised-word-boundary-v1``. | See source/tests |
| [scripts/evaluate_text_recall_crops.py](../scripts/evaluate_text_recall_crops.py) | Frozen text-recall maps -> alternative word crops -> real pinned PARSeq. | text-recall-crops |
| [scripts/evaluate_text_recall_learned_crops.py](../scripts/evaluate_text_recall_learned_crops.py) | Reuse trained kernels/coherent votes to separate the same frozen text masks. | text-recall-crops |
| [scripts/evaluate_text_recall_refinements.py](../scripts/evaluate_text_recall_refinements.py) | Frozen Text+Kernel, native384 and crop-refinement ablations with real PARSeq. | text-recall-refinements |
| [scripts/evaluate_totaltext_bert_correction.py](../scripts/evaluate_totaltext_bert_correction.py) | Prepare, score and evaluate frozen Total-Text v4 PARSeq with genuine BERT. | See source/tests |
| [scripts/evaluate_totaltext_bert_crop_selection.py](../scripts/evaluate_totaltext_bert_crop_selection.py) | Frozen Total-Text crop selection: prepare -> BERT infer -> evaluate -> export. | See source/tests |
| [scripts/evaluate_totaltext_bert_methods.py](../scripts/evaluate_totaltext_bert_methods.py) | Approximate, frozen Total-Text development battery: casing, edits, PLL, scene context. | totaltext-bert-methods-v1 |
| [scripts/evaluate_totaltext_bert_padded_evidence.py](../scripts/evaluate_totaltext_bert_padded_evidence.py) | Separate approximate padding experiment; immutable localization, expanded RGB. | totaltext-bert-visual-correction-v1 |
| [scripts/evaluate_totaltext_bert_visual_correction.py](../scripts/evaluate_totaltext_bert_visual_correction.py) | Six frozen TRAIN-development controls for approximate PARSeq/BERT correction. | See source/tests |
| [scripts/evaluate_totaltext_compact_v2_selected_ocr.py](../scripts/evaluate_totaltext_compact_v2_selected_ocr.py) | APPROXIMATE TEST padding OCR and mask-size diagnostic; numeric outputs only. | See source/tests |
| [scripts/evaluate_totaltext_detector_slots.py](../scripts/evaluate_totaltext_detector_slots.py) | Independent DeepSolo word-slot test over frozen Total-Text PARSeq outputs. | See source/tests |
| [scripts/evaluate_totaltext_frequency_gated_skip_parseq_val.py](../scripts/evaluate_totaltext_frequency_gated_skip_parseq_val.py) | Restartable TRAIN-holdout pad-0 PARSeq check for the 100-epoch frequency skip. | See source/tests |
| [scripts/evaluate_totaltext_highres_contour_padding.py](../scripts/evaluate_totaltext_highres_contour_padding.py) | Restartable TEST-300 contour/padding/PARSeq diagnostic for three native-256 nets. | See source/tests |
| [scripts/evaluate_totaltext_highres_routing.py](../scripts/evaluate_totaltext_highres_routing.py) | Transfer frozen CLIP and paper context routers to cached high-resolution OCR. | See source/tests |
| [scripts/evaluate_totaltext_letterbox_500_words_train.py](../scripts/evaluate_totaltext_letterbox_500_words_train.py) | TRAIN-only word/PARSeq audit of the fixed 500-epoch letterbox branches. | See source/tests |
| [scripts/evaluate_totaltext_letterbox_pad0_test.py](../scripts/evaluate_totaltext_letterbox_pad0_test.py) | Frozen letterbox + pad-0 PARSeq on official Total-Text TEST-300. | See source/tests |
| [scripts/evaluate_totaltext_letterbox_threshold065_test.py](../scripts/evaluate_totaltext_letterbox_threshold065_test.py) | Frozen TRAIN-selected 0.65 threshold on Total-Text TEST; no parameter search. | See source/tests |
| [scripts/evaluate_totaltext_letterbox_threshold_sweep_train.py](../scripts/evaluate_totaltext_letterbox_threshold_sweep_train.py) | Frozen letterbox probability-threshold diagnostic on TRAIN validation only. | See source/tests |
| [scripts/evaluate_totaltext_neuspell.py](../scripts/evaluate_totaltext_neuspell.py) | Approximate frozen Total-Text letterbox -> PARSeq -> spelling-tuned BERT. | See source/tests |
| [scripts/evaluate_totaltext_partial_encoder_parseq_val.py](../scripts/evaluate_totaltext_partial_encoder_parseq_val.py) | Restartable TRAIN-holdout pad0 PARSeq for the isolated partial-encoder branches. | See source/tests |
| [scripts/evaluate_totaltext_text_detectors.py](../scripts/evaluate_totaltext_text_detectors.py) | Frozen Total-Text TEST comparison: CRAFT/DBNet/FAST regions -> genuine PARSeq. | See source/tests |
| [scripts/evaluate_totaltext_train_padding_sweep.py](../scripts/evaluate_totaltext_train_padding_sweep.py) | TRAIN-only native128 padding diagnostic with one frozen mask per image. | See source/tests |
| [scripts/evaluate_totaltext_v1_epoch447_ocr_padding.py](../scripts/evaluate_totaltext_v1_epoch447_ocr_padding.py) | Stream v1 epoch447 padding crops directly to qualified PARSeq; save JSON only. | See source/tests |
| [scripts/evaluate_totaltext_v1_epoch485_cpu_control.py](../scripts/evaluate_totaltext_v1_epoch485_cpu_control.py) | Stream v1 epoch485 control padding crops directly to qualified PARSeq; save JSON only. | See source/tests |
| [scripts/evaluate_totaltext_v1_findcontours_parseq.py](../scripts/evaluate_totaltext_v1_findcontours_parseq.py) | Evaluate a trained Total-Text practical target with native-128 contours + PARSeq. | See source/tests |
| [scripts/evaluate_totaltext_v4_curved_crops.py](../scripts/evaluate_totaltext_v4_curved_crops.py) | Compare tight rectangular and curved-component crops on Total-Text TEST. | See source/tests |
| [scripts/evaluate_totaltext_v4_image_frequency_baseline.py](../scripts/evaluate_totaltext_v4_image_frequency_baseline.py) | Re-evaluate the frozen high-res residual baseline on the identical TRAIN holdout. | See source/tests |
| [scripts/evaluate_totaltext_v4_image_frequency_val_parseq.py](../scripts/evaluate_totaltext_v4_image_frequency_val_parseq.py) | Frozen TRAIN-holdout pad-0 contours and PARSeq for image/frequency ablations. | See source/tests |
| [scripts/evaluate_totaltext_word_ablation_parseq_val.py](../scripts/evaluate_totaltext_word_ablation_parseq_val.py) | Restartable TRAIN-holdout pad0 PARSeq for four word-training ablations. | See source/tests |
| [scripts/evaluate_totaltext_word_f1_separator_parseq_val.py](../scripts/evaluate_totaltext_word_f1_separator_parseq_val.py) | PARSeq on frozen Total-Text TRAIN-holdout crops from two word-F1 runs. | See source/tests |
| [scripts/evaluate_totaltext_word_kernel_parseq_val.py](../scripts/evaluate_totaltext_word_kernel_parseq_val.py) | Restartable TRAIN-holdout pad-0 PARSeq check for the 100-epoch word-kernel model. | See source/tests |
| [scripts/evaluate_totaltext_yolo_direct_parseq.py](../scripts/evaluate_totaltext_yolo_direct_parseq.py) | Recognize frozen YOLO word boxes directly with genuine PARSeq; no group selection. | See source/tests |
| [scripts/evaluate_totaltext_yolo_improvements.py](../scripts/evaluate_totaltext_yolo_improvements.py) | Exploratory TEST battery: OCR-supported filtering, crop views and multiscale YOLO. | See source/tests |
| [scripts/evaluate_totaltext_yolo_scale_agreement.py](../scripts/evaluate_totaltext_yolo_scale_agreement.py) | Cached, exploratory TEST validation of exact reading agreement across640/960. | See source/tests |
| [scripts/evaluate_totaltext_yolo_word_slots.py](../scripts/evaluate_totaltext_yolo_word_slots.py) | Evaluate a pinned YOLO word-object checkpoint as slots for frozen PARSeq crops. | See source/tests |
| [scripts/evaluate_unet_crop_links_ocr.py](../scripts/evaluate_unet_crop_links_ocr.py) | PARSeq on learned-crop exports; original RGB and immutable per-backend cache. | See source/tests |
| [scripts/evaluate_unet_detection_methods.py](../scripts/evaluate_unet_detection_methods.py) | Frozen word instances -> axis/oriented/Bézier RGB -> qualified PARSeq. | See source/tests |
| [scripts/evaluate_unet_seeded_selective_ocr.py](../scripts/evaluate_unet_seeded_selective_ocr.py) | Fixed PARSeq evaluation of the six qualified seeded/selective replay arms. | See source/tests |
| [scripts/evaluate_unet_word_detection_test.py](../scripts/evaluate_unet_word_detection_test.py) | Frozen single U-Net and Hybrid384 detection on Total-Text TEST (APPROXIMATE). | See source/tests |
| [scripts/evaluate_unet_word_research_ocr.py](../scripts/evaluate_unet_word_research_ocr.py) | PARSeq on frozen exported crops, after segmentation models are unloaded. | See source/tests |
| [scripts/evaluate_unet_word_research_v2_ocr.py](../scripts/evaluate_unet_word_research_v2_ocr.py) | PARSeq on corrective exports, with separate cache contracts per execution backend. | See source/tests |
| [scripts/evaluate_uniform_unet_fusion.py](../scripts/evaluate_uniform_unet_fusion.py) | Freeze annotation-free TEST routes, then evaluate all predeclared budgets. | See source/tests |
| [scripts/evaluate_uniform_utility_fusion.py](../scripts/evaluate_uniform_utility_fusion.py) | Freeze all common-utility TEST routes before metric/fallback access. | shared-pipeline-code-tour |
| [scripts/evaluate_v1_local_context_train.py](../scripts/evaluate_v1_local_context_train.py) | CPU-only TRAIN evaluation of whole-caption versus local-caption v1 routing. | See source/tests |
| [scripts/evaluate_v1_rescue_test.py](../scripts/evaluate_v1_rescue_test.py) | Frozen TEST evaluation of practical-remap-v1.1-rescue against v1. | See source/tests |
| [scripts/evaluate_word_centerline_ocr.py](../scripts/evaluate_word_centerline_ocr.py) | Qualify a completed centerline experiment and compare its crops with real PARSeq. | See source/tests |
| [scripts/evaluate_word_multitask_crops.py](../scripts/evaluate_word_multitask_crops.py) | Frozen-map contour/seeded/refined crop comparison with real, cached PARSeq. | word-guided-contours |
| [scripts/evaluate_word_slot_methods.py](../scripts/evaluate_word_slot_methods.py) | Held-out-TRAIN threshold selection and harmonized paired word-slot evaluation. | word-slot-methods |
| [scripts/evaluate_yolo_dont_care.py](../scripts/evaluate_yolo_dont_care.py) | Rescore frozen four-dataset YOLO/PARSeq outputs; ignore regions affect metrics only. | See source/tests |
| [scripts/export_adapted_word_train.py](../scripts/export_adapted_word_train.py) | GT-free expert outputs on TRAIN, or on TEST after gate qualification. | See source/tests |
| [scripts/export_essential_portable_router.py](../scripts/export_essential_portable_router.py) | Export the existing qualified OCR18+mask12 Ridge without refitting. | budget-distilled-router |
| [scripts/export_essential_transfer_deepsolo.py](../scripts/export_essential_transfer_deepsolo.py) | GT-blind, restartable native DeepSolo export on explicitly supplied ICDAR images. | deepsolo-icdar2015-gap, essential-router-icdar-transfer |
| [scripts/export_frozen_quality_tables.py](../scripts/export_frozen_quality_tables.py) | Lossless table export from completed approximate routing JSON; no inference. | See source/tests |
| [scripts/export_ic13_unet_expert.py](../scripts/export_ic13_unet_expert.py) | Export qualified full or decoder-heldout U-Net/PARSeq observations. | icdar13-unet-yolo-procedure |
| [scripts/export_scene_text_algorithm1_crops.py](../scripts/export_scene_text_algorithm1_crops.py) | Export paper-Algorithm-1 approximate crops for one scene-text TEST split. | See source/tests |
| [scripts/export_train_blip2_captions.py](../scripts/export_train_blip2_captions.py) | Export one pinned BLIP-2 caption per frozen ICDAR13 TRAIN image. | See source/tests |
| [scripts/export_train_native_masks.py](../scripts/export_train_native_masks.py) | Export frozen author-protocol TRAIN-229 native masks for CPU diagnostics. | See source/tests |
| [scripts/export_train_native_probabilities.py](../scripts/export_train_native_probabilities.py) | Export frozen epoch-0098 TRAIN-229 native sigmoid probability maps. | See source/tests |
| [scripts/export_train_v1_parseq_inputs.py](../scripts/export_train_v1_parseq_inputs.py) | Persist compact TRAIN practical-remap-v1 crops and PARSeq outputs. | See source/tests |
| [scripts/export_uniform_encoder_context.py](../scripts/export_uniform_encoder_context.py) | Export frozen shared ResNet features and qualify one FIT-only PCA projection. | See source/tests |
| [scripts/export_uniform_fallback_rereads.py](../scripts/export_uniform_fallback_rereads.py) | Prepare real PARSeq fallback-box readings on native TRAIN for all3 domains. | See source/tests |
| [scripts/export_uniform_unet.py](../scripts/export_uniform_unet.py) | Shared validation-only crop-policy selection and RGB-only expert exports. | See source/tests |
| [scripts/export_word_slot_deepsolo.py](../scripts/export_word_slot_deepsolo.py) | Legacy Python3.8 batch export for frozen DeepSolo specialist training labels. | See source/tests |
| [scripts/finalize_graph_occlusion_run.py](../scripts/finalize_graph_occlusion_run.py) | Finalize checksums after the detached graph/occlusion battery terminates. | See source/tests |
| [scripts/finish_icdar13_word_refit.py](../scripts/finish_icdar13_word_refit.py) | Validation-only policy/gate selection followed by complete IC13 TEST evaluation. | icdar13-word-refit |
| [scripts/fit_essential_yolo_head.py](../scripts/fit_essential_yolo_head.py) | Fit one matched30-feature YOLO Ridge on original FIT1028, without TEST fitting. | essential-router-threshold-analysis |
| [scripts/generate_totaltext_bert_captions.py](../scripts/generate_totaltext_bert_captions.py) | Approximate SHA-bound Florence scene context for frozen Total-Text OCR inputs. | See source/tests |
| [scripts/grounded_word_editor.py](../scripts/grounded_word_editor.py) | Visual versus equal-capacity text-only explicit edit transducers. | grounded-word-recovery |
| [scripts/grounded_word_search.py](../scripts/grounded_word_search.py) | Learned versus fixed queries for unread proposals and new tile localizations. | grounded-word-recovery |
| [scripts/icdar_budget_data.py](../scripts/icdar_budget_data.py) | Explicit image populations and annotation-only metric boundaries. | icdar-trained-budget-router |
| [scripts/icdar_budget_stages.py](../scripts/icdar_budget_stages.py) | Frozen experts, native map features and validation-only lightweight heads. | icdar-trained-budget-router |
| [scripts/infer_deepsolo_word_f1.py](../scripts/infer_deepsolo_word_f1.py) | Apply a sealed APPROXIMATE DeepSolo word-F1 policy to new images, without GT. | deepsolo-word-f1-study |
| [scripts/infer_ic13_resolution.py](../scripts/infer_ic13_resolution.py) | Separate detector and PARSeq stages for validation-selected YOLO resolution. | See source/tests |
| [scripts/infer_uniform_word_pipeline.py](../scripts/infer_uniform_word_pipeline.py) | Run the shared approximate word pipeline from an RGB-only JSON manifest. | shared-pipeline-code-tour, uniform-pipeline-current-results, uniform-unet-fusion |
| [scripts/inspect_adaspot_mask_align.py](../scripts/inspect_adaspot_mask_align.py) | Posthoc geometry coverage and fixed-first-eight visualizations, no tuning. | See source/tests |
| [scripts/inspect_deepsolo_candidates.py](../scripts/inspect_deepsolo_candidates.py) | CPU-safe provenance/config identity check; does not qualify model weights. | See source/tests |
| [scripts/manage_uniform_decoder_oof.py](../scripts/manage_uniform_decoder_oof.py) | Freeze and supervise resource-aware decoder-heldout training and exports. | uniform-unet-fusion |
| [scripts/measure_router_weight_storage.py](../scripts/measure_router_weight_storage.py) | Read-only storage/tensor-size comparison; no GPU execution or precision change. | essential-router-budget50 |
| [scripts/neural_macs.py](../scripts/neural_macs.py) | Read-only dense-arithmetic counting, including functional attention projections. | See source/tests |
| [scripts/package_guarded_word_gt.py](../scripts/package_guarded_word_gt.py) | Small measured V1/V2/V3 target comparison; never exports RGB OCR crops. | See source/tests |
| [scripts/package_native128_failure_review.py](../scripts/package_native128_failure_review.py) | Render a few measured diagnostic figures, not an exported OCR crop dataset. | See source/tests |
| [scripts/package_text_recall_refinements_review.py](../scripts/package_text_recall_refinements_review.py) | Offline representative review of completed crop refinements; no inference. | text-recall-refinements |
| [scripts/package_totaltext_adaptive_boxes_visuals.py](../scripts/package_totaltext_adaptive_boxes_visuals.py) | One composite per TRAIN image: RGB, pixel GT, v1, v2 and filled v3. | See source/tests |
| [scripts/package_totaltext_curved_box_v4_overlays.py](../scripts/package_totaltext_curved_box_v4_overlays.py) | Package TRAIN-only pixel-GT/v4 overlays as one composite per image. | See source/tests |
| [scripts/package_totaltext_curved_box_v4_visuals.py](../scripts/package_totaltext_curved_box_v4_visuals.py) | One composite JPEG per TRAIN image: RGB, GT, v1, rejected v3, curved v4. | See source/tests |
| [scripts/package_totaltext_highres_residual_pad0_review.py](../scripts/package_totaltext_highres_residual_pad0_review.py) | One composite PNG per TEST image: RGB, both GT masks, prediction and all pad-0 crops. | See source/tests |
| [scripts/package_totaltext_letterbox_failure_examples.py](../scripts/package_totaltext_letterbox_failure_examples.py) | CPU-only audit and composite examples of frozen Total-Text letterbox failures. | See source/tests |
| [scripts/package_totaltext_no_area_test_visuals.py](../scripts/package_totaltext_no_area_test_visuals.py) | Package exactly one visual audit image per Total-Text TEST sample. | See source/tests |
| [scripts/package_totaltext_original_gt_triple_failures.py](../scripts/package_totaltext_original_gt_triple_failures.py) | Package comparable Total-Text TEST words missed by three GT-oracle OCR paths. | See source/tests |
| [scripts/package_totaltext_refined_word_comparisons.py](../scripts/package_totaltext_refined_word_comparisons.py) | Package the largest TRAIN v1-to-refined changes for visual target inspection. | See source/tests |
| [scripts/package_totaltext_v4_gt_crop_failures.py](../scripts/package_totaltext_v4_gt_crop_failures.py) | Package GT-v4 oracle crops where both PARSeq crop shapes are wrong. | See source/tests |
| [scripts/package_totaltext_v4_resolution256_failures.py](../scripts/package_totaltext_v4_resolution256_failures.py) | Package strict or punctuation-normalized failures of GT-v4 256 OCR. | See source/tests |
| [scripts/package_totaltext_word_region_comparisons.py](../scripts/package_totaltext_word_region_comparisons.py) | Lossless mask pairs plus compact per-image visual comparisons in one ZIP. | See source/tests |
| [scripts/plot_common_pipeline.py](../scripts/plot_common_pipeline.py) | Plot saved common-pipeline results; no fitting or inference. | See source/tests |
| [scripts/plot_essential_thresholds.py](../scripts/plot_essential_thresholds.py) | Standalone scientific F1/fallback frontiers; no model inference or selection. | essential-router-threshold-analysis |
| [scripts/plot_icdar13_hypothesis_tests.py](../scripts/plot_icdar13_hypothesis_tests.py) | Render saved numerical results only, in an existing plotting environment. | See source/tests |
| [scripts/plot_word_slot_methods.py](../scripts/plot_word_slot_methods.py) | Standalone scientific PNG/PDF comparisons; no model imports or inference. | See source/tests |
| [scripts/predict_center_guided_word_crops.py](../scripts/predict_center_guided_word_crops.py) | Experimental fresh two-U-Net inference on one RGB image, without annotations. | center-guided-word-crops |
| [scripts/prepare_ctw1500_recognition.py](../scripts/prepare_ctw1500_recognition.py) | Qualify official CTW1500 end-to-end annotations against frozen images. | See source/tests |
| [scripts/prepare_ctw1500_separated_word_gt.py](../scripts/prepare_ctw1500_separated_word_gt.py) | Export separated CTW1500 TEST word targets from a pinned source archive. | separated-word-gt |
| [scripts/prepare_deepsolo_host.py](../scripts/prepare_deepsolo_host.py) | Create an isolated, non-Docker DeepSolo host runtime from frozen inputs. | deepsolo-runtime |
| [scripts/prepare_fixed_expert_budget.py](../scripts/prepare_fixed_expert_budget.py) | Prepare TRAIN-only fixed-emission cases and authenticated fallback targets. | lightweight-icdar13-90 |
| [scripts/prepare_fixed_expert_retention.py](../scripts/prepare_fixed_expert_retention.py) | Adapt authenticated TT/IC15 experts to the same fixed-emission gate trainer. | See source/tests |
| [scripts/prepare_ic13_complement.py](../scripts/prepare_ic13_complement.py) | Freeze a small U-Net complement to YOLO on selection images only. | budget-target-router |
| [scripts/prepare_ic13_multiscale.py](../scripts/prepare_ic13_multiscale.py) | Validation-only canonical-word fusion of three frozen YOLO/PARSeq resolutions. | See source/tests |
| [scripts/prepare_ic13_slot_budget.py](../scripts/prepare_ic13_slot_budget.py) | Validation-only raw-slot admission and post-correction canonical text NMS. | See source/tests |
| [scripts/prepare_ic13_split_generic.py](../scripts/prepare_ic13_split_generic.py) | TRAIN-only dataset Generic preparation; frozen official TEST resource unchanged. | See source/tests |
| [scripts/prepare_ic13_unet_oof.py](../scripts/prepare_ic13_unet_oof.py) | Freeze conditional U-Net decoder folds using the existing YOLO image splits. | icdar13-unet-yolo-procedure |
| [scripts/prepare_ic13_word_risk.py](../scripts/prepare_ic13_word_risk.py) | Image-group OOF word confidence and discarded-candidate risk, FIT-only training. | budget-target-router |
| [scripts/prepare_icdar13_fold_teachers.py](../scripts/prepare_icdar13_fold_teachers.py) | Prepare and independently audit TRAIN-only Generic teachers for future OOF gates. | icdar13-unet-dense-instances |
| [scripts/prepare_icdar13_word_refit.py](../scripts/prepare_icdar13_word_refit.py) | Audit actual IC13 glyph masks and create isolated connected full-word targets. | icdar13-word-refit |
| [scripts/prepare_icdar2015_synthtext_cache.py](../scripts/prepare_icdar2015_synthtext_cache.py) | Recover the pinned existing SynthText cohort and cache audited v2 targets. | See source/tests |
| [scripts/prepare_icdar2015_word_regions.py](../scripts/prepare_icdar2015_word_regions.py) | Prepare and audit isolated ICDAR2015 TRAIN word masks using Total-Text operations. | icdar2015-totaltext-transfer |
| [scripts/prepare_lightweight_detector_train.py](../scripts/prepare_lightweight_detector_train.py) | TRAIN-only external lightweight detector/PARSeq source qualification. | See source/tests |
| [scripts/prepare_scene_text_dataset.py](../scripts/prepare_scene_text_dataset.py) | Build or verify a checksum-frozen scene-text dataset manifest. | scene-text-datasets |
| [scripts/prepare_scene_text_segmentation.py](../scripts/prepare_scene_text_segmentation.py) | Prepare CTW1500, ICDAR2015, and Total-Text segmentation datasets. | See source/tests |
| [scripts/prepare_selected_deepsolo_export.py](../scripts/prepare_selected_deepsolo_export.py) | Create a GT-free DeepSolo export plan for an already frozen exact image quota. | See source/tests |
| [scripts/prepare_separated_word_gt.py](../scripts/prepare_separated_word_gt.py) | Export isolated, auditable masks of separated words from downloaded GT. | separated-word-gt |
| [scripts/prepare_separated_word_training.py](../scripts/prepare_separated_word_training.py) | Bridge new Total-Text word masks to the existing approximate V4 trainer. | See source/tests |
| [scripts/prepare_tighter_word_gt.py](../scripts/prepare_tighter_word_gt.py) | Generate isolated smaller word GT from the complete separated-v1 delivery. | tighter-word-gt-v2 |
| [scripts/prepare_totaltext_adaptive_word_boxes.py](../scripts/prepare_totaltext_adaptive_word_boxes.py) | Prepare filled adaptive word-box targets from frozen Total-Text TRAIN data. | See source/tests |
| [scripts/prepare_totaltext_curved_box_v4.py](../scripts/prepare_totaltext_curved_box_v4.py) | Prepare the filled curved-boundary v4 Total-Text TRAIN target. | See source/tests |
| [scripts/prepare_totaltext_refined_word_regions.py](../scripts/prepare_totaltext_refined_word_regions.py) | Prepare a tighter, pixel-GT-guided curved-word TRAIN target (isolated v2). | See source/tests |
| [scripts/prepare_totaltext_text_detectors.py](../scripts/prepare_totaltext_text_detectors.py) | Download versioned detector weights, check upstream digests and freeze full SHA256. | See source/tests |
| [scripts/prepare_totaltext_word_regions.py](../scripts/prepare_totaltext_word_regions.py) | Prepare isolated curved-word Total-Text TRAIN targets; never alters source. | See source/tests |
| [scripts/prepare_uniform_encoder_control.py](../scripts/prepare_uniform_encoder_control.py) | Check the upper-encoder arm against the completed frozen-encoder control. | See source/tests |
| [scripts/prepare_uniform_geometric_control.py](../scripts/prepare_uniform_geometric_control.py) | Prepare the shared geometric-augmentation intervention against the old control. | See source/tests |
| [scripts/preview_icdar2015_improvement_masks.py](../scripts/preview_icdar2015_improvement_masks.py) | Render measured target maps with nearest interpolation, without model output. | See source/tests |
| [scripts/preview_icdar2015_improvement_predictions.py](../scripts/preview_icdar2015_improvement_predictions.py) | Compare saved predictions on validation examples with gains and losses. | icdar2015-unet-improvement |
| [scripts/preview_icdar2015_word_transfer.py](../scripts/preview_icdar2015_word_transfer.py) | Render a scientific target comparison; panels are GT, not model predictions. | icdar2015-totaltext-transfer |
| [scripts/probe_detailed_scene_captions.py](../scripts/probe_detailed_scene_captions.py) | GT-blind development and qualification of longer native scene captions. | detailed-scene-captions |
| [scripts/probe_icdar2015_seed_partition.py](../scripts/probe_icdar2015_seed_partition.py) | Fixed-VAL geometry screen of confidence-based single-output mask splitting. | See source/tests |
| [scripts/probe_qwen_batch.py](../scripts/probe_qwen_batch.py) | Qualify native caption batches under the frozen development-selected prompt. | See source/tests |
| [scripts/profile_ag_unet.py](../scripts/profile_ag_unet.py) | Emit the static M8 contract or profile one canonical CUDA forward pass. | ag-unet-implementation-decisions |
| [scripts/profile_blip2.py](../scripts/profile_blip2.py) | Standalone BLIP-2 feasibility profiler; always emits a JSON success/OOM record. | See source/tests |
| [scripts/profile_deepsolo.py](../scripts/profile_deepsolo.py) | Run one isolated candidate load/inference and preserve its JSON response. | See source/tests |
| [scripts/profile_parseq.py](../scripts/profile_parseq.py) | Run the qualified APPROXIMATE PARSeq adapter on one image. | See source/tests |
| [scripts/profile_uniform_unet_compute.py](../scripts/profile_uniform_unet_compute.py) | Count profiler-supported FLOPs of the qualified U-Net on CPU, without GT. | uniform-unet-fusion |
| [scripts/profile_word_center_offsets.py](../scripts/profile_word_center_offsets.py) | Real auxiliary U-Net forward/backward smoke; synthetic input, no trained-result claim. | word-center-offsets |
| [scripts/prune_compact_v2_checkpoints.py](../scripts/prune_compact_v2_checkpoints.py) | Remove only inventoried old compact-v2 weights; preserve masks and run metadata. | See source/tests |
| [scripts/prune_lightweight_unet_router.py](../scripts/prune_lightweight_unet_router.py) | Exploratory validation-only evidence check before retaining extra gate signals. | lightweight-unet-router |
| [scripts/prune_segmentation_checkpoints.py](../scripts/prune_segmentation_checkpoints.py) | Keep only the manifest-selected checkpoint in completed segmentation runs. | See source/tests |
| [scripts/qualify_blip_visual_only.py](../scripts/qualify_blip_visual_only.py) | Compare explicitly pruned native BLIP visual queries against authenticated TRAIN caches. | encoder-family-router |
| [scripts/qualify_corrected_unet_transfer.py](../scripts/qualify_corrected_unet_transfer.py) | Replay the complete prediction-only corrected U-Net on authenticated source RGB. | essential-router-icdar-transfer |
| [scripts/qualify_final_detector_refit.py](../scripts/qualify_final_detector_refit.py) | Qualify a predeclared fixed-epoch detector refit against its frozen gate. | See source/tests |
| [scripts/qualify_guarded_word_dataloader.py](../scripts/qualify_guarded_word_dataloader.py) | Real CPU DataLoader boundary smoke for the opt-in V3 training dataset. | guarded-word-gt-v3 |
| [scripts/qualify_guarded_word_gt.py](../scripts/qualify_guarded_word_gt.py) | Qualify dynamic V3 targets on complete TRAIN, with no training/GPU/crop job. | guarded-word-gt-v3 |
| [scripts/qualify_icdar2015_target70_ocr_device.py](../scripts/qualify_icdar2015_target70_ocr_device.py) | Full fixed-TRAIN-validation CPU/GPU OCR and output-policy equivalence. | icdar2015-target70 |
| [scripts/qualify_mpnet_scoring.py](../scripts/qualify_mpnet_scoring.py) | Qualify the pinned APPROXIMATE MPNet and all named scoring interpretations. | See source/tests |
| [scripts/qualify_text_encoder_legacy.py](../scripts/qualify_text_encoder_legacy.py) | Export only a pinned legacy ResNet50 encoder for cross-environment checks. | See source/tests |
| [scripts/qualify_text_encoder_main.py](../scripts/qualify_text_encoder_main.py) | Verify every converted text-pretrained encoder stage against legacy outputs. | See source/tests |
| [scripts/qualify_totaltext_adaptive_word_boxes.py](../scripts/qualify_totaltext_adaptive_word_boxes.py) | Verify filled v3 TRAIN masks and per-word topology at native 128x128. | See source/tests |
| [scripts/qualify_totaltext_curved_box_v4.py](../scripts/qualify_totaltext_curved_box_v4.py) | Audit curved-box v4 TRAIN targets and native-128 word connectivity. | See source/tests |
| [scripts/qualify_totaltext_refined_word_regions.py](../scripts/qualify_totaltext_refined_word_regions.py) | Audit isolated Total-Text refined TRAIN targets at native 128x128 resolution. | See source/tests |
| [scripts/qualify_uniform_multiscale_masks.py](../scripts/qualify_uniform_multiscale_masks.py) | Blend authenticated native/scaled VAL maps, read fresh crops, then replay. | See source/tests |
| [scripts/qualify_uniform_multiscale_native_dependency.py](../scripts/qualify_uniform_multiscale_native_dependency.py) | Link a cached blend to its original genuine native-neural qualification. | See source/tests |
| [scripts/qualify_uniform_multiscale_validation_banks.py](../scripts/qualify_uniform_multiscale_validation_banks.py) | Bind existing qualified full-FIT validation views to the matching FIT policy. | See source/tests |
| [scripts/qualify_uniform_multiscale_words.py](../scripts/qualify_uniform_multiscale_words.py) | Cached native/second-scale word augmentation and independent VAL replay. | See source/tests |
| [scripts/qualify_uniform_scaled_frontend.py](../scripts/qualify_uniform_scaled_frontend.py) | RGB-only resolution export, native parity and hard-budget validation replay. | See source/tests |
| [scripts/qualify_uniform_shared_multiscale.py](../scripts/qualify_uniform_shared_multiscale.py) | Sequential three-domain VAL screen with a predeclared shared selection rule. | See source/tests |
| [scripts/qualify_word_graph_decoder.py](../scripts/qualify_word_graph_decoder.py) | Calibration-free real-RGB CUDA gradient probe, NOT a spotting result. | word-graph-recovery |
| [scripts/qualify_word_slot_mae_teacher.py](../scripts/qualify_word_slot_mae_teacher.py) | Correct and qualify MAE pretraining inputs before any MAE-supervised fitting. | word-slot-methods |
| [scripts/reassess_word_slot_alphanumeric.py](../scripts/reassess_word_slot_alphanumeric.py) | Rescore saved word-slot experts without symbols and refit TRAIN-only gates. | word-slot-alphanumeric |
| [scripts/recalibrate_fixed_expert_emission.py](../scripts/recalibrate_fixed_expert_emission.py) | Select an existing emission grid using image-excluded TRAIN predictions. | See source/tests |
| [scripts/recheck_icdar13_word_refit.py](../scripts/recheck_icdar13_word_refit.py) | Independent completed IC13 refit recheck, input equivalence and target bounds. | icdar13-word-refit |
| [scripts/recover_ic13_raw_slots.py](../scripts/recover_ic13_raw_slots.py) | Recover discarded low-joint-confidence PARSeq hypotheses on frozen YOLO boxes. | See source/tests |
| [scripts/refit_uniform_multiscale_heads.py](../scripts/refit_uniform_multiscale_heads.py) | Refit one common head family on matching views; never opens TEST outputs. | See source/tests |
| [scripts/render_detailed_captions.py](../scripts/render_detailed_captions.py) | Standalone original-versus-improved caption review with unmodified text. | detailed-scene-captions |
| [scripts/replay_qwen_insertion_controls.py](../scripts/replay_qwen_insertion_controls.py) | Fixed insertion controls on historical Qwen outputs; no new model inference. | text-reconstruction-tests |
| [scripts/replay_router_improved_deepsolo.py](../scripts/replay_router_improved_deepsolo.py) | Approximate frozen-router replay with independently qualified DeepSolo profiles. | router-improved-deepsolo-replay |
| [scripts/replay_totaltext_bert_crop_selection.py](../scripts/replay_totaltext_bert_crop_selection.py) | Reuse genuine frozen BERT scores only for exactly identical observables. | See source/tests |
| [scripts/replay_trained_router_vocabularies.py](../scripts/replay_trained_router_vocabularies.py) | Re-evaluate frozen ICDAR gates with qualified Generic/Weak DeepSolo outputs. | trained-router-vocabulary-replay |
| [scripts/replay_uniform_validation_checkpoints.py](../scripts/replay_uniform_validation_checkpoints.py) | Compare retained neural checkpoints on the complete frozen validation pipeline. | See source/tests |
| [scripts/report_budget_distilled_router.py](../scripts/report_budget_distilled_router.py) | Audited half-budget router comparison, validation cuts and CPU cost evidence. | budget-distilled-router |
| [scripts/report_common_parity_threshold.py](../scripts/report_common_parity_threshold.py) | Export calibrated versus TEST-diagnostic F1/cost tables and scientific figures. | common-parity-threshold |
| [scripts/report_common_pipeline.py](../scripts/report_common_pipeline.py) | Join qualified shared-pipeline experiments without selecting TEST methods. | See source/tests |
| [scripts/report_creative_context_router.py](../scripts/report_creative_context_router.py) | Export all six context methods, validation-only group choices and diagnostics. | creative-context-router |
| [scripts/report_deepsolo_word_f1_profiles.py](../scripts/report_deepsolo_word_f1_profiles.py) | Export the global TRAIN winner and native-only TRAIN winner as explicit profiles. | deepsolo-word-f1-study |
| [scripts/report_detailed_caption_comparison.py](../scripts/report_detailed_caption_comparison.py) | Present complete Qwen/BLIP2/trained-router results without selecting on TEST. | See source/tests |
| [scripts/report_grounded_context_router.py](../scripts/report_grounded_context_router.py) | Descriptive signal diagnostics and scientific plots for an audited frozen run. | grounded-context-router |
| [scripts/report_icdar13_hypothesis_tests.py](../scripts/report_icdar13_hypothesis_tests.py) | Independent fit/count audits and paired summaries for the completed IC13 tests. | See source/tests |
| [scripts/report_icdar13_matched_word_training.py](../scripts/report_icdar13_matched_word_training.py) | Export audited matched-loss curves, counts and historical emission controls. | icdar13-matched-word-training |
| [scripts/report_icdar_budget_router.py](../scripts/report_icdar_budget_router.py) | Export audited ICDAR routing tables, cheap deployment heads and static plots. | icdar-trained-budget-router |
| [scripts/report_router_lexicon_retraining.py](../scripts/report_router_lexicon_retraining.py) | Combine independently audited retrained gates and matched previous controls. | See source/tests |
| [scripts/report_router_validation.py](../scripts/report_router_validation.py) | Export audited validation-calibrated router results and cause controls. | router-validation-study |
| [scripts/report_shadow_inventory_router.py](../scripts/report_shadow_inventory_router.py) | Audited inventory-router tables and fixed-decision paired bootstrap intervals. | shadow-inventory-router |
| [scripts/report_trained_router_vocabulary_replay.py](../scripts/report_trained_router_vocabulary_replay.py) | Readable before/after tables and standalone scientific routing curves. | trained-router-vocabulary-replay |
| [scripts/report_unet_token_router.py](../scripts/report_unet_token_router.py) | Export audited matched U-Net/YOLO router curves without rerunning inference. | unet-token-router |
| [scripts/report_validated_budget_variant.py](../scripts/report_validated_budget_variant.py) | Bind an audited frozen variant selector to independently audited TEST results. | See source/tests |
| [scripts/report_word_slot_methods.py](../scripts/report_word_slot_methods.py) | Independent Hungarian audit and complete fixed-budget word-slot result report. | word-slot-methods |
| [scripts/reread_deepsolo_boxes.py](../scripts/reread_deepsolo_boxes.py) | Read real cached DeepSolo boxes with the already qualified PARSeq model. | lightweight-icdar13-90 |
| [scripts/rescore_ctw1500_v1_parseq.py](../scripts/rescore_ctw1500_v1_parseq.py) | Attach qualified CTW1500 text-line GT to frozen V1/PARSeq outputs. | See source/tests |
| [scripts/rescore_totaltext_highres_padding_symbols.py](../scripts/rescore_totaltext_highres_padding_symbols.py) | CPU-only post-hoc OCR scoring: casefold plus Unicode P/S removal. | See source/tests |
| [scripts/retrain_icdar_lexicon_router.py](../scripts/retrain_icdar_lexicon_router.py) | Retrain native ICDAR gates against Generic / TRAIN-only Weak teachers. | See source/tests |
| [scripts/reuse_image_context_generators.py](../scripts/reuse_image_context_generators.py) | Reuse valid historical controls only after exact input/function equivalence checks. | image-context-correction |
| [scripts/review_native128_masks.py](../scripts/review_native128_masks.py) | Exact data-derived GT/prediction cards and full-TRAIN target-size statistics. | See source/tests |
| [scripts/review_native128_ocr.py](../scripts/review_native128_ocr.py) | APPROXIMATE matched TEST OCR across all retained native128 model families. | See source/tests |
| [scripts/review_paper_captions.py](../scripts/review_paper_captions.py) | Descriptive, GT-blind caption review against the exact Figure 4 scene. | See source/tests |
| [scripts/router_cause_gpu.py](../scripts/router_cause_gpu.py) | Frozen-model crop and missing-word interventions, with raw paired evidence. | router-cause-tests |
| [scripts/router_cause_observations.py](../scripts/router_cause_observations.py) | CPU controls, fixed-budget tests and bounded reporting for cause interventions. | router-cause-tests |
| [scripts/router_cause_training.py](../scripts/router_cause_training.py) | Matched seeds/folds and single-factor context-router interventions. | router-cause-tests |
| [scripts/router_gt_crop_control.py](../scripts/router_gt_crop_control.py) | Approximate GT-crop PARSeq control; never a deployable inference method. | router-validation-study |
| [scripts/router_localization_control.py](../scripts/router_localization_control.py) | Nested one-to-one crop/recognition/filter coverage diagnostics. | router-validation-study |
| [scripts/run_adaspot_gap.py](../scripts/run_adaspot_gap.py) | Resumable, checksum-bound Text+Kernel+Gap and matched-control TRAIN study. | See source/tests |
| [scripts/run_adaspot_gap_context.py](../scripts/run_adaspot_gap_context.py) | Matched 1x1/3x3 Gap probes with TRAIN-only hard-positive weighting. | adaspot-gap-context |
| [scripts/run_adaspot_joint.py](../scripts/run_adaspot_joint.py) | Train and test genuine shared-feature AdaSpot; explicit APPROXIMATE mode. | adaspot-joint |
| [scripts/run_approximate_e2e.py](../scripts/run_approximate_e2e.py) | Run one APPROXIMATE M6 sample with a supplied segmentation mask. | See source/tests |
| [scripts/run_boundary_inventory_router.py](../scripts/run_boundary_inventory_router.py) | TRAIN/VAL-only policy selection for a bounded CRAFT correction of prior routing. | shadow-inventory-router |
| [scripts/run_budget_distilled_router.py](../scripts/run_budget_distilled_router.py) | Approximate DeepSolo-supervised cheap gate; validation choices precede TEST. | budget-distilled-router |
| [scripts/run_clip_high_recall_admission.py](../scripts/run_clip_high_recall_admission.py) | Filter frozen high-recall PARSeq crops; score without labels, then evaluate. | See source/tests |
| [scripts/run_clip_image_router.py](../scripts/run_clip_image_router.py) | Score every Text+Kernel/PARSeq output, calibrate and audit whole-image routing. | clip-image-router |
| [scripts/run_clip_image_router_fallback.py](../scripts/run_clip_image_router_fallback.py) | Execute only whole-image DeepSolo requests from an audited CLIP router run. | clip-image-router |
| [scripts/run_clip_image_router_word_advantage.py](../scripts/run_clip_image_router_word_advantage.py) | Fixed native/CLIP word-versus-background whole-image router. | clip-image-router |
| [scripts/run_clip_innovative_verification.py](../scripts/run_clip_innovative_verification.py) | Extend a completed v1 CLIP run with isolated experimental verification rules. | See source/tests |
| [scripts/run_clip_router_e2e.py](../scripts/run_clip_router_e2e.py) | Frozen TRAIN125/TEST300 CLIP image router and full-scene DeepSolo evaluation. | clip-router-e2e |
| [scripts/run_clip_router_literature_v3.py](../scripts/run_clip_router_literature_v3.py) | Isolated, pinned CLIP-family tests motivated by scene-text literature. | clip-router-literature-models |
| [scripts/run_clip_router_prompt_ablation.py](../scripts/run_clip_router_prompt_ablation.py) | APPROXIMATE CLIP/SigLIP2 prompt/router comparisons on frozen epoch61 OCR. | clip-router-prompt-ablation |
| [scripts/run_clip_specialized_routers.py](../scripts/run_clip_specialized_routers.py) | APPROXIMATE native Long-CLIP/LLM2CLIP routers on the frozen OCR population. | clip-specialized-routers |
| [scripts/run_clip_verification_suite.py](../scripts/run_clip_verification_suite.py) | Download pinned CLIP snapshots, run v1/v2 sequentially, and summarize. | See source/tests |
| [scripts/run_clip_word_verification.py](../scripts/run_clip_word_verification.py) | Pinned-model, fixed-box PARSeq/CLIP/caption TRAIN diagnostic. | clip-word-verification |
| [scripts/run_common_budget_pipeline.py](../scripts/run_common_budget_pipeline.py) | Train one shared APPROXIMATE correction/risk/count-utility procedure. | common-budget-pipeline |
| [scripts/run_contextual_boundary_selection.py](../scripts/run_contextual_boundary_selection.py) | Controlled APPROXIMATE boundary reranking with image-separated OOF proposals. | contextual-boundary-selection |
| [scripts/run_creative_context_router.py](../scripts/run_creative_context_router.py) | Six approximate context-router experiments with immutable frozen base branches. | creative-context-router |
| [scripts/run_crop_group_postfilters.py](../scripts/run_crop_group_postfilters.py) | Filter frozen crop-group outputs without rerunning segmentation, PARSeq or CLIP. | See source/tests |
| [scripts/run_crop_group_word_selection.py](../scripts/run_crop_group_word_selection.py) | Run dataset-independent selectors on frozen crop-group artifacts, then evaluate. | See source/tests |
| [scripts/run_deepsolo_textocr_comparison.py](../scripts/run_deepsolo_textocr_comparison.py) | Paired APPROXIMATE DeepSolo evaluation; never retune a router or replace old caches. | deepsolo-textocr-ic13-comparison |
| [scripts/run_deepsolo_word_f1_study.py](../scripts/run_deepsolo_word_f1_study.py) | APPROXIMATE low-floor/resolution study, using separate existing Python runtimes. | deepsolo-word-f1-study |
| [scripts/run_detailed_caption_comparison.py](../scripts/run_detailed_caption_comparison.py) | Explicit Qwen caption ablation with the frozen common router comparison. | detailed-scene-captions |
| [scripts/run_directional_affinity_split.py](../scripts/run_directional_affinity_split.py) | APPROXIMATE learned-affinity instance decoder; CAL cross-validation gates TEST. | directional-transport-results |
| [scripts/run_directional_closedness.py](../scripts/run_directional_closedness.py) | APPROXIMATE CAL-qualified ownership-closedness readout; reject negative pilots. | directional-transport-results |
| [scripts/run_directional_transport.py](../scripts/run_directional_transport.py) | APPROXIMATE parameter-matched decoder study with image-only native PARSeq. | directional-transport-results, directional-transport-unet |
| [scripts/run_encoder_family_router.py](../scripts/run_encoder_family_router.py) | Train matched CLIP/BLIP/combined approximate image-deferral heads. | encoder-family-router |
| [scripts/run_essential_icdar_transfer.py](../scripts/run_essential_icdar_transfer.py) | Four-way ICDAR transfer of the frozen essential gate, with explicit stage caches. | essential-router-icdar-transfer |
| [scripts/run_essential_router_budget.py](../scripts/run_essential_router_budget.py) | APPROXIMATE budget-specific small-head selection and expert-unseen calibration. | essential-router-budget50 |
| [scripts/run_graph_occlusion_followups.py](../scripts/run_graph_occlusion_followups.py) | Four controlled APPROXIMATE follow-ups; all choices use calibration word F1. | graph-occlusion-followups |
| [scripts/run_graph_occlusion_research.py](../scripts/run_graph_occlusion_research.py) | Controlled fragment GNN and clean-to-occluded U-Net pilots on explicit CUDA. | graph-occlusion-research |
| [scripts/run_grounded_context_router.py](../scripts/run_grounded_context_router.py) | Approximate grounded-context ablations, frozen base branches and TRAIN-only fit. | grounded-context-router |
| [scripts/run_grounded_word_recovery.py](../scripts/run_grounded_word_recovery.py) | Three controlled APPROXIMATE recovery experiments; immutable-source execution. | grounded-word-recovery |
| [scripts/run_guarded_v3_graph_cut.py](../scripts/run_guarded_v3_graph_cut.py) | Frozen V3/447: TRAIN-calibrated graph-cut vs matched geodesic and TEST OCR. | guarded-v3-graph-cut |
| [scripts/run_guarded_v3_training_ocr.py](../scripts/run_guarded_v3_training_ocr.py) | Durable sequential V3 training -> selected-checkpoint PARSeq -> independent audit. | See source/tests |
| [scripts/run_icdar13_dense_instances.py](../scripts/run_icdar13_dense_instances.py) | Prepare, cache, train and qualify an approximate U-Net center/extent decoder. | icdar13-unet-dense-instances |
| [scripts/run_icdar13_essential_word_heuristics.py](../scripts/run_icdar13_essential_word_heuristics.py) | Freeze a small validation-selected gate before scoring CAL and complete TEST. | See source/tests |
| [scripts/run_icdar13_hypothesis_tests.py](../scripts/run_icdar13_hypothesis_tests.py) | Matched APPROXIMATE vocabulary, crop, cross-fitting and TT-replay experiments. | icdar13-hypothesis-tests |
| [scripts/run_icdar13_matched_word_training.py](../scripts/run_icdar13_matched_word_training.py) | Run frozen, matched ICDAR13 loss controls and selection-only PARSeq policies. | icdar13-matched-word-training |
| [scripts/run_icdar13_trace_batch.py](../scripts/run_icdar13_trace_batch.py) | Collect resumable ICDAR13 APPROXIMATE traces without computing Table 5 accuracy. | See source/tests |
| [scripts/run_icdar13_transfer_duration.py](../scripts/run_icdar13_transfer_duration.py) | Compare fixed Total-Text transfer and IC13/mixed training durations. | icdar13-transfer-duration |
| [scripts/run_icdar2015_improvement_evaluation.py](../scripts/run_icdar2015_improvement_evaluation.py) | Sequential validation/selection/TEST after the matched training job exits. | icdar2015-unet-improvement |
| [scripts/run_image_context_correction.py](../scripts/run_image_context_correction.py) | Controlled APPROXIMATE image-wide OCR correction battery on frozen observations. | image-context-correction |
| [scripts/run_image_context_joint_policy.py](../scripts/run_image_context_joint_policy.py) | Predeclared CAL-only composition of frozen image-context correction and FP filtering. | image-context-correction |
| [scripts/run_legacy_global_context.py](../scripts/run_legacy_global_context.py) | Add image-wide context to the previously qualified word-LM feature baseline. | image-context-correction |
| [scripts/run_lexical_budget_router.py](../scripts/run_lexical_budget_router.py) | Select conservative light edits, fit target-utility heads and freeze budgets. | budget-target-router |
| [scripts/run_lightweight_unet_router.py](../scripts/run_lightweight_unet_router.py) | APPROXIMATE feature-cost ablations and calibration of a frozen small U-Net gate. | lightweight-unet-router |
| [scripts/run_lightweight_word_selection.py](../scripts/run_lightweight_word_selection.py) | TRAIN-only admission experiment for a lightweight fixed-budget OCR pipeline. | lightweight-icdar13-90 |
| [scripts/run_mask_crop_budget.py](../scripts/run_mask_crop_budget.py) | Qualified fixed-PARSeq budget ablation for the TRAIN-selected Guarded V3. | mask-crop-three-tests |
| [scripts/run_mask_crop_learned_tests.py](../scripts/run_mask_crop_learned_tests.py) | Isolated split/merge selection and recognition-supervised crop refinement. | mask-crop-three-tests |
| [scripts/run_mask_probability_refiner.py](../scripts/run_mask_probability_refiner.py) | Controlled full-image residual refinement battery; APPROXIMATE mode only. | mask-probability-refiner |
| [scripts/run_occlusion_four_tests.py](../scripts/run_occlusion_four_tests.py) | Four APPROXIMATE component experiments; frozen segmentation/PARSeq, TRAIN-only selection. | occlusion-four-tests |
| [scripts/run_paper_caption_comparison.py](../scripts/run_paper_caption_comparison.py) | BLIP-2/MPNet/fuzzy vs qualified trained routers, three complete benchmarks. | paper-caption-router-comparison |
| [scripts/run_paper_image_router_baseline.py](../scripts/run_paper_image_router_baseline.py) | Staged BLIP2/MPNet whole-image baseline against frozen CLIP/PARSeq outputs. | paper-image-router-baseline |
| [scripts/run_router_cause_tests.py](../scripts/run_router_cause_tests.py) | Predeclared approximate tests of five context-router failure hypotheses. | router-cause-tests |
| [scripts/run_router_validation_study.py](../scripts/run_router_validation_study.py) | Approximate matched router validation without refitting after calibration. | router-validation-study |
| [scripts/run_selective_word_repair_tests.py](../scripts/run_selective_word_repair_tests.py) | Four isolated APPROXIMATE selective word-repair tests on qualified frozen inputs. | selective-word-repair-tests |
| [scripts/run_shadow_inventory_router.py](../scripts/run_shadow_inventory_router.py) | Frozen CRAFT inventory and active-region audits for approximate image routing. | shadow-inventory-router |
| [scripts/run_table5_smoke.py](../scripts/run_table5_smoke.py) | Run one real-data APPROXIMATE downstream smoke with frozen segmentation. | reproduction-plan |
| [scripts/run_teacher_student_tests.py](../scripts/run_teacher_student_tests.py) | Five controlled APPROXIMATE pilots; explicit artifacts, fit-only updates and word F1. | teacher-student-tests |
| [scripts/run_teacher_student_yolo.py](../scripts/run_teacher_student_yolo.py) | Isolated, already-installed Ultralytics 8.1.0 student training/inference. | teacher-student-tests |
| [scripts/run_text_recall_resolution.py](../scripts/run_text_recall_resolution.py) | Matched text coverage experiments: loss x input resolution, APPROXIMATE only. | text-recall-resolution |
| [scripts/run_text_reconstruction_tests.py](../scripts/run_text_reconstruction_tests.py) | Three APPROXIMATE frozen-OCR tests; explicit artifacts and TRAIN-only selection. | text-reconstruction-tests |
| [scripts/run_token_context_router.py](../scripts/run_token_context_router.py) | Full-TRAIN approximate token-context, transcript and dense-support routers. | token-context-router |
| [scripts/run_totaltext_crop_clip_battery.py](../scripts/run_totaltext_crop_clip_battery.py) | One-command Total-Text TRAIN crop/PARSeq/CLIP battery (approximate). | See source/tests |
| [scripts/run_totaltext_crop_group_test_suite.py](../scripts/run_totaltext_crop_group_test_suite.py) | Official TEST-300: frozen threshold/window crops, PARSeq, groups, 14 selectors. | See source/tests |
| [scripts/run_totaltext_crop_recovery_suite.py](../scripts/run_totaltext_crop_recovery_suite.py) | Run all GT-blind crop generators, PARSeq and post-hoc normalized oracle recall. | See source/tests |
| [scripts/run_trainable_image_router.py](../scripts/run_trainable_image_router.py) | Train and evaluate approximate image routers against a fixed-budget GT oracle. | yolo-trainable-router |
| [scripts/run_unet_auxiliary_ocr.py](../scripts/run_unet_auxiliary_ocr.py) | Matched detached/connected auxiliary CTC supervision of the Text+Kernel U-Net. | unet-auxiliary-ocr |
| [scripts/run_unet_cold_light_router.py](../scripts/run_unet_cold_light_router.py) | Exploratory fallback fitting on the 125 images excluded from light-model gradients. | unet-minimum-fallback |
| [scripts/run_unet_corrected_token_router.py](../scripts/run_unet_corrected_token_router.py) | APPROXIMATE corrected U-Net token gate; fixed experts, newly fitted routing. | unet-corrected-token-router |
| [scripts/run_unet_crop_links.py](../scripts/run_unet_crop_links.py) | Isolated learned second-pass experiment on frozen TRAIN predictions. | See source/tests |
| [scripts/run_unet_detection_methods.py](../scripts/run_unet_detection_methods.py) | Resumable APPROXIMATE text pretraining/fusion/progressive-kernel experiments. | See source/tests |
| [scripts/run_unet_minimum_fallback.py](../scripts/run_unet_minimum_fallback.py) | APPROXIMATE U-Net/corrector to DeepSolo routing at minimum development cost. | unet-minimum-fallback |
| [scripts/run_unet_seeded_selective.py](../scripts/run_unet_seeded_selective.py) | Replay qualified crop predictions with seeded links and a GT-free gate. | See source/tests |
| [scripts/run_unet_token_router.py](../scripts/run_unet_token_router.py) | Train matched U-Net/YOLO token routers with the qualified improved fallback. | unet-token-router |
| [scripts/run_unet_word_research.py](../scripts/run_unet_word_research.py) | Matched, resumable APPROXIMATE glyph/word and auxiliary-head experiments. | See source/tests |
| [scripts/run_unet_word_research_v2.py](../scripts/run_unet_word_research_v2.py) | Matched, resumable APPROXIMATE corrective word experiments. | See source/tests |
| [scripts/run_unet_word_research_v2_postprocess.py](../scripts/run_unet_word_research_v2_postprocess.py) | Three GT-blind extraction methods on the SAME trained v1 embedding maps. | See source/tests |
| [scripts/run_uniform_photometric_pilot.py](../scripts/run_uniform_photometric_pilot.py) | Prepare matched continuations and qualify the IC13 pilot before other domains. | See source/tests |
| [scripts/run_uniform_unet_training.py](../scripts/run_uniform_unet_training.py) | Prepare/retrain identical Text+Kernel U-Nets on three explicit TRAIN splits. | shared-pipeline-code-tour, uniform-unet-fusion |
| [scripts/run_word_centerline.py](../scripts/run_word_centerline.py) | Resumable APPROXIMATE frozen-U-Net centerline/radius experiment. | See source/tests |
| [scripts/run_word_evidence_graphs.py](../scripts/run_word_evidence_graphs.py) | Isolated APPROXIMATE typed/ownership/character/active-observation experiments. | word-evidence-graphs |
| [scripts/run_word_evidence_suite.py](../scripts/run_word_evidence_suite.py) | Three isolated APPROXIMATE word-evidence tests with persistent phase receipts. | word-evidence-suite |
| [scripts/run_word_graph_recovery.py](../scripts/run_word_graph_recovery.py) | Four isolated APPROXIMATE word-F1 experiments, with fixed qualified inputs. | word-graph-recovery |
| [scripts/run_word_instance_extensions.py](../scripts/run_word_instance_extensions.py) | Three isolated local-word ablations, with matched controls and TRAIN calibration. | word-instance-extensions |
| [scripts/run_word_multitask.py](../scripts/run_word_multitask.py) | Isolated, restartable matched word-head ablation; reuses existing training I/O. | word-multitask, word-multitask-results-v1 |
| [scripts/run_word_slot_methods.py](../scripts/run_word_slot_methods.py) | Versioned APPROXIMATE multi-scale word-slot training and paired OCR battery. | word-slot-methods |
| [scripts/run_word_slot_specialist.py](../scripts/run_word_slot_specialist.py) | Run a completed specialist and a fitted image/word router on a new RGB file. | word-slot-methods |
| [scripts/run_yolo_context_quality.py](../scripts/run_yolo_context_quality.py) | APPROXIMATE contextual image-quality probes; frozen models and TRAIN-only cuts. | yolo-context-quality-router |
| [scripts/run_yolo_coverage_resolution.py](../scripts/run_yolo_coverage_resolution.py) | APPROXIMATE text-completeness gate and paired YOLO640/960/1280 experiments. | yolo-text-coverage-resolution |
| [scripts/run_yolo_f1_quality_study.py](../scripts/run_yolo_f1_quality_study.py) | Approximate offline quality-target study: frozen OCR/DeepSolo, no training. | yolo-f1-quality-study |
| [scripts/run_yolo_frozen_consistency.py](../scripts/run_yolo_frozen_consistency.py) | APPROXIMATE frozen-model routing: scale/crop stability, coverage and CLIP controls. | yolo-training-free-consistency-router |
| [scripts/run_yolo_residual_cost_router.py](../scripts/run_yolo_residual_cost_router.py) | Offline approximate residual-region/error-cost router with frozen DeepSolo. | yolo-residual-cost-router |
| [scripts/run_yolo_residual_vqa.py](../scripts/run_yolo_residual_vqa.py) | APPROXIMATE frozen BLIP2 missing-text probes with TRAIN-only image routing. | yolo-residual-text-vqa |
| [scripts/run_yolo_router_comparison.py](../scripts/run_yolo_router_comparison.py) | APPROXIMATE paired YOLO/PARSeq: fixed CLIP versus section3.5 router. | yolo-paired-router-comparison |
| [scripts/screen_icdar2015_unet_resolutions.py](../scripts/screen_icdar2015_unet_resolutions.py) | Frozen-weight resolution screening on the fixed TRAIN validation100 only. | icdar2015-unet-improvement |
| [scripts/select_adaspot_gap_ocr.py](../scripts/select_adaspot_gap_ocr.py) | Calibrate deployable Gap checkpoints against actual word OCR, TRAIN-only. | See source/tests |
| [scripts/select_fallback_reread_policy.py](../scripts/select_fallback_reread_policy.py) | Choose one low-cost fallback transcript replacement on TRAIN validation. | See source/tests |
| [scripts/select_fallback_word_fusion.py](../scripts/select_fallback_word_fusion.py) | Select a no-extra-network fallback transcript fusion on TRAIN/validation. | See source/tests |
| [scripts/select_ic13_resolution.py](../scripts/select_ic13_resolution.py) | Freeze selection-only resolution/admission, then stage the stronger light expert. | See source/tests |
| [scripts/select_icdar13_yolo_word_expert.py](../scripts/select_icdar13_yolo_word_expert.py) | Choose an adapted word detector checkpoint/crop policy on TRAIN validation. | See source/tests |
| [scripts/select_icdar2015_improvement.py](../scripts/select_icdar2015_improvement.py) | Freeze one completed branch using validation correct-word F1, before TEST. | icdar2015-unet-improvement |
| [scripts/select_icdar2015_target70.py](../scripts/select_icdar2015_target70.py) | Freeze the completed source/long/native comparison without reading TEST. | icdar2015-target70 |
| [scripts/select_one_se_fixed_gate.py](../scripts/select_one_se_fixed_gate.py) | Cost-aware one-standard-error selection over independently audited TRAIN trials. | See source/tests |
| [scripts/select_risk_admission_gate.py](../scripts/select_risk_admission_gate.py) | Select GT-free word-risk admission and fixed-budget routing on TRAIN/validation. | See source/tests |
| [scripts/select_uniform_lexical_policy.py](../scripts/select_uniform_lexical_policy.py) | TRAIN-validation-only shared emission/lexicon control for the trained U-Nets. | uniform-unet-fusion |
| [scripts/select_validated_budget_variant.py](../scripts/select_validated_budget_variant.py) | Select an approximate budget pipeline using only frozen validation counts. | See source/tests |
| [scripts/study_icdar2015_single_unet_multiview.py](../scripts/study_icdar2015_single_unet_multiview.py) | Fixed-VAL multiscale/flip study of one frozen U-Net; never reads TEST. | See source/tests |
| [scripts/summarize_clip_router_prompt_ablation.py](../scripts/summarize_clip_router_prompt_ablation.py) | Summarize sealed prompt experiments; choose model only by TRAIN45 evidence. | clip-router-prompt-ablation |
| [scripts/summarize_scene_text_v1_parseq_suite.py](../scripts/summarize_scene_text_v1_parseq_suite.py) | Collect the six frozen scene-text V1+PARSeq TEST diagnostics. | See source/tests |
| [scripts/summarize_totaltext_partial_encoder.py](../scripts/summarize_totaltext_partial_encoder.py) | Read-only comparison of the two completed held-out TRAIN diagnostics. | See source/tests |
| [scripts/summarize_totaltext_word_ablation.py](../scripts/summarize_totaltext_word_ablation.py) | Paired TRAIN-holdout comparisons; never reads TEST or selects thresholds. | See source/tests |
| [scripts/summarize_unet_crop_links.py](../scripts/summarize_unet_crop_links.py) | Audit four completed arms and compare exact PARSeq word identities. | See source/tests |
| [scripts/summarize_unet_seeded_selective.py](../scripts/summarize_unet_seeded_selective.py) | Report geometry, recognition and matched ablations without model promotion. | See source/tests |
| [scripts/summarize_unet_word_research.py](../scripts/summarize_unet_word_research.py) | Compare completed heads and per-word OCR gains/losses without promotion. | See source/tests |
| [scripts/summarize_unet_word_research_v2.py](../scripts/summarize_unet_word_research_v2.py) | Compare corrective experiments; preserve frozen and continued baselines. | See source/tests |
| [scripts/supervise_ic13_unet_oof.py](../scripts/supervise_ic13_unet_oof.py) | Run the frozen U-Net folds sequentially, then audit and export each fold. | icdar13-unet-yolo-procedure |
| [scripts/supervise_uniform_pipeline_smoke.py](../scripts/supervise_uniform_pipeline_smoke.py) | Qualify the RGB-only chain on three small batches without stopping other jobs. | See source/tests |
| [scripts/supervise_uniform_unet_posttraining.py](../scripts/supervise_uniform_unet_posttraining.py) | Qualify common emissions and word-risk/fusion gates as each U-Net completes. | See source/tests |
| [scripts/supervise_uniform_unet_training.py](../scripts/supervise_uniform_unet_training.py) | Resource-aware sequential smoke/audit/full training; never interrupts other jobs. | uniform-unet-fusion |
| [scripts/supervise_uniform_utility_oof.py](../scripts/supervise_uniform_utility_oof.py) | After all neural banks qualify, refit/audit the same common v3 gate on CPU. | uniform-unet-fusion |
| [scripts/sweep_clip_router_e2e.py](../scripts/sweep_clip_router_e2e.py) | Descriptive threshold sweep of the frozen B16 image router, separate splits. | clip-router-prompt-ablation, clip-router-threshold-sweep |
| [scripts/sweep_totaltext_v4_gt_mask_resolution.py](../scripts/sweep_totaltext_v4_gt_mask_resolution.py) | GT-v4 target-resolution sensitivity, with frozen native-128 control. | See source/tests |
| [scripts/sweep_totaltext_v4_predicted_mask_resolution.py](../scripts/sweep_totaltext_v4_predicted_mask_resolution.py) | TEST-300 PARSeq diagnostic from frozen v4 predicted binary masks. | See source/tests |
| [scripts/token_context_data.py](../scripts/token_context_data.py) | Checksum-bound full-TRAIN inputs and frozen YOLO/PARSeq/DeepSolo branches. | token-context-router |
| [scripts/token_context_training.py](../scripts/token_context_training.py) | Fold-only auxiliary support fitting and paired-benefit context-router training. | token-context-router |
| [scripts/train_adaspot_gap_decoder.py](../scripts/train_adaspot_gap_decoder.py) | Train a genuine Gap-specific attention decoder, keeping Text/Kernel immutable. | adaspot-gap-decoder |
| [scripts/train_adaspot_gap_fulltrain.py](../scripts/train_adaspot_gap_fulltrain.py) | Full official TRAIN Gap refit; frozen prior choices, separate TEST evaluation. | adaspot-gap-fulltrain |
| [scripts/train_adaspot_gap_normalized.py](../scripts/train_adaspot_gap_normalized.py) | Fit/decision-calibrate/evaluate Gap heads without using evaluation GT for choices. | adaspot-gap-normalized |
| [scripts/train_adaspot_gap_probe.py](../scripts/train_adaspot_gap_probe.py) | Balanced frozen-feature Gap probe, fully isolated from qualified M/K weights. | See source/tests |
| [scripts/train_crossfit_fixed_gate.py](../scripts/train_crossfit_fixed_gate.py) | Select a fixed-emission image gate using FIT cross-fitting and validation. | See source/tests |
| [scripts/train_fixed_expert_budget.py](../scripts/train_fixed_expert_budget.py) | Train common word-risk/count gates without modifying fixed expert emissions. | See source/tests |
| [scripts/train_guarded_v3.py](../scripts/train_guarded_v3.py) | APPROXIMATE V3 training with the V2 split/protocol and bounded checkpoint retention. | See source/tests |
| [scripts/train_ic13_fulltrain_oof.py](../scripts/train_ic13_fulltrain_oof.py) | Image-group OOF head selection, followed by a full TRAIN fit and fixed-budget test. | budget-target-router |
| [scripts/train_icdar13_matched_router_generic.py](../scripts/train_icdar13_matched_router_generic.py) | Fit a NEW small native U-Net gate against the qualified90.17% Generic expert. | icdar13-matched-router-generic |
| [scripts/train_icdar13_text_encoder_words.py](../scripts/train_icdar13_text_encoder_words.py) | Matched word-U-Net adaptation with ImageNet versus text-pretrained encoders. | icdar13-unet-dense-instances |
| [scripts/train_icdar13_word_refit.py](../scripts/train_icdar13_word_refit.py) | Train isolated IC13 full-word/kernel controls; all selection remains on TRAIN validation. | icdar13-word-refit, shared-pipeline-code-tour |
| [scripts/train_icdar13_yolo_words.py](../scripts/train_icdar13_yolo_words.py) | Controlled head-only/full YOLO word-box adaptation, TRAIN/validation only. | See source/tests |
| [scripts/train_icdar2015_encoder_decay.py](../scripts/train_icdar2015_encoder_decay.py) | Matched encoder-L2 continuation; fixed TRAIN validation selects raw words. | See source/tests |
| [scripts/train_icdar2015_synthetic.py](../scripts/train_icdar2015_synthetic.py) | Synthetic text pretraining and fixed-split ICDAR fine-tuning with real AdamW. | See source/tests |
| [scripts/train_icdar2015_target70.py](../scripts/train_icdar2015_target70.py) | Long approximate ICDAR training; fixed TRAIN validation selects actual words. | icdar2015-target70 |
| [scripts/train_icdar2015_target70_ohem.py](../scripts/train_icdar2015_target70_ohem.py) | OHEM/full-encoder continuation; fixed TRAIN validation selects raw correct words. | See source/tests |
| [scripts/train_icdar2015_totaltext_transfer.py](../scripts/train_icdar2015_totaltext_transfer.py) | Train the isolated approximate ICDAR2015 U-Net transfer; never load TEST/OCR. | icdar2015-totaltext-transfer, icdar2015-word-f1 |
| [scripts/train_icdar2015_totaltext_weights.py](../scripts/train_icdar2015_totaltext_weights.py) | Explicit Total-Text weight warm starts; fixed ICDAR TRAIN word selection. | See source/tests |
| [scripts/train_icdar2015_unet_improvement.py](../scripts/train_icdar2015_unet_improvement.py) | Matched approximate continuations; selection uses fixed TRAIN validation only. | icdar2015-unet-improvement |
| [scripts/train_icdar_budget_router.py](../scripts/train_icdar_budget_router.py) | Approximate per-dataset YOLO/U-Net gates with separate selection/calibration. | icdar-trained-budget-router |
| [scripts/train_kernel_only.py](../scripts/train_kernel_only.py) | Resumable CUDA fine-tuning supervised only by K, then fixed-padding OCR. | kernel-only-training |
| [scripts/train_kernel_only_fresh.py](../scripts/train_kernel_only_fresh.py) | New Kernel U-Net, fresh decoder/head, local ImageNet encoder only. | kernel-only-fresh |
| [scripts/train_lightweight_word_router.py](../scripts/train_lightweight_word_router.py) | Freeze admission, conditional fallback complement and a pre-fallback gate. | See source/tests |
| [scripts/train_totaltext_compact_v2_word_selection.py](../scripts/train_totaltext_compact_v2_word_selection.py) | Restart compact-v2 with held-out word metrics and at most four checkpoint files. | See source/tests |
| [scripts/train_totaltext_frequency_gated_skip_100ep.py](../scripts/train_totaltext_frequency_gated_skip_100ep.py) | TRAIN-only 100-epoch feature-wavelet gated-skip Total-Text ablation. | See source/tests |
| [scripts/train_totaltext_letterbox_500_separator_pair.py](../scripts/train_totaltext_letterbox_500_separator_pair.py) | Fork one frozen letterbox checkpoint into two isolated 500-epoch TRAIN runs. | See source/tests |
| [scripts/train_totaltext_letterbox_word_f1_separator_head.py](../scripts/train_totaltext_letterbox_word_f1_separator_head.py) | Two isolated TRAIN-only letterbox runs: word-F1 control and separator head. | See source/tests |
| [scripts/train_totaltext_partial_encoder_50ep.py](../scripts/train_totaltext_partial_encoder_50ep.py) | Matched 50-epoch TRAIN-only continuations; no TEST or OCR in selection. | See source/tests |
| [scripts/train_totaltext_v4_highres_ablation.py](../scripts/train_totaltext_v4_highres_ablation.py) | Isolated TRAIN-only Total-Text v4 native-256 segmentation ablation. | See source/tests |
| [scripts/train_totaltext_v4_image_frequency_ablation.py](../scripts/train_totaltext_v4_image_frequency_ablation.py) | TRAIN-only, restartable practical image/frequency ablations for Total-Text v4. | See source/tests |
| [scripts/train_totaltext_word_ablation_50ep.py](../scripts/train_totaltext_word_ablation_50ep.py) | Four isolated practical TRAIN interventions and a matched50-epoch control. | See source/tests |
| [scripts/train_totaltext_word_kernel_100ep.py](../scripts/train_totaltext_word_kernel_100ep.py) | One isolated 100-epoch Total-Text word-kernel TRAIN/validation experiment. | See source/tests |
| [scripts/train_uniform_pooled_gate.py](../scripts/train_uniform_pooled_gate.py) | Validation-only pooled image-router study with explicit hard fallback budgets. | See source/tests |
| [scripts/train_uniform_risk_fusion.py](../scripts/train_uniform_risk_fusion.py) | Fit one identical word-risk/expected-error/fusion procedure per dataset. | uniform-unet-fusion |
| [scripts/train_uniform_utility_fusion.py](../scripts/train_uniform_utility_fusion.py) | Fit an explicit common count-utility family and conservative fallback fusion. | uniform-unet-fusion |
| [scripts/train_uniform_word_admission.py](../scripts/train_uniform_word_admission.py) | Validation-only pooled light-word classifier, with native cut retained as control. | See source/tests |
| [scripts/train_uniform_word_union.py](../scripts/train_uniform_word_union.py) | Fit/audit one pooled word fusion model with unchanged qualified image gates. | See source/tests |
| [scripts/train_word_center_offsets.py](../scripts/train_word_center_offsets.py) | Restartable APPROXIMATE auxiliary center-vector training on qualified Total-Text. | word-center-offsets |
| [scripts/train_word_center_votes.py](../scripts/train_word_center_votes.py) | Matched APPROXIMATE coherent-center-vote loss training; unchanged U-Net architecture. | word-center-vote-loss |
| [scripts/unet_token_router_stages.py](../scripts/unet_token_router_stages.py) | Qualified frozen expert stages and matched token-head fitting/evaluation. | unet-token-router |
| [scripts/uniform_decoder_folds.py](../scripts/uniform_decoder_folds.py) | Conditional decoder-heldout supervision for the common trained pipeline. | uniform-unet-fusion |
| [scripts/verify_separated_word_gt.py](../scripts/verify_separated_word_gt.py) | Independent full-artifact verification using OpenCV binary components. | separated-word-gt |
| [scripts/verify_tighter_word_gt.py](../scripts/verify_tighter_word_gt.py) | Independently verify compact v2 outputs against their unchanged v1 sources. | tighter-word-gt-v2 |
| [scripts/word_evidence_active.py](../scripts/word_evidence_active.py) | A graph acquires at most two visual observations; matched-cost controls. | See source/tests |
| [scripts/word_evidence_consensus.py](../scripts/word_evidence_consensus.py) | GPU alignment and spatially grounded character consensus, APPROXIMATE only. | See source/tests |
| [scripts/word_graph_geometry_stages.py](../scripts/word_graph_geometry_stages.py) | True U-Net-free proposal/deformation training and matched fresh controls. | word-graph-recovery |
| [scripts/word_graph_regional_stages.py](../scripts/word_graph_regional_stages.py) | FIT-verified complete-word region distillation: GT, frozen and EMA controls. | word-graph-recovery |
| [scripts/write_deepsolo_runtime_manifest.py](../scripts/write_deepsolo_runtime_manifest.py) | Create an untracked runtime manifest from explicit candidate/checkpoint pairs. | See source/tests |

## segmentation

| Source | Purpose | Experiment guides |
| --- | --- | --- |
| [segmentation/__init__.py](../segmentation/__init__.py) | Approximate segmentation-reproduction components. | See source/tests |
| [segmentation/adaspot_gap.py](../segmentation/adaspot_gap.py) | Orientation-independent inter-word corridors, paired targets and masked loss. | See source/tests |
| [segmentation/adaspot_gap_decoder.py](../segmentation/adaspot_gap_decoder.py) | Explicit class-balanced spatial Gap supervision, approximate experiment. | See source/tests |
| [segmentation/adaspot_joint.py](../segmentation/adaspot_joint.py) | APPROXIMATE joint supervision: predicted ROIs, matched labels, detached gate. | See source/tests |
| [segmentation/config.py](../segmentation/config.py) | Mode-isolated configuration validation for segmentation experiments. | See source/tests |
| [segmentation/ctw1500_word_gt.py](../segmentation/ctw1500_word_gt.py) | Parse the pinned VimTS word-GT variant, distinct from CTW line GT. | See source/tests |
| [segmentation/data.py](../segmentation/data.py) | Checksum-aware segmentation manifests and paired image/mask loading. | See source/tests |
| [segmentation/directional_transport.py](../segmentation/directional_transport.py) | Isolated APPROXIMATE native512 targets and training-only ownership edges. | directional-transport-results, directional-transport-unet |
| [segmentation/evaluate.py](../segmentation/evaluate.py) | Deterministic APPROXIMATE ICDAR13 segmentation evaluation. | See source/tests |
| [segmentation/evaluate_trajectory.py](../segmentation/evaluate_trajectory.py) | Post-hoc test-set trajectory for an already completed segmentation run. | See source/tests |
| [segmentation/guarded_word_dataset.py](../segmentation/guarded_word_dataset.py) | Training-facing V3 wrapper: transport guard evidence through DataLoader workers. | See source/tests |
| [segmentation/guarded_word_gt.py](../segmentation/guarded_word_gt.py) | Opt-in APPROXIMATE native word-target repair, after the real paired transform. | See source/tests |
| [segmentation/icdar13_manifests.py](../segmentation/icdar13_manifests.py) | Generate deterministic M9 manifests for the adopted ICDAR13 split assumption. | See source/tests |
| [segmentation/icdar13_matched_training.py](../segmentation/icdar13_matched_training.py) | Approximate, isolated loss controls for the ICDAR13 word-mask refit. | icdar13-matched-word-training |
| [segmentation/icdar13_word_refit.py](../segmentation/icdar13_word_refit.py) | Approximate ICDAR2013 full-word / kernel transfer with qualified geometry. | icdar13-word-refit, shared-pipeline-code-tour |
| [segmentation/icdar2015_encoder_decay.py](../segmentation/icdar2015_encoder_decay.py) | Controlled approximate encoder-decay ablation with fresh encoder moments. | See source/tests |
| [segmentation/icdar2015_improvement_runtime.py](../segmentation/icdar2015_improvement_runtime.py) | Shared provenance/validation for the isolated approximate improvement suite. | icdar2015-unet-improvement |
| [segmentation/icdar2015_improvements.py](../segmentation/icdar2015_improvements.py) | Isolated approximate ICDAR2015 resolution, ownership and word-loss variants. | icdar2015-unet-improvement |
| [segmentation/icdar2015_multiview.py](../segmentation/icdar2015_multiview.py) | GT-blind merging of hypotheses from one U-Net at several input views. | See source/tests |
| [segmentation/icdar2015_operating_policy.py](../segmentation/icdar2015_operating_policy.py) | GT-blind components and OCR crops from approximate U-Net probability maps. | See source/tests |
| [segmentation/icdar2015_seed_partition.py](../segmentation/icdar2015_seed_partition.py) | Approximate single-logit confidence seeds; preserve 8-connected word gaps. | See source/tests |
| [segmentation/icdar2015_synthetic_training.py](../segmentation/icdar2015_synthetic_training.py) | Explicit AdamW stages and counted synthetic supervision; approximate only. | See source/tests |
| [segmentation/icdar2015_synthtext_cache.py](../segmentation/icdar2015_synthtext_cache.py) | Audited synthetic target cache; original labels and RGB bytes are preserved. | See source/tests |
| [segmentation/icdar2015_target70.py](../segmentation/icdar2015_target70.py) | Approximate target70 supervision, accumulated updates and frozen OCR replay. | See source/tests |
| [segmentation/icdar2015_target70_ohem.py](../segmentation/icdar2015_target70_ohem.py) | Per-image weighted hard-negative BCE with explicit negative-only behavior. | See source/tests |
| [segmentation/icdar2015_target70_selection.py](../segmentation/icdar2015_target70_selection.py) | Freeze approximate ICDAR model/policy choices exclusively from TRAIN VAL100. | See source/tests |
| [segmentation/icdar2015_totaltext_weights.py](../segmentation/icdar2015_totaltext_weights.py) | Explicit, checksum-pinned Total-Text weight transfer; approximate only. | See source/tests |
| [segmentation/icdar2015_transfer.py](../segmentation/icdar2015_transfer.py) | Explicit approximate ICDAR2015 word targets, paired geometry and validation. | See source/tests |
| [segmentation/icdar2015_word_size.py](../segmentation/icdar2015_word_size.py) | GT-blind size admission for cached ICDAR2015 component proposals. | See source/tests |
| [segmentation/kernel_only_training.py](../segmentation/kernel_only_training.py) | Only Kernel and validity enter this supervised loss. | See source/tests |
| [segmentation/mask_probability_refiner.py](../segmentation/mask_probability_refiner.py) | Paired full-image supervision and losses for residual probability refinement. | See source/tests |
| [segmentation/metrics.py](../segmentation/metrics.py) | Explicit foreground IoU/F1 aggregation modes for segmentation. | See source/tests |
| [segmentation/recognition_crop_refiner.py](../segmentation/recognition_crop_refiner.py) | Experimental crop geometry learned through a frozen PARSeq recognizer. | See source/tests |
| [segmentation/scene_text_preparation.py](../segmentation/scene_text_preparation.py) | Prepare auditable binary segmentation targets for three scene-text datasets. | icdar2015-totaltext-transfer |
| [segmentation/separated_word_gt.py](../segmentation/separated_word_gt.py) | GT-derived, separated word regions for APPROXIMATE supervision only. | See source/tests |
| [segmentation/smoke.py](../segmentation/smoke.py) | Tiny synthetic end-to-end smoke for the APPROXIMATE segmentation path. | See source/tests |
| [segmentation/synthtext_words.py](../segmentation/synthtext_words.py) | Read-only SynthText word polygons; deterministic SHA-bound external subset. | See source/tests |
| [segmentation/text_recall.py](../segmentation/text_recall.py) | Isolated recall/resolution experiments; original scientific paths stay frozen. | See source/tests |
| [segmentation/text_recall_provenance.py](../segmentation/text_recall_provenance.py) | Preserve the one known CUDA-initialization failure, before any training. | See source/tests |
| [segmentation/tighter_word_gt.py](../segmentation/tighter_word_gt.py) | Smaller filled word cores, an explicit APPROXIMATE supervision experiment. | See source/tests |
| [segmentation/totaltext_adaptive_word_boxes.py](../segmentation/totaltext_adaptive_word_boxes.py) | Filled, piecewise word-box TRAIN targets for Total-Text. | See source/tests |
| [segmentation/totaltext_curved_box_v4.py](../segmentation/totaltext_curved_box_v4.py) | Smooth, filled Total-Text word polygons for a separate TRAIN target. | agent-experiment-review-2026-10-11 |
| [segmentation/totaltext_image_frequency_augment.py](../segmentation/totaltext_image_frequency_augment.py) | Isolated TRAIN-only image augmentations for practical Total-Text ablations. | See source/tests |
| [segmentation/totaltext_word_ablation.py](../segmentation/totaltext_word_ablation.py) | Practical losses, paired TRAIN zoom, and inference-only pixel-link grouping. | icdar2015-target70-ohem |
| [segmentation/totaltext_word_kernel.py](../segmentation/totaltext_word_kernel.py) | Practical word nuclei targets and GT-blind simultaneous region expansion. | word-multitask |
| [segmentation/totaltext_word_regions.py](../segmentation/totaltext_word_regions.py) | Disjoint curved word targets from Total-Text polygons (PRACTICAL_APPROXIMATION). | adaspot-gap |
| [segmentation/train.py](../segmentation/train.py) | Real-manifest segmentation training entry point. | ag-unet-implementation-decisions, reproduction-plan |
| [segmentation/training.py](../segmentation/training.py) | Auditable training primitives for APPROXIMATE segmentation runs. | See source/tests |
| [segmentation/transforms.py](../segmentation/transforms.py) | Explicit paired transforms for the APPROXIMATE segmentation branch. | See source/tests |
| [segmentation/unet_auxiliary_ocr.py](../segmentation/unet_auxiliary_ocr.py) | TRAIN-only word reader supervision and matched text/kernel objectives. | See source/tests |
| [segmentation/unet_crop_links.py](../segmentation/unet_crop_links.py) | Frozen-prediction crop sampling and word-instance supervision; no GT crop boxes. | See source/tests |
| [segmentation/unet_dense_instances.py](../segmentation/unet_dense_instances.py) | Geometry targets, loss and deterministic proposals for a U-Net instance head. | icdar13-unet-dense-instances |
| [segmentation/unet_detection_methods.py](../segmentation/unet_detection_methods.py) | Nested word nuclei and masked objectives, reusing existing word targets/loss. | See source/tests |
| [segmentation/unet_word_box_field.py](../segmentation/unet_word_box_field.py) | Dense box targets and word-balanced loss; primary recognition GT is unchanged. | See source/tests |
| [segmentation/unet_word_research.py](../segmentation/unet_word_research.py) | Paired glyph/word supervision and declared experimental auxiliary losses. | See source/tests |
| [segmentation/unet_word_research_v2.py](../segmentation/unet_word_research_v2.py) | Declared corrective objectives; original glyph/word datasets stay immutable. | agent-experiment-review-2026-10-11 |
| [segmentation/uniform_encoder_adaptation.py](../segmentation/uniform_encoder_adaptation.py) | Explicit upper-encoder adaptation; the legacy frozen path remains unchanged. | See source/tests |
| [segmentation/uniform_instance_objective.py](../segmentation/uniform_instance_objective.py) | Isolated owner-balanced text loss, preserving the qualified kernel objective. | See source/tests |
| [segmentation/uniform_multiscale_masks.py](../segmentation/uniform_multiscale_masks.py) | Explicit probability averaging on the native U-Net half-pixel grid. | See source/tests |
| [segmentation/uniform_photometric_training.py](../segmentation/uniform_photometric_training.py) | RGB-only training perturbations over the qualified uniform word dataset. | See source/tests |
| [segmentation/uniform_unet_fusion.py](../segmentation/uniform_unet_fusion.py) | Shared, explicit APPROXIMATE word-U-Net and fusion experiment boundaries. | shared-pipeline-code-tour, uniform-unet-fusion |
| [segmentation/word_center_offsets.py](../segmentation/word_center_offsets.py) | Word-balanced dense attraction targets and training for the auxiliary U-Net. | See source/tests |
| [segmentation/word_center_votes.py](../segmentation/word_center_votes.py) | APPROXIMATE endpoint-consensus/neighbor loss for the unchanged offset U-Net. | See source/tests |
| [segmentation/word_centerline.py](../segmentation/word_centerline.py) | Native word skeleton targets and isolated APPROXIMATE radius supervision. | See source/tests |
| [segmentation/word_checkpoint_metrics.py](../segmentation/word_checkpoint_metrics.py) | APPROXIMATE native-mask and operational-box metrics for checkpoint selection. | See source/tests |
| [segmentation/word_evidence.py](../segmentation/word_evidence.py) | Small CPU-trained modules; all segmentation weights and evaluation maps frozen. | See source/tests |
| [segmentation/word_instance_data.py](../segmentation/word_instance_data.py) | Read-only, hash-bound source maps and separated inference/training inputs. | See source/tests |
| [segmentation/word_instance_extensions.py](../segmentation/word_instance_extensions.py) | Prompt-specific masks and sampled, class-balanced maximin connectivity loss. | See source/tests |
| [segmentation/word_multitask.py](../segmentation/word_multitask.py) | Approximate word-instance targets and configurable masked multi-task loss. | See source/tests |
| [segmentation/word_multitask_evaluation.py](../segmentation/word_multitask_evaluation.py) | Validation and visual exports built on the existing geometry/pixel metrics. | See source/tests |
| [segmentation/word_multitask_provenance.py](../segmentation/word_multitask_provenance.py) | JSON-stable contracts and a narrow, auditable repair for smoke-only runs. | See source/tests |
| [segmentation/word_slot_data.py](../segmentation/word_slot_data.py) | Checksum-bound word/owner alignment and explicit letterbox coordinates. | See source/tests |
| [segmentation/word_slot_objectives.py](../segmentation/word_slot_objectives.py) | Explicit, independently switchable objectives for the APPROXIMATE slot study. | word-slot-methods |

## tests

| Source | Purpose | Experiment guides |
| --- | --- | --- |
| [tests/caption/test_caption_foundation.py](../tests/caption/test_caption_foundation.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/caption/test_decord_wheel_repair.py](../tests/caption/test_decord_wheel_repair.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/caption/test_detailed_scene.py](../tests/caption/test_detailed_scene.py) | No module docstring; see linked source and tests | detailed-scene-captions |
| [tests/caption/test_florence_scene.py](../tests/caption/test_florence_scene.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/caption/test_lavis_captioner_candidates.py](../tests/caption/test_lavis_captioner_candidates.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/caption/test_paper_batch_blip2.py](../tests/caption/test_paper_batch_blip2.py) | No module docstring; see linked source and tests | paper-caption-router-comparison |
| [tests/caption/test_qwen_batch_scene.py](../tests/caption/test_qwen_batch_scene.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/caption/test_qwen_scene.py](../tests/caption/test_qwen_scene.py) | No module docstring; see linked source and tests | detailed-scene-captions |
| [tests/data/test_icdar13_manifests.py](../tests/data/test_icdar13_manifests.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/data/test_icdar13_mask_conversion.py](../tests/data/test_icdar13_mask_conversion.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/data/test_segmentation_dataset.py](../tests/data/test_segmentation_dataset.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/data/test_segmentation_transforms.py](../tests/data/test_segmentation_transforms.py) | No module docstring; see linked source and tests | reproduction-plan, word-multitask |
| [tests/deepsolo/test_candidates_and_reduction.py](../tests/deepsolo/test_candidates_and_reduction.py) | No module docstring; see linked source and tests | word-multitask |
| [tests/deepsolo/test_contract.py](../tests/deepsolo/test_contract.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/deepsolo/test_host_setup.py](../tests/deepsolo/test_host_setup.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/deepsolo/test_source_identity.py](../tests/deepsolo/test_source_identity.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/deepsolo/test_worker_inference.py](../tests/deepsolo/test_worker_inference.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_adaspot_gap_decode.py](../tests/localize/test_adaspot_gap_decode.py) | Barrier boundaries, fused kernels, deterministic BFS and original coordinates. | See source/tests |
| [tests/localize/test_algorithm1.py](../tests/localize/test_algorithm1.py) | No module docstring; see linked source and tests | reproduction-plan |
| [tests/localize/test_appearance_word_remap.py](../tests/localize/test_appearance_word_remap.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_bilinear_mask_backprojection.py](../tests/localize/test_bilinear_mask_backprojection.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_bilinear_threshold_train.py](../tests/localize/test_bilinear_threshold_train.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_center_guided_word_crops.py](../tests/localize/test_center_guided_word_crops.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_classic_word_regions.py](../tests/localize/test_classic_word_regions.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_coherent_probability_groups.py](../tests/localize/test_coherent_probability_groups.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_contiguous_span_candidates.py](../tests/localize/test_contiguous_span_candidates.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_crop_recovery.py](../tests/localize/test_crop_recovery.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_crop_word_groups.py](../tests/localize/test_crop_word_groups.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_curved_component_crop.py](../tests/localize/test_curved_component_crop.py) | The curved cutout must isolate one mask component without moving its box. | See source/tests |
| [tests/localize/test_directional_affinity_split.py](../tests/localize/test_directional_affinity_split.py) | No module docstring; see linked source and tests | directional-transport-results |
| [tests/localize/test_directional_closedness.py](../tests/localize/test_directional_closedness.py) | No module docstring; see linked source and tests | directional-transport-results |
| [tests/localize/test_directional_component_neighbors.py](../tests/localize/test_directional_component_neighbors.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_external_text_detectors.py](../tests/localize/test_external_text_detectors.py) | Geometry/RGB boundary checks independent of optional pretrained runtimes. | See source/tests |
| [tests/localize/test_gap_word_regions.py](../tests/localize/test_gap_word_regions.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_graph_occlusion_followups.py](../tests/localize/test_graph_occlusion_followups.py) | Focused scientific tests for the four follow-up experiments. | graph-occlusion-followups |
| [tests/localize/test_graph_occlusion_research.py](../tests/localize/test_graph_occlusion_research.py) | Scientific contracts for fragment graphs and occlusion distillation. | See source/tests |
| [tests/localize/test_group_probability_cache.py](../tests/localize/test_group_probability_cache.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_grouped_word_blocks.py](../tests/localize/test_grouped_word_blocks.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_gt_mask_localization_oracle.py](../tests/localize/test_gt_mask_localization_oracle.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_independent_word_slots.py](../tests/localize/test_independent_word_slots.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_line_word_blocks.py](../tests/localize/test_line_word_blocks.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_multihypothesis_word_proposals.py](../tests/localize/test_multihypothesis_word_proposals.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_multiscale_component_persistence.py](../tests/localize/test_multiscale_component_persistence.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_native_algorithm1.py](../tests/localize/test_native_algorithm1.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_native_repository_contours.py](../tests/localize/test_native_repository_contours.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_occlusion_four_tests.py](../tests/localize/test_occlusion_four_tests.py) | No module docstring; see linked source and tests | occlusion-four-tests |
| [tests/localize/test_practical_candidate_selection.py](../tests/localize/test_practical_candidate_selection.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_practical_candidate_selection_v2.py](../tests/localize/test_practical_candidate_selection_v2.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_practical_remap_failure_analysis.py](../tests/localize/test_practical_remap_failure_analysis.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_probability_threshold_word_connectivity.py](../tests/localize/test_probability_threshold_word_connectivity.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_repository_contours.py](../tests/localize/test_repository_contours.py) | Differential parity with the frozen author source, plus scientific edge cases. | See source/tests |
| [tests/localize/test_repository_contours_train.py](../tests/localize/test_repository_contours_train.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_rlsa_gap_postsplit.py](../tests/localize/test_rlsa_gap_postsplit.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_structural_word_groups.py](../tests/localize/test_structural_word_groups.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_supervised_word_boundary_train.py](../tests/localize/test_supervised_word_boundary_train.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_teacher_student_tests.py](../tests/localize/test_teacher_student_tests.py) | Scientific boundary tests for the isolated teacher/student pilots. | teacher-student-tests |
| [tests/localize/test_text_recall_crops.py](../tests/localize/test_text_recall_crops.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_totaltext_train_padding_sweep.py](../tests/localize/test_totaltext_train_padding_sweep.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_train_crop_completeness.py](../tests/localize/test_train_crop_completeness.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_train_crop_glyph_audit.py](../tests/localize/test_train_crop_glyph_audit.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_train_v1_ocr_failure_audit.py](../tests/localize/test_train_v1_ocr_failure_audit.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_unet_crop_refinement.py](../tests/localize/test_unet_crop_refinement.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_unet_detection_instances.py](../tests/localize/test_unet_detection_instances.py) | No module docstring; see linked source and tests | unet-detection-methods |
| [tests/localize/test_unet_seeded_selective.py](../tests/localize/test_unet_seeded_selective.py) | Meaningful invariants for soft link growth and GT-free selective replacement. | See source/tests |
| [tests/localize/test_word_center_voting.py](../tests/localize/test_word_center_voting.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_word_centerline_decode.py](../tests/localize/test_word_centerline_decode.py) | GT-blind instance identity, disk competition, and oracle representation. | word-centerline |
| [tests/localize/test_word_crop_refinements.py](../tests/localize/test_word_crop_refinements.py) | No module docstring; see linked source and tests | unet-detection-methods |
| [tests/localize/test_word_detector_training.py](../tests/localize/test_word_detector_training.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_word_evidence_graphs.py](../tests/localize/test_word_evidence_graphs.py) | Scientific boundaries for visual ownership, alignment and active observation. | See source/tests |
| [tests/localize/test_word_graph_cut.py](../tests/localize/test_word_graph_cut.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/localize/test_word_graph_recovery.py](../tests/localize/test_word_graph_recovery.py) | Scientific invariants for non-destructive graphs and word-level objectives. | word-graph-recovery |
| [tests/localize/test_word_guided_contours.py](../tests/localize/test_word_guided_contours.py) | No module docstring; see linked source and tests | unet-detection-methods |
| [tests/models/test_adaspot_gap_architecture.py](../tests/models/test_adaspot_gap_architecture.py) | Full architecture compatibility and strict warm-start behavior. | See source/tests |
| [tests/models/test_adaspot_joint_components.py](../tests/models/test_adaspot_joint_components.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/models/test_adaspot_mask_align.py](../tests/models/test_adaspot_mask_align.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/models/test_ag_unet_architecture.py](../tests/models/test_ag_unet_architecture.py) | No module docstring; see linked source and tests | reproduction-plan |
| [tests/models/test_ag_unet_profiler.py](../tests/models/test_ag_unet_profiler.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/models/test_directional_transport.py](../tests/models/test_directional_transport.py) | No module docstring; see linked source and tests | directional-transport-results, directional-transport-unet |
| [tests/models/test_icdar2015_improvements.py](../tests/models/test_icdar2015_improvements.py) | Norm migration retains every convolution/affine tensor and freezes encoder. | icdar2015-unet-improvement |
| [tests/models/test_icdar2015_ohem_encoder.py](../tests/models/test_icdar2015_ohem_encoder.py) | Full feature gradients must reach early layers while source inference is exact. | See source/tests |
| [tests/models/test_icdar2015_target70_encoder.py](../tests/models/test_icdar2015_target70_encoder.py) | Actual gradients and preserved decoder optimizer state, without downloads. | icdar2015-target70 |
| [tests/models/test_joint_word_instance_unet.py](../tests/models/test_joint_word_instance_unet.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/models/test_kernel_only.py](../tests/models/test_kernel_only.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/models/test_kernel_only_fresh.py](../tests/models/test_kernel_only_fresh.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/models/test_text_pretrained_encoder.py](../tests/models/test_text_pretrained_encoder.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/models/test_totaltext_frequency_gated_skip.py](../tests/models/test_totaltext_frequency_gated_skip.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/models/test_unet_adv_attention.py](../tests/models/test_unet_adv_attention.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/models/test_unet_auxiliary_ocr.py](../tests/models/test_unet_auxiliary_ocr.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/models/test_unet_detection_architecture.py](../tests/models/test_unet_detection_architecture.py) | No module docstring; see linked source and tests | unet-detection-methods |
| [tests/models/test_word_graph_decoder.py](../tests/models/test_word_graph_decoder.py) | Real encoder compatibility and trainable U-Net-free decoder qualification. | word-graph-recovery |
| [tests/models/test_word_slot_unet.py](../tests/models/test_word_slot_unet.py) | Slot/pixel leakage, exact prior reuse, sampling and conflicting-gradient tests. | word-slot-methods |
| [tests/pipeline/test_approximate_e2e.py](../tests/pipeline/test_approximate_e2e.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/pipeline/test_e2e_comparison_diagnostic.py](../tests/pipeline/test_e2e_comparison_diagnostic.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/pipeline/test_frozen_segmentation.py](../tests/pipeline/test_frozen_segmentation.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/pipeline/test_icdar13_trace_batch.py](../tests/pipeline/test_icdar13_trace_batch.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/pipeline/test_joined_t3.py](../tests/pipeline/test_joined_t3.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/pipeline/test_lightweight_context.py](../tests/pipeline/test_lightweight_context.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/pipeline/test_routing_suite.py](../tests/pipeline/test_routing_suite.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/pipeline/test_uniform_scaled_frontend.py](../tests/pipeline/test_uniform_scaled_frontend.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/pipeline/test_uniform_word_pipeline.py](../tests/pipeline/test_uniform_word_pipeline.py) | No module docstring; see linked source and tests | shared-pipeline-code-tour |
| [tests/recognition/test_adapted_word_policy.py](../tests/recognition/test_adapted_word_policy.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_bert_correction.py](../tests/recognition/test_bert_correction.py) | Scientific boundaries and genuine tiny-BERT CPU integration (random fixture). | See source/tests |
| [tests/recognition/test_bert_crop_selection.py](../tests/recognition/test_bert_crop_selection.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_bert_method_selection.py](../tests/recognition/test_bert_method_selection.py) | Selective edits and immutable-KEEP ranking boundaries; no dataset labels. | See source/tests |
| [tests/recognition/test_bert_visual_correction.py](../tests/recognition/test_bert_visual_correction.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_blip_visual_only.py](../tests/recognition/test_blip_visual_only.py) | No module docstring; see linked source and tests | encoder-family-router |
| [tests/recognition/test_boundary_inventory_router.py](../tests/recognition/test_boundary_inventory_router.py) | Quota and protected-decision guarantees for bounded inventory corrections. | shadow-inventory-router |
| [tests/recognition/test_budget_distilled_router.py](../tests/recognition/test_budget_distilled_router.py) | Scientific boundaries for the cheap, paired-expert supervised gate. | budget-distilled-router |
| [tests/recognition/test_budget_target_oracle.py](../tests/recognition/test_budget_target_oracle.py) | No module docstring; see linked source and tests | budget-target-router |
| [tests/recognition/test_clip_crop_admission.py](../tests/recognition/test_clip_crop_admission.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_clip_image_router.py](../tests/recognition/test_clip_image_router.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_clip_image_router_execution.py](../tests/recognition/test_clip_image_router_execution.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_clip_innovative_verifier.py](../tests/recognition/test_clip_innovative_verifier.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_clip_literature_runtime.py](../tests/recognition/test_clip_literature_runtime.py) | Scientific boundary checks for v3; tiny random models are structural tests. | See source/tests |
| [tests/recognition/test_clip_router_context_chunks.py](../tests/recognition/test_clip_router_context_chunks.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_clip_router_evaluation.py](../tests/recognition/test_clip_router_evaluation.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_clip_router_prompt_ablation.py](../tests/recognition/test_clip_router_prompt_ablation.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_clip_router_threshold_sweep.py](../tests/recognition/test_clip_router_threshold_sweep.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_clip_specialized_reporting.py](../tests/recognition/test_clip_specialized_reporting.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_clip_specialized_runtime.py](../tests/recognition/test_clip_specialized_runtime.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_clip_word_verifier.py](../tests/recognition/test_clip_word_verifier.py) | No module docstring; see linked source and tests | clip-word-verification |
| [tests/recognition/test_common_budget_pipeline.py](../tests/recognition/test_common_budget_pipeline.py) | No module docstring; see linked source and tests | common-parity-threshold |
| [tests/recognition/test_common_parity_threshold.py](../tests/recognition/test_common_parity_threshold.py) | Scientific boundaries of validation-only scalar F1 parity calibration. | common-parity-threshold |
| [tests/recognition/test_conservative_ocr_edit_gates_train.py](../tests/recognition/test_conservative_ocr_edit_gates_train.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_contextual_boundary_selection.py](../tests/recognition/test_contextual_boundary_selection.py) | Focused scientific checks for boundary selection and leakage boundaries. | contextual-boundary-selection |
| [tests/recognition/test_contextual_lm_verifier_runtime.py](../tests/recognition/test_contextual_lm_verifier_runtime.py) | No downloaded models: test scoring masks, frozen state, and failure boundaries. | See source/tests |
| [tests/recognition/test_contextual_ocr_correction.py](../tests/recognition/test_contextual_ocr_correction.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_contextual_ocr_fewshot_train.py](../tests/recognition/test_contextual_ocr_fewshot_train.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_contextual_ocr_plaintext_train.py](../tests/recognition/test_contextual_ocr_plaintext_train.py) | CPU-only protocol tests; no model loading or inference. | contextual-ocr-plaintext-train-v2 |
| [tests/recognition/test_contextual_quality_features.py](../tests/recognition/test_contextual_quality_features.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_contextual_quality_router.py](../tests/recognition/test_contextual_quality_router.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_creative_context_router.py](../tests/recognition/test_creative_context_router.py) | Scientific boundaries for six context signals and TRAIN error retrieval. | creative-context-router |
| [tests/recognition/test_crop_group_refinements.py](../tests/recognition/test_crop_group_refinements.py) | No module docstring; see linked source and tests | crop-group-refinements |
| [tests/recognition/test_crop_group_test_suite.py](../tests/recognition/test_crop_group_test_suite.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_crop_recovery_suite.py](../tests/recognition/test_crop_recovery_suite.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_deepsolo_icdar2015_gap.py](../tests/recognition/test_deepsolo_icdar2015_gap.py) | Meaningful boundaries for validation-selected DeepSolo filtering. | deepsolo-icdar2015-gap |
| [tests/recognition/test_deepsolo_lexicon.py](../tests/recognition/test_deepsolo_lexicon.py) | Source-order, edit boundary and denominator safeguards for DeepSolo lexicons. | deepsolo-icdar-lexicons |
| [tests/recognition/test_deepsolo_lexicon_deployment.py](../tests/recognition/test_deepsolo_lexicon_deployment.py) | Prediction-only lexicon qualification and strict identity/failure boundaries. | deepsolo-icdar-lexicons |
| [tests/recognition/test_deepsolo_lexicon_guard.py](../tests/recognition/test_deepsolo_lexicon_guard.py) | Confidence and open-vocabulary protection without access to evaluation labels. | deepsolo-icdar-lexicons |
| [tests/recognition/test_deepsolo_word_f1.py](../tests/recognition/test_deepsolo_word_f1.py) | No module docstring; see linked source and tests | deepsolo-word-f1-study |
| [tests/recognition/test_detailed_caption_comparison.py](../tests/recognition/test_detailed_caption_comparison.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_dont_care_evaluation.py](../tests/recognition/test_dont_care_evaluation.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_encoder_family_inference.py](../tests/recognition/test_encoder_family_inference.py) | No module docstring; see linked source and tests | encoder-family-router |
| [tests/recognition/test_encoder_family_router.py](../tests/recognition/test_encoder_family_router.py) | No module docstring; see linked source and tests | encoder-family-router |
| [tests/recognition/test_essential_conditional_oracle.py](../tests/recognition/test_essential_conditional_oracle.py) | No module docstring; see linked source and tests | essential-router-threshold-analysis |
| [tests/recognition/test_essential_icdar_transfer.py](../tests/recognition/test_essential_icdar_transfer.py) | No module docstring; see linked source and tests | essential-router-icdar-transfer, essential-router-threshold-analysis |
| [tests/recognition/test_essential_portable_router.py](../tests/recognition/test_essential_portable_router.py) | No module docstring; see linked source and tests | budget-distilled-router |
| [tests/recognition/test_essential_router_budget.py](../tests/recognition/test_essential_router_budget.py) | No module docstring; see linked source and tests | essential-router-budget50 |
| [tests/recognition/test_essential_threshold_deployment.py](../tests/recognition/test_essential_threshold_deployment.py) | No module docstring; see linked source and tests | essential-router-threshold-analysis |
| [tests/recognition/test_essential_threshold_frontier.py](../tests/recognition/test_essential_threshold_frontier.py) | No module docstring; see linked source and tests | essential-router-threshold-analysis |
| [tests/recognition/test_essential_word_heuristics.py](../tests/recognition/test_essential_word_heuristics.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_f1_quality_study.py](../tests/recognition/test_f1_quality_study.py) | No module docstring; see linked source and tests | yolo-f1-quality-study |
| [tests/recognition/test_fallback_feasibility.py](../tests/recognition/test_fallback_feasibility.py) | No module docstring; see linked source and tests | icdar13-word-refit |
| [tests/recognition/test_fallback_reread.py](../tests/recognition/test_fallback_reread.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_fallback_word_fusion.py](../tests/recognition/test_fallback_word_fusion.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_fixed_expert_budget.py](../tests/recognition/test_fixed_expert_budget.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_fixed_expert_deployment.py](../tests/recognition/test_fixed_expert_deployment.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_fixed_refit_boundary.py](../tests/recognition/test_fixed_refit_boundary.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_fold_safe_teacher.py](../tests/recognition/test_fold_safe_teacher.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_geometry_context_reconstruction_train.py](../tests/recognition/test_geometry_context_reconstruction_train.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_grounded_context_router.py](../tests/recognition/test_grounded_context_router.py) | Grounded inference boundaries: spatial completeness, local controls and schemas. | grounded-context-router |
| [tests/recognition/test_grounded_word_recovery.py](../tests/recognition/test_grounded_word_recovery.py) | Scientific boundary tests for RGB-grounded word recovery. | grounded-word-recovery |
| [tests/recognition/test_group_output_filter.py](../tests/recognition/test_group_output_filter.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_group_word_selection.py](../tests/recognition/test_group_word_selection.py) | No module docstring; see linked source and tests | clip-word-verification |
| [tests/recognition/test_gt_word_group_selector.py](../tests/recognition/test_gt_word_group_selector.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_hard_fallback_budget.py](../tests/recognition/test_hard_fallback_budget.py) | No module docstring; see linked source and tests | shared-pipeline-code-tour |
| [tests/recognition/test_ic13_complement_risk.py](../tests/recognition/test_ic13_complement_risk.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_ic13_multiscale.py](../tests/recognition/test_ic13_multiscale.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_icdar13_annotations.py](../tests/recognition/test_icdar13_annotations.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_icdar13_hypothesis_tests.py](../tests/recognition/test_icdar13_hypothesis_tests.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_icdar13_matched_router.py](../tests/recognition/test_icdar13_matched_router.py) | No module docstring; see linked source and tests | icdar13-matched-router-generic |
| [tests/recognition/test_icdar13_trace_diagnostics.py](../tests/recognition/test_icdar13_trace_diagnostics.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_icdar2015_word_f1.py](../tests/recognition/test_icdar2015_word_f1.py) | Word F1 boundaries: symbols, duplicates, spatial matching and empty outputs. | icdar2015-word-f1 |
| [tests/recognition/test_icdar_budget_deployment.py](../tests/recognition/test_icdar_budget_deployment.py) | Single-image deployment and annotation-free input-boundary tests. | icdar-trained-budget-router |
| [tests/recognition/test_icdar_budget_router.py](../tests/recognition/test_icdar_budget_router.py) | No module docstring; see linked source and tests | icdar-trained-budget-router |
| [tests/recognition/test_image_context_correction.py](../tests/recognition/test_image_context_correction.py) | Scientific boundaries for global OCR context, correction and rejection. | image-context-correction |
| [tests/recognition/test_lexical_budget_router.py](../tests/recognition/test_lexical_budget_router.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_lightweight_unet_router.py](../tests/recognition/test_lightweight_unet_router.py) | No module docstring; see linked source and tests | lightweight-unet-router |
| [tests/recognition/test_lightweight_validation_pruning.py](../tests/recognition/test_lightweight_validation_pruning.py) | No module docstring; see linked source and tests | lightweight-unet-router |
| [tests/recognition/test_lightweight_word_selection.py](../tests/recognition/test_lightweight_word_selection.py) | No module docstring; see linked source and tests | lightweight-icdar13-90 |
| [tests/recognition/test_line_token_evaluation.py](../tests/recognition/test_line_token_evaluation.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_neuspell_correction.py](../tests/recognition/test_neuspell_correction.py) | NeuSpell decoding contracts. Tiny random weights are engineering fixtures. | See source/tests |
| [tests/recognition/test_ocr_evidence_generation_battery_train.py](../tests/recognition/test_ocr_evidence_generation_battery_train.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_ocr_prompt_comparison_train.py](../tests/recognition/test_ocr_prompt_comparison_train.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_ocr_target_bound_battery_train.py](../tests/recognition/test_ocr_target_bound_battery_train.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_one_se_fixed_gate.py](../tests/recognition/test_one_se_fixed_gate.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_paper_caption_comparison.py](../tests/recognition/test_paper_caption_comparison.py) | No module docstring; see linked source and tests | paper-caption-router-comparison |
| [tests/recognition/test_paper_image_router_evaluation.py](../tests/recognition/test_paper_image_router_evaluation.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_parseq_adapter.py](../tests/recognition/test_parseq_adapter.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_parseq_candidate_ranking.py](../tests/recognition/test_parseq_candidate_ranking.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_parseq_candidate_ranking_script.py](../tests/recognition/test_parseq_candidate_ranking_script.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_parseq_crop_objective.py](../tests/recognition/test_parseq_crop_objective.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_parseq_crop_sensitivity.py](../tests/recognition/test_parseq_crop_sensitivity.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_parseq_visual_margin_context_gate_train.py](../tests/recognition/test_parseq_visual_margin_context_gate_train.py) | No module docstring; see linked source and tests | parseq-visual-margin-context-gate-train-v1 |
| [tests/recognition/test_probability_threshold_crops_parseq.py](../tests/recognition/test_probability_threshold_crops_parseq.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_recovered_train_crops.py](../tests/recognition/test_recovered_train_crops.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_replacement_fallback_counts.py](../tests/recognition/test_replacement_fallback_counts.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_residual_cost_router.py](../tests/recognition/test_residual_cost_router.py) | No module docstring; see linked source and tests | yolo-residual-cost-router |
| [tests/recognition/test_residual_text_vqa.py](../tests/recognition/test_residual_text_vqa.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_retention_audit_case_selection.py](../tests/recognition/test_retention_audit_case_selection.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_retention_source_boundary.py](../tests/recognition/test_retention_source_boundary.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_risk_admission_deployment.py](../tests/recognition/test_risk_admission_deployment.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_risk_selection_policy.py](../tests/recognition/test_risk_selection_policy.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_router_cause_analysis.py](../tests/recognition/test_router_cause_analysis.py) | Independent numerical guards for matched cause-test reporting. | router-cause-tests |
| [tests/recognition/test_router_cause_tests.py](../tests/recognition/test_router_cause_tests.py) | Invariants for controlled approximate router cause tests. | router-cause-tests |
| [tests/recognition/test_router_fallback_replay.py](../tests/recognition/test_router_fallback_replay.py) | No module docstring; see linked source and tests | trained-router-vocabulary-replay |
| [tests/recognition/test_router_improved_deepsolo_replay.py](../tests/recognition/test_router_improved_deepsolo_replay.py) | Metric and frozen-routing boundaries for improved-fallback comparisons. | router-improved-deepsolo-replay |
| [tests/recognition/test_router_localization_control.py](../tests/recognition/test_router_localization_control.py) | No module docstring; see linked source and tests | router-validation-study |
| [tests/recognition/test_router_validation.py](../tests/recognition/test_router_validation.py) | Validation must stay outside fitting and bind cuts to identical weights. | router-validation-study |
| [tests/recognition/test_scene_text_datasets.py](../tests/recognition/test_scene_text_datasets.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_scene_text_v1_evaluation.py](../tests/recognition/test_scene_text_v1_evaluation.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_segmentation_target_component_topology.py](../tests/recognition/test_segmentation_target_component_topology.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_segmentation_word_topology.py](../tests/recognition/test_segmentation_word_topology.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_selected_deepsolo_evidence.py](../tests/recognition/test_selected_deepsolo_evidence.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_selected_deepsolo_export.py](../tests/recognition/test_selected_deepsolo_export.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_selective_word_repair.py](../tests/recognition/test_selective_word_repair.py) | No module docstring; see linked source and tests | selective-word-repair-tests |
| [tests/recognition/test_shadow_inventory_router.py](../tests/recognition/test_shadow_inventory_router.py) | Scientific boundaries for independent-inventory whole-image routing. | shadow-inventory-router |
| [tests/recognition/test_text_coverage_audit.py](../tests/recognition/test_text_coverage_audit.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_text_coverage_execution.py](../tests/recognition/test_text_coverage_execution.py) | Failure-boundary fixtures; these do not qualify GPU inference or scientific results. | See source/tests |
| [tests/recognition/test_text_coverage_router.py](../tests/recognition/test_text_coverage_router.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_text_reconstruction.py](../tests/recognition/test_text_reconstruction.py) | Scientific boundary checks for text-only reconstruction, not model efficacy. | text-reconstruction-tests |
| [tests/recognition/test_token_context_router.py](../tests/recognition/test_token_context_router.py) | Scientific invariants for the three approximate context router proposals. | token-context-router |
| [tests/recognition/test_totaltext_crop_clip_battery.py](../tests/recognition/test_totaltext_crop_clip_battery.py) | No module docstring; see linked source and tests | clip-word-verification |
| [tests/recognition/test_trainable_image_router.py](../tests/recognition/test_trainable_image_router.py) | No module docstring; see linked source and tests | shadow-inventory-router, yolo-trainable-router |
| [tests/recognition/test_training_free_consistency_audit.py](../tests/recognition/test_training_free_consistency_audit.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_training_free_consistency_execution.py](../tests/recognition/test_training_free_consistency_execution.py) | Boundary fixtures; no substituted outputs qualify a scientific experiment. | See source/tests |
| [tests/recognition/test_training_free_consistency_router.py](../tests/recognition/test_training_free_consistency_router.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_two_track_diagnostics.py](../tests/recognition/test_two_track_diagnostics.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_unet_corrected_token_audit.py](../tests/recognition/test_unet_corrected_token_audit.py) | No module docstring; see linked source and tests | unet-corrected-token-router |
| [tests/recognition/test_unet_corrected_token_interfaces.py](../tests/recognition/test_unet_corrected_token_interfaces.py) | No module docstring; see linked source and tests | unet-corrected-token-router |
| [tests/recognition/test_unet_expert_records.py](../tests/recognition/test_unet_expert_records.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_unet_fallback_coverage.py](../tests/recognition/test_unet_fallback_coverage.py) | Missing-text coverage must depend on native predicted geometry, never labels. | unet-minimum-fallback |
| [tests/recognition/test_unet_fallback_transfer.py](../tests/recognition/test_unet_fallback_transfer.py) | Regression for array/sample alignment during frozen-router transfer. | unet-minimum-fallback |
| [tests/recognition/test_unet_minimum_fallback.py](../tests/recognition/test_unet_minimum_fallback.py) | Scientific checks for minimum-cost spatial word fallback. | unet-minimum-fallback |
| [tests/recognition/test_unet_token_routing.py](../tests/recognition/test_unet_token_routing.py) | Scientific boundaries for frozen U-Net token routing and transferred cuts. | unet-token-router |
| [tests/recognition/test_uniform_encoder_context.py](../tests/recognition/test_uniform_encoder_context.py) | No module docstring; see linked source and tests | shared-pipeline-code-tour |
| [tests/recognition/test_uniform_fallback.py](../tests/recognition/test_uniform_fallback.py) | No module docstring; see linked source and tests | shared-pipeline-code-tour |
| [tests/recognition/test_uniform_lexical_risk.py](../tests/recognition/test_uniform_lexical_risk.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_uniform_multiscale_words.py](../tests/recognition/test_uniform_multiscale_words.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_uniform_observation_bank.py](../tests/recognition/test_uniform_observation_bank.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_uniform_pooled_gate.py](../tests/recognition/test_uniform_pooled_gate.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_uniform_risk_fusion.py](../tests/recognition/test_uniform_risk_fusion.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_uniform_utility_fusion.py](../tests/recognition/test_uniform_utility_fusion.py) | No module docstring; see linked source and tests | shared-pipeline-code-tour |
| [tests/recognition/test_uniform_word_admission.py](../tests/recognition/test_uniform_word_admission.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_uniform_word_union.py](../tests/recognition/test_uniform_word_union.py) | No module docstring; see linked source and tests | shared-pipeline-code-tour |
| [tests/recognition/test_validated_budget_report.py](../tests/recognition/test_validated_budget_report.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_validated_budget_variant.py](../tests/recognition/test_validated_budget_variant.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_word_slot_alphanumeric.py](../tests/recognition/test_word_slot_alphanumeric.py) | Symbol-free matching, empty denominators and normalized router consistency. | word-slot-alphanumeric |
| [tests/recognition/test_word_slot_methods.py](../tests/recognition/test_word_slot_methods.py) | Denominators, split identity, deployable gate lifecycle and coordinate tests. | word-slot-methods |
| [tests/recognition/test_word_unet_policy.py](../tests/recognition/test_word_unet_policy.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_yolo_router_comparison.py](../tests/recognition/test_yolo_router_comparison.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_yolo_scale_agreement.py](../tests/recognition/test_yolo_scale_agreement.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/recognition/test_yolo_word_refinement.py](../tests/recognition/test_yolo_word_refinement.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/research/test_comparison.py](../tests/research/test_comparison.py) | Protect scientific reporting boundaries without inference or external data. | See source/tests |
| [tests/research/test_documents.py](../tests/research/test_documents.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/research/test_library.py](../tests/research/test_library.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scoring/test_budgeted_paired.py](../tests/scoring/test_budgeted_paired.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scoring/test_budgeted_paired_e2e.py](../tests/scoring/test_budgeted_paired_e2e.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scoring/test_caption_routing_train.py](../tests/scoring/test_caption_routing_train.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scoring/test_context_candidate_ranking.py](../tests/scoring/test_context_candidate_ranking.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scoring/test_context_foundation.py](../tests/scoring/test_context_foundation.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scoring/test_local_context.py](../tests/scoring/test_local_context.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scoring/test_paper_image_router.py](../tests/scoring/test_paper_image_router.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scoring/test_parseq_mask_region_audit.py](../tests/scoring/test_parseq_mask_region_audit.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scoring/test_persistent_context_pipeline.py](../tests/scoring/test_persistent_context_pipeline.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scoring/test_routing_threshold_train.py](../tests/scoring/test_routing_threshold_train.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scoring/test_train_blip2_caption_export.py](../tests/scoring/test_train_blip2_caption_export.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_adaspot_gap_context.py](../tests/scripts/test_adaspot_gap_context.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_adaspot_gap_decoder.py](../tests/scripts/test_adaspot_gap_decoder.py) | No module docstring; see linked source and tests | adaspot-gap-decoder |
| [tests/scripts/test_adaspot_gap_fulltrain.py](../tests/scripts/test_adaspot_gap_fulltrain.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_adaspot_gap_normalized.py](../tests/scripts/test_adaspot_gap_normalized.py) | No module docstring; see linked source and tests | adaspot-gap-normalized |
| [tests/scripts/test_adaspot_gap_ocr.py](../tests/scripts/test_adaspot_gap_ocr.py) | Frozen three-arm crops, exact matching, resume and exported-map integrity. | See source/tests |
| [tests/scripts/test_adaspot_gap_ocr_selection.py](../tests/scripts/test_adaspot_gap_ocr_selection.py) | No module docstring; see linked source and tests | adaspot-gap-ocr-selection |
| [tests/scripts/test_adaspot_gap_probe.py](../tests/scripts/test_adaspot_gap_probe.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_adaspot_gap_runner.py](../tests/scripts/test_adaspot_gap_runner.py) | Exact checkpoint/RNG resume, isolation, denominator and failure boundaries. | See source/tests |
| [tests/scripts/test_adaspot_joint_runner.py](../tests/scripts/test_adaspot_joint_runner.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_adaspot_mask_align_runner.py](../tests/scripts/test_adaspot_mask_align_runner.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_adaspot_phase0.py](../tests/scripts/test_adaspot_phase0.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_audit_deepsolo_word_f1.py](../tests/scripts/test_audit_deepsolo_word_f1.py) | No module docstring; see linked source and tests | deepsolo-word-f1-study |
| [tests/scripts/test_audit_icdar2015_word_f1.py](../tests/scripts/test_audit_icdar2015_word_f1.py) | Synthetic artifact boundary test for the independent scientific auditor. | icdar2015-word-f1 |
| [tests/scripts/test_audit_icdar_budget_router.py](../tests/scripts/test_audit_icdar_budget_router.py) | Tests for independent route denominators and fixed-cardinality certificates. | icdar-trained-budget-router |
| [tests/scripts/test_cached_totaltext_padding.py](../tests/scripts/test_cached_totaltext_padding.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_center_vote_training_launcher.py](../tests/scripts/test_center_vote_training_launcher.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_cleanup_failed_unet_detection_run.py](../tests/scripts/test_cleanup_failed_unet_detection_run.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_collaboration.py](../tests/scripts/test_collaboration.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_compact_v2_selected_ocr.py](../tests/scripts/test_compact_v2_selected_ocr.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_completed_checkpoint_deduplication.py](../tests/scripts/test_completed_checkpoint_deduplication.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_deepsolo_short_words.py](../tests/scripts/test_deepsolo_short_words.py) | No module docstring; see linked source and tests | deepsolo-short-words |
| [tests/scripts/test_deepsolo_textocr_comparison.py](../tests/scripts/test_deepsolo_textocr_comparison.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_deepsolo_word_f1_fusion.py](../tests/scripts/test_deepsolo_word_f1_fusion.py) | No module docstring; see linked source and tests | deepsolo-word-f1-study |
| [tests/scripts/test_deepsolo_word_f1_study.py](../tests/scripts/test_deepsolo_word_f1_study.py) | No module docstring; see linked source and tests | deepsolo-word-f1-study |
| [tests/scripts/test_diagnose_totaltext_original_gt_components_parseq.py](../tests/scripts/test_diagnose_totaltext_original_gt_components_parseq.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_diagnose_unet_crop_refinement.py](../tests/scripts/test_diagnose_unet_crop_refinement.py) | Runner evidence, preprocessing, isolation and failure-contract tests. | See source/tests |
| [tests/scripts/test_evaluate_center_guided_crops.py](../tests/scripts/test_evaluate_center_guided_crops.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_evaluate_icdar2015_transfer_word_f1.py](../tests/scripts/test_evaluate_icdar2015_transfer_word_f1.py) | ICDAR2015 TEST proposal, coordinate and evaluation configuration checks. | icdar2015-word-f1 |
| [tests/scripts/test_evaluate_text_recall_crops.py](../tests/scripts/test_evaluate_text_recall_crops.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_evaluate_text_recall_learned_crops.py](../tests/scripts/test_evaluate_text_recall_learned_crops.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_evaluate_text_recall_refinements.py](../tests/scripts/test_evaluate_text_recall_refinements.py) | No module docstring; see linked source and tests | unet-detection-methods |
| [tests/scripts/test_evaluate_totaltext_highres_routing.py](../tests/scripts/test_evaluate_totaltext_highres_routing.py) | Boundary tests for transfer routing; no GPU, downloads or new raster files. | See source/tests |
| [tests/scripts/test_evaluate_unet_word_detection_test.py](../tests/scripts/test_evaluate_unet_word_detection_test.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_export_scene_text_algorithm1_crops.py](../tests/scripts/test_export_scene_text_algorithm1_crops.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_finish_icdar13_word_refit.py](../tests/scripts/test_finish_icdar13_word_refit.py) | Explicit empty-read rejection and raw observation preservation. | icdar13-word-refit |
| [tests/scripts/test_guarded_v3_graph_cut.py](../tests/scripts/test_guarded_v3_graph_cut.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_guarded_v3_training_ocr.py](../tests/scripts/test_guarded_v3_training_ocr.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_icdar13_transfer_duration.py](../tests/scripts/test_icdar13_transfer_duration.py) | No module docstring; see linked source and tests | icdar13-transfer-duration |
| [tests/scripts/test_icdar2015_improvement_evaluation.py](../tests/scripts/test_icdar2015_improvement_evaluation.py) | Dynamic-scale component/crop geometry and approximate evaluation isolation. | icdar2015-unet-improvement |
| [tests/scripts/test_icdar2015_improvement_failures.py](../tests/scripts/test_icdar2015_improvement_failures.py) | GPU failures produce explicit failed artifacts and propagate to the caller. | icdar2015-unet-improvement |
| [tests/scripts/test_icdar2015_improvement_selection.py](../tests/scripts/test_icdar2015_improvement_selection.py) | Predeclared TRAIN-only resolution and epoch tie breaks. | icdar2015-unet-improvement |
| [tests/scripts/test_icdar2015_target70_device_equivalence.py](../tests/scripts/test_icdar2015_target70_device_equivalence.py) | Device qualification fails on changed words or filter membership. | See source/tests |
| [tests/scripts/test_icdar2015_target70_final_selection.py](../tests/scripts/test_icdar2015_target70_final_selection.py) | Completion gates and validation-only selection; fixtures are not model results. | See source/tests |
| [tests/scripts/test_icdar2015_target70_independent_audit.py](../tests/scripts/test_icdar2015_target70_independent_audit.py) | Independent population/filter replay includes empty outputs and all GT. | See source/tests |
| [tests/scripts/test_icdar2015_target70_oracle_geometry.py](../tests/scripts/test_icdar2015_target70_oracle_geometry.py) | Perfect-localization diagnostics must retain RGB corner geometry. | icdar2015-target70 |
| [tests/scripts/test_icdar2015_target70_test_evaluation.py](../tests/scripts/test_icdar2015_target70_test_evaluation.py) | Unchanged TEST populations, exact qualification binding and raw-cache integrity. | See source/tests |
| [tests/scripts/test_infer_deepsolo_word_f1.py](../tests/scripts/test_infer_deepsolo_word_f1.py) | No module docstring; see linked source and tests | deepsolo-word-f1-study |
| [tests/scripts/test_kernel_only_fresh_runner.py](../tests/scripts/test_kernel_only_fresh_runner.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_kernel_only_padding.py](../tests/scripts/test_kernel_only_padding.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_kernel_only_training_runner.py](../tests/scripts/test_kernel_only_training_runner.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_mask_crop_three_tests.py](../tests/scripts/test_mask_crop_three_tests.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_mask_to_crop_failures.py](../tests/scripts/test_mask_to_crop_failures.py) | Failure categories, padding oracle identities, and small-word edit counts. | See source/tests |
| [tests/scripts/test_native128_crop_limit.py](../tests/scripts/test_native128_crop_limit.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_native128_failures.py](../tests/scripts/test_native128_failures.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_native128_review.py](../tests/scripts/test_native128_review.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_package_totaltext_highres_residual_pad0_review.py](../tests/scripts/test_package_totaltext_highres_residual_pad0_review.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_package_totaltext_letterbox_failure_examples.py](../tests/scripts/test_package_totaltext_letterbox_failure_examples.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_package_totaltext_original_gt_triple_failures.py](../tests/scripts/test_package_totaltext_original_gt_triple_failures.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_package_totaltext_v4_gt_crop_failures.py](../tests/scripts/test_package_totaltext_v4_gt_crop_failures.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_package_totaltext_v4_resolution256_failures.py](../tests/scripts/test_package_totaltext_v4_resolution256_failures.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_prepare_ctw1500_recognition.py](../tests/scripts/test_prepare_ctw1500_recognition.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_prune_compact_v2_checkpoints.py](../tests/scripts/test_prune_compact_v2_checkpoints.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_prune_segmentation_checkpoints.py](../tests/scripts/test_prune_segmentation_checkpoints.py) | Safety checks for completed-run checkpoint pruning. | See source/tests |
| [tests/scripts/test_repository_language.py](../tests/scripts/test_repository_language.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_rescore_totaltext_highres_padding_symbols.py](../tests/scripts/test_rescore_totaltext_highres_padding_symbols.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_research_synthesis.py](../tests/scripts/test_research_synthesis.py) | Scientific reporting: paired populations, micro metrics and matching bounds. | See source/tests |
| [tests/scripts/test_scene_text_yolo_thresholds.py](../tests/scripts/test_scene_text_yolo_thresholds.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_scene_text_yolo_transfer.py](../tests/scripts/test_scene_text_yolo_transfer.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_sweep_totaltext_v4_gt_mask_resolution.py](../tests/scripts/test_sweep_totaltext_v4_gt_mask_resolution.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_sweep_totaltext_v4_predicted_mask_resolution.py](../tests/scripts/test_sweep_totaltext_v4_predicted_mask_resolution.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_synthtext_download.py](../tests/scripts/test_synthtext_download.py) | Offline engineering fixtures, never genuine SynthText qualification. | synthtext-download, unet-detection-methods |
| [tests/scripts/test_text_detector_suite.py](../tests/scripts/test_text_detector_suite.py) | Scientific mode, artifact integrity and GT-blind crop selection boundaries. | See source/tests |
| [tests/scripts/test_text_recall_resolution.py](../tests/scripts/test_text_recall_resolution.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_totaltext_bert_correction.py](../tests/scripts/test_totaltext_bert_correction.py) | Frozen-source/schema tests; fixture reports do not represent model performance. | See source/tests |
| [tests/scripts/test_totaltext_bert_methods.py](../tests/scripts/test_totaltext_bert_methods.py) | Frozen method-battery and caption boundaries; pretrained scores are not simulated. | See source/tests |
| [tests/scripts/test_totaltext_bert_padded_evidence.py](../tests/scripts/test_totaltext_bert_padded_evidence.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_totaltext_bert_visual_correction.py](../tests/scripts/test_totaltext_bert_visual_correction.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_totaltext_frequency_gated_skip_parseq_val.py](../tests/scripts/test_totaltext_frequency_gated_skip_parseq_val.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_totaltext_highres_contour_padding.py](../tests/scripts/test_totaltext_highres_contour_padding.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_totaltext_neuspell.py](../tests/scripts/test_totaltext_neuspell.py) | Frozen-source and model boundary tests; synthetic weights imply no accuracy claim. | See source/tests |
| [tests/scripts/test_totaltext_v4_perfect_mask_localization_train.py](../tests/scripts/test_totaltext_v4_perfect_mask_localization_train.py) | Accounting tests for the TRAIN perfect-target localization ceiling. | See source/tests |
| [tests/scripts/test_totaltext_word_f1_separator_parseq_val.py](../tests/scripts/test_totaltext_word_f1_separator_parseq_val.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_unet_auxiliary_ocr_audit.py](../tests/scripts/test_unet_auxiliary_ocr_audit.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_unet_auxiliary_ocr_runner.py](../tests/scripts/test_unet_auxiliary_ocr_runner.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_unet_crop_links_runner.py](../tests/scripts/test_unet_crop_links_runner.py) | Config isolation, exact checkpoint resume, paired reporting and fallback behavior. | See source/tests |
| [tests/scripts/test_unet_detection_runner.py](../tests/scripts/test_unet_detection_runner.py) | No module docstring; see linked source and tests | unet-detection-methods |
| [tests/scripts/test_unet_seeded_selective_runner.py](../tests/scripts/test_unet_seeded_selective_runner.py) | Audit boundaries: dataset denominator, cache backend, smoke mode and OOM. | See source/tests |
| [tests/scripts/test_unet_word_research_runner.py](../tests/scripts/test_unet_word_research_runner.py) | Resume, provenance, empty outputs and OCR protocol qualification. | See source/tests |
| [tests/scripts/test_unet_word_research_v2_runner.py](../tests/scripts/test_unet_word_research_v2_runner.py) | Phase-aware resume, config gating, source qualification and frozen OCR. | See source/tests |
| [tests/scripts/test_uniform_decoder_folds.py](../tests/scripts/test_uniform_decoder_folds.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_uniform_utility_supervision.py](../tests/scripts/test_uniform_utility_supervision.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_word_center_offset_training_launcher.py](../tests/scripts/test_word_center_offset_training_launcher.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_word_centerline_ocr.py](../tests/scripts/test_word_centerline_ocr.py) | No module docstring; see linked source and tests | word-centerline |
| [tests/scripts/test_word_centerline_runner.py](../tests/scripts/test_word_centerline_runner.py) | Resume, population, crop-export and scientific-mode boundaries. | word-centerline |
| [tests/scripts/test_word_instance_extensions_runner.py](../tests/scripts/test_word_instance_extensions_runner.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_word_multitask_crops.py](../tests/scripts/test_word_multitask_crops.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_yolo_direct_parseq.py](../tests/scripts/test_yolo_direct_parseq.py) | Coordinate, evaluation and mode boundaries of the direct YOLO crop experiment. | See source/tests |
| [tests/scripts/test_yolo_four_dataset_comparison.py](../tests/scripts/test_yolo_four_dataset_comparison.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_yolo_router_comparison_runner.py](../tests/scripts/test_yolo_router_comparison_runner.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/scripts/test_yolo_word_slots.py](../tests/scripts/test_yolo_word_slots.py) | Identity and geometry boundaries for the lightweight YOLO word-slot run. | See source/tests |
| [tests/segmentation/conftest.py](../tests/segmentation/conftest.py) | Test-local isolation for process-global PyTorch backend settings. | See source/tests |
| [tests/segmentation/test_adaspot_gap.py](../tests/segmentation/test_adaspot_gap.py) | Gap target geometry, paired augmentation, masked gradients and optimizer safety. | See source/tests |
| [tests/segmentation/test_author_protocol.py](../tests/segmentation/test_author_protocol.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_center_vote_objective.py](../tests/segmentation/test_center_vote_objective.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_ctw1500_word_gt.py](../tests/segmentation/test_ctw1500_word_gt.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_evaluation.py](../tests/segmentation/test_evaluation.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_evaluation_trajectory.py](../tests/segmentation/test_evaluation_trajectory.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_export_train_native_probabilities.py](../tests/segmentation/test_export_train_native_probabilities.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_guarded_word_gt.py](../tests/segmentation/test_guarded_word_gt.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_ic13_unet_oof_splits.py](../tests/segmentation/test_ic13_unet_oof_splits.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_icdar13_matched_training.py](../tests/segmentation/test_icdar13_matched_training.py) | Source-loss fidelity, invalid-pixel gradients, emissions and split isolation. | icdar13-matched-word-training |
| [tests/segmentation/test_icdar13_word_refit.py](../tests/segmentation/test_icdar13_word_refit.py) | Word-mask identity, strict source compatibility and partial-encoder gradients. | icdar13-word-refit |
| [tests/segmentation/test_icdar2015_encoder_decay.py](../tests/segmentation/test_icdar2015_encoder_decay.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_icdar2015_improvement_targets.py](../tests/segmentation/test_icdar2015_improvement_targets.py) | Word-mask separation and geometry across explicit ICDAR2015 resolutions. | icdar2015-unet-improvement |
| [tests/segmentation/test_icdar2015_multiview.py](../tests/segmentation/test_icdar2015_multiview.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_icdar2015_ohem_loss.py](../tests/segmentation/test_icdar2015_ohem_loss.py) | Hard negatives must affect only the declared pixels and preserve empty images. | See source/tests |
| [tests/segmentation/test_icdar2015_ohem_training_io.py](../tests/segmentation/test_icdar2015_ohem_training_io.py) | Omitting unused owners must preserve used tensors/RNG and accumulated updates. | See source/tests |
| [tests/segmentation/test_icdar2015_operating_crops.py](../tests/segmentation/test_icdar2015_operating_crops.py) | GT-free crop generation preserves components, original boxes and tiny words. | icdar2015-target70 |
| [tests/segmentation/test_icdar2015_seed_partition.py](../tests/segmentation/test_icdar2015_seed_partition.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_icdar2015_synthetic_training.py](../tests/segmentation/test_icdar2015_synthetic_training.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_icdar2015_synthtext_cache.py](../tests/segmentation/test_icdar2015_synthtext_cache.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_icdar2015_target70.py](../tests/segmentation/test_icdar2015_target70.py) | Target70 preserves metrics and qualifies new supervision/update semantics. | icdar2015-target70 |
| [tests/segmentation/test_icdar2015_totaltext_transfer.py](../tests/segmentation/test_icdar2015_totaltext_transfer.py) | Scientific boundaries for ICDAR2015 targets and the Total-Text U-Net transfer. | icdar2015-totaltext-transfer |
| [tests/segmentation/test_icdar2015_totaltext_weights.py](../tests/segmentation/test_icdar2015_totaltext_weights.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_icdar2015_word_size.py](../tests/segmentation/test_icdar2015_word_size.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_kernel_only_training.py](../tests/segmentation/test_kernel_only_training.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_mask_probability_refiner.py](../tests/segmentation/test_mask_probability_refiner.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_metrics.py](../tests/segmentation/test_metrics.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_pipeline_smoke.py](../tests/segmentation/test_pipeline_smoke.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_python38_compatibility.py](../tests/segmentation/test_python38_compatibility.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_scene_text_preparation.py](../tests/segmentation/test_scene_text_preparation.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_separated_word_gt.py](../tests/segmentation/test_separated_word_gt.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_separated_word_training.py](../tests/segmentation/test_separated_word_training.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_text_recall.py](../tests/segmentation/test_text_recall.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_text_recall_provenance.py](../tests/segmentation/test_text_recall_provenance.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_tighter_word_gt.py](../tests/segmentation/test_tighter_word_gt.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_totaltext_adaptive_500_epoch_config.py](../tests/segmentation/test_totaltext_adaptive_500_epoch_config.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_totaltext_adaptive_word_boxes.py](../tests/segmentation/test_totaltext_adaptive_word_boxes.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_totaltext_curved_box_v4.py](../tests/segmentation/test_totaltext_curved_box_v4.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_totaltext_curved_box_v4_training.py](../tests/segmentation/test_totaltext_curved_box_v4_training.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_totaltext_image_frequency_ablation.py](../tests/segmentation/test_totaltext_image_frequency_ablation.py) | Focused CPU checks for isolated Total-Text image/frequency interventions. | See source/tests |
| [tests/segmentation/test_totaltext_letterbox_500_separator_pair.py](../tests/segmentation/test_totaltext_letterbox_500_separator_pair.py) | Two frozen letterbox continuation branches; CPU-only unit checks. | See source/tests |
| [tests/segmentation/test_totaltext_letterbox_500_words_train.py](../tests/segmentation/test_totaltext_letterbox_500_words_train.py) | TRAIN word-identity comparison and checkpoint reuse guards; CPU only. | See source/tests |
| [tests/segmentation/test_totaltext_letterbox_pad0_test.py](../tests/segmentation/test_totaltext_letterbox_pad0_test.py) | CPU-only checks for frozen letterbox TEST evaluation and restart provenance. | See source/tests |
| [tests/segmentation/test_totaltext_letterbox_threshold_sweep_train.py](../tests/segmentation/test_totaltext_letterbox_threshold_sweep_train.py) | TRAIN-only threshold diagnostic invariants; no CUDA inference. | See source/tests |
| [tests/segmentation/test_totaltext_partial_encoder.py](../tests/segmentation/test_totaltext_partial_encoder.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_totaltext_v4_highres_ablation.py](../tests/segmentation/test_totaltext_v4_highres_ablation.py) | Isolated high-resolution practical ablation contracts. | See source/tests |
| [tests/segmentation/test_totaltext_word_ablation.py](../tests/segmentation/test_totaltext_word_ablation.py) | CPU fixtures for instance adjacency, paired crops, losses and provenance. | See source/tests |
| [tests/segmentation/test_totaltext_word_f1_separator_head.py](../tests/segmentation/test_totaltext_word_f1_separator_head.py) | Focused CPU checks for the isolated practical word-F1/separator ablation. | See source/tests |
| [tests/segmentation/test_totaltext_word_kernel.py](../tests/segmentation/test_totaltext_word_kernel.py) | No module docstring; see linked source and tests | directional-transport-results, unet-detection-methods |
| [tests/segmentation/test_totaltext_word_regions.py](../tests/segmentation/test_totaltext_word_regions.py) | No module docstring; see linked source and tests | unet-detection-methods |
| [tests/segmentation/test_training.py](../tests/segmentation/test_training.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_training_completion.py](../tests/segmentation/test_training_completion.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_unet_auxiliary_ocr_supervision.py](../tests/segmentation/test_unet_auxiliary_ocr_supervision.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_unet_crop_links.py](../tests/segmentation/test_unet_crop_links.py) | Predicted-window supervision, reciprocal connectivity, and crop coordinates. | See source/tests |
| [tests/segmentation/test_unet_dense_instances.py](../tests/segmentation/test_unet_dense_instances.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_unet_detection_targets.py](../tests/segmentation/test_unet_detection_targets.py) | No module docstring; see linked source and tests | unet-detection-methods |
| [tests/segmentation/test_unet_word_box_field.py](../tests/segmentation/test_unet_word_box_field.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_unet_word_research.py](../tests/segmentation/test_unet_word_research.py) | Scientific boundaries of paired glyph/word supervision and new heads. | See source/tests |
| [tests/segmentation/test_unet_word_research_v2.py](../tests/segmentation/test_unet_word_research_v2.py) | Corrective instance geometry, protected warmup and explicit label semantics. | See source/tests |
| [tests/segmentation/test_uniform_encoder_adaptation.py](../tests/segmentation/test_uniform_encoder_adaptation.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_uniform_instance_objective.py](../tests/segmentation/test_uniform_instance_objective.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_uniform_multiscale_masks.py](../tests/segmentation/test_uniform_multiscale_masks.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_uniform_photometric_training.py](../tests/segmentation/test_uniform_photometric_training.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_uniform_unet_fusion.py](../tests/segmentation/test_uniform_unet_fusion.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_word_center_offsets.py](../tests/segmentation/test_word_center_offsets.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_word_centerline.py](../tests/segmentation/test_word_centerline.py) | Geometry, supervised gradients, and frozen-source checks for line targets. | word-centerline |
| [tests/segmentation/test_word_checkpoint_metrics.py](../tests/segmentation/test_word_checkpoint_metrics.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_word_evidence_suite.py](../tests/segmentation/test_word_evidence_suite.py) | Scientific boundaries for the three frozen Text+Kernel evidence tests. | word-evidence-suite |
| [tests/segmentation/test_word_instance_extensions.py](../tests/segmentation/test_word_instance_extensions.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/segmentation/test_word_multitask.py](../tests/segmentation/test_word_multitask.py) | CPU scientific boundary tests for the additive word-head experiment. | unet-detection-methods, word-multitask, word-multitask-results-v1 |
| [tests/segmentation/test_word_multitask_provenance.py](../tests/segmentation/test_word_multitask_provenance.py) | Contract serialization and complete launcher-stage boundary regressions. | word-multitask-results-v1 |
| [tests/test_config.py](../tests/test_config.py) | No module docstring; see linked source and tests | See source/tests |
| [tests/test_main_environment.py](../tests/test_main_environment.py) | No module docstring; see linked source and tests | word-multitask |
| [tests/test_package_imports.py](../tests/test_package_imports.py) | No module docstring; see linked source and tests | See source/tests |
