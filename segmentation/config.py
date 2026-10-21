"""Mode-isolated configuration validation for segmentation experiments."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


class SegmentationConfigError(ValueError):
    pass


class StrictSegmentationBlocked(SegmentationConfigError):
    pass


def load_segmentation_config(path: str | Path) -> DictConfig:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise SegmentationConfigError(f"Segmentation config does not exist: {config_path}")
    config = OmegaConf.load(config_path)
    if not isinstance(config, DictConfig):
        raise SegmentationConfigError("Segmentation config root must be a mapping")
    if config.get("experiment_kind") != "segmentation":
        raise SegmentationConfigError("experiment_kind must be 'segmentation'")
    mode = config.get("reproduction_mode")
    if mode not in ("strict", "approximate"):
        raise SegmentationConfigError("Segmentation mode must be strict or approximate")
    if config_path.parent.name in ("strict", "approximate") and config_path.parent.name != mode:
        raise SegmentationConfigError("Config namespace and reproduction_mode do not match")
    expected_label = f"{mode} reproduction"
    if config.get("claim_label") != expected_label:
        raise SegmentationConfigError(f"{mode} mode requires claim_label '{expected_label}'")
    if mode == "strict":
        blockers = OmegaConf.select(config, "strict_blockers", default=[])
        if config.get("runnable") is not False or not blockers:
            raise SegmentationConfigError("Strict segmentation must preserve an actionable blocker list")
        raise StrictSegmentationBlocked(
            "STRICT segmentation is blocked before execution: " + ", ".join(str(item) for item in blockers)
        )
    if config.get("runnable") is not True:
        raise SegmentationConfigError("Approximate segmentation config must explicitly be runnable")
    author_protocol = (
        OmegaConf.select(config, "training.protocol", default=None)
        == "author_clarified_icdar13_v1"
        or OmegaConf.select(config, "architecture.implementation", default=None)
        == "author_clarified_ag_unet"
    )
    required = {
        "architecture.contract_schema_version": 1,
        "transforms.image_size": [256, 256],
        "transforms.mask_size": [128, 128],
        "localization.nmax": 10,
    }
    if author_protocol:
        required.update({
            "architecture.implementation": "author_clarified_ag_unet",
            "architecture.identity": "author-clarified-ag-unet-2048x1024-v1",
        })
    else:
        required.update({
            "architecture.implementation": "figure_derived_ag_unet",
            "architecture.identity": "approximate-ag-unet-figure3-v1",
        })
    if OmegaConf.select(config, "run.kind", default=None) != "real_evaluation":
        required.update({
            "training.optimizer": "Adam", "training.learning_rate": 1e-4,
            "training.weight_decay": 0.01, "training.batch_size": 8,
        })
        required["training.loss"] = (
            "masked_BCEWithLogitsLoss" if author_protocol else "BCEWithLogitsLoss"
        )
    for dotted, expected in required.items():
        actual = OmegaConf.select(config, dotted, default=None)
        if actual != expected:
            raise SegmentationConfigError(f"{dotted} must be {expected!r}; got {actual!r}")
    choices = config.get("scientific_choices")
    if not choices or any(
        choice.get("classification") not in (
            "PAPER_SPECIFIED", "ASSUMPTION", "REPOSITORY_SPECIFIED"
        )
        for choice in choices
    ):
        raise SegmentationConfigError("Approximate choices require supported classifications")
    if OmegaConf.select(config, "run.kind", default=None) == "real_training":
        _validate_real_training_config(config)
    if OmegaConf.select(config, "run.kind", default=None) == "real_evaluation":
        _validate_real_evaluation_config(config)
    return config


def _validate_real_evaluation_config(config: DictConfig) -> None:
    if OmegaConf.select(config, "architecture.implementation", default=None) == (
        "author_clarified_ag_unet"
    ):
        _validate_author_protocol_evaluation_config(config)
        return

    required = {
        "run.result_class": "approximate reproduction", "evaluation.batch_size": 1,
        "checkpoint.artifact_id": "ART-ICDAR13-M9-APPROX-TRAINED-CHECKPOINT",
        "checkpoint.sha256": "335a003ae00cf21d159151b9568e10eafde12d6ae524bf7505e1d62ae4c3bfe2",
        "data.test.manifest_sha256": "8bffaa558f687d5ae6a6e41a30a4a51db682124da99d4c2496aa3fa4a2a001d8",
        "evaluation.probability_activation": "sigmoid", "evaluation.probability_threshold": 0.5,
        "evaluation.threshold_comparison": "greater_than_or_equal",
        "evaluation.threshold_resolution": "native_128x128_logits",
        "evaluation.binary_mask_backprojection": "nearest",
        "evaluation.evaluation_resolution": "original_gt_hxw",
        "evaluation.gt_conversion": "icdar13-task2.2-exact-white-background-v1",
        "evaluation.dont_care": "foreground", "evaluation.valid_mask": "all_pixels",
        "evaluation.primary_aggregation": "per_image_macro",
        "evaluation.diagnostic_aggregation": "global_micro",
        "evaluation.empty_foreground_policy": "perfect_if_both_empty",
    }
    for dotted, expected in required.items():
        actual = OmegaConf.select(config, dotted, default=None)
        if actual != expected:
            raise SegmentationConfigError(f"{dotted} must be {expected!r}; got {actual!r}")


def _validate_author_protocol_evaluation_config(config: DictConfig) -> None:
    required = {
        "run.result_class": "approximate reproduction",
        "architecture.identity": "author-clarified-ag-unet-2048x1024-v1",
        "architecture.freeze_encoder": True,
        "architecture.frozen_batchnorm_state": True,
        "architecture.bottleneck_channels": 1024,
        "architecture.decoder_channels": [512, 256, 128, 64],
        "checkpoint.artifact_id": (
            "ART-ICDAR13-AUTHOR-PROTOCOL-APPROX-TRAINED-CHECKPOINT"
        ),
        "checkpoint.selection": "training_foreground_iou",
        "evaluation.batch_size": 1,
        "evaluation.probability_activation": "sigmoid",
        "evaluation.probability_threshold": 0.5,
        "evaluation.threshold_comparison": "greater_than_or_equal",
        "evaluation.threshold_resolution": "native_128x128_logits",
        "evaluation.binary_mask_backprojection": "none",
        "evaluation.evaluation_resolution": "native_128x128",
        "evaluation.evaluation_resolution_classification": "ASSUMPTION",
        "evaluation.gt_resize_interpolation": "bilinear",
        "evaluation.gt_binarization_threshold": 0.5,
        "evaluation.validity_resize_interpolation": "nearest",
        "evaluation.primary_aggregation": "per_image_macro",
        "evaluation.diagnostic_aggregation": "global_micro",
        "evaluation.empty_foreground_policy": "perfect_if_both_empty",
        "evaluation.dont_care_classification": "ASSUMPTION",
    }
    for dotted, expected in required.items():
        actual = OmegaConf.select(config, dotted, default=None)
        if actual != expected:
            raise SegmentationConfigError(f"{dotted} must be {expected!r}; got {actual!r}")
    dont_care = OmegaConf.select(config, "evaluation.dont_care", default=None)
    gt_conversion = OmegaConf.select(config, "evaluation.gt_conversion", default=None)
    valid_mask = OmegaConf.select(config, "evaluation.valid_mask", default=None)
    allowed_mask_protocols = {
        (
            "foreground",
            "icdar13-task2.2-exact-white-background-v1",
            "all_pixels",
        ),
        (
            "excluded_by_validity_mask",
            "icdar13-task2.2-author-validity-mask-v1",
            "paired_txt_colors",
        ),
        (
            "excluded_by_validity_mask",
            "icdar13-task2.2-task1-box-zero-overlap-validity-v1",
            "task1_care_box_zero_overlap_colors",
        ),
    }
    if (dont_care, gt_conversion, valid_mask) not in allowed_mask_protocols:
        raise SegmentationConfigError(
            "Author evaluation requires a complete foreground/all-pixels or "
            "explicit validity-mask Don\'t Care exclusion protocol"
        )
    data_contracts = {
        (
            "foreground",
            "icdar13-task2.2-exact-white-background-v1",
            "all_pixels",
        ): (
            "ART-ICDAR13-M10-APPROX-TEST-MANIFEST",
            "8bffaa558f687d5ae6a6e41a30a4a51db682124da99d4c2496aa3fa4a2a001d8",
        ),
        (
            "excluded_by_validity_mask",
            "icdar13-task2.2-task1-box-zero-overlap-validity-v1",
            "task1_care_box_zero_overlap_colors",
        ): (
            "ART-ICDAR13-M10-TASK1-OVERLAP-IGNORE-TEST-MANIFEST",
            "7e9267e39fd439a28374875c250bacef9a1c1a6d3a0b98c8647dcac7e69f7d15",
        ),
    }
    expected_data_contract = data_contracts.get(
        (dont_care, gt_conversion, valid_mask)
    )
    if expected_data_contract is not None:
        expected_artifact, expected_manifest_sha256 = expected_data_contract
        if config.data.test.artifact_id != expected_artifact:
            raise SegmentationConfigError(
                "data.test.artifact_id must be {!r}; got {!r}".format(
                    expected_artifact, config.data.test.artifact_id
                )
            )
        if config.data.test.manifest_sha256 != expected_manifest_sha256:
            raise SegmentationConfigError(
                "data.test.manifest_sha256 must be {!r}; got {!r}".format(
                    expected_manifest_sha256, config.data.test.manifest_sha256
                )
            )
    for dotted in (
        "checkpoint.sha256",
        "checkpoint.training_run_manifest_sha256",
        "data.test.manifest_sha256",
    ):
        digest = OmegaConf.select(config, dotted, default=None)
        if not isinstance(digest, str) or len(digest) != 64:
            raise SegmentationConfigError(f"{dotted} must be an explicit SHA256 digest")
        try:
            int(digest, 16)
        except ValueError as error:
            raise SegmentationConfigError(f"{dotted} must be hexadecimal") from error


def _validate_real_training_config(config: DictConfig) -> None:
    """Reject incomplete or silently adapted M9 real-training configurations."""

    protocol = OmegaConf.select(config, "training.protocol", default=None)
    if protocol == "author_clarified_icdar13_v1":
        _validate_author_protocol_training_config(config)
        return
    if protocol in (
        "paper_hyperparameters_scene_text_v1",
        "paper_hyperparameters_multidataset_balanced_v1",
        "practical_totaltext_adaptive_word_boxes_500_v1",
        "practical_totaltext_curved_word_boxes_500_v1",
        "practical_totaltext_separated_words_500_v1",
        "practical_totaltext_tighter_words_500_v2",
    ):
        _validate_scene_text_training_config(config, protocol=str(protocol))
        return

    required = {
        "run.result_class": "approximate reproduction",
        "architecture.encoder_weights": "IMAGENET1K_V2",
        "architecture.encoder_weights_artifact_id": "ART-RESNET50-IMAGENET1K-V2-CANDIDATE",
        "training.precision": "float32",
        "training.num_workers": 0,
        "training.drop_last": False,
        "validation.enabled": True,
        "validation.role": "validation",
        "validation.source": "held_out_subset_of_official_training_split",
        "validation.checkpoint_semantics": "metrics_only_last_checkpoint",
        "validation.leakage_policy": "disjoint_sample_ids_and_resolved_paths",
        "data.split_policy_artifact_id": "ART-ICDAR13-M9-APPROX-SPLIT-POLICY",
        "data.split_policy_fingerprint_sha256": (
            "3a7c3d4226e735b2e4b57fef5f8c84fa9522ad1b60852d86229a975ff07d6124"
        ),
        "data.training_mask_policy_artifact_id": (
            "ART-ICDAR13-M9-APPROX-TRAINING-MASK-POLICY"
        ),
        "data.training_mask_policy_fingerprint_sha256": (
            "ca318685a8fea2a0c162d80986acb88d60ec950fdec440652ff4d57bf4c0adf3"
        ),
    }
    for dotted, expected in required.items():
        actual = OmegaConf.select(config, dotted, default=None)
        if actual != expected:
            raise SegmentationConfigError(f"{dotted} must be {expected!r}; got {actual!r}")
    for dotted in (
        "run.name",
        "data.train.manifest",
        "data.train.artifact_id",
        "data.validation.manifest",
        "data.validation.artifact_id",
    ):
        value = OmegaConf.select(config, dotted, default=None)
        if not isinstance(value, str) or not value:
            raise SegmentationConfigError(f"{dotted} must be an explicit non-empty string")
    if OmegaConf.select(config, "data.train.role", default=None) != "train":
        raise SegmentationConfigError("data.train.role must be 'train'")
    epochs = OmegaConf.select(config, "training.epochs", default=None)
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 2:
        raise SegmentationConfigError(
            "M9 real training requires an explicit multi-epoch count >= 2"
        )
    if OmegaConf.select(config, "validation.protocol_classification", default=None) != "ASSUMPTION":
        raise SegmentationConfigError(
            "The unresolved validation protocol must remain an explicit ASSUMPTION"
        )


def _validate_author_protocol_training_config(config: DictConfig) -> None:
    required = {
        "run.result_class": "approximate reproduction",
        "architecture.encoder_weights": "IMAGENET1K_V2",
        "architecture.freeze_encoder": True,
        "architecture.frozen_batchnorm_state": True,
        "architecture.bottleneck_channels": 1024,
        "architecture.bottleneck_width_classification": "ASSUMPTION",
        "architecture.output_stride": 2,
        "data.train.role": "train",
        "data.train.expected_sample_count": 229,
        "transforms.horizontal_flip_probability": 0.5,
        "transforms.rotation_degrees": [-15.0, 15.0],
        "transforms.rotation_degrees_classification": "ASSUMPTION",
        "transforms.mask_interpolation": "bilinear",
        "transforms.target_binarization_threshold": 0.5,
        "transforms.validity_interpolation": "nearest",
        "training.epochs": 100,
        "training.checkpoint_selection": "training_foreground_iou",
        "training.precision": "float32",
        "training.num_workers": 8,
        "training.drop_last": False,
        "checkpoint_selection.dataset": "training_all_229",
        "checkpoint_selection.metric": "foreground_iou",
        "checkpoint_selection.augmentation": "disabled",
        "checkpoint_selection.tie_break": "earliest_epoch",
        "validation.enabled": False,
        "validation.held_out_count": 0,
    }
    for dotted, expected in required.items():
        actual = OmegaConf.select(config, dotted, default=None)
        if actual != expected:
            raise SegmentationConfigError(f"{dotted} must be {expected!r}; got {actual!r}")
    for dotted in ("run.name", "data.train.manifest", "data.train.artifact_id"):
        value = OmegaConf.select(config, dotted, default=None)
        if not isinstance(value, str) or not value:
            raise SegmentationConfigError(f"{dotted} must be an explicit non-empty string")


def _validate_scene_text_training_config(config: DictConfig, *, protocol: str) -> None:
    extended_boxes = protocol in {
        "practical_totaltext_adaptive_word_boxes_500_v1",
        "practical_totaltext_curved_word_boxes_500_v1",
        "practical_totaltext_separated_words_500_v1",
        "practical_totaltext_tighter_words_500_v2",
    }
    required = {
        "run.result_class": "approximate reproduction",
        "architecture.implementation": "author_clarified_ag_unet",
        "architecture.encoder_weights": "IMAGENET1K_V2",
        "architecture.freeze_encoder": True,
        "architecture.frozen_batchnorm_state": True,
        "architecture.bottleneck_channels": 1024,
        "architecture.bottleneck_width_classification": "ASSUMPTION",
        "architecture.output_stride": 2,
        "data.train.role": "train",
        "transforms.horizontal_flip_probability": 0.5,
        "transforms.rotation_degrees": [-15.0, 15.0],
        "transforms.rotation_degrees_classification": "ASSUMPTION",
        "transforms.mask_interpolation": "bilinear",
        "transforms.target_binarization_threshold": 0.5,
        "transforms.validity_interpolation": "nearest",
        "training.epochs": 500 if extended_boxes else 100,
        "training.checkpoint_selection": "training_foreground_iou",
        "training.precision": "float32",
        "training.num_workers": 8,
        "training.drop_last": False,
        "checkpoint_selection.metric": "foreground_iou",
        "checkpoint_selection.augmentation": "disabled",
        "checkpoint_selection.tie_break": "earliest_epoch",
        "validation.enabled": False,
        "validation.held_out_count": 0,
    }
    expected_sampling = (
        "balanced_by_dataset_oversample_to_largest"
        if protocol == "paper_hyperparameters_multidataset_balanced_v1"
        else "epoch_shuffle"
    )
    required["training.sampling.strategy"] = expected_sampling
    if protocol == "practical_totaltext_adaptive_word_boxes_500_v1":
        required.update({
            "run.name": "total-text-adaptive-filled-word-boxes-v3-500ep",
            "run.variant_class": "PRACTICAL_APPROXIMATION",
            "data.train.expected_sample_count": 1255,
            "data.train.datasets": ["total_text"],
            "data.train.manifest_sha256": "bd3697b63318d6412cab8aca668ca82f67c443d1313d4b03b2e3adbd6767c9bf",
        })
    elif protocol == "practical_totaltext_curved_word_boxes_500_v1":
        required.update({
            "run.name": "total-text-filled-curved-word-boxes-v4-500ep",
            "run.variant_class": "PRACTICAL_APPROXIMATION",
            "data.train.expected_sample_count": 1255,
            "data.train.datasets": ["total_text"],
            "data.train.manifest_sha256": "7ad8525d66294e07b2d63be0dd68d08a2e83dbecc04a4d55fb65cd6110c51bea",
        })
    elif protocol == "practical_totaltext_separated_words_500_v1":
        required.update({
            "run.name": "total-text-separated-words-v1-500ep",
            "run.variant_class": "PRACTICAL_APPROXIMATION",
            "data.train.expected_sample_count": 1255,
            "data.train.datasets": ["total_text"],
            "data.train.artifact_id": "ART-TOTALTEXT-SEPARATED-WORDS-V1",
            "data.train.target_source": "v4_wider_gaps_separated_word_gt_v1",
            "training.separation_weight": 1.0,
        })
    elif protocol == "practical_totaltext_tighter_words_500_v2":
        required.update({
            "run.name": "total-text-tighter-words-v2-500ep",
            "run.variant_class": "PRACTICAL_APPROXIMATION",
            "data.train.expected_sample_count": 1255,
            "data.train.datasets": ["total_text"],
            "data.train.artifact_id": "ART-TOTALTEXT-COMPACT-WORDS-V2",
            "data.train.target_source": "compact_word_cores_v2",
            "training.separation_weight": 1.0,
        })
    for dotted, expected in required.items():
        actual = OmegaConf.select(config, dotted, default=None)
        if actual != expected:
            raise SegmentationConfigError(f"{dotted} must be {expected!r}; got {actual!r}")
    for dotted in ("run.name", "data.train.manifest", "data.train.artifact_id"):
        value = OmegaConf.select(config, dotted, default=None)
        if not isinstance(value, str) or not value:
            raise SegmentationConfigError(f"{dotted} must be an explicit non-empty string")
    sample_count = OmegaConf.select(config, "data.train.expected_sample_count", default=None)
    datasets = OmegaConf.select(config, "data.train.datasets", default=None)
    if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count < 1:
        raise SegmentationConfigError("data.train.expected_sample_count must be positive")
    if not datasets or len(set(str(value) for value in datasets)) != len(datasets):
        raise SegmentationConfigError("data.train.datasets must list unique dataset names")
    if protocol == "paper_hyperparameters_multidataset_balanced_v1" and len(datasets) != 3:
        raise SegmentationConfigError("Balanced multi-dataset protocol requires three datasets")
    manifest_digest = OmegaConf.select(config, "data.train.manifest_sha256", default=None)
    if not isinstance(manifest_digest, str) or len(manifest_digest) != 64:
        raise SegmentationConfigError("data.train.manifest_sha256 must be an explicit SHA256")
    try:
        int(manifest_digest, 16)
    except ValueError as error:
        raise SegmentationConfigError("data.train.manifest_sha256 must be hexadecimal") from error


__all__ = [
    "SegmentationConfigError",
    "StrictSegmentationBlocked",
    "load_segmentation_config",
]
