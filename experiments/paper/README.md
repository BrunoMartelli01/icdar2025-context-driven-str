# Paper baseline and scientific specification

[All families](../README.md) · [Code index](../CODE_INDEX.md) · [Saved results](../results/README.md)

Category membership is navigation, not a shared evaluation protocol or a completion claim.

## Approximate AG-UNet Implementation Decisions

ID: `ag-unet-implementation-decisions`. [Guide](<../../docs/AG_UNET_IMPLEMENTATION_DECISIONS.md>).

Code: [scripts/profile_ag_unet.py](<../../scripts/profile_ag_unet.py>), [segmentation/train.py](<../../segmentation/train.py>).

Configuration: [configs/approximate/segmentation_icdar13.yaml](<../../configs/approximate/segmentation_icdar13.yaml>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Author Clarifications and Reproduction Update

ID: `author-clarifications-and-reproduction-update`. [Guide](<../../docs/Author Clarifications and Reproduction Update.md>).

Code: [models/unet_adv.py](<../../models/unet_adv.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Recognition and Context Foundation Decisions

ID: `recognition-context-implementation-decisions`. [Guide](<../../docs/RECOGNITION_CONTEXT_IMPLEMENTATION_DECISIONS.md>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Scientific Reproduction Plan

ID: `reproduction-plan`. [Guide](<../../docs/REPRODUCTION_PLAN.md>).

Code: [localize/mask_to_blocks.py](<../../localize/mask_to_blocks.py>), [models/unet_adv.py](<../../models/unet_adv.py>), [scripts/run_table5_smoke.py](<../../scripts/run_table5_smoke.py>), [segmentation/train.py](<../../segmentation/train.py>).

Configuration: [configs/approximate/segmentation_icdar13.yaml](<../../configs/approximate/segmentation_icdar13.yaml>), [configs/approximate/segmentation_table3_icdar13.yaml](<../../configs/approximate/segmentation_table3_icdar13.yaml>), [configs/strict/segmentation_icdar13.yaml](<../../configs/strict/segmentation_icdar13.yaml>).

Tests: [tests/data/test_segmentation_transforms.py](<../../tests/data/test_segmentation_transforms.py>), [tests/localize/test_algorithm1.py](<../../tests/localize/test_algorithm1.py>), [tests/models/test_ag_unet_architecture.py](<../../tests/models/test_ag_unet_architecture.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Reproduction Specification

ID: `reproduction-spec`. [Guide](<../../docs/REPRODUCTION_SPEC.md>).

Code: [models/segmenter.py](<../../models/segmenter.py>), [pipeline/runner.py](<../../pipeline/runner.py>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.

## Strict Reproduction Status

ID: `strict-reproduction-status`. [Guide](<../../docs/STRICT_REPRODUCTION_STATUS.md>).

Results/status: read the dated guide and its source audits; no numerical comparison registered here.
