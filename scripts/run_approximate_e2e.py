#!/usr/bin/env python3
"""Run one APPROXIMATE M6 sample with a supplied segmentation mask."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from caption.blip2_captioner import BLIP2ApproxConfig, BLIP2Captioner
from deepsolo.adapter import DeepSoloDockerAdapter, DeepSoloSubprocessAdapter
from deepsolo.candidates import CANDIDATES
from deepsolo.reduction import REDUCTION_POLICIES
from localize.mask_to_blocks import APPROXIMATE_TIE_BREAK, BlockParam
from pipeline.approximate_e2e import ApproximateE2EConfig, ApproximateE2EPipeline
from recognition.parseq import (
    PARSEQ_GIT_SOURCE,
    PARSEQ_SOURCE_TYPES,
    ParseqApproxConfig,
    ParseqApproxRecognizer,
)
from scoring.semantic import MPNetApproxConfig, MPNetSemanticSimilarity
from scoring.strategies import STRATEGIES


class SuppliedMaskSegmenter:
    def __init__(self, mask: np.ndarray, source: str) -> None:
        self.mask = mask
        self.source = source
        self.config = type("SuppliedMaskConfig", (), {"device": "cpu"})()

    def __call__(self, image_rgb: np.ndarray):
        if self.mask.shape != image_rgb.shape[:2]:
            raise ValueError("Supplied mask and image must have identical spatial dimensions")
        return {"mask": self.mask.copy()}

    def release(self):
        self.mask = None
        return {"supplied_mask_reference_released": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parseq-source", type=Path, required=True)
    parser.add_argument(
        "--parseq-source-type", choices=PARSEQ_SOURCE_TYPES,
        default=PARSEQ_GIT_SOURCE,
    )
    parser.add_argument("--parseq-source-archive", type=Path)
    parser.add_argument("--parseq-checkpoint", type=Path, required=True)
    transport = parser.add_mutually_exclusive_group(required=True)
    transport.add_argument("--deepsolo-python", type=Path)
    transport.add_argument("--deepsolo-docker-container")
    parser.add_argument(
        "--deepsolo-container-python", default="/opt/context-str-deepsolo/bin/python"
    )
    parser.add_argument("--deepsolo-container-workdir", default="/workspace/DeepSolo")
    parser.add_argument("--deepsolo-container-image-tag")
    parser.add_argument("--deepsolo-container-image-id")
    parser.add_argument("--deepsolo-runtime-manifest", type=Path, required=True)
    parser.add_argument("--deepsolo-source-root", type=Path, required=True)
    parser.add_argument("--deepsolo-candidate", choices=tuple(CANDIDATES), required=True)
    parser.add_argument("--deepsolo-reduction", choices=REDUCTION_POLICIES, required=True)
    parser.add_argument("--deepsolo-score-threshold", type=float)
    parser.add_argument("--deepsolo-lexicon-type", type=int, choices=(1, 2, 3))
    parser.add_argument("--scoring-strategy", choices=STRATEGIES, required=True)
    parser.add_argument(
        "--lexical-mode",
        choices=("raw", "lowercase", "normalized_whitespace", "unicode_normalized"),
        required=True,
    )
    parser.add_argument("--min-area", type=int, required=True)
    parser.add_argument("--padding", type=int, required=True)
    parser.add_argument("--connectivity", type=int, choices=(4, 8), default=8)
    parser.add_argument("--equal-area-tie-break", default=APPROXIMATE_TIE_BREAK)
    args = parser.parse_args()

    image = np.asarray(Image.open(args.image).convert("RGB"))
    mask = np.asarray(Image.open(args.mask).convert("L"))
    mask_hash = hashlib.sha256(args.mask.read_bytes()).hexdigest()
    segmenter = SuppliedMaskSegmenter(
        mask,
        "supplied mask engineering fixture {} sha256={}".format(args.mask.resolve(), mask_hash),
    )
    recognizer = ParseqApproxRecognizer(ParseqApproxConfig(
        runtime_source=args.parseq_source,
        checkpoint=args.parseq_checkpoint,
        device="cuda:0",
        runtime_source_type=args.parseq_source_type,
        runtime_source_archive=args.parseq_source_archive,
    ))
    captioner = BLIP2Captioner(BLIP2ApproxConfig(
        device="cuda:0", dtype="float16", placement="cpu_offload", quantization="none",
    ))
    semantic = MPNetSemanticSimilarity(MPNetApproxConfig(device="cpu", precision="float32"))
    if args.deepsolo_docker_container:
        runtime_environment = {}
        if args.deepsolo_container_image_tag:
            runtime_environment["CONTEXT_STR_DEEPSOLO_IMAGE_TAG"] = (
                args.deepsolo_container_image_tag
            )
        if args.deepsolo_container_image_id:
            runtime_environment["CONTEXT_STR_DEEPSOLO_IMAGE_ID"] = (
                args.deepsolo_container_image_id
            )
        fallback = DeepSoloDockerAdapter(
            args.deepsolo_docker_container,
            args.deepsolo_runtime_manifest,
            host_source_root=args.deepsolo_source_root,
            container_python=args.deepsolo_container_python,
            container_workdir=args.deepsolo_container_workdir,
            runtime_environment=runtime_environment,
        )
    else:
        fallback = DeepSoloSubprocessAdapter(
            args.deepsolo_python,
            args.deepsolo_runtime_manifest,
            host_source_root=args.deepsolo_source_root,
        )
    pipeline = ApproximateE2EPipeline(
        ApproximateE2EConfig(
            localization=BlockParam(
                min_area=args.min_area,
                pad=args.padding,
                connectivity=args.connectivity,
                tie_break=args.equal_area_tie_break,
            ),
            segmentation_source=segmenter.source,
            scoring_strategy=args.scoring_strategy,
            lexical_mode=args.lexical_mode,
            deepsolo_candidate=args.deepsolo_candidate,
            deepsolo_reduction_policy=args.deepsolo_reduction,
            deepsolo_inference_options={
                "device": "cuda:0",
                "score_threshold": args.deepsolo_score_threshold,
                "lexicon_type": args.deepsolo_lexicon_type,
            },
        ),
        segmenter=segmenter,
        recognizer=recognizer,
        captioner=captioner,
        semantic=semantic,
        fallback=fallback,
    )
    trace = pipeline.run(image, image_path=args.image)
    pipeline.write_json(trace, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
