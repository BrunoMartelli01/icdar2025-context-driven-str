#!/usr/bin/env python3
"""Run one real-data APPROXIMATE downstream smoke with frozen segmentation."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import traceback

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from caption.blip2_captioner import BLIP2ApproxConfig, BLIP2Captioner
from deepsolo.adapter import DeepSoloSubprocessAdapter
from localize.mask_to_blocks import BlockParam
from pipeline.approximate_e2e import ApproximateE2EConfig, ApproximateE2EPipeline
from pipeline.config import get_optional, get_required, load_config, resolved_yaml
from pipeline.frozen_segmentation import FrozenAuthorProtocolSegmenter, FrozenSegmentationConfig
from recognition.parseq import (
    PARSEQ_RUNTIME_ARCHIVE_SHA256,
    ParseqApproxConfig,
    ParseqApproxRecognizer,
)
from scoring.semantic import MPNetApproxConfig, MPNetSemanticSimilarity


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _path_from_repository(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segmentation-checkpoint", type=Path, required=True)
    parser.add_argument("--parseq-source", type=Path, required=True)
    parser.add_argument("--parseq-source-archive", type=Path)
    parser.add_argument("--parseq-checkpoint", type=Path, required=True)
    parser.add_argument("--deepsolo-python", type=Path, required=True)
    parser.add_argument("--deepsolo-runtime-manifest", type=Path, required=True)
    parser.add_argument("--deepsolo-source-root", type=Path, required=True)
    return parser.parse_args()


def execute(args: argparse.Namespace) -> dict[str, object]:
    cfg = load_config(args.config)
    if get_required(cfg, "run.kind") != "real_data_end_to_end_smoke":
        raise ValueError("This CLI accepts only a real_data_end_to_end_smoke config")
    if get_required(cfg, "run.table_5_claim") is not False:
        raise ValueError("Smoke configuration must explicitly forbid a Table 5 claim")
    if get_required(cfg, "fallback.transport") != "local_python_subprocess":
        raise ValueError("This host smoke requires fallback.transport=local_python_subprocess")
    if (
        str(get_required(cfg, "recognition.primary.runtime_source_archive_sha256"))
        != PARSEQ_RUNTIME_ARCHIVE_SHA256
    ):
        raise ValueError("PARSeq runtime source archive SHA256 is not the frozen identity")

    image_path = args.image.resolve()
    image = np.asarray(Image.open(image_path).convert("RGB"))
    segmentation = FrozenAuthorProtocolSegmenter(
        FrozenSegmentationConfig(
            evaluation_config=_path_from_repository(
                str(get_required(cfg, "segmentation.evaluation_config"))
            ),
            checkpoint=args.segmentation_checkpoint,
            device=str(get_required(cfg, "segmentation.device")),
        )
    )
    recognizer = ParseqApproxRecognizer(
        ParseqApproxConfig(
            runtime_source=args.parseq_source,
            checkpoint=args.parseq_checkpoint,
            device=str(get_required(cfg, "recognition.device")),
            runtime_source_type=str(
                get_required(cfg, "recognition.primary.runtime_source_type")
            ),
            runtime_source_archive=args.parseq_source_archive,
        )
    )
    captioner = BLIP2Captioner(
        BLIP2ApproxConfig(
            model_id=str(get_required(cfg, "caption.model_id")),
            revision=str(get_required(cfg, "caption.revision")),
            prompt=str(get_required(cfg, "caption.prompt")),
            generation=dict(OmegaConf.to_container(cfg.caption.generation, resolve=True)),
            device=str(get_required(cfg, "caption.device")),
            dtype=str(get_required(cfg, "caption.dtype")),
            placement=str(get_required(cfg, "caption.placement")),
            quantization=str(get_required(cfg, "caption.quantization")),
        )
    )
    semantic = MPNetSemanticSimilarity(
        MPNetApproxConfig(
            model_id=str(get_required(cfg, "scoring.embedder_model_id")),
            revision=str(get_required(cfg, "scoring.embedder_revision")),
            normalize_embeddings=bool(get_required(cfg, "scoring.normalize_embeddings")),
            precision=str(get_required(cfg, "scoring.embedder_precision")),
            device=str(get_required(cfg, "scoring.embedder_device")),
        )
    )

    fallback = DeepSoloSubprocessAdapter(
        args.deepsolo_python,
        args.deepsolo_runtime_manifest,
        host_source_root=args.deepsolo_source_root,
    )

    pipeline = ApproximateE2EPipeline(
        ApproximateE2EConfig(
            localization=BlockParam(
                min_area=int(get_required(cfg, "localize.min_area")),
                pad=int(get_required(cfg, "localize.padding")),
                connectivity=int(get_required(cfg, "localize.connectivity")),
                tie_break=str(get_required(cfg, "localize.equal_area_tie_break")),
                nmax=int(get_required(cfg, "localize.nmax")),
            ),
            segmentation_source="{} sha256={}".format(
                get_required(cfg, "segmentation.checkpoint_artifact_id"),
                get_required(segmentation.segmentation_config, "checkpoint.sha256"),
            ),
            scoring_strategy=str(get_required(cfg, "scoring.strategy")),
            lexical_mode=str(get_required(cfg, "scoring.lexical_mode")),
            deepsolo_candidate=str(get_required(cfg, "fallback.candidate")),
            deepsolo_reduction_policy=str(get_required(cfg, "fallback.reduction_policy")),
            deepsolo_inference_options={
                "device": str(get_required(cfg, "fallback.device")),
                "score_threshold": get_optional(cfg, "fallback.score_threshold"),
                "lexicon_type": get_optional(cfg, "fallback.lexicon_type"),
            },
            alpha=float(get_required(cfg, "scoring.alpha")),
            beta=float(get_required(cfg, "scoring.beta")),
            tau=float(get_required(cfg, "scoring.tau")),
            result_class=str(get_required(cfg, "run.result_class")),
            t3_selection=str(get_required(cfg, "recognition.t3_selection")),
            t3_selection_classification=str(
                get_optional(
                    cfg,
                    "recognition.t3_selection_classification",
                    "ASSUMPTION",
                )
            ),
        ),
        segmenter=segmentation,
        recognizer=recognizer,
        captioner=captioner,
        semantic=semantic,
        fallback=fallback,
    )
    trace = pipeline.run(
        image,
        image_path=image_path,
        artifact_root=args.output.resolve().parent / "intermediates",
    )
    trace["success"] = True
    rendered_config = resolved_yaml(cfg)
    trace["configuration"] = {
        "path": str(args.config.resolve()),
        "resolved_yaml": rendered_config,
        "resolved_yaml_sha256": hashlib.sha256(rendered_config.encode("utf-8")).hexdigest(),
    }
    return trace


def _failure_record(args: argparse.Namespace, error: Exception) -> dict[str, object]:
    response = getattr(error, "response", None)
    return {
        "schema_version": 1,
        "success": False,
        "reproduction_mode": "APPROXIMATE",
        "result_class": "failed approximate reproduction smoke test",
        "claims": {"table_5": False, "table_6": False, "table_7": False},
        "inputs": {
            "config": str(args.config.resolve()),
            "image": str(args.image.resolve()),
            "segmentation_checkpoint": str(args.segmentation_checkpoint.resolve()),
            "parseq_source": str(args.parseq_source.resolve()),
            "parseq_source_archive": (
                str(args.parseq_source_archive.resolve())
                if args.parseq_source_archive is not None
                else None
            ),
            "parseq_checkpoint": str(args.parseq_checkpoint.resolve()),
            "deepsolo_python": str(args.deepsolo_python.absolute()),
            "deepsolo_runtime_manifest": str(args.deepsolo_runtime_manifest.resolve()),
            "deepsolo_source_root": str(args.deepsolo_source_root.resolve()),
            "intermediate_root": str(args.output.resolve().parent / "intermediates"),
        },
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "deepsolo_response": response.to_dict() if response is not None else None,
        },
        "warnings": [
            "This failed run is not a Table 5 result.",
            "No fallback output or alternate model was substituted after failure.",
        ],
    }


def main() -> None:
    args = parse_args()
    try:
        trace = execute(args)
    except Exception as error:
        failure = _failure_record(args, error)
        ApproximateE2EPipeline.write_json(failure, args.output)
        traceback.print_exc()
        print("Failure record: {}".format(args.output.resolve()), file=sys.stderr)
        raise SystemExit(1)
    ApproximateE2EPipeline.write_json(trace, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
