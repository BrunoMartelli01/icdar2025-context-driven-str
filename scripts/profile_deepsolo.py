#!/usr/bin/env python3
"""Run one isolated candidate load/inference and preserve its JSON response."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepsolo.adapter import (
    DeepSoloDockerAdapter, DeepSoloProcessError, DeepSoloSubprocessAdapter,
)
from deepsolo.candidates import CANDIDATES
from deepsolo.reduction import REDUCTION_POLICIES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    transport = parser.add_mutually_exclusive_group(required=True)
    transport.add_argument("--deepsolo-python", type=Path)
    transport.add_argument("--docker-container")
    parser.add_argument(
        "--container-python", default="/opt/context-str-deepsolo/bin/python"
    )
    parser.add_argument("--container-workdir", default="/workspace/DeepSolo")
    parser.add_argument("--container-image-tag")
    parser.add_argument("--container-image-id")
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--deepsolo-source-root", type=Path, required=True)
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    parser.add_argument("--reduction-policy", choices=REDUCTION_POLICIES, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "cuda:0"), required=True)
    parser.add_argument("--score-threshold", type=float)
    parser.add_argument("--lexicon-type", type=int, choices=(1, 2, 3))
    args = parser.parse_args()
    if args.docker_container:
        runtime_environment = {}
        if args.container_image_tag:
            runtime_environment["CONTEXT_STR_DEEPSOLO_IMAGE_TAG"] = args.container_image_tag
        if args.container_image_id:
            runtime_environment["CONTEXT_STR_DEEPSOLO_IMAGE_ID"] = args.container_image_id
        adapter = DeepSoloDockerAdapter(
            args.docker_container,
            args.runtime_manifest,
            host_source_root=args.deepsolo_source_root,
            container_python=args.container_python,
            container_workdir=args.container_workdir,
            runtime_environment=runtime_environment,
        )
    else:
        adapter = DeepSoloSubprocessAdapter(
            args.deepsolo_python,
            args.runtime_manifest,
            host_source_root=args.deepsolo_source_root,
        )
    try:
        response = adapter.infer(
            image_path=args.image,
            candidate_id=args.candidate,
            inference_options={
                "device": args.device,
                "score_threshold": args.score_threshold,
                "lexicon_type": args.lexicon_type,
            },
            reduction_policy=args.reduction_policy,
        )
        payload = response.to_dict()
        exit_code = 0
    except DeepSoloProcessError as error:
        if error.response is None:
            raise
        payload = error.response.to_dict()
        exit_code = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
