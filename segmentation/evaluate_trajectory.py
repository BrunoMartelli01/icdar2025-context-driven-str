"""Post-hoc test-set trajectory for an already completed segmentation run.

This is deliberately labeled a leaky diagnostic. Repeated inspection of the
official test set cannot select a checkpoint or support an unbiased Table 3 claim.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

from omegaconf import OmegaConf

from segmentation.config import load_segmentation_config
from segmentation.data import file_sha256
from segmentation.evaluate import ROOT, SegmentationEvaluationError, evaluate


EXPECTED_EPOCHS = 100
TRAJECTORY_NAME = "icdar13-author-protocol-test-trajectory"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_canonical_json(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def _inside(root: Path, path: Path) -> bool:
    return root == path or root in path.parents


def _validate_training_run(
    run_directory: Path, config: Any
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    manifest_path = run_directory / "run_manifest.json"
    if not manifest_path.is_file():
        raise SegmentationEvaluationError(
            f"Training run manifest does not exist: {manifest_path}"
        )
    manifest_sha256 = file_sha256(manifest_path)
    if manifest_sha256 != str(config.checkpoint.training_run_manifest_sha256):
        raise SegmentationEvaluationError("Training run manifest SHA256 mismatch")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SegmentationEvaluationError(
            f"Cannot read training run manifest: {error}"
        ) from error
    if manifest.get("status") != "SUCCESS" or manifest.get("success") is not True:
        raise SegmentationEvaluationError("Trajectory requires a successful full training run")
    architecture = manifest.get("architecture") or {}
    if architecture.get("identity") != str(config.architecture.identity):
        raise SegmentationEvaluationError("Training architecture identity mismatch")
    epochs = manifest.get("epochs")
    if not isinstance(epochs, list) or len(epochs) != EXPECTED_EPOCHS:
        raise SegmentationEvaluationError(
            f"Trajectory requires exactly {EXPECTED_EPOCHS} source epochs"
        )
    if [record.get("epoch") for record in epochs] != list(range(1, EXPECTED_EPOCHS + 1)):
        raise SegmentationEvaluationError("Training epochs must be contiguous from 1 to 100")
    selected = manifest.get("checkpoint") or {}
    if selected.get("sha256") != str(config.checkpoint.sha256):
        raise SegmentationEvaluationError("Configured selected checkpoint does not match the run")
    for record in epochs:
        relative = record.get("checkpoint")
        digest = record.get("checkpoint_sha256")
        if not isinstance(relative, str) or not relative.endswith(".pt"):
            raise SegmentationEvaluationError("Epoch checkpoint path is invalid")
        checkpoint = (run_directory / relative).resolve()
        if not _inside(run_directory, checkpoint):
            raise SegmentationEvaluationError("Epoch checkpoint escapes the training run")
        if not checkpoint.is_file():
            raise SegmentationEvaluationError(f"Epoch checkpoint is missing: {checkpoint}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise SegmentationEvaluationError("Epoch checkpoint SHA256 is invalid")
    return manifest, epochs, manifest_sha256


def _trajectory_record(
    epoch_record: Mapping[str, Any], report: Mapping[str, Any], report_path: Path
) -> dict[str, Any]:
    aggregates = report["aggregates"]
    primary = aggregates["primary_per_image_macro"]
    micro = aggregates["diagnostic_global_micro"]
    return {
        "schema_version": 1,
        "epoch": int(epoch_record["epoch"]),
        "global_step": int(epoch_record["global_step"]),
        "checkpoint": str(epoch_record["checkpoint"]),
        "checkpoint_sha256": str(epoch_record["checkpoint_sha256"]),
        "sample_count": int(aggregates["sample_count"]),
        "failure_count": int(aggregates["failure_count"]),
        "macro_foreground_iou": float(primary["foreground_iou"]),
        "macro_foreground_f1": float(primary["foreground_f1"]),
        "micro_foreground_iou": float(micro["foreground_iou"]),
        "micro_foreground_f1": float(micro["foreground_f1"]),
        "elapsed_seconds": float(report["runtime"]["elapsed_seconds"]),
        "report": str(report_path),
    }


def trajectory_summary(
    records: list[Mapping[str, Any]], *, source_epoch_count: int
) -> dict[str, Any]:
    if not records:
        raise SegmentationEvaluationError("Trajectory contains no completed epochs")
    by_epoch = sorted(records, key=lambda record: int(record["epoch"]))
    highest_iou = max(by_epoch, key=lambda record: float(record["macro_foreground_iou"]))
    highest_f1 = max(by_epoch, key=lambda record: float(record["macro_foreground_f1"]))
    return {
        "schema_version": 1,
        "result_class": "leaky test-as-validation diagnostic",
        "scientific_mode": "approximate",
        "test_set_reused_across_epochs": True,
        "checkpoint_selection_permitted": False,
        "table3_claim_permitted": False,
        "source_epoch_count": int(source_epoch_count),
        "completed_epoch_count": len(by_epoch),
        "complete": len(by_epoch) == int(source_epoch_count),
        "highest_observed_macro_iou": dict(highest_iou),
        "highest_observed_macro_f1": dict(highest_f1),
        "epochs": [dict(record) for record in by_epoch],
    }


def run_trajectory(
    *,
    config_path: Path,
    training_run: Path,
    data_root: Path,
    output_root: Path,
    device_name: str,
    start_epoch: int = 1,
    end_epoch: int = EXPECTED_EPOCHS,
) -> dict[str, Any]:
    if start_epoch < 1 or end_epoch > EXPECTED_EPOCHS or start_epoch > end_epoch:
        raise SegmentationEvaluationError("Epoch range must stay within 1..100")
    resolved_output = output_root.resolve()
    if _inside(ROOT, resolved_output):
        raise SegmentationEvaluationError("Trajectory output must remain outside the repository")
    config = load_segmentation_config(config_path)
    run_directory = training_run.resolve()
    manifest, epochs, manifest_sha256 = _validate_training_run(run_directory, config)
    trajectory_root = resolved_output / TRAJECTORY_NAME
    trajectory_root.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()
    records = []

    print(
        "trajectory_start label=LEAKY_TEST_AS_VALIDATION epochs={}-{} "
        "checkpoint_selection_permitted=no table3_claim_permitted=no".format(
            start_epoch, end_epoch
        ),
        flush=True,
    )
    for epoch_record in epochs:
        epoch = int(epoch_record["epoch"])
        if epoch < start_epoch or epoch > end_epoch:
            continue
        epoch_root = trajectory_root / "epochs" / f"epoch-{epoch:04d}"
        result_root = epoch_root / "result"
        report_path = result_root / "table3.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("checkpoint", {}).get("sha256") != epoch_record["checkpoint_sha256"]:
                raise SegmentationEvaluationError(
                    f"Existing epoch {epoch} result has a different checkpoint SHA256"
                )
        else:
            epoch_root.mkdir(parents=True, exist_ok=True)
            epoch_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
            OmegaConf.update(epoch_config, "run.name", "result")
            OmegaConf.update(
                epoch_config, "checkpoint.sha256", epoch_record["checkpoint_sha256"]
            )
            OmegaConf.update(epoch_config, "checkpoint.selected_epoch", epoch)
            OmegaConf.update(
                epoch_config, "checkpoint.global_step", epoch_record["global_step"]
            )
            epoch_config_path = epoch_root / "resolved_evaluation_config.yaml"
            OmegaConf.save(epoch_config, epoch_config_path)
            checkpoint = (run_directory / epoch_record["checkpoint"]).resolve()
            report = evaluate(
                config_path=epoch_config_path,
                data_root=data_root,
                checkpoint=checkpoint,
                output_root=epoch_root,
                device_name=device_name,
            )
            if not report["aggregates"]["complete"]:
                raise SegmentationEvaluationError(f"Epoch {epoch} evaluation is incomplete")
        report["result_class"] = "leaky test-as-validation diagnostic"
        report["diagnostic"] = {
            "test_set_reused_across_epochs": True,
            "checkpoint_selection_permitted": False,
            "table3_claim_permitted": False,
        }
        _atomic_json(report_path, report)
        record = _trajectory_record(epoch_record, report, report_path)
        records.append(record)
        print(
            "trajectory_epoch={:03d}/100 macro_fgIoU={:.6f} macro_F1={:.6f} "
            "failures={} elapsed={:.1f}s".format(
                epoch,
                record["macro_foreground_iou"],
                record["macro_foreground_f1"],
                record["failure_count"],
                record["elapsed_seconds"],
            ),
            flush=True,
        )

    summary = trajectory_summary(records, source_epoch_count=len(epochs))
    summary.update({
        "training_run": str(run_directory),
        "training_run_manifest_sha256": manifest_sha256,
        "training_selected_checkpoint": manifest["checkpoint"],
        "test_config": str(config_path.resolve()),
        "test_config_sha256": file_sha256(config_path),
        "requested_epoch_range": [start_epoch, end_epoch],
        "runtime_seconds": time.perf_counter() - started_at,
    })
    _atomic_json(trajectory_root / "trajectory.json", summary)
    (trajectory_root / "trajectory.jsonl").write_text(
        "".join(_canonical_json(record) + "\n" for record in records),
        encoding="utf-8",
    )
    print(
        "trajectory_complete completed={} highest_iou_epoch={} highest_iou={:.6f} "
        "highest_f1_epoch={} highest_f1={:.6f}".format(
            summary["completed_epoch_count"],
            summary["highest_observed_macro_iou"]["epoch"],
            summary["highest_observed_macro_iou"]["macro_foreground_iou"],
            summary["highest_observed_macro_f1"]["epoch"],
            summary["highest_observed_macro_f1"]["macro_foreground_f1"],
        ),
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/approximate/segmentation_icdar13_author_protocol_test.yaml",
    )
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--end-epoch", type=int, default=EXPECTED_EPOCHS)
    args = parser.parse_args()
    run_trajectory(
        config_path=args.config,
        training_run=args.training_run,
        data_root=args.data_root,
        output_root=args.output_root,
        device_name=args.device,
        start_epoch=args.start_epoch,
        end_epoch=args.end_epoch,
    )


if __name__ == "__main__":
    main()
