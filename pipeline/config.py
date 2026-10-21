"""Small, mode-isolated OmegaConf loader with provenance validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from omegaconf import DictConfig, ListConfig, OmegaConf


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_REGISTRY = REPOSITORY_ROOT / "provenance" / "ARTIFACTS.yaml"
VALID_MODES = {"strict", "approximate", "demo"}
APPROXIMATE_CLASSIFICATIONS = {"ASSUMPTION", "REPOSITORY_SPECIFIED", "HARDWARE_DEVIATION"}
MISSING = object()


class ConfigValidationError(ValueError):
    """Raised when configuration cannot safely identify a scientific run."""


def get_required(cfg: DictConfig, dotted: str) -> Any:
    """Return a required value, rejecting missing, unreadable, null, or empty fields."""

    try:
        value = OmegaConf.select(cfg, dotted, default=MISSING, throw_on_resolution_failure=True)
    except Exception as error:
        raise ConfigValidationError(f"Cannot resolve required config field '{dotted}': {error}") from error
    if value is MISSING or value is None or value == "":
        raise ConfigValidationError(f"Required config field '{dotted}' is missing or empty")
    return value


def get_optional(cfg: DictConfig, dotted: str, default: Any = None) -> Any:
    """Return an explicitly optional engineering value, preserving resolution errors."""

    try:
        return OmegaConf.select(cfg, dotted, default=default, throw_on_resolution_failure=True)
    except Exception as error:
        raise ConfigValidationError(f"Cannot resolve optional config field '{dotted}': {error}") from error


def parse_config(path: str | Path) -> DictConfig:
    """Parse exactly one config file; cross-namespace inheritance is unsupported."""

    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise ConfigValidationError(f"Configuration file does not exist: {config_path}")
    try:
        cfg = OmegaConf.load(config_path)
    except Exception as error:
        raise ConfigValidationError(f"Cannot read configuration {config_path}: {error}") from error
    if not isinstance(cfg, DictConfig):
        raise ConfigValidationError(f"Configuration root must be a mapping: {config_path}")
    if "defaults" in cfg or "extends" in cfg:
        raise ConfigValidationError("Config inheritance is disabled; load one mode-local file")

    mode = get_required(cfg, "reproduction_mode")
    if mode not in VALID_MODES:
        raise ConfigValidationError(f"Unknown reproduction_mode '{mode}'")
    namespace = config_path.parent.name
    if namespace in VALID_MODES and namespace != mode:
        raise ConfigValidationError(
            f"Config namespace '{namespace}' does not match reproduction_mode '{mode}'"
        )
    return cfg


def _artifact_index(path: Path) -> dict[str, dict[str, Any]]:
    try:
        registry = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except Exception as error:
        raise ConfigValidationError(f"Cannot read artifact registry {path}: {error}") from error
    if not isinstance(registry, dict) or not isinstance(registry.get("artifacts"), list):
        raise ConfigValidationError(f"Artifact registry has an invalid schema: {path}")
    artifacts = registry["artifacts"]
    index = {item.get("id"): item for item in artifacts if isinstance(item, dict) and item.get("id")}
    if len(index) != len(artifacts):
        raise ConfigValidationError("Artifact registry contains a missing or duplicate artifact ID")
    return index


def _walk_artifact_references(value: Any, path: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(value, (DictConfig, dict)):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if str(key).endswith("artifact_id") and child not in (None, ""):
                if not isinstance(child, str):
                    raise ConfigValidationError(f"Artifact ID at '{child_path}' must be a string")
                yield child_path, child
            elif str(key).endswith("artifact_ids") and child not in (None, ""):
                if not isinstance(child, (ListConfig, list)):
                    raise ConfigValidationError(f"Artifact IDs at '{child_path}' must be a list")
                for index, artifact_id in enumerate(child):
                    if not isinstance(artifact_id, str):
                        raise ConfigValidationError(
                            f"Artifact ID at '{child_path}[{index}]' must be a string"
                        )
                    yield f"{child_path}[{index}]", artifact_id
            else:
                yield from _walk_artifact_references(child, child_path)
    elif isinstance(value, (ListConfig, list)):
        for index, child in enumerate(value):
            yield from _walk_artifact_references(child, f"{path}[{index}]")


def _validate_scientific_choices(cfg: DictConfig, artifacts: dict[str, dict[str, Any]]) -> None:
    choices = get_optional(cfg, "scientific_choices", [])
    if not isinstance(choices, (ListConfig, list)):
        raise ConfigValidationError("scientific_choices must be a list")
    for index, choice in enumerate(choices):
        if not isinstance(choice, (DictConfig, dict)):
            raise ConfigValidationError(f"scientific_choices[{index}] must be a mapping")
        classification = choice.get("classification")
        if classification not in APPROXIMATE_CLASSIFICATIONS:
            raise ConfigValidationError(
                f"scientific_choices[{index}] has unsupported classification '{classification}'"
            )
        artifact_ids = choice.get("artifact_ids", [])
        ambiguity_ids = choice.get("ambiguity_ids", [])
        if not artifact_ids and not ambiguity_ids:
            raise ConfigValidationError(
                f"scientific_choices[{index}] must reference provenance artifact_ids and/or ambiguity_ids"
            )
        for artifact_id in artifact_ids:
            if artifact_id not in artifacts:
                raise ConfigValidationError(
                    f"scientific_choices[{index}] references unknown artifact ID '{artifact_id}'"
                )


def validate_config(
    cfg: DictConfig,
    artifact_registry: str | Path = DEFAULT_ARTIFACT_REGISTRY,
) -> DictConfig:
    """Validate mode boundaries, required safeguards, and every artifact reference."""

    if not isinstance(cfg, DictConfig):
        raise ConfigValidationError("Configuration must be an OmegaConf DictConfig")
    mode = get_required(cfg, "reproduction_mode")
    if mode not in VALID_MODES:
        raise ConfigValidationError(f"Unknown reproduction_mode '{mode}'")
    artifacts = _artifact_index(Path(artifact_registry).resolve())
    references = list(_walk_artifact_references(cfg))
    for field, artifact_id in references:
        artifact = artifacts.get(artifact_id)
        if artifact is None:
            raise ConfigValidationError(f"Unknown artifact ID '{artifact_id}' at '{field}'")
        classification = artifact.get("decision_classification")
        if classification in APPROXIMATE_CLASSIFICATIONS and mode != "approximate":
            raise ConfigValidationError(
                f"Approximate artifact '{artifact_id}' at '{field}' requires approximate mode"
            )

    recognizer = get_required(cfg, "recognition.primary.implementation")
    if mode == "strict":
        if get_required(cfg, "claim_label") != "strict reproduction":
            raise ConfigValidationError("Strict mode must use claim_label 'strict reproduction'")
        if recognizer != "parseq_s":
            raise ConfigValidationError("Strict mode requires PARSeq-S; demo recognizers are forbidden")
        if get_required(cfg, "fallback.enabled") is not True:
            raise ConfigValidationError("Strict mode cannot disable the DeepSolo fallback")
        if get_required(cfg, "fallback.implementation") != "deepsolo":
            raise ConfigValidationError("Strict mode requires DeepSolo as the fallback implementation")
        for field in (
            "segmentation.architecture_artifact_id",
            "segmentation.checkpoint_artifact_id",
            "recognition.primary.checkpoint_artifact_id",
            "recognition.primary.runtime_artifact_id",
            "caption.checkpoint_artifact_id",
            "scoring.embedder_artifact_id",
            "fallback.checkpoint_artifact_id",
        ):
            get_required(cfg, field)
        for field, artifact_id in references:
            artifact = artifacts[artifact_id]
            if artifact.get("status") == "BLOCKING" or artifact.get("blocks_strict_reproduction"):
                raise ConfigValidationError(
                    f"Artifact '{artifact_id}' at '{field}' is BLOCKING for strict reproduction"
                )
    elif mode == "approximate":
        if get_required(cfg, "claim_label") != "approximate reproduction":
            raise ConfigValidationError(
                "Approximate mode must use claim_label 'approximate reproduction'"
            )
        _validate_scientific_choices(cfg, artifacts)
        if recognizer == "parseq_s":
            checkpoint_id = get_required(cfg, "recognition.primary.checkpoint_artifact_id")
            runtime_id = get_required(cfg, "recognition.primary.runtime_artifact_id")
            if checkpoint_id == runtime_id:
                raise ConfigValidationError("PARSeq checkpoint and runtime provenance must be separate")
            if artifacts[checkpoint_id].get("artifact_type") not in {
                "checkpoint",
                "checkpoint_candidate",
            }:
                raise ConfigValidationError("PARSeq checkpoint artifact does not identify checkpoint bytes")
            if artifacts[runtime_id].get("artifact_type") not in {
                "source_code",
                "source_code_candidate",
            }:
                raise ConfigValidationError("PARSeq runtime artifact does not identify source code")
    else:
        if get_required(cfg, "claim_label") != "demo only":
            raise ConfigValidationError("Demo mode must use claim_label 'demo only'")
    return cfg


def load_config(
    path: str | Path,
    artifact_registry: str | Path = DEFAULT_ARTIFACT_REGISTRY,
) -> DictConfig:
    """Parse and validate one mode-local configuration file."""

    return validate_config(parse_config(path), artifact_registry)


def resolved_yaml(cfg: DictConfig) -> str:
    """Serialize a fully resolved configuration for experiment metadata."""

    return OmegaConf.to_yaml(cfg, resolve=True)


__all__ = [
    "ConfigValidationError",
    "get_optional",
    "get_required",
    "load_config",
    "parse_config",
    "resolved_yaml",
    "validate_config",
]
