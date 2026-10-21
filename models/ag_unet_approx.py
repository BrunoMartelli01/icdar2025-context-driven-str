"""Figure-derived APPROXIMATE AG-UNet reconstruction.

See ``docs/AG_UNET_IMPLEMENTATION_DECISIONS.md``. This module must not be
presented as the exact paper architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal, Mapping, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


EncoderWeights = Optional[Literal["IMAGENET1K_V2"]]

ARCHITECTURE_IDENTITY = "approximate-ag-unet-figure3-v1"
ARCHITECTURE_CONTRACT_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
EXPECTED_PARAMETER_COUNT = 48_454_733
EXPECTED_TRAINABLE_PARAMETER_COUNT = 48_454_733
EXPECTED_STATE_DICT_TENSOR_COUNT = 456
EXPECTED_STATE_DICT_SCHEMA_FINGERPRINT = (
    "bc146d1e3742a63f9b64ad8aded6ff92fbde656f8386e1a88d0e8a109ab04b5f"
)


def state_dict_schema_fingerprint(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash sorted state keys, shapes, and dtypes; never hash parameter values."""

    if any(not isinstance(key, str) for key in state_dict):
        raise ValueError("State-dict schema requires string keys")
    digest = sha256()
    for key, tensor in sorted(state_dict.items()):
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("State-dict schema requires string keys and tensor values")
        shape = ",".join(str(dimension) for dimension in tensor.shape)
        dtype = str(tensor.dtype)
        if dtype.startswith("torch."):
            dtype = dtype[len("torch."):]
        digest.update(f"{key}\0{shape}\0{dtype}\n".encode("utf-8"))
    return digest.hexdigest()


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        padding = kernel_size // 2
        conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False
        )
        norm = nn.BatchNorm2d(out_channels)
        super().__init__(conv, norm, nn.ReLU(inplace=True))
        nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.ones_(norm.weight)
        nn.init.zeros_(norm.bias)


class DecoderConvBlock(nn.Sequential):
    """Repository-specified two-convolution decoder choice."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            ConvNormAct(in_channels, out_channels, kernel_size=3),
            ConvNormAct(out_channels, out_channels, kernel_size=3),
        )


class SpatialAttentionGate(nn.Module):
    """Explicit approximate additive spatial attention gate."""

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.gate_projection = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.skip_projection = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.coefficient = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self, gate: torch.Tensor, skip: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if gate.shape[-2:] != skip.shape[-2:]:
            raise ValueError("Attention gate and skip must already have matching spatial shapes")
        coefficient = self.coefficient(
            self.gate_projection(gate) + self.skip_projection(skip)
        )
        return skip * coefficient, coefficient


class AttentionDecoderStage(nn.Module):
    def __init__(self, decoder_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.project = ConvNormAct(decoder_channels, out_channels, kernel_size=1)
        self.attention = SpatialAttentionGate(
            gate_channels=out_channels,
            skip_channels=skip_channels,
            inter_channels=max(1, out_channels // 2),
        )
        self.decode = DecoderConvBlock(out_channels + skip_channels, out_channels)

    def forward(
        self, decoder: torch.Tensor, skip: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        upsampled = F.interpolate(
            decoder, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        projected = self.project(upsampled)
        gated_skip, coefficient = self.attention(projected, skip)
        decoded = self.decode(torch.cat((projected, gated_skip), dim=1))
        return decoded, projected, gated_skip, coefficient


@dataclass(frozen=True)
class ApproximateAGUNetConfig:
    encoder_weights: EncoderWeights
    freeze_encoder: bool
    claim_label: str = "approximate reproduction"
    evidence_basis: str = "FIGURE_INFERRED"
    output_stride: int = 2
    architecture_identity: str = ARCHITECTURE_IDENTITY
    architecture_contract_schema_version: int = ARCHITECTURE_CONTRACT_SCHEMA_VERSION


class ApproximateAGUNet(nn.Module):
    """Approximate ResNet-50 Attention-Gated U-Net with 128-square logits."""

    def __init__(self, *, encoder_weights: EncoderWeights, freeze_encoder: bool) -> None:
        super().__init__()
        if encoder_weights not in (None, "IMAGENET1K_V2"):
            raise ValueError("encoder_weights must be None or 'IMAGENET1K_V2'")
        weights = ResNet50_Weights.IMAGENET1K_V2 if encoder_weights else None
        self.config = ApproximateAGUNetConfig(encoder_weights, freeze_encoder)
        self.encoder = resnet50(weights=weights)
        if freeze_encoder:
            self.encoder.requires_grad_(False)

        self.bottleneck = ConvNormAct(2048, 512, kernel_size=3)
        self.decoder3 = AttentionDecoderStage(512, 1024, 512)
        self.decoder2 = AttentionDecoderStage(512, 512, 256)
        self.decoder1 = AttentionDecoderStage(256, 256, 128)
        self.decoder0 = AttentionDecoderStage(128, 64, 64)
        self.head = nn.Conv2d(64, 1, kernel_size=1)
        nn.init.kaiming_normal_(self.head.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.head.bias)

    def _encode(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        e0 = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(image)))
        e1 = self.encoder.layer1(self.encoder.maxpool(e0))
        e2 = self.encoder.layer2(e1)
        e3 = self.encoder.layer3(e2)
        e4 = self.encoder.layer4(e3)
        return e0, e1, e2, e3, e4

    def forward_with_trace(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("ApproximateAGUNet input must have shape Bx3xHxW")
        if image.shape[0] < 1:
            raise ValueError("ApproximateAGUNet input batch must contain at least one image")
        if not image.is_floating_point():
            raise ValueError("ApproximateAGUNet input must use a floating-point dtype")
        model_tensor = self.head.weight
        if image.dtype != model_tensor.dtype:
            raise ValueError(
                "ApproximateAGUNet input dtype must match model parameters; "
                f"got input={image.dtype}, model={model_tensor.dtype}"
            )
        if image.device != model_tensor.device:
            raise ValueError(
                "ApproximateAGUNet input device must match model parameters; "
                f"got input={image.device}, model={model_tensor.device}"
            )
        height, width = image.shape[-2:]
        if height < 32 or width < 32 or height % 32 or width % 32:
            raise ValueError(
                "ApproximateAGUNet spatial dimensions must each be at least 32 and "
                f"divisible by 32; got HxW={height}x{width}"
            )
        e0, e1, e2, e3, e4 = self._encode(image)
        bottleneck = self.bottleneck(e4)
        d3, g3, s3, a3 = self.decoder3(bottleneck, e3)
        d2, g2, s2, a2 = self.decoder2(d3, e2)
        d1, g1, s1, a1 = self.decoder1(d2, e1)
        d0, g0, s0, a0 = self.decoder0(d1, e0)
        logits = self.head(d0)
        trace = {
            "e0": e0,
            "e1": e1,
            "e2": e2,
            "e3": e3,
            "e4": e4,
            "bottleneck": bottleneck,
            "gate3": g3,
            "skip3": s3,
            "attention3": a3,
            "d3": d3,
            "gate2": g2,
            "skip2": s2,
            "attention2": a2,
            "d2": d2,
            "gate1": g1,
            "skip1": s1,
            "attention1": a1,
            "d1": d1,
            "gate0": g0,
            "skip0": s0,
            "attention0": a0,
            "d0": d0,
            "logits": logits,
        }
        return logits, trace

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_trace(image)
        return logits

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @property
    def architecture_identity(self) -> str:
        return ARCHITECTURE_IDENTITY

    @property
    def state_dict_schema_fingerprint(self) -> str:
        return state_dict_schema_fingerprint(self.state_dict())

    def checkpoint_metadata(self) -> dict[str, int | str]:
        actual_fingerprint = self.state_dict_schema_fingerprint
        if actual_fingerprint != EXPECTED_STATE_DICT_SCHEMA_FINGERPRINT:
            raise RuntimeError(
                "Cannot create checkpoint metadata for drifted architecture schema: "
                f"expected {EXPECTED_STATE_DICT_SCHEMA_FINGERPRINT}, got {actual_fingerprint}"
            )
        return {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "architecture_identity": ARCHITECTURE_IDENTITY,
            "architecture_contract_schema_version": ARCHITECTURE_CONTRACT_SCHEMA_VERSION,
            "state_dict_schema_fingerprint": EXPECTED_STATE_DICT_SCHEMA_FINGERPRINT,
        }

    def architecture_summary(self) -> dict[str, Any]:
        """Return the frozen, machine-readable static M8 architecture contract."""

        state = self.state_dict()
        actual_fingerprint = state_dict_schema_fingerprint(state)
        if self.parameter_count != EXPECTED_PARAMETER_COUNT:
            raise RuntimeError(
                "Approximate AG-UNet topology drift: expected "
                f"{EXPECTED_PARAMETER_COUNT} parameters, got {self.parameter_count}"
            )
        if len(state) != EXPECTED_STATE_DICT_TENSOR_COUNT:
            raise RuntimeError(
                "Approximate AG-UNet state schema drift: expected "
                f"{EXPECTED_STATE_DICT_TENSOR_COUNT} tensors, got {len(state)}"
            )
        if actual_fingerprint != EXPECTED_STATE_DICT_SCHEMA_FINGERPRINT:
            raise RuntimeError(
                "Approximate AG-UNet state schema fingerprint drift: expected "
                f"{EXPECTED_STATE_DICT_SCHEMA_FINGERPRINT}, got {actual_fingerprint}"
            )
        return {
            "schema_version": ARCHITECTURE_CONTRACT_SCHEMA_VERSION,
            "architecture_identity": ARCHITECTURE_IDENTITY,
            "reproduction_mode": "approximate",
            "claim_label": "approximate reproduction",
            "canonical_input_shape": [1, 3, 256, 256],
            "supported_input": {
                "layout": "BCHW",
                "channels": 3,
                "batch_size": "positive integer",
                "dtype": "floating point matching model parameters; selected branch float32",
                "height": "integer >= 32 and divisible by 32",
                "width": "integer >= 32 and divisible by 32",
            },
            "output": {
                "name": "foreground_logits",
                "channels": 1,
                "activation": None,
                "spatial_rule": "H/2 x W/2",
                "canonical_shape": [1, 1, 128, 128],
                "classification": "ASSUMPTION",
                "ambiguity_id": "A-005",
            },
            "modules": {
                "encoder": {
                    "type": "torchvision.models.resnet50",
                    "registered_but_not_executed": ["encoder.avgpool", "encoder.fc"],
                    "stages": {
                        "e0": {"module": "encoder.conv1+bn1+relu", "channels": [3, 64]},
                        "e1": {"module": "encoder.maxpool+layer1", "channels": [64, 256]},
                        "e2": {"module": "encoder.layer2", "channels": [256, 512]},
                        "e3": {"module": "encoder.layer3", "channels": [512, 1024]},
                        "e4": {"module": "encoder.layer4", "channels": [1024, 2048]},
                    },
                },
                "bottleneck": {
                    "module": "bottleneck",
                    "type": "3x3 Conv2d(no bias)+BatchNorm2d+ReLU",
                    "channels": [2048, 512],
                },
                "decoder3": self._decoder_contract(512, 1024, 256, 512),
                "decoder2": self._decoder_contract(512, 512, 128, 256),
                "decoder1": self._decoder_contract(256, 256, 64, 128),
                "decoder0": self._decoder_contract(128, 64, 32, 64),
                "head": {"module": "head", "type": "1x1 Conv2d", "channels": [64, 1]},
            },
            "canonical_tensors": {
                "e0": [1, 64, 128, 128],
                "e1": [1, 256, 64, 64],
                "e2": [1, 512, 32, 32],
                "e3": [1, 1024, 16, 16],
                "e4": [1, 2048, 8, 8],
                "bottleneck": [1, 512, 8, 8],
                "gate3": [1, 512, 16, 16],
                "skip3": [1, 1024, 16, 16],
                "attention3": [1, 1, 16, 16],
                "d3": [1, 512, 16, 16],
                "gate2": [1, 256, 32, 32],
                "skip2": [1, 512, 32, 32],
                "attention2": [1, 1, 32, 32],
                "d2": [1, 256, 32, 32],
                "gate1": [1, 128, 64, 64],
                "skip1": [1, 256, 64, 64],
                "attention1": [1, 1, 64, 64],
                "d1": [1, 128, 64, 64],
                "gate0": [1, 64, 128, 128],
                "skip0": [1, 64, 128, 128],
                "attention0": [1, 1, 128, 128],
                "d0": [1, 64, 128, 128],
                "logits": [1, 1, 128, 128],
            },
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "trainable_parameter_count_selected_branch": EXPECTED_TRAINABLE_PARAMETER_COUNT,
            "selected_branch_freeze_encoder": False,
            "state_dict": {
                "tensor_count": EXPECTED_STATE_DICT_TENSOR_COUNT,
                "schema_fingerprint": EXPECTED_STATE_DICT_SCHEMA_FINGERPRINT,
                "fingerprint_algorithm": "SHA256 over sorted key\\0comma-shape\\0dtype\\n",
            },
            "initialization": {
                "selected_encoder_weights": "IMAGENET1K_V2",
                "selected_encoder_weights_artifact": "ART-RESNET50-IMAGENET1K-V2-CANDIDATE",
                "selected_encoder_weights_classification": "REPOSITORY_SPECIFIED",
                "paper_encoder_weights": "UNRESOLVED (A-006)",
                "unweighted_encoder": "torchvision ResNet-50 initialization under torch seed",
                "new_convolutions": "Kaiming normal; fan_out/ReLU except head fan_in/linear",
                "new_batch_norm": "weight=1, bias=0",
                "new_biases": "zero",
                "seed_policy": "call segmentation.training.seed_everything before construction",
            },
            "strict_blockers": ["A-001", "A-002", "A-003", "A-004", "A-005", "A-006"],
        }

    @staticmethod
    def _decoder_contract(
        decoder_channels: int,
        skip_channels: int,
        attention_inter_channels: int,
        out_channels: int,
    ) -> dict[str, Any]:
        return {
            "type": "AttentionDecoderStage",
            "named_submodules": [
                "project",
                "attention.gate_projection",
                "attention.skip_projection",
                "attention.coefficient",
                "decode",
            ],
            "upsample": "bilinear size=skip, align_corners=false",
            "project": {
                "type": "1x1 Conv2d(no bias)+BatchNorm2d+ReLU",
                "channels": [decoder_channels, out_channels],
            },
            "attention": {
                "type": "additive one-channel spatial gate",
                "gate_channels": out_channels,
                "skip_channels": skip_channels,
                "inter_channels": attention_inter_channels,
                "coefficient_channels": 1,
            },
            "decode": {
                "type": "two 3x3 Conv2d(no bias)+BatchNorm2d+ReLU",
                "channels": [out_channels + skip_channels, out_channels, out_channels],
            },
        }

    def make_checkpoint_payload(self) -> dict[str, Any]:
        return {**self.checkpoint_metadata(), "model_state_dict": self.state_dict()}

    def load_compatible_checkpoint(self, payload: Mapping[str, Any]) -> None:
        """Validate architecture metadata and load a state dict strictly."""

        if not isinstance(payload, Mapping):
            raise ValueError("Approximate AG-UNet checkpoint must be a mapping")
        expected_metadata = self.checkpoint_metadata()
        for field, expected in expected_metadata.items():
            actual = payload.get(field)
            if actual != expected:
                raise ValueError(
                    f"Approximate AG-UNet checkpoint {field} mismatch: "
                    f"expected {expected!r}, got {actual!r}"
                )
        checkpoint_state = payload.get("model_state_dict")
        if not isinstance(checkpoint_state, Mapping):
            raise ValueError("Approximate AG-UNet checkpoint model_state_dict must be a mapping")
        if any(not isinstance(key, str) for key in checkpoint_state):
            raise ValueError("Approximate AG-UNet checkpoint state keys must be strings")
        expected_state = self.state_dict()
        missing = sorted(set(expected_state) - set(checkpoint_state))
        unexpected = sorted(set(checkpoint_state) - set(expected_state))
        if missing:
            raise ValueError(f"Approximate AG-UNet checkpoint has missing keys: {missing}")
        if unexpected:
            raise ValueError(f"Approximate AG-UNet checkpoint has unexpected keys: {unexpected}")
        shape_mismatches = {
            key: {"expected": tuple(expected_state[key].shape), "actual": tuple(value.shape)}
            for key, value in checkpoint_state.items()
            if isinstance(value, torch.Tensor) and value.shape != expected_state[key].shape
        }
        if shape_mismatches:
            raise ValueError(
                f"Approximate AG-UNet checkpoint has shape mismatches: {shape_mismatches}"
            )
        invalid_values = sorted(
            key for key, value in checkpoint_state.items() if not isinstance(value, torch.Tensor)
        )
        if invalid_values:
            raise ValueError(
                "Approximate AG-UNet checkpoint has non-tensor state values: "
                f"{invalid_values}"
            )
        dtype_mismatches = {
            key: {"expected": str(expected_state[key].dtype), "actual": str(value.dtype)}
            for key, value in checkpoint_state.items()
            if value.dtype != expected_state[key].dtype
        }
        if dtype_mismatches:
            raise ValueError(
                f"Approximate AG-UNet checkpoint has dtype mismatches: {dtype_mismatches}"
            )
        actual_fingerprint = state_dict_schema_fingerprint(checkpoint_state)
        if actual_fingerprint != EXPECTED_STATE_DICT_SCHEMA_FINGERPRINT:
            raise ValueError(
                "Approximate AG-UNet checkpoint state schema is incompatible: expected "
                f"{EXPECTED_STATE_DICT_SCHEMA_FINGERPRINT}, got {actual_fingerprint}"
            )
        try:
            self.load_state_dict(checkpoint_state, strict=True)
        except RuntimeError as error:
            raise ValueError(
                f"Approximate AG-UNet checkpoint is otherwise incompatible: {error}"
            ) from error

    def load_state_dict(
        self, state_dict: Mapping[str, torch.Tensor], strict: bool = True, assign: bool = False
    ) -> Any:
        if strict is not True:
            raise ValueError("ApproximateAGUNet prohibits non-strict checkpoint loading")
        return super().load_state_dict(state_dict, strict=True, assign=assign)


__all__ = [
    "ARCHITECTURE_CONTRACT_SCHEMA_VERSION",
    "ARCHITECTURE_IDENTITY",
    "CHECKPOINT_SCHEMA_VERSION",
    "EXPECTED_PARAMETER_COUNT",
    "EXPECTED_STATE_DICT_SCHEMA_FINGERPRINT",
    "EXPECTED_STATE_DICT_TENSOR_COUNT",
    "EXPECTED_TRAINABLE_PARAMETER_COUNT",
    "ApproximateAGUNet",
    "ApproximateAGUNetConfig",
    "AttentionDecoderStage",
    "SpatialAttentionGate",
    "state_dict_schema_fingerprint",
]
