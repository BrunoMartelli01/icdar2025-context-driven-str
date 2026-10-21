# Approximate AG-UNet Implementation Decisions

## Status and claim boundary

This document defines the runnable segmentation architecture introduced for
Macro-Milestone 4 and frozen/qualified in M8. It is an **APPROXIMATE** reconstruction. It is not the repaired
historical `AdvancedUNet`, an author checkpoint architecture, or a STRICT reproduction.

M8 freezes this branch as **`approximate-ag-unet-figure3-v1`**, architecture-contract
schema version 1. The deterministic machine-readable contract is
[`provenance/AG_UNET_ARCHITECTURE_M8.json`](../provenance/AG_UNET_ARCHITECTURE_M8.json).
Changing topology, named state tensors, or any architectural assumption requires a new
identity/version rather than mutating this contract.

The primary evidence is Section 3.2 and Figure 3 of arXiv:2503.15639v2. The frozen
repository implementation is secondary evidence. `FIGURE_INFERRED` below is an
evidence basis, not an additional decision classification: each such choice is
classified `ASSUMPTION` as required by `AGENTS.md`.

## Fixed evidence and selected architecture

| Item | Approximate implementation | Classification | Evidence/rationale |
|---|---|---|---|
| Encoder | torchvision ResNet-50 feature stages `64@128`, `256@64`, `512@32`, `1024@16`, `2048@8` for a 256-square input | `PAPER_SPECIFIED` family and shapes | Section 3.2 and Figure 3 |
| Encoder weights | Explicit caller choice; approximate real-training config selects `IMAGENET1K_V2`; tests/synthetic smoke use no downloaded weights | `REPOSITORY_SPECIFIED` | Frozen author repository; exact paper revision remains A-006 |
| Bottleneck | One 3×3 `2048→512` convolution, BatchNorm, ReLU at 8×8 | `ASSUMPTION` (`FIGURE_INFERRED`) | Paper specifies compression with convolution+BN+activation but not kernel/output channels; 512 is inferred from Figure 3's first decoder feature |
| Upsampling | Bilinear ×2, `align_corners=False`, then a 1×1 channel projection | `ASSUMPTION` | Paper labels upsampling but gives no method; chosen explicitly for the approximation |
| Attention gates | Additive gate with 1×1 projections, BatchNorm, ReLU, and one-channel sigmoid spatial coefficient multiplied into the encoder skip | `ASSUMPTION` | Paper supplies only inputs `x`/`g` and cites attention gates; exact equations/channels remain A-004 |
| Gate intermediate channels | Half the decoder output channels | `ASSUMPTION` | Not specified by paper or frozen repository |
| Skip set | Gate `1024@16`, `512@32`, `256@64`, and `64@128` encoder features | `ASSUMPTION` (`FIGURE_INFERRED`) | Figure 3 depicts gates in each shown skip; exact count is not stated in prose |
| Decoder channels | `512@16`, `256@32`, `128@64`, `64@128` | `ASSUMPTION` (`FIGURE_INFERRED`) | Directly follows the labels visible in Figure 3 but the tensor-producing operations are absent |
| Decoder block | Concatenate projected decoder feature and gated encoder skip, then two 3×3 Conv-BN-ReLU layers | `REPOSITORY_SPECIFIED` | Preserves the frozen repository's conventional decoder block; Figure 3 does not specify it |
| Prediction head | One 1×1 convolution from 64 channels to one foreground logit channel | `REPOSITORY_SPECIFIED` | Frozen repository behavior; compatible with binary foreground intent but not detailed by paper |
| Logit resolution | 128×128 for a 256×256 input; no final full-resolution decoder stage | `ASSUMPTION` (`FIGURE_INFERRED`) | Matches the last labeled Figure 3 feature and Section 4.1's 128×128 training masks; A-005 remains unresolved |
| Encoder training state | Explicit configurable freeze flag; approximate smoke config uses trainable encoder unless stated otherwise | `ASSUMPTION` | “All other modules remain frozen” does not unambiguously say whether the encoder is frozen |
| Initialization | torchvision defaults for selected encoder weights; Kaiming-normal convs and unit/zero BatchNorm parameters for new layers | `ASSUMPTION` | Paper does not state non-encoder initialization |
| Input geometry | Positive BCHW batches, three floating-point channels matching the model dtype/device, and height/width each at least 32 and divisible by 32; scientific approximate config uses 256×256 float32 | `ASSUMPTION` plus `PAPER_SPECIFIED` configured size | Supported shapes produce `H/2×W/2` logits; other shapes fail before encoder execution |

## Shape contract at 256×256

| Tensor | Channels | Spatial size |
|---|---:|---:|
| Encoder stem `e0` | 64 | 128×128 |
| Encoder `e1` | 256 | 64×64 |
| Encoder `e2` | 512 | 32×32 |
| Encoder `e3` | 1024 | 16×16 |
| Encoder `e4` | 2048 | 8×8 |
| Compressed bottleneck `b` | 512 | 8×8 |
| Decoder `d3` | 512 | 16×16 |
| Decoder `d2` | 256 | 32×32 |
| Decoder `d1` | 128 | 64×64 |
| Decoder `d0` | 64 | 128×128 |
| Foreground logits | 1 | 128×128 |

The implementation exposes these tensors through an opt-in shape-trace method for
architecture conformance tests. It has **48,454,733 parameters** when all encoder
parameters are trainable. This calculated count is an implementation property, not a
paper claim.

Each represented skip is gated. At the canonical input, the projected decoder gate,
encoder skip, and one-channel coefficient shapes are respectively
`512/1024/1@16×16`, `256/512/1@32×32`, `128/256/1@64×64`, and
`64/64/1@128×128`. Spatial equality is checked by the gate rather than repaired there.

## M8 frozen conformance and checkpoint contract

- Exact total and selected-branch trainable parameters: `48,454,733`; freezing the
  encoder is an explicit alternate training-state choice and leaves `22,897,701`
  trainable parameters without changing topology. To preserve the established count and
  schema, the full torchvision encoder keeps registered `avgpool`/`fc` modules, although
  feature extraction never executes them; this inherited repository behavior is now
  explicit rather than silently removed.
- State dictionary: 456 tensors. Its schema fingerprint is
  `bc146d1e3742a63f9b64ad8aded6ff92fbde656f8386e1a88d0e8a109ab04b5f`, computed
  as SHA-256 over sorted `key\0comma-separated-shape\0dtype\n` records. Parameter
  values are intentionally excluded.
- Strict checkpoint loading requires checkpoint schema version 1, architecture identity,
  architecture-contract schema version, schema fingerprint, exact keys, shapes, and
  dtypes. Missing/unexpected keys, shape/dtype mismatches, incorrect metadata, and
  `strict=False` are rejected actionably.
- Initialization is `ASSUMPTION`: the selected approximate real-weight candidate is
  torchvision `IMAGENET1K_V2` (`REPOSITORY_SPECIFIED`; exact revision/hash and paper
  selection remain A-006). Tests use no downloaded weights. New convolution layers use
  the frozen Kaiming policies in the machine contract; BatchNorm affine parameters are
  one/zero; biases are zero. Repository determinism requires seeding before construction.
- A-005 is not resolved: this identity emits one foreground-logit channel at `H/2×W/2`
  (128×128 for the canonical input), with no sigmoid in the model.

## Physical-host batch-1 qualification

The CUDA report is deliberately separate from the static architecture artifact. With the
main environment activated, run:

```bash
python -m scripts.profile_ag_unet --device cuda:0 --precision float32 --output /tmp/ag_unet_m8_cuda.json
```

This performs evaluation-only inference with `IMAGENET1K_V2`, input
`[1,3,256,256]`, and no training. It reports success/failure, OOM status and actionable
guidance, device/name, input/output shapes, batch size, precision, weight artifact and
classification, plus peak allocated/reserved bytes.

The first physical-host profiler attempt failed before model construction because PyTorch
2.4.1+cu124 had not initialized the lazy CUDA context/allocator when peak statistics were
reset. Fresh-process diagnosis showed that a one-element allocation on `cuda:0` resolves
that profiler-only failure. The profiler now allocates and deletes that tensor, synchronizes,
empties the cache, and only then resets peak statistics, excluding initialization from the
model measurement. This does not alter the model, weights, precision, or architecture
identity.

The corrected profiler passed on the physical NVIDIA GeForce RTX 3070 using `cuda:0`,
float32, `IMAGENET1K_V2`, batch size 1, input `[1,3,256,256]`, and output
`[1,1,128,128]`. It reported `success=true`, `oom=false`, 264,071,680 peak allocated
bytes, and 289,406,976 peak reserved bytes. The complete physical suite passed 142 tests
in 30.68 seconds and `git diff --check` passed. These are engineering qualification facts
recorded separately from the static architecture contract; no training was performed and
no scientific classification or STRICT blocker changed.

## Strict blockers preserved for the M8 identity

At the M8 freeze, STRICT construction/execution remained blocked by A-001 through A-008, T-001 through
T-004, DS-001/DS-003, and E-002/E-003. In particular, no author model definition or
checkpoint establishes the exact decoder, attention equations, bottleneck, ResNet-50
weight revision, output resolution, normalization, threshold, training duration,
dataset regime, or metric aggregation. The approximate implementation must use the
`approximate reproduction` claim label and must never be loaded by namespace fallback
from `configs/strict/`.

## M9 real-training contract

M9 reuses this frozen architecture and adds no topology choice. The real-data entry point is
`python -m segmentation.train`; it accepts only mode-local `run.kind=real_training`
configuration and explicit data/output roots. Configured manifest paths resolve from the
repository root by default (or an explicit `--manifest-root`), while sample paths resolve
only from `--data-root`. It requires two checksum-bearing manifests
whose `artifact_id` values exist as upstream-primary dataset/dataset-split entries in
`provenance/ARTIFACTS.yaml`. Each manifest declares `train` or `validation`; synthetic,
demo, smoke, and test manifests are rejected. Training and validation IDs and resolved
image/mask paths must be disjoint.
The output root must be outside the repository so generated checkpoints and run artifacts
cannot accidentally enter Git.

For ICDAR13, M9 freezes the `ASSUMPTION` policy
`icdar13-task2.2-sha256-ranked-holdout-v1`: seed 202503, raw SHA-256 digest ranking of
`UTF8("202503:" + numeric_sample_id)`, first 23 validation and remaining 206 training,
then numeric manifest order. Its machine record and two per-file-checksummed manifests are
under `manifests/`. Physical regeneration covered all 229 pairs with zero intersection and
was byte-identical. M9 training now converts exact RGB white to background and every other
RGB value to foreground before configured resizing. Nearest mask interpolation preserves
binary targets. Don't Care remains foreground and TXT annotations are audit-only under an
explicit `ASSUMPTION`; BCE and its unmasked contract are unchanged. The manifests are
APPROXIMATE-training-ready, while DS-003 remains a STRICT blocker.

The epoch contract retains paper-specified Adam, learning rate `1e-4`, weight decay `0.01`,
batch size 8, BCE, 256×256 inputs, and 128×128 masks. No scheduler, early stopping, best
checkpoint, AMP, gradient accumulation, or alternate optimizer is available. The
APPROXIMATE ICDAR13 configuration explicitly classifies two epochs, Adam defaults,
trainable encoder, transforms, held-out validation source, seed, dataset-specific training,
and last-checkpoint semantics as `ASSUMPTION`/`REPOSITORY_SPECIFIED`; none enters STRICT.

Every epoch writes deterministic JSONL metrics and an architecture-aware checkpoint with
model, optimizer, completed epoch/global step, complete Python/NumPy/CPU-CUDA PyTorch RNG
state, resolved-config and dataset-manifest identities, and runtime metadata. Resume checks
the complete run/training contract before strict state loading and RNG restoration. The run
manifest records checkpoint SHA-256, dataset/sample-manifest checksums, assumptions,
blockers, validation loss, device/precision, and peak allocated/reserved CUDA memory.
Failures, including OOM, are written to `failure.json` and re-raised.

The physical M9 qualification used the one-batch-per-epoch real-data smoke path, which
changes the result class to `smoke-test result` explicitly:

```bash
python -m segmentation.train --config configs/approximate/segmentation_icdar13.yaml --data-root "${DATA_ROOT}" --output-root "${RUN_ROOT}" --device cuda --qualification-max-steps 1
```

On the physical NVIDIA GeForce RTX 3070 8192 MiB (driver 595.84, torch 2.4.1+cu124), the
CUDA float32 run used 48,454,733 parameters, batch 8, images `[8,3,256,256]`, and masks
`[8,1,128,128]`. With two configured epochs, `qualification-max-steps=1` produced one train
step per epoch and `global_step=2`. Train losses were `0.9355729818344116` and
`0.9342144727706909`; validation losses were `0.2795393764972687` and
`0.3452925980091095`. It reported `success=true`, `oom=false`, 2,373,442,560 peak allocated
bytes, and 2,552,233,984 peak reserved bytes. The final external checkpoint SHA-256 is
`f06db536167ccd3d881adababaf00b2d6cefbb8cf6524bd22d303a908540e45e`; no output artifact is
tracked. The physical suite passed 171 tests in 38.56 seconds and `git diff --check` passed.
This is a short APPROXIMATE engineering qualification, not full scientific training or a
Table 3 result, and it does not resolve T-001 or any other STRICT blocker.

## Separate author-clarified protocol identity

The 2026-08-29 author clarification does not mutate the frozen M8/M9 identity or its
checkpoints. `AuthorProtocolAGUNet` instead uses identity
`author-clarified-ag-unet-2048x1024-v1`: a frozen ResNet-50 encoder and BN state, one
assumed `2048->1024` Conv-BN-ReLU bottleneck, decoder outputs `512/256/128/64`, additive
one-channel spatial attention that preserves skip widths, and one-channel logits at
128x128 for 256x256 input. The extra historical 256x256 logit stage is absent.

The author confirms all-229/no-validation ICDAR13 training, 100 epochs, flip probability
0.5, bilinear target resize to 128x128 plus threshold 0.5, per-pixel Don’t Care BCE
validity, and best training-foreground-IoU checkpoint selection. The runnable config
records the bottleneck width/block, exact decoder operations, attention intermediate
widths, +/-15 degree rotation and probability, validity resize, optimizer defaults,
weights/normalization, and IoU aggregation/threshold as unresolved assumptions or
repository choices. Consequently the branch remains an approximate reproduction and
requires new physical RTX 3070 VRAM qualification before its 100-epoch run.
