# Reproduction Environment Boundaries

## Status and scope

This document records the Macro-Milestone 3 environment design, the isolated PARSeq
compatibility qualification, and creation/validation of `context-str-main` on
2026-08-18. The PARSeq dependency gate is qualified for the **APPROXIMATE** main
environment only. Macro-Milestone 6 subsequently qualified the isolated Docker
runtime recorded in `docs/DEEPSOLO_RUNTIME.md` on the physical host. Sandbox network
and GPU limitations are not reinterpreted as host failures.

Operational follow-up relocated the validated main environment from temporary storage
to the persistent, Git-ignored project-local path
`/home/bruno/projects/icdar2025-context-driven-str/.venvs/context-str-main`. The
project-local path is an equivalent user-owned environment directory; it also keeps
the uv 0.12.5 executable and CPython 3.11.11 base installation outside `/tmp`.

No dependency choice in this document is `PAPER_SPECIFIED`. Version classifications
here describe software compatibility evidence only:

- `UPSTREAM_REQUIRED`: an exact constraint in a frozen upstream source.
- `UPSTREAM_SUPPORTED`: an upstream-supported pairing or range, but not the paper's
  selected runtime.
- `ENGINEERING_CHOICE`: a reproducibility/compatibility selection made for this
  repository.
- `UNRESOLVED`: no compatible selection is established yet.

## Physical host

The dependency-free `scripts/check_system.py` probe reported:

| Item | Physical-host value |
|---|---|
| Distribution | Ubuntu 26.04 LTS |
| Kernel | 7.0.0-29-generic |
| CPU / architecture | QEMU Virtual CPU version 2.5+ / x86_64 |
| RAM | 32,572,747,776 bytes total; 28,432,310,272 bytes available at probe time |
| Repository filesystem | 134,679,105,536 bytes total; 108,883,230,720 bytes free at probe time |
| System Python | CPython 3.14.4 at `/usr/bin/python3.14` |
| Git | 2.53.0 |
| GPU | NVIDIA GeForce RTX 3070, 8192 MiB |
| NVIDIA driver | 595.84 |
| `nvidia-smi` CUDA field | 13.2 (driver compatibility, not toolkit installation) |
| `nvcc` | Not found |
| `CUDA_HOME` / CUDA PATH entries | Not present |
| Conventional system toolkit root | Not found |

The sandbox could locate `nvidia-smi` but could not communicate with the driver. The
reported GPU values therefore come from the approved physical-host execution, not from
the sandbox result.

## Why two environments are mandatory

The main pipeline needs a maintained PyTorch/Transformers/SentenceTransformers stack.
DeepSolo's frozen official revision instead specifies Python 3.8, PyTorch 1.9.0,
torchvision 0.10.0, CUDA 11.1, Detectron2 0.6, and old NumPy/OpenCV packages. Combining
these constraints would either downgrade the main pipeline or silently alter the
DeepSolo runtime. The environments therefore must not share Python, PyTorch,
torchvision, CUDA runtime, NumPy, or OpenCV.

DeepSolo runs through the M6 versioned JSON-file subprocess boundary plus referenced
image files. The main process writes request IDs, image paths/hashes, an explicit
candidate ID, inference options, and an explicit reduction policy. DeepSolo returns
raw ordered geometry, transcription, confidence, config/source/checkpoint identifiers,
failure state, and timing. The caller propagates nonzero exit status, malformed records,
timeouts, and missing outputs. This boundary prevents legacy imports and CUDA extension
state from leaking into the main process.

## Installed `context-str-main` stack

The installed stack targets an official PyTorch CUDA binary rather than the host's
Python 3.14. It is intentionally conservative relative to the newest packages. The
environment is managed by `uv 0.12.5` at
`/home/bruno/projects/icdar2025-context-driven-str/.venvs/context-str-main`; its
complete solved package state and validation reports are captured under
`environments/main/`. The complete environment tree is ignored and must not be
committed.

| Dependency | Installed version | Classification | Basis |
|---|---:|---|---|
| Python | 3.11.11 | `ENGINEERING_CHOICE` | Supported by the selected PyTorch and ML libraries; avoids relying on host Python 3.14 wheel availability. |
| PyTorch | 2.4.1 | `ENGINEERING_CHOICE` | Stable modern candidate; official pairing with torchvision 0.19.1 and CUDA 12.4. Conflicts with PARSeq v1.0.0 metadata. |
| torchvision | 0.19.1 | `UPSTREAM_SUPPORTED` | Official PyTorch 2.4.1 pairing. |
| PyTorch CUDA runtime | 12.4 | `UPSTREAM_SUPPORTED` | Official binary runtime; compatible with the RTX 3070 and the newer 595.84 driver. |
| NumPy | 1.26.4 | `ENGINEERING_CHOICE` | Mature pre-2.0 ABI chosen to reduce older PARSeq utility risk. |
| Pillow | 10.4.0 | `ENGINEERING_CHOICE` | Approved main-stack pin. The isolated qualification used the runtime snapshot's 10.2.0; strict load, preprocessing, and inference were rechecked successfully with 10.4.0 in the final environment. |
| OpenCV | 4.10.0.84 | `ENGINEERING_CHOICE` | Pinned manylinux wheel; no CUDA compilation required for current main tasks. |
| Transformers | 4.47.1 | `ENGINEERING_CHOICE` | Nearest stable release to the pinned BLIP-2 config's recorded `4.47.0.dev0`; supports BLIP-2. |
| Accelerate | 1.2.1 | `ENGINEERING_CHOICE` | Supports explicit device placement/offload mechanics needed to investigate the 8 GB BLIP-2 candidate; using offload in an experiment remains a `HARDWARE_DEVIATION`. |
| SentenceTransformers | 3.3.1 | `ENGINEERING_CHOICE` | Supports the pinned MPNet artifact format with Transformers 4.47.1. |
| RapidFuzz | 3.11.0 | `ENGINEERING_CHOICE` | Replaces the repository's mutable `>=3.7` constraint with a reproducible candidate; it is not paper evidence. |
| OmegaConf | 2.3.0 | `ENGINEERING_CHOICE` | Stable typed/dotted configuration access for repository infrastructure. |
| pytest | 8.3.4 | `ENGINEERING_CHOICE` | Development/test runner only. |
| timm | 0.9.16 | `UPSTREAM_REQUIRED` for the selected PARSeq runtime snapshot | Exact pin in official commit `1902db04...`; strict checkpoint loading was qualified. |
| PyTorch Lightning | 2.2.0.post0 | `UPSTREAM_REQUIRED` for the selected PARSeq runtime snapshot | Exact official runtime pin; compatible with PyTorch 2.x. |
| torchmetrics | 1.3.1 | `UPSTREAM_REQUIRED` for the selected PARSeq runtime snapshot | Exact official runtime pin. |
| NLTK / LMDB | 3.8.1 / 1.4.1 | `UPSTREAM_REQUIRED` for the selected PARSeq runtime snapshot | Exact official core/test pins used by the qualification. |
| PyYAML | 6.0.1 | `UPSTREAM_REQUIRED` for the selected PARSeq runtime snapshot | Exact official runtime pin used for model configuration loading. |
| setuptools | 69.1.0 | `UPSTREAM_REQUIRED` for the selected PARSeq runtime snapshot | Supplies `pkg_resources`, required transitively by the pinned Lightning runtime. |

Transformers 4.47.1 requires Python 3.9 or newer and its source declares PyTorch 1.11
or newer. SentenceTransformers 3.3.x recommends Python 3.9+, PyTorch 1.11+, and
Transformers 4.34+. The pinned BLIP-2 configuration records Transformers
`4.47.0.dev0`. The installed stable 4.47.1 selection therefore avoids silently loading
that artifact through an older pre-update BLIP-2 implementation.

### Approximate-artifact compatibility gate

Only the pinned PARSeq checkpoint was downloaded into a temporary qualification
directory. BLIP-2 and MPNet weights were not downloaded:

| Candidate | Frozen provenance identity | Gate result |
|---|---|---|
| PARSeq-S | Checkpoint `ART-PARSEQ-V100-PARSEQ-CANDIDATE`; runtime `ART-PARSEQ-PYTORCH2-RUNTIME-CANDIDATE` | **COMPATIBLE_FOR_APPROXIMATE_MAIN**; `ENGINEERING_CHOICE`, not strict evidence |
| BLIP-2 | `ART-BLIP2-HF-FLANT5XL-REPOSITORY-CANDIDATE`; revision `0eb0d3b4...` | Metadata/config is compatible with the proposed Transformers 4.47.1 stack; weights remain untested and unavailable locally. |
| MPNet | `ART-ALL-MPNET-BASE-V2-REPOSITORY-CANDIDATE`; revision `e8c3b32e...` | Artifact format is supported by the proposed SentenceTransformers/Transformers stack; weights remain untested and unavailable locally. |

The frozen PARSeq v1.0.0 package metadata remains incompatible with the modern main
stack: it requires PyTorch 1.10.2, torchvision 0.11.3, Lightning 1.4.9, torchmetrics
0.6.2, and timm 0.4.12. The checkpoint itself remains unchanged at release v1.0.0,
commit `315d19be8ef473a864950ab497a649a69e37c6a4`, with SHA256
`bb5792a68e367476abca029cbf8699abc805f3d3dc7e57aae45c8ec4f7b7cd00`.

The separately pinned official runtime is commit
`1902db043c029a7e03a3818c616c06600af574be`. Official commit
`970f2662db03d90191135ad0f2ceefa9e980e4a7` introduced the PyTorch/Lightning 2.x
requirements and Boolean attention masks; follow-up `056c792d...` corrected the mask
tensor dtype. The selected later commit includes both changes. The earlier
`4cdf0bf9...` refactor moved inference into a nested plain `nn.Module`; the official
pretrained loader consequently targets `model.model`, preserving the unprefixed
v1.0.0 checkpoint schema. The encoder construction was also adapted to newer `timm`
APIs by disabling the classifier/class token at construction instead of deleting those
modules afterward; the patch grid, architecture hyperparameters, parameter names, and
tensor shapes remained compatible.

The qualification used a temporary environment, not `context-str-main`: Python
3.11.11, PyTorch 2.4.1+cu124, torchvision 0.19.1+cu124, NumPy 1.26.4, Pillow 10.2.0,
timm 0.9.16, Lightning 2.2.0.post0, and torchmetrics 1.3.1. All 175 checkpoint tensors
loaded strictly with no missing, unexpected, or shape-mismatched keys. Architecture
hyperparameters remained 32x128 input, 4x8 patches, 384 embedding dimensions, 12/6-head
encoder, 1/12-head decoder, maximum label length 25, autoregressive greedy decoding,
and one refinement iteration.

The 94-character training charset, 36-character test charset, and special-token IDs
(EOS 0, BOS 95, PAD 96) are unchanged. Newer code renames the displayed special tokens
from `<eos>/<bos>/<pad>` to `[E]/[B]/[P]` and adds an optional raw-decode API; normal
EOS-truncated greedy decoding is unchanged. The official inference transform remains
RGB, bicubic resize to 32x128, `ToTensor`, and normalization by mean/std 0.5. Although
torchvision's transform representation now reports `antialias=True` instead of the old
`antialias=None`, all six official demo images produced byte-identical float32 input
tensors across the two tested runtimes.

The modern CUDA runtime and original v1.0.0/PyTorch 1.10.2 CPU runtime produced the
same `1x26x95` logits shape and identical decoded strings on all six official images;
the maximum absolute per-token confidence difference was
`8.344650268554688e-07`. Modern batch-size-1 inference was deterministic and used
106,748,928 bytes peak allocated and 119,537,664 bytes peak reserved CUDA memory on the
RTX 3070. The legacy comparison required clearing an executable-stack flag on the
temporary PyTorch shared library to run on the hardened host; no checkpoint byte was
modified.

This pairing is therefore `COMPATIBLE_FOR_APPROXIMATE_MAIN`, `APPROXIMATE`, and an
`ENGINEERING_CHOICE`. It does not show which PARSeq checkpoint or code the paper used,
so strict PARSeq remains `UNRESOLVED` and blocking. A third PARSeq environment is not
recommended for the approximate pipeline.

On host `paper`, the exact runtime commit was acquired as GitHub's commit archive
because HTTPS Git cloning unexpectedly requested credentials. The frozen tarball SHA256
is `f57289cb1fc7fe41f488a52ea48f8a7f114252ef405b0b64efc1d528c29b48ef`. The
APPROXIMATE adapter and asset checker accept this source form only when that hash,
the archive root revision, all extracted file bytes, executable bits, and the absence
of extra extracted files are verified. This transport choice is an `ASSUMPTION`; it
does not alter or strengthen the paper-level PARSeq evidence.

The main-environment validator retains the RTX 3070 as its default hardware target.
For an explicitly selected host GPU it accepts `--expected-gpu-fragment`, records the
actual device and marks a non-3070 expectation as `HARDWARE_DEVIATION`; it does not
reinterpret a successful RTX 3090 run as target-hardware equivalence.

On 2026-09-03, host `paper` recreated the exact 70-package snapshot in the user-local
`.venvs/context-str-main-py311` environment with project-local uv 0.12.5 and managed
CPython 3.11.11. `uv pip check` passed, and the sorted installed state exactly matched
`environments/main/installed.txt`. Physical CUDA qualification passed on the NVIDIA
GeForce RTX 3090 with PyTorch/torchvision `2.4.1+cu124`/`0.19.1+cu124`, CUDA runtime
12.4, and cuDNN 90100. The basic CUDA check peaked at 8,521,216/23,068,672 bytes
allocated/reserved. The archive-verified PARSeq runtime strictly loaded all 175 tensors,
decoded `cute-184.jpg` as `SALMON`, and peaked at
107,701,760/119,537,664 bytes allocated/reserved. The qualification JSON SHA256 is
`23f4e739966ef686b8ef3646e77164f4410d41bec94423bbba9bf9e70e9ed685`.
This qualifies the host environment for APPROXIMATE work; the RTX 3090 remains a
`HARDWARE_DEVIATION` from the primary RTX 3070 target and supplies no paper provenance.

The same host environment subsequently qualified the pinned MPNet/scoring branch on
CPU in float32. Repeated batch embeddings were bit-identical, batch-versus-single
encoding differed by at most `8.195638656616211e-08`, and the frozen raw lexical and
three named aggregation paths executed at `alpha=0.6`, `beta=0.4`, `tau=0.8`. The
qualification JSON SHA256 is
`32ce49e73b12a0e57ba1ec91cdf6b4dc29e33716679fe51bfb0d9a13da09a1db`.
This is an APPROXIMATE runtime qualification, not a Table 5 result and not evidence for
the paper's unresolved MPNet/scoring choices.

The final `context-str-main` check retained the unchanged checkpoint and separately
pinned runtime source. All 175 tensors again loaded strictly with no missing or
unexpected keys, and `cute-184.jpg` decoded as `SALMON` on the RTX 3070. The CUDA
tensor/matrix smoke test peaked at 8,521,216 bytes allocated and 23,068,672 bytes
reserved; PARSeq inference peaked at 107,701,760 bytes allocated and 119,537,664 bytes
reserved. `uv pip check` reported all 70 installed packages compatible. Neither
Detectron2 nor DeepSolo's `adet` package is importable in this environment.

## Isolated `context-str-deepsolo` candidate stack

| Dependency | Proposed version | Classification | Basis |
|---|---:|---|---|
| Python | 3.8.18 | `ENGINEERING_CHOICE` within upstream 3.8 requirement | Last Python 3.8 maintenance release. |
| PyTorch | 1.9.0+cu111 | `UPSTREAM_REQUIRED` | Frozen DeepSolo installation instructions. |
| torchvision | 0.10.0+cu111 | `UPSTREAM_REQUIRED` | Frozen DeepSolo installation instructions. |
| CUDA runtime | 11.1 wheel/runtime | `UPSTREAM_REQUIRED` | Frozen DeepSolo/Detectron2 binary pairing. |
| Detectron2 | 0.6, commit `d1e04565...` | `UPSTREAM_REQUIRED` | `ART-DEEPSOLO-RUNTIME-CANDIDATE` / `SRC-DETECTRON2-V06`. |
| DeepSolo | commit `dbadae99...` | `UPSTREAM_REQUIRED` | Frozen official source in provenance. |
| NumPy / OpenCV | 1.20.0 / 4.5.5.62 | `UPSTREAM_REQUIRED` | Frozen DeepSolo requirements. |
| Remaining pip packages | Exact versions in `environments/deepsolo/environment.yml` | `UPSTREAM_REQUIRED` | Frozen DeepSolo requirements. |

The table preserves the earlier environment specification. The qualified container
reports Python 3.8, but its exact micro version was not included in the final host
record and must not be inferred from the proposed 3.8.18 value. A single checkpoint is
now qualified for APPROXIMATE inference below; `D-001` remains blocking for STRICT.

### M6 Docker qualification boundary

The physical host qualified Docker image
`context-str-deepsolo:m6-qualified-20260818` at immutable local image ID
`sha256:30e0...755c`. It contains Python 3.8, PyTorch 1.9.0+cu111, torchvision
0.10.0+cu111, CUDA 11.1, cuDNN 8.0.5, and Detectron2 0.6. `adet._C` was built in the
persistent frozen source for sm_86. The ResNet-50 Total-Text config/checkpoint pairing
loaded and ran real CUDA inference with cuDNN enabled. The final staged-worker repository
smoke returned three detections in 0.2597780159994727 seconds with 869,216,256 bytes
peak allocated and 1,270,874,112 bytes peak reserved CUDA memory. The physical main
environment (PyTorch 2.4.1+cu124, RTX 3070) passed all 122 repository tests. Exact hashes, build flags,
mounts, CPU-feature requirements, memory, timing, and diagnostic history are canonical
in `docs/DEEPSOLO_RUNTIME.md`.

This resolves the APPROXIMATE runtime gate for that single candidate. It does not
qualify the other seven candidate configurations or identify what the paper used.
The original Docker base-image construction history was not captured, so a Dockerfile
rebuild remains a reconstruction until separately validated; the qualified local image
ID is authoritative for M6.

The canonical validation command is:

```bash
DEEPSOLO_ENV_FILE="$PWD/docker/deepsolo/.env" scripts/deepsolo_smoke.sh
```

The qualified manifest is tracked without model bytes under
`docker/deepsolo/runtime_manifest.json`. Other manifests remain generated outside Git
with `scripts.write_deepsolo_runtime_manifest`; hashes are computed, not guessed.

The qualified Docker runtime supplies its own CUDA 11.1 toolkit and successfully built
the DeepSolo custom extension. This does not add a CUDA toolkit to `context-str-main`
or the host environment.

## CUDA strategy

For the installed main environment, strategy **A** is selected: NVIDIA driver plus the
PyTorch-provided CUDA 12.4 runtime. PyTorch, torchvision, OpenCV, Transformers,
SentenceTransformers, and the current repository code do not require local CUDA
extension compilation. A system CUDA Toolkit would add mutable compiler/header state
without providing a current benefit.

The `nvidia-smi` value 13.2 is only the maximum CUDA compatibility advertised by driver
595.84. It neither supplies `nvcc` nor proves that a system toolkit exists. No system
CUDA Toolkit should be installed for `context-str-main`.

The historically qualified M6 DeepSolo CUDA compilation remains confined to its
container and persistent external source. On host `paper`, where Docker is
administratively inaccessible, the separately qualified 2026-09-02 alternative compiles
only inside the `context-str-deepsolo-host` venv and external frozen source using the
already installed `/usr/local/cuda-11.1`. Its RTX-3090 subprocess qualification is an
APPROXIMATE `HARDWARE_DEVIATION`. It never installs legacy packages into
`context-str-main`, which continues to use only its PyTorch-provided CUDA runtime.

## Environment creation and snapshots

No Conda/Mamba executable was present on the physical host. The already-qualified
`uv` toolchain was therefore retained as the `context-str-main` manager rather than
introducing a second environment manager. The environment was solved at
`/tmp/context-str-main` with CPython 3.11.11, then relocated byte-for-byte to the
canonical persistent path under `.venvs/`. Its sorted package inventory exactly
matches the captured 70-package snapshot, and `uv pip check` passes. Console-script
shebangs and the venv's Python base were relocated so the canonical environment has no
runtime dependency on the temporary environment or temporary uv Python installation.

Post-relocation CPU PARSeq strict loading and inference passed with all 175 tensors
and the expected `SALMON` decode. A direct post-relocation physical-GPU rerun remains
operationally blocked because the current no-escalation execution sandbox cannot
access NVML/CUDA; this is recorded explicitly rather than reusing the earlier physical
GPU result as though it were a new run.

`environments/main/installed.txt` is the sorted output captured from `uv pip freeze`
after the successful solve; it is an installed-state snapshot, not a fabricated lock.
`environment.yml` records direct creation pins and the checkpoint/runtime separation,
`environment_metadata.json` records platform and provenance, and `validation.json`
records the successful physical-GPU validation. No cache, source checkout, model
weight, or dataset bytes are included in those snapshots.

## Primary compatibility sources

- PyTorch official version pairs: <https://pytorch.org/get-started/previous-versions/>
- Transformers 4.47.1 source/PyPI metadata: <https://github.com/huggingface/transformers/tree/v4.47.1>
- SentenceTransformers installation requirements: <https://sbert.net/docs/installation.html>
- Frozen PARSeq v1.0.0 source/checkpoint: <https://github.com/baudm/parseq/tree/v1.0.0>
- Qualified official PyTorch 2 runtime: <https://github.com/baudm/parseq/tree/1902db043c029a7e03a3818c616c06600af574be>
- Frozen DeepSolo source: <https://github.com/ViTAE-Transformer/DeepSolo/tree/dbadae995035246bad3376c7a44c015c69e9b313>
- Detectron2 v0.6: <https://github.com/facebookresearch/detectron2/tree/v0.6>
