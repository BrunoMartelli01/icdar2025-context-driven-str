# DeepSolo M6 Qualified Runtime

## Status boundary

This is the canonical operational record for the Macro-Milestone 6 DeepSolo
runtime. Future work must read it before changing, running, or qualifying DeepSolo.

The runtime and one ResNet-50 Total-Text candidate are qualified only for an
**APPROXIMATE** reproduction on the physical RTX 3070 host. Docker process
isolation and the sequential model lifecycle are `HARDWARE_DEVIATION` choices.
The paper does not identify its exact checkpoint, dataset mapping, lexicon
protocol, or multi-output reduction, so **STRICT remains blocked**.

## Immutable source and candidate identities

| Item | Qualified identity |
|---|---|
| DeepSolo source | commit `dbadae995035246bad3376c7a44c015c69e9b313` |
| Detectron2 source identity | v0.6, commit `d1e04565d3bec8719335b88be9e9b961bf3ec464` |
| Candidate artifact | `ART-DEEPSOLO-R50-TOTALTEXT-CANDIDATE` |
| Architecture/config | `TransformerPureDetector`, ResNet-50, `configs/R_50/TotalText/finetune_150k_tt_mlt_13_15.yaml` |
| Config SHA256 | `0da913239e4b3f11ae1bad19e0b104996f7c3f3607aa2a7f4b1f52e3838b702d` |
| Container checkpoint | `/opt/checkpoints/tt_res50_finetune_synth-tt-mlt-13-15.pth` |
| Checkpoint SHA256 | `ac2a01b3d8df0ee42e4632ff57a0d3a20a5706fb0ac1c5ae33b2455d0d90da68` |
| Parameters | 42,588,150 total; 42,365,750 trainable |

The pairing loaded successfully with the expected architecture and inference
fields. These bytes remain an upstream-candidate selection by `ASSUMPTION`; load
success does not associate them with the paper authors' experiments.

## Qualified Docker runtime

| Item | Physical-host qualification |
|---|---|
| Container | `context-str-deepsolo-build` |
| Image tag | `context-str-deepsolo:m6-qualified-20260818` |
| Image ID | `sha256:30e0e99d23279060e72b22fe3911a2e42b72de55e70be6349fb06ec37b35755c` |
| Restart policy | `unless-stopped` |
| Python | `/opt/context-str-deepsolo/bin/python`, Python 3.8 |
| PyTorch / torchvision | `1.9.0+cu111` / `0.10.0+cu111` |
| CUDA / cuDNN | CUDA 11.1 / cuDNN 8.0.5 |
| Detectron2 | 0.6 |
| GPU | NVIDIA RTX 3070, CUDA architecture `sm_86` |

The authoritative object is the local committed Docker image with this exact ID.
The original base-image construction history was not captured. Therefore
[`docker/deepsolo/Dockerfile`](../docker/deepsolo/Dockerfile) is a reconstruction
and validation wrapper around an explicitly supplied legacy base. A new image is
not qualified merely because it builds or imports successfully.

## Persistent source and mounts

The qualified host used:

```text
/home/bruno/projects/deps/DeepSolo-upstream/DeepSolo
  -> /workspace/DeepSolo
```

The host path is qualification metadata, not a reusable default. Scripts obtain
host paths from the ignored `docker/deepsolo/.env`. New containers bind-mount
checkpoints at `/opt/checkpoints`; checkpoints must never be copied into this
repository or reconstructed image layers. The container workdir is
`/workspace/DeepSolo`.

### Host-verified source identity

The persistent source directory is a subdirectory of the host Git checkout; its
`.git` metadata is in the checkout parent and is intentionally not mounted into the
container. The qualified image also does not need or provide the `git` executable.
Before every adapter inference, the main host process runs `git rev-parse HEAD`
against configurable `DEEPSOLO_SOURCE_ROOT` and requires
`dbadae995035246bad3376c7a44c015c69e9b313`. Failure to execute Git, resolve the
checkout, or match the frozen revision aborts before the worker starts.

The adapter adds that result as invocation-scoped
`runtime.source_identity_verification` metadata to the copied runtime manifest. The
container worker requires the verification record and matching frozen revision, but
does not run Git. It independently retains config SHA256 and checkpoint SHA256
verification against the files visible inside the container. The tracked manifest
does not claim that host verification has already occurred.

The existing qualified container has its verified checkpoint at `/opt/checkpoints`
inside the snapshot and must not be recreated merely to add a bind mount.
`DEEPSOLO_CHECKPOINT_DIR` is required only when reconstructing or recovering a new
container, where the start script bind-mounts external checkpoint bytes read-only.

## CUDA extension build

`adet._C` was built successfully in place in the persistent source with:

```text
CUDA_HOME=/usr/local/cuda
CC=/usr/bin/gcc
CXX=/usr/bin/g++
TORCH_CUDA_ARCH_LIST=8.6
MAX_JOBS=2
FORCE_CUDA=1
```

The `_C*.so`, `build/`, and Python caches are external generated artifacts and
must not be committed. The exact historical build command was not captured.
`scripts/deepsolo_build.sh` uses a conventional frozen-upstream editable pip build
as an explicit reconstruction choice; its success does not transfer the qualified
image identity.

## Proxmox CPU requirement and SIGILL history

The VM originally used Proxmox CPU type `x86-64-v2-AES`. The guest and container
saw a generic QEMU CPU without AVX, AVX2, or FMA. Full cuDNN-enabled ResNet
convolution inference then exited with `SIGILL`. Setting
`torch.backends.cudnn.enabled=False` allowed diagnostic inference, proving that
model/checkpoint loading and the non-cuDNN route were viable, but it was never
accepted as the final runtime.

The VM CPU type was changed to `host`. Guest and container now expose an Intel
Xeon E7-8880 v4 with AVX, AVX2, and FMA. With the same legacy stack, normal cuDNN
inference passes. Smoke scripts verify all three flags on host and container.
`cudnn=False` is diagnostic history only.

Final runtime configuration:

```python
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
```

## Real inference qualification

A constructed 720×1280 image containing `DEEPSOLO TEST 2026` produced three
detections. Returned fields were `bd`, `ctrl_points`, `pred_classes`,
`rec_scores`, `recs`, and `scores`; decoded examples included `2026`, `TEST`, and
`DEPSOLO`. The final persistent-container, CPU-host, cuDNN-enabled smoke reported:

```text
inference_seconds = 0.3091638289988623
peak_allocated = 869216256 bytes
peak_reserved = 1270874112 bytes
DEEPSOLO CUDNN-ON CPU-HOST: PASS
exit=0
```

The repository smoke script creates a deterministic engineering fixture with the
same dimensions and three text elements. Its rendering parameters were not part
of the earlier record, so it is not a byte-identical recreation or evaluation
sample.

The final repository-owned Docker boundary smoke passed on the physical host on
2026-08-19. It exercised the staged versioned JSON worker and recorded
`success=true`, three detections, recognized strings `deepsolo`, `2026`, and
`test`, selected output `deepsolo test 2026`, device `cuda:0`, inference time
0.2597780159994727 seconds, peak allocated CUDA memory 869,216,256 bytes, and
peak reserved CUDA memory 1,270,874,112 bytes. The main physical-host environment
was PyTorch 2.4.1+cu124 with CUDA available on an NVIDIA GeForce RTX 3070; its
final repository suite passed 122 tests in 18.67 seconds.

## Separate-process benchmark

Five warmups preceded 20 timed inferences; each returned three detections.

| Backend | Mean ms | Median ms | Min ms | Max ms | FPS |
|---|---:|---:|---:|---:|---:|
| cuDNN enabled | 214.6753376000106 | 209.9483005000593 | 207.56017600001542 | 241.08452599989505 | 4.6581969367307074 |
| cuDNN disabled, diagnostic only | 227.90747514998202 | 221.539443499978 | 215.52423200000703 | 270.0066390000302 | 4.387745506556628 |

These are physical-host qualification measurements, not paper latency or Table 4
reproduction. The final runtime is the cuDNN-enabled row.

## Versioned boundary and lifecycle

The main environment never imports Detectron2 or DeepSolo. The Docker adapter
copies a UUID-scoped request bundle into the persistent container and runs the
existing version-1 JSON worker. Request/response fields, hashes, raw text, token
and instance confidence, predicted classes, geometry, latency, placement, memory,
runtime metadata, reduction policy, and structured traceback behavior are
preserved.

The bundle stages this repository's `deepsolo/` package beside the absolute
entrypoint `context_str_m6_deepsolo_json_worker.py`. Before importing `deepsolo`,
the bootstrap derives the bundle root from its own resolved path, inserts that root
at `sys.path[0]`, clears any preloaded `deepsolo` modules, and invalidates import
caches. It then verifies that both `deepsolo.__file__` and
`deepsolo.worker.__file__` resolve under the staged package root; any other origin
fails explicitly. Resolved entrypoint/package/worker paths are retained in response
metadata for diagnosis.

Dispatch therefore does not depend on the historical `run_deepsolo_worker.py`
basename, current-directory lookup, or ambient `PYTHONPATH`. The Docker working
directory remains `/workspace/DeepSolo` for upstream imports after the M6 worker has
been selected. This prevents DeepSolo/Detectron2 CLI collisions while preserving the
version-1 `--request`/`--response` interface.

Physical-host diagnostics confirmed that this bootstrap selects the staged worker and
that its `--help` exposes the M6 JSON arguments. A later argparse failure had a distinct
cause inside inference: the frozen source's `adet.data.augmentation` import first
executes `adet/data/__init__.py`, whose `builtin.py` parses process arguments at import
time. The already-valid M6 `--request`/`--response` arguments therefore caused
`SystemExit(2)` after correct worker dispatch.

The repository worker no longer imports `adet.data`. ResNet-50 never needs the ViTAE
padding branch. For ViTAEv2-S, the private inference helper reproduces the frozen
upstream `Pad` behavior through Detectron2 `PadTransform`: divisor 32, zero padding on
the bottom/right only, and no padding on an already divisible dimension. The padding
semantics are `REPOSITORY_SPECIFIED`; implementing them locally is engineering process
isolation and is not a scientific preprocessing change. ResizeShortestEdge behavior is
unchanged.

DeepSolo runs only when the selected M5 scoring strategy produces
`needs_fallback=true`. Segmentation, PARSeq, BLIP-2, and MPNet are explicitly
released before Docker DeepSolo starts. Concurrent BLIP-2/DeepSolo residency is
prohibited.

Multi-output mapping remains unresolved. Available `ASSUMPTION` policies are
`highest_score`, `source_order_space_join`, and `reading_order_space_join`; raw
outputs are retained regardless of reduction.

## Physical-host smoke

Copy `.env.example` to the ignored `.env`, set absolute source/checkpoint paths,
then run from the repository root:

```bash
DEEPSOLO_ENV_FILE="$PWD/docker/deepsolo/.env" scripts/deepsolo_smoke.sh
```

The script validates the exact image ID, persistent container/restart policy,
host/container AVX+AVX2+FMA, source and checkpoint hashes, model load, parameter
counts, cuDNN flags, real CUDA inference, response schema, and three detections.
The final physical-host run passed with the measurements above. It writes
generated files outside Git. Normal pytest never executes Docker/GPU.

## Qualified non-Docker host subprocess on `paper`

Docker on `paper` is administratively managed and unavailable to the reproduction
user. It is a read-only/inaccessible boundary: repository work must not change daemon,
socket, groups, toolkit, images, or containers. Consequently the qualified Docker path
is `BLOCKED` on this host unless authorized access is supplied later.

The existing `DeepSoloSubprocessAdapter` permits a separate unprivileged Python 3.8
process without importing Detectron2 into `context-str-main`. For this host only, the
repository supplies `scripts/prepare_deepsolo_host.py` and
`scripts/deepsolo_host_smoke.sh`. They preserve the frozen DeepSolo revision,
checkpoint/config hashes, legacy package versions, JSON schema, failure propagation,
and conditional fallback semantics. The environment is compiled against the already
installed `/usr/local/cuda-11.1`; no system package or Docker mutation is performed.
The host preparation accepts either a `git_checkout`, verified with `git rev-parse
HEAD`, or a `github_commit_archive`. The archive path is fail-closed: callers must
provide the frozen revision, a commit-marker file, the original archive, and its
expected SHA256; the report records each item without claiming a Git HEAD. The frozen
archive for commit `dbadae995035246bad3376c7a44c015c69e9b313` is the official GitHub
codeload commit archive and has SHA256
`828c63e1c19f07d03ae094e4dd39a1eb093b0660a604bef75324f8e716693bfb`.
Because `scikit-image==0.15.0` has no CPython 3.8 wheel, host setup builds its
sdist after installing the pinned NumPy build prerequisite. `wheel==0.37.1` is
a build-only `ASSUMPTION`, paired with the already frozen
`setuptools==59.5.0`; the runtime report and pip freeze record both versions.
The CUDA 11.1 Detectron2 wheel selected by the frozen `detectron2==0.6`
requirement reports the exact PEP 440 local version `0.6+cu111`; qualification
checks that full installed identity rather than misclassifying the local build
tag as a version change.
The qualified provenance identity is
`ART-DEEPSOLO-PAPER-HOST-RUNTIME-CANDIDATE`.

This choice is a `HARDWARE_DEVIATION`: it replaces the previously qualified Docker
process isolation with a local venv subprocess because administrative container access
is unavailable. Python 3.8.10 is an `ASSUMPTION` because the Docker record established
only Python 3.8, not its micro version. The physical `paper` qualification used an
NVIDIA GeForce RTX 3090 rather than the primary RTX 3070 and is therefore not a
target-hardware performance result.

The route was qualified for **APPROXIMATE** use on 2026-09-02. The exact environment
audit, `adet._C` import, frozen source/archive/config/checkpoint checks, strict checkpoint
compatibility, staged repository worker, CUDA inference, response schema, and memory
record all passed. The 720x1280 fixture produced `deepsolo`, `test`, and `2026`, reduced
in reading order to `deepsolo test 2026`. Model loading took
0.7447327279951423 seconds, inference took 0.147001147037372 seconds, and total worker
time was 0.8934014710830525 seconds. Peak CUDA memory was 869,216,256 bytes allocated
and 1,270,874,112 bytes reserved. The strict compatibility record contained no missing,
unexpected, or incorrectly shaped keys and reported 42,588,150 total parameters and
42,365,750 trainable parameters.

Qualification record hashes are:

| Record | SHA256 |
|---|---|
| environment report | `1e1c5763bbb645eb55a877de77484acfd1ec13c52129e807ad960b800e8f2d76` |
| pip freeze | `91473ada63f96a4f73c195c01dc5faafce7574cf5a5ff9deee8e990da493f277` |
| setup report | `b513d841d4b025f95cd9f3e95da2b0daec239ef169b6e962eaf0aeda1b900096` |
| runtime manifest | `c52a65b7d844cd35bcf6e38e3cb58c5194851c8327cbdaba9b313869ec2c7ffa` |
| smoke image | `31c00a1ee43108e70849a1cdc015a2013837d4a8fb8339b9b168295bf9cdfa2f` |
| smoke response | `82dac26c016515f88b625a654acfa2d9aa026d420f5871f789d5cd1ad37ce4a8` |

The first repository-boundary attempt exposed an engineering defect: resolving the
venv `bin/python` symlink selected `/usr/bin/python3.8`, bypassed `pyvenv.cfg`, and made
installed modules such as `cv2` unavailable. The adapter and runtime-manifest writer now
preserve the absolute venv entry-point path without resolving the symlink. The saved
successful response then exposed a stale postcondition expecting Detectron2 `0.6`
rather than its installed PEP 440 identity `0.6+cu111`; the smoke now consumes the same
frozen version constant as the environment audit. Neither correction changes model
inputs, weights, configuration, or inference semantics.

This qualification does not inherit or replace the M6 Docker identity and does not
support a STRICT or Table 5 claim. The paper's DeepSolo checkpoint selection and
multi-output mapping remain unresolved.
