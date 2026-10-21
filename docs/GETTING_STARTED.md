# Getting started on a new machine

Run commands from the repository root. Linux/Git/Python 3.8+ are sufficient for
navigation. Scientific code uses the pinned main Python 3.11.11 environment;
DeepSolo uses its own legacy stack. Do not combine them.

## 1. Clone and explore

Clone the URL supplied by the maintainer, then enter the clone:

```bash
git clone https://github.com/BrunoMartelli01/icdar2025-context-driven-str.git
cd icdar2025-context-driven-str
python3 -m scripts.collaboration check
python3 -m research groups
python3 -m research find 'DeepSolo' --limit 5
```

The URL is the intended collaboration destination; its existence/access must be
confirmed when publishing. A private repository requires an invited account.
Read [AGENTS.md](../AGENTS.md), [the map](REPOSITORY_MAP.md), and
[external assets](EXTERNAL_ASSETS.md). For documentation-only work no model or
dataset download is needed. `research compare/check/overview` require exact saved
reports; `groups/find/section` work on code/documentation alone.

## 2. Create an isolated main environment when needed

The following command **installs** a repo-local uv 0.12.5, Python 3.11.11 and
the exact package versions in `environments/main/installed.txt`. Run it only
when you intend to create that environment. It uses network access and requires
system `python3-venv`; it does not install system packages or DeepSolo.

```bash
bash scripts/create_main_environment_host.sh
.venvs/context-str-main-py311/bin/python -c 'import torch; print(torch.__version__)'
```

This is an installed-version snapshot for Linux/CUDA 12.4, not a hash-complete
dependency lock. Read [the snapshot](../environments/main/README.md) and
[environment boundaries](ENVIRONMENTS.md) before changing platforms. A newly
created environment still needs qualification on the actual machine. The
historical root `requirements.txt` must not replace this procedure.

Set external roots explicitly; choose locations with enough disk space:

```bash
export CONTEXT_STR_DATA_ROOT=/absolute/path/to/datasets
export CONTEXT_STR_ARTIFACTS_ROOT=/absolute/path/to/artifacts
export CONTEXT_STR_RUNS_ROOT=/absolute/path/to/runs
export HF_HOME="$CONTEXT_STR_ARTIFACTS_ROOT/huggingface-home"
export HF_HUB_CACHE="$CONTEXT_STR_ARTIFACTS_ROOT/huggingface"
export TORCH_HOME="$CONTEXT_STR_ARTIFACTS_ROOT/torch"
```

These are shell conventions for the examples. Dataset/model programs still
receive their documented path flags; they do not all read these variables.
Never commit populated credentials or local environment files.

## 3. Run offline component checks

Using the already-created main environment:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venvs/context-str-main-py311/bin/python -m pytest -q \
  tests/research tests/scripts/test_collaboration.py tests/test_config.py \
  tests/test_package_imports.py tests/localize/test_algorithm1.py \
  tests/deepsolo/test_contract.py
```

These tests use local/synthetic fixtures and mocks. No upstream model download
or live DeepSolo process is required. For a separate synthetic segmentation
forward smoke on CPU:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venvs/context-str-main-py311/bin/python -m pytest -q \
  tests/segmentation/test_pipeline_smoke.py
```

This is a **smoke-test result**, never a paper accuracy result. The complete
research suite has additional environment/data requirements; a passing subset
must not be reported as full-suite qualification.

## 4. Obtain scientific inputs and qualify the host

Follow [EXTERNAL_ASSETS.md](EXTERNAL_ASSETS.md) to retrieve the exact paper,
source revisions, candidate weights and dataset manifests. Check all recorded
hashes before loading. Candidate artifacts are APPROXIMATE only when the paper
choice is unresolved. A missing strict artifact must stop execution.

For GPU work, inspect current hardware/processes before each run:

```bash
hostname -f
whoami
pwd
nvidia-smi
```

The historical target was an RTX 3070/8 GB. PAPER has also hosted RTX 3090 runs;
past memory measurements do not qualify the current GPU. Record actual hardware,
precision, input shape, batch size, device placement, allocated/reserved VRAM
peaks, failures and all hardware deviations. BLIP-2 is a particular memory risk.

For a PARSeq/main-host qualification, with the verified source layout in the
asset guide and the cache directory present:

```bash
bash scripts/qualify_main_host.sh "$CONTEXT_STR_ARTIFACTS_ROOT" 'YOUR GPU MODEL'
```

That script uses the external PARSeq commit archive and checkpoint, and writes
an external qualification report. It intentionally requires a real CUDA device.
For CPU-only work use the offline tests above; do not fabricate a GPU pass.

## 5. Prepare data, then launch one explicit experiment

See [dataset preparation](EXTERNAL_ASSETS.md#datasets-and-prepared-targets) for
the required download layout. The portable underlying entry points are:

```bash
.venvs/context-str-main-py311/bin/python -m scripts.prepare_scene_text_segmentation \
  --download-root "$CONTEXT_STR_DATA_ROOT" \
  --output-root "$CONTEXT_STR_DATA_ROOT/scene-segmentation-v1"

.venvs/context-str-main-py311/bin/python -m segmentation.train \
  --config configs/approximate/segmentation_total_text_author_hyperparameters.yaml \
  --data-root "$CONTEXT_STR_DATA_ROOT/scene-segmentation-v1" \
  --manifest-root "$CONTEXT_STR_DATA_ROOT/scene-segmentation-v1" \
  --output-root "$CONTEXT_STR_RUNS_ROOT/total-text-new-run" \
  --device cuda --qualification-max-steps 1
```

This is an **approximate reproduction**, not a strict result. Qualify before a
full run. Inspect saved resolved configuration, dataset hashes, memory and
failure reports before removing `--qualification-max-steps 1` for full training.
Do not overwrite a historical run directory. Different preparation bytes may
fail frozen manifest hashes; investigate rather than editing hashes to pass.

For DeepSolo read [DEEPSOLO_RUNTIME.md](DEEPSOLO_RUNTIME.md), the M6 manifest,
and strict-status entries first. An independently rebuilt runtime needs new
qualification; building successfully does not recreate the original image ID.

## 6. Work on a branch

Follow [CONTRIBUTING.md](../CONTRIBUTING.md). Keep one scientific component per
change, retain original audits, run relevant tests, and record unresolved inputs.
Historical host scripts and configs may contain absolute paths. Prefer the
underlying Python module and a separate, explicitly supplied local path file.
