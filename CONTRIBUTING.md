# Contributing

Read [AGENTS.md](AGENTS.md), [the repository map](docs/REPOSITORY_MAP.md), and
[setup](docs/GETTING_STARTED.md) first. These instructions apply to humans and
AI agents. Documentation/navigation work can be done without datasets or a GPU.

## A bounded change

1. Start a branch for one component or scientific question. State the intended
   behavior and link its ambiguity ID or evidence source.
2. Before changing a scientific component, inspect its implementation and the
   corresponding section of `papers/2503.15639v2.pdf`.
3. Classify each scientific choice as `PAPER_SPECIFIED`, `REPOSITORY_SPECIFIED`,
   `ASSUMPTION`, `HARDWARE_DEVIATION`, or `UNRESOLVED`. Record evidence, alternatives,
   and blockers in `docs/PAPER_AMBIGUITIES.md` when the task permits edits.
4. Keep `configs/strict/`, `configs/approximate/`, and `configs/demo/` isolated.
   Supply paths explicitly. Do not promote an approximate choice to a strict default.
5. Add proportional tests, run relevant smoke/unit checks, inspect the complete
   diff, and report failures without suppressing them.

## Local checks

```bash
python3 -m scripts.collaboration check
python3 -m research groups
```

With the main environment already installed:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venvs/context-str-main-py311/bin/python -m pytest -q \
  tests/research tests/scripts/test_collaboration.py tests/test_config.py \
  tests/test_package_imports.py tests/localize/test_algorithm1.py \
  tests/deepsolo/test_contract.py
```

GitHub CI runs source/navigation checks on CPU. It does not qualify a GPU,
DeepSolo, the complete test suite, or a paper result. Component tests can require
additional packages, data, or historical paths; select the documented environment
instead of installing everything into the main environment.

## Evidence and generated indexes

On a code-only clone, after changing documents:

```bash
python3 -m scripts.build_experiment_library
python3 -m scripts.build_experiment_library --check
python3 -m research catalog --source-only
python3 -m research catalog --source-only --check
```

On a full workspace, supply the same explicit external roots used for that
catalog; see [docs/README.md](docs/README.md). Preserve original audits and
the [historical workspace inventory](docs/history/WORKSPACE_AUDIT_INDEX.json).
A new catalog is a navigation view, not evidence of scientific success.
Use English for maintained prose and labels. Preserve original scientific
payloads and follow [the language policy](docs/LANGUAGE_POLICY.md).

## What belongs in a pull request

Include the concrete change, its evidence, tests and failures, scientific
classification, and limitations. Keep result classes and protocols separate.
For a run, preserve resolved configuration, code identity, environment, hardware,
seed, dataset/checkpoint hashes, per-sample outputs, failures and denominators in
an external run directory. Add only a small provenance record or link to the PR.
A new training run does not by definition recreate a missing historical checkpoint.

Never commit datasets, model weights, environments, credentials, caches or large
outputs. `scripts.collaboration check` also rejects prohibited tracked files,
symlinks, oversized source files and common credential formats. This focused
check is not a complete secret audit. Do not force-add ignored artifacts.

Do not automatically commit other people's changes, rewrite history, upgrade
packages or launch GPU work as part of a documentation PR.
