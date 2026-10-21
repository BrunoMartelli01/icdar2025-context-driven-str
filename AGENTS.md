# Scientific Reproduction Engineering Rules

These instructions apply permanently to all Codex work in this repository. They govern implementation, experiments, evaluation, documentation, and reporting for the reproduction of:

> *A Lightweight Context-Driven Training-Free Network for Scene Text Segmentation and Recognition* (arXiv:2503.15639v2)

The authoritative local paper is `papers/2503.15639v2.pdf`. Before changing a scientific component, inspect both its existing implementation and the corresponding paper section.

## Persistent host and repository navigation

- The user-designated execution machine is **`bmartelli@paper.dinfo.unfi.it`**
  (recorded verbatim from the user's instruction). Existing historical documents
  also contain `paper.dinfo.unifi.it`; do not silently substitute the spelling.
- Before constructing commands, check `hostname -f`, `whoami`, and `pwd`. If
  already executing as `bmartelli` on `paper`, run commands directly; do not SSH
  back into the same machine. From another machine, use the user-designated SSH
  destination and verify connectivity before depending on it.
- Repository on PAPER: `/extra1/bmartelli/projects/icdar2025-context-driven-str`.
  Operational paths and local/SSH command examples are in `docs/HOST_OPERATIONS.md`.
  Pass dataset, run, checkpoint and cache locations explicitly to scientific tools;
  these host notes are not scientific configuration defaults.
- Start repository exploration with `docs/README.md`, `docs/AUDIT_INDEX.md`, and
  `python3 -m research groups`. Use `python3 -m research find QUERY` and
  `python3 -m research section AMBIGUITY_ID` for targeted evidence reads.
- For a source-only collaboration clone, also read `docs/GETTING_STARTED.md`,
  `docs/REPOSITORY_MAP.md`, and `docs/EXTERNAL_ASSETS.md`. Run
  `python3 -m scripts.collaboration check` before preparing a Git snapshot.
  Historical external evidence is inventoried in
  `docs/history/WORKSPACE_AUDIT_INDEX.json`; its presence there does not mean
  the external bytes are included or downloadable.
- `.rgignore` excludes generated artifacts and environments from generic searches.
  Search a specific excluded audit with `rg --no-ignore` and its exact path.
  Preserve historical audits; summaries and indices are navigation aids, not
  replacements for source evidence. Verify current GPU hardware before each run;
  historical RTX 3070 qualifications do not qualify PAPER's hardware.

## Scientific objective

The objective is scientific reproduction of the paper, not merely making the demo run.

Use this priority order for all decisions:

1. Fidelity to the paper.
2. Reproducibility.
3. Correctness.
4. Testability.
5. Compatibility with target hardware.
6. Performance.
7. Convenience.

Never silently replace a paper component with an easier alternative. Clearly distinguish strict paper reproduction from demos, debugging paths, approximations, and hardware-adapted experiments.

## Strict paper-reproduction components

The strict paper-reproduction configuration must use:

- An Attention-Gated U-Net segmentation architecture with a pretrained ResNet-50 encoder.
- Connected components or contours for localization.
- Component-area filtering for localization. Bounding-box area is not an equivalent silent substitute.
- Padding around localized components.
- Sorting localized components by component area.
- A maximum of 10 localized text blocks per image.
- PARSeq-S for recognition.
- BLIP-2 for scene captioning.
- MPNet embeddings and cosine similarity for semantic similarity.
- Levenshtein/fuzzy matching for lexical similarity.
- Main scoring parameters `alpha=0.6`, `beta=0.4`, and `tau=0.8`.
- DeepSolo as the fallback recognizer.

TrOCR may remain available for demos and debugging, but it MUST NOT replace PARSeq-S in experiments presented as paper reproduction. DeepSolo MUST NOT be removed, disabled, or replaced in experiments claiming to reproduce the paper's recognition results.

## Scientific evidence and ambiguity handling

Never silently resolve an ambiguity or inconsistency in the paper or repository. Every implementation decision affecting scientific behavior must be classified as exactly one of:

- `PAPER_SPECIFIED`: explicitly supported by the paper.
- `REPOSITORY_SPECIFIED`: supplied by the original repository but not explicitly supported by the paper.
- `ASSUMPTION`: a necessary, reasoned choice not established by either source.
- `HARDWARE_DEVIATION`: a choice made because the target hardware differs from the paper environment.
- `UNRESOLVED`: a scientifically material question that has not yet been resolved.

Record assumptions, evidence, alternatives considered, and unresolved scientific ambiguities in `docs/PAPER_AMBIGUITIES.md`. Create or update that document only when the active task authorizes repository modifications. Never invent a value merely to match a reported result.

Preserve and investigate at least these known ambiguities:

- Exact AG-UNet decoder and channel architecture.
- Bottleneck compression.
- Segmentation mask and logit resolution.
- ResNet-50 pretrained weight revision.
- Image normalization.
- COCO-TS dataset size and protocol.
- ICDAR13 evaluation split.
- Exact BLIP-2 checkpoint.
- Exact BLIP-2 decoding parameters.
- Whether caption length means words or tokens.
- Exact MPNet checkpoint.
- PARSeq-S checkpoint, revision, charset, and decoding.
- Fuzzy-string preprocessing.
- Context-score aggregation.
- Recognition-accuracy definition.
- DeepSolo checkpoint selection.
- Handling multiple DeepSolo text outputs.
- FLOP calculation protocol.

If an ambiguity blocks a strict result, report it as `UNRESOLVED`; do not conceal it behind a default.

## Target hardware and hardware deviations

Primary development hardware is:

- Ubuntu Linux.
- One NVIDIA RTX 3070.
- 8 GB VRAM.
- Single-GPU execution.

The paper's environment is not identical to this machine. Classify a hardware-driven change as `HARDWARE_DEVIATION` whenever it differs from the paper, including:

- Mixed precision.
- CPU offload.
- Quantization.
- Reduced batch size.
- Gradient accumulation.
- Sequential model loading.
- Separate model processes.
- Activation checkpointing.

Do not present a result obtained with an undocumented hardware deviation as an exact reproduction. Prefer paper fidelity over memory optimization. If an optimization can change model outputs, document that risk and evaluate its effect where feasible.

## Environments and dependency isolation

Before changing, running, or qualifying DeepSolo, read
`docs/DEEPSOLO_RUNTIME.md`, the canonical M6 Docker/runtime qualification record,
and the DeepSolo entries in `docs/STRICT_REPRODUCTION_STATUS.md`.

Do not force every component into one Python environment when dependencies conflict. The initial audit found major incompatibilities between the modern main pipeline and the legacy DeepSolo/Detectron2 stack.

Prefer isolated environments, especially:

1. A main reproduction environment.
2. A DeepSolo environment.

Integration between environments may use deterministic, versioned file-based interfaces or controlled subprocesses. Define schemas, preserve ordering and encodings, propagate failures, and test the boundary. Do not silently swallow subprocess, model-loading, or inference errors.

Do not install or upgrade dependencies until the active task explicitly authorizes environment creation or package installation.

## GPU memory discipline

Never assume all models fit simultaneously in 8 GB VRAM. BLIP-2 is a known memory risk.

GPU-intensive changes must be tested for:

- Successful execution.
- Peak allocated VRAM.
- Peak reserved VRAM.
- Out-of-memory behavior and error reporting.

Record the model set, precision, device placement, input shape, and batch size with memory measurements. Model unloading, offload, quantization, or precision changes that can affect output must be documented using the classification above.

## Scientific modes and configuration boundaries

Never hardcode:

- Dataset paths.
- Checkpoint paths.
- Hugging Face cache paths.
- Model revisions.
- Experimental thresholds.

Every scientific experiment MUST run in one of two explicitly selected modes. The mode must be recorded in the resolved configuration and result metadata.

### STRICT mode

STRICT mode is reserved for exact paper-reproduction work:

- Use only parameters, artifacts, checkpoints, revisions, and procedures supported by authoritative evidence and recorded provenance.
- Any unresolved parameter or artifact classified as blocking MUST cause an explicit, actionable failure before execution. Do not guess, default, substitute, or continue with a partial configuration.
- A repository default or implementation choice is not sufficient evidence by itself. `REPOSITORY_SPECIFIED`, `ASSUMPTION`, approximate, and demo choices MUST NOT silently enter STRICT mode.
- A previously unresolved choice may enter STRICT mode only after authoritative evidence resolves it and the corresponding ambiguity and provenance records are updated.

### APPROXIMATE mode

APPROXIMATE mode permits scientifically useful work when strict evidence is unavailable:

- Explicitly documented assumptions and evidence-limited substitutions are permitted only when the strict choice is unresolved.
- Record every assumption, substitution, and relevant hardware deviation in the experiment configuration and scientific documentation, including its source classification, rationale, expected effect, and linked ambiguity or provenance entry where applicable.
- Every output and reported result MUST be labeled `approximate reproduction`; it MUST NOT be presented as a strict reproduction.
- Approximate choices MUST NOT overwrite, mutate, or become defaults for STRICT configuration.

Use isolated configuration namespaces:

- `configs/strict/` for STRICT mode.
- `configs/approximate/` for APPROXIMATE mode.
- `configs/demo/` for demonstrations and debugging, which do not support scientific reproduction claims.

Configuration lookup MUST NOT fall back across these namespaces. In particular, STRICT mode MUST NOT inherit from approximate or demo defaults. Missing strict configuration, checkpoints, revisions, evidence, or datasets must cause an explicit, actionable error. Validate resolved configurations before starting an experiment and save the fully resolved configuration with its results.

## Reproducibility requirements

Where possible:

- Pin package versions and platform/CUDA builds.
- Pin model revisions rather than mutable `main` or `latest` references.
- Record external repository commit hashes.
- Record cryptographic checkpoint hashes.
- Record dataset manifests, split manifests, and checksums.
- Set and record random seeds across Python, NumPy, PyTorch, CUDA, workers, and augmentations.
- Record deterministic/nondeterministic backend settings.
- Log the complete experiment configuration and software/hardware metadata.
- Save scientifically relevant intermediate outputs, including masks, localized blocks, captions, recognizer outputs, similarity scores, routing decisions, and fallback outputs.

Do not rely on mutable model revisions for final experiments. Preserve enough provenance to rerun an experiment independently and determine exactly which inputs, code, dependencies, and weights produced it.

## Testing and milestone completion

Every implementation milestone requires tests proportional to its scientific and engineering risk. Before declaring a milestone complete:

1. Run relevant unit tests.
2. Run relevant smoke tests.
3. Report every failure explicitly.
4. Inspect the complete git diff.
5. Confirm that unrelated files were not modified.

Use focused tests for tensor shapes, channel compatibility, preprocessing, coordinate mappings, component-area filtering and ordering, crop limits, string normalization, score calculations, threshold boundaries, checkpoint compatibility, configuration isolation, deterministic outputs, and cross-environment interfaces as applicable.

Never suppress, catch-and-ignore, or downgrade an error merely to make a test pass. Do not silently substitute dummy outputs or alternate models after a failure.

Never silently use `strict=False` for scientifically important checkpoint loading. If non-strict loading is genuinely required, explicitly justify it, list every missing and unexpected key, test the resulting model, and document the decision and its classification.

## Evaluation and claims

Never claim that a paper result has been reproduced unless the complete corresponding official evaluation dataset and protocol have actually been run.

Keep these result classes separate in filenames, metadata, tables, and prose:

- Smoke-test result.
- Partial-dataset result.
- Approximate reproduction.
- Strict reproduction.

Store experiment results in machine-readable form in addition to human-readable summaries. Preserve per-sample outputs sufficient to audit aggregate metrics. Do not omit failed samples from a denominator without an explicit protocol rule and reported count.

The final scientific target includes reproduction or documented analysis of:

- Table 3: segmentation foreground IoU and F1.
- Table 5: recognition accuracy.
- Table 6: `alpha`/`beta`/`tau` ablation.
- Table 7: context-description ablation.

Metric implementations must state aggregation level, normalization, ignore/care handling, matching rules, thresholds, and treatment of empty predictions. If the paper leaves any of these unspecified, classify and document the decision before reporting results.

## Incremental development workflow

Work incrementally. Do not implement multiple major scientific components in one change unless explicitly requested.

Before modifying a component:

1. Inspect its existing implementation.
2. Inspect the corresponding paper section.
3. Identify paper-specified and unspecified behavior.
4. Make the smallest scientifically justified change.
5. Add appropriate tests.
6. Run the relevant tests.
7. Report assumptions, unresolved questions, and deviations.

Do not rewrite working components unnecessarily. Preserve unrelated user changes. A convenience improvement must not alter the strict scientific path unless that change is separately justified and tested.

## Git and artifact hygiene

Do not commit:

- Datasets.
- Model weights or checkpoints.
- Hugging Face caches.
- Generated masks or crops.
- Large experiment outputs.
- Secrets, credentials, or private download tokens.

Use ignore rules and lightweight manifests where appropriate, but modify ignore files only when authorized by the active task. Never perform destructive git operations unless explicitly requested. Do not automatically commit changes unless explicitly requested.

## Current repository status

The initial read-only audit established that this repository is an incomplete inference scaffold, not a complete scientific reproduction. Preserve this context during future work.

Known blockers include:

- Broken and incomplete AG-UNet implementation.
- No segmentation training pipeline.
- No dataset loaders.
- No evaluation implementation.
- PARSeq adapter is a stub and is not genuine PARSeq sequence decoding.
- DeepSolo adapter is a stub.
- BLIP-2 configuration does not uniquely match the paper.
- Configuration handling is broken.
- No trained AG-UNet checkpoint.
- No tests.
- No reproducible dependency lock.

Do not describe any of these blockers as resolved until the corresponding implementation, provenance, tests, and scientific validation are complete.
