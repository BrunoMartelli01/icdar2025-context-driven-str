# Research and comparison guide

Browse [experiment families](../experiments/README.md), [code by package](../experiments/CODE_INDEX.md),
and [offline numerical summaries](../experiments/results/README.md).
Use `python3 -m research experiments`, `experiment ID`, and `results GROUP` to
follow an experiment from its guide to implementation, configuration, tests,
and results. `compare GROUP` rechecks the original external report instead.

This is the entry point to the reproduction workspace. Original audits remain
the evidence: the index and tables are derived views, with no new inference,
training, or changes to scientific implementations.

For a new collaboration clone, start with [GETTING_STARTED.md](GETTING_STARTED.md),
[REPOSITORY_MAP.md](REPOSITORY_MAP.md), and [EXTERNAL_ASSETS.md](EXTERNAL_ASSETS.md).
The clone contains code and metadata; external reports require explicit restoration.
The complete historical inventory is preserved in
[history/WORKSPACE_AUDIT_INDEX.json](history/WORKSPACE_AUDIT_INDEX.json).

## Where to start

| Need | Source or command |
| --- | --- |
| Correct machine, directories, environments, SSH | [HOST_OPERATIONS.md](HOST_OPERATIONS.md) |
| Prepared readable comparisons | [RESULTS_OVERVIEW.md](RESULTS_OVERVIEW.md) |
| Complete list of registered comparisons | `python3 -m research groups` |
| Documents and audits within the indexed scope | [AUDIT_INDEX.md](AUDIT_INDEX.md) |
| Section positions, hashes, and original metadata | [AUDIT_INDEX.json](AUDIT_INDEX.json) |
| Refactor outcome and preservation checks | [REFACTOR_AUDIT.md](REFACTOR_AUDIT.md) |
| STRICT status and historical qualifications | [STRICT_REPRODUCTION_STATUS.md](STRICT_REPRODUCTION_STATUS.md) |
| Specification and author clarifications | [REPRODUCTION_SPEC.md](REPRODUCTION_SPEC.md), [clarifications](<Author Clarifications and Reproduction Update.md>) |
| Decisions, assumptions, and ambiguities | [PAPER_AMBIGUITIES.md](PAPER_AMBIGUITIES.md) |
| Source and weight identities | [SOURCES.yaml](../provenance/SOURCES.yaml), [ARTIFACTS.yaml](../provenance/ARTIFACTS.yaml) |
| New word-instance experiments and duplication checks | [WORD_INSTANCE_EXTENSIONS.md](WORD_INSTANCE_EXTENSIONS.md), [local novelty audit](WORD_INSTANCE_NOVELTY_AUDIT.json) |
| Small RGB/mask/probability correctors: architectures and training | [MASK_PROBABILITY_REFINER.md](MASK_PROBABILITY_REFINER.md), [local novelty audit](MASK_PROBABILITY_REFINER_NOVELTY.json) |

The opening of `STRICT_REPRODUCTION_STATUS.md`, the upstream README, and some
implementation notes describe historical milestones. Do not infer a recent
experiment's status from the date of an older paragraph alone. Follow its
specific audit, receipt, and report, preserving contradictions as such.

## Search without loading the entire history

The CLI requires only standard Python3. It does not import PyTorch, install
packages, or modify experiments. From the repository directory:

```bash
python3 -m research find 'native128'
python3 -m research section NATIVE128-GT-REVIEW-001
python3 -m research section TABLE5-SOURCE-METRIC-AUDIT-001
python3 -m research find 'DeepSolo' --limit 10
```

`section` returns the source text with its path and line number; ambiguous titles
require a more specific query or `--file`. `find` searches literally, without
interpreting queries as commands or regular expressions. Ordinary `rg` searches
exclude artifacts and environments through `.rgignore`. For an excluded audit,
use `rg --no-ignore 'PATTERN' exact/path/to/audit.json`.

## Compare results

```bash
python3 -m research groups
python3 -m research compare native128-pad0 --sort f1
python3 -m research compare native128-padoriginal6
python3 -m research compare refinements --format json
python3 -m research compare multitask-geometry --format csv
```

External sources require explicit roots. On PAPER:

```bash
export CONTEXT_STR_RUNS_ROOT=/extra1/bmartelli/context-str-runs
export CONTEXT_STR_ARTIFACTS_ROOT=/extra1/bmartelli/context-str-artifacts
python3 -m research compare kernel-fresh
python3 -m research compare research-v2-heads
python3 -m research compare yolo-test-equal
python3 -m research compare clip-encoders-budget
python3 -m research overview
python3 -m research check
```

`--runs-root` and `--artifacts-root` are also accepted **before** the subcommand.
No external path is guessed. Missing files or changed hashes produce an error
identifying the source; the command does not recreate results through inference.
`groups` works without datasets, checkpoints, or run directories.

`--format json` preserves full numerical values, protocol, references, and
pointers to original rows. CSV repeats the protocol, split, class, and source
hash for every row. Markdown rounds only the display. To save a comparison,
add `--output /path/to/new.csv` or `.json`; exclusive creation prevents
existing sources or exports from being overwritten.

## Available comparisons

| Family | Groups | Essential limitation |
| --- | --- | --- |
| Checkpoints with output128 | `native128-pad0/1/2/4/6/8`, `native128-padoriginal6` | 15 checkpoints/14 states; explored TEST300; historical bbox-area150, cap10, no ignored predictions |
| Text, Kernel, Center | `multitask-geometry` | Native instance geometry; neither OCR nor pixel F1 |
| Word research | `research-v1`, `research-v1-top10`, `research-v2-heads`, `research-v2-postprocess` | Distinct controls and budgets; epoch 50 postprocessing separate from new heads |
| Crops and coverage | `crop-links`, `frozen-crops`, `learned-crops`, `refinements`, `center-votes` | Reused TRAIN125; suite-specific ignore and coordinate policies |
| Gap and Kernel-only | `gap`, `kernel-extracted`, `kernel-fresh` | Extracting a jointly trained head and training a fresh network are different experiments |
| YOLO and routers | `yolo-test-equal`, `yolo-train-equal`, `yolo-test-transferred` | Lexical matching without IoU; descriptive budgets separate from transferred thresholds |
| Specialized CLIP encoders | `clip-encoders-budget`, `clip-encoders-threshold` | Fixed `margin_scene_en` prompt family, rather than each encoder's different winner |
| GT crops | `gt-crop-diagnostic` | Diagnostic oracle, excluded from the inference-method overview |

The initial registry contains **26 groups / 225 rows**, selected from inspected
structured reports. Not every audit is an importable numerical result: missing
sources, historical results that cannot be rerun, qualifications, failures, and
proposals remain accessible through the index and source documents. Other rows
in the source reports remain available and are not deleted.

Do not merge groups into a global ranking. Precision, recall, F1, and
`TN0=TP/(TP+FP+FN)` are derived only from counts under the declared protocol,
checking recorded rates when available. `None`/`UNRESOLVED` denotes division
without a denominator; it is not converted into success. Empty predictions
count according to the original evaluator and are not removed by the comparator.
`images` is checked only when present in the source summary; it is not invented.

VRAM columns in `multitask-geometry` are original **training** peaks; seconds in
`clip-encoders-*` concern feature export. These are not comparable end-to-end
timings. Model, precision, batch, shape, checkpoint, device, and other measurements
remain in the hash-identified source reports. No latency/FLOP benchmark is
fabricated when a common protocol is absent.

## Update without losing evidence

1. Read the specific source, audit, and paper section before changing a scientific
   component. A descriptive comparison does not authorize automatic promotion
   of a model into the pipeline.
2. Preserve reports, resolved configurations, per-sample data, and qualifications.
   Update `research/experiments.json` only after checking protocol, provenance,
   SHA256, split, denominators, budget, and row selection. Keep a comparison
   separate if any of these boundaries change.
3. Run `python3 -m research check` with the necessary roots. Every error remains
   an error; the command does not skip failed rows to produce a table.
4. Create a new overview export using `overview --output`, compare it with
   `RESULTS_OVERVIEW.md`, then deliberately update the derived view.
   `overview --check docs/RESULTS_OVERVIEW.md` verifies freshness.
5. Regenerate the index after documentation changes:

```bash
python3 -m research catalog --source-only
python3 -m research catalog --source-only --check
```

These commands build the portable source index. To index a full workspace,
omit `--source-only` and supply the same two PAPER variables above.
The catalog records selected roots, maximum depth, and exclusions; it does not
claim to index every file on every disk or in every cache. Catalog hashes verify
document freshness, not their authors' scientific claims. The index excludes
itself to avoid circular dependencies.

Reporting decision: `REPORTING-INDEX-001` in `PAPER_AMBIGUITIES.md`.
