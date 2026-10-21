# Repository map for people and AI agents

Read [AGENTS.md](../AGENTS.md) before edits. Paths below are repository-relative;
historical `/home/bruno`, `/extra1/bmartelli`, `/tmp`, and `/var/tmp` paths in
audits identify past runs, not portable defaults.

## Code by responsibility

| Directory | Responsibility / useful entry points |
| --- | --- |
| `models/` | AG-U-Net and exploratory architectures; `ag_unet_approx.py`, `unet_adv.py` |
| `segmentation/` | Datasets, transforms, training, pixel metrics, word targets; `train.py`, `config.py`, `data.py`, `smoke.py` |
| `localize/` | Mask-to-block Algorithm 1, component filtering/ranking, crop extraction; `mask_to_blocks.py` |
| `recognition/` | Genuine PARSeq adapter (`parseq.py`), annotation parsers, evaluation, research routers |
| `caption/` | BLIP-2 adapter and caption cache |
| `scoring/` | MPNet/lexical similarity, explicit score aggregation and research routing |
| `deepsolo/` | Versioned JSON boundary, candidates, source/checkpoint validation, isolated worker |
| `pipeline/`, `cli/` | Mode validation and orchestration; `pipeline/config.py`, `pipeline/approximate_e2e.py` |
| `diagnostics/` | Exploratory analyses with experiment-specific metrics and assumptions |
| `research/` | Standard-library evidence navigation; `experiments.json` pins saved report identities |
| `experiments/` | Family guides, a package-level code index, the experiment catalog, and small verified result summaries |
| `scripts/` | Executable modules and historical host launchers; `collaboration.py` checks/exports source |
| `configs/strict/` | Evidence-gated specifications; blockers must fail explicitly |
| `configs/approximate/` | Labeled assumptions, candidate artifacts and research experiments |
| `configs/demo/` | Demonstrations/debugging, without scientific claims |
| `environments/`, `docker/deepsolo/` | Version snapshots and isolated runtime reconstruction |
| `manifests/`, `provenance/` | Small split/checksum/source records; data and model bytes stay external |
| `tests/` | Component tests, mocks and small synthetic localization fixtures |
| `docs/` | Guides, source evidence, dated reports, decisions, indexes |
| `Figures/` | Original upstream illustrations, with authorship retained |
| `papers/` | Retrieval instructions for the authoritative PDF |

## Choose a starting point

| Work | Read first | Relevant tests |
| --- | --- | --- |
| Navigation, source export | `docs/README.md`, `research/`, `scripts/collaboration.py` | `tests/research/`, `tests/scripts/test_collaboration.py` |
| Segmentation/training | `docs/AG_UNET_IMPLEMENTATION_DECISIONS.md`, paper architecture/training sections | `tests/models/`, `tests/segmentation/`, `tests/data/` |
| Localization | Paper Algorithm 1, `localize/mask_to_blocks.py`, `docs/REPOSITORY_CONTOURS_VARIANT.md` | `tests/localize/` |
| PARSeq/context | `docs/RECOGNITION_CONTEXT_IMPLEMENTATION_DECISIONS.md`, `provenance/ARTIFACTS.yaml` | `tests/recognition/test_parseq_adapter.py`, `tests/scoring/test_context_foundation.py` |
| DeepSolo | `docs/DEEPSOLO_RUNTIME.md`, M6 manifest, DeepSolo entries in strict status | `tests/deepsolo/` (ordinary tests use mocks) |
| New dataset | `docs/SCENE_TEXT_DATASETS.md`, `configs/approximate/datasets/scene_text_sources_v1.yaml` | `tests/recognition/test_scene_text_datasets.py`, `tests/segmentation/test_scene_text_preparation.py` |
| Word instances/mask refinement | `docs/RESEARCH_SYNTHESIS_20261021.md`, the specific component guide | Matching `tests/segmentation/` / `tests/models/` modules |

Use `python3 -m research find QUERY` and `section AMBIGUITY_ID` for targeted
reads. Do not replace the comprehensive ambiguity log with a summary or infer
current evidence from the initial audit alone.

## Scientific boundaries

The main reproduction target remains Tables 3, 5, 6 and 7. Recognition accuracy,
matching rules and several checkpoint identities remain unresolved. End-to-end
word F1, geometry-only recall, pixel F1, and lexical multiset scores are distinct
quantities. Read the group's protocol and denominator before comparing numbers.
Explored TEST populations and reused TRAIN holdouts are marked in the reports.

Historical `.sh` launchers often encode PAPER paths. On another host prefer the
underlying `python -m ... --help` module and explicitly supplied paths.

## Source-export boundary

`python3 -m scripts.collaboration export --destination /absolute/new/directory`
creates a source-only copy without staging or changing the original Git tree.
It copies Git-visible files from documented source directories, validates them,
regenerates the destination's local audit index, and records per-file SHA256
plus the original Git base in `COLLABORATION_EXPORT.json`.
The original inventory is preserved under `docs/history/`.
The export can start its own Git history; upstream attribution and base identity
remain recorded without claiming an official author release.
It includes the workspace's existing uncommitted research code. A focused
onboarding test run does not qualify every included experimental component.

The package layout and Python entry points remain stable so historical commands
and source snapshots stay meaningful. The refactor adds a navigable experiment
layer instead of moving scientific modules between packages. Use
`python3 -m research experiment ID` to find one experiment's exact code,
configuration and tests; `experiments/CODE_INDEX.md` covers all Python modules.
