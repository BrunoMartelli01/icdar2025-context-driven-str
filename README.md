# Context-driven scene text recognition: reproduction workspace

Collaborative scientific reproduction and exploratory research for
[*A Lightweight Context-Driven Training-Free Network for Scene Text Segmentation
and Recognition*](https://arxiv.org/abs/2503.15639v2), ICDAR 2025.
This workspace derives from the [authors’ repository](https://github.com/Ritabrata04/icdar2025-context-driven-str).
Paper authors: Ritabrata Chakraborty, Palaiahnakote Shivakumara, Umapada Pal,
and Cheng-Lin Liu. The historical introduction is preserved in
[docs/history/UPSTREAM_README.md](docs/history/UPSTREAM_README.md).

**Start here:** [experiment families](experiments/README.md) · [saved results](experiments/results/README.md)
· [code index](experiments/CODE_INDEX.md) · [setup](docs/GETTING_STARTED.md) · [code and task map](docs/REPOSITORY_MAP.md)
· [external assets](docs/EXTERNAL_ASSETS.md) · [contributing](CONTRIBUTING.md)
· [instructions for AI agents](AGENTS.md).

## What this repository contains

The paper pipeline is AG-U-Net/ResNet-50 → localized text blocks → PARSeq-S,
with BLIP-2 scene context, MPNet/lexical scoring, and DeepSolo fallback.
Code, isolated configuration namespaces, tests, dataset manifests, provenance,
and historical research reports are included. Datasets, weights, environments,
generated masks/crops, and large run outputs stay outside Git.

**STRICT reproduction remains blocked by unresolved scientific evidence.**
Implemented adapters and local qualifications do not establish the paper’s
checkpoint identities or evaluation protocol. Experiments under
`configs/approximate/` are labeled **approximate reproduction**; demos cannot
support reproduction claims. CTW1500, YOLO/CLIP routers, word-instance models,
and mask refiners are research extensions with their own protocols.

Read [the strict status](docs/STRICT_REPRODUCTION_STATUS.md),
[ambiguities](docs/PAPER_AMBIGUITIES.md), and
[the dated research synthesis](docs/RESEARCH_SYNTHESIS_20261021.md) together.
Older milestone notes describe earlier states; follow specific source/audit
evidence instead of treating a historical paragraph as current.

## Explore without installing ML dependencies

From the repository root, using Python 3.8+ and Git:

```bash
python3 -m scripts.collaboration check
python3 -m research groups
python3 -m research experiments --family word-instances
python3 -m research experiment mask-crop-three-tests
python3 -m research results native128-pad0
python3 -m research find 'Table 5' --limit 5
python3 -m research section TABLE5-SOURCE-METRIC-AUDIT-001
```

These commands neither download models nor run inference. For training/model
tests use the pinned Python 3.11 main environment in [setup](docs/GETTING_STARTED.md).
DeepSolo requires a separate legacy environment. The root `requirements.txt`
is historical upstream material; it is not a complete environment specification.

## Find evidence and reproduce work

| Task | Entry point |
| --- | --- |
| Understand modules and select a bounded task | [Repository map](docs/REPOSITORY_MAP.md) |
| Obtain datasets, paper, checkpoints, caches or outputs | [External assets](docs/EXTERNAL_ASSETS.md), [machine-readable inventory](provenance/EXTERNAL_ASSETS.json) |
| Inspect experiments and limitations | [Research guide](docs/README.md), [results overview](docs/RESULTS_OVERVIEW.md) |
| Find source evidence | [Audit index](docs/AUDIT_INDEX.md), `python3 -m research find QUERY` |
| Run scientific code | [Setup](docs/GETTING_STARTED.md), component documentation |
| Work on the PAPER machine | [Host operations](docs/HOST_OPERATIONS.md) |
| Propose changes or report issues | [Contribution guide](CONTRIBUTING.md), GitHub templates |

The clone includes small numerical summaries in Markdown, JSON, and CSV.
`research results GROUP` verifies and reads these summaries without model weights
or run directories. `research compare/check/overview` verifies original reports
and requires their exact sources. The summary does not replace per-sample evidence.
Read the [English/evidence policy](docs/LANGUAGE_POLICY.md) and the
[refactor audit](docs/ENGLISH_REFACTOR.md). No historical external archive is
claimed to be publicly available.

## Attribution and reuse

The upstream README states MIT, but no upstream `LICENSE` file was present in
the inspected workspace. A license grant for this combined workspace has not
been established here; no new license is inferred from a badge. Resolve this
with the owners before redistribution under a specific license. Dataset,
checkpoint and third-party terms remain separate; see
[access and licenses](docs/EXTERNAL_ASSETS.md#access-and-licenses).
