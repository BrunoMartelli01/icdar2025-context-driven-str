# Grounded context inventory and local verification router

2026-10-23. User-authorized **approximate reproduction extension**, not strict
paper reproduction or Table 5 recognition accuracy. The user asked to check
novelty against existing work and implement the genuinely different signals.
Paper section 3.5, Figure 4 and Algorithm 2 were inspected before implementation.

## Novelty check

| Proposal | Existing evidence | Decision and new boundary |
| --- | --- | --- |
| Add captions, neighbors or generic geometry | `docs/YOLO_CONTEXT_QUALITY_ROUTER.md`, `docs/OCR_EVIDENCE_GENERATION_BATTERY_TRAIN_V1.md`, `docs/GEOMETRY_CONTEXT_RECONSTRUCTION_TRAIN_V1.md`, `docs/PARSEQ_VISUAL_MARGIN_CONTEXT_GATE_TRAIN_V1.md` already investigate these | Not a new method on its own; retain existing controls |
| Mask the target and compare CLIP scores | `docs/CLIP_INNOVATIVE_VERIFICATION.md` already tests target occlusion, counterfactual scores and word foils | Not claimed as new; the new verifier uses a generative visual model on local line crops, with target-erased and blank controls |
| Ask BLIP2 whether residual text exists | `docs/YOLO_RESIDUAL_TEXT_VQA.md` already tests original/masked/blank whole-image Yes/No with captions | Not repeated; the new task asks about one exact predicted spelling and a foil, preserves neighboring line pixels, and measures word-specific visual support |
| Independent region inventory | Florence OCR was inspected in small caption diagnostics; detector slots and U-Net residual proposals already exist | New routing evidence: frozen Florence `<OCR_WITH_REGION>` plus `<DENSE_REGION_CAPTION>` from original RGB only, without YOLO words or boxes; compare region coverage and associated readings afterward |
| Rich visual input to supervised routing | `docs/YOLO_TRAINABLE_ROUTER.md` uses 46/90 scalar features | New ablation: frozen original-image Florence visual representations enter the small router directly, versus core scalars alone |

The novelty is relative to this repository's implemented router inputs and
controls, not a claim of a novel general algorithm. Machine-readable decisions
and source references: `docs/GROUNDED_CONTEXT_ROUTER_NOVELTY.json`.

## Explicit experiment

- Reuse the exact 125 TRAIN / 300 TEST image branches, historical TRAIN80 fit
  and VAL45 partition, 983/2204 care words and complete DeepSolo ViTAEv2-S+TextOCR
  outputs. Neither YOLO/PARSeq predictions nor fallback outputs are changed.
- Independent inventory uses the already local, immutable
  Florence-2-large-ft revision `4a12a2b54b7016a48a22037fbd62da90cd566f2a`.
  This is an explicitly chosen model, not Qwen3-VL or a claim to reproduce it.
  No downloads, package installations or visual-model training are needed.
- Original image and task token only. OCR quadrilaterals, dense region labels,
  raw generations, token IDs and incompleteness flags are preserved. Both a
  missing EOS and reaching the token cap indicate incomplete/ambiguous output:
  the decoder can force an EOS at the cap. Mean original
  image visual tokens are L2 normalized; their 1024 coordinates are available
  to visual ablations. Independent predictions are not ground truth.
- Native-pixel quadrilateral coverage permits multiple YOLO words inside one
  independent line. Character edit agreement uses spatially associated raw
  readings with Unicode casefold and whitespace normalization only. Punctuation,
  digits, repetitions and invalid/no-area independent predictions are explicit.
  Text-bearing object labels generate risks, not inferred missing-word truth.
- At most one suspicious nonempty predicted word per image is checked, selected
  by frozen visual quality and disagreement with the independent inventory.
  Neighbors are bounded by explicit geometry and a five-word cap. Query the
  pinned frozen BLIP2-FLAN-T5-XL about the exact spelling and a deterministic
  reversed/changed foil on original local, target-erased and uniform blank views.
  Never append a caption or use GT to form a candidate. Preserve every prompt,
  Yes/No teacher-forced sum log probability including EOS, normalized support,
  pixel hash and number of unverified words. These are not calibrated F1 scores.
- Five feature sets: core46, inventory61, verified75, visual1070, combined1099.
  Ridge and small ExtraTrees regress delta TP and delta prediction count, with
  the fitting-only all-DeepSolo F1 reference. Hyperparameters use TRAIN80 inner
  five-fold CV, with fold-only scalers/labels/reference. VAL45 selects families
  and scalar cuts. No refit125 and no TEST-based promotion.
- Freeze all choices before new TEST visual inference. TEST probes and vectors
  bind the already-existing choices hash. Compare exact 20/30/50% image bypass
  budgets and transferred scalar thresholds, retaining all candidates and old
  controls. Report accepted quality, final TP/FP/FN, F1, TN0 and same-budget oracle.

Metric: within-image exact casefold word multisets, punctuation retained,
evaluation-only native256 ignore, all care words retained, no spatial IoU.
F1 aggregates global TP/FP/FN; TN0 is TP/(TP+FP+FN). The mapping to the paper's
recognition accuracy remains unresolved. TRAIN125 is a historical holdout,
not official full TRAIN1255; DeepSolo was fine-tuned on Total-Text TRAIN and
TEST has already been repeatedly explored. All results remain exploratory.

## Runtime and execution

Actual PAPER hardware is RTX3090, not the historical RTX3070. Models load
sequentially, FP32, vision batch1 and Yes/No language batch2; every stage records input shapes, model state hashes,
runtime, peak allocated/reserved CUDA bytes and loading compatibility. Staged
BLIP2 logits are checked against native forward. Errors/OOM propagate with a
preserved failure receipt, without model/precision/device substitution. Incremental
probe time excludes cached upstream inference and DeepSolo: bypass fraction alone
does not establish acceleration. Whole-image Florence processing has its native
preprocessing; a word crop only becomes larger inside the local verifier.

```bash
PYTHONPATH=artifacts/diagnostics/lavis-captioner-candidates-20260914/runtime-deps \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venvs/context-str-main-py311/bin/python -u -m scripts.run_grounded_context_router \
  --mode approximate \
  --config configs/approximate/grounded_context_router_v1.json \
  --paths configs/approximate/grounded_context_router_local_paths_v1.json
```

The existing isolated einops target is reused without installation. Add
`--preflight` for integrity checks or `--smoke` for separate20-image qualification.
Audit completed outputs with:

```bash
.venvs/context-str-main-py311/bin/python -m scripts.analyze_grounded_context_router \
  --run /explicit/completed/run
```

Implementation: `recognition/grounded_context_router.py`,
`recognition/grounded_context_runtime.py`, `scripts/run_grounded_context_router.py`,
`scripts/analyze_grounded_context_router.py`.
Tests: `tests/recognition/test_grounded_context_router.py`.
Scientific classifications: GROUNDED-CONTEXT-ROUTER-001 in `docs/PAPER_AMBIGUITIES.md`.

## Sources and limitations

[Florence-2](https://arxiv.org/abs/2311.06242) and its
[official model card](https://huggingface.co/microsoft/Florence-2-large-ft)
support the independent task/region interfaces.
[TIFA](https://arxiv.org/abs/2303.11897) motivates decomposed visual questions;
our word-verification/routing experiment is not its benchmark reproduction.
Neither source establishes that these signals improve our router.

Independent inventories can miss text or hallucinate it. Object boxes include
background; geometry approximates curved reading order; target masking may erase
neighbor pixels; a one-word check leaves other words unchecked; model likelihood
may reflect language priors. Eighty fit images may be insufficient for raw visual
features. These are measured hypotheses, not perfect-selection guarantees.

## Verification status

Initial focused tests:30 PASS across the new geometry/control tests and existing
trained-router/residual-VQA tests. Initial smoke `7960a04d3361` failed before
saving any per-image inventory: Florence's outer generation config has EOS=None
while generation delegates to its language decoder. The explicit decoder EOS
contract now uses the actual delegated generation config. Failure evidence is
preserved; no output or runtime error is ignored. Completed run evidence is
reported below only after full inference and independent audit.

Smoke20 `1c6f60b1be54` completed the original neural/fit/evaluation boundary.
Inspection found a pinned processor issue: the generated decoder-start EOS
was attached to the first OCR label. Preserve full raw output and remove only
that structural token position before the official parser. Thirty real generated
task outputs pass the processor regression; ordinary OCR punctuation is retained.

Full attempt `748ea7bb5931` completed125 qualified TRAIN inventories, then was
deliberately interrupted before fitting or TEST inference to correct forced-EOS
incompleteness accounting. Its notice and outputs remain immutable. The final
protocol may reuse this explicit TRAIN source with `--reuse-train-inventory`:
neural runtime bytes, model/config/generation contracts, RGB population, complete
frozen-state receipt and every raw output hash must match. Rebinding records
preserves original fields and hashes the source; no TEST probes are imported.
The final independent audit rechecks all reused bytes and token parsing.

## Completed result: no improvement

Final full run:
`/extra1/bmartelli/context-str-artifacts/grounded-context-router-v1/full/f9ac718a9aa5`.
Plan SHA256:
`bed354abf739b5a835238e11db47efdeaa0cca3fe75c529edf4e826380c66dd8`.
Results SHA256:
`4cc9986599bc23c4f7a8cd621af4ab040a021b8bf1637d3ce64fe186b55ab646`.

The methods are distinct extensions of prior inputs, but **none improves the
strongest previous same-budget controls in this experiment**. VAL45 selects
`trees_core` at every quota, excluding all new evidence. Do not promote a
TEST-favorable alternative or replace the old router on these measurements.

| TEST images bypassing DeepSolo | VAL-selected family | TRAIN125 final F1 | TEST300 final F1 | Previous strongest same-budget control TEST F1 | Eligible oracle TEST F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 60 /300 (20%) | trees_core | 78.440% | 80.018% | 80.174% | 82.875% |
| 90 /300 (30%) | trees_core | 77.460% | 79.624% | 79.835% | 82.914% |
| 150 /300 (50%) | trees_core | 75.319% | 78.387% | 78.473% | 82.320% |

Previous references are historical exploratory controls, not new TEST selection:
base_proxy_f1 / risk_visual_consensus / risk_visual_mass, respectively.
All-DeepSolo F1 stays78.624% TRAIN125 /80.357% TEST300. Fit80, VAL45 and
TRAIN125 scores remain separately exported; TRAIN125 includes fitted images.

Matched ExtraTrees feature ablations, with the same explicit grids/CV protocol:

| Input | TEST F1 at20% bypass | TEST F1 at30% bypass | TEST F1 at50% bypass |
| --- | ---: | ---: | ---: |
| Previous core scalar inputs | 80.018% | 79.624% | 78.387% |
| Core + independent inventory | 78.945% | 78.770% | 77.675% |
| Core + inventory + local verification | 79.063% | 78.392% | 77.343% |
| Core + frozen visual coordinates | 78.878% | 77.860% | 76.514% |
| Core + all new evidence | 77.925% | 77.236% | 76.159% |

All Ridge variants are also preserved in results/CSV and do not beat the
corresponding previous references. These are point estimates; fixed-gate paired
bootstrap intervals are exploratory and uncorrected for multiple comparisons.

## Diagnostic evidence, not new selector tuning

`scripts/report_grounded_context_router.py` requires the PASS audit and writes
`signal_diagnostics.json` plus PNG/PDF curves. These diagnostics use all images,
including ineligible/zero-filled scalar inputs; they do not fit/select a model
or search TEST thresholds/directions. AUROC for image F1>=.8 is diagnostic,
not the final micro-F1 gain objective.

- TEST independent reading-agreement AUROC73.68%, versus80.48% for the previous
  base_proxy_f1 scalar under this all-image diagnostic. Missing-region risk is
  68.97%; the local counterfactual margin is58.60%. TRAIN agreement AUROC81.83%
  falls on TEST; small data and distribution change are plausible explanations,
  not experimentally isolated causes.
- Local exact-word median Yes support is0.633 TEST, versus0.502 for the foil,
  0.528 after target erasure and0.565 on blank inputs. The additional visual
  signal remains weak and influenced by language priors; none is a calibrated
  correctness probability.
- OCR generation reaches the cap or otherwise remains incomplete on3/125 TRAIN
  and6/300 TEST images; dense-region generation is incomplete on1TEST image.
  One TEST inventory is empty. Preserve these images and all care denominators.
- A total412 words get local probes (121TRAIN /291TEST), at most one per image.
  Every other predicted word remains unchecked by this additional verifier.

## Final qualification

Independent audit PASS:425 vectors,2174 independent OCR regions,412 local probes,
300 routing points,4800 held-out predictions and115 independent cached model
refits. The audit verifies immutable raw reuse, official token parsing, original/
erased/blank RGB hashes, answer-score normalization, model fit/validation scopes,
CV/family/threshold choices and complete routing metrics. All47 focused tests
PASS, including injected OOM, freeze-before-TEST and cache-drift rejection.

Florence native vision shape1x3x768x768, FP32 vision batch1; peak allocated
4,223,270,400 bytes and reserved4,605,345,792 bytes across recorded stages.
BLIP2 local views use224x224, FP32 vision batch1 /language batch2; peak allocated
15,955,174,912 bytes, reserved16,091,447,296 bytes. This is RTX3090-qualified,
not an8GB qualification. State hashes stay unchanged; no real OOM occurred.

Measured stage seconds: TRAIN inventory334.77 (reused), TRAIN verification77.09,
TEST inventory1038.25, TEST verification223.18. Stage timers exclude initial model
loading/source-integrity work; local probe records also preserve per-word times.
Extra evidence adds substantial computation and establishes no acceleration.

Initial plot export in the main environment failed because matplotlib is absent.
The existing YOLO analysis environment supplies it without installation or neural
reruns; preserve `report_attempt_main_environment_failure.json` and
`report_runtime.json`. Original metrics and independent audit remain unchanged.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venvs/context-str-yolo-py311/bin/python -m scripts.report_grounded_context_router \
  --run /extra1/bmartelli/context-str-artifacts/grounded-context-router-v1/full/f9ac718a9aa5
```

The completed run includes `report.md`, `comparison.csv`, full per-family/threshold
metrics, per-image probes/feature vectors, portable trained models, CV traces,
fixed choices, paired intervals and the standalone scientific figures.
