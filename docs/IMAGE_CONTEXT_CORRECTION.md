# Image-wide OCR correction battery

**Approximate reproduction / exploratory transcription research.** The user
authorized multiple tests per failure category, including a method that uses
the complete original OCR transcript and all other detected words to decide
whether and where to divide a reading. This changes downstream transcription;
U-Net, RGB boxes, PARSeq and STRICT/default routing remain unchanged.

**Completed and audited, 2026-10-27.** Read the
[detailed results and failure analysis](IMAGE_CONTEXT_CORRECTION_RESULTS.md).
Portable numerical evidence and qualification are preserved in
[`provenance/IMAGE_CONTEXT_CORRECTION_V1.json`](../provenance/IMAGE_CONTEXT_CORRECTION_V1.json).

Before implementation, the authoritative `papers/2503.15639v2.pdf` sections
3.3–3.5, frozen ByT5 reconstruction, copied-character head, contextual boundary
selection and their failure diagnostics were inspected. The paper's MPNet/fuzzy
caption routing does not specify this corrector. The original CoNLL2021
LSTM/BiLSTM method is not implemented or reproduced by this battery.

Implementation: `recognition/image_context_correction.py`,
`scripts/run_image_context_correction.py`. Configuration:
`configs/approximate/image_context_correction_v1.json`. Tests:
`tests/recognition/test_image_context_correction.py`.
Independent replay: `scripts/audit_image_context_correction.py`. Detailed
post-hoc diagnosis: `scripts/analyze_image_context_correction.py`.
Context-specificity and named-case diagnosis:
`scripts/diagnose_image_context_failures.py`. The supplementary exact legacy
feature baseline is replayed by `scripts/run_legacy_global_context.py`, using
`configs/approximate/image_context_legacy_features_v1.json`.
Decision: IMAGE-CONTEXT-CORRECTION-001 in `docs/PAPER_AMBIGUITIES.md`.

## New method and controlled context tests

Image-wide Counterfactual Boundary Selection (ICBS) is a descriptive experimental
name, not an established-method or novelty claim. Original readings are the KEEP
reference. Every proposal is scored against that same original under:

1. Target-only BERT PLL.
2. Original horizontal OCR neighbours.
3. All other original OCR observations in their inherited reading order.
4. All other original OCR observations as a second-sequence evidence inventory.
5. Local, ordered and inventory evidence together.
6. An explicitly incorrect-image context control, cycling within the same role.

The target is removed from its own context, as are all parents of a fragment
merge. Every remaining token enters a window; none is silently truncated.
Long observations can span multiple windows. Context windows are bounded by the
largest candidate's token requirement and the declared BERT limit. Pool target
PLL changes with mean/min/max/variance, confidence-weighted mean and the fraction
of positive windows. These are counterfactual language features, not calibrated
word-correctness probabilities. Original context is frozen: corrected neighbours
never feed later corrections. Window pooling is not one joint forward pass over
an arbitrarily long transcript. Inventory order is stable, not permutation invariant.

Use the same boundary pool and HGB capacity for the six context comparisons.
Additional global-context controls use class-balanced HGB, conservative aligned
training labels, and a trained listwise MLP with explicit zero-logit KEEP.
Listwise softmax targets distribute mass over beneficial alternatives, or KEEP
when none is beneficial. Different groups can still conflict at inference;
greedy parent consumption makes that resolution explicit.

## Training data, characters and fragments

Three matched one-epoch/6000-record ByT5-small recipes start from the same pinned
pretrained weights, use the same optimizer/update budget and image-separated
three-fold OOF predictions, plus a final model on all FIT images:

- Historical corruption recipe, with fold-only lexical priors and operations.
- High-coverage real OCR/GT pairs plus random synthetic character noise.
- The same alignment policy plus TRAIN-derived empirical character confusions.

Clean alignment requires covered care words, rejects partially covered neighbours
and incompatible multiline groups, and removes conflicting source/target pairs.
Those are assumptions, not perfect manually verified labels. The empirical
recipe is not a reproduction of ACL2024 glyph-similarity generation. Matched
budgets compare whole recipes; sampled record identities and corruption draws
need not be identical. Preserve pairs, labels, conflicts, confusion tables,
training IDs, held-out IDs, checkpoints and loading/memory receipts.

Generate bounded beam alternatives on original single observations and horizontal
two-observation groups. Preserve raw outputs, EOS completion and validation
decisions. Bounded character edits can substitute/delete/insert letters; original
empty/numeric/punctuation observations stay available. Character and fragment
pool ablations compare target-only and global-context selection. Expanded-pool
and aligned-label controls are separate. Raw single-observation generator arms
measure the whole training recipe without a learned selector.

Boundary-pool membership here means equal compact Unicode-casefold strings;
case normalization can differ from the historical strict character-copy control.
Those controls remain separate. OOF models see fewer FIT images than final
models despite matched record/update budgets; this distribution difference is
explicit, rather than an independent-generalization claim.

## False positives and evaluation

Three separate filters compare exact-read probability, exact-read plus care-GT
coverage proxy, and overlapping lower-score duplicate rejection. Coverage is
annotation supervision, not physical text-presence truth. Zero-threshold KEEP
is available. Report deletions, false positives and correct words lost together.
Filtering and correction are isolated comparisons; their gains are not added.

A separately predeclared supplementary composition compares filter-first with
correct-first/protect-changed-parent policies, for both the boundary and expanded
correctors and exact/presence-aware filters. Joint CAL-only threshold selection
uses the same explicit grids, objectives and damage caps. This larger search can
overfit CAL. It is not an independent validation or an assumed sum of gains.
Code: `recognition/image_context_joint_policy.py`,
`scripts/run_image_context_joint_policy.py`; configuration:
`configs/approximate/image_context_joint_policy_v1.json`. Outputs remain under
the separate `joint/` directory with per-image predictions and count replay.

Reuse 1030 FIT /100 calibration /125 evaluation /300 TEST. ByT5 OOF generation
and lexical priors exclude the predicted image's fold. Fit selector labels only
on FIT. Freeze all thresholds and filter choices using calibration before reading
evaluation/TEST transcripts in this run. Compare correct-count-first and F1-first
calibration under damage caps0/2; these caps do not guarantee TEST safety.

Primary metric is per-image micro casefold whitespace-token multiset, retaining
punctuation, every care GT, duplicate and empty observations, without IoU or
ignore-based prediction neutralization. It is transcription, not spatial spotting.
Every arm operates offline on saved PARSeq observations. DeepSolo usage in this
correction-only experiment is0/300 TEST images (0.00%), not an inferred paper
fallback rate or an online routing/latency measurement; reports retain counts
and percentages for each evaluated role.
Historical repeated exploration, source calibration exposure and known identical
TEST578/TRAIN212 and TEST656/TRAIN81 prevent independent-validation claims.
Report paired image bootstrap intervals conditional on fixed models, not full
training uncertainty, and a duplicate-excluded sensitivity.

## Execution and qualification

External root:
`/extra1/bmartelli/context-str-runs/image-context-correction-battery-20261025-v1`.
The current exact-budget execution is `run_v2/full/` under that root; the
original `full/` is preserved as stopped/invalid for matched-recipe comparison.

The first full run exposed empty synthetic strings discarded after filling the
record budget (e.g.5987/5991 instead of6000). It stopped before final selection
or evaluation. Replenishment now rejects empty strings before counting, and
an unmet exact budget raises an explicit error. Regression coverage includes
both noise recipes, successful replenishment and impossible-budget failure.
The first shortfall test expected a budget-specific error but received the older
empty-set error; validation order was corrected, retaining both explicit failures.
All105 focused/regression tests pass after the correction.

`scripts/reuse_image_context_generators.py` verifies unchanged training/inference
function ASTs, identical configuration/data/runtime, exact regenerated historical
training records and cryptographic checkpoint/prediction copies before reusing
the four unaffected historical controls. The two affected recipes are trained
anew. Preserve `run_v2/full/generator_reuse.json` and
`run_v2/exact_budget_preflight.json`: all eight new training datasets have exactly
6000records/750updates. Completed model stages must also match those data hashes.

Earlier GPU smokes1/2 stopped at occupied-memory preflight, without model OOM.
Smoke3 completed42images and passed the independent1044-image/arm count audit.
The unchanged training/inference engine also passes strict checkpoint reload,
with all172 saved tensors bit-exact. These checks establish execution validity,
not correction efficacy; read the current full-run completion and audit.
Use an immutable source snapshot, explicit paths, and the existing environment:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 /explicit/python -u \
  -m scripts.run_image_context_correction --mode approximate \
  --config configs/approximate/image_context_correction_v1.json \
  --output /explicit/new/output --stage all --smoke
```

Remove `--smoke` for the complete battery. Stages are prepare, generators, pool,
language, features, selectors, filters and report. Resume requires unchanged
configuration, source and sealed completed artifacts. Incomplete generator stages
require a fresh output; errors and OOM propagate with a failure receipt. No model
or package download is performed. CUDA FP32, no AMP/TF32, sequential model loading,
actual GPU, batches/shapes, allocated/reserved peaks and runtime are recorded.
RTX3090 execution does not qualify RTX3070/8GiB. The completed primary audit
passes 45675 image/arm count checks, the joint audit passes 16800 image/policy
checks, and the legacy control reproduces 525 earlier predictions exactly.

## Completed findings

All profiles below use saved PARSeq output on the same 300 TEST images; the
metric is the lexical transcription protocol defined above. Recovered/lost
counts refer to exact matched care words relative to original OCR.

| Frozen profile | F1 | Recovered / lost | False positives | DeepSolo images |
| --- | ---: | ---: | ---: | --- |
| Original OCR | 52.5855% | 0 / 0 | 1502 | 0/300 (0.00%) |
| Primary global-context boundary correction | 53.3861% | 26 / 0 | 1498 | 0/300 (0.00%) |
| Fragment + boundary proposals with context | 53.9933% | 47 / 0 | 1498 | 0/300 (0.00%) |
| Expanded proposals with context | 53.8522% | 49 / 1 | 1514 | 0/300 (0.00%) |
| Exact-read FP filter | 60.5834% | 0 / 24 | 783 | 0/300 (0.00%) |
| Care-coverage-aware FP filter | 60.3539% | 0 / 9 | 834 | 0/300 (0.00%) |
| Primary joint supplement | 60.4320% | 2 / 9 | 833 | 0/300 (0.00%) |
| Expanded + exact-read filter | 61.2986% | 19 / 24 | 776 | 0/300 (0.00%) |
| Expanded + care-coverage-aware filter | 61.0589% | 19 / 9 | 827 | 0/300 (0.00%) |

Global minus same-pool word-only is +0.3516 F1 points, conditional 95% interval
[-0.0120,+0.7546]. Matched relevant inventory minus wrong-image inventory is
-0.2978 points, interval [-0.7817,+0.1044]. Thus the whole-image implementation
does not establish a benefit from image-specific context. Adding context to
the exact previous word-LM features gives a small positive signal (+0.3269
points, interval [+0.0162,+0.6850]), without isolating semantic relevance.
All are explored-cohort, single-seed conditional comparisons.

The expanded pool has an exact GT-assisted maximum of 1474 correct words,
versus 1322 originally. Even with free removal of every unmatched word, its
optimistic F1 bound is 80.1523%; at that maximum, reaching 80% allows at most
7 remaining FP. This bound applies only to the current proposals. Expand
useful textual candidates, improve acceptance supervision and validate on
independent images before expecting a large step from threshold tuning.

The exact-read filter's F1 gain includes losing 24 correct words; annotation
coverage protection reduces that to 9. Filtering may remove wrongly read real
text as well as background/ignored/partial detections. CAL damage caps are not
TEST guarantees. Character-level reading-order diagnostics also fail to improve
when expanded correction is added to either filter. Full tradeoffs, matched
training mixtures, named failures, duplicate sensitivity, runtime/memory,
preserved execution failures and next experiments are in the linked report.
