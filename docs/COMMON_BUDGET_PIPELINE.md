# Shared budgeted OCR pipeline for Total-Text and ICDAR2015

**Approximate reproduction / research extension.** The user authorizes new
training and requires one coherent procedure, with independent learned weights,
that preserves or improves four qualified full-TEST results. Dataset-optimized
DeepSolo variants are explicitly allowed. Preserve exact logical fallback
budgets, all images and all care words:

| Dataset | Version | Baseline lexical micro F1 % | DeepSolo images/total (%) |
| --- | --- | ---: | --- |
| Total-Text | U-Net |79.9904145699|150/300 (50.00%)|
| Total-Text | YOLO |80.9557605867|150/300 (50.00%)|
| ICDAR2015 | U-Net |78.9333333333|228/500 (45.60%)|
| ICDAR2015 | YOLO |79.9489795918|228/500 (45.60%)|

Paper3.2/3.3/3.5 and existing source implementations were inspected before
scientific changes. This trained utility router is an extension of the paper's
fallback idea; it is not Algorithm2's training-free MPNet/fuzzy gate. The metric
is per-image exact multiset matching after Unicode casefold/isalnum, then micro
TP/FP/FN; no spatial match, all care/empty denominators retained, native256
ignore/care handling inherited from each qualified source. All results must be
labelled approximate reproduction. Historical TEST exposure remains explicit.

## Shared procedure declared before evaluation

1. Keep qualified PARSeq/light predictions and optimized DeepSolo outputs as
   authenticated expert inputs. Use the same inference record schema, correction
   implementation, feature functions, learning code and budget rule in all four
   cases. No predictor branches on dataset identity.
2. Choose from the same prediction-only lexical correction grid on selection.
   Absence of a vocabulary disables editing; ICDAR2015 uses the existing
   benchmark Weak TRAIN/TEST resources. The no-edit option is always eligible.
   Weak benchmark side information is not a general-image vocabulary guarantee.
3. Extract common raw/post OCR, native token, lexical-edit, box-geometry and
   text/optional-kernel coverage descriptors. A missing kernel is explicitly
   marked and its six coordinates set to zero, not interpreted as a real mask.
   YOLO's map group requires the corresponding U-Net's inference too.
4. Fit word correctness proxies using FIT-only labels and five RGB-grouped
   folds. FIT images receive word-risk predictions from other image folds;
   selection/CAL/TEST use a full-FIT model. All upstream light experts remain
   fixed and do not become OOF through this operation.
5. Train normalized light-minus-fallback TP and prediction-count regressors.
   Every case uses the identical Ridge/HGB/ExtraTrees recipes, feature groups,
   reference grid, nonempty-guard alternatives and top3/5/10 model ensembles.
   Select on selection micro F1 at the scaled user budget. Within0.1 F1 points
   of its best, prefer fewer stored scalars, then higher selection F1. No
   TEST score or label selects a method. No refit follows selection.
6. Rank scores by the same predicted count utility, then RGB hash and sample
   ID. Keep exactly N-K light images. K remains dataset-specific by the user's
   instruction. Empty outputs can be scored; the guard is validation-selected.
   An impossible guard/budget combination fails rather than changing K.

Total-Text uses1128 FIT images (original1028 FIT plus100 previously calibrated
TRAIN),62 selection and63 CAL from the125 expert-gradient-heldout images.
ICDAR2015 uses900 FIT/50 selection/50 CAL. These are both approximately90/5/5
TRAIN partitions. RGB identities must be disjoint across roles and TEST.
The CAL split is reserved for independent diagnostics, not checkpoint or
feature selection in this first exact-budget study. Baselines used different
fitting/selection histories; comparison is at the output/cost boundary, not an
isolated causal effect of the learning objective.

## Boundaries and execution

The initial shared procedure unifies correction/risk/routing around existing
experts. Its light experts still retain qualified learned architectures,
resolutions and upstream corrections: TT U-Net Text+Kernel/GNN/verifier/editor,
IC15 long768 native U-Net, and frozen YOLO640/PARSeq. Changing these experts is
a separate stage if scientifically justified; this first study does not claim
their underlying training protocols are identical. Dataset-specific DeepSolo
checkpoints, resizes and postprocessing remain as authorized.

Entry points: `recognition/common_budget_pipeline.py`,
`scripts/run_common_budget_pipeline.py`; explicit profile:
`configs/approximate/common_budget_pipeline_v1.json`.
External root:
`/extra1/bmartelli/context-str-runs/common-budget-pipeline-20261029-v1`.
Resolved configuration, executed source, per-image observations/maps, separate
TRAIN/TEST labels, fresh models, all selection trials and failures are saved.
Only existing environments are used. New work initially runs on CPU over
authenticated caches; zero new neural or DeepSolo inference calls are expected.
Logical routing is not online latency. Any later GPU stage needs its own
hardware check, memory measurement and explicit qualification.

Classification: PAPER_SPECIFIED architecture/fallback motivation;
REPOSITORY_SPECIFIED qualified source experts/resources/count boundaries;
ASSUMPTION common features, correction grid, OOF proxies, two count targets,
model/reference/ensemble grid, selection and quota rule;
HARDWARE_DEVIATION CPU float64 cache/model processing;
UNRESOLVED generalization, historical TEST/upstream exposure, vocabulary
distribution shift and meeting all four targets under the common procedure.

Complete results and their independent qualifications are recorded below.
Prior outputs are preserved irrespective of the comparison outcome.

## Completed first run and the next validation control

The shared v1 smoke executes all four cases and the independent smoke audit
passes. Its first auditor failed on a constant image aspect ratio represented
with tiny floating-point variance: the independently reconstructed scaler did
not apply StandardScaler's constant-feature error bound. The audit was repaired
against the installed implementation, with a regression test; fitted models and
features were unchanged and the failed report is retained.

The full v1 run completed all four fits and froze every choice before evaluation:

| Dataset | Version | v1 TEST F1 % | Difference versus baseline, points | DeepSolo images/total (%) |
| --- | --- | ---: | ---: | --- |
| Total-Text | U-Net |79.7119|-0.2785|150/300 (50.00%)|
| Total-Text | YOLO |80.9388|-0.0169|150/300 (50.00%)|
| ICDAR2015 | U-Net |79.0943|+0.1609|228/500 (45.60%)|
| ICDAR2015 | YOLO |80.8403|+0.8913|228/500 (45.60%)|

This run does **not** meet the user's four no-degradation conditions. Do not
promote it as their solution. Its full independent audit is separate from the
smoke audit and is retained under `full/audit_report_v1` when completed.

The next control is declared in
`configs/approximate/common_budget_pipeline_pooled_validation_v2.json` before
its execution. Exact batch quotas do not need a scalar-threshold calibration
population. Pool the existing selection and CAL cohorts into125 TT validation
images and100 IC15 validation images; keep1128/900 FIT and every TEST image
unchanged. Select correction/head/reference/ensemble with the same finite grid
and same criterion on that larger heldout-from-current-gate-fit validation set.
This is one shared90/10 TRAIN protocol; no separate scalar-cut CAL remains.
It is a research follow-up after v1 TEST exposure, not an untouched confirmation.
No test-label trial search is introduced.

## User acceptance revision and ICDAR2013 extension

The user subsequently accepts Total-Text F1 strictly above78.2% for both
versions, while retaining the existing150/300 (50.00%) quota. ICDAR2015 retains
its no-degradation references and228/500 (45.60%). Under those revised criteria,
the completed shared v1 meets all four requirements. Preserve the original
small TT losses in the comparison rather than changing the baseline numbers.
`user_acceptance_criteria_v2.json` records this steering. The pooled-validation
control completed its smoke; its full run was prepared but is unnecessary for
the revised target and has not been executed.

The user also requests testing this procedure against90% on ICDAR2013. The
extension profile `configs/approximate/common_budget_pipeline_icdar13_v1.json`
preserves the shared v1 correction/feature/word-risk/head/reference/ensemble
recipe and separate validation rule. Preserve original FIT183/selection23/CAL23,
all233 TEST images/1095 care words and61/233 (26.18%) DeepSolo decisions.
Evaluate TT-derived U-Net, recovered-slot YOLO, and YOLO with its qualified
U-Net complement separately; the third is not called YOLO alone. The same
authenticated split-Generic TRAIN teacher is used across all three; TEST uses
the identical optimized Generic DeepSolo cache. TRAIN Generic vocabulary
inherited from all229 TRAIN words is an upstream dependency, so FIT-only word
folds do not establish an untouched end-to-end estimate. No IC13 TEST trial
chooses a policy. The90% value is a test target, not evidence of achievement.

## Completed qualified results

The common TT/IC15 v1 and the IC13 extension both completed. Every shared
feature/training/reference/guard/ensemble configuration field agrees across
the two source snapshots; the AST hashes of all14 scientific core functions
also agree. This verifies shared behavior beyond entrypoint naming. The
selected feature sets, regularization and ensembles differ because the same
selection rule learns from each dataset, not because TEST chooses a recipe.

| Dataset | Version | Shared procedure F1 % | Previous qualified F1 % | Difference, points | DeepSolo images/total (%) |
| --- | --- | ---: | ---: | ---: | --- |
| Total-Text | U-Net |79.7119|79.9904|-0.2785|150/300 (50.00%)|
| Total-Text | YOLO |80.9388|80.9558|-0.0169|150/300 (50.00%)|
| ICDAR2015 | U-Net |79.0943|78.9333|+0.1609|228/500 (45.60%)|
| ICDAR2015 | YOLO |80.8403|79.9490|+0.8913|228/500 (45.60%)|
| ICDAR2013 | U-Net |84.4294|82.3416|+2.0878|61/233 (26.18%)|
| ICDAR2013 | YOLO |86.6067|86.9814|-0.3747|61/233 (26.18%)|
| ICDAR2013 | YOLO + U-Net complement |86.9202|87.7275|-0.8073|61/233 (26.18%)|

The user's revised empirical TT/IC15 requirements are satisfied: TT remains
strictly above78.2%, both IC15 versions improve their qualified point references,
and every requested budget is exact. This is not a population no-regression
guarantee. Conditional paired2000-image-bootstrap95% intervals for the four
differences, in table order, are [-1.1022,+0.5456], [-1.0990,+1.0339],
[-1.1212,+1.5215] and [+0.0214,+1.8004] points. They omit fitting/selection
uncertainty and prior TEST exposure. Only the IC15 YOLO interval excludes zero
in this conditional analysis; no untouched generalization claim follows.

The shared method **does not reach90% on IC13**. Replaying an exact privileged
image selector on the same new expert outputs gives:

| IC13 version | Actual shared gate F1 % | GT-perfect F1 % | DeepSolo images/total (%) |
| --- | ---: | ---: | --- |
| U-Net |84.4294|88.8469|61/233 (26.18%)|
| YOLO |86.6067|91.2150|61/233 (26.18%)|
| YOLO + U-Net complement |86.9202|91.8744|61/233 (26.18%)|

These optima have zero exact optimality residual and consume TEST labels only
for diagnosis. They are not deployed routing features. The fixed U-Net expert
cannot reach90% through image selection alone at61 images; the YOLO experts
have theoretical room, but the trained common gate does not locate that subset.
Its best observed IC13 mixed result remains3.0798 points below90%. Its gap to
its same-output oracle is4.9542 points. The earlier qualified mixed gate still
has the higher IC13 point score and is preserved, not replaced on that claim.
IC13 uses183 fitting images versus1128 TT and900 IC15, with only23 selection
images; its benchmark vocabulary and upstream fitting dependencies also persist.
These are plausible generalization limitations, not isolated causal proof.

## Qualification, deployment and evidence

The full scientific fits comprise441 image count-head bundles and42 word-risk
models across seven cases; smoke fits are excluded. Ten focused new regression
tests and the relevant lexicon/oracle tests qualify the new boundaries. Separate
real-cache smokes cover all four TT/IC15 cases, pooled-validation engineering
and all three IC13 cases. Complete independent reports verify7,492 per-image
count records,7,392 selection trials,35 word-OOF refits, selected count-head
refits and2,887 annotation-free deployment score replays. Maximal score replay
error is below1.1e-14. No new GPU, recognizer or DeepSolo neural inference was
needed; these figures describe CPU training/replay, not online latency.

Main qualification: `full/audit_report_v2/independent_audit.json`; IC13:
`icdar13_v1/audit_report_v1/independent_audit.json`. The main independent audit
was repeated from an immutable source snapshot to bind its executed auditor
and deployment source explicitly. The first audit/qualification manifest and
all earlier failures remain preserved. Each run's `deployment_qualification.json`
binds the selected models, correction policy, configuration and audit by hash.

The shared inference API receives qualified light rows, original image size and
predicted maps. It receives no GT, validity mask or actual DeepSolo result:

```python
from recognition.common_budget_deployment import CommonBudgetPipeline

pipeline = CommonBudgetPipeline(run_root, case_name)
observations = {
    sample_id: pipeline.score(light_rows, image_wh, predicted_maps)
    for sample_id, light_rows, image_wh, predicted_maps in light_outputs
}
decision = pipeline.route_batch(
    observations, rgb_hashes, fallback_images=requested_fallback_images
)
# Use decision['light_rows'] for accepted images and invoke the qualified
# dataset-specific DeepSolo backend for exactly decision['fallback_ids'].
```

Use `full` with `totaltext__unet`, `totaltext__yolo`, `icdar2015__unet` or
`icdar2015__yolo`; use `icdar13_v1` with `icdar13__unet`, `icdar13__yolo` or
`icdar13__yolo_unet`. Full-batch ranking gives an exact image quota; it is not
a transferred scalar threshold or a streaming latency measurement. Missing or
changed qualification/model/vocabulary bytes fail explicitly.

The joint machine-readable report, shared-recipe identity proof, complete
model-size inventory and figure are in `final_report_v1` under the external
root. A portable completion receipt is
`provenance/COMMON_BUDGET_PIPELINE_COMPLETION_V1.json`. The trained router
uses OCR/map/lexical/risk features without new CLIP or BLIP encoders; qualified
upstream TT correction modules and the YOLO map dependency remain part of
the stated expert cost. This completes the revised common TT/IC15 target and
the requested IC13 test as approximate reproduction, with the IC13 goal unmet.

## Subsequent scalar-threshold parity study

The new user objective replaces fixed fallback quotas with maximal light-image
acceptance at the measured all-DeepSolo F1. See
[COMMON_PARITY_THRESHOLD.md](COMMON_PARITY_THRESHOLD.md) for the shared
CAL-only selection rule, unchanged full TEST evaluation, independently checked
frontier and explicitly separate TEST-label diagnostic. The previous fixed-quota
results and models remain preserved; they do not establish DeepSolo parity.
