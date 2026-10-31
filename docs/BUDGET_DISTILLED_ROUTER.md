# Cheap DeepSolo-supervised router at half-image acceptance

**Approximate reproduction / research extension.** The active request is to
approach perfect expert selection while retaining approximately half the images
on the U-Net/PARSeq path, with low incremental routing cost. Paper Section3.5 /
Algorithm2 and existing token, lightweight and validation routers were inspected.
This is a trained research gate, not the paper's training-free MPNet/fuzzy rule.

## Predeclared experiment

Keep the qualified Text+Kernel ResNet50 Attention U-Net, genuine PARSeq and
improved native ViTAEv2-S/TextOCR DeepSolo outputs fixed. Use the existing1128
fitting /125 validation /300 TEST split, after the same two cross-TEST RGB
exclusions. No refit follows validation selection or threshold calibration.

Train a small CPU head to predict paired utility
`2*(TP_light-TP_DeepSolo)-F_reference*(N_light-N_DeepSolo)` divided by
`sqrt(1+selected_raw_rows)`. The reference F1 uses fitting DeepSolo outputs only.
Annotations and actual fallback predictions are training/evaluation inputs,
never deployment inputs. This is a cost-sensitive deferral adaptation inspired
by [Mozannar and Sontag, ICML2020](https://proceedings.mlr.press/v119/mozannar20b.html),
not a reproduction of their classifier loss or its guarantees for OCR micro F1.

Compare OCR summaries, spatial predicted-map residual summaries, and context
pooled from the existing U-Net encoder/decoder. Shared context requires no new
encoder family or second expert forward during deployment. Extract the same
frozen U-Net activations offline for this experiment, replay-check probabilities,
and record actual GPU peaks and added pooling time. Fit PCA on fitting images
only. No CLIP or BLIP model is loaded or needed by any deployment candidate.

Compare real examples against an explicitly synthetic training augmentation:
remove selected raw PARSeq outputs and optionally add already decoded rejected
outputs. Recompute exact lexical targets against the fixed annotations and
DeepSolo outputs. These are counterfactual candidate subsets, not new measured
OCR inference. Their fixed original maps/context can reveal omissions still
visible in the segmentation maps, but cannot simulate every unseen mask failure.

Select among24 predeclared small heads using mean validation hybrid lexical F1
at40/50/60% light acceptance. Bind45/50/55% (plus40/60%) scalar cuts to the exact
selected model using its validation scores. Report independently the predeclared
batch top-half rule, which uses scores and a fixed quota without TEST labels.
Freeze model and cuts before processing TEST observations/outcomes in this run.

## Boundaries and qualifications

- ASSUMPTION: descriptors, spatial grid, shared pooling/projection, utility
  normalization, finite candidate grid and counterfactual augmentation.
- REPOSITORY_SPECIFIED: frozen expert artifacts, cached improved DeepSolo,
  exact source lexical normalization, ignore policy and deterministic splits.
- HARDWARE_DEVIATION: observed RTX3090 FP32 sequential frozen extraction;
  existing environments; CPU small-head training/inference. No8GB qualification.
- UNRESOLVED:1128 fitting expert predictions retain their original training
  exposure. Augmentation is an attempted robustness intervention, not genuine
  expert out-of-fold training. Validation was historically used for expert
  checkpoint selection; TEST is historically explored. Small validation and
  search variability limit generalization claims.

Primary metric is per-image lexical multiset casefold+isalnum micro F1, without
localization IoU; source native256 care/ignore handling and all empty/symbol-only
denominators remain. Every router result includes DeepSolo images/total/percent.
Policies are logical cached-output routing; new DeepSolo inference is zero.
Record head+feature latency and shared pooling separately from expert inference;
do not infer end-to-end acceleration from parameter counts alone.

Implementation: `recognition/budget_distilled_router.py`,
`scripts/run_budget_distilled_router.py`. Explicit approximate configurations:
`configs/approximate/budget_distilled_router_v1.json` and its local paths profile.
The predeclaration above precedes completion; audited results follow.

## Completed studies and negative result

All three protocols completed24 heads each. All candidate choices and scalar
cuts were frozen on their declared validation images before this run read TEST
observations/outcomes. The second and third protocols are explicitly exploratory
follow-ups after the first study; they are not independent confirmation.

| Protocol | Router fitting / validation images | Selected head | TEST F1 at half quota (%) | DeepSolo images/total (%) | Incremental CPU observations + head median |
| --- | --- | --- | ---: | --- | ---: |
| Original split | 1128 /125 | Shared U-Net context, augmentation,64 depth2 boosted trees | 77.4178 | 150/300 (50.00%) | 7.233ms plus shared pooling |
| Expert-unseen adaptation | 1203 /50 | OCR/layout, augmentation,233-coefficient ridge | 77.5131 | 150/300 (50.00%) | 5.441ms |
| Expert-unseen fitting only | 75 /50 | Shared U-Net context, augmentation,249-coefficient ridge | 76.7189 | 150/300 (50.00%) | 5.546ms plus shared pooling |
| Previous U-Net token reference | 1128 /125 | Existing220035-parameter token head | 77.8892 | 150/300 (50.00%) | Not remeasured |
| Conditional U-Net GT oracle | Privileged GT choice; not deployment | Fixed expert outputs/eligibility | 82.5465 | 150/300 (50.00%) | Not applicable |
| Improved native DeepSolo endpoint | Frozen fallback | ViTAEv2-S/TextOCR | 82.6540 | 300/300 (100.00%) | Not remeasured |

The adaptation protocol deterministically moves75 former expert-holdout images
into router fitting with12x weights;50 remaining images select the model/cuts.
The pure-unseen protocol uses only those75 images to fit both router and PCA;
it excludes the1128 expert-fitting examples and selects on50 validation images
at exactly50% acceptance. Thus these comparisons vary supervision and are not
architecture-only ablations. Both reuse qualified features; neither runs new
U-Net or DeepSolo inference. No existing pipeline default is replaced.

No selected new gate improves the previous half-budget reference. The adaptation
ridge gives a0.3761-point observed loss at the same150/300 (50.00%) fallback
cost, with conditional paired95% interval[-1.8560,+1.1841] points. This does not
establish superiority or equivalence. Shared context and candidate-subset
augmentation do not close the oracle gap in these tested configurations.
The main study's conditional gap is5.1287 points at150/300 (50.00%) DeepSolo;
the adaptation gap is5.0334 points at that same cost.

### Validation-derived scalar cuts

These are operational single-image cuts of unchanged selected weights. Their
actual acceptance differs from the separately predeclared exact-half batch rule.

| Protocol | Validation target | Frozen cut | TEST accepted light | TEST F1 (%) | DeepSolo images/total (%) |
| --- | ---: | ---: | ---: | ---: | --- |
| Original split | 50% | -1.006188 | 142/300 | 78.2195 | 158/300 (52.67%) |
| Expert-unseen adaptation | 50% | -1.111763 | 141/300 | 77.8917 | 159/300 (53.00%) |
| Expert-unseen fitting only | 50% | -2.059818 | 144/300 | 77.3789 | 156/300 (52.00%) |

The original-split scalar policy cannot be called better than the previous
78.0138% policy by comparing these F1 values alone: its158/300 (52.67%)
fallback cost differs from the prior153/300 (51.00%). Use the exact-half table
to compare selector quality at matched cost. All40/45/50/55/60% cuts and batch
quota results are retained in each report/CSV.

### Artifacts, qualification and deployment boundary

External roots under `/extra1/bmartelli/context-str-artifacts/budget-distilled-router-v1/full/`:

- `7036878d0a54`: original split; independent replay7933 PASS.
- `b50d00c6c020`: expert-unseen adaptation; independent replay6433 PASS.
- `d509f2365b9b`: expert-unseen fitting only; independent replay1873 PASS.

Each root includes `deployment.json`, original immutable code snapshot,
per-image predictions, `training_observations.npz/.json`, all24 trials,
validation cuts, conditional GT oracles, completion/audits, and `report_v2/`
with Markdown,CSV,PNG andSVG. Portable identities and selected metrics are
saved in `provenance/budget_distilled_router_20261027_completion.json`.

The adaptation deployment is57184 bytes including metadata and standardization;
it needs only existing OCR observations and predicted maps.233 fitted head
coefficients and5.441ms median CPU observation/head processing do not establish
an end-to-end speedup. Shared context candidates require additional pooling:
source measurements were9.436ms median in TRAIN and1.580ms median in TEST,
with shared-GPU contention and copies included. These source measurements
must not be added blindly into a clean latency benchmark.

Original frozen extraction used RTX3090 FP32,batch1,RGB512. TRAIN allocated /
reserved peaks552595456/778043392 bytes; TEST550760448/801112064 bytes.
All1553 probability maps replay exactly (maximum difference0), and state hashes
are unchanged. No new DeepSolo,CLIP orBLIP inference occurs. No model-run
failure or OOM occurred. The initial engineering smoke completed24 fits and
its own routing checks; the final focused suite passes64 tests.

Qualification history: the first supplemental auditor compared independent
refits bit-for-bit with an advanced-indexed Fortran-order matrix, unlike the
original C-order `np.stack`. It reported coefficient differences up to
5.5823e-15 and mean differences2.6645e-14. A diagnostic rerun preserved that
failure. Matching the original tensor memory layout restored exact comparisons;
no tolerance, learned weights, predictions or selection rule was changed.
All three final independent audits pass. Later entry-point hardening adds
explicit feature-setting validation, complete-run no-op checks, and GPU peak
metadata on errors; executed scientific snapshots remain immutable.
The pure-unseen configuration initially retained the preceding adaptation's
descriptive1203-image/12x-weight entry. Its resolved partitions,actual fitting
tensors and independent audit establish75 fitting images with unit real-view
weights; the current source configuration removes that stale prose. No
scientific runtime parameter or historical artifact is changed by this erratum.

Example deployment API: `predict_observation(deployment, rows, candidates,
image_wh, probabilities, shared_context=None)`. `shared_context` is required
only when declared by the selected bundle. It returns a utility score and
validation-cut decisions. For a batch with a fixed acceptance budget, use
`rank_accept(scores, ids, 0.5)`, with deterministic ties and no annotations.

## Verified existing alternative on the corrected U-Net expert

The repository also contains the earlier qualified
[essential budget gate](ESSENTIAL_ROUTER_BUDGET50.md). It uses a different,
corrected U-Net/PARSeq expert with existing GNN/RGB-verifier/text-editor stages.
Its standalone lexical F1 is66.2278%, DeepSolo0/300 (0%), versus63.0532%,
DeepSolo0/300 (0%), for the uncorrected expert studied above. Its existing
30-feature Ridge was trained previously; this task independently rechecked and
exported it, and does not claim that earlier training as a new experiment.

A fresh run of `scripts/audit_essential_router_budget.py` passes all20 candidate
selection replays,3 calibration cuts,300 single-image feature replays and900
decisions. Its62 selection and63 calibration images are disjoint subsets of the
125 expert-unseen population. Every threshold uses the unchanged trained head.

More importantly, all600 per-image light/fallback count vectors match those of
the corrected-expert token reference. Thus the following difference compares
routing on identical expert outputs, although router fitting/selection histories
differ and this is not an architecture-only causal ablation.

| Corrected-expert policy | TEST lexical F1 (%) | DeepSolo images/total (%) |
| --- | ---: | --- |
| Previous corrected-expert token head | 78.4428 | 150/300 (50.00%) |
| Existing essential gate, exact-half batch rule | 79.9904 | 150/300 (50.00%) |
| Existing essential gate, validation50% scalar cut | 80.0383 | 151/300 (50.33%) |
| Conditional corrected-expert GT oracle | 83.6555 | 150/300 (50.00%) |

The equal-budget observed gain is1.5476 points, closing29.69% of that reference's
gap to its conditional oracle;3.6650 points remain at150/300 (50.00%) DeepSolo.
The paired RGB-cluster2000-draw interval is[+0.0286,+3.0539] points, conditional
on fixed models/routes and excluding training/search uncertainty and historical
TEST exposure. The ratio-oracle optimality certificate is exactly zero. This
positive comparison must not be attributed to the72 new uncorrected-expert heads.

`recognition/essential_portable_router.py` now deploys the existing head using
NumPy and the same18 OCR +12 predicted-map descriptors. It preserves31 fitted
coefficients/intercept and60 normalization scalars:728 numerical bytes. No
CLIP/BLIP or DeepSolo result is needed to make the routing decision. Existing
upstream correction costs remain. All300 TEST scores replay within3.56e-15,
and all900 scalar-policy decisions/counts match. CPU observation/head median
was1.669ms in the independently checked export; this excludes experts and cold
imports and is not an online pipeline benchmark.

The portable export and source-independent audit are under
`/extra1/bmartelli/context-str-artifacts/budget-distilled-router-v1/qualified_essential_portable/`.
The maintained exporter produces the same numerical head and gates in
`qualified_essential_portable_v2/`, with metadata kept separate from parameter
counts. Reproduction requires explicit paths:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  .venvs/context-str-main-py311/bin/python -m scripts.export_essential_portable_router \
  --mode approximate \
  --source-root /absolute/qualified/essential/budget_validation_v1 \
  --source-audit /absolute/independent/source_recheck.json \
  --output-root /absolute/new/export-directory
```

The final focused suite includes69 passing tests. The three new negative studies
remain unchanged and preserved. This verified existing alternative supplies a
small deployment with better observed matched-expert routing; it does not
establish strict paper reproduction or perfect selection.
