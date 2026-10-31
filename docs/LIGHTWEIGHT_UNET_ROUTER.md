# Lightweight corrected-U-Net router: retained signals and calibration

Approximate reproduction / research extension. Complete TEST300 / 2204 care words. Exact casefold+isalnum lexical micro F1, source native256 ignore; no localization IoU. Counts describe logical cached-output routing, not online speedups.

## Implemented reduction

The conservative feature-pruning follow-up retains 18 OCR/geometry descriptors and ridge alpha 1: 19 fitted head coefficients plus 36 scalar normalization values. The previous attention head has 220035 parameters. CLIP/BLIP, per-word embedding extraction, global image/tile context, token auxiliary heads and predicted-map coverage are not required by this selected gate. The original corrected U-Net/PARSeq/GNN/RGB-verifier/text-editor expert stays frozen; its upstream cost is not removed.

The deployed single-image API accepts only OCR observations and image dimensions. All 300 TEST outputs and 2400 policy decisions match the stored batch predictions, with no map/context reads or extra encoder calls. Model and provenance overhead affect serialized size; coefficient counts are not a full-pipeline model-size or latency claim.

## Feature evidence

The predeclared run fits 50 heads: ten feature sets x three ridge penalties/two shallow-tree budgets. FIT 1028 fits all transforms/heads; selection 125 chooses candidates; CAL 100 fixes policies of those exact heads. No refit follows calibration. The native DeepSolo comparator is now consistent on TRAIN and TEST.

The predeclared cost/tolerance selector chooses core38. Its selection mean F1 is only 0.0912 points above OCR-only; its conditional 95% interval is [-0.3274,+0.5680]. Token statistics alone add 0.0504 points, interval [-0.0905,+0.2054]. No tested extra feature set has a positive lower paired interval against OCR-only. These are conditional exploratory comparisons after candidate hyperparameter selection, not confirmatory null-hypothesis tests or proof that context is universally useless.

A separately saved exploratory follow-up therefore retains extra features only with positive validation evidence; it chooses OCR-only before loading TEST metrics in that follow-up. Original core38 choices/results remain unchanged. The TEST had already been evaluated and is historically explored; this is not an untouched confirmatory experiment.

| Feature set | Mean selection F1 difference versus OCR-only, points | Conditional95% interval | DeepSolo images/total (%) for each compared method |
| --- | ---: | ---: | --- |
| ocr | +0.0000 | [+0.0000, +0.0000] | 113/125 (90.40%); 100/125 (80.00%); 87/125 (69.60%) |
| ocr_tokens | +0.0504 | [-0.0905, +0.2054] | 113/125 (90.40%); 100/125 (80.00%); 87/125 (69.60%) |
| ocr_mask | +0.0724 | [-0.6402, +0.7018] | 113/125 (90.40%); 100/125 (80.00%); 87/125 (69.60%) |
| core | +0.0912 | [-0.3274, +0.5680] | 113/125 (90.40%); 100/125 (80.00%); 87/125 (69.60%) |
| core_local | -0.0457 | [-0.5917, +0.4863] | 113/125 (90.40%); 100/125 (80.00%); 87/125 (69.60%) |
| core_global | -0.0118 | [-0.7248, +0.7218] | 113/125 (90.40%); 100/125 (80.00%); 87/125 (69.60%) |
| core_blip | -0.1276 | [-0.9544, +0.6225] | 113/125 (90.40%); 100/125 (80.00%); 87/125 (69.60%) |
| all | -0.2119 | [-0.8727, +0.4480] | 113/125 (90.40%); 100/125 (80.00%); 87/125 (69.60%) |
| all_no_mask | +0.3350 | [-0.0957, +0.8357] | 113/125 (90.40%); 100/125 (80.00%); 87/125 (69.60%) |
| all_no_tokens | +0.0048 | [-0.8692, +0.8086] | 113/125 (90.40%); 100/125 (80.00%); 87/125 (69.60%) |

Every selection mean averages 10/20/30% light quotas with the same DeepSolo costs for both compared methods: 113/125 (90.40%), 100/125 (80.00%), 87/125 (69.60%), respectively. The simultaneous exploratory intervals and all individual count/route points are preserved in choice_frozen.json and trials.json.

## Frozen calibration policies on TEST

| Rule | Frozen cutoff or normalized correction | F1 % | DeepSolo images/total (%) | Conditional95% delta versus all DeepSolo, points |
| --- | ---: | ---: | ---: | ---: |
| calibration_f1_budget_0.0 | 0.314654 | 82.3753 | 274/300 (91.33%) | [-0.5748, -0.0398] |
| calibration_f1_budget_0.0025 | -0.060090 | 81.7487 | 218/300 (72.67%) | [-1.5349, -0.3506] |
| calibration_f1_budget_0.005 | -0.670763 | 78.6354 | 135/300 (45.00%) | [-5.6114, -2.6635] |
| calibration_f1_budget_0.01 | -1.319135 | 76.0646 | 94/300 (31.33%) | [-8.6163, -4.7406] |
| conformal_0.05 | 0.598977 | 82.6540 | 300/300 (100.00%) | [+0.0000, +0.0000] |
| conformal_0.1 | 0.355051 | 82.6540 | 300/300 (100.00%) | [+0.0000, +0.0000] |
| conformal_0.2 | 0.198878 | 82.6540 | 298/300 (99.33%) | [+0.0000, +0.0000] |
| raw_zero | 0.000000 | 81.7856 | 228/300 (76.00%) | [-1.4320, -0.3742] |

All-native DeepSolo reference: F1 82.6540%, DeepSolo 300/300 (100.00%). All policies were fixed using CAL only; confidence intervals condition on the trained models and routes and omit search/training variability.

The empirical zero-loss CAL policy uses utility cutoff 0.314654. It scores 81.5104% on CAL with DeepSolo 94/100 (94%), versus all-DeepSolo 81.4574% with 100/100 (100%). On TEST it scores 82.3753%, DeepSolo 274/300 (91.33%), a 0.2787-point observed loss from the all-native endpoint. Validation-only calibration is necessary but does not guarantee transfer from a small, historically exposed cohort. Larger allowed empirical losses rapidly increase TEST damage; they are not recommended as a demonstrated accuracy-preserving deployment.

The original predeclared core38 zero-loss-CAL rule accepts 77/100 CAL images but sends only 89/300 TEST images to DeepSolo (29.67%) and scores 76.4153%. This failed primary result is preserved. Feature reduction alone does not fix threshold reliability.

## Meaningful general rule

The general decision criterion is a strictly positive lower bound on normalized advantage. A one-sided split-conformal residual quantile is fitted on CAL 100 for the frozen gate. The zero comparison has a stable meaning; the residual correction still depends on representative calibration data. Under exchangeability the nominal claim concerns marginal utility coverage, not the fraction of harmful accepted predictions or preservation of global micro F1.

At nominal alpha 0.05/0.10 the method keeps all 300/300 images with DeepSolo (100%): F1 remains 82.6540%, with no useful bypass. At alpha 0.20 it sends 298/300 (99.33%) to DeepSolo and retains the same 82.6540% F1. This is a conservative result, not a large accuracy-preserving compute saving.

References: [calibrated deferral](https://proceedings.mlr.press/v162/verma22c.html), [split conformal calibration](https://arxiv.org/html/2107.07511v6), [risk-control scope](https://arxiv.org/abs/2208.02814v4). The first paper is motivation rather than an implemented OvA model, and no direct CRC theorem is asserted for nonmonotone aggregate F1.

## Descriptive quota comparison

| U-Net acceptance | Lightweight OCR-only F1 % | Previous corrected token F1 %, descriptive | DeepSolo images/total (%) for both |
| ---: | ---: | ---: | ---: |
| 0% | 82.6540 | 82.6540 | 300/300 (100.00%) |
| 10% | 82.3753 | 82.2746 | 270/300 (90.00%) |
| 20% | 81.7814 | 81.9570 | 240/300 (80.00%) |
| 30% | 81.3008 | 81.2981 | 210/300 (70.00%) |
| 40% | 81.0630 | 80.0966 | 180/300 (60.00%) |
| 50% | 79.6916 | 78.4428 | 150/300 (50.00%) |

This comparison is descriptive: the old gate fitted all 1253 TRAIN images against the original DeepSolo profile, while the new heads fit 1028 with the native comparator and separate selection/calibration. It is not a causal isolation of attention removal. No TEST quota was used to select the new gate or a cutoff.

## Qualification and limitations

Focused suite: 34 passing tests. Engineering smoke passes all 50 fits and its independent audit. Full audit passes 14 source files, 4 feature populations, 2 FIT-only projections, 50 fitting scopes, 30 independently solved weighted ridge normal equations, 20 tree-budget/initialization checks, 150 selection routes, 40 empirical CAL optima, 30 conformal corrections, 80 CAL routes and 140 TEST routes. The pruned deployment independently replays all 300 single-image predictions and 2400 decisions.

No new expert inference, GPU runs, installations or changes to existing expert weights/datasets occur. FP64 CPU feature/head computation with four threads is documented as HARDWARE_DEVIATION. Head timings exclude upstream OCR and optional context generation, and do not establish end-to-end latency. Original source expert TRAIN/calibration exposure, small validation cohorts, historical TEST exploration and exploratory pruning remain unresolved scientific limitations.

The implementation achieves a much cheaper router. It does not yet establish substantial bypass with unchanged recognition quality.

## Reproducible implementation

Source: `recognition/lightweight_unet_router.py`; runner: `scripts/run_lightweight_unet_router.py`; independent audit: `scripts/audit_lightweight_unet_router.py`; explicit mode/configuration: `configs/approximate/lightweight_unet_router_v1.json`. The separate exploratory selection stage is `scripts/prune_lightweight_unet_router.py`. Tests: `tests/recognition/test_lightweight_unet_router.py`, `tests/recognition/test_lightweight_validation_pruning.py`, plus coverage/corrected-token regressions.

Immutable full execution source: `source_snapshot_v2`; full result: `full_v1`; exploratory OCR deployment: `validation_pruned_v1/selected_deployment.joblib`, all under `/extra1/bmartelli/context-str-runs/lightweight-unet-router-20261027-v1`. The first two cohorts are official TRAIN subsets, not new official benchmark splits. CAL100 and TEST have historical expert exposure as explained above.

The head target is `(2 deltaTP - FIT native DeepSolo F1 deltaN)/(observed U-Net word count+1)`, with FIT sample weights equal to that same observable count+1. Coverage threshold 0.5/area 4, PCA 8, model budgets and selection tolerance are explicit ASSUMPTIONs in configuration. Frozen native crops and expert outputs remain unchanged.

Portable evidence: [completion record](../provenance/lightweight_unet_router_v1_completion.json). Scientific classifications: `LIGHTWEIGHT-UNET-ROUTER-001` in `PAPER_AMBIGUITIES.md`. Source background: [corrected token expert](UNET_CORRECTED_TOKEN_ROUTER.md) and [qualified native fallback](UNET_TOKEN_ROUTER.md).
