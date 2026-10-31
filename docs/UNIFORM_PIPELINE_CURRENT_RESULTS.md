# Uniform pipeline: current verified result

The objective was resumed on2026-10-31 with the original DeepSolo counts as
hard limits. See [the active budget-constrained study](UNIFORM_HARD_BUDGET.md).
The larger-fallback controls below are historical and cannot satisfy that
resumed objective.

Approximate reproduction / research extension. The formal word-F1 thresholds
are exceeded by the exploratory common95% fallback control. Primary IC13 and
the previously validation-frozen common80% control remain below target.
The95% control was highlighted after TEST; it is not independent validation.

| Dataset | Primary budget: F1; DeepSolo images/total (%) | Frozen common80%: F1; DeepSolo images/total (%) | Exploratory95%: F1; DeepSolo images/total (%) |
| --- | --- | --- | --- |
| ICDAR2013 | 84.6116%; 61/233 (26.18%) | 88.3268%; 187/233 (80.26%) | 89.9814%; 222/233 (95.28%) |
| ICDAR2015 | 79.9257%; 228/500 (45.60%) | 82.7444%; 400/500 (80.00%) | 82.8381%; 475/500 (95.00%) |
| Total-Text | 80.7919%; 150/300 (50.00%) | 81.8312%; 240/300 (80.00%) | 82.4899%; 285/300 (95.00%) |

All233/500/300 official TEST images and1095/2077/2204 care words are retained.
Metric: per-image Unicode casefold/alphanumeric word multisets, micro TP/FP/FN,
native256 care/ignore, no spatial IoU matching. No failed image is omitted.
These are authenticated cached expert replays; actual new DeepSolo calls for
this full metric evaluation are0. They are not paper Table5 accuracy or strict
reproduction results. Full per-image outputs and independent audits are external.

The common method is RGB768 Attention U-Net with frozen text-pretrained
ResNet50, text/kernel maps384, area-ranked kernel expansion and up to50 crops;
genuine native PARSeq; scalar word-correctness model; shared FIT-only PCA64 of
already-computed encoder features; two-count HGB image utility; exact batch
routing; one pooled learned word-fusion classifier on selected DeepSolo images.
No extra neural model or encoder forward is added by the context or fusion.
Thresholds and weights are learned per domain from the same procedure; the
PCA and fusion classifier are shared. Identity remains a validation candidate.

The image router runs **before DeepSolo**. From the U-Net maps, PARSeq words,
word probabilities, geometry and pooled encoder context, two regression heads
estimate the change in correct words and in emitted words from using the
fallback. Their F1-oriented utility ranks images; the selected quota fixes the
number routed to DeepSolo. Training uses annotated expert outputs, while
inference uses observable predictions only. The fusion step runs **after** the
selected fallback outputs arrive and chooses words from the two experts. It
does not decide which images should invoke DeepSolo. ICDAR2013 follows this
same procedure; its learned weights, vocabulary and calibrated thresholds
remain dataset-specific.

All3 U-Nets completed30 epochs using the same initialization/loss/optimizer
recipe; selected epochs25/30/15 for IC13/IC15/TT. All15 additional conditional
decoder-heldout fits and their crop exports qualify. The extra offline training
covers193980 image-epochs, with zero extra deployment networks. Encoder/initial
decoder upstream exposure, nonnested stacking, benchmark vocabularies and
historical adaptive TEST/reused validation remain material limitations.

The v6b RGB-only chain passes real12-image engineering probes per domain:
DeepSolo10/12 (83.33%) each, with12 real U-Net calls and original-crop PARSeq.
Model load, selected-only legacy handoff, no-GT input, memory and error
propagation are independently checked. This small probe does not establish
full-dataset online latency or fresh full-dataset quality.

The frontend has81987817 neural parameters; with conditional DeepSolo the
distinct one-domain model set has115736671. The U-Net alone reports297.3053
GFLOPs at768 under the declared partial profiler convention (33.0339 at256).
Cold12-image controls already show startup/frontend overhead can make the
whole system slower than DeepSolo alone. No general speedup is claimed.
IC15's qualified fallback peaks near6.95GB allocated and12.22GB reserved on
RTX3090;8GB hardware is not qualified by this experiment.

| Dataset | Current pipeline, seconds; DeepSolo images/total (%) | DeepSolo alone, seconds; images/total (%) |
| --- | --- | --- |
| ICDAR2013 | 15.228; 10/12 (83.33%) | 8.929; 12/12 (100.00%) |
| ICDAR2015 | 20.906; 10/12 (83.33%) | 13.334; 12/12 (100.00%) |
| Total-Text | 14.472; 10/12 (83.33%) | 8.280; 12/12 (100.00%) |

These are separate one-off cold runs on the same twelve RGB images and neural
assets, including setup and I/O. They are engineering observations, not a
randomized latency benchmark or timings for the full common95% evaluation.
See `provenance/UNIFORM_COLD_BATCH_COST_V6B.json`.

Initial authorized deletion removed35 verified non-improving checkpoints,
recovering22.0495GiB. Selected/final weights and all evidence were preserved.
Completed-fold optimizer compaction separately preserves identical final model
tensors, histories and audit receipts. No datasets, weights or large outputs
are committed.

Entry point: `python -m scripts.infer_uniform_word_pipeline`. Use explicit
`configs/approximate/uniform_pipeline_deployment_v6b.json`, dataset, reference
policy quota and RGB-only manifest. Complete manifests are external under
`final_rgb_manifests`; common95 quotas are222/233 (95.28%),475/500 (95.00%),
285/300 (95.00%). Experimental outputs are under `word_union_v6b` and
`live_pipeline_smoke_v6b` in the main run root. See
[full protocol and history](UNIFORM_UNET_FUSION.md) and portable receipt
`provenance/UNIFORM_WORD_UNION_EVALUATION_V6B.json`.

Final evidence checks pass all three training, decoder-bank, full cached metric
and live engineering audits, all21 deployment input pins,41 focused unit tests,
and both generated navigation checks. The derived scope receipt is
`provenance/UNIFORM_PIPELINE_FINAL_SCOPE_AUDIT_V1.json`; earlier negative receipts
are preserved. Numerical targets are reached at high fallback, while a
computationally advantageous solution at the original budgets remains an open
research objective.
