# Essential router weights and validation calibration near half fallback

Date: 2026-10-27. **Approximate reproduction, historically explored TotalText TEST 300.**
The target is the nominal numerical value 79.8 with 45–55% of images routed to
DeepSolo. The paper Table 5 calls 79.8 recognition accuracy without an operational
metric definition. Our symbol-free lexical micro F1 is not a verified equivalent
protocol. The paper's TotalText fallback count is unavailable.

The corrected U-Net expert and native DeepSolo caches are unchanged. Decisions
are logical selections over cached outputs; new DeepSolo inference images:0.
No end-to-end latency or wall-clock saving is claimed.

## Frozen validation protocol

The original lightweight library contains 50 heads fitted on 1028 TRAIN images.
The validation 125 held out from U-Net gradients is the population established by
[the exposure study](ROUTER_VALIDATION_STUDY.md). A deterministic SHA256 ordering
with seed 20261027 partitions it into 62 model-selection and 63 calibration images.
No model is refitted after selection or calibration. The old CAL 100 was seen
by U-Net gradients; it is not used to set the revised cutoff. Validation 125
was previously used for upstream checkpoint selection; this does not establish
an untouched validation experiment for the complete correction chain.

Twenty existing candidates with no extra encoder pass the predeclared cost
filter. Selection uses micro F1 at 50% U-Net acceptance, then prefers fewer stored
scalars within 0.25 F1 points of the best candidate. The chosen head is Ridge
alpha 1 using 18 OCR summaries and 12 predicted-mask coverage descriptors.
The maximum-acceptance and component thresholds come from the pinned source
configuration, rather than TEST tuning. The gate sees predicted text and maps,
never annotations, validity masks or actual fallback outputs.

The scalar cut is a calibration score quantile. Accept the corrected U-Net
above the cut; exact ties use stable sample IDs. The primary calibration
acceptance is 50%;45% and 55% are predeclared controls. A fixed scalar quantile
does not guarantee the same fallback fraction on a different distribution.
The optional exact 50% batch ranking is a budget control without a risk guarantee.

## TEST results

All 300 images and 2204 care words are retained. Normalization uses Unicode
casefold followed by alphanumeric characters only. Counts use per-image lexical
multisets, no spatial matching, and the inherited native 256 ignore/care policy;
empty or unmatched care predictions are not silently discarded. See the
[source metric boundary](UNET_CORRECTED_TOKEN_ROUTER.md).

| Frozen CAL U-Net acceptance | Score cutoff | TEST F1 % | DeepSolo images/total (%) | Nominal 79.8 and 45–55% budget |
| ---: | ---: | ---: | ---: | --- |
|45%|−0.4388694973|80.1819|163/300 (54.33%)|Both met|
|50%, primary|−0.5549093274|80.0383|151/300 (50.33%)|Both met|
|55%|−0.7041223687|79.0106|137/300 (45.67%)|Accuracy target missed|
|Exact 50% batch control|Rank current scores|79.9904|150/300 (50.00%)|Both met, descriptive control|

Primary counts: TP 1670, FP 299, FN 534. An image bootstrap with 2000 draws yields
F1 interval 77.4556–82.2922% for this fixed policy, DeepSolo 151/300 (50.33%).
It omits fitting/selection uncertainty and does not erase historical TEST exposure.
This is a nominal target achievement, not a reproduced paper result or evidence
that the true performance exceeds 79.8.

The initial frozen OCR-only primary using the old CAL 100 is preserved:
F1 81.0060%, DeepSolo 176/300 (58.67%), outside the requested 45–55% range.
Its predeclared 55% acceptance control gave F1 79.9518%, DeepSolo 158/300 (52.67%).
That TEST control was not promoted to a newly selected primary.

## Incremental router weights

Decimal units; saved tensor dtypes are preserved. Shared U-Net, PARSeq, GNN,
RGB verifier, byte corrector and DeepSolo weights are excluded. Weight storage
is separate from activation memory, software overhead and online latency.

| Component | Tensor/normalization bytes | Serialized artifact bytes |
| --- | ---: | ---: |
|Original OCR18 essential head|440|21530|
|Selected OCR18+mask12 essential head|728|28146 including frozen policies/provenance|
|Previous token attention head|880140|922046|
|CLIP native state, including buffers|598485140|598641023|
|Full native BLIP2|15769786368|15769954544|
|Previous head plus both complete encoders|16369151648|Sum of source artifacts, not simultaneous VRAM|

The selected Ridge requires 31 coefficients/intercept and 60 normalization values,
91 FP64 scalars. No CLIP or BLIP encoder is needed. Its serialized deployment
also carries metadata; it is not the numerical parameter count. The previous
pipeline generated CLIP and BLIP features in sequential stages, so summed
weights must not be called a measured peak VRAM requirement.

The follow-up [encoder-family training study](ENCODER_FAMILY_ROUTER.md) tests
whether a larger head can make CLIP alone, BLIP alone, or their combination
worth its cost, and qualifies an explicitly reduced BLIP visual loader.

## Artifacts and verification

External root:
`/extra1/bmartelli/context-str-runs/essential-router-budget50-20261027-v1`.
`budget_validation_v1` contains `frozen_choices.json`, `deployment.joblib`,
`test_scores.json`, `results.json`, `completion.json`, and `independent_audit.json`.
`weight_comparison.json` records exact checkpoint-header and tensor measurements;
`scripts/measure_router_weight_storage.py` independently replays the read-only
analysis into `weight_comparison_replay.json` with identical measured sizes.
All original failed runs and upstream audits are preserved.

The independent audit passed 20 candidate-selection replays,3 calibration cuts,
4 TEST route/count sums,300 authenticated single-image feature replays and 900
decisions. Maximum batch versus single-image score error was 3.56e-15.
Original weighted-Ridge fitting was independently checked in the source audit.

Maintained implementation: `scripts/run_essential_router_budget.py`,
`scripts/audit_essential_router_budget.py`, `recognition/essential_budget_router.py`.
Configuration: `configs/approximate/essential_router_budget50_v1.json`.
Tests: `tests/recognition/test_essential_router_budget.py`.
Scientific classification: `ESSENTIAL-BUDGET50-001` in `docs/PAPER_AMBIGUITIES.md`.
