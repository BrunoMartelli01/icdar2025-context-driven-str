# Six context signals for whole-image OCR routing

2026-10-23. User-authorized **approximate reproduction extension**. Implement
and compare all six proposed ideas while preserving the complete YOLO640/PARSeq
and DeepSolo ViTAEv2-S+TextOCR image branches. This is neither the paper's
training-free router nor a strict Table5 recognition-accuracy reproduction.

Before changes, inspect local paper3.5/Figure4/Algorithm2, the official PARSeq
runtime's decode/mask/attention implementation, PARSeq paper3.2/3.3, SceneFaith's
fixed-target-patch protocol and E2STR's retrieval context mechanism. Existing
caption/inventory/VQA/crop-stability/rendered-CLIP experiments remain controls.

## Six implementations and their boundaries

1. **Fixed-word context transplant.** Preserve the native target patch, a guard
   band, coordinates, canvas and framing across original, gray and foreign
   backgrounds. A fourth foreign-background-without-target view controls prior
   support. Donors are dissimilar frozen visual embeddings from TRAIN80 only,
   excluding identical RGB; no GT/transcription selects a donor. A local expanded
   view preserves readable scale in BLIP2's224px processor. This tests local
   context, not the exact synthetic SceneFaith benchmark or arbitrary global scenes.
   Exact-word/foil Yes/No probabilities are uncalibrated support, not certificates.
2. **PARSeq internal-context sensitivity.** Keep the original crop and prediction
   fixed. Teacher-force that predicted string under official forward and reverse
   masks, a fixed odd/even permutation, and null context. Save target probability
   and argmax IDs at each position. Character summaries exclude EOS. This is not
   four independent recognizers and does not change any output word. A native
   no-EOS decode can yield26 symbols; the conditional probe supports at most25.
   Such words remain predictions with explicit probe abstention, never truncation.
3. **Repeated glyph consistency.** Read the decoder's existing cross-attention
   without modifying its forward result. Monotonic attention centers give
   approximate native character windows, not annotations or proof of causal
   attribution. Compare edge patches across distinct same-line word boxes;
   strongly overlapping duplicate boxes do not count as independent support.
   Contradictory labels of visually similar patches produce risk. Absent pairs
   explicitly mean unavailable evidence; consistent errors can still pass.
4. **Partial structural constraints.** In observed English menu/price or
   opening-hours contexts, measure unpaired prices/times and invalid clock forms.
   Context and line grouping are predictions. Standalone prices/times can be
   legitimate, so these are soft features, not word insertion or hard rejection.
   Unstructured scenes receive explicit missing evidence.
5. **Context-conditioned inverse-graphics prototype.** Choose among six hashed
   DejaVu fonts and three fixed line angles using other confident same-line words,
   excluding the target. Compare symmetric nearest-edge distances for target and
   a deterministic spelling foil. Gray edges reduce color/polarity dependence.
   This is a restricted font/angle prototype, not a full perspective/color/lighting
   generative renderer. Unknown styles, unsupported strings or absent neighbors
   may abstain. It differs from earlier CLIP comparison of generic rendered words.
6. **Memory of TRAIN branch errors.** Retrieve nearest frozen image embeddings,
   optionally combined with standardized core quality features. Reference labels
   are fitting-TRAIN deltaTP and delta-predicted-word count. Query ID and identical
   RGB are excluded, including during TRAIN evaluation. This uses supervised
   reference outcomes despite having no new neural updates. It does not reproduce
   the separately trained E2STR architecture or insert GT demonstration strings.

Neural models remain frozen. Geometry/glyph/font/structure features are GT-blind.
The supervised router and error memory deliberately use TRAIN labels. The word
list itself remains the complete original light branch or complete fallback.

## Frozen comparison protocol

- Audited source: `grounded-context-router-v1/full/f9ac718a9aa5`, with pinned plan,
  result and independent audit hashes. Reuse125 TRAIN /300 TEST,983/2204 care
  words, historical TRAIN80 fit /VAL45, complete frozen fallback and core46 inputs.
- Seven scalar feature sets: core46, swap61, permutation58, glyph50, structure54,
  render53 and combined92. Combined includes the five measured signal groups;
  error memory remains a separate predictor. Cross each with Ridge/ExtraTrees;
  add two memory variants and five old frozen controls:21 total families.
- Inner five-fold GroupKFold uses original RGB hashes, with fold-only fitting
  labels/scalers/reference F1/memories. Fixed background donor images are an
  unlabeled TRAIN80 bank, not per-fold labeled exemplars. CV selects explicit
  hyperparameters by final word microF1. Do not fit or scale on VAL/TEST.
- VAL45 chooses one family and scalar cut at20/30/50% requested bypass, including
  the five old controls as candidates. Fixed declared family order breaks ties.
  Primary: trees_all. Fit final models on80 only; do not refit125.
- Freeze choices before new TEST neural probes. TEST raw probes/vectors bind
  the already-written choices hash. Preserve all candidate outcomes and transferred
  scalar cuts, even if the selected policy or a new method loses performance.
- Supervised regression/memory rank by2*deltaTP_hat minus fitting-only all-DeepSolo
  F1*deltaPred_hat. The reference is a linearized score; selection/evaluation use
  actual full microF1 and an exact same-budget oracle, never oracle inference.

Metric: exact Unicode-casefold within-image word multisets, punctuation retained,
evaluation-only native256 ignore, no spatial IoU. Retain every care word and all
branch counts. F1 is global2TP/(2TP+FP+FN), TN0=TP/(TP+FP+FN). Good-image F1>=.8
is a separate diagnostic, not the regression objective. TRAIN125 includes fitted
images and is not official fullTRAIN1255. DeepSolo was fine-tuned on Total-Text
TRAIN and historical TEST has already been explored. Paper accuracy equivalence
remains unresolved; no benchmark-superiority claim follows.

## Files, resources and execution

Core features: `recognition/creative_context_signals.py`.
Neural probes: `recognition/creative_context_runtime.py`.
Error retrieval: `recognition/context_memory_router.py`.
Runner: `scripts/run_creative_context_router.py`.
Audit: `scripts/audit_creative_context_router.py`.
Tests: `tests/recognition/test_creative_context_router.py`.

Explicit profiles: `configs/approximate/creative_context_router_v1.json` and
`configs/approximate/creative_context_router_local_paths_v1.json`.

```bash
PYTHONPATH=artifacts/diagnostics/lavis-captioner-candidates-20260914/runtime-deps \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venvs/context-str-main-py311/bin/python -u -m scripts.run_creative_context_router \
  --mode approximate \
  --config configs/approximate/creative_context_router_v1.json \
  --paths configs/approximate/creative_context_router_local_paths_v1.json
```

Use --preflight to check source artifacts; --smoke uses a separate20-image run.
No installations/downloads. Actual PAPER GPU is RTX3090. Sequential frozen neural
models use FP32 vision batch1; BLIP2 Yes/No language batch2. Record source/code/
checkpoint hashes, original crop and intervention pixels, input shapes, state
identity, peak allocated/reserved VRAM, per-stage seconds and failures/OOM.
Peak8GB compatibility and end-to-end acceleration are not inferred from bypass.
Cached source Florence embeddings/captions/segmentation/DeepSolo costs remain
outside the new probe timers and must enter any future full-pipeline speed claim.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venvs/context-str-main-py311/bin/python -m scripts.audit_creative_context_router \
  --run /explicit/completed/run
```

All raw outputs, character spatial proxies, rendered-style decisions, donors,
features, portable regressors/memories, CV traces, frozen choices and per-split
metrics stay external. No weights, datasets or image copies enter Git.

## Qualification and evidence

Initial20-image smoke `0e25d1faba3b` completed all neural/CPU/fit/evaluation phases.
Independent audit PASS:504 routing points,1170 held-out predictions,187 independent
refits. Read-only attention capture gives exactly unchanged logits and all model
state hashes remain identical. Subsequent explicit capacity-edge abstention and
configuration validation preserve supported-word behavior; all44 focused tests
PASS before the final full run. No failed image is omitted from any denominator.

Scientific classifications: CREATIVE-CONTEXT-ROUTER-001 in
`docs/PAPER_AMBIGUITIES.md`. The two capacity-edge strings are data abstentions,
not dummy outputs or replacement predictions. No real OOM occurred in smoke.

Sources: [PARSeq](https://arxiv.org/abs/2207.06966),
[SceneFaith](https://arxiv.org/abs/2609.34781),
[E2STR](https://arxiv.org/abs/2311.13120).
These motivate hypotheses; they do not establish that these router versions work.
Final results are recorded below only after full completion and independent audit.

## Completed full run and cache replay

Final run:
`/extra1/bmartelli/context-str-artifacts/creative-context-router-v1/full/5ef7dbab9e1e`.
Plan SHA256:
`fc9bdb171875bcd42fb5ef0656d743b9ec95bbc76c7c30cbed5e7e545424d3b3`.
Result SHA256:
`cf3a959547d863c5999febd331eaf7503705ac1c518526149d6420bac10c5d42`.

The initial full run `1c80fadc34f6` completed all inference/evaluation, but its
first independent feature replay rejected observed time metadata: Python regex
pairs were tuples while the immutable JSON cache held lists. No numerical
feature differed. Preserve its `initial_audit_replay_difference.json` and all
original outputs. Time metadata is now explicitly JSON-compatible; a regression
test checks exact cache round trips.

The final run uses `--reuse-probes-from` with the explicit original full source.
Reuse requires identical configuration, populations, donors, qualified frozen
neural runtimes and pixel functions, complete stage receipts and per-file hashes.
New fitting must reproduce every model hash, CV result and VAL choice before
importing TEST probes. The final audit also verifies identical source input
hashes, byte-preserved neural data and every numerical result/decision relative
to the original attempt. No GPU inference, threshold change, new TEST tuning or
scientific tolerance relaxation resolves this serialization issue.

## Selected policy, TRAIN and TEST kept separate

| Images bypassing DeepSolo | Frozen VAL choice | TRAIN125 final F1 | TEST300 final F1 | Historical reference TEST F1 | Accepted TEST images with light F1>=.8 |
| --- | --- | ---: | ---: | ---: | ---: |
| 20% /60TEST | trees_swap | 78.753% | 80.027% | 80.174% | 47 /60 |
| 30% /90TEST | trees_swap | 77.778% | 79.982% | 79.835% | 63 /90 |
| 50% /150TEST | trees_render | 75.811% | 78.476% | 78.473% | 77 /150 |

Historical references are the already explored base_proxy_f1,
risk_visual_consensus and risk_visual_mass, respectively. They are not new TEST
model selection. All-DeepSolo F1 is78.624% TRAIN125 /80.357% TEST300. FullTRAIN125
includes fitted images; Fit80 and VAL45 scores remain separate in the CSV.

Per-method variants below are chosen using frozen VAL scores within each group,
never the most favorable TEST variant:

| Method | TEST20% F1 | TEST30% F1 | TEST50% F1 |
| --- | ---: | ---: | ---: |
| Context transplant | 80.027% | 79.982% | 78.343% |
| PARSeq internal contexts | 79.945% | 79.652% | 78.823% |
| Repeated glyph consistency | 80.220% | 79.845% | 78.360% |
| Structural constraints | 80.110% | 79.790% | 77.677% |
| Context-conditioned rendering | 79.594% | 79.750% | 78.476% |
| TRAIN error memory | 79.586% | 77.463% | 75.742% |
| Combined five measured feature groups | 78.890% | 79.139% | 77.827% |

The overall policy does not achieve a confirmed gain over the historical
references. At30%, its nominal+0.147pp interval is[-0.490,+0.764]pp; at50%,
rendering gives+0.003pp with[-0.788,+0.812]pp. The PARSeq-context family at50%
gives+0.350pp, interval[-0.416,+1.154]pp. It is a candidate for a new validation
experiment, not promoted over the pre-frozen overall choice. These are10,000
paired-image fixed-gate bootstrap intervals without multiplicity correction.

## Interpretation and signal coverage

- PARSeq contexts offer the most interesting50%-budget point estimate with
  relatively cheap additional decoding. Its standalone applicable-image quality
  AUROC is65.55%, so it does not reliably certify image F1>=.8 by itself.
- Repeated glyph comparisons are available for only1TRAIN image and9TEST images
  (23TEST pairs). The tiny20%-budget point gain is not evidence that glyph
  reasoning generalizes; adding sparse features can perturb a small tree model.
- The structural prototype applies to8TRAIN /6TEST images. Its English price/
  hours rules provide little coverage of heterogeneous Total-Text scenes.
- Rendering applies to99TRAIN /243TEST images,595/1412 words respectively.
  Its selected50% result is effectively equal to the historical best control.
  Its isolated applicable-image margin AUROC is51.92%, demonstrating weak
  discrimination under this restricted font/angle model.
- Context transplant has291TEST word probes and an applicable-image foreign
  target-support AUROC54.27%. Stable contextual answers need not be correct.
- Supervised error retrieval does not improve this80-example reference bank;
  its50% result is75.742%. This does not establish that larger/different memories
  cannot work. Combined feature models also do not improve the tested policy.
- Many new signals inspect already detected words. Missing-word evidence still
  comes mainly from previous coverage features. At50% the selected policy keeps
  only77/150 good80 images; perfect separation remains far away. The eligible
  same-budget oracle remains82.320% final F1, a3.845pp gap from the selected policy.

Applicable-only AUROC diagnostics use different subsets and are not a ranking
of final system utility. No image or care word is removed from final F1/TN0.
The repeated glyph and structure variants are restricted prototypes, not tests
of every possible implementation of those broader ideas.

## Final engineering/scientific verification

**45 focused tests PASS.** The full independent audit passes425 images,
504 routing points,9360 held-out predictions and197 independent cached refits.
The runner made558 tabular/memory fits. All neural weights remain unchanged.
There are3227 original predicted words,14418 attention-guided character windows,
412 context-swap word probes, and exactly two explicit capacity abstentions.
Model/choice/input identity across the cache fix is verified, not assumed.

Measured original neural-stage seconds: PARSeq22.32TRAIN /48.13TEST;
BLIP2 swap98.92TRAIN /238.79TEST. These exclude initial model load and cached
upstream/fallback costs. PARSeq uses1x3x32x128 FP32, peak allocated228,328,960
bytes/reserved245,366,784. BLIP2 uses224px vision batch1 and language batch2,
peak allocated16,050,999,296bytes/reserved16,204,693,504 on RTX3090. This does
not qualify an8GB deployment or demonstrate end-to-end acceleration.

`scripts/report_creative_context_router.py` exports all family/scalar metrics,
TRAIN/VAL/TEST counts, per-method VAL choices, paired intervals, evidence coverage,
applicable-only diagnostics and PNG/PDF plots. The existing YOLO analysis
environment supplies matplotlib, without installation:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venvs/context-str-yolo-py311/bin/python -m scripts.report_creative_context_router \
  --run /extra1/bmartelli/context-str-artifacts/creative-context-router-v1/full/5ef7dbab9e1e
```

The final external directory contains `report.md`, `comparison.csv`,
`analysis_summary.json`, `creative_context_comparison.png/.pdf`, complete raw
probes/features/models/CV/choices and `independent_audit.json`.
