# Research status and directions: masks, instances, crops

2026-10-21. **Approximate reproduction / exploratory saved-output analysis.**
No new training/neural inference. Other research results are read from completed
runs and linked to their audit SHA256, not inferred from launch notes. In
particular, word-instance suite is now SUCCESS/complete,2026-10-21T01:35UTC,
although its guide still describes launch-time status. STRICT/production
configurations stay unchanged.

## Operational conclusion

Best-supported direction:**generate local alternatives and select a complete,
compatible crop set under an explicit budget**. V3 improves795→870 words without
changing U-Net/PARSeq. Automatically deforming every crop is negative in two
OCR-supervision experiments and RGB Mask-Align proxy. This does not rule out
alignment/OCR supervision in other architectures; it identifies current priorities.

Text+Kernel is an important additional reference. Text/Kernel are text regions/
word nuclei, not separate letter/word networks. Gap full1 reaches1311/2204 versus
its M/K control1293. These are not a controlled ranking against V3: resolution
512/native 256 versus 256/native 128, letterbox/stretch geometry, targets, training,
budgets, ignore differ. Gap's measured gain is18 words over its own control;
1311–870 is not an isolated Gap effect.

## Parallel research update

| Research thread/population | Verified internal comparison | Interpretation |
| --- | --- | --- |
| V3, TEST300/2204 | Selector795→870; F1 37,06→39,60%; +92/-17 words | Actual precision/recall gain, still exploratory |
| V3, same TEST | Cap removed795→904; F1 37,06→36,96% | Real recovery, more false positives |
| M/K and Gap, TEST300/2204 |1293→1311; F1 53,96→54,31% | Small gain; Gap decoder adds 32,6M parameters to frozen58,2M |
| Mask-Align, same Gap versusl | Original RGB1311, bilinear1300, rotated1307, mask1170 | Mask-guided RGB warp worsens; aligned-feature OCR was not trained/tested |
| Six local modules, TRAIN holdout125/983 | Prompt/affinity-maximin select off; geometry522, geometry+OCR520, baseline520 |20real epochs per arm, audit PASS; no OCR benefit of new loss over matched versusl |
| YOLO/DeepSolo, TEST300, IoU-free metric | Coverage640 F1 77,32→77,60%;960/1280 worsen | Uncertain gate benefit: F1 interval[-0,46,+1,03]points. Not comparable to localized-crop F1 |

Diagnostic Gap10 reads1324 words but F1 54,25% is below full1. It does not replace
predeclared epoch 1 after observing TEST. Earlier Gap+2 on holdout125 was confined
to one image; refit1255 includes that group in training and is not a fresh holdout
evaluation.

Mask-Align genuinely emitted64x8x32/64x16x64 features and checked gradients;
its PARSeq reading instead concerns resampled RGB. Do not confuse these findings.
Predicted-mask support coverage proves neither real-character coverage nor
causes of regression.

YOLO gate still accepts only 41/150 fully correct images at50% budget, even with
coverage: high confidence does not certify completeness. Budget policies rank
a batch, unlike transferred scalar thresholds; their cost exceeds U-Net alone.

Gap1 maximum geometric matching covers1612 of 2204 words;1311 are also read
exactly.592 annotations still lack one-to-one geometric matches. Localization-
reading gap includes incomplete/contaminated crops; not pure PARSeq error or a
per-word disjoint causal taxonomy.

## Additional analysis performed

Paired per-image bootstrap,10000 draws, seed 20261021,95% percentile intervals.
Models/policies fixed; intervals exclude training variability, parameter search,
TEST reuse.

| Comparison with own versusl | F1 delta, points | Conditional95% interval | Images with more/fewer correct words |
| --- | ---: | --- | ---: |
| V3 selector | +2,54 | [+1,41,+3,69] |60/14 |
| V3 uncapped | -0,10 | [-1,73,+1,75] |34/0 |
| V3 learned refiner | -1,26 | [-2,22,-0,29] |28/46 |
| Gap full1 versus M/K | +0,34 | [-0,09,+0,85] |14/2 |
| Mask-Align RGB versus original | -5,84 | [-7,12,-4,58] |12/104 |

V3 gain is not driven by one image: largest per-image net gain5. Per-image counts
differ from 92 word gains/17 losses; both can occur within one image. Gap adds
correct words, but conditional-bootstrap F1 improvement remains compatible
with zero. Lower TRAIN loss alone cannot justify greater capacity.

### Crop complementarity: potential, not a new result

Saved readings reconstruct maximum one-to-one matching, replaying all 300 images:

| Candidate set | Correctly matched words |
| --- | ---: |
| Baseline cap10 |795 |
| All original components |904 |
| Current split/merge selector |870 |
| Original components + selector output union, oracle |996 |
| Complete local pool, oracle |1032 |
| Local pool + all original components, oracle |1141 |

996 indicates complementarity between discarded components/local transformations.
No current selector achieves996 or1141. GT matching omits some alternative-group
exclusivity constraints: these are pool potentials, not validated feasible
configurations. Simply summing both systems' gains is invalid.

A further descriptive comparison finds116 words read by V3+selector but not
Gap1,754 shared,557 Gap1-only. GT/order verified identical. GT-identity union1427
requires an oracle over two differently priced/protocol systems; it does not
propose two U-Nets as a new reference.

### Previously unquantified cost

Selector uses PARSeq readings for alternatives. **2190 emitted crops do not mean
2190 crops requiring recognition**: pool5537 proposals,5165 unique RGB crops,
versus 2086 baseline. Cold-start reading count is **2,48 times** larger; this does
not imply equal latency multiplication. Reused historical caches do not guarantee
savings on new images. This motivates cheap preselection before PARSeq and
equal-budget quality reporting.

## Promising directions, in order

1. **Transfer discrete selector to Text+Kernel and measure Gap contribution.**
   2x2: M/K versus M/K+Gap1, original assembly versus selected alternatives.
   Same samples, PARSeq, matching, ignore, budget within pairs; TRAIN/calibration-only
   choices. Reuse existing letterbox generators: native 128/stretch generation
   is not directly compatible with 256/letterbox maps. Transfer is unperformed;
   V3 gain75 does not predict its outcome. Keep cheaper M/K control rather than
   assuming an extra Gap decoder is necessary. Before selector training, measure
   TRAIN pool potential: missing candidate words require better generation.
   Gap full1255 already saw the old holdout; new validation needs sealed data/
   roles unused by that model.
2. **Adaptive budget/pre-recognizer selection.** Pool beyond-cap components with
   a few local alternatives. Learn TRAIN crop quality from RGB/mask/provenance
   without unread-crop OCR; run PARSeq on a restricted budget and compare full
   selector. Optimize alternative groups/conflicts using truncated, multiword,
   duplicate negatives, not only independent proposal correctness.996/1141
   motivate testing, not expected results. Measure F1/recall versus unique OCR
   crops/actual time, and original-success gains/losses. Coverage/confidence are
   possible signals, not completeness certificates.
3. **Later research only: OCR on aligned features.** Mask-Align enables testing
   the same small trained OCR head on axial versus Mask-Align features, same
   frozen backbone, capacity, data, budget, selection. This is the previously
   identified AdaSpot next step, not an existing result. The head must learn
   feature-domain inputs; frozen PARSeq on warped RGB does not measure that
   potential. Leave gating/deep heads/joint training for later experiments to
   isolate the changed factor.

Word-level calibration is motivated by
[On Calibration of Scene-Text Recognition Models](https://arxiv.org/abs/2012.12643),
which studies recognizer overconfidence. Calibrating reading probability does
not recover missing characters/unproposed words; crop quality/completeness needs
a separate objective.
[Mask TextSpotter v3](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560681.pdf)
and [FOTS](https://openaccess.thecvf.com/content_cvpr_2018/html/Liu_FOTS_Fast_Oriented_CVPR_2018_paper.html)
support segmentation proposals/aligned features for trained recognition, not
our frozen RGB warp efficacy.
[PSENet](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Shape_Robust_Text_Detection_With_Progressive_Scale_Expansion_Network_CVPR_2019_paper.html)
motivates separated kernels/expansion, already partly represented by M/K and not
new local novelty.
[More recent SemiETS, CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/html/Luo_SemiETS_Integrating_Spatial_and_Content_Consistencies_for_Semi-Supervised_End-to-end_Text_CVPR_2025_paper.html)
studies spatial/content consistency/reliable pseudo-labels: a direction for
unannotated data/domain transfer, not the first need of this already annotated
pool. Links motivate hypotheses, not evidence of the proposed combinations' gains.

## Lower priorities and validation

Without a newly isolated factor, do not repeat continuous OCR-loss refiners, local
prompts, maximin on identical maps, general RGB warping, global YOLO resolution
increase, or fresh Kernel-only networks. Existing controls are negative/modest;
this does not prove principles impossible. Do not rename existing tests as new
directions.

Before adding capacity, freeze word-level evaluation with separate data roles
and test on data unused for choice search; repeat selector across seeds. Current
bootstrap replaces neither. Two identical local TRAIN/TEST image files
(212↔578,81↔656) are already recorded: primary300/2204 stays separate from
sensitivity298/2187. Gain persists there, but historically explored298 still is
not an untouched benchmark.

## Evidence and reproducibility

Main sources:

- [V3: complete comparison](</extra1/bmartelli/context-str-runs/mask-crop-three-tests-20261021-v1/comparison.md>).
- [Gap fullTRAIN](ADASpot_GAP_FULLTRAIN.md) and [Mask-Align](ADASpot_MASK_ALIGN.md).
- [Six local modules: complete run](</extra1/bmartelli/context-str-runs/word-instance-extensions-20261021-v1/comparison.md>) and `independent_audit.json`.
- [YOLO/coverage: comparison/uncertainty](YOLO_TEXT_COVERAGE_RESOLUTION.md).

New analysis:`artifacts/diagnostics/research-synthesis-20261021-v1/`.
`results/bootstrap.json`,`results/candidate_oracles.json`,
`results/cross_family_overlap.json`,`candidate_workload.json`,
`source_verification.json`.319 source-file SHA256 values and links between
results/audits from four other research threads verified. Numerical self-tests/
original-count replay PASS; no original modified.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venvs/context-str-main-py311/bin/python \
  artifacts/diagnostics/research-synthesis-20261021-v1/analyze.py \
  --mode approximate --config configs/approximate/research_synthesis_20261021_v1.json \
  --output /path/to/new/report
```

Classifications:`REPOSITORY_SPECIFIED` for reused protocols/sources;
`ASSUMPTION` for conditional bootstrap/descriptive oracles;
`UNRESOLVED` for transfer, training variance, cross-family causality, feature-OCR
benefit. Research-priority ranking is a data-motivated judgment, not an achieved
scientific result.
