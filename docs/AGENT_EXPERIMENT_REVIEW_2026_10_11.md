# Agent experiment audit —11 October2026

Result class:**approximate reproduction / experimental extensions**.
This document analyzes existing results without launching training, changing
checkpoints, or promoting pipeline variants. Benefits of new proposals remain
**UNRESOLVED**. Existing choices retain classifications/references in
[PAPER_AMBIGUITIES.md](PAPER_AMBIGUITIES.md).

## Main finding

Controlled comparisons most strongly support **word kernels** for separation.
Several changes increase coverage/recover words but destroy previously correct
crops or add many false positives. Measure correct, lost, recovered, precision,
recall, Hmean together. More correct words alone does not establish overall gain.

`unet-word-research-v2` and `unet-crop-links-v1` are now complete: six V2 training
branches have50 recorded epochs;`crop_word`/`crop_links` have20. Their OCR reports
state `SUCCESS`. Pending-training statements in implementation guides are
historical and do not describe these newer artifacts. No duplicate training
is needed to inspect outcomes.

## Protocol and comparability

These comparisons use Total-Text TRAIN holdout125 images/983 care words,
split seed 202503; training uses the other1.130 images. Repeated experiments/
selections make this holdout non-independent. These are not official TEST,
Table3, or Table5 results.

OCR true positives require maximum one-to-one matching of exclusive original-RGB
bboxes, IoU >=0,50, and exact PARSeq transcription after `casefold`. Punctuation
stays in the primary metric. Given `TP` and `N` care crops:

- precision = `TP / N`;
- recall = `TP / 983`;
- Hmean = `2 * TP / (N + 983)`.

These are text+crop end-to-end metrics, not fixed-crop recognizer accuracy.
All983 words stay in denominators. Only wholly ignored instances are excluded
after localization; partly ignored boxes stay whole. Original protocols state
ignore-coordinate systems. Native geometry and RGB-projected support metrics
are not interchangeable.

V1/V2/crop-links use RTX3090 FP32 with separately loaded segmentation/PARSeq;
V2 batch 8, crop-training batch 4. Recorded hardware deviations do not qualify
RTX3070. Centers/refinements audits reuse exported maps and qualified CPU
PARSeq readings. CPU refinement verification reproduces historical GPU text
for all 1.037 control care crops; do not generalize this to different maps,
devices, or batches.

## Research V2: complete results

Current source:[comparison.json](/extra1/bmartelli/context-str-runs/unet-word-research-v2/comparison.json)
and [comparison.md](/extra1/bmartelli/context-str-runs/unet-word-research-v2/comparison.md).
[V2 guide](UNET_WORD_RESEARCH_V2.md) describes choices/earlier CPU qualification;
preliminary CPU rows must not replace this GPU run.

| Variant | Selected epoch | Native word F1 | Crop care | OCR /983 | OCR precision | OCR recall | OCR Hmean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen reference | 2 from source | 64,26% | 1.037 | 520 | 50,14% | 52,90% | 51,49% |
| Control continuation | 0 | 64,26% | 1.037 | 520 | 50,14% | 52,90% | 51,49% |
| Glyph isolated | 50 | 64,32% | 1.038 | 520 | 50,10% | 52,90% | 51,46% |
| Glyph staged | 7 | 64,26% | 1.037 | 520 | 50,14% | 52,90% | 51,49% |
| Threshold frozen | 0 | 64,26% | 1.037 | 520 | 50,14% | 52,90% | 51,49% |
| DB core | 14 | 64,34% | 1.050 | 522 | 49,71% | 53,10% | 51,35% |
| Links isolated | 33 | 5,84% | 20.200 | 490 | 2,43% | 49,85% | 4,63% |

DB core recovers five/loses three words: net+2 does not improve Hmean. Glyph
heads do not improve OCR. Continuation/threshold select the checkpoint before
additional training; another 50 epochs do not yield a preferable model under
the declared criterion.

Postprocessing instead uses **identical V1 embedding epoch 50 maps**, which are
neither control maps nor best embedding epoch 0 maps:

| Extraction from identical maps | Word F1 | Crop care | OCR /983 | Precision | Recall | Hmean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hard | 50,90% | 1.563 | 498 | 31,86% | 50,66% | 39,12% |
| Watershed | 61,24% | 1.104 | 500 | 45,29% | 50,86% | 47,92% |
| Soft embedding cost | 61,52% | 1.104 | 494 | 44,75% | 50,25% | 47,34% |

Watershed removes substantial hard-aggregation fragmentation; soft embedding cost
adds no OCR gain. All three preserve identical foreground. Do not attribute
this to segmentation improvement.

### Why pixel links fail here

Auditing all 125 image exports, excluding recorded `ignored_instance_ids`, counts
actual proposal-support area:

| Variant | Crop care | Median native area | Area = 1 pixel | Area <= 4 pixel |
| --- | ---: | ---: | ---: | ---: |
| Reference | 1.037 | 182 | 5 | 25 |
| Links isolated | 20.200 | 1 | 17.022 | 18.410 |
| DB core | 1.050 | 179 | 5 | 26 |

84,27% of links proposals are single pixels. Split-involved words:722/983 versus
83/983 control. These are **observed results**.

The [inference rule](../localize/unet_word_research_v2.py) requires both link
directions to have probability >=0,5; each isolated foreground pixel becomes its
own instance. The [loss](../segmentation/unet_word_research_v2.py) supervises
positive GT anchors; inference uses predicted foreground. Training loss falls
1,007 ->0,548, but epoch 50 emits35.732 crops, worse than selected epoch 33.
These are **implementation/log facts**.

Whether the dominant bottleneck is affinity calibration, foreground false
positives, reciprocal threshold, or frozen-feature capacity remains **UNRESOLVED**.
These hypotheses need isolation. Area filtering would remove fragments without
proving missing-letter reconstruction; never silently apply it to hide failure.
Logs do not support simply increasing epochs.

## Second pass trained on crops

Sources:[comparison.json](/extra1/bmartelli/context-str-runs/unet-crop-links-v1/comparison.json),
[comparison.md](/extra1/bmartelli/context-str-runs/unet-crop-links-v1/comparison.md),
[protocol](UNET_CROP_LINKS.md).

| Variant | RGB word F1 | Crop care | OCR /983 | Precision | Recall | Hmean | Gained / lost versus versusl |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Control | 63,62% | 1.054 | 521 | 49,43% | 53,00% | 51,15% | 0 / 0 |
| Crop word/kernel | 60,25% | 1.301 | 524 | 40,28% | 53,31% | 45,88% | 75 / 72 |
| Crop links | 6,43% | 18.851 | 476 | 2,53% | 48,42% | 4,80% | 63 / 108 |
| Reused global links V2 | 5,82% | 20.298 | 490 | 2,41% | 49,85% | 4,61% | 20 / 51 |

Word/kernel crop training increases geometric matches648→688, but OCR gains/
losses nearly cancel and precision drops sharply. Split-involved words rise
124→217. The second pass produces useful candidates, but wholesale base-crop
replacement is not a demonstrated solution. Links repeat fragmentation;
global-links reuses V2 rather than providing independent replication.

**Why520 and 521 are different baselines:**V2 constructs boxes from native
geometry with inverse letterbox; crop-links projects supports to RGB/evaluates
there. Counts/ignore give1.037 versus 1.054 care proposals. Do not attribute+1
to a new network: control checkpoint is identical. Compare each variant against
its own protocol's exported control.

Earlier [untrained recropping](UNET_CROP_REFINEMENT.md) already showed increased
“MOTORS” fragmentation in image147 smoke only; it was not quantitative holdout
improvement and must not be confused with the new run.

## Text+Kernel, centers, and conservative crops

[Complete multitask comparison](WORD_MULTITASK_RESULTS_V1.md) provides the
clearest kernel-addition causal control: word F1 58,96%(Text),62,13%(Text+Kernel),
58,41%(Text+Kernel+Center). Kernels reduce merged words293→171, increasing
splits48 ->95. Center model improves pixel metrics but worsens word metrics.
Instance separation is distinct from foreground prediction alone.

Later [V1 research](UNET_WORD_RESEARCH.md) reaches520 control OCR. Shared-decoder
glyph gives517, adaptive threshold 505. Learned top10 ranking improves429 ->438
on its own frozen set; not a directly comparable gain against uncapped520 control.

[Centers V2 report](../artifacts/center-guided-word-crops-v2/REPORT.md) compares
offset networks against a different425-OCR baseline. V2 improves V1 from 335 to
348 words, but loses152 base successes while recovering75. The
[loss diagnostic](../artifacts/diagnostics/center-guided-v2-word-loss-audit-v1/REPORT.md)
separates observable factors:

- 113/152 losses have no IoU >=0,50 box;39 have admissible geometry/wrong text.
  None is attributed to matching competition.
- Lost-word median annotated-support crop coverage falls96,8% ->71,1%; mean
  other-word contamination rises3,9% ->21,6%.
- Center error falls only 8,66 ->8,42 pixels; median word height13 pixels.
 189 words retain multiple supported centers;128 centers include pixels from
  multiple words.132 words lack any base-mask pixel.
- Incomplete/multiword crops can receive extremely high OCR confidence: FISHING
  becomes FISH, THE/FINEST become THEFINEST. Confidence does not certify completeness.

Conservative selection **has already been measured**, not newly proposed:

| Policy | Crop care | OCR /983 | Precision | Recall | Hmean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base centers experiment | 886 | 425 | 47,97% | 43,23% | 45,48% |
| Global V2 + expansion | 1.065 | 348 | 32,68% | 35,40% | 33,98% |
| Forbid merges/mixed; local changes only when confidence improves | 946 | 451 | 47,67% | 45,88% | 46,76% |

Third policy recovers34/loses8. Post-hoc reused-holdout evidence does not prove
generalization. Better vote reliability, background supervision, adaptive scales
are **ASSUMPTION** with **UNRESOLVED** benefit, not validated causal explanations.

## Resolution, recall, and frozen rules

[Resolution suite](TEXT_RECALL_RESOLUTION.md) is complete in external JSON.
`control512`/`tversky512` both select epoch 0;`control768` epoch 50;`tversky768`
epoch 10, threshold 0,90. Greater mean per-word coverage does not automatically
improve pixel recall/complete-word rate. Fixed Tversky768 mask reads443 words
with contours,358 with V2 offsets,525 with transferred kernels: instance grouping
strongly affects outcomes here.

[Refinement suite](TEXT_RECALL_REFINEMENTS.md), with
[machine-readable results](../artifacts/text-recall-refinements-v1-qualified/summary.json),
isolates other frozen-map rules:

| Source / rule | Crop care | OCR /983 | Precision | Recall | Hmean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Joint512 baseline | 1.037 | 520 | 50,14% | 52,90% | 51,49% |
| Joint512 seeded-only / soft fallback | 1.006 | 520 | 51,69% | 52,90% | 52,29% |
| Hybrid256 baseline | 1.143 | 525 | 45,93% | 53,41% | 49,39% |
| Hybrid256 seeded-only | 960 | 514 | 53,54% | 52,29% | 52,91% |
| Hybrid384 baseline | 1.213 | 528 | 43,53% | 53,71% | 48,09% |
| Hybrid384 bounded growth | 1.234 | 530 | 42,95% | 53,92% | 47,81% |

Joint512 filtering retains exactly520 correct words while removing31 wrong
proposals. Highest observed suite Hmean, Hybrid256 seeded-only, sacrifices11 words
relative to its baseline and uses two checkpoints. Grid384 adds only three
words over Hybrid256 and more false positives. This does not establish that
resolution increase alone solves the problem.

### GT crops:838/983, not confirmation of earlier93%

Reproducible diagnostic with identical PARSeq/983 GT bboxes gives fixed-identity
accuracy **838/983 =85,25%**. Earlier93% was user-reported, without enough protocol/
provenance in these artifacts to verify comparability. This does not establish
93% was wrong; it is not the value measured here.

Among463 Joint512-unread words,334 lack a predicted IoU >=0,50 bbox;262 of these
are correctly read from their GT bbox. Another129 have admissible boxes but
no exact text, and 63 are GT-box-readable. Total325 current failures are readable
with annotated crops, supporting coverage, purity, geometry work. GT bbox is a
diagnostic, not an inference candidate/absolute ceiling: some predictions correctly
read words whose GT rectangles are not recognized.

## Avoid duplication and bound the new experiment

Already implemented: kernel watershed/BFS, center heatmaps, offsets/voting/consensus,
hard/soft PAN-like embeddings, global/crop links, adaptive thresholds, DB raster core,
auxiliary glyphs, top10 ranking, Tversky loss/input 768, frozen/retrained recrops,
distance markers, bounded expansion, fallback filtering, conservative family
selection, unlearned PCA/Bezier rectification, erosion/reconstruction. Details:
[crop recovery](TOTALTEXT_CROP_RECOVERY_SUITE.md),[text recall crops](TEXT_RECALL_CROPS.md),
[center audit](../artifacts/diagnostics/center-guided-v2-word-loss-audit-v1/REPORT.md).
Variants must be identified as ablations/corrections, not wholly new methods.

The suite being prepared adds a **private centerline+radius head on frozen
baseline**, classified **ASSUMPTION**, not yet evaluated/no measured benefit.
Longitudinal lines differ from center-point heatmaps/offsets. Lines alone do
not define crop height: radius or explicit word-mask linkage is needed.
[Existing curved-GT construction](../segmentation/totaltext_curved_box_v4.py)
already uses centerlines to build polygons, but does not train a line-prediction
head. New targets must not overwrite originals. Precise choices/ambiguities
belong in the new suite documentation.

## Audit evidence and limitations

For every V2/crop-links branch,`run_manifest.json`/`epochs.jsonl` were read;
epoch count/final epoch, OCR `SUCCESS`, recorded `comparison -> ocr_comparison`/
`comparison -> validation` hash equality checked. Precision/recall/Hmean were
derived from OCR counts; recorded protocols inspected. Singleton audit uses
`foreground_area` from every V2 export after recorded validation ignore-ID
exclusions. Training/OCR were not rerun; this is not complete requalification of
every checkpoint, crop, image, or repository function.

SHA256 values of inspected files:

| File | SHA256 |
| --- | --- |
| `/extra1/bmartelli/context-str-runs/unet-word-research-v1/comparison.json` | `c6704de0e313a9f419f8d3aca7060a69e3f8f669e39b0cc4f397881d81a9f88d` |
| `/extra1/bmartelli/context-str-runs/unet-word-research-v2/comparison.json` | `cc737b83301bf5d6b7b2f31a59ee5745dee77359c016c2393d68fcc021b0b010` |
| `/extra1/bmartelli/context-str-runs/unet-word-research-v2/ocr_comparison.json` | `43bdc6f6374f4211585644b325d7542286bc39702503519eea9b1b46b73abdd6` |
| `/extra1/bmartelli/context-str-runs/unet-crop-links-v1/comparison.json` | `e680d21bb67a8a15a6c9ee9b0d2d13900b282b21406794befc61a7b680f8f8f5` |
| `/extra1/bmartelli/context-str-runs/unet-crop-links-v1/ocr_comparison.json` | `f7e998e19417284d386f792ab469eaf769d78230ebc8bf49d87c35a3ebdb90cd` |
| `/extra1/bmartelli/context-str-runs/totaltext-text-recall-resolution-v1/comparison.json` | `b7165e8fde959b9ddba2721219e0e06ef9fca8c82b9c8cebf53fdb41bd6f5632` |
| `artifacts/text-recall-refinements-v1-qualified/summary.json` | `c076c22b713d9854ac663acb4e5bb90a2319ce6324aeb2f71a8ab445924d0ddc` |
| `artifacts/center-guided-word-crops-v2/summary.json` | `f08981d713688389f91c7acfbc1bc454e3e8df6c58eb87acf481c73bc6629e67` |
| `artifacts/diagnostics/center-guided-v2-word-loss-audit-v1/policy_ablation.json` | `bc80bfd4ae3d612db4668ec577e92e7aa70551182dfb42eb78a2592083b581d4` |

## Priorities for next tests

1. Retain Text+Kernel/base crop as control. Separately measure segmentation,
   instance reconstruction, crop selection; report same-GT-identity gains/losses.
2. First qualify centerline/radius targets, pairing, coordinates, ignore, small/curved/
   nearby words, GT-free inference, strict warm start; then real smoke with allocated/
   reserved peaks/explicit OOM. Compare frozen baseline under identical protocol,
   without promising gains.
3. Compare conservative selection against existing measured rules, same population/
   cost/cap/qualified readings. Do not reward recoveries while excluding losses/FP.
4. Freeze targets, loss, thresholds, checkpoint criterion, recipes before subsequent
   evaluation. Use validation independent of this repeatedly inspected holdout
   and declared-seed replications before generalization claims; keep official
   TEST separate from hyperparameter search. Future results remain approximate
   reproduction until STRICT evidence/protocol are satisfied.
