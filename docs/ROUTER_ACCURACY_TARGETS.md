# Accuracy gaps and conditional U-Net requirements

Source-based numerical analysis,2026-10-16; **approximate reproduction / local
diagnostic**, not an official benchmark comparison. No new model inference or
threshold fitting. Target paper Table5 nominal78.4% R50 and79.8% ViTAEv2-S
are interpreted as recognition F1 based on `PAPER_TABLE5_SOURCE_AUDIT.md`;
their exact protocol remains unresolved.

All main rows below use TEST300 /2,204 care words, the same local exact-casefold
image-word multiset metric, punctuation retained, no bbox IoU and native256
ignore. Accepted-image budgets are descriptive149 for U-Net /150 for YOLO.
Differences are **numerical percentage-point gaps**, not validated benchmark
deficits. Changing the budget changes each router's accuracy.

| Version | Local TEST F1 | Gap to78.4 | Gap to79.8 |
|---|---:|---:|---:|
| U-Net + PARSeq only |53.43%|24.97pp|26.37pp|
| U-Net + approximate paper router + DeepSolo |66.79%|11.61pp|13.01pp|
| U-Net + original CLIP B16 + DeepSolo |68.99%|9.41pp|10.81pp|
| U-Net + DataComp-L14 + DeepSolo |69.49%|8.91pp|10.31pp|
| U-Net + DFN-H14 + DeepSolo |69.01%|9.39pp|10.79pp|
| U-Net + Long-CLIP-B + DeepSolo |68.49%|9.91pp|11.31pp|
| U-Net + Long-CLIP-L + DeepSolo |68.82%|9.58pp|10.98pp|
| U-Net + LLM2CLIP + DeepSolo |68.30%|10.10pp|11.50pp|
| YOLO + PARSeq only |68.83%|9.57pp|10.97pp|
| YOLO + approximate paper router + DeepSolo |73.05%|5.35pp|6.75pp|
| YOLO + original CLIP B16 + DeepSolo |73.45%|4.95pp|6.35pp|
| DeepSolo R50 on every image |75.74%|2.66pp|4.06pp|

This is not a comparison of each version's TEST-optimal threshold. The transferred
TRAIN half-coverage cuts give YOLO/CLIP72.90% F1 at189 acceptances and
YOLO/paper70.92% at197, as already recorded. More fallback may improve
accuracy; no new deployment cut is chosen from TEST labels.

## Can a router fix the current U-Net by itself?

An exact GT-only choice between every image's frozen U-Net/PARSeq and DeepSolo
branches maximizes dataset microF1 via the previously tested fractional optimizer.
Without a fallback budget constraint, the current U-Net branches reach an oracle
F1 **78.0052%**, using72 light-branch images and228 DeepSolo images. Thus
even perfect selection does not reach the nominal78.4 or79.8 targets under this
local protocol. This does not prove an official benchmark ceiling.

At exactly149 light-branch images, the oracle is **75.0919%**. It measures
remaining routing headroom at that budget; it is not a realizable GT-free gate.
For comparison, the previously frozen YOLO oracle is **79.9155%**, with135
YOLO and165 DeepSolo images. The current YOLO branch pair has sufficient
numerical headroom under this local metric, unlike the current U-Net pair.

## What should the U-Net branch achieve?

There is **no identifiable required pixel segmentation F1** from these data.
Pixel F1 does not determine instance splitting/merging, missing-word recall,
crop coverage, PARSeq errors or overlap of errors with DeepSolo. An OCR word
F1 requirement concerns the entire U-Net+localization+PARSeq branch.

For a concrete conditional calculation, freeze the original B16 gate's149
accepted images and all151 rejected images' DeepSolo outputs:

- Accepted images contain977 care GT words,978 predicted care words and619 TP.
  Accepted-branch word F1 is63.3248%.
- Frozen fallback contributes855 TP in1,091 predicted words.
- Combined system:2,069 predicted words,1,474 TP,2,204 GT; F1=68.9913%.

If the accepted branch still emits978 care words and the rejected branch remains
unchanged, then `F1_total=2*(855+TP_accepted)/(2204+2069)`.

| Nominal total-system target | Minimum accepted TP to exceed it | Gain over619 | Accepted precision | Accepted recall | Accepted word F1 |
|---|---:|---:|---:|---:|---:|
|78.4%|821 /978|+202|83.95%|84.03%|83.99%|
|79.8%|850 /978|+231|86.91%|87.00%|86.96%|

The integer counts yield actual total F1=78.4461% and79.8034%, respectively.
Targets are treated as exact displayed values; the authors' unrounded results
are unavailable. These are conditional requirements on the **currently accepted
subset**, not minimum all-image U-Net F1, pixel F1, or guaranteed accuracy.
Changing geometry, output count or gate decisions requires recalculation.
Missing readings can be caused by localization or PARSeq; this calculation does
not attribute202/231 errors exclusively to the segmentation architecture.

## Evidence

Reused frozen results:
`/extra1/bmartelli/context-str-artifacts/clip-specialized-comparison-v1/final/comparison.json`,
`/extra1/bmartelli/context-str-artifacts/yolo-paired-routers-v1/full/results.json`,
and completed425-image word branches. The pure U-Net53.43% value here uses
the strict-text word protocol and must not be mixed with prior symbol-stripped
or detector-only F1 figures near60%. The new Kernel-only/padding results are
TRAIN holdout125 only and cannot supply a TEST paper gap.

Machine-readable counts, numerical gaps, conditional requirements, both U-Net
oracles, source hashes and verification receipt are outside Git:
`/extra1/bmartelli/context-str-artifacts/router-accuracy-gap-analysis-20261016/`.
Scientific record: `ROUTER-ACCURACY-TARGETS-001`. No scientific source code,
historical predictions, thresholds, weights or environments were changed.
