# Frozen word-crop refinements

**Approximate reproduction / experimental extensions**, not strict paper Table5.
Paper sections3.2-3.3 and Algorithm1 specify segmentation/component localization;
the extra kernels and interventions below are separately classified in TRF-001.
Historical code, checkpoints, datasets and result directories are preserved.

## Frozen experiment

Reuse official TRAIN1130/held-out125, seed202503, all983 care words and qualified
v4 original RGB/annotations. No TEST or checkpoint/threshold reselection occurs.
The source control is research-v1 Text+Kernel selected epoch2, RGB512/native256.
Hybrid text is selected Tversky768 epoch10/threshold.90, kernels from the original
Text+Kernel epoch61. Reference threshold remains.65 and kernel threshold.50.
All three checkpoints, source contracts, exports, dataset and split are verified.

Sources: joint `reference512`, transferred `hybrid256`, transferred `hybrid384`.
Native384 uses the saved original sigmoid384 text output, not upsampled256 text.
Predicted256 kernel probabilities are bilinearly interpolated with
`align_corners=False`. Native384 target/ignore/owners are regenerated from
full-resolution qualified data using the existing validation transform.
Inference canvas depends only on RGB dimensions. Original RGB boxes/GT and
one-to-one IoU>=.50 plus exact Unicode casefold matching remain common to all
arms. Punctuation stays in the primary metric; P/S removal is secondary.
Wholly ignored instances are excluded after proposal generation; partial boxes
stay complete. No area filter, padding, top10 or OCR-driven proposal selection.
Native topology on384 is a separate raster diagnostic, not directly identical
to256 topology. No missing/failed image is excluded from a denominator.

For each source, compare five inference-only recipes:

| Recipe | Change from the same-source baseline |
| --- | --- |
| `baseline` | Existing kernel marker-controlled watershed |
| `refined` | Existing shared-kernel fragment unions; bridge text.50, maximum gap3 at256, kernel merge enabled, no center input |
| `bounded_growth` | Preserve every high-threshold instance ID and flood text>=.50 within Euclidean radius3 at256; no seed-unreachable component added |
| `soft_fallback` | Retain seeded instances and fallback instances whose maximum predicted kernel>=.35 |
| `seeded_only` | Explicit recall-sacrificing diagnostic: remove every instance without kernel>=.50 |

Distances scale by native_size/256; the existing refinement's integer distance
rounds upward, growth's radius remains floating point. Parameters are frozen in
`configs/approximate/text_recall_refinements_v1.json` before any GT/OCR scoring.
This is not PSENet reproduction, adaptive parameter selection or a deployable
gain established on independent data. The fallback rules do not alter NN maps.

Separate GT-box diagnostic: recognize each of the983 original GT bbox crops,
keeping its annotated identity. GT crops never seed/filter prediction recipes.
Report fixed-pair GT-crop accuracy and, for every predicted arm, failures with
no viable IoU.50 crop, viable crop/wrong text, or one-to-one competition. Each
failure is further classified by whether its own GT crop is recognized exactly.
The GT oracle row is never reported as achieved detector performance.

## Running

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venvs/context-str-main-py311/bin/python -u \
  -m scripts.evaluate_text_recall_refinements --mode approximate \
  --reference-run /extra1/bmartelli/context-str-runs/unet-word-research-v1/control \
  --text-run /extra1/bmartelli/context-str-runs/totaltext-text-recall-resolution-v1/branches/tversky768 \
  --kernel-run /extra1/bmartelli/context-str-runs/totaltext-word-multitask-v1/text_kernel \
  --output-root artifacts/text-recall-refinements-v1-qualified --action all \
  --parseq-source /extra1/bmartelli/context-str-artifacts/parseq/parseq-1902db043c029a7e03a3818c616c06600af574be \
  --parseq-source-archive /extra1/bmartelli/context-str-artifacts/parseq/parseq-1902db043c029a7e03a3818c616c06600af574be.tar.gz \
  --parseq-checkpoint /extra1/bmartelli/context-str-artifacts/parseq/parseq-bb5792a6.pt \
  --reuse-run-root artifacts/text-recall-parseq-crops-v1 \
  --reuse-run-root artifacts/text-recall-learned-crops-v1
```

Use `--action preflight`, `geometry` or `recognize` to separate stages.
Only CPU FP32/PARSeq AR=True/refine1/batch8 is supported by this qualified run.
This is a HARDWARE_DEVIATION, with no segmentation inference/GPU qualification.
All reference-family RGB crop hashes run afresh on CPU, including any shared
across other arms. Remaining exact RGB crops may reuse complete hash-qualified
results; the existing16-crop replay remains mandatory. No downloads/installations.
Completed contracts/exports/OCR hashes are verified before reuse. Output must be
isolated; changed scientific configuration/source requires a new directory.

Results retain resolved config and source/artifact hashes, integer instance PNGs,
original RGB crop hashes, proposal audit trails, raw OCR, per-image matches,
all15 inference rows, separate oracle metrics and per-word GT-crop diagnosis.

## Qualification status

The initial real geometry attempt in `artifacts/text-recall-refinements-v1`
failed explicitly: the frozen native256 PNG writer rejects384 shapes. No OCR or
aggregate result was produced. An isolated adapter now reuses that writer for256
and exports384 integer IDs losslessly as uint16 with conflict checking. The
failed protocol/artifacts remain; the corrected run uses the separate directory
above. The added regression checks preserve ID1023 and reject changed caches.

Completed qualification: **104 focused/related tests passed**. All15 inference
arms and the separate GT diagnostic completed125 images/983 words. Real CPU
PARSeq recognized5,655 new crops and reused1,873 qualified exact RGB crops;
7,528 unique crops total,478.91 seconds,16/16 exact cache replays, no OOM.
Fresh CPU reference agrees with historical CUDA text on **1,037/1,037 care
crops**, zero raw-text differences. All1,727 unique reference-family crop hashes
executed afresh on CPU.125 original reference masks/native metrics and125
canonical hybrid masks/crop receipts replay exactly.

The independent auditor recomputed all2,000 image/arm rows, replayed18,049 RGB
crops, verified3,875 export hashes and all7,528 OCR record hashes, and confirmed
identical historical GT/annotation hashes. Receipt:
`provenance/text_recall_refinements_v1_results.json`; frozen audit source:
`provenance/text_recall_refinements_independent_audit.py.txt`.
No source model, historical result, checkpoint or strict configuration changed.
The independently running research-v2 training suite is not restarted or
represented as completed by this experiment.

## Complete observed results

Counts below require original-box IoU>=.50 **and** exact casefold PARSeq text.
All rows divide recall by983 GT words; precision divides by care proposals.
Hmean=2TP/(care proposals+983). These are micro word-level end-to-end metrics,
not pixel F1, fixed-pair recognizer accuracy or strict paper recognition accuracy.

| Source | Recipe | Care proposals | Correct /983 | Precision | Recall | Hmean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Joint Text+Kernel512 | baseline | 1037 | 520 | 50.14% | 52.90% | 51.49% |
| Joint Text+Kernel512 | refined | 1035 | 520 | 50.24% | 52.90% | 51.54% |
| Joint Text+Kernel512 | bounded_growth | 1044 | 521 | 49.90% | 53.00% | 51.41% |
| Joint Text+Kernel512 | soft_fallback | 1006 | 520 | 51.69% | 52.90% | 52.29% |
| Joint Text+Kernel512 | seeded_only | 1006 | 520 | 51.69% | 52.90% | 52.29% |
| Hybrid256 | baseline | 1143 | 525 | 45.93% | 53.41% | 49.39% |
| Hybrid256 | refined | 1127 | 523 | 46.41% | 53.20% | 49.57% |
| Hybrid256 | bounded_growth | 1173 | 524 | 44.67% | 53.31% | 48.61% |
| Hybrid256 | soft_fallback | 968 | 514 | 53.10% | 52.29% | 52.69% |
| Hybrid256 | seeded_only | 960 | 514 | 53.54% | 52.29% | 52.91% |
| Hybrid384 | baseline | 1213 | 528 | 43.53% | 53.71% | 48.09% |
| Hybrid384 | refined | 1196 | 525 | 43.90% | 53.41% | 48.19% |
| Hybrid384 | bounded_growth | 1234 | 530 | 42.95% | 53.92% | 47.81% |
| Hybrid384 | soft_fallback | 990 | 517 | 52.22% | 52.59% | 52.41% |
| Hybrid384 | seeded_only | 979 | 516 | 52.71% | 52.49% | 52.60% |

The modest single-model improvement preserves the exact same520 correct GT
identities while removing31 incorrect care proposals (+0.80 Hmean percentage
points). Soft and hard fallback filters happen to agree on this control; that
does not establish equivalence on new data. The highest observed Hmean52.91%
uses transferred two-checkpoint Hybrid256 maps and removes11 words correctly
read by that source's baseline. It is a precision/recall tradeoff, not better
foreground recall or a qualified lightweight single-model improvement.

Native384 baseline gains7 and loses4 correct GT identities versus Hybrid256
baseline: net+3. Its word localization rises657→667 but exact OCR525→528 and
proposal count1143→1213, lowering Hmean. Both read78/248 small GT words, using
the declared proxy of bbox shorter side<=8 pixels on the canonical256 letterbox.
Bounded384 growth reaches530 correct but Hmean47.81%; more geometric matches
are insufficient to establish better crops. Refinement offers at most a small
precision effect here and does not increase the number of correct words.

## GT-box diagnosis and remaining limitation

Fixed-identity GT bbox PARSeq reads **838/983 =85.25%**. This is a diagnostic
reference, not an absolute ceiling: some predicted crops read a word correctly
when its GT rectangle does not. These crops are never used to seed prediction.

| Failure group | Joint512 baseline | Of these, GT crop reads correctly | Hybrid256 baseline | Of these, GT crop reads correctly |
| --- | ---: | ---: | ---: | ---: |
| No predicted bbox with IoU>=.50 | 334 | 262 | 325 | 247 |
| Viable bbox but no exact OCR candidate | 129 | 63 | 133 | 72 |

Thus325 currently incorrect reference words, and319 incorrect Hybrid256 words,
are read correctly from their own GT rectangle. This supports improving coverage
and crop purity before expecting a broad recognizer fine-tuning gain. GT bbox
compatibility alone does not prove complete text, absence of neighbors or correct
orientation. Mask/grid enlargement alone did not resolve this limitation.

Full merge/split rates, localization geometry, raw texts, per-image gains/losses
and all per-word diagnoses are retained in `summary.json`, `paired_changes.json`
and `gt_crop_diagnostic.json`. No variant is silently promoted. Additional
reused-validation tuning requires a new declared experiment; independent
generalization and the pending research-v2 interventions remain unresolved.

## Offline review

```bash
.venvs/context-str-main-py311/bin/python -m scripts.package_text_recall_refinements_review \
  --root artifacts/text-recall-refinements-v1-qualified \
  --output artifacts/text-recall-refinements-review-v1.zip
```

The completed ZIP has12 diagnostic examples including gains, losses, typical
and dense scenes; green/red boxes agree with exact matching. The displayed
variant has the highest observed Hmean among15 declared arms, a post-hoc visual
choice rather than independent model selection. All aggregate results and
GT-diagnostic counts accompany the images. CRC/content hashes and the real
visual preview pass. Existing ZIPs and outputs are never overwritten.

Download from a local machine:

```bash
scp bmartelli@paper.dinfo.unifi.it:/extra1/bmartelli/projects/icdar2025-context-driven-str/artifacts/text-recall-refinements-review-v1.zip .
```
