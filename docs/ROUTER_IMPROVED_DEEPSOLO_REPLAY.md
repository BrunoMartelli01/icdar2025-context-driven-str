# Frozen routers with improved DeepSolo and two transcription metrics

**Approximate reproduction, completed on 2026-10-26.** This user-authorized
recalculation separates punctuation normalization from replacing the fallback.
It preserves every original router score and expert output. Paper Section3.5 /
Algorithm2 motivates fallback routing; these lexical metrics and confidence
profiles are research extensions, not strict reproduction or Table5 accuracy.
All fallback counts below describe **logical routing on cached outputs**, not
fresh online DeepSolo calls or measured latency. Percentages use every evaluated
image as the denominator.

## Why the previous numbers differ

The completed [router cause tests](ROUTER_CAUSE_TESTS.md) retained punctuation:
`coffee!` did not match `coffee`. The separate
[DeepSolo optimization study](DEEPSOLO_WORD_F1_STUDY.md) used Unicode casefold
followed by `isalnum`, on both OCR and GT. This removes punctuation, symbols and
whitespace, while preserving letters, accents and numbers. It does not split
one prediction into several words or remove symbol-only care annotations.
Empty normalized strings never match. Repeated words retain multiplicity.

Both metrics below sum TP/FP/FN over all images, retain native256 ignore rules,
exclude ignored predictions, retain invalid-envelope false positives, and use
lexical matching within each image without bbox IoU. Every TEST result includes
all300 images and all2,204 care words. `F1=2TP/(2TP+FP+FN)` and
`accuracy_TN0=TP/(TP+FP+FN)` are separate quantities.

## DeepSolo endpoint replay

All profiles use the qualified ViTAEv2-S+TextOCR checkpoint; the improvements
come from confidence filtering, a lower detection floor and, for fusion, a
larger second view. Selections were made in the earlier study using TRAIN125,
without selecting a new winner on TEST.

| Frozen fallback profile | Images sent to DeepSolo | Punctuation retained: TEST F1 % | Symbols removed: TEST F1 % | Symbols removed: TEST accuracy TN0 % |
| --- | ---: | ---: | ---: | ---: |
| Original native, score>=0.4 | 300/300 (100%) | 80.3571 | 81.7308 | 69.1057 |
| First improved profile, score>=0.4 / character confidence>=0.9 | 300/300 (100%) | 81.0181 | 82.3977 | 70.0647 |
| Global TRAIN word winner: native+large fusion | 300/300 (100%) | 81.1000 | 82.4516 | 70.1427 |
| Separate native-only TRAIN winner | 300/300 (100%) | 81.1848 | 82.6540 | 70.4362 |

The user's approximately82.3 result corresponds to the first improved82.3977%
endpoint. All five available improved profile/population aggregates replayed
exactly against the independently audited earlier reports. The global winner
remains fusion even though the separate native-only choice scores higher on TEST.

The original DeepSolo endpoint gains1.3736 F1 points from normalization alone.
Its TP rises1755→1785 with unchanged denominators. The light YOLO/PARSeq branch
gains68.8291→74.2581% from the same normalization, TP1496→1614. Therefore applying
normalization only to DeepSolo would misstate the router's relative performance.
This light-only endpoint sends0/300 (0%) images to DeepSolo; the original
DeepSolo normalization comparison sends300/300 (100%).
The improved native-only fallback gains a further0.9233 points over original
DeepSolo when both use the symbol-free metric.

## Frozen decisions with the improved fallback

The following values are final-system TEST micro F1 with symbols removed. Bypass
means using every saved YOLO/PARSeq prediction on the accepted image and DeepSolo
on every other image. No transcript is repaired in this replay. Counts, ranking,
score ties and eligible-image restrictions remain fixed from the saved routers.

| Images bypassing DeepSolo | Images sent to DeepSolo, every strategy in this row | Token router + native-only profile F1 % | Combined router + native-only profile F1 % | Combined router + global fusion profile F1 % | Native-profile GT oracle F1 % |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 /300 (0%) | 300/300 (100%) | 82.6540 | 82.6540 | 82.4516 | 82.6540 |
| 30 /300 (10%) | 270/300 (90%) | 82.7684 | 82.3055 | 82.1220 | 84.3735 |
| 60 /300 (20%) | 240/300 (80%) | 82.4615 | 81.9002 | 81.5666 | 84.8414 |
| 90 /300 (30%) | 210/300 (70%) | 82.0440 | 81.3422 | 81.1014 | 84.8414 |
| 150 /300 (50%) | 150/300 (50%) | 80.8248 | 80.8702 | 80.6202 | 84.2983 |

At50% bypass (150/300,50% sent to DeepSolo), the native-profile combined router has accuracy TN0=67.8841% and
F1=80.8702%, versus70.4362% /82.6540% for all native-profile DeepSolo. The F1
loss is1.7838 points, conditional95% paired RGB-cluster interval
[−2.8295, −0.7631]. The corresponding oracle remains3.4281 points above this
router. It is a GT-privileged diagnostic, not a deployable selector.

At10% bypass (270/300,90% sent to DeepSolo), the predeclared token arm gives82.7684%, +0.1144 points over the
native fallback; the supplementary95% interval is[−0.2601, +0.5239]. With global
fusion it gives82.6941%, +0.2425 points, interval[−0.1806, +0.7415]. These small
numerical gains do not establish an improvement. Intervals were added after the
main results for **every** saved router/profile/metric at the already evaluated
10% quota, without promoting a TEST winner. They are conditional on fixed
models/routes, with no multiple-comparison correction.

For a comparison that retains punctuation throughout, the historical combined
router changes78.4441→78.8839% when replacing the original fallback with the
native-only improved one. For symbol-free evaluation, the corresponding change
is80.3427→80.8702%; all four router values send150/300 (50%) images to DeepSolo.
Those two effects must not be conflated with the larger
change caused by switching the metric itself.

## TRAIN populations and training limitations

TRAIN and TEST remain separate in all outputs:

- Complete TRAIN1253: original and first improved confidence0.9 profiles are
  available from the original native outputs. Symbol-free all-DeepSolo F1 is
  82.4384% /83.1731%; combined-router OOF F1 at50% is79.2380% /79.6621%.
  All-DeepSolo sends1253/1253 (100%) images; both OOF routers and their50% oracles
  send627/1253 (50.04%), reflecting rounding of the accepted-image count.
  The corresponding same-budget oracles are83.2637% /83.6482%. Therefore the
  earlier finding that even the TRAIN50% oracle cannot beat all-DeepSolo applies
  to the punctuation-retaining metric; it does not transfer to this symbol-free
  evaluation. With punctuation retained, the improved confidence0.9 oracle is
  still82.1064%, below all-DeepSolo83.1502%.
- Shared TRAIN125: the lower-floor native and large-view outputs support all
  four fallback profiles on exactly the earlier optimization population. Native
  all-DeepSolo F1 is80.1959%, and global fusion80.2974%. These values are not
  substitutes for complete TRAIN1253 results; both send125/125 (100%) images
  to DeepSolo.
- Complete TEST300: all four fallback profiles are available.

There are no lower-floor or larger-view outputs for the other1,128 TRAIN images.
The replay reports this coverage explicitly rather than fabricating replacements.
All26 saved score sets are evaluated: six original arms plus20 cause-test trials.
TRAIN uses grouped OOF scores; TEST uses final fitted scores. The original
routers were trained to predict benefit against the older fallback under the
punctuation-retaining metric. This is a frozen-policy transfer evaluation;
training new targets against the improved fallback is a separate experiment.

## Reproducibility and completion evidence

Implementation: `scripts/replay_router_improved_deepsolo.py`.
Scientific and host paths use separate explicit approximate configurations:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venvs/context-str-main-py311/bin/python -u \
  -m scripts.replay_router_improved_deepsolo \
  --mode approximate \
  --config configs/approximate/router_improved_deepsolo_replay_v1.json \
  --paths configs/approximate/router_improved_deepsolo_replay_local_paths_v1.json \
  --output-root /absolute/fresh/external/output
```

This is CPU evaluation of cached predictions; no GPU inference, package
installation, foundation-model update or STRICT configuration change occurred.
The completed external run is
`/extra1/bmartelli/context-str-artifacts/router-improved-deepsolo-replay-v1/full/20261026`.
It preserves resolved configurations, input/code hashes, complete per-image
evaluation rows, accepted IDs, aggregates, paired intervals and exact rational
same-budget oracle certificates. `figures/improved_fallback.png` and `.pdf`
compare the metric changes and native-profile quota curves, with a source/hash
manifest. Results SHA256:
`50bb622923fe832e3fe37263170062c59fd9b52ea92e01a850d1cef390722ef2`.

Completion SUCCESS; independent count audit PASS with14,138 separate count/route
checks and five improved endpoint replays. The oracle is covered by an exhaustive
small-case regression test. All33 focused replay/DeepSolo/tabular-router tests
passed. An initial navigation check detected stale generated indices; rebuilding
the library and source catalog resolved it. Portable evidence:
[completion summary](../provenance/router_improved_deepsolo_20261026_completion.json).
Scientific classifications and remaining ambiguities are recorded as
`ROUTER-IMPROVED-DEEPSOLO-REPLAY-001` in [PAPER_AMBIGUITIES.md](PAPER_AMBIGUITIES.md).
