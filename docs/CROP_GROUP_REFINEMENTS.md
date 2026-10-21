# Crop-group refinement diagnostic

This is an **approximate reproduction** experiment on frozen Total-Text crop,
PARSeq, and probability artifacts. It does not modify the paper's localization
algorithm, strict configuration, checkpoints, or recognizer. The paper's section
3.3 and Algorithm 1 specify padded connected-component crops; section 3.5 uses
PARSeq-S. They do not specify cross-threshold grouping or the refinements below.

## What changed

1. **Reference for choosing a crop within a group.** The previous score compared
   each crop's soft-mask mass with one geometrically chosen crop. If that crop
   omitted a letter, it received perfect support by definition. The new score
   compares each crop with the coordinate-wise inner-quartile envelope of all
   *core* boxes in its group. Shared multiword parents do not expand this
   reference. The score remains the geometric mean of this support and the
   unchanged balanced PARSeq character confidence. No text, GT, or dictionary
   chooses the reference.
2. **Confidence guard for a crop switch.** Compare the selected robust-reference
   crop with the original selected crop. Accept the switch only if the balanced
   PARSeq character confidence does not decrease. If mixing the two assignments
   would reuse one original crop in two groups, revert the colliding groups to
   their original choices. This is a zero-margin comparison of two existing
   scores, with no vocabulary or GT-dependent rule. All retained selections
   receive the robust-reference score for the subsequent duplicate check.
3. **Second-stage output grouping.** After one crop is chosen per original group,
   two outputs are treated as alternate hypotheses only when their boxes have
   IoU at least 0.9 and their normalized readings are identical or one character
   apart (for words of length at least four). The higher-scoring output remains.
   This is intentionally conservative because two real words can have
   overlapping boxes. It does not change the original core groups or their
   one-per-group oracle.

All three choices are `ASSUMPTION`, isolated in
`configs/approximate/crop_group_refinements_v1.json`. The original selector is
retained as a control. No model is loaded or rerun. The experiment saves all
selected outputs, score components, rejections, per-image matches, resolved
configuration, hashes, and result class outside the repository.

## Measured frozen results

Care words are counted after Unicode casefold and removal of punctuation and
symbols, with axis-aligned bbox IoU >= 0.5 and one-to-one matching. All GT words
remain in the denominator. These are local diagnostic metrics, **not** official
Total-Text polygon Hmean.

| Split | Method | Correct words | Outputs | Available group oracle |
|---|---|---:|---:|---:|
| TRAIN held-out 125 (983 GT) | frozen baseline | 732 | 7,522 | 781 |
| TRAIN held-out 125 | robust reference | 740 | 7,522 | 781 |
| TRAIN held-out 125 | robust + conservative output grouping | 740 | 7,456 | 781 |
| TRAIN held-out 125 | confidence guard + output grouping | 738 | 7,462 | 781 |
| TEST 300 (2,204 GT) | frozen min-side-3 baseline | 1,722 | 16,846 | 1,823 |
| TEST 300 | robust reference | 1,743 | 16,846 | 1,823 |
| TEST 300 | robust + conservative output grouping | 1,743 | 16,659 | 1,823 |
| TEST 300 | confidence guard + output grouping | **1,745** | **16,678** | 1,823 |

On TEST, the robust score recovers 50 baseline-missed words and loses 29
baseline-correct words, for a net gain of 21. Conservative output grouping
removes 187 outputs without changing the 1,743 correct-word count. Even the
smaller output set, 16,659, is far above the 2,204 annotated words; this
experiment does not solve group admission or final spotting precision.

The confidence guard recovers 31 baseline-missed TEST words and loses 8 that
the baseline had correct, a net gain of 23. On TRAIN it recovers 10 and loses
4, a net gain of 6. This is a lower-regret alternative to the unrestricted
robust selector, though the latter scores two more TRAIN words. The guarded
result still emits 16,678 TEST outputs from 19,212 original core groups. No
automatic near-word-count grouping method has been validated by this run.

An all-core-box envelope scored 728/983 on TRAIN, versus 740 for the inner
quartile. With the same one-edit rule, lowering the identical-reading box-IoU
bound to 0.75 retained 740/983 TRAIN words from 7,255 outputs, but retained
only 1,742/2,204 TEST words from 16,162 outputs: it discarded one correct
reading whose larger competing box fell just below the matching IoU. A 0.5
bound retained 1,722 TEST words from 15,158 outputs. The 0.9 bound avoids
these observed losses; the complete exploratory sweep is saved in
`/tmp/totaltext-crop-group-refinements-v1/output_merge_probe.json`.
A median mask-probability admission gate with baseline and
two-crop rescue yielded 719/983 from 5,522 outputs. Those exploratory controls
are not the selected primary method. The quartile and conservative near-identity
rules are structural hypotheses, not proven transfer-safe parameters. The TEST
run is exploratory because earlier experiments inspected this same split.

We also tried the robust selector on the previously frozen proposal-lineage
groups, after the same min-side-3 crop filtering. On TEST, lineage grouping
reduced the original 19,212 groups to 16,365. Its former selector found 1,546
correct words; the robust selector found 1,597, still below 1,743 with the
original groups. Adding the frozen cross-window footprint links reduced the
groups to 15,364 and found 1,552 correct with the robust selector. This probe
is recorded in `/tmp/totaltext-crop-group-refinements-v1/structural_selector_probe.json`.
It demonstrates that more merging requires a selector that handles conflicting
readings within larger groups; the lineage variants are not adopted here.
An additional no-model vote over the selected original-group readings found
1,558 correct TEST words from 14,086 outputs with lineage groups, and 1,478
from 13,156 with cross-window footprints. The exploratory record is
`/tmp/totaltext-crop-group-refinements-v2/lineage_vote_probe.json`; agreement
among noisy hypotheses does not yet make these larger groups reliable.

## Reproduction

Run focused tests:

```bash
.venvs/context-str-main-py311/bin/python -m pytest -q tests/recognition/test_crop_group_refinements.py
```

The CPU runner accepts explicit frozen run, manifest and output paths:

```bash
PYTHONPATH=. .venvs/context-str-main-py311/bin/python -m scripts.evaluate_crop_group_refinements \
  --run /tmp/crop-group-word-selection/0d7113261cc3 \
  --manifest artifacts/totaltext-curved-word-boxes-v4/manifests/total_text_train.yaml \
  --output /tmp/totaltext-crop-group-refinements-v2/train-full

PYTHONPATH=. .venvs/context-str-main-py311/bin/python -m scripts.evaluate_crop_group_refinements \
  --run /tmp/totaltext-minside3-group-test \
  --manifest /extra1/bmartelli/datasets/context-str/scene-segmentation-v1/manifests/total_text_test.yaml \
  --output /tmp/totaltext-crop-group-refinements-v2/test-full
```

The runner freezes all selection shards before opening evaluation GT. Results
are in `/tmp/totaltext-crop-group-refinements-v2/{train-full,test-full}/results.json`.
