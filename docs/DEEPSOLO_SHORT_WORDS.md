# DeepSolo minimum-three-character diagnostic

**Approximate reproduction.** User-requested reassessment of the completed
[DeepSolo study](DEEPSOLO_WORD_F1_STUDY.md) after excluding short words. This is
a new evaluated word population, with fixed saved predictions and previously
TRAIN-selected confidence/fusion policies. No threshold is refitted on TEST,
no model is loaded and no new GPU inference is performed.

## Exact rule and evidence

Apply Unicode casefold followed by `isalnum`, as in the preceding symbol-free
study. Keep strings with **at least three normalized characters**, independently
on GT and predictions. Numbers such as `123` remain eligible. Repeated words
remain separate instances. Original ignore flags and invalid-envelope handling
are preserved. A long incorrect prediction near an excluded short GT can still
be an FP: the diagnostic adds no GT-overlap neutralization.

This rule removes241 of2204 care GT entries, leaving1963. All300 TEST images
remain included; none has an empty filtered GT in this dataset. The earlier383
exclusion count concerned the **full** upstream word-spotting predicate, which
also applies numeric, punctuation and character restrictions. It is not the
number of short words alone.

Inspected the authoritative local paper sections4.1/4.2 and Table5, existing
word/geometry evaluators and the frozen upstream `include_in_dictionary` and
`MIN_LENGTH_CARE_WORD=3` at commit
`dbadae995035246bad3376c7a44c015c69e9b313`. The paper does not specify this
symmetric normalized-length protocol. Full official polygon, care, ignore and
transcription equivalence remains unresolved; scores below do not reproduce
the official83.6% result.

## TEST300 results

Word F1 uses per-image exact normalized multiset matching and micro counts:
`F1=2TP/(2TP+FP+FN)`. The second metric additionally requires bboxIoU>=0.5
and maximum one-to-one matching; it is not official polygon IoU>0.5.

| Fixed ViTAE+TextOCR profile | Original word F1 % | Length-filtered word F1 % | Filtered TP/FP/FN | DeepSolo images |
| --- | ---: | ---: | --- | --- |
| Ordinary native | 81.731 | 82.781 | 1625/338/338 | 300/300 (100%) |
| Native-only TRAIN winner | 82.654 | 83.674 | 1576/228/387 | 300/300 (100%) |
| Global TRAIN winner, two-view fusion | 82.452 | 83.980 | 1591/235/372 | 300/300 (100%) |

The native profile's filtered precision/recall are87.361%/80.285%. Fusion gives
87.130%/81.049%. Filtered localized word F1 is82.578% for ordinary native,
83.462% for the native profile and83.188% for fusion. Care predictions excluded
by length number201,212 and261 respectively.

DeepSolo coverage is300/300 images(100%) for every profile: this is replay of
saved DeepSolo outputs, with **zero new online inference calls**. It is not a
new latency measurement or a router avoiding DeepSolo. Fusion reuses both
previously exported resolutions. The numerical increase reflects the changed
evaluated population, with no change to model weights or raw OCR outputs.

## Reproduction and validation

Configuration: `configs/approximate/deepsolo_short_words_v1.json`.
Runner: `scripts/evaluate_deepsolo_short_words.py`.
Tests: `tests/scripts/test_deepsolo_short_words.py`.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venvs/context-str-main-py311/bin/python \
  -m scripts.evaluate_deepsolo_short_words --mode approximate \
  --config configs/approximate/deepsolo_short_words_v1.json \
  --run SOURCE_FUSION_RUN --output-root FRESH_EXTERNAL_OUTPUT
```

The configuration pins source prediction, deployment-profile and original-report
SHA256 values. The resolved configuration records code hashes, policies and
the length threshold. Every unfiltered aggregate must first replay its historical
report exactly. Independent Hungarian matching verifies1800 image/metric cases
(three profiles,300 images,two metrics).

All23 focused tests passed, including seven new boundary/denominator cases.
These cover length after symbol removal, inclusive three-character admission,
digits, Unicode letters, duplicates, ignore flags, invalid predictions and
empty filtered GT without suppressing remaining FP. The full300-image CPU
reassessment passed with per-image predictions, filtered GT and removed counts.

External evidence:
`/extra1/bmartelli/context-str-artifacts/deepsolo-short-words-20261026-v1/`.
Portable summary: `provenance/deepsolo_short_words_v1_completion.json`.
Scientific decisions: `DEEPSOLO-SHORT-WORDS-20261026` in
`docs/PAPER_AMBIGUITIES.md`. Historical metrics and STRICT defaults are preserved.
