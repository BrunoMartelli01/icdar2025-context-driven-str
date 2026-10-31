# Symbol-free comparison of word-slot U-Net, PARSeq and DeepSolo

This user-directed follow-up changes text comparison for the completed
[word-slot study](WORD_SLOT_METHODS.md). It is an **approximate reproduction /
metric reassessment**, with unchanged expert weights and saved RGB inference.
The previous punctuation-retaining result remains historical evidence.

## Comparison rule

[word_slot_text.py](../recognition/word_slot_text.py) implements the explicit
`alphanumeric_casefold` policy: Unicode casefold, then keep only Unicode letters
and numbers using `isalnum`. Delete punctuation, symbols and whitespace on both
the predicted text and GT. Thus `FISHERMAN’S` matches `fishermans`, `B&B` matches
`bb`, and `CO-OP` matches `coop`. Accents and Unicode letters remain; no accent
removal or compatibility/NFKC folding is introduced.

Raw OCR and annotation strings are preserved. Empty normalized strings cannot
match; symbol-only GT and predictions remain in their original denominators.
Original-frame bbox IoU>=0.5, integer pixel-edge truncation, one-to-one matching,
all-output counting and the care/ignore protocol remain unchanged. This is not
an official Total-Text polygon evaluation or a resolved paper metric.

Matching, threshold fitting, router quality labels, text-agreement features and
replacement equivalence use the same policy. Other features retain raw text,
geometry and native confidences. Existing OCR/segmentation training is unchanged;
this study does not claim training with a modified charset.

## Separate normalization and calibration effects

[reassess_word_slot_alphanumeric.py](../scripts/reassess_word_slot_alphanumeric.py)
first scores the historical thresholds and routing decisions with symbols
ignored. It then refits the same declared threshold grids and image-group OOF
routers using TRAIN calibration only. Primary student selection still maximizes
word-router OOF calibration F1; descriptive TEST ranking does not select a model.
The [explicit configuration](../configs/approximate/word_slot_alphanumeric_v1.json)
pins the source protocol, original metric/selection files and prediction-hash
manifest. Derived analysis links to read-only source predictions and checkpoints;
analysis outputs, refitted decisions and metrics use a separate directory.

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m scripts.reassess_word_slot_alphanumeric --mode approximate --config configs/approximate/word_slot_alphanumeric_v1.json --output FRESH_OUTPUT
```

No model inference, GPU execution, dependency installation or expert retraining
is required. The `casefold` policy retains the earlier behavior for historical
configurations; no STRICT configuration or historical report is overwritten.
Frozen serialized routers obtain the comparison rule from their resolved router
configuration through [word_slot_routing.py](../recognition/word_slot_routing.py).

## Measured outcome, 2026-10-25

TEST still contains300 images and2204 care words.55 GT strings lose symbols;
none becomes empty. Calibration has802 care words with no removed GT symbols.
One of983 reused evaluation care entries becomes empty and remains counted.

| Endpoint | Earlier punctuation-retaining F1 % | Symbol-free F1 % | DeepSolo images/total, both comparisons |
| --- | ---: | ---: | ---: |
| Frozen DeepSolo | 78.836 | 80.190 |300/300 (100%) |
| Pixel-PIL U-Net control | 57.227 | 61.006 |0/300 (0%) |
| Recognition alignment, descriptive new standalone best | 57.200 | 61.136 |0/300 (0%) |
| Historical primary word gate, decisions frozen | 77.935 | 79.265 |300/300 (100%) |
| Primary word gate, recalibrated | 77.935 | 79.793 |300/300 (100%) |

Both primary selections choose `dense_matching`, whose new standalone F1 is
60.656% with82 exclusive GT matches and601 DeepSolo-only matches. Its refitted
word router recovers11 DeepSolo errors, loses13 correct words and adds19FP.
The difference from normalized DeepSolo is-0.396 percentage points, with paired
image-bootstrap95% interval[-0.877,+0.073]. The interval includes0; this screen
does not establish a reliable gain. The image gate reaches78.367% with242 logical
DeepSolo calls; fixed-half reaches75.857% with150 calls. Word routing uses both
experts on all300 images.

The secondary calibration-only DeepSolo confidence filter reaches80.338% F1;
it removes175FP while losing112TP and recovers no new words. It is not a
specialist improvement. The complete report retains all15 student/control rows,
the fixed-policy normalization diagnostic, refitted policies and per-word output.

One other predeclared arm, the `content_strip` word router, descriptively reaches
80.325% F1 with DeepSolo300/300 (100%):6 recovered words,3 lost correct and3 fewerFP. Its+0.135-point interval
[-0.135,+0.416] includes0. Its OOF calibration F1 was below dense matching, so it
was not the primary selected method. This small TEST point gain is below the
score-filter control and is not promoted as a confirmed improvement.

## Target paper comparison and mandatory routing cost

The authoritative [paper](../papers/2503.15639v2.pdf), Table5, reports79.8% for
its TotalText ViTAEv2-S pipeline, labeled recognition accuracy. Our higher
observed numbers exceed that reported number nominally. Earlier negative
conclusions concern primary calibration selection and the frozen DeepSolo
comparator; they must not be read as saying every result is below the paper's
reported number.

| Endpoint, symbol-free comparison | Our word F1 % | DeepSolo images/total | DeepSolo % | Nominal difference from reported79.8, points |
| --- | ---: | ---: | ---: | ---: |
| Word router, content strip, descriptive best | 80.325 |300/300 |100.00 |+0.525 |
| Image router, complementary specialist, descriptive best | 79.882 |277/300 |92.33 |+0.082 |
| Primary word router, dense matching | 79.793 |300/300 |100.00 |-0.007 |
| Primary image router, dense matching | 78.367 |242/300 |80.67 |-1.433 |
| Primary fixed-half image router, dense matching | 75.857 |150/300 |50.00 |-3.943 |
| Frozen DeepSolo baseline | 80.190 |300/300 |100.00 |+0.390 |

Every router score is reported with its DeepSolo image count, total and
percentage. Counts are independently reconstructed from the saved per-image
decisions. Image-router counts are logical selective decisions over cached
predictions; word routing requires both experts for every image. These are not
measured online latency or FLOP savings. The best observed image policy bypasses
DeepSolo on23/300 images (7.67%). Neither descriptive best is the primary
OOF-calibration-selected student.

The paper does not provide a specific TotalText fallback-image count. Its
ICDAR13/ICDAR15 counts and approximate compute reduction cannot supply that
missing TotalText count. Exact equivalence between the paper's recognition
accuracy and our bbox/alphanumeric F1, including normalization and care/ignore
handling, remains UNRESOLVED (TABLE5-SOURCE-METRIC-AUDIT-001). Therefore the
nominal numeric excess is not yet an independently established improvement
under the identical official protocol.

Audited reporting artifacts are under
`/extra1/bmartelli/context-str-runs/word-slot-alphanumeric-20261025-v1/router_reporting_v1/`.
Portable reporting audit:
[word_slot_router_reporting_v1.json](../provenance/word_slot_router_reporting_v1.json).

## Verification and limitations

All81 focused/regression tests pass, including explicit router image-count
reporting, symbol-only denominators,
normalization/IoU conjunction, repeated hypotheses, unchanged raw strings,
quality/merge agreement and configuration mismatch. The original default metric
reproduces8400 saved image/method results, and the historical serialized router
reproduces12750 image/word decisions. The normalized report independently audits
33900 cases with Hungarian matching; the new serialized router checks12750
cases with exact readings, boxes, ordering, expert selection and decisions.

An initial qualification rejected whole-dict equality because scikit-learn's
`expit` and the NumPy `logaddexp` sigmoid differ by at most1.11e-16 in float64.
The final audit permits absolute1e-12 only for `router_probability`, while all
scientific outputs/actions remain exact and missing/nonfinite values fail.
The first failure, its diagnostic and source snapshot remain archived; the
complete analysis is repeated in a fresh output directory.

Full results: `/extra1/bmartelli/context-str-runs/word-slot-alphanumeric-20261025-v1/full_v2/`.
Portable evidence: [word_slot_alphanumeric_v1_completion.json](../provenance/word_slot_alphanumeric_v1_completion.json).
Tests: [test_word_slot_alphanumeric.py](../tests/recognition/test_word_slot_alphanumeric.py).
The historical TEST exploration, inherited prior/calibration exposure, one seed,
known cross-split RGB duplicates and unresolved TEST399 label remain limitations.
No model or routing policy is promoted. Decisions are classified in
`WORD-SLOT-ALPHANUMERIC-001` in [PAPER_AMBIGUITIES.md](PAPER_AMBIGUITIES.md).
