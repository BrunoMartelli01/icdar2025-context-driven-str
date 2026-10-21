# CTW1500: word recovery inside annotated text lines

**Approximate reproduction / word recovery diagnostic.** This evaluates the
existing frozen YOLO/PARSeq TEST predictions as words. It is a separate protocol
version; historical full-line scores and source artifacts remain auditable.
Scientific record: CWR-001 in `docs/PAPER_AMBIGUITIES.md`.

```bash
bash scripts/test_ctw_words.sh
```

The command runs on CPU without model inference, installation or downloads.
`--limit 1 --output /tmp/ctw1500-word-recovery-audit-smoke-fresh` runs a real smoke test.
Paths and output directory are configured in the separate approximate local
profile. Outputs include resolved provenance, per-image selected IDs/tokens/
matching edges, `results.json`, `comparison.csv` and `independent_audit.json`.
The current defaults use the audit v2 approximate profiles and write
`/tmp/ctw1500-word-recovery-audit-v2/`. Historical v1 results are retained.

## Evaluation rules

1. Normalize all care line transcriptions and frozen predictions with Unicode
   casefold and removal of punctuation/symbol categories. Split on whitespace.
   Removing punctuation does not invent word boundaries (`A-B` becomes `ab`).
2. Each GT token is a separate occurrence retaining the annotated line region.
   `GO GO` contributes two words. Do not invent GT boxes for individual words.
3. Require exact normalized token equality and at least 0.5 of the prediction
   rectangle area inside the GT line's original axis bbox. This reuses the prior
   line-fragment diagnostic's support rule. It is not word-box IoU: a small
   word box can legitimately cover only a fraction of a complete line.
4. Use maximum one-to-one bipartite matching. Every predicted token and GT token
   can contribute at most one match, even across overlapping line boxes. Keep
   repeated word occurrences; extra predictions are false positives.
5. Ignored GT lines contribute zero target words, with their count reported.
   Predictions over ignored regions remain FP, as in the source experiment.
   Care lines that normalize to no tokens contribute zero words and their count
   is reported. A normalized-empty prediction remains one FP. No failed sample
   can be dropped silently; validation/metric errors stop the experiment.
6. Micro precision is correct/predicted word units, recall is correct/GT tokens,
   and word recovery F1 equals Hmean: `2TP/(2TP+FP+FN)`. An additional image-wide
   multiset count without geometry is saved as a distinct diagnostic.

The complete source has 500 TEST images, 2,687 care lines and 388 ignored lines.
After normalization, two care lines are empty and the others contain **4,546
word occurrences**. This denominator is derived from source transcriptions,
never tuned to match a reported performance.

## Predictions and controls

Keep YOLO original640/direct crops, qualified PARSeq, joint threshold0.60 and
same-reading NMS0.5. The audit compares all95 detector floors0.05..0.99, step0.01,
and a separate analytic bound, on frozen readings. The analytic predicate is
`sqrt(detector_score) >= joint_threshold`, because OCR quality cannot exceed1.
Its nominal cutoff is joint-threshold squared,0.36 at joint0.60. The direct
square-root comparison also retains representable floating-point ties that a
comparison against the rounded square might reject. Verify exact equality of
the analytic method's selected IDs against the unfiltered final selection and
the source baseline. Scalar floors are separate measured controls.
GT is used only to evaluate already frozen predictions; it does not choose
crops, transcriptions, grouping or filtering.

This cached analysis does not claim an actual reduction of model forwards in
the original run. Altered OCR batch composition needs inference qualification.
All TEST controls are exploratory; no threshold is fitted or made a deployment
default using the new word scores.

## Complete TEST results

All 500 source images and 12,594 original crops are included. No failed sample
is omitted. Joint0.60 yields the following word-occurrence scores:

| YOLO prefilter | Eligible crops | Correct words /4,546 | Outputs | Precision | Recall | Recovery F1/Hmean |
|---|---:|---:|---:|---:|---:|---:|
| 0.05 | 12,594 | 3,508 | 5,245 | 66.88% | 77.17% | 71.66% |
| 0.36 | 5,691 | 3,508 | 5,245 | 66.88% | 77.17% | 71.66% |
| 0.40 | 5,426 | 3,466 | 5,121 | 67.68% | 76.24% | 71.71% |
| 0.50 | 4,788 | 3,289 | 4,675 | 70.35% | 72.35% | 71.34% |
| 0.60 | 4,083 | 3,006 | 4,050 | 74.22% | 66.12% | 69.94% |
| 0.70 | 3,066 | 2,497 | 3,060 | 81.60% | 54.93% | 65.66% |

The reference has 3,508 TP, 1,737 FP and 1,038 FN. Its image-wide lexical
multiset diagnostic without spatial support gives 3,530 matches; the primary
score above requires line-region support. The earlier full-line metric remains
1,106/2,687 correct lines and27.89% Hmean under its separate bbox-IoU protocol.
The new result changes the evaluated unit and support rule, not predictions.

Qualification: 14 focused unit tests and a real single-image smoke pass.
Independent Unicode normalization, rectangle support and SciPy bipartite
matching verify all3,000 per-image controls. Safe floors preserve every frozen
selected candidate ID. Results: `/tmp/ctw1500-word-recovery-v1/results.json`.

## Code audit and complete threshold grid

The audited run confirms all3,000 historical per-image controls: counts, selected
IDs, eligible matching edges and matched GT word indices are unchanged. Every
physical care/ignore transcription and canonical polygon/bbox matches cached
GT. Every OCR candidate matches its detector slot, native confidence and crop
shape. The parser clips16 raw annotation polygons before source bookkeeping;
this is recorded separately from the source's post-parser clipping counter.

Two objectives give different choices:

| Objective | Detector cutoff | Eligible crops | Correct words | Precision | Recall | Recovery F1/Hmean |
|---|---:|---:|---:|---:|---:|---:|
| Preserve the reference outputs using a GT-independent bound | nominal0.36 | 5,691 | 3,508 | 66.88% | 77.17% | 71.66% |
| Highest F1 observed in the TEST grid | 0.41 | 5,343 | 3,448 | 68.03% | 75.85% | 71.73% |

0.36 is the largest tested scalar cutoff with exactly the same output IDs across
all500 images. The F1-selected0.41 saves348 additional crops but loses60 correct
words, for a0.071 percentage-point F1 gain. This TEST-selected maximum does not
establish generalization or replace the analytic primary method. The bound
changes whenever the final joint threshold changes;0.36 is not a free parameter.

Audit fixes: guard floating-point boundary ties with the direct bound, validate
the declared metric semantics, reject duplicate candidate identities and
noninteger counts, and verify cached GT/proposals against physical sources.
These fixes do not alter the measured historical results. Original detector,
recognizer, strict configuration and model inference are unchanged.

Qualification: **26 focused tests**, real smoke, all500 TEST images and
**48,000 independently verified word controls**. Independent Unicode/rectangle
code and SciPy matching agree. No failed sample is omitted. Audit artifacts:
`/tmp/ctw1500-word-recovery-audit-v2/results.json`, `independent_audit.json` and
`code_review_audit.json`.

The reference contains13 predicted tokens eligible for more than one GT line
and26 GT word occurrences repeated within their lines;15 repeated occurrences
are matched. One-to-one assignment caps their counts, but exact word positions
remain unknown. These measured cases reinforce the proxy qualification below.

## Limits

This measures **line-supported word recovery**, not official CTW full-line
spotting Hmean or precise word localization. A matched word may be in the wrong
position within its line. Repeated words and duplicate boxes cannot be fully
disambiguated without word geometry. Axis bboxes may include background around
curved lines. These constraints also mean the new score cannot be directly
compared with official polygon scores or treated as a change in model quality.

The authoritative paper sections3.5/4.1 describe PARSeq and recognition accuracy
on ICDAR13/ICDAR15/Total-Text; CTW and this token evaluation are an explicitly
user-requested extension, outside strict paper reproduction.
