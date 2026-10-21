# Frozen YOLO/PARSeq evaluation with don't-care neutralization

This is an **approximate reproduction / local word-recovery diagnostic**, using
the existing external YOLO experiment. It is not strict reproduction of the
paper, Table5 recognition accuracy, or a complete official benchmark score.
Scientific decisions are recorded under DCE-001 in `PAPER_AMBIGUITIES.md`.

## Run

From the repository root:

```bash
bash scripts/test_yolo_evaluation.sh
```

The command rescales nothing and runs no models: it rescores already frozen
TEST detector/OCR outputs on CPU. It uses the existing isolated Python runtime
for polygon intersections, without installing packages or loading DeepSolo.
Local paths belong to
`configs/approximate/yolo_dont_care_word_evaluation_local_paths_v1.json`.
Resolved configuration, sources, package versions, code/model identities and
per-image outputs are retained under `/tmp/yolo-dont-care-word-evaluation-v1/`.
An identical rerun verifies/reuses immutable artifacts; a changed configuration
or implementation requires a new `--output` directory.

For a smoke test:

```bash
bash scripts/test_yolo_evaluation.sh --limit 1 --output /tmp/yolo-dont-care-word-evaluation-qualified-smoke-v1
```

## What changes

1. Keep the original detector boxes, RGB crop hashes and PARSeq readings.
2. Apply the original GT-blind joint-confidence selection at0.60 and
   same-reading NMS at IoU>0.5. The analytic pre-OCR bound is
   `sqrt(detector_score)>=0.60`, nominally detector confidence0.36.
3. For each final predicted rectangle, compute intersection with every original
   don't-care GT polygon, divided by the **prediction's** area.
4. If that fraction is **strictly greater than0.5 for any single polygon**,
   neutralize the prediction in the metrics. Exactly0.5 does not qualify.
   Do not replace polygons with their enclosing boxes or union ignore regions.
5. Perform unchanged care-word evaluation on the remaining predictions.

Ignored GT regions contribute no FN. Neutralized predictions contribute neither
TP nor FP. If a rectangle covers both an ignored and a care region, ignore
neutralization takes precedence. No NMS is rerun after neutralization, and no
suppressed prediction is resurrected. GT is never provided to cropping,
recognition, confidence selection or NMS. The number of generated outputs and
actual OCR work therefore stay unchanged; fewer outputs enter the score.

Native Polygon3 implements the inspected benchmark rule. Original annotation
vertices are retained, including vertices outside the image: per-vertex
clamping could alter the polygon's in-image shape. Every intersection and ignore
decision is checked with independent Shapely geometry. Shapely `make_valid` is
allowed only inside this logged independent audit; primary vertices are never
repaired. Invalid/zero-area annotations are explicitly logged. Total-Text has
one single-point ignored region, which neutralizes nothing.

## Care matching and denominators

- ICDAR13, ICDAR2015 and Total-Text: exact whole-word equality after Unicode
  casefold and removal of all Unicode punctuation/symbols, plus axis bbox
  IoU>=0.5; maximum one-to-one matching. Every care word remains in the
  denominator; a normalized-empty care reading cannot match a prediction.
- CTW1500: count whitespace-delimited word occurrences inside care line
  annotations after the same normalization. Require exact token equality and
  at least0.5 of a prediction's area inside the care line axis bbox. Repetitions
  retain separate occurrences, with maximum one-to-one matching. Two
  normalized-empty care lines contribute zero words and remain documented in
  the prior qualified word audit. There are no fabricated GT word boxes.
- Empty normalized readings are rejected by the existing final selector.
  The standalone CTW scorer would count an empty prediction as one FP.
- Micro precision=`TP/predicted_words`, recall=`TP/GT_words`,
  F1=Hmean=`2*TP/(predicted_words+GT_words)`. A zero denominator returns0.
  No failed image is omitted from the dataset.

The inspected official evaluator also uses polygon IoU for care matching,
different matching/normalization, optional word-spotting/lexicon rules, and
full-line outputs on CTW. Those differences are unresolved here. Only
don't-care neutralization is aligned with its rule. Our values must not be
advertised as official end-to-end Hmean or compared directly as a benchmark rank.

## Full TEST results

Fixed joint confidence0.60; analytic detector prefilter (nominal0.36).

| Dataset | Images | Correct / care words | Outputs produced | Neutralized | Outputs evaluated | Precision | Recall | Previous F1/Hmean | Updated F1/Hmean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ICDAR13 |233|862 /1,095|1,038|0|1,038|83.04%|78.72%|80.83%|80.83%|
| ICDAR2015 |500|1,401 /2,077|2,336|541|1,795|78.05%|67.45%|63.49%|72.37%|
| Total-Text |300|1,597 /2,204|2,205|155|2,050|77.90%|72.46%|72.44%|75.08%|
| CTW1500 word proxy |500|3,508 /4,546|5,245|507|4,738|74.04%|77.17%|71.66%|75.57%|

No correct words are lost at the primary setting. The gain comes entirely from
neutralizing previously counted FPs. ICDAR13's acquired Task1 annotations have
no don't-care regions. Ignored-region counts are 0 on ICDAR13, 3,153 on
ICDAR2015, 343 on Total-Text and 388 on CTW.
All 34,310 original proposals and all 10,824 selected outputs remain unchanged.

The nominal0.36 cutoff retains12,242 proposals eligible for OCR and all baseline
final IDs. This is a count derived from frozen readings, not a new inference run
or a measured speedup. GT-dependent neutralization cannot be used as a deployed
prefilter because ignore polygons are evaluation annotations.

The previous95-cutoff grid is retained as an **exploratory TEST** comparison.
The largest tested cutoff preserving every produced output is0.36 in each
dataset. Highest updated TEST F1 occurs at0.48 on ICDAR13 (81.47%), at the
baseline plateau on ICDAR2015/Total-Text, and0.38 on CTW (75.61%,3,493 correct,
15 fewer than the baseline). These maxima do not change the common selection
rule and are not independent validation of new thresholds.

## Qualification

- 55 focused tests pass, including the preceding CTW tests.
- Real one-image-per-dataset smoke and full1,533-image TEST rescoring pass.
- 147,168 per-image cutoff controls are checked; identical selected sets reuse
  previously checked scores. 21,080 distinct before/after scores agree with
  independent Unicode/rectangle code and SciPy bipartite matching.
- All145,083 polygon intersections and threshold decisions agree between
  native Polygon3 and Shapely. Runtime: Python3.8.10, Polygon3 3.0.9.1,
  Shapely2.0.0; compiled-extension hashes are in `independent_audit.json`.
- Every baseline selected ID and non-CTW baseline word metric agrees with its
  frozen source. All48,000 earlier CTW controls agree before neutralization.
- Frozen care/ignore annotations are verified against physical GT files. For
  ignore evaluation,8 Total-Text and5 CTW ignored polygons retain original
  out-of-frame vertices rather than canonical per-vertex clipping.
- No new crops, inference, image modification, GPU model allocation or package
  installation. Original experiments and strict configuration are preserved.

Artifacts: `results.json`, `comparison.csv`, `resolved_configuration.json`,
`polygon_request.json`, `polygon_response.json`, `independent_audit.json`,
`code_review_audit.json`, and `evaluation/<dataset>__<sample>.json` containing
original/evaluated/neutralized IDs, GT polygons, matching evidence and scores.
Failures are explicit in `failure.json`; worker failures propagate to the caller.
