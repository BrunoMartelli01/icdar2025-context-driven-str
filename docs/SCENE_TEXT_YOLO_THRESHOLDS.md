# Cached YOLO/PARSeq confidence controls and CTW1500 failures

**Approximate reproduction; exploratory TEST analysis.** This analyzes the
completed 1,233-image RTX3090 transfer experiment, preserving all proposal
geometry, OCR readings, normalization, care/ignore handling and matching. No
model is loaded and no new crop is recognized. Scientific record: YTS-001.

```bash
bash scripts/test_scene_text_yolo_thresholds.sh
```

The command reads the frozen GPU outputs using the existing YOLO environment
and writes `/tmp/scene-text-yolo-thresholds-v1/results.json` and `comparison.csv`.
Paths live in the separate approximate path profile. `--output` permits a fresh
external directory for changed analysis code. Immutable per-image controls,
candidate scores and failure diagnostics are retained. Mismatched source code,
configuration, runtime identity, OCR shards or annotation bytes fail explicitly.

## Why 25,992 crops

One direct crop was recognized for each YOLO proposal surviving its initial
confidence floor0.05 and proposal NMS0.7. This recall-oriented proposal set was
saved so later filters could be tested without model inference. These counts
refer to all TEST images combined, before final confidence filtering:

| Dataset | Images | Recognized crops | Final output at joint0.60 |
|---|---:|---:|---:|
| ICDAR13 | 233 | 2,574 | 1,038 |
| ICDAR2015 | 500 | 10,824 | 2,336 |
| CTW1500 | 500 | 12,594 | 5,245 |
| Total | 1,233 | 25,992 | 8,619 |

For the existing joint score `sqrt(detector_score * OCR_quality)`,
`OCR_quality <=1`. Thus any crop with `sqrt(detector_score) < threshold` cannot
pass the final joint filter. At threshold0.60, only1,116/2,871/5,691 proposals
remain eligible before PARSeq: **9,678 combined**, a62.77% reduction. This bound
retains every currently admitted candidate under the frozen readings. It is an
evaluation of a possible pre-OCR filter, not a claim that the original run used
fewer recognizer forwards. Rerunning altered batches still requires execution
qualification; the current run retained low-score proposals for the controls.

## CTW1500 uses lines; this recognizer predicts words

The qualified PARSeq source config has94 printable non-space characters,
`remove_whitespace: true`, and max label length25. The adapter creates this
qualified model without changing the vocabulary. Its decoded output cannot
include spaces. The comparison metric preserves whitespace, so exact matching
of a multiword GT line is structurally unavailable to a single word output.

The baseline joint0.60 result partitions as:

| Care CTW GT | Units | Exact correct |
|---|---:|---:|
| One normalized whitespace token | 1,599 | 1,106 |
| Multiple normalized tokens | 1,086 | 0 |
| Empty after punctuation/symbol removal | 2 | 0 |

The2,687 care lines contain4,546 normalized whitespace tokens; this token count
is descriptive and supplies no word boxes or official word metric. Of the1,086
missed multiword lines, **1,007** have at least one correctly recognized proper
token fragment inside the GT axis bbox, and **858** have at least two distinct
fragments. A fragment counts if at least half its predicted box lies within the
line bbox. GT is used only in this diagnostic, never to combine/choose outputs.

Examples retained in the machine-readable report:

| Sample | Complete GT line | Separate correctly read fragments |
|---|---|---|
| ctw1500:1131 | IS DELICIOUS | IS, DELICIOUS |
| ctw1500:1131 | IS BACK | IS, BACK |
| ctw1500:1449 | HARBOR POINT | HARBOR, POINT |
| ctw1500:1449 | WINTON WOODS | WINTON, WOODS |
| ctw1500:1194 | CELEBRATING QUEEN ELIZABETH II | CELEBRATING, QUEEN, ELIZABETH, II |

The69.17% single-token recall is a diagnostic subset, not official CTW Hmean.
The whole-line Hmean also penalizes separate word outputs because they fail the
full-line transcription match. A GT-blind word-to-line grouping/ordering stage
would be a new experiment; scalar confidence changes do not create line strings.
The existing ignore policy and axis-box geometry remain additional limitations.

## Threshold results

The identical grid0.05..0.99 in steps0.01 was applied to:

1. Raw YOLO confidence; retain current detector proposal NMS.
2. Existing joint confidence plus same-normalized-reading NMS at bbox IoU>0.5.

All190 controls per dataset are retained. Postfilter thresholds act on saved
proposals; floors below0.05 or changed detector proposal NMS need new inference.
Maximum one-to-one matching and casefold/punctuation/symbol normalization are
unchanged. CTW remains a separate exact-line diagnostic, never pooled with words.

### One shared threshold

Cells below show **exact correct / Hmean**:

| Joint threshold | ICDAR13 words | ICDAR2015 words | CTW1500 lines |
|---|---:|---:|---:|
| 0.50 | 883 /78.84% | 1,486 /58.52% | 1,168 /27.02% |
| 0.60 frozen baseline | 862 /80.83% | 1,401 /63.49% | 1,106 /27.89% |
| 0.62 exploratory shared improvement | 857 /81.31% | 1,379 /64.32% | 1,088 /27.93% |

Among tested joint thresholds improving all three baselines,0.62 maximizes the
smallest absolute Hmean gain. It removes290 final outputs but loses45 exact
correct units in total. Threshold0.50 instead recovers21/85/62 additional exact
units while retaining more false positives. Recognition recovery and Hmean
therefore favor different operating points in these measured controls.

### Per-dataset TEST maxima (exploratory)

| Dataset | Joint threshold | Correct | Outputs | Precision | Recall | Hmean |
|---|---:|---:|---:|---:|---:|---:|
| ICDAR13 | 0.69 | 838 | 950 | 88.21% | 76.53% | 81.96% |
| ICDAR2015 | 0.68 | 1,272 | 1,833 | 69.39% | 61.24% | 65.06% |
| CTW1500 lines | 0.62 | 1,088 | 5,105 | 21.31% | 40.49% | 27.93% |

Raw detector-confidence maxima are0.49/81.12%,0.50/61.54%,0.38/26.92%,
respectively; they do not outperform the joint method on these TEST splits.
The original inference config remains frozen at joint0.60. All maxima and the
shared0.62 observation were selected using TEST labels and require independent
validation before a new generalization claim or deployment setting.

## Qualification

16 focused tests pass, including threshold-before/after-NMS equivalence at all
grid values and exact score ties, safe pre-OCR bounds, whitespace-preserving
line diagnostics and one-to-one matching. All1,233 images were rescored.
An independent confidence/NMS/normalization/IoU implementation and SciPy
matcher verified all **234,270 per-image threshold controls**,25,992 candidate
scores, GT denominators, eligible-crop counts and aggregate P/R/Hmean.
`independent_audit.json` records the complete PASS result. No new GPU execution,
scientific component substitution or strict configuration change is introduced.
