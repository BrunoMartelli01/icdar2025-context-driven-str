# Caption router versus the qualified lightweight router

Approximate reproduction. This user-authorized experiment implements the
closest evidence-supported caption/text gate around the qualified U-Net/PARSeq
and YOLO/PARSeq experts on Total-Text, ICDAR2013 and ICDAR2015. The authoritative
local paper sections 3.3–3.5, Figure 4 and Algorithm 2 and existing caption,
MPNet/fuzzy, whole-image PARSeq and shared trained-router code were inspected.
See `PAPER-CAPTION-COMPARISON-001` in `PAPER_AMBIGUITIES.md`.

## Protocol declared before inference

Use the common qualified 2026-10-29 experts and trained image-gate decisions.
These are validation-selected eligible experts, not a new global ranking of
every historical checkpoint. Total-Text uses the corrected Text+Kernel U-Net;
ICDAR2013 uses its qualified Total-Text-derived corrected U-Net and recovered
YOLO outputs; ICDAR2015 uses long768 U-Net and YOLO640. The same existing
validation-selected lexical corrections are retained for both routers. Those
expert improvements, benchmark vocabularies and YOLO are extensions of the
paper. No segmentation or recognizer weights are changed here.

Generate T1 with genuine native PARSeq from the complete image, T2 with pinned
BLIP-2 Flan-T5-XL from RGB alone, and T3 by spatially ordering and joining all
frozen light expert readings. Keep duplicates and empty outputs. BLIP-2 uses
the existing Figure-4-motivated paragraph prompt and the paper's medium
`min_length=40,max_length=80` decoder lengths. Three deterministic beams and
length penalty 1 are explicit assumptions; exact author weights, complete prompt,
decoding and tokenizer length convention remain unresolved. No OCR/GT enters
caption prompts, no caption editing, annotation-driven profile selection or
TEST threshold optimization is permitted.

Choose T1 if its MPNet cosine with T2 is strictly greater than T3's; otherwise
choose T3. Compute the chosen candidate's `C=0.6*cosine+0.4*fuzz_ratio/100`;
accept at `C>=0.8`, otherwise use the existing qualified full-image DeepSolo
cache. Fuzzy comparison keeps raw case and punctuation. Report a separate
joined-T3-only gate to measure caption routing without whole-image candidate
replacement. Empty selected output/caption or MPNet overflow beyond 384 tokens
forces fallback; no silent semantic truncation is used. T1 emits exactly one
native sequence, without invented boxes or whitespace splitting.

Report the literal 0.8 threshold and a separately labelled exact batch budget:
Total-Text 150/300 (50.00%), ICDAR2013 61/233 (26.18%), ICDAR2015 228/500 (45.60%). The latter ranks
the same confidence without TEST labels, with RGB-hash ties; it changes the
paper's threshold policy and is not its reported operating point. Also transfer
a GT-independent CAL quantile cut at the same target fallback fraction. No
new gate training is performed. Historical TEST exposure remains a limitation.

## Metric and execution

Match words exactly as per-image multisets after Unicode casefold and removal
of all non-alphanumeric characters, then sum TP/FP/FN over every TEST image.
This removes punctuation and symbols, preserves repeated-word multiplicity,
and has no spatial IoU requirement. Frozen care/ignore assignments and invalid
envelope handling remain unchanged. Empty care predictions remain FP; missing
words remain FN, even when T1 wins. `F1=2TP/(2TP+FP+FN)`. All-light and all-
DeepSolo controls and per-image counts are saved. This metric is not established
as the paper's recognition-accuracy definition.

Run `scripts.run_paper_caption_comparison` with explicit approximate config,
mode and output root. The runner pins inputs and neural artifacts, seals the
resolved plan, raw native generation results, CUDA allocated/reserved peaks,
source shapes, precision and batch size; it propagates errors with failure
receipts. Existing packages only. Actual hardware must be checked before each
GPU run. RTX 3090 CUDA FP32 T1 and FP16 BLIP-2, sequential loading, batch-one T1
and native batch-eight captioning, and CPU FP32 MPNet are documented hardware
deviations. Batch-one versus batch-eight decoding matched 17/18 smoke captions;
the one changed TRAIN caption is preserved as an explicit floating-arithmetic
output risk. No scalar-output equivalence is claimed. Shared batch memory and
timing are recorded per image with explicit batch IDs; those repeated timings
must not be summed as individual-image call times. No new DeepSolo neural
inference is needed; routing counts describe decisions on cached predictions
and do not measure online latency.

Complete results and qualification are recorded below.

Entry points: `recognition/paper_caption_comparison.py`,
`caption/paper_batch_blip2.py`, `scripts/run_paper_caption_comparison.py`,
`scripts/audit_paper_caption_comparison.py`.
Configuration: `configs/approximate/paper_caption_comparison_batch8_v2.json`.
Tests: `tests/recognition/test_paper_caption_comparison.py`,
`tests/caption/test_paper_batch_blip2.py`.

```bash
.venvs/context-str-main-py311/bin/python -m scripts.run_paper_caption_comparison \
  --mode approximate \
  --config configs/approximate/paper_caption_comparison_batch8_v2.json \
  --output-root /extra1/bmartelli/context-str-runs/paper-caption-comparison-20261030-v1/full_v2 \
  --stage all
```

## Caption similarity review requested by the user

The exact Figure 4 scene is Total-Text TEST1194 (Cannery Row directory). It was identified visually against PDF page 9. Its raw generated caption is:

> a sign with a map of san francisco and a trash can next to a sign that reads san francisco bay area san francisco bay area san francisco bay area san francisco bay area san francisco bay area san francisco bay area san fr

The paper example contains 72 words, versus 44 generated here. MPNet cosine between the two descriptions is 0.607295; this is semantic overlap, not a factuality or style certificate. The candidate recognizes a map/sign and nearby trash can, but introduces unsupported San Francisco text/location and repeats it. It omits the red rightward arrow, curved upper sign, map colors and paths, barrel/bin details, trees, wooden building and daylight described in the paper.

The paper paragraph takes 100 tokens with this candidate T5 tokenizer, beyond its configured maximum decoder length 80. This finding is specific to this candidate tokenizer; it does not identify the author checkpoint or resolve the paper's token/word convention. Neither caption editing nor a new TEST-selected profile follows the review.

A second visual review uses three CAL images per dataset, selected by frozen RGB hash without annotations. Direct image facts were recorded before reading their generated captions. Seven of those nine examples show substantial repetitive continuation; two offer short coherent descriptions with fewer attributes than Figure 4. This small descriptive sample is not a population factuality benchmark.

Across complete TEST captions, repeated-trigram fraction is `1 - unique_trigrams / all_trigrams`, after casefold/alphanumeric word normalization. A caption is counted below when that fraction exceeds 0.20. This detects repetition, not factual correctness.

| Dataset | TEST captions | Median words | Within 40–80 words | Repetition fraction >0.20 |
| --- | ---: | ---: | ---: | ---: |
| Total-Text | 300 | 34.0 | 119/300 (39.67%) | 187/300 (62.33%) |
| ICDAR2013 | 233 | 38 | 114/233 (48.93%) | 144/233 (61.80%) |
| ICDAR2015 | 500 | 44.0 | 269/500 (53.80%) | 335/500 (67.00%) |

Conclusion: this checkpoint/profile does **not** reproduce the illustrated caption quality. This limits interpretation of the caption router comparison; its scores cannot establish the quality or results of the unknown author configuration. Detailed unchanged captions, statistics and image views are in `caption_review/caption_review.json`, `caption_review/caption_review.html`, and `caption_visual_review.json` under the external experiment root.

## Complete TEST results

All 1,033 official TEST images are retained: Total-Text 300/2,204 care words, ICDAR2013 233/1,095, and ICDAR2015 500/2,077. The six-case comparison exactly replays the common trained router's original outputs and aggregate metrics. It does not run a new global checkpoint search. Every table below is **approximate reproduction**, using the punctuation-free lexical micro word metric above.

At the literal paper threshold, every image requests DeepSolo. The U-Net and YOLO rows coincide because neither light output is accepted:

| Dataset | F1 % at tau0.8, both experts | DeepSolo images/total (%) |
| --- | ---: | --- |
| Total-Text | 82.6540 | 300/300 (100.00%) |
| ICDAR2013 | 90.1707 | 233/233 (100.00%) |
| ICDAR2015 | 82.7905 | 500/500 (100.00%) |

These are all-DeepSolo controls produced by this threshold, not evidence that the gate saves inference or reproduces the author's operating point. For `C=0.6S+0.4L`, even `S=1` requires `L>=0.5` to reach 0.8; comparing a full descriptive/repetitive paragraph with an OCR transcript often yields low lexical similarity. The unresolved caption/checkpoint/aggregation choices remain material.

At the **same exact frozen fallback budgets**, the observed comparison is:

| Dataset | Expert | Paper-selected T1/T3 F1 % | Caption T3-only F1 % | Trained common router F1 % | DeepSolo images/total (%), all three gates |
| --- | --- | ---: | ---: | ---: | --- |
| Total-Text | U-Net | 75.8519 | 76.1183 | 79.7119 | 150/300 (50.00%) |
| Total-Text | YOLO | 78.9242 | 79.1493 | 80.9388 | 150/300 (50.00%) |
| ICDAR2013 | U-Net | 78.8637 | 79.1031 | 84.4294 | 61/233 (26.18%) |
| ICDAR2013 | YOLO | 85.7143 | 86.4662 | 86.6067 | 61/233 (26.18%) |
| ICDAR2015 | U-Net | 70.6177 | 75.7403 | 79.0943 | 228/500 (45.60%) |
| ICDAR2015 | YOLO | 73.4558 | 76.8657 | 80.8403 | 228/500 (45.60%) |

The common trained router has higher observed point F1 in all six matched-budget cases. This is conditional evidence for these qualified experts and previously explored TEST sets, not a population or statistical superiority claim. The joined-T3 ablation changes the confidence ranking and emits all light readings, so its difference is not a pure isolated effect of candidate replacement. T1 remains one PARSeq sequence; missing GT words remain FN.

CAL-only quantile transfers, all-light/all-DeepSolo controls, complete TP/FP/FN and every accepted/fallback image are preserved in `full_v2/results.json`, `comparison.csv`, `comparison.md`, and `per_image_scores.json`. CAL cuts need not reproduce their target TEST fraction; the report always shows the actual count. The exact-budget rows rank a complete batch and are not streaming scalar thresholds.

## Verification and preserved failures

The focused unit suite passes 72 tests, including native RGB-only batched generation, output cardinality, OOM propagation without alternate captions, raw fuzzy scoring, inclusive thresholds, semantic ties, duplicate words, punctuation normalization, T1 output cardinality, overflow fallback and complete denominators. Both scalar and batch-eight real smokes complete, with independent word/routing/semantic audits. A full independent audit is recorded separately.

Earlier failures remain preserved: the first smoke called the existing asset verifier with the wrong argument shape; a singleton smoke could not satisfy its exact budget because its only OCR output was empty; and a cold CUDA resume reset peak statistics before initialization. The API and CUDA initialization were corrected; the guarded singleton failure was handled by a larger smoke, not by accepting empty OCR or altering a scientific threshold. An intermediate navigation check reported expected stale generated views before rebuilding. No inference/model error was swallowed.

The active full run uses its sealed source snapshot and batch-eight v2 configuration. GPU allocated/reserved peaks, final independent audit, source and artifact hashes, and the caption review are bound in the completion receipt. No packages were installed, no DeepSolo neural calls were made, and no commit was created.

The full independent audit **PASS** verifies 2,608 sealed neural output shards, 4,132 independently recomputed word-count records, all 54 routing policies, 2,608 score decisions, and 23 model-weight/processor files. It re-encodes every relevant distinct text and reconstructs MPNet cosines independently; maximum absolute semantic difference is 1.75e-07, below the predeclared 2e-6 batching tolerance.

Measured full-run GPU peaks on the RTX 3090 are 10,077,775,872/10,989,076,480 allocated/reserved bytes for BLIP-2 FP16 batch 8 (native224 image preprocessing), and 131,752,448/153,092,096 bytes for PARSeq FP32 batch 1 (native32x128 inputs). Original image shapes are preserved per shard. No OOM occurred; a simulated OOM unit test confirms propagation without fabricated captions. These measurements do not qualify an 8 GB RTX 3070 run.

Portable completion receipt: `provenance/PAPER_CAPTION_ROUTER_COMPARISON_COMPLETION_V1.json`. External root: `/extra1/bmartelli/context-str-runs/paper-caption-comparison-20261030-v1`. The completed run is `full_v2`; earlier smoke/failure evidence remains in adjacent directories.
