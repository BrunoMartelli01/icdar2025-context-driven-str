# BERT selection of existing Total-Text crops

APPROXIMATE extension; TRAIN development only. This addresses the user's many-crop
high-recall / low-precision workflow. It does not alter STRICT localization,
recognition, fallback, scoring or configuration. Scientific record:
`TT-BERT-CROP-SELECTION-V1` in `docs/PAPER_AMBIGUITIES.md`.

The inspected local paper sections3.3–3.5 specify component crops, PARSeq-S,
BLIP-2 and MPNet/lexical/DeepSolo context routing. Cross-threshold groups,
Florence descriptions and BERT crop admission are experimental extensions.

## Source and separation of decisions

Reuse the complete125-image TRAIN crop-group refinement artifact, specifically
`guarded_consolidated`: 18,912 source crops, 8,683 groups, 7,462 retained outputs,
738 correct under the inherited local metric and983 care GT. Source readings,
original boxes/crop hashes, masks, grouping and PARSeq checkpoint stay fixed.

The new selector tests two decisions separately:

1. **Within-group ranking:** keep the baseline plus at most two distinct native
   PARSeq readings from competing crops. Geometry must overlap the baseline
   at bboxIoU>=0.5 and robust visual score must be>=0.85 times the baseline.
   Compare log visual-score ratio +0.15 times BERT mean log-likelihood delta;
   change only for gain>0.1. Ties/unknown reference keep the baseline. Conflicting
   assignments revert all involved groups to baseline, so neither a group nor
   an original crop is reused.
2. **Admission:** filter the selected groups using confidence-only, physical-only
   or physical+BERT scores. No new crop or spelling is generated. Numbers,
   accented text and proper names have no dictionary blacklist.

## Genuine BERT and context

Use cached `google-bert/bert-base-uncased`, immutable revision
`86b5e0934494bd15c9632b12f734a8a67f723594`, FP32. Verify all tokenizer/model
files and every learned tensor at loading; allow only the four explicitly
listed unused pooler/next-sentence keys. This is a general pretrained masked
language model, not a checkpoint trained to detect text/background or rank crops.

[Salazar etal2020](https://aclanthology.org/2020.acl-main.240/) motivates frozen
MLM hypothesis rescoring. [Kauf/Ivanova2023](https://aclanthology.org/2023.acl-short.80/)
motivates masking the target subwords at and to the right of each scored subword.
The reused target-only **mean** PLL-word-l2r is our adaptation; it is not a
calibrated probability that an image region contains a correctly read word.

Context is frozen before BERT: reliable native PARSeq anchors with balanced
confidence>=0.85, absolute maskmean>=0.4 and at least two normalized characters.
Near-overlapping anchors are spatially deduplicated. Up to three same-line
neighbors per side are used, excluding the target group and groups sharing
parent crop candidates. All alternatives in a group receive identical context;
newly selected readings never feed later decisions. The scene control adds
checksum-bound Florence-2-large-ft descriptions as a separate BERT segment,
truncated at80 whitespace-separated words. Captions may be factually wrong.

## Physical evidence and operating points

The absolute mask mean uses the continuous integral divided by the rectangle's
area in the letterbox probability canvas. Relative group completeness can be1
even on a poor region; it is not substituted for absolute foreground evidence.
The physical score is the geometric mean of balanced PARSeq confidence,
absolute mask mean, and `min(1, crop_height_in_mask_cells/2)`.

The joint admission score is physical times
`sigmoid((BERT_mean_PLL+6)/2)^0.25`. Unknown or overlong BERT targets explicitly
abstain to the physical score. This arbitrary scale is classified ASSUMPTION,
not the paper's alpha/beta/tau and not an estimated posterior probability.

Seven predeclared admission families use thresholds0,0.05,...,0.95. All curves
are reported, alongside unfiltered baseline/local/scene rankings. For each
family select the highest precision point retaining at least95% of the **same
GT identities** correct in the baseline. Recovering other GT does not excuse
losing protected baseline words. This uses TRAIN annotations for development
threshold selection; transfer must freeze the threshold and cannot be called
independent validation on this reused sample. No TEST run or tuning is performed.

## Evaluation and artifacts

Inference reads only the frozen observable `inputs.json`, its protocol/runtime
and pinned BERT files. Labels are stored separately. Evaluation first checks all
inference hashes and complete population, then opens labels. Every sample stays
in the denominator, with no failed sample silently omitted.

Metric: micro precision/recall/Hmean proxy, exact Unicode casefold comparison
after removing Unicode punctuation and symbols, preserving accents/digits/spaces,
axis-aligned bboxIoU>=0.5, maximum-cardinality one-to-one matching, all source
care GT. Empty predictions have precision0. Inherited source care/ignore handling
is explicit in config. This differs from official polygon Total-Text evaluation
and from earlier BERT spelling experiments that preserved punctuation; their
counts must not be mixed.

Artifacts include observable candidate/physical/context evidence, separate
labels, full token likelihoods, ranking gains, collision traces, admission
scores, per-image matches, every threshold result, selected operating points
and a label-free selected-crop export with original IDs/boxes/RGB hashes.
Sealed files cannot be overwritten by changed code/config/input/runtime.

CPU FP32 execution uses fourthreads and batch16; cached Florence and PARSeq
outputs are reused. This is HARDWARE_DEVIATION, not paper timing or RTX3070
qualification. Optional CUDA is explicit with no automatic CPU fallback;
allocated/reserved peaks and propagated OOM receipts are recorded.

## Run

No packages or new weights need installation when the existing pinned cache
is available. Configure paths through CLI arguments, never through code edits.

```bash
bash scripts/run_totaltext_bert_crop_selection_host.sh \
  --source-dir /tmp/totaltext-crop-group-refinements-v2/train-full \
  --captions /tmp/totaltext-bert-florence-context-v1/captions.json \
  --model-dir /extra1/bmartelli/context-str-artifacts/huggingface/models--google-bert--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594 \
  --output-dir /tmp/totaltext-bert-crop-selection-v2 \
  --reuse-inference-dir /tmp/totaltext-bert-crop-selection-v1 \
  --device cpu
```

The launcher runs prepare/infer/evaluate/export and propagates every failure.
Its exported operating point is the best eligible family on TRAIN, even if the
visual-only control beats BERT. Every per-family point remains inspectable.
Use `--limit 1` with a fresh output directory for a separate SMOKE_TEST.

Omit `--reuse-inference-dir` with a fresh output directory to compute new BERT
scores. Cache reuse checks the full observable sample population byte-value
equivalence, immutable model/runtime identity, original payload/receipt hashes,
unchanged masking implementation and authenticated historical scoring-function
AST. Missing, corrupt or incompatible cache entries fail rather than fabricate
or silently recompute scores. Inference opens neither source nor destination
labels. The final configuration revision makes every policy field mandatory and
exposes the originally fixed anchor-length2 and anchor-deduplicationIoU0.5;
all numeric/scientific decisions remain unchanged. Full neural scores were
computed once on125 images, then replayed into the final protocol with explicit
cache provenance. Original source snapshots are preserved with the neural run.

For integration by other crop generators use
`recognition/bert_crop_selection.py`: freeze `observable_sample`, score native
readings with `score_requests`, then `selections` and `apply_filter`. The CLI
adapter deliberately requires the authenticated TRAIN crop-group/refinement
schema rather than silently accepting incompatible OCR formats or TEST data.

## Verification record

The first genuine one-image BERT inference completed with62 scored crop/context
pairs. Its later evaluation was rejected because code changed during continued
implementation; the checksum failure was not bypassed. A fresh finalized run
is required. Unit coverage includes mask-coordinate averages, geometry/native
reading preservation, context self-exclusion, unknown scores, ties, group/crop
conflicts, threshold boundaries, isolated configurations, corrupt payloads,
batched subword ordering and recorded/propagated simulated OOM.

The finalized genuine one-image launcher smoke completed all143 controls and
export. An initial auxiliary export audit searched for the string `ground_truth`
and matched legitimate `ground_truth_use` provenance metadata; its corrected
structural check confirms no annotation/matching fields enter the export.

## Measured full TRAIN development result

All125 images/983 care GT were evaluated. The observable shortlist has9,595
native readings,1,711 groups with alternatives,2,036 reliable OCR anchors and
4,826 groups with local context. Genuine CPU FP32 BERT inference produced19,190
valid crop/context scores with no unknown/token-limit failures, in502.5seconds;
this is local development timing, not paper or GPU performance. The label file
was unavailable during the entire genuine neural process, and source/destination
labels were unavailable during final cache replay.

| Decision / selected TRAIN point | Correct | Outputs | Precision proxy | Recall proxy | Baseline words lost |
|---|---:|---:|---:|---:|---:|
| Frozen source |738|7,462|9.89%|75.08%|0|
| Confidence-only,>=0.95 |704|2,806|25.09%|71.62%|34|
| Physical admission,>=0.70 |721|4,439|16.24%|73.35%|17|
| Baseline crops + BERT local admission,>=0.05 |735|7,384|9.95%|74.77%|3|
| Baseline crops + BERT/Florence admission,>=0.25 |718|6,632|10.83%|73.04%|20|
| Unfiltered BERT crop ranking,local |651|7,462|8.72%|66.23%|97|
| Unfiltered BERT crop ranking,Florence |689|7,462|9.23%|70.09%|62|

Local ranking recovers10 new GT but loses97; Florence ranking recovers13 but
loses62. Neither ranking retains95% of baseline-correct GT even before filtering,
so their three admission families have **no eligible operating point**. This is
a negative BERT result; it is not promoted merely because the user requested BERT.

The exported recommendation is `confidence@0.95`: remove4,656 outputs(62.4%),
retain704/738=95.4% of baseline-correct words, and remove4,622 unmatched outputs.
It uses native PARSeq confidence and does not depend on BERT in production.
BERT and Florence remain optional comparison tools in the isolated module.
Precision25.1% still leaves2,102 unmatched outputs: the80–90% target is not met.
The experiment supports a need for stronger visual text/background evidence,
not a claim that linguistic plausibility can detect a valid crop.

Final report: `/tmp/totaltext-bert-crop-selection-v2/results.json`.
Selected native crops:
`/tmp/totaltext-bert-crop-selection-v2/selected-crops-confidence@0.95.json`.
The JSON contains125 images, original IDs/boxes/readings/crop hashes and no GT.
All143 control results, per-image matches and recovered/lost identities are kept.

To apply the exported confidence rule to a new observable crop pool, keep the
TRAIN-selected threshold fixed and use the existing confidence implementation:

```python
from recognition.group_word_selection import reading_confidence

# baseline: existing one-crop-per-group selection; rows: native PARSeq records.
# minimum_confidence: explicit argument, 0.95 for the selected TRAIN point.
by_id = {row["candidate_id"]: row for row in rows}
selected = [row for row in baseline if reading_confidence(
    by_id[row["candidate_id"]]["ocr"])["confidence_balanced"] >= minimum_confidence]
```

This model-free path was checked against all125 exported images and produces
exactly the same2,806 original crop IDs. A separate SciPy sparse bipartite
matching audit, with independently implemented normalization/IoU, confirms
704 correct /983 care GT. Receipt:
`/tmp/totaltext-bert-crop-selection-v2/independent_final_audit.json`.
Report SHA256:`66159fa62e6f8d987c400abd4d6282de03a8832d082c267fd06033df8e74a1e9`;
export SHA256:`940c933e5b3088ab8f25aa37c1cc27161b04d06b59add3e14e1f295b24472160`.

Focused tests:74 pass, including explicit-policy requirements, cache observation
and algorithm mismatch, masking/batching, geometry, normalization interfaces,
one-use assignments, threshold boundaries, corrupt payloads and propagated OOM.
The genuine batching audit differs from the original per-context scorer by at
most3.814697265625e-6 token log-probability, below the1e-5 tolerance. CPU smoke,
full neural inference, complete evaluation, cache replay and export are separate
checks; none establishes RTX3070 GPU-memory qualification or independent TEST
performance. Existing core modules and other agents' edits are preserved.
