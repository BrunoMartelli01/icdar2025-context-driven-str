# Completed CLIP comparison: B/32, B/16, and L/14

The requested comparison is complete: three checkpoints, 40 rules each,
125 Total-Text TRAIN images, 1.173 fixed boxes, all 983 care GT in the denominator.
These are **approximate reproduction** results on development data, not
independent TEST, recognition of new words, or reproduced Table 5. The separate
DeepSolo slot experiment does not contribute to these results.

## Main results

Each value counts exact Unicode-casefold readings with punctuation retained,
under one-to-one bbox matching at IoU >=0.5. Alternatives are existing PARSeq
strings; all methods retain identical boxes. Original PARSeq reaches 368/983;
selecting the highest-confidence PARSeq alternative reaches 383/983 for all models.

| Rule | B/32 | B/16 | L/14 |
|---|---:|---:|---:|
| Crop + written-word prompt |362|361|370|
| Crop + three-prompt ensemble |361|365|374|
| Crop + word and scene description |355|356|366|
| Crop + contextual residual |369|364|371|
| PARSeq/CLIP rank fusion |378|382|386|
| Conservative PARSeq/crop/wide-crop agreement |370|370|376|
| Visual comparison with rendered candidates |371|374|383|
| Before/after target-covering difference + PARSeq agreement |372|370|374|
| PARSeq/visible-target/rendered-candidate agreement |369|370|371|

Fusion uses equal-weight average midranks of PARSeq confidence and CLIP
crop-ensemble score. It was declared before outcomes, not newly threshold-tuned
on these results. L/14 is the best observed checkpoint under this rule: +18 over
original baseline, +3 over confidence selection. This remains exploratory among
40 rules.

| L/14 rule | Correct /983 | Gained | Lost | Changed |
|---|---:|---:|---:|---:|
| PARSeq confidence |383|26|11|351|
| PARSeq/CLIP rank fusion |386|19|1|241|
| Conservative PARSeq/crop/wide-crop agreement |376|8|0|33|
| Rendered candidates, aspect-preserving canvas |383|19|4|323|
| Rendered candidates, square-stretched crop |375|13|6|331|
| Rendered candidates, mean of both crops |381|17|4|330|
| Visual control against reversed strings |371|12|9|264|

Gains/losses compare correct GT IDs with the original baseline, not the
confidence arm. A change need not affect exact matching. No observed sample loss
does not guarantee no losses on new images.

## Do descriptions help the prompt?

Directly adding descriptions worsens crop selection in every model:
B/32 362→355, B/16 361→356, L/14 370→366. Contextual residual gives mixed results
and does not beat the best PARSeq control. With L/14, true description gives
366 versus 365 when swapped from another image: this experiment does not
establish useful real-context benefit.

Descriptions are verified Florence-2-large-ft caches, not BLIP-2. They contain
the GT word in 160 of 482 uniquely GT-associated anchors and may include their
own OCR suggestions. They neither generate candidates nor derive from GT.
Prompts respect the 77-token limit without truncating candidates. Shorter,
local, or text-separated contexts were not tested and are not ruled out.

## Word correctness and letter discrimination

On 482 anchors overlapping exactly one care GT, PARSeq confidence separates
correct/incorrect better than individual CLIP scores. The 688 missing and three
ambiguous associations are excluded only from this diagnostic; all 983 GT stay
in the main metric. These 482 cases contain 365 correct and 117 incorrect words.
AUROC is not a calibrated probability of correctness.

| Correctness indicator | B/32 AUROC | B/16 AUROC | L/14 AUROC |
|---|---:|---:|---:|
| PARSeq confidence |0.823|0.823|0.823|
| CLIP crop + written word |0.615|0.613|0.593|
| CLIP crop + word and description |0.562|0.584|0.539|
| Visual CLIP, rendered candidate |0.710|0.715|0.725|

Representing candidates as images is therefore more promising than simple
prompts for this diagnostic. Among 67 correct/incorrect Levenshtein-distance-one
pairs, L/14 rendered candidates choose correctly in82.1%, matching PARSeq control;
written prompt reaches68.7%, description prompt62.7%. Correlated pairs do not
provide independent significance evidence. Font, length, semantics, and style
can also affect visual comparison.

A data-motivated direction retains PARSeq candidates and uses L/14 as visual
confirmation: fusion gives more net recoveries; conservative gating limits
changes. Both need a frozen rule and independent verification before final
selection. Reversed strings/stretching do not improve simple L/14 visual
comparison in this population.

## Provenance, hardware, and checks

Protocols:
[v1](../configs/approximate/clip_word_verification_train_v1.json) and
[v2](../configs/approximate/clip_innovative_verification_train_v2.json).
Revisions: B/32 `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`,
B/16 `57c216476eefef5ab752ec549e440a49ae4ae5f3`,
L/14 `32bd64288804d66eefd0ccbe215aa642df71cc41`. All weights have verified
upstream checksums. The B/16 receipt resolves the earlier checksum limitation
for this run: SHA256
`ec89c7b09c749a60aae3c9cd910516f24b58214a7df060b48962d14c469cfbf0`.
Historical config and CPU results remain intact.

Host: RTX 3090, separate sequential models, float32, batch 16, input [3,224,224],
seed 20261007, TF32 disabled, deterministic backends. These are
`HARDWARE_DEVIATION`; they do not qualify the RTX 3070 target.

| Model | Peak allocated bytes | Peak reserved bytes | v1 scoring s | v2 scoring s |
|---|---:|---:|---:|---:|
| B/32 |687653888|767557632|30.77|141.31|
| B/16 |796634624|926941184|37.13|225.26|
| L/14 |2023449600|2181038080|71.32|610.66|

Timings sum per-image recorded scoring and exclude downloads, initial loading,
and evaluation; they are not end-to-end latency benchmarks. Every variant
completed successfully without reported OOM. Codex does not expose CUDA; this
verification uses host GPU outputs.

Independent verification recalculated all metrics in six reports from 750 shards
and checked inputs, pools, selections, weights, code hashes, and provenance.
All28 v1 controls remain within 40 v2 rules for each checkpoint.53 CLIP/control
tests pass;762 original artifacts are unchanged in the audit. No original output
was modified. Historical CUDA-preflight failures/CPU smoke remain preserved
and are not GPU results.

Artifacts outside Git:

- Original report: `/tmp/clip-verification-host-suite/6920d0b0c030/comparison.json`.
- Complete table: `/tmp/clip-verification-host-suite/6920d0b0c030/COMPARISON.md`.
- Audit/summary: `/tmp/clip-verification-analysis-6920d0b0c030/verified_results.json`.
- Repeat audit: `.venvs/context-str-main-py311/bin/python /tmp/clip-verification-analysis-6920d0b0c030/audit_completed_suite.py`.

Original report SHA256:
`2cdbf5b858ad181eff5322a512d41264e613af109a2ef5287e34fcbcf6eb828a`.
Method risks/classifications remain in `CLIP-WORD-VERIFICATION-V1` and
`CLIP-INNOVATIVE-VERIFICATION-V2` in `PAPER_AMBIGUITIES.md`. Calibration, TEST
generalization, other captioners, and pretrained spelling projections remain
`UNRESOLVED`.
