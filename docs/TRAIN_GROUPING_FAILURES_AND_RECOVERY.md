# TRAIN: localization errors, crop completeness, and OCR recovery

Updated:19 September2026.

This is the starting point for grouping work, collecting results, conclusions,
and limitations from practical-remap-v1 error analysis.
Classification: **APPROXIMATE_DIAGNOSTIC / approximate reproduction**.
Neither reproduced Table 5 nor a new TEST evaluation.

## 1. Frozen state and constraints

- ICDAR13 TRAIN:229 images,849 care GT words.
- Author-protocol epoch-0098 checkpoint:
  SHA256 `b450f19e79f649f8873a8b8db0d9a25ab0a7f144ac5f8676a2a80d5c1d8fd2f2`.
- Native128 binary prediction, sigmoid>=0.5; historical cache/nearest backprojection
  verified. No resolution/threshold changes in these audits.
- Frozen practical-remap-v1 baseline, padding6/top10 unchanged.
- Paper-faithful Algorithm1, v1, v2, segmentation, scoring, routing unchanged.
- Qualified PARSeq; V1 text reused from cache.
- CPU geometry analysis. Only subsequent diagnostic-crop OCR was run by the
  user on PAPER; Codex did not execute CUDA.
- No training, grid, new segmentation, TEST use, or parameter selection.

## 2. Verified TRAIN baseline

| Metric | Result |
|---|---:|
| GT words |849|
| Predictions available for diagnostic association |745|
| Total V1 crops |1203|
| PARSeq exact casefold |502/849 = 59,13%|
| PARSeq RapidFuzz casefold>=0,8 |578/849 = 68,08%|
| GT with proposal IoU>=0,5 |568/849|
| Exact errors, including missing outputs |347|

Historical association: maximum-IoU crop for each GT, available only with positive
intersection. Missing predictions stay in denominators. GT association is ONLY
diagnostic, never routing.

## 3. First audit: which crop reaches PARSeq?

| Observed situation among 347 errors | GT words |
|---|---:|
| No overlapping retained crop |104|
| IoU<0,5 crop, not multiword |102|
| Multiword crop |75|
| IoU>=0,5 crop, not multiword, wrong OCR |66|

These rows are mutually exclusive, not four established causes.75 counts failed
GT words, not all dataset multiword crops.

**Key finding:** among 66 IoU>=0,5 errors,49 exclude at least one entire annotated
glyph;62 clip at least part of glyphs. Only4 contain all GT glyph pixels.
BBoxIoU>=0,5 therefore does NOT guarantee a complete PARSeq word. Historical
`GOOD_BOX_OCR_ERROR` does not imply intrinsic recognizer error.

None of 347 errors has an exact transcription in another already-generated V1
crop geometrically overlapping the same word. Simply reranking existing V1
outputs cannot recover them.

## 4. Second audit: missing glyphs and extraneous text

Count original GT glyph pixels, not bbox area. Other care words are separated
from Don't Care. Even one extraneous pixel is recorded, but alone does not prove
a significant merge.

| Crop state among 347 errors | Words |
|---|---:|
| Missing |104|
| Incomplete, no other care-word pixels |159|
| Incomplete, with other care-word pixels |28|
| Complete, with other care-word pixels |52|
| Complete, no other care-word pixels |4|

The last4 include Don't Care: arlboro/Marlboro, ESP/ESPI, TO/MOTO, KIA/OKIA.
Do not change annotations/metrics to make them correct; document visible text
versus evaluated care text.

Among291 incomplete/missing-crop errors:

- 152 have some predicted support on every annotated glyph;
- 139 have at least one unsupported glyph;76 have no support on any glyph;
- 173 exclude own-word pixels predicted as segmentation foreground.

These partially overlapping views must not be summed. Minimal predicted support
does not establish good glyph reconstruction. In83 errors, native-mask precision
and recall>=0,8 within the word region, but crop IoU<0,5: downstream problems
can follow locally good predictions.

Perfect completeness is not necessary for every recognition:98 of 502 correct
words still clip some glyph pixels.

## 5. Geometric oracle: recovery from existing predicted components

Counterfactual: union bbox of raw predicted components associated with GT, plus
frozen padding6. No new prediction. This GT-assisted union is NOT an inference rule.

- 173 incomplete/missing errors improve included own foreground.
- 89 become fully included.
- 80 become complete without other care-word/Don't Care pixels.
- Union is not universally beneficial:8 errors lose coverage;9 gain other
  care-word pixels.

Composition of 80 cases selected before OCR:

| Previous geometric category | Cases |
|---|---:|
| GROUPING_UNDERMERGE |49|
| SUCCESS by IoU>=0,5, but clipped crop |25|
| TOP10_TRUNCATION |5|
| OTHER |1|

The5 top10 cases cannot automatically be attributed to the linking rule.
These80 were selected using GT/known failures: a conditioned diagnostic sample,
not independent validation.

## 6. GPU OCR on 80 cases

Local report checked against user-provided SHA; counts recalculated from 80
per-word rows. PARSeq ran only on RECOVERED/GT_BOX crops; V1 stayed cached.

| PARSeq input | Exact casefold | Fuzzy>=0,8 |
|---|---:|---:|
| Original V1 |0/80|25/80|
| RECOVERED, diagnostic union |78/80 = 97,5%|80/80|
| GT_BOX |78/80 = 97,5%|79/80|

The same two words remain wrong in both controls:

| Sample / zero-based GT index | GT | RECOVERED | GT_BOX |
|---|---|---|---|
|117 / 7|`(412)`|`(412")`|`(412")`|
|313 / 0|`GENTLEMEN`|`CENTLEMEN`|`CENTLENIEN`|

For78 words, crop correction alone enables exact reading without changing network/
recognizer. GT_BOX also fails for both remaining words; further grouping is not
a demonstrated solution for them.

**Accounting counterfactual**, keeping all other outputs unchanged:
502+78 =580/849 =68,32%, versus 59,13% (+9,19 points).
NOT new end-to-end accuracy, NOT a global top10 guarantee, NOT achieved performance
of a GT-free localizer. Analogous fuzzy would be633/849, still diagnostic
substitution rather than an evaluated pipeline.

## 7. Conclusions and decisions

1. Prioritize grouping/crops for the recoverable subset before U-Net retraining:
   OCR demonstrates real headroom with unchanged masks.
2. Segmentation is not proven sufficient everywhere. Unsupported glyphs/lost
   small text need separate diagnosis.
3. Optimize more than IoU: measure glyph coverage, extraneous text, recognition,
   loss of V1 successes.
4. Do not indiscriminately increase gaps/padding or merge every fragment:
   prior overmerges/worsened counterfactuals remain real risks.
5. Never use GT, correct text, or OCR outcome to decide inference groups.
6. This audit implemented/promoted no new rule.

## 8. Proposed next step, not performed

- Inspect missing edges/components in78 recoveries; separate undermerge,
  IoU-undetected clipping, and top10.
- Compare geometrically similar links that would merge different words.
- Test whether a simple prediction/geometry-only rule distinguishes these;
  do not invent thresholds to fit78 examples.
- Compare future variants with V1 across all TRAIN, counting recoveries,
  losses among 502 correct words, and new overmerges.
- Freeze protocol before any TEST; repeated TRAIN study requires independent
  validation to establish generalization.

## 9. Artifacts, hashes, and commands

Paths below are repository-root-relative. No earlier artifact was overwritten
for this summary.

| Artifact | SHA256 |
|---|---|
|`artifacts/diagnostics/train-v1-ocr-failure-audit-v1/results.json`|`82acfe8d860aee1a0f7bf70b9308a83aefb4abc17561105c74324d8b990fc56a`|
|`artifacts/diagnostics/train-v1-ocr-failure-audit-v1/glyph-clipping.json`|`3592cbe10dfabbee852b0f33cb35e660d32bc38b69b457fc747de2dfafc4218b`|
|`artifacts/diagnostics/train-crop-completeness-v1/results.json`|`f49d244e1e6c439bb9ec72e7fdb004ef2029bfe5ee405877f7d5f7e32a269cf9`|
|`artifacts/diagnostics/train-recovered-crops-parseq-v1/results.json`|`6dce36b5ac37f6a097a90dfcdc11bda74f6651377aa677f198a8cefe5dabf17c`|

All four JSON SHA values verified locally on 19 September2026. Earlier audit
CSV/PNG hashes remain in their reports/READMEs. Persistent OCR cache:
`artifacts/diagnostics/train-recovered-crops-parseq-v1/ocr-cache/`.

PAPER launcher, with progress/resume from saved readings:

```bash
cd /extra1/bmartelli/projects/icdar2025-context-driven-str
bash diagnostics/run_recovered_train_crops_host.sh /extra1/bmartelli
```

No relaunch is needed to inspect results; segmentation is not rerun. Recorded
checks:66 focused audit/completeness tests;7 OCR launcher/helper tests;
py_compile, bash syntax, git diff--check passed. No Codex CUDA test.

## 10. Scientific provenance and open limitations

- `PAPER_SPECIFIED`: components/contours, bboxes, padding, area filtering, top10(section3.3).
- `ASSUMPTION`: practical V1, best-IoU diagnostic association, glyph-color counters,
  GT oracle, casefold exactness, conditioned80-case selection.
- `HARDWARE_DEVIATION`: standalone PARSeq batch 1 on PAPER.
- `UNRESOLVED`: comparable Table5 accuracy; these reports do not resolve paper
  recognition unit, matching, multi-output handling.
- Per-word GT_BOX now exists for these 80, not all 849 words. Older aggregate
 836/849 GT_BOX cannot identify every error.
- Initial READMEs predate GPU OCR; section6 updates the prior missing-recovered-
  crop-OCR limitation.

Decision references: TRAIN-V1-OCR-FAILURE-001 and TRAIN-CROP-COMPLETENESS-001
in [PAPER_AMBIGUITIES.md](PAPER_AMBIGUITIES.md).
