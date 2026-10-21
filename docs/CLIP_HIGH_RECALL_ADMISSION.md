# CLIP and PARSeq on the high-recall pool

**Approximate reproduction**, Total-Text TRAIN development. This experiment
addresses precision in a crop pool already able to recover many words. It does
not change STRICT, generators, PARSeq, DeepSolo, boxes, or transcriptions.
Local paper sections3.3–3.5 do not specify CLIP admission.

## Existing experiments found

The repository contains richer pools than the first CLIP comparison:

| Pool/result | Crops or outputs | Correct words available/selected |
|---|---:|---:|
| Baseline/thresholds/windows, before selection |18.912|782 oracle /983|
| Complete-link IoU0.55 groups |8.683 groups|781 group oracle /983|
| guarded_consolidated group selection |7.462 outputs|738 selected /983|
| All generators without final variations |40.380|811 oracle /983|
| All generators including final variations |472.358|854 oracle /983|

Sources: `TOTALTEXT_CROP_WORD_GROUPS.md`, `CROP_GROUP_REFINEMENTS.md`,
`TOTALTEXT_CROP_RECOVERY_SUITE.md`, frozen artifacts under
`/tmp/totaltext-crop-recovery-suite/d957d64fb251/` and
`/tmp/totaltext-crop-group-refinements-v2/train-full/`.
An oracle measures GT-assisted reading availability, not selector performance.
Larger pools also include transformed crops; testing them would require
reconstructing the exact pixels of their respective recipes.

The first test uses the 7.462 already-selected outputs, recall75.08% and
precision9.89%. They contain correct readings for 738 words; every subsequent
choice only filters, without changing crops or adding words. The PARSeq
confidence>=0.95 control retains 704 from 2.806 outputs, precision25.09%,
recall71.62%. This is the reference to improve upon.

This pool's normalization removes Unicode punctuation/symbols in addition to
casefold. The earlier fixed-box CLIP comparison retained punctuation; its
368/405/983 counts are not directly interchangeable with these. Matching remains
one-to-one bboxIoU>=0.5, not polygon Hmean.

## Authentic inputs and annotation separation

Reuse sealed `inputs.json` from the crop experiment, not BERT scores or choices.
For every output, check ID, box, raw reading, RGB hash, and confidence against
native PARSeq caches and original selection. The config pins SHA values for both
provenance files. All images/crops are checked against original RGB pixels before
CLIP inference. Tiny crops explicitly receive `channels_last`.

Confidence>=0.5 prefilter:5.410 crops, containing738 correct words in the prior
TRAIN control, recalculated in this session. This threshold is motivated by
already-observed data: an explicit development choice, not a transfer guarantee
or paper parameter. All exclusions are recorded; the other2.052 predictions
remain in non-CLIP controls and the reference. Excluded crops never enter CLIP
arms, even with final threshold zero.

Inference reads only observables, images, and weights. All scores and rank-based
decisions are saved before opening `evaluation_labels.json`. Descriptions are
not used in this test. Annotations are subsequently used to evaluate/select a
TRAIN curve point, with that use declared.

## Signals tested

- **Word/crop correspondence:** cosine between crop and written-word prompt,
  plus the normalized average of three templates already used previously.
- **Visual correspondence:** render the same PARSeq word with Sans, Serif, Mono,
  black/white and white/black; compare the crop with the normalized average of
  six visual embeddings. Square canvases retain the field of view. Fonts,
  sizes, polarities, and rendering hashes are saved.
- **Text versus background:** cosine difference between readable-text and
  background/texture/shape-without-text prompt ensembles.
- **Word versus background:** difference between the word prompt and background
  ensemble, potentially reducing reliance on visual plausibility alone.
- **Glyph versus uniform images:** difference between the rendered candidate and
  normalized average of black/white images without letters. This controls
  background features; it does not segment letters.

These contrasts are hypotheses; CLIP is not claimed to be a calibrated text
detector. Semantics, font, reading length, and crop distribution may affect all
signals.

Each signal becomes its empirical rank, with midrank ties, on active TRAIN
without labels. Equal-weight geometric means combine PARSeq and CLIP ranks;
two combinations also include glyph and text/background signals together.
The CDF reference is saved to apply the same transform to new data without
recalibrating on TEST. Ranks are not probabilities.

## Precision while retaining recall

All grid points0,0.05,...,0.95,0.97,0.99 are reported for 15 families plus baseline:
331 checks overall. For each family, TRAIN selects maximum precision retaining
at least95% of the **same GT identities** correct in baseline: at least702 of 738.
Recovering other identities does not compensate for losing protected words.

Confidence and physical-score are controls. If they outperform CLIP arms, the
report keeps them as best rather than forcing a positive CLIP result. All983
care GT remain in the denominator, including unrecoverable words. Proxy
precision/recall/Hmean, correct counts, outputs, gains/losses, and per-image
matches are saved. Exports preserve IDs/boxes/readings/RGB without annotations;
threshold selection remains explicitly TRAIN-guided.

## Execution

From the host GPU terminal, with L/14 already available:

```bash
bash scripts/run_clip_high_recall_admission_host.sh
```

Default: L/14 CUDA:0, FP32, batch 16, one model per process. No installation or
download. Log: `/tmp/clip-high-recall-admission-host.log`. The report is printed
at completion under `/tmp/clip-high-recall-admission-v1/<model>/<protocol-hash>/`.

Other checkpoints: `--model vit_b32` or `--model vit_b16`. Initial CPU smoke:

```bash
bash scripts/run_clip_high_recall_admission_host.sh --model vit_b32 --device cpu --limit 1
```

Full local B/32 CPU uses `--model vit_b32 --device cpu`, without a limit.
Add `--check` to inspect inputs without inference. Model, weights/config/tokenizer,
software, code, device, prefilter, and sample define the protocol. Altered or
incompatible caches fail; payload SHA detects changed scores. Historical results
are never overwritten. Repeated loads have separate receipts; interrupted runs
resume only compatible shards. CUDA unavailability/OOM propagate without
changing device, precision, batch, or model.

## Checks

59 focused tests pass: ranks/ties, protected GT identities, crop preservation,
zero-threshold exclusions, geometry, corrupt caches, tiny RGB, and a simulated
memory error recorded/propagated. Real B/32 CPU smoke completed on the first
image:26 outputs,15 scored crops, two GT words. Its best point retains 2/2 from 2
outputs versus 2/2 from 3 under confidence0.95. This verifies execution only,
not precision improvement over the entire pool.

Full B/32 CPU completed on all 125 images and all 331 checks, with 5.410 genuinely
scored crops and 605.15s recorded scoring.

| TRAIN point | Correct | Outputs | Proxy precision | Proxy recall | Lost from baseline |
|---|---:|---:|---:|---:|---:|
| High-recall baseline |738|7.462|9.89%|75.08%|0|
| PARSeq confidence>=0.95 |704|2.806|25.09%|71.62%|34|
| PARSeq + CLIP ensemble, score>=0.45 |706|2.784|25.36%|71.82%|32|
| PARSeq + CLIP word/background, score>=0.45 |714|2.798|25.52%|72.63%|24|
| PARSeq + glyph, score>=0.20 |717|4.435|16.17%|72.94%|21|
| PARSeq + residual glyph + text/background, score>=0.35 |712|3.234|22.02%|72.43%|26|

Best observed point: `native_word_advantage@0.45`. CLIP score is
`cos(crop, ensemble_word_prompt) - cos(crop, ensemble_background_prompt)`;
final score is the square root of the product of empirical ranks for this
contrast and PARSeq confidence. It retains 714/738=96.75% of baseline-correct
identities. Improvement over confidence0.95 is small: ten more correct words,
eight fewer outputs, +0.43 percentage points precision. This establishes neither
generalization nor high precision.

A further **post-hoc** control fixes output count to2.798: selecting the 2.798
highest PARSeq confidences gives703 correct versus 714 for PARSeq/CLIP. This is
an equal-budget ranking diagnostic, not a deployment rule or GT-derived word
budget for new images.

The audit checks125 score shards and 125 selections, payload/code hashes, crop
identities, and all 331 aggregates against per-image evaluations.
Report: `/tmp/clip-high-recall-admission-v1/vit_b32/7423fe6400b8/results.json`,
SHA256 `baa1ba9bf9bf5e862e9c6ec05f57e204a72abae64647e3924d0f13a978e8ca80`;
audit: `independent_audit.json` in the same directory. Exports for selected
points: `selected_outputs.json`; future-data transform: `calibration.json`.

Real L/14 CPU smoke also completed on image14,15 scored crops, all 331 checks,
and 2/2 GT retained. L/14 GPU on this new pool still requires the initial command;
old-test results do not replace that measurement. Precision/placement/batching
are `HARDWARE_DEVIATION`; seed is a reproducible `ASSUMPTION`. CPU GPU peaks
are null; no RTX3070 qualification is inferred from smoke. Protocol and
classifications: `CLIP-HIGH-RECALL-ADMISSION-V1` in `PAPER_AMBIGUITIES.md`
and config under `configs/approximate/`.
