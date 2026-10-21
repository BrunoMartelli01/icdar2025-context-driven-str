# Training-free contextual image-quality routing

APPROXIMATE,`approximate reproduction`, exploratory. Not Table5 or its official
protocol. Local paper §3.5/Algorithm2 already inspected: context verification is
extended to estimate **per-image YOLO640/PARSeq F1 ≥0,8**. Original words remain
identical; rejected images go entirely to DeepSolo.

## Frozen protocol

Isolated configs:`configs/approximate/yolo_context_quality_v1.json` and
`configs/approximate/yolo_context_quality_local_paths_v1.json`.
Classifications/limits:`YOLO-CONTEXT-QUALITY-001`,`PAPER_AMBIGUITIES.md`.

- Same125TRAIN/300TEST,640-pixel YOLO, PARSeq,425 image-only BLIP2 captions,
 425 complete DeepSolo ViTAEv2-S+TextOCR branches, and audited controls.
- TRAIN80: label-free CDF/family choice; TRAIN45: final thresholds. Receipt frozen
  before new TEST signals. Historically explored TRAIN/TEST do not support
  untouched-validation claims.
- No optimizer, gradient update, learned classifier, correction, word filter,
  adapted weight, GT dictionary, installation, or download.
- Separate FP32 models on PAPER's actually verified GPU. Record snapshots/hashes,
  before/after state, tokens/inputs, allocated/reserved memory, failures. CDFs/
  thresholds are empirical calibration, not network training.

## Experiments

1. Literal anchors: quoted/attributed tokens, such as “word PETROSAINS on it”.
   Exact multiset recall; edit matching diagnostic only. Caption without evidence:
   unavailable signal/neutral multiplier. Attributed text marked ambiguous;
   captions are fallible/non-exhaustive.
2. MPNet: own-caption transcription compatibility minus mean of three unrelated
   TRAIN80 captions. No truncation above384 tokens.
3. CLIP: original RGB crop, word, deterministic GT-free mutations; margin against
   most similar mutation. Simple prompt, own caption, unrelated TRAIN80 caption.
   Per-image20° percentile of margins. Above77 tokens means explicit abstention,
   never truncation.
4. Frozen FLAN-T5-base: teacher-forced Yes/No probability for literal support/
   coherence, unrelated-caption control. EOS targets, pad excluded. Text-only judge
   does not verify pixels, completeness, or F1.

Controls: coverage_baseline, all_checks_softcrop, historical contextual C3.
Predeclared primary family: base_clip_context; all 17 families reported.
Fixed combinations:`sqrt(baseline * CDF_TRAIN80(signal))`; literal contradiction
`baseline * caption_recall`. No fitted weights.

Sources:[CLIPScore](https://aclanthology.org/2021.emnlp-main.595/) evaluates
image/description compatibility;[VQAScore](https://arxiv.org/abs/2404.01291) uses
visual-answer probabilities;[TIFA](https://arxiv.org/abs/2303.11897) decomposes
alignment into questions. Text-only FLAN here **does not implement VQAScore**;
visual/residual verification would be a distinct later experiment.

## Metrics

Source metric: per-image casefold multiset, punctuation retained, no bboxIoU.
Don't Care only in evaluator; all 983/2204 care words remain in denominators.
Empty outputs cannot bypass. Separate accepted-image F1≥0,8 purity, its recall/
binomial interval, and final micro F1/TN0. Empirical80/90/95% good-image thresholds
require at least5 TRAIN45 acceptances; no valid threshold means full fallback.
Budgets10/20/30/40/50% use GT-free ranking with ID ties; report unfilled budgets.

Only120/300 lightweight TEST images reach F1≥0,8: at150 acceptances, theoretical
maximum purity is80%. Context cannot certify undetected words; existing coverage
remains an independent fallible signal. Better purity does not imply better
micro F1 versus DeepSolo. End-to-end speed claims must include captions, maps,
and verifiers.

## Execution and status

On PAPER, from repository, without SSH back to itself:

```bash
.venvs/context-str-yolo-py311/bin/python -m scripts.run_yolo_context_quality \
  --mode approximate \
  --config configs/approximate/yolo_context_quality_v1.json \
  --paths configs/approximate/yolo_context_quality_local_paths_v1.json
```

Add `--preflight` for initial checks,`--smoke` for 6 images. External outputs:
`yolo-context-quality-v1/{smoke,full}/HASH`, with plan, code, features, receipts,
thresholds, results. Restart reuses verified stages only.

Initial20 focused tests and real6-image/model smoke passed. Smoke
`1720a648e62e`, results SHA7360f1ba8f716a05b48749d4dbd437b13ab807e74d9f96b497616d8c630452cf.
Full comparison completed:`full/5e1900aa7bc2` below artifacts root.
Plan SHA6a922ea6b7e377440fcbf14f8d63b368fb598fea4ad37942d2a95206478847eb;
results SHA4aa89b7de8b1b66bfb2deb320db33997a0955d4005163dbec848594fa9d8c4c4.
Independent audit PASS:425 features,3.227 original RGB crops,884 routing points,
10 unchanged-state receipts; optimizer_steps0. No CLIP overflow/failure in full
run.29TRAIN/106TEST captions have extracted anchors(not annotations).

## Results

Fixed20% budget:25TRAIN/60TEST retained images. Purity is the share of retained
images with F1≥0,8, not word accuracy.

| Method | Good TRAIN | Good TEST | Final TEST F1 |
|---|---:|---:|---:|
| Prior coverage |15/25|39/60|79,8346%|
| Prior softcrop consistency |16/25|48/60|79,9633%|
| Primary: consistency + caption CLIP |8/25|38/60|79,6143%|
| Consistency + literal anchors |15/25|47/60|79,8531%|
| Consistency + contrastive MPNet |13/25|47/60|79,4678%|
| Consistency + FLAN literal support |14/25|47/60|80,0092%|
| Consistency + FLAN coherence |14/25|45/60|80,2203%|

Predeclared primary **does not improve** classification. Relative to softcrop,
purity difference is−16,67 points, paired95% bootstrap[−28,79;−4,88]. FLAN
coherence gives a small exploratory micro-F1 increase,+0,257 points,
95% interval[−0,347;+0,883], with fewer good images. This neither establishes
superiority nor promotes a post-TEST choice. At30/50% bypass, its F1 is77,745/74,397%,
below control79,384/76,891%.

Empirical90% TRAIN45 thresholds do not give that TEST purity: softcrop baseline
63accepted/50good=79,37%; anchors43/35=81,40%. Primary CLIP, MPNet, combined FLAN
have no90% threshold with minimum 5 examples: full fallback, without substituting
more permissive thresholds. Intervals condition on fixed gates,10.000 image
resamples, seed 20261021, no multiple-exploration correction; all 17 methods remain
in results.json and analysis/paired_comparisons_v2.json, with tie checks.

Total added time120,46s excludes cached captions/OCR/maps/DeepSolo. Not an
end-to-end benchmark. A later BLIP2 visual-residual experiment is isolated in
yolo-residual-vqa-v1 with separate protocol.

Residual comparison completed:[YOLO_RESIDUAL_TEXT_VQA.md](YOLO_RESIDUAL_TEXT_VQA.md).
The contextual VQA primary retains 37/60goodTEST and the plain control42/60,
versus 48/60softcrop.No new family is promoted.Complete17-method threshold and
predicted-word/count tables with separate train/test are in
analysis/results_table.csv of the context run.Cached replay preserves exactly
the same result SHA without new inference.65 focused tests pass overall.
TRAIN125 denotes the historical segmentation holdout, not full official TRAIN.
Context run peak allocation1,233,462,272bytes, reservation1,415,577,600.
