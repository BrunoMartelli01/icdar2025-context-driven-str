# Training-free visual residual-text test

APPROXIMATE / `approximate reproduction`, exploratory, not strict reproduction.
Follow-up to the contextual17-family comparison; no final word changes.
Retain125TRAIN/300TEST and complete DeepSolo ViTAEv2-S+TextOCR. Existing BLIP2
image-only captions were never generated from PARSeq texts.

## Hypothesis and controls

Classifying F1≥0,8 also requires detecting words never localized. Hide the union
of original YOLO640 boxes and query the same frozen BLIP2:
“Is there any readable text or number visible in the image?”

Predeclared views: original image, boxes covered by gray128, the same boxes blurred
(global blur radius 5% of shorter side, applied inside boxes only), gray control
image. Outside pixels remain identical; overlaps are order-independent. Boxes
may also cover unrecognized text. Questions are simple or prefixed with the
same image-only caption; gray control measures caption-induced “yes” without
visible text.

Per question: teacher-forced Yes/No including EOS, summed sequence probabilities,
normalized over alternatives. These are not calibrated OCR-correctness
probabilities. Use maximum residual score as risk; compare original-image score
and context-free condition. Retain all 12 methods/predeclared primary without
TEST selection.

TRAIN80-only CDF/family, TRAIN45-only thresholds. Freeze before new TEST inference.
Budget/threshold definitions follow the contextual comparison; all 983/2204 care
words stay in denominators. No training, optimizer, gradient, adapted weights,
installation, downloads, word filtering. Historically explored TEST does not
become untouched validation in this follow-up.

## Runtime and qualification

Local BLIP2-FLAN-T5-XL snapshot0eb0d3b46c14c1f8c7680bca2693baafdb90bb28;
config/two weight shards checked against caption hashes. Current RTX3090, FP32,
no offload/AMP/quantization; batch 1 vision/2 answers. Staged path reuses visual
queries only, uses native processor with 32 image slots, and compares logits with
original Transformers4.47.1 forward.

Initial smoke da648d652c5b rejected before inference by an incorrect legacy-token
path check. Failure preserved; no OOM/substituted output. Corrected smoke
9e0381dc8341: PASS on 6 images, maximum logit error1,52588e-5 within tolerance2e-4,
unchanged state. Independent6-image/18-view/624-point audit PASS. Peak
15.954.364.928 allocated/16.091.447.296 reserved bytes:**not8GB-qualified**.
Time150,20s includes loading/checksums, not the entire pipeline.

Model sees224×224: small writing may disappear or mask borders may affect scores.
Residual-text presence does not verify spelling; results are experimental
hypotheses, not F1≥0,8 certificates.

## Command

```bash
.venvs/context-str-yolo-py311/bin/python -m scripts.run_yolo_residual_vqa \
  --mode approximate \
  --config configs/approximate/yolo_residual_vqa_v1.json \
  --paths configs/approximate/yolo_residual_vqa_local_paths_v1.json
```

`--smoke` isolates qualification;`--preflight` checks inputs. External outputs:
`yolo-residual-vqa-v1/{smoke,full}/HASH`, without duplicated checkpoints.
Scientific classification:`YOLO-RESIDUAL-VQA-001` in ambiguity log.
Inspired by [VQAScore](https://arxiv.org/abs/2404.01291), without claiming model/
benchmark reproduction; no new network trained here.

Full run completed;65 focused tests pass, including injected OOM propagation
with explicit failure receipt and no substitute results.No real OOM occurred.

## Completed results

Run:`/extra1/bmartelli/context-str-artifacts/yolo-residual-vqa-v1/full/83b4067d4722`.
Plan SHA86a514590bc09326e83653c7106c8313987551efd7f5490ae288d80afdf8a7df;
results SHA7f28edd97e321027b4bf7797d8f6717d35c0bbc21421f74ae4f038393c954ba3.
Independent auditPASS:425features,1,275 original/masked RGBviews,624routing
points.Counts, masks, Yes/No probabilities, TRAIN cuts and unchanged model state
verified.No omitted images; all 983/2204 care words remain in denominators.
TRAIN125 is the historical segmentation holdout, not the entire official TRAIN
set.DeepSolo has been fine-tuned on TotalTextTRAIN; keep these diagnostics apart.

At20% bypass,25TRAIN/60TEST images are retained.Good means per-image local word
F1>=.8; combinedF1 instead aggregates TP/FP/FN over both word branches.

| Method | Good TRAIN | Good TEST | Combined TEST F1 | TEST TN0 |
|---|---:|---:|---:|---:|
| Previous coverage |15/25|39/60|79.8346%|66.4373%|
| Previous softcrop consistency |16/25|48/60|79.9633%|66.6157%|
| Consistency + original/residual VQA drop, plain |14/25|42/60|79.9542%|66.6031%|
| Primary: same VQA drop with caption |13/25|37/60|78.7448%|64.9414%|

None of the ten new variants beats softcrop good-image counts at20/30/40/50%
bypass.TRAIN80-only selection retains softcrop at20% and prior coverage at
30/40/50%; no VQA method is promoted from TEST.The fixed primary loses18.33
purity points and 1.2184 combinedF1 points at20%; the plain control loses6good
images.All12methods and scalar cuts remain in results.json.

The plain-combination90%-target cut is0.6838599820802586: TEST77accepted/52good,
67.53%, not90%.The contextual primary has no valid90% cut with at least5TRAIN45
examples and therefore abstains on all images.Gray blank P(Yes) is0.5241plain;
with captions, the median is0.8694TRAIN/0.8884TEST despite absent pixel text.
This exposes language-prior bias, not calibrated OCR/F1 confidence.

Paired bootstrap10,000images, seed 20261021, fixed gates, no multiple-comparison
correction: primary ΔF1−1.2184points,95% interval[−2.8145;−0.0161].Plain control
ΔF1−0.0090, interval[−0.6257;+0.5996].These are exploratory fixed-gate intervals,
not a claim of superior paper reproduction.

Added runtime494.73s includes loading/state hashes and excludes cached upstream
inference.Peak allocation15,954,364,928bytes, reservation16,091,447,296; native-logit
error1.52588e-5, unchanged states.No8GB qualification or end-to-end speed claim.
Cached replay preserves the same plan/results hashes without new inference.
Full CSV with predicted words/TP/FP/FN and separate train/test is
analysis/results_table.csv; intervals/tie diagnostics are in
analysis/paired_comparisons_v2.json.Do not replace existing gates with these
features on the basis of the current measurements.
