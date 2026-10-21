# Select for actual OCR — approximate reproduction

ADASpot-GAP-OCR-SELECTION-001. The completed10epoch decoder trial chose epoch1
by native geometry on100TRAIN-calibration images. On the reused125image
evaluation set it localized649RGB words versus648, but exact OCR523versus524.
Independent audit passed. Geometry did not predict the desired OCR gain.

This is an explicit subsequent, post-hoc method-development step, not independent
confirmation. Neither earlier experiment nor its scientific source is modified.
Freeze the new rule/grid before reading calibration OCR: among every deployable
checkpoint (off baseline, saved best and saved final last), evaluate the existing
thresholds.5/.75/.9/.95/.99 and cut/preserve on the SAME100calibration images.
Epochs3/5 have calibration maps but no deployable saved parameters; they cannot
be selected without replaying training. No new training or capacity is added.

`ASSUMPTION`: maximize actual localized+exact-casefold read word count; ties
maximize end-to-end Hmean, then earlier epoch/higher threshold/cut. Freeze all
crop geometry before reading each image's transcripts. Count every care word and
all care proposals, including empty or duplicated readings. Use qualified genuine
PARSeq AR1 and original-RGB bboxIoU>=.5; exact casefold keeps punctuation. This
is an end-to-end objective, not the official paper recognition-accuracy claim.

`REPOSITORY_SPECIFIED`: source10full epochs/2580CUDAupdates and its saved best/
last, all original dataset/input/checkpoint hashes, same1030fit/100calibration/
125evaluation split, original GT reader and existing assembly variants. Keep
all21candidate per-image results, frozen crops, readings, choice and masks.
Hash-bound same-crop/same-qualified-AR1 prediction cache may be reused. Missing
recognitions execute on actual GPU; no GT is a model input or prediction filter
except the pre-existing declared care/ignore handling.

`HARDWARE_DEVIATION`: single actual CUDA GPU FP32/batch4, segmentation and PARSeq
loaded sequentially, memory and OOM saved without CPU fallback. No8GB/latency
claim. Evaluation images and labels load only after choice is saved.

`UNRESOLVED`: the criterion was redesigned after examining the preceding
evaluation failure; reused evaluation is exploratory. The100calibration images
were seen in M/K pretraining, so only Gap fitting was held out. Independent
transfer, multiple seeds and the full shared-feature AdaSpot remain unresolved.
If calibration selects off or the result fails on evaluation, report it rather
than force a positive epoch or cherry-pick an evaluation configuration.

Completed on RTX3090 at
`/extra1/bmartelli/context-str-runs/adaspot-gap-ocr-selection-v1-20261019`.
All21candidates on100calibration images/802care words. Baseline525exact/801
proposals; selected best epoch1/.9/preserve527exact/806proposals, Hmean
.6550218341→.6554726368. All3902unique calibration crops recognized on GPU;
selection saved BEFORE evaluation access. Same explicit original criteria and
reused source10epoch training, no new network training in this selection stage.

Full125evaluation images/983care words:

| Measure | Baseline | OCR-selected Gap |
| --- | ---: | ---: |
| RGB localized words | 648 | 649 |
| Native localized words | 650 | 651 |
| Exact localized+read words | 524 | 526 |
| Proposals | 1038 | 1041 |
| Recall | 53.3062% | 53.5097% |
| Precision | 50.4817% | 50.5283% |
| OCR Hmean | 51.8555% | 51.9763% |
| Recovered / damaged words | — | 2 / 0 |
| GT words touched by fused components | 202 | 204 |

Small+2word/+0.2035percentage-point recall improvement; Hmean+0.1208points.
No claim of robust or statistically established gain. This DOES NOT meet the
proposal's fusion-reduction hypothesis; the conservative support restoration
preserves crops but can restore any-pixel fusion contacts. Full independent TEST
and multiple seeds remain necessary before a strong generalization claim. The
geometry-selected cut arm's523words is retained as a negative result, not hidden.

Actual GPUFP32/batch4; selected segmentation inference peak2,167,429,120bytes
allocated/2,258,632,704reserved; calibration PARSeq138,519,040/155,189,248;
evaluation PARSeq235,095,040/270,532,608; no OOM. Existing qualified same-crop
AR1 cache reused where possible; every missing crop was inferred on GPU.

Independent audit PASS:2100calibration image-arm cases,250evaluation OCR and
250native cases, all candidate selection/populations, crop and scientific source
hashes checked independently. Result SHA256:
`a613738be46c7206b9a2327c66289316a901545060e73d20bec87ef409da82b8`.
The two gains are ORDER and HERE in total_text:972; no other evaluation word
changes. Thus the observed benefit is concentrated in a single image, not broad
evidence of transfer.70focused tests pass. Two initial fixture errors were fixed
and documented in the preceding experiment records; no runtime failure/OOM or
suppressed test in these completed runs. Three independent adaptation/selection
audits pass; historical source hashes remain unchanged.
