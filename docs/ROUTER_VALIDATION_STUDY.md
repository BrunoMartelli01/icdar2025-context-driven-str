# Router generalization and held-out validation calibration

**Approximate reproduction / user-authorized research extension.** The user
requests evidence about generalization and thresholds chosen on validation.
Paper Section3.5/Algorithm2, existing token training, score preprocessing and
the qualified U-Net training contracts were inspected before implementation.
This trained router remains separate from the paper's training-free rule.

## Findings motivating the new protocol

The previous cuts used predictions held out from individual CV heads, rather
than their fitting examples. They were subsequently applied to a different
head refitted on the entire1253-image population. That model change invalidates
the assumption that raw score calibration is preserved.

More substantially, the frozen U-Net's qualified1130/125 split was recoverable
from its manifest and original SHA256 ranking. Excluding the two known exact
TRAIN/TEST copies leaves1128 expert fitting images and125 expert holdout images.
There are no RGB overlaps between these groups or the remaining TEST population.
The expert used the125 holdout for checkpoint/model selection, but not gradient
updates. Historical reuse limits any claim of fresh validation.

Under the current lexical alphanumeric metric, U-Net+PARSeq F1 is75.4657% on
its1128 fitting images with0/1128 (0%) DeepSolo,62.3213% on125 holdout images
with0/125 (0%) DeepSolo, and63.0532% on TEST300 with0/300 (0%) DeepSolo.
Thus the large expert quality gap already occurs inside official TRAIN before
the router. Treating the mixed1253 population as a homogeneous proxy for unseen
expert outputs was optimistic. Precise counts and confound checks are saved
by the diagnostic study; these observations alone do not isolate every cause.

## Predeclared correction

- Fit U-Net and YOLO token heads only on the1128-image fit group.
- Evaluate5/10/20 checkpoints on the125 held-out validation group and select
  the earliest best combined micro F1 at50% acceptance.
- Choose scalar cuts only from that same checkpoint's validation scores.
  Bind every cut to the actual model hash. Never refit after calibration.
- Use three fixed initialization seeds; the first is primary. Others diagnose
  variance and are not selected by TEST performance.
- Freeze all decisions before scoring TEST with the exact selected weights.
  Keep batch-quota diagnostics separate from transferred scalar thresholds.
- Preserve frozen experts, OCR filter0.7, CLIP/BLIP inputs, improved DeepSolo
  outputs and evaluation rules. No fresh foundation or DeepSolo inference.

Configurations: `configs/approximate/router_validation_v1.json` and
`configs/approximate/router_validation_local_paths_v1.json`.
Runner: `scripts/run_router_validation_study.py`.
Boundary tests: `tests/recognition/test_router_validation.py`.

## Evidence and qualifications

Additional diagnostic controls compare expert fitting/holdout/TEST cohorts,
word-count and text-size strata, raw versus filtered OCR, and OOF/refitted
predictions for identical images. The new validation protocol tests whether
keeping identical weights restores acceptance calibration. A single frozen
expert and historically reused validation do not establish a universal causal
explanation or an unbiased new benchmark.

Every policy reports DeepSolo images/total/percentage. Counts are logical
routing on cached experts; actual new DeepSolo inference is0. FP32 small token
heads use the observed RTX3090 with deterministic math attention, cached CPU
features, batch8 and logged allocated/reserved peaks. These choices are
HARDWARE_DEVIATION; the experiment does not qualify online8GB operation.

The completed run and controls are reported below. Qualification passed54
focused tests, a real24/8/4-image smoke with154 checks, the full-run audit
with462 checks, and4477 supplemental checks. There are no full-run failure
or OOM records. An initial supplemental-summary export failed on a string
versus Path conversion; it was corrected before any summary was written.
This did not change inference, training or metrics.


## Completed evidence and diagnosis

External root: `router-validation-v1/full/9dc6c9951d67`.
Portable summary: [completion manifest](../provenance/router_validation_20261027_completion.json).
Full [report](/extra1/bmartelli/context-str-artifacts/router-validation-v1/full/9dc6c9951d67/report_v2/report.md)
and [396-row CSV](/extra1/bmartelli/context-str-artifacts/router-validation-v1/full/9dc6c9951d67/report_v2/validation_thresholds.csv).
FIT scores in the report are resubstitution diagnostics; only validation selects
epoch/cuts, and TEST evaluates the unchanged selected model.

The controls support two distinct problems. Expert-fitting U-Net outputs are
substantially better than expert-unseen outputs. The router therefore learns
relative-benefit labels mostly from an optimistic expert population. Original
router OOF validation holds examples out of the router, but most examples remain
fitting images for the frozen U-Net. Separately, transferring a raw cutoff from
those fold heads to a different full-refit head changes score calibration.

### Isolating recognition from crop extraction

The GT-crop control matches125 expert-fitting images to the validation word-count
bins. PARSeq paired recognition accuracy is815/891=91.4703% on those fitting
images,892/983=90.7426% on validation and2071/2204=93.9655% on TEST. These
controls send0/125,0/125 and0/300 images to DeepSolo (0%). They are annotated
crop recognition diagnostics, not deployable pipeline F1.

On the same matched125 fitting images the actual U-Net/PARSeq pipeline has
76.5027% lexical F1 with0/125 DeepSolo (0%), versus62.3213% on validation
with0/125 (0%) and63.0532% on TEST with0/300 (0%). The frozen reader does
not show the pipeline's large seen/unseen collapse when given correct rectangles.
This localizes the principal bottleneck to region extraction/crop quality.

The gap also persists within GT word-count and relative-height strata. For
example, with at most3 GT words, fitting/validation/TEST U-Net F1 is82.46%,
61.22%,60.57%, with0/324,0/30,0/80 DeepSolo images respectively (0%).
Thus global density or smaller TEST text alone does not account for the gap.

### Spatial coverage partition

A separate deterministic maximum one-to-one bbox IoU>=0.5 diagnostic finds raw
crop coverage of81.0622% on expert-fitting images,65.5137% on validation and
68.8748% on TEST, with0/1128,0/125 and0/300 DeepSolo images (0%). Failed
coverage includes missed, merged and shifted envelopes. It is not the primary
lexical metric, and does not mean that no text pixel is inside a failed envelope.

For all2204 TEST care words, nested edge-set matching partitions the diagnostic
denominator into686 without geometric coverage,205 geometrically covered but
without a correct raw reading,62 correct raw readings removed by the confidence
filter, and1251 retained spatially correct readings. These are GT-based error
controls with0/300 DeepSolo (0%), not inference inputs. Primary lexical TP is
1268 because that metric does not require spatial matching.

The confidence sweep retains0.7 in the deployed source. TEST lexical F1 is
57.253%/62.240%/63.053%/62.268% at sequence cuts0/0.5/0.7/0.9, all with
0/300 DeepSolo (0%). Adjusting this filter within the predeclared grid does
not recover the localization gap.

### What changed in the router protocol

The same original1253 images and count targets expose the calibration change:
U-Net mean predicted DeltaTP bias is+0.1860 words in router OOF predictions
and+0.8236 after full refit. The predicted utility bias increases from+0.1007
to+1.3542 in the source score units. These are pre-routing residual diagnostics;
no routing policy or new DeepSolo call is evaluated in this comparison.

The correction fits only1128 images, selects a checkpoint using125 held-out
validation images, and derives cuts from that exact checkpoint. There is no
refit after calibration. The main seed selects20 epochs for U-Net and10 for
YOLO. Two additional predetermined seeds check variance; none is chosen using
TEST. All six checkpoints verify fitting membership and threshold/model hashes.

### Validation50% operating point

| Population | Expert | Cut | Accepted light | DeepSolo images/total (%) | Hybrid F1 (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| VALIDATION | UNET | -0.667425 | 62/125 | 63/125 (50.40%) | 75.5776 |
| VALIDATION | YOLO | -1.040588 | 62/125 | 63/125 (50.40%) | 77.5378 |
| TEST | UNET | -0.667425 | 147/300 | 153/300 (51.00%) | 78.0138 |
| TEST | YOLO | -1.040588 | 147/300 | 153/300 (51.00%) | 80.8300 |

The two main validation-derived cuts both accept147/300 TEST images and send
153/300 to DeepSolo (51.00%). The quota now transfers closely without a TEST
quantile or a TEST-selected threshold. U-Net F1 is78.0138%; YOLO80.8300%,
both at that153/300 (51.00%) DeepSolo cost.

At exactly150/300 (50.00%) TEST images sent to DeepSolo, the new descriptive
rankings give77.8892%/80.8400% U-Net/YOLO F1, versus78.1122%/80.9558% for
the earlier routers at the same150/300 (50.00%) cost. Calibration is corrected;
the selector ranking has not materially improved. The frozen experts and
eligibility still give the same conditional150/300 (50.00%) GT oracles,
82.5465% for U-Net and84.2983% for YOLO.

Across the three predetermined seeds, every validation50% cut is applied to its
unchanged model. The complete report retains each seed's actual acceptance,
DeepSolo count/percentage and F1; no best TEST seed is promoted.

## Consequences and remaining causal work

Improve U-Net localization/crop quality and train the gate using expert-unseen
predictions, such as U-Net out-of-fold predictions. This would align router
fitting labels with validation/TEST expert quality. It requires additional
expert training rather than further cut tuning. The present controls isolate
the main failing stage and the calibration protocol; they do not uniquely
separate mask overfitting, instance assembly, crop geometry and typography.

The checkpoint's parent Text+Kernel run and its continuation have the same
qualified holdout identity; the parent manifest hash matches the continuation
contract. Shared scientific code hashes match the original cache run. GT-crop
PARSeq is frozen CPU FP32/batch32; no foundation or DeepSolo model is retrained.
Small token-head memory is recorded per checkpoint. Measured maximum selected
head allocated/reserved bytes across all six seeds:

- UNET: 123864576/138412032 bytes, FP32, batch8,980 CLIP patch tokens and32 cached BLIP queries plus padded word tokens.

- YOLO: 140251136/163577856 bytes, FP32, batch8,980 CLIP patch tokens and32 cached BLIP queries plus padded word tokens.


At the validation50% cut, the primary U-Net hybrid loses4.6183pp against
all-DeepSolo on validation (63/125 images sent to DeepSolo,50.40%, versus
125/125,100%) and4.6402pp on TEST (153/300,51.00%, versus300/300,100%).
This operating point's deferral penalty is now similar across validation and
TEST. It does not improve the underlying expert coverage or oracle gap.

The20-crop cap is an additional explicit constraint: TEST has10 images with
more than20 care words, so even perfect one-word crops under that cap would
miss at least128 of2204 words. This is a capacity lower bound, not a causal
assignment of128 observed failures to truncation; actual spatially uncovered
TEST words are686. The cap alone cannot explain the whole coverage deficit.
