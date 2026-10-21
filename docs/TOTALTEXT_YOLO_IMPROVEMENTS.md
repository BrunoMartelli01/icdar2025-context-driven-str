# Exploratory YOLO failure analysis and improvement battery

This is an **approximate reproduction**, exploratory TEST experiment. Baseline
TEST failures motivated method families. Numeric controls are generic and
declared in `totaltext_yolo_improvements_v1.json` before the new outcomes.
There is no training on TEST and no claim of an untouched final validation.
Paper section3.3/Algorithm1 and section3.5 were inspected along with the
existing YOLO/direct-PARSeq implementations. The paper uses component crops,
padding and PARSeq-S; this experiment uses an external object-box detector and
the repository-qualified PARSeq checkpoint, not an established PARSeq-S.

## Baseline causes

At detector score0.40, 1,574/2,204 GT words are correct. The remaining630 are:

| Failure | GT words |
| --- | ---: |
| A valid lower-score box exists but score0.40 removes it | 345 |
| Only partial/loose boxes, best IoU below0.5 but above0.1 | 131 |
| No nearby predicted box (best IoU below0.1) | 37 |
| A valid high-score box exists but reading/matching fails | 117 |

Of the117 reading failures,62 best boxes cover less than90% of the GT envelope.
This diagnostic uses GT geometry for analysis only. One-character words are a
weak group (6/54 correct); the third-party training conversion excluded them.

Of793 unmatched predictions at score0.40,270 overlap ignored annotations,
186 have partial/loose/multiword overlap with care GT,188 have no nearby care
GT,144 overlap care boxes but have wrong readings, and5 duplicate a correct
reading. The primary historical metric counts ignored-overlap predictions;
this battery retains it for comparison. Ignored GT never filters inference.
Some visual examples also show incorrect historical transcriptions (e.g.
`ROBRTSON` where the image reads `ROBERTSON`, and `h` where it reads `Hong`).
Labels are preserved and these are not repaired using TEST outcomes.

Baseline analysis and English contact sheets are at
`/tmp/yolo-failure-improvements/baseline_failures.json` and sibling JPG files.

## Fixed interventions

1. **Admission:** geometrically combine YOLO confidence with balanced PARSeq
   character confidence. Test joint thresholds0.5/0.6; raw YOLO score controls
   remain0.05/0.25/0.40. Reject empty normalized readings only in explicitly
   named nonempty/joint arms. The joint value is an uncalibrated score.
2. **Duplicates:** NMS at IoU0.5, either purely geometric or requiring equal
   normalized OCR readings. Distinct adjacent words are not grouped by text
   alone; spatial overlap is required.
3. **Crop views:** original boxes scoring>=0.25 get5% shrink,5% expansion,
   and15% expansion along each box axis. Report fixed variants and choose a
   view by balanced PARSeq confidence, with baseline winning ties. Evaluation
   keeps the original detection bbox, so padding cannot inflate localization
   scores. This does not rectify curved/rotated text.
4. **Detector detail:** whole-image960 inference and a separate union of
   original640 predictions with four640 overlapping windows (20% overlap,
   tile score floor0.25). Tile coordinates map to the original stored raster;
   deterministic NMS0.5 merges duplicate boxes. This implements a small fixed
   sliced-inference control inspired by [SAHI](https://docs.ultralytics.com/guides/sahi-tiled-inference),
   without installing SAHI or claiming its exact implementation. Resolution
   and NMS controls are supported by [Ultralytics](https://docs.ultralytics.com/modes/predict).

The detector/recognizer weights are unchanged and hash checked. Every new
detector and OCR shard freezes geometry, crop pixels, model scores and
execution receipts before GT evaluation. Model processes are sequential,
FP32 batch32, CPU threads4, seed0; CPU use is a `HARDWARE_DEVIATION`. GPU peaks
and OOM errors are recorded where CUDA is explicitly used. No silent device
fallback occurs. All300 images and2,204 care GT remain in the denominator.
The reported micro precision/recall/Hmean is the same local normalized-word
bbox proxy at IoU0.5; official ignore-aware polygon Hmean remains separate.

Run the complete suite on a GPU host:

```bash
bash scripts/test_totaltext_yolo_improvements.sh
```

For CPU, add `--device cpu`. The default output is declared by the local paths
profile and separated by device. `--stage cached` evaluates only admission and
NMS on the frozen baseline, without new inference. `--limit 1` is genuine smoke.

## Measured TEST results

All300 images and2,204 care GT completed. The initial39 controls required21,991
new PARSeq readings,131.89seconds recorded YOLO inference and453.30seconds
recorded new OCR processing on CPU FP32. Twelve focused tests and genuine
smoke pass. An independent sparse matching audit reproduced all39 controls
and verified35,228 crop pixel hashes, coordinates, source images and receipts.

| Method / control | Outputs | Correct | Precision | Recall | Local Hmean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original640, YOLO score0.40 | 2,367 | 1,574 | 66.50% | 71.42% | 68.87% |
| 640, joint0.60 + same-reading NMS | 2,205 | 1,597 | 72.43% | 72.46% | 72.44% |
| 960, joint0.60 + same-reading NMS | 2,391 | 1,556 | 65.08% | 70.60% | 67.73% |
| Four windows + whole640, joint0.60 | 3,168 | 1,607 | 50.73% | 72.91% | 59.83% |
| Best-confidence crop view, joint0.60 + same-reading NMS | 2,319 | 1,616 | 69.69% | 73.32% | 71.46% |
| Fixed5% crop shrink, YOLO score0.40 | 2,367 | 1,550 | 65.48% | 70.33% | 67.82% |
| Fixed5% padding, YOLO score0.40 | 2,367 | 1,565 | 66.12% | 71.01% | 68.48% |
| Fixed15% padding, YOLO score0.40 | 2,367 | 1,404 | 59.32% | 63.70% | 61.43% |

Resolution/window controls do not improve this metric. At score0.40, geometry
coverage is1,691 original640,1,701 whole960 and1,827 windows, whereas correct
readings are1,574/1,547/1,597. More covered GT is accompanied by more OCR
failures and false positives. At score0.05,960 increases geometry coverage by
only3 words and correct readings by2, with1,122 more outputs. Window fusion
also loses some original low-score crops through NMS. Fixed shrink/padding
loses24/9/170 correct words; these interventions should not become defaults.
Confidence-chosen views have a modest benefit, not enough to justify all extra
readings for the highest Hmean. The maximum recall control is crop_best at
score0.05:1,845/2,204 correct (83.71%) among8,318 outputs.

The least expensive improved family is original640 plus geometric-mean
detector/OCR confidence. Exact-reading NMS removes9 further outputs without
losing correct words at joint0.60. An algebraic pre-OCR bound follows from
`joint=sqrt(detector*OCRquality)` and `OCRquality<=1`: detector score must be
at least `joint_threshold²`. For joint0.60, detector floor0.36 leaves2,564 crops
for recognition instead of the complete8,318-crop diagnostic pool. A cached
audit confirms identical selected IDs in all300 images; this is an exact
necessary-condition prune, not a new tuned threshold or new neural run.
The optimized original YOLO-score0.40 baseline would itself need only2,367
recognitions. Performance timing of this reduced execution is not measured.

### Additional cached scale agreement

After the initial battery, a separate exploratory config declared exact
normalized-reading agreement and box IoU>=0.5 across640/960. Hungarian matching
uses each detection once, scores a pair by the geometric mean of its joint
scores, and emits original640 coordinates. The same0.5/0.6 controls and
same-reading NMS0.5 are tested. No new inference is needed for this analysis.
The full one-to-one/score/geometry/sparse-matching audit and its focused unit
test pass. The shell command also runs this second config after new inference.

| Agreement threshold | Outputs | Correct | Precision | Recall | Local Hmean |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 2,290 | 1,629 | 71.14% | 73.91% | 72.50% |
| 0.60 | 1,989 | 1,533 | 77.07% | 69.56% | 73.12% |

Agreement0.60 reduces false positives, at the cost of64 correct words relative
to640 joint0.60. Agreement0.50 offers more recall. The earlier compact DeepSolo
slot pipeline has1,499/1,886 outputs, local Hmean73.30%. This comparison uses
the same local metric, not official polygon spotting; two-scale YOLO inference
and additional OCR have a cost and are not free improvements.

Reports: `/tmp/yolo-failure-improvements/v1/cpu/full/results.json`,
`scale_agreement/results.json` and their sibling audit files. Base report SHA256
`6c13ab1aa10e8be5cfd1dc9aa7d45a91d9e97170a19fb09f2f3765cbf0c7d539`;
agreement SHA256 `b6f818612965140b312ab16ef6809c59fac8b51d6bbc73d11c36e73bf067bd6b`.
GT-free prediction exports, source snapshots, per-image receipts, comparison
PNG/SVG and the baseline failure contact sheets are preserved. Selecting an
operating point after reviewing TEST is explicitly exploratory; calibration
and validation on independent data are still needed for transfer claims.
