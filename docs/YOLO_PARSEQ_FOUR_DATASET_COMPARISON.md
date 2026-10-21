# YOLO/PARSeq: four TEST datasets and per-image compute estimates

**Approximate reproduction / external object-detector experiment.** This adds
the completed Total-Text direct640 run to the prior ICDAR13/ICDAR2015/CTW1500
controls. Paper sections3.3/3.5 and its Table4 were inspected. The qualified
PARSeq checkpoint is not established as the paper's PARSeq-S artifact, and the
paper's FLOP protocol remains unresolved. Scientific record: YFC-001.

```bash
bash scripts/test_yolo_four_datasets.sh
```

The command reuses frozen accuracy outputs, rescans Total-Text thresholds and
profiles arithmetic on CPU using existing packages/weights. No installation or
download occurs. The default output is
`/tmp/yolo-parseq-four-dataset-comparison-v2/`: `results.json`, `comparison.csv`,
`compute_profile.json`, per-image Total-Text controls and independent audits.
`--stage thresholds` skips model profiling. Use `--output` for a fresh external
output directory when changing config/code. Approximate scientific and path
profiles are isolated under `configs/approximate/`.

## Accuracy comparison

All results use the same original640 detector boxes, direct crops without added
padding, qualified PARSeq and joint confidence/same-reading NMS. Text matching
is exact after casefold and Unicode punctuation/symbol removal; whitespace is
preserved. Original-axis bbox IoU>=0.5, maximum one-to-one matching and micro
counts apply. Ignored GT is excluded, predictions over ignored regions remain
FP. This is a **local bbox proxy**, not official polygon Hmean. CTW uses full
lines and is a separate diagnostic.

Total-Text detector inference was RTX3090 FP32 and recognition CPU FP32; the
other three completed recognition runs used RTX3090 FP32. These source-device
differences are explicitly retained. Checkpoint/runtime identities agree.
Total-Text's physical care transcriptions and canonically clipped boxes match
the frozen GT exactly:300 images/2,204 care words/343 ignored/20 clipped polygons.
High-resolution, tiles and crop variants are excluded from this comparison.

| Dataset | Images | Correct at joint0.60 | Outputs | Precision | Recall | Hmean |
|---|---:|---:|---:|---:|---:|---:|
| ICDAR13 | 233 | 862/1,095 words | 1,038 | 83.04% | 78.72% | 80.83% |
| ICDAR2015 | 500 | 1,401/2,077 words | 2,336 | 59.97% | 67.45% | 63.49% |
| Total-Text | 300 | 1,597/2,204 words | 2,205 | 72.43% | 72.46% | 72.44% |
| CTW1500 | 500 | 1,106/2,687 lines | 5,245 | 21.09% | 41.16% | 27.89% |

Each scalar grid has95 values0.05..0.99, step0.01, for raw detector and joint
confidence. Per-dataset maxima and shared settings are selected using TEST;
they are exploratory and do not automatically change the original0.60 setting.

| Dataset | Hmean0.60 | Hmean0.62 | Maximum observed joint Hmean | Threshold |
|---|---:|---:|---:|---:|
| ICDAR13 | 80.83% | 81.31% | 81.96% | 0.69 |
| ICDAR2015 | 63.49% | 64.32% | 65.06% | 0.68 |
| Total-Text | 72.44% | 72.78% | 73.08% | 0.64 |
| CTW1500 lines | 27.89% | 27.93% | 27.93% | 0.62 |

The common0.62 improves all four baselines by the earlier max-min gain rule.
For Total-Text it returns1,568 correct/2,105 outputs; its individual0.64 maximum
returns1,547/2,030. These Hmean improvements remove some correct outputs along
with false positives. CTW's vocabulary/line mismatch is unchanged; see the
earlier `SCENE_TEXT_YOLO_THRESHOLDS.md` diagnosis.

## What one image costs

Count **2 FLOPs per multiply-accumulate (MAC)**. Included: dense Conv2d/Linear,
complete MultiheadAttention Q/K/V/output projections, and attention QK/AV matrix
products in both ViT and decoder. MHA projections use functional linear calls,
so counting only Linear module hooks would miss them. Fused ViT attention also
needs its matrix products counted. MHA children are excluded from other hooks
to prevent double counting. Profiling hooks do not change model outputs/weights.

Excluded: pointwise scaling/residual/bias operations, activations, normalization,
softmax, embedding/index operations, preprocessing, tokenizer logic, confidence
filtering and NMS. Thus these are **dominant-arithmetic estimates**, not complete
operator totals or exact paper Table4 FLOPs. They describe YOLO/PARSeq only.

Fused YOLO at square640/batch1 is **28.4327424 GFLOPs** under this convention.
The earlier28.65 THOP figure used a different supported-operator scope including
unfused normalization. Actual rectangular LetterBox tensors lower the mean to
17.06..21.24 GFLOPs in these datasets. All1,233 recorded shapes match the shape
formula; Total-Text's300 shapes are derived from the qualified single-image
Ultralytics8.1.0 auto LetterBox/stride32 preprocessing.

PARSeq depends on autoregressive early EOS and the longest decoding path in each
batch. Sample all original crops in four evenly spaced images per dataset,
using original per-image batch32 order, RGB32x128 and refine1. No GT chooses the
profiling images. Total16 images/264 crops are profiled on CPU FP32; all sampled
decoded strings match their cached CPU/GPU outputs. Per-crop sampled means are
6.58/7.59/7.36/6.99 GFLOPs respectively. A separate forced26-step single-crop
profile gives **8.207660544 GFLOPs**, a conservative counted-arithmetic budget;
it is not the deployed early-EOS path or an accuracy output.

For each dataset:

`estimated mean pipeline FLOPs = exact mean detector dense FLOPs + mean actual crop count * sampled mean recognizer dense FLOPs per crop`

| Dataset | Mean recognized crops/image | Mean YOLO G | Sampled PARSeq G/crop | Current pipeline estimate G/image | Hypothetical safe prefilter estimate G/image |
|---|---:|---:|---:|---:|---:|
| ICDAR13 | 11.05 | 21.12 | 6.58 | 93.8 | 52.6 |
| ICDAR2015 | 21.65 | 17.06 | 7.59 | 181.4 | 60.7 |
| Total-Text | 27.73 | 20.70 | 7.36 | 224.8 | 83.6 |
| CTW1500 | 25.19 | 21.24 | 6.99 | 197.3 | 100.8 |

The current pipeline reads **all proposals before the final filter**; final
output counts cannot be substituted for recognizer calls. The hypothetical
column uses `sqrt(detector_score)>=0.60` before OCR, retaining all currently
admitted candidates under frozen readings. It holds sampled per-crop compute
fixed; changed batch composition/EOS and numerical output equivalence require
an actual rerun. These are estimates across all TEST images, using sampled OCR
cost. Sampling ranges and per-image estimates are saved; they are not latency
measurements or statistical confidence intervals. Exact GPU conditional cost
for every original crop and target RTX3070 qualification remain unresolved.

## Verification and failures resolved

13 focused tests pass, including real Ultralytics LetterBox geometry and
independent Torch-profiler checks on Conv/Linear/MHA/ViT dense operations.
The first MHA/ViT assertions compared against all profiler FLOPs, which also
include48/80 attention scaling multiplies in the synthetic fixtures. The tests
now compare the declared dense scope; no count or neural output was changed.
The first Total-Text scan stopped on source annotation4's out-of-image border
point. Reusing the existing canonical clipping helper restores exactly the
previous frozen GT; no label is removed or denominator changed. The incomplete
v1 output is preserved and the qualified run uses a new v2 directory.

An independent normalization/confidence/NMS/IoU implementation plus SciPy
matching verifies all57,000 new Total-Text per-image controls and8,318 candidate
scores. Combined with prior audited234,270 controls, the four-dataset comparison
covers1,533 images/34,310 recognized crops. A second independent TorchDispatch
operator counter exactly matches **14,216,371,200 YOLO MACs** and
**4,103,830,272 PARSeq full-length MACs**. Its initial primitive-only audit missed
atomic inference-mode Linear/Conv2d/SDPA calls; including the actually observed
operator schemas resolves that audit failure without changing the module
counter. Dependency deprecation and checkpoint load warnings remain visible.
