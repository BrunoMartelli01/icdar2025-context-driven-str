# YOLO routing with inference-only text coverage and resolution controls

**Approximate reproduction; exploratory experiment.** This user-authorized
extension tests whether a confidence gate can recognize incomplete image OCR
without sending nearly every image to DeepSolo. It also checks larger YOLO
inputs. It is not the paper's section3.5 router or an official benchmark score.
The paper's sections3.3/3.5 and Algorithm2, existing YOLO/PARSeq implementation,
frozen CLIP calibration, and U-Net map geometry were inspected first.
Scientific classifications are recorded under
`YOLO-TEXT-COVERAGE-RESOLUTION-001` in `PAPER_AMBIGUITIES.md`.

## Method and control

The control keeps the existing PARSeq/CLIP word-versus-background image score:
the minimum over word-level square roots of native-confidence and CLIP empirical
CDF ranks. CDFs are frozen on the same80 TRAIN calibration images. New coverage
is predicted foreground probability mass covered by the union of final word
rectangles, measured separately on the Text and Kernel U-Net maps. Rectangles
are exclusive integer XYXY in the original stored RGB frame, projected through
the existing square center-pad/nearest256 transform. Padded canvas is excluded.
Only NPZ `probabilities` are read by the feature extractor; GT, validity and
owner arrays must not enter inference. Probability threshold0.5 and diagnostic
uncovered-component area4 are declared assumptions. Empty masks are neutral;
empty OCR remains ineligible for bypass.

The gate mixes the confidence score and TRAIN-CDF rank of coverage with
weights0/0.25/0.5/0.75. Field/weight are selected on TRAIN80 at50% bypass by
combined microF1; equal performance retains the simpler confidence control.
The remaining45 TRAIN images are separate validation. Every resolution's
selector is sealed before any new TEST feature is computed. All300 TEST
images, including failed/empty readings, remain in the evaluation population.

Primary bypass is50%; additional controls are30% and70%. A batch rank budget
uses score and deterministic sample-ID tie breaking, with no GT, to achieve
exactly90/150/210 TEST bypass images. This is a transductive batch policy,
not a single-image fixed threshold. Scalar TRAIN thresholds are transferred
and reported separately with their actual TEST bypass counts. CDF ranks and
mixture scores are not calibrated probabilities of complete recognition.

All fallback outputs are the same425 complete frozen ViTAEv2-S+TextOCR
DeepSolo predictions. An image either retains all light-branch words or all
DeepSolo words; words from the branches are not merged. TRAIN refers to the
existing125-image segmentation holdout, not all1,255 training images; upstream
DeepSolo was fine-tuned on TotalTextTRAIN, so TRAIN is diagnostic only.

## Resolution and execution

YOLO uses letterbox scaling/padding to its configured `imgsz`, historically640,
with stride rounding and rectangular inputs. Test640/960/1280, recording actual
model tensor shapes. PARSeq always receives crops cut from the original stored
RGB image, subsequently resized by its native32x128 transform. There is no
additional JPEG compression or cropped-raster reuse from the detector input.

The640 branch replays frozen detector/OCR/CLIP outputs and asserts per-image
metric equality. TEST960 reuses the audited CPU FP32 YOLO/OCR cache. TRAIN960
and TRAIN/TEST1280 obtain new CUDA FP32 YOLO and CPU FP32 PARSeq outputs.
Final admission remains joint-confidence0.6 and same-reading NMS0.5. New
detector floor0.35 is below the necessary0.36 joint-confidence bound, keeping
every potentially eligible candidate. Native YOLO NMS/max detections match
the previous run. Device differences are recorded: the960/1280 comparison
does not isolate resolution from CPU/CUDA rounding effects.

Use the existing isolated YOLO environment, four CPU threads, deterministic
algorithms, TF32 disabled, YOLO CUDA FP32 batch1, PARSeq CPU FP32 batch32 and
then CLIP CUDA FP32 batch16. Detector/OCR and CLIP are loaded sequentially.
The actual host is PAPER with RTX3090; this does not qualify the primary8GB
RTX3070 hardware. Save input shapes, allocated/reserved CUDA peaks, co-resident
processes and explicit failures/OOM. No packages are installed or upgraded.

Run from the repository on PAPER:

```bash
.venvs/context-str-yolo-py311/bin/python -u -m scripts.run_yolo_coverage_resolution --mode approximate --check
.venvs/context-str-yolo-py311/bin/python -u -m scripts.run_yolo_coverage_resolution --mode approximate --smoke
.venvs/context-str-yolo-py311/bin/python -u -m scripts.run_yolo_coverage_resolution --mode approximate
```

Scientific and host-path profiles are separate under `configs/approximate/`;
neither affects STRICT. Results, original crop hashes, maps, intermediate
features, frozen fits, per-image branches, code snapshot and execution receipts
are saved outside Git, under the explicit output profile. Repeated execution
checks sealed contracts and source/input hashes before cache reuse.

## Evaluation and current evidence

Use the existing within-image exact Unicode-casefold word multiset metric,
punctuation retained, no spatial IoU. All983 TRAIN and2,204 TEST care words
remain. Ignore is evaluation-only: nonempty native256 support wholly outside
the validity mask is neutralized; light branch uses exclusive rectangles and
frozen DeepSolo retains its original polygon treatment. Report precision,
recall, F1 and requested TN0=TP/(TP+FP+FN), prediction counts, complete accepted
images and actual bypass. These are not interchangeable with official
polygon WordSpotting F1 or the target paper's incompletely specified accuracy.

The complete run finished all125 TRAIN /300 TEST images at every resolution,
with no failed or omitted sample. The primary640 gate selected Text coverage
with weight0.25 on TRAIN80. At960 calibration retained confidence-only;1280
selected Kernel coverage with weight0.25. No TEST correctness selected these
weights. At the50% TEST budget,150 images bypass DeepSolo at every resolution.

| Gate / YOLO resolution | TRAIN125 TN0 | Validation45 TN0 | TEST300 TN0 | TEST300 F1 | TEST care predictions |
|---|---:|---:|---:|---:|---:|
| Original confidence /640 |57.95%|59.08%|63.03%|77.32%|2,082|
| Coverage mixture /640 |58.01%|60.27%|63.39%|77.60%|2,121|
| Selected gate /960 |55.64%|57.29%|59.65%|74.72%|2,116|
| Selected gate /1280 |53.42%|55.22%|56.65%|72.33%|2,190|

The640 gain is small: TP+21, FP+18, FN-21, F1+0.2739pp and TN0+0.3648pp.
Conditional paired-image bootstrap,10,000 draws, fixed gates: TN0 difference
95% interval[-0.6091,+1.3578]pp; F1[-0.4582,+1.0253]pp. Both cross zero;
superiority is not established. Accepted-word precision rises81.14%→82.14%,
but complete accepted images stay41/150. The gate still does not certify
complete recognition. At1280 the selected coverage gate is worse than its
confidence-only control on TEST (TN0 56.65% versus57.24%).

|640 batch bypass|Bypass images|DeepSolo images|Final TN0|Final F1|Care predictions|
|---|---:|---:|---:|---:|---:|
|30%|90|210|65.38%|79.07%|2,139|
|50%|150|150|63.39%|77.60%|2,121|
|70%|210|90|59.48%|74.59%|2,113|

These are batch budgets, not fixed scalar cuts. Transferred640 thresholds
0.313315445/0.261307676/0.185572650 (targets30/50/70% on TRAIN80) accept
160/186/237 TEST images, with TN0 63.39/61.03/57.02% and F1
77.59/75.80/72.63%. In particular, the middle TRAIN threshold does not enforce
50% TEST bypass. The first threshold happens to accept53.33% TEST images;
that observed coverage does not recalibrate its score into a probability.
The common all-image ViTAE DeepSolo reference is TN0 67.16%, F1 80.36%.

Larger inputs do not improve the tested system. Original TEST dimensions are
larger than640 in192/300 images, larger than960 in129 and larger than1280 in50.
Actual1280 tensors include `[1,3,960,1280]`; the parser input remains32x128.
Thus the detector does downscale many originals at640, but simply increasing
its input size hurts with this checkpoint and admission policy. This does
not establish that all high-resolution detectors would behave the same way.

Sixty-three focused tests pass, including the independent counter/CDF checks
and a simulated CUDA OOM boundary that requires an explicit failure receipt
and propagated exception. Genuine GPU smoke completed one TRAIN/TEST image
per resolution; smoke values are not dataset results. No real OOM occurred.
The independent full audit verifies1,275 feature records and10,284 original
crop pixel hashes, map coverage, per-image word denominators, TRAIN references,
all rank-budget decisions and scalar thresholds. New-stage maximum allocated
VRAM was796,635,136bytes, reserved1,210,056,704bytes. This excludes generating
cached U-Net/DeepSolo outputs and is not full-pipeline throughput or8GB
qualification. No active compute job was co-resident at start.

Full run:
`/extra1/bmartelli/context-str-artifacts/yolo-coverage-resolution-v1/full/6c927d8d98d1`.
Results SHA256:`dceeff9307be7e6e11b6122ed750ab8f8d49f2c603e9d02e9ca37d578d7e5d79`;
plan SHA256:`0d74e407430c502a376e386550cd4024d241894443a3051153c44a07f113ce01`.
`analysis/report.md`, `analysis/comparison.csv` and `analysis/verification.json`
preserve the detailed result, uncertainty and independent checks. To rerun
the audit without model inference:

```bash
.venvs/context-str-yolo-py311/bin/python -u -m scripts.analyze_yolo_coverage_resolution --run /extra1/bmartelli/context-str-artifacts/yolo-coverage-resolution-v1/full/6c927d8d98d1
```
