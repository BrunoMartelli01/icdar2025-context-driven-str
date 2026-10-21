# YOLO: paired CLIP / paper-router comparison

**Approximate reproduction; exploratory comparison.** The authors' missing
U-Net-only score does not prevent a controlled router comparison: hold YOLO,
PARSeq, images, annotations and DeepSolo constant, and change only the gate.
It does prevent attributing their published pipeline score to their router.
Authority inspected: local `papers/2503.15639v2.pdf`, sections3.3/3.5,
Algorithm2 and Table5. Decisions: `YOLO-PAIRED-ROUTERS-001` in
`docs/PAPER_AMBIGUITIES.md`.

The existing TextOCR YOLOv8s640 direct-box model, joint-confidence0.60 and
same-reading NMS0.5 produce exactly the previous2,205 TEST outputs. Reuse all
frozen CPU FP32 PARSeq TEST readings; generate the missing125 TRAIN YOLO/OCR
samples. Keep original figure4_single image-only BLIP2 captions, whole-image
PARSeq T1 and the same425 complete DeepSolo R50 branches. Both routers see
every final YOLO word, including evaluation-ignored words. They replace all
outputs of a rejected image with full-image DeepSolo; there is no word mixing.

The paper baseline chooses T1/T3 by MPNet cosine and tests the selected
candidate's0.6S+0.4L. T1 must pass its own threshold and produces one raw
sequence. T3 joins all YOLO words. CLIP uses the previously fixed B16
word-versus-background prompts and minimum over sqrt(native-rank × CLIP-rank).
Fresh CDF references and six coverage cuts are fit only on TRAIN80; the
remaining45 images do not select a family. The sealed threshold receipt
records **zero TEST features at freeze**. No TEST label fits a new router.
The existing YOLO selection family had already been explored on TEST, so this
is not untouched validation. TRAIN125 is the earlier held-out subset for our
segmentation, not all1,255 TRAIN images; upstream DeepSolo was fine-tuned on
TotalTextTRAIN, so TRAIN results remain diagnostic.

## Common local metric

Within each image, match whole words one-to-one by exact Unicode casefold,
with punctuation retained, without bbox IoU. T1 is one unlocalized prediction,
without splitting or an invented box. Keep all983 TRAIN /2,204 TEST care words;
empty predictions leave FN and invalid envelopes remain unmatchable FP.

Ignore is evaluation-only and follows the preceding routing experiments:
native256 letterboxed support must be nonempty and entirely invalid to be
neutralized. YOLO uses the exclusive predicted rectangle; DeepSolo retains its
original polygon support. T1 is always care. Report precision TP/(TP+FP),
recall TP/(TP+FN), F1 and requested **TN0=TP/(TP+FP+FN)**.

These figures must **not** be mixed with the older YOLO report's75.08% F1:
that report removes Unicode punctuation/symbols, uses spatial matching and
neutralizes polygon-overlap>0.5. Here the audited detector/OCR outputs are
unchanged; the protocol is different and is common to both routers/fallback.
YOLO's spatial exact-casefold diagnostic is separately saved in results.json.

## Results at equal fallback budgets

Exact62 TRAIN /150 TEST acceptances are descriptive rank budgets, with
GT-independent sample-ID tie breaking. They are not transferred scalar cuts.

| System | TRAIN F1 | TRAIN TN0 | TEST F1 | TEST TN0 |
|---|---:|---:|---:|---:|
| YOLO/PARSeq only |64.24%|47.32%|68.83%|52.47%|
| YOLO + paper baseline + DeepSolo |72.64%|57.04%|73.05%|57.54%|
| YOLO + CLIP + DeepSolo |72.76%|57.18%|73.45%|58.04%|
| DeepSolo on every image |79.01%|65.30%|75.74%|60.95%|

At150 TEST acceptances, CLIP improves numerically by0.40 F1 points or
0.50 TN0 points over the paper baseline. Conditional paired-image bootstrap,
5,000 draws: TN0 difference IC95% **[-1.67,+2.59] percentage points**.
This does not establish superiority. Native confidence and joint-confidence
controls are in comparison.csv; they are competitive at several budgets.
The paper baseline is better at the small30-image budget. No family or
deployment threshold is selected using these TEST observations.

CLIP accepts700 care words at81.14% precision; the paper baseline accepts
1,075 at76.28%. CLIP's final system has TP1530/FP432/FN674, compared with
paper TP1556/FP500/FN648. Thus CLIP removes more FP but also loses correct
words. Complete accepted images are41/150 for CLIP and28/150 for paper;
high word precision does not certify that all scene text was recovered.

| Accepted TEST images | CLIP final TN0 | Paper final TN0 |
|---:|---:|---:|
|15|60.83%|61.37%|
|30|60.53%|61.44%|
|75|59.27%|59.92%|
|150|58.04%|57.54%|
|225|54.57%|53.60%|
|270|53.19%|51.86%|

## Real scalar thresholds transferred from TRAIN

The CLIP half-coverage TRAIN80 cut is0.1683146048116919. It accepts68/125
TRAIN and189/300 TEST images: TEST F1=72.90%, TN0=57.35%,1,969 final care
words. The paper half-coverage cut0.2630696011697477 accepts65/125 TRAIN and
197/300 TEST: TEST F1=70.92%, TN0=54.94%,2,077 final care words. These use
different fallback counts and must not be called the equal-cost comparison.
Distribution shift means a TRAIN50% coverage cut need not accept50% on TEST.
All six actual scalar cuts and transferred counts are in the artifacts.

Original paper tau0.8 accepts **zero** YOLO/T1 images, leaving DeepSolo on all
300 TEST images. Its resulting75.74% F1 is the fallback score, not evidence
of a useful selector or replication of the paper's claimed fallback savings.

## Headroom and paper comparison

An exact **GT-only oracle**, maximizing micro F1/TN0 between the two frozen
branches, gives TEST TP1703/FP355/FN501, F1 **79.92%**, TN0 **66.55%**,
with135 YOLO images and165 DeepSolo. Fixed-budget oracle results are also
saved. This is an upper bound, not a deployable router or a measured result
of CLIP. There is room above either standalone branch, but current semantic
confidence does not recover that selection reliably. A useful next selector
should estimate the relative benefit of using YOLO versus DeepSolo and
missed-word risk, rather than only transcription plausibility.

Table5 labels its TotalText values **recognition accuracy**:78.4% for
ResNet50 and79.8% for ViTAEv2-S. Its matching, denominator and ignore protocol,
exact PARSeq-S and DeepSolo checkpoints remain unresolved. Our F1/TN0 cannot
establish beating those numbers, including the numerically higher oracle.
The legitimate result is a local paired detector-controlled comparison.

## Execution and qualification

```bash
.venvs/context-str-yolo-py311/bin/python -m scripts.run_yolo_router_comparison --mode approximate
.venvs/context-str-yolo-py311/bin/python -m scripts.audit_yolo_router_comparison --root /extra1/bmartelli/context-str-artifacts/yolo-paired-routers-v1/full
.venvs/context-str-text-detectors-py311/bin/python scripts/audit_yolo_router_comparison.py --root /extra1/bmartelli/context-str-artifacts/yolo-paired-routers-v1/full --render-only
```

Isolated approximate scientific/path profiles:
`configs/approximate/yolo_router_comparison_v1.json` and
`configs/approximate/yolo_router_comparison_local_paths_v1.json`. No package
installation, downloads or new DeepSolo/caption inference. Generated artifacts
remain outside Git at
`/extra1/bmartelli/context-str-artifacts/yolo-paired-routers-v1/full/`:
resolved config, source snapshots/hashes, per-image TRAIN OCR and all425
CLIP/MPNet features, frozen TRAIN thresholds, word branches, JSON/CSV/Markdown
comparison, curves, independent audit and supplemental fixed-gate bootstrap.

RTX3090 sequential FP32 YOLO CUDA batch1, PARSeq CPU batch32, CLIP CUDA
batch16 and MPNet CPU are explicit `HARDWARE_DEVIATION` choices. Seed20261012,
four CPU threads, deterministic algorithms, TF32 off. Actual YOLO input tensor
shapes include the first warmup; PARSeq32×128 and CLIP224×224 shapes/batches
are recorded. Peak allocated/reserved: YOLO169,325,568/213,909,504 bytes;
CLIP796,635,136/1,090,519,040. No OOM or failed sample in the completed run;
these figures do not qualify the entire paper pipeline on8GB hardware.

**67 focused tests pass.** The genuine one-image-per-split smoke completed;
full independent audit validates425 features,3,227 selected RGB crops,
all425 fallback word metrics and128 policy points. A first smoke failed
because the wrapper omitted `Namespace.output_root`; it was corrected and
rerun in a separate immutable smoke-v2 directory. A later test collection
failed due to duplicate test filenames; the runner test module was renamed
before the successful67-test run. Both development failures are recorded;
no failed experiment image is removed from a denominator.

Frozen full plan SHA256:
`7de57915ebf6e2dd8abf7da1eaefa6880fc9ccfc2bddc767be597b937d8e3eab`.
Results SHA256:
`4d452a6462c199d94ddd630db99d71c4230d2e4f63cc792085e9fd90a9681de1`.
