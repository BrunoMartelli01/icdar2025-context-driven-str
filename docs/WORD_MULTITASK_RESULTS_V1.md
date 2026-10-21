# Completed word multi-task experiment v1

Result class: **approximate reproduction**, exploratory held-out TRAIN geometry.
All three branches completed 100 epochs and the independent replay of their
selected checkpoints. Completion was audited on 2026-10-08. The experiment root
is `/extra1/bmartelli/context-str-runs/totaltext-word-multitask-v1`.
The original checkpoints, reports, configurations and visualizations are preserved.

The strongest measured variant is **Text + Kernel**. Adding the center head with
the frozen v1 loss and post-processing configuration does not improve word F1.
This finding concerns this one seeded experiment and its validation protocol.

## Matched comparison

All branches share the qualified v4 targets, 1,130 training images, 125 validation
images / 983 care words, seed 202503, ImageNet V2 encoder, paired preprocessing,
Adam settings, 100 epochs, and the text BCE+Dice loss. Resolved contracts and all
100 training sample-order fingerprints match apart from the variant name.
Text is the new matched BCE+Dice control; the original historical BCE runs are
separate references in `comparison.json` and are not causal head controls.

Word precision, recall and F1 use global maximum-cardinality one-to-one matching
at native bounding-box IoU >= .50. Merge rate counts care words touched by a
prediction touching at least two words; split rate counts care words touched by
at least two predictions. Both rates divide by all 983 expected care words.
Touching means any shared native pixel. Entirely invalid predictions are ignored;
partially invalid predictions retain their whole boxes. No validation image is
omitted, and no validation word is lost in the native targets.

Percentages below use the predeclared watershed pipeline. The text-only variant
uses foreground connected components because it has no auxiliary maps.

| Variant | Selected epoch | Precision | Recall | F1 | Merge rate | Split rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Text, BCE+Dice control | 56 | 62.19% | 56.05% | 58.96% | 29.81% | 4.88% |
| Text + Kernel | 61 | 60.74% | 63.58% | 62.13% | 17.40% | 9.66% |
| Text + Kernel + Center | 66 | 55.09% | 62.16% | 58.41% | 26.65% | 10.89% |

Auditable counts are respectively 551/625/611 matched words, 886/1,029/1,109
care proposals, 293/171/262 words affected by merges, and 48/95/107 words
affected by splits. Checkpoint selection is validation word F1, then fewer merged
words, recall, macro foreground IoU and earliest epoch, in that order.

Text + Kernel gains **3.17 percentage points F1**, 7.53 points recall and reduces
merge rate by **12.41 points** versus the matched Text control. Precision falls
1.45 points and split rate rises 4.78 points. This supports better separation
with a measurable increase in fragmentation. The center variant loses .55 points
F1 versus Text and 3.71 points versus Text + Kernel. A general claim that center
supervision cannot help would require further evidence.

## Pixel metrics and existing post-processing ablation

| Variant | Foreground macro IoU | Foreground macro F1 | Foreground micro IoU | Foreground micro F1 |
| --- | ---: | ---: | ---: | ---: |
| Text | 68.69% | 80.29% | 66.86% | 80.14% |
| Text + Kernel | 67.19% | 79.09% | 65.90% | 79.45% |
| Text + Kernel + Center | 69.81% | 81.21% | 68.48% | 81.29% |

These are the existing valid-pixel macro/micro metrics, with a perfect score when
prediction and target are both empty. The center variant has the best pixel
scores while its word F1 is lower, illustrating why pixel scores alone do not
establish successful instance separation.

The already exported post-processing ablation uses the same selected checkpoint
and probability maps for each model; there was no extra training or inference:

| Variant | Connected components F1 | Kernel BFS F1 | Watershed F1 |
| --- | ---: | ---: | ---: |
| Text | 58.96% | 58.96% | 58.96% |
| Text + Kernel | 61.73% | 62.13% | 62.13% |
| Text + Kernel + Center | 56.89% | 58.32% | 58.41% |

The kernel-trained text map already improves word F1 with connected components;
seeded separation adds .40 points at this checkpoint. This is a diagnostic of
the selected maps, not a separately optimized or checkpoint-selected CC protocol.
For three heads, watershed exceeds CC by 1.52 points, but the complete pipeline
still falls behind Text + Kernel. Existing audit counters report only 62 retained
center peaks across 125 images at threshold .50 / NMS radius 3, 12 kernel
components split by centers, and zero center rescue markers. Those counters are
not center detection precision/recall. Calibration or supervision problems are
hypotheses; center markers and the center confidence landscape have not been
isolated from the effect of training in this experiment.

## Hardware, size and limits

Actual hardware is **NVIDIA RTX 3090** (`HARDWARE_DEVIATION` relative to the
development target RTX 3070), with sequential segmentation-only execution,
FP32, batch 8, RGB 512x512 and native maps 256x256. No OOM was reported. Training
peak allocated memory was 3.943/3.946/3.946 GiB; peak reserved memory was
4.986 GiB for each branch. This does not qualify execution on an RTX 3070.

Parameter counts are 58,155,081 / 58,155,146 / 58,155,211. The kernel and center
channels add only 65 and 130 parameters to the Text control. This preserves the
backbone's size without establishing that the existing 58M backbone is small
in absolute terms.

There is one seed, checkpoints are selected on the same repeatedly used TRAIN
holdout, and no confidence interval or independent TEST evaluation is available.
The experiment does not measure PARSeq recognition accuracy. These results do
not reproduce the paper's official polygon detection protocol, Table 3 or Table 5,
and do not resolve existing STRICT evidence gates.

## Verification and provenance

The saved `comparison.json` was rebuilt exactly using the existing comparison
function, including the two historical references. SHA256s of selected/latest
checkpoints, encoder, scientific sources, dataset manifest, preprocessing config,
375 proposal JSONs, 375 map archives, 375 prediction PNGs and 375 target PNGs
were checked. Per-image geometry, aggregate word metrics, pixel macro/micro
metrics and original-coordinate PARSeq boxes were recomputed from the saved maps.
All 125 samples and their targets match across variants. No new training or model
inference was performed for this audit.

The lightweight machine-readable audit is
`provenance/word_multitask_v1_completion_audit.json`. Full per-sample evidence
remains in each branch's `validation.json` and `validation_exports/`.
Comparison SHA256:
`9e933af73a56363038cf511e7b3dff12767cb5d8e1da1e529e5777d62c6e290b`.

The original root `failure.json` records the earlier JSON-contract failure before
training began; its timestamp predates all successful training manifests. It is
preserved as historical evidence and does not describe the completed run.
The first completion-audit attempt looked for the dataset manifest relative to
the repository rather than the recorded target root and failed explicitly; the
correctly resolved audit then passed. No experiment artifact was altered.

The two focused test modules passed **35 tests**:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venvs/context-str-main-py311/bin/python \
  -m pytest tests/segmentation/test_word_multitask.py \
  tests/segmentation/test_word_multitask_provenance.py -q
```

The full-suite failures recorded during implementation in `docs/WORD_MULTITASK.md`
remain part of the verification record; the full suite was not rerun for this
artifact audit. The recommended existing checkpoint for further qualified
word-geometry work is `text_kernel/best.pt`, selected at epoch 61, SHA256
`e65691dda944478e9d8b4f022d45d0d3b21edc7c995274554b4c5ca31336766f`.

## Offline visual archive

The requested ZIP is
`artifacts/word-multitask-visualizations-v1.zip`. Extract it and open
`word-multitask-visualizations-v1/index.html` in a browser. It includes all 125
validation samples for all three variants, common three-map GT visualizations,
original-coordinate proposal JSONs, architecture SVGs and the saved reports.
The gallery works offline, with no installation or model execution.

Run the following on the computer receiving the archive, using the server's SSH
host name (`paper` in the supplied terminal prompt):

```bash
scp bmartelli@paper:/extra1/bmartelli/projects/icdar2025-context-driven-str/artifacts/word-multitask-visualizations-v1.zip .
unzip word-multitask-visualizations-v1.zip
```

Model weights and raw datasets remain outside the archive. Exact source-image
and export hashes are retained in the proposal files and reports. The archive
manifest records SHA256s for its payload files.
