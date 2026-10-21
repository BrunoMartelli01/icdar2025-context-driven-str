# Small RGB/mask/probability correction battery

**Approximate reproduction / experimental extension**, decision MASK-REFINER-001.
User requested online research, multiple architectures and training variants on
2026-10-21. The authoritative local paper sections3.2–3.3/Algorithm1 and the
existing implementation were inspected before changes. Neither extra refinement
training nor Text/Kernel maps are specified by that paper.

## Literature and local novelty

- [CascadePSP, CVPR2020](https://arxiv.org/abs/2005.02551),
  [official implementation](https://github.com/hkchengrex/CascadePSP): image/mask
  conditioning and global/local refinement motivate broader spatial context.
- [PointRend, CVPR2020](https://arxiv.org/abs/1912.08193): adaptive point selection
  motivates concentrating supervision on errors/boundaries. Our dense weighted
  loss does not implement its subdivision algorithm or point rendering.
- [SegRefiner, NeurIPS2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/fc0cc55dca3d791c4a0bb2d8ddeefe4f-Abstract-Conference.html),
  [official implementation](https://github.com/MengyuWang826/SegRefiner): repeated
  conditional correction motivates a shared three-pass experiment. Our recurrent
  U-Net is not discrete diffusion and does not reproduce SegRefiner.

Sources inspected2026-10-21. These are research inspirations, not claims of
literature novelty or replicas of published architectures/results. No external
model weights, source code or additional packages are used by these correctors.
Primary-source CVF HTML requests returned403; arXiv, official repositories and
NeurIPS records supplied the evidence.

The local audit, [MASK_PROBABILITY_REFINER_NOVELTY.json](MASK_PROBABILITY_REFINER_NOVELTY.json),
preserves hashed near-prior evidence. Existing crop U-Nets segment RGB patches
with a copied pretrained decoder; previous tiny prompt models predict a new
single-word mask with a7x7 receptive field. V2 DB/threshold heads already make
residual/threshold adjustments using backbone features. Thus neither residual
learning, crop refinement, kernels nor longer training is claimed as locally
new by itself. The new controlled combination is an independent small full-image
corrector conditioned on RGB, both coarse probabilities, both thresholded masks
and canvas, with exact initial map identity, broader context and matched training
ablations. Previous Gap decoders learn an extra separator; their full-TRAIN1255
refit has a different data protocol and is not the control of this battery.

## Models and supervision

Input is original RGB centered on its square letterbox and resized directly to
native256 with PIL bilinear, scaled[-1,1]; source Text/Kernel probabilities,
thresholded masks(.65/.5) and RGB canvas make eight channels. GT never enters
the inference interface. Targets are the saved v4 filled word mask and its
existing independently shrunk .5distance kernels. Binary glyph stroke masks
are not character instances and are not refiner inputs.

| Arm | Architecture | Trainable parameters | Training difference |
| --- | --- | ---: | --- |
| shallow_balanced | Three3x3 layers,24channels | 12,218 | Balanced BCE/Dice |
| dilated_balanced | Five3x3 layers,24channels, dilations1/2/4/8/16 | 22,634 | Same |
| unet_balanced | Channels16/32/64/128,3down/upsamples, global pooled context, GroupNorm | 505,954 | Same |
| unet_error_focus | Same U-Net and initialization | 505,954 | Additional4x weight on source errors/3x3GT boundaries |
| unet_corruption | Same U-Net and initialization | 505,954 | On50%fit images,3/5pixel erosion/dilation of predicted probability maps |
| unet_iterative | Same shared U-Net,3passes | 505,954 | Equal loss at all3passes |

Each zero-initialized two-channel output predicts bounded additive logit
corrections,12*tanh(raw/12). For numerical stability, logit input is clamped
to[1e-4,1-1e-4]. Inference adds the sigmoid *difference* to the original
probability, then clips[0,1]; a zero correction is bitwise identity, including
0,1 and threshold values. The stable-logit BCE proxy and emitted probabilities
can differ by up to epsilon per pass at saturated source values. Dice and
preservation use emitted probabilities. This numerical proxy is explicit.
The shared iterative network recomputes probability/mask inputs after each pass.

Train all models from scratch; original segmentation and qualified PARSeq are
frozen. Positive and negative BCE are separately normalized per image/channel,
then averaged over present classes. Add.5Dice with smoothing1 and.05valid mean
absolute change from input probabilities. Ignore pixels contribute no gradient;
all-invalid batches skip updates and are counted. Clip gradient norm at5.
GT error/boundary weights appear only in training loss. Corruption uses only
predicted maps; targets remain unchanged. GroupNorm has no running statistics.

## Budget, selection and evaluation

Same seeded1030fit/100calibration/125evaluation IDs as the previous local-word
battery. The source segmenter previously saw the100calibration images; these
are held out from the corrector only. Its frozen canonical fit predictions are
not out-of-fold. The125evaluation images are a repeatedly reused officialTRAIN
holdout, not untouched TEST. Single-seed results do not establish generalization.

**80complete whole-image epochs**, every fit ID once each epoch, every visible
valid word supervised. Batch8 gives129updates/epoch,10,320per arm when no batch
is wholly invalid. This is82,400whole-image exposures per arm, rather than the
previous one sampled word window/image/epoch. AdamW3e-4, decay1e-4, cosine to1e-5,
seed20261020; exact paired horizontal flip and0/90/180/270rotation without
interpolation. The rotation/flip distribution and budget are assumptions, not
paper settings. U-Net loss/corruption arms share initialization, ordering and
paired augmentation. Different architectures have unequal parameter counts.

Calibrate only at epochs0/5/20/40/80. Epoch0 is qualified by exact source-map
identity across the declared population. Choose most correctly localized/read
words, then OCR Hmean, then geometric Hmean; off wins full ties and earlier
checkpoints win subsequent ties. Run all80epochs regardless of the selected
epoch. Finish selection for all six arms before scoring any125evaluation arm.
No held-out winner is automatically promoted to deployment.

Unchanged source watershed,Text>=.65,Kernel>=.5; no area filter,padding or cap.
Global correction can add/delete/split/join instances, unlike earlier fixed
one-output-per-seed heads. Final instances are frozen before scoring annotations;
exclude only wholly invalid supports, preserve mixed supports whole. Original
exclusive RGB bboxIoU>=.5 maximum one-to-one matching; exact Unicode casefold
with punctuation retained for OCR; empty/duplicate care predictions remain.
Every evaluation arm retains all125images and983care words. Pixel micro TP/FP/FN,
IoU/F1 are over valid native256 pixels at.65/.5, empty prediction+GT gives1.
These are not official polygon metrics, paper Table3 or Table5 results.

Baseline must replay calibration622localized/525OCR/720proposals and evaluation
647localized/520OCR/1037proposals exactly. Store complete configurations, split
IDs, source/RGB/annotation/checkpoint hashes, per-epoch seen IDs, weights and
optimizer/scheduler state, native probabilities/labels, per-image proposals,
originalRGB crop hashes, genuine PARSeq outputs/confidences and paired gains/losses.
Off evaluation rows replay source predictions; learned rejected heads have only
calibration results, not a held-out learned-head score.

## Qualification and execution

Configuration: `configs/approximate/mask_probability_refiner_v1.json`.
Runner: `python -m scripts.run_mask_probability_refiner`, explicit `--mode`,
`--config`, `--evidence-root`, `--word-root`, `--parseq-source`,
`--parseq-checkpoint`, `--parseq-archive`, `--ocr-cache-source`, `--output-root`,
`--device` and `--action {preflight,smoke,all}`. Scientific locations have no
implicit defaults. Outputs must be external and separate from inputs. The
qualified launch uses a source snapshot, fresh output root, process lock,
detached supervisor and independent post-run count/replay audit.

Existing main Python3.11 environment only. Actual smoke hardware RTX3090/24GB,
FP32batch8, deterministic algorithms, noAMP/TF32, four CPU threads,workers0.
Sequential arms plus resident frozen PARSeq; source maps/targets on CPU.
This differs from the paper and is HARDWARE_DEVIATION, not8GBqualification.
Smoke peak allocated2,422,149,120bytes/reserved2,623,537,152bytes, noOOM.
All six arms made two real CUDA updates on16fitimages, with nonzero body/head
changes; PARSeq state remained exact. This is execution qualification, not
performance evidence.39focused tests pass, including interruption/resume tensor
identity, ignored-pixel gradients, saturated/threshold initial identity,
paired transforms, range and namespace/denominator checks. Full launch/result
receipts live outside the repository at the locations recorded below.

## Live full run

Launched2026-10-21T09:16:57UTC on PAPER:
`/extra1/bmartelli/context-str-runs/mask-refinement-battery-20261021-v1`.
`launch_receipt.json`, `command.json`, `source_snapshot.json` pin the launch;
992snapshot files passed the same39tests. Snapshot code is isolated from later
workspace edits. SupervisorPID1681343, runnerPID1681344 at launch.
`pipeline_status.json` tracks training and the independent final audit;
`results/progress.json` tracks arm/epoch, `suite.log` records updates.
Completion writes `results/results.json`, `results/comparison.md` and
`independent_audit.json`. Those final result files do not exist until completion.

The full calibration reference replayed exactly:100images,715care words,
622localized,525correctOCR,720proposals. Initial shallow calibration checkpoints
gave513OCR at5epochs and497at20, versus525source. These are calibration results,
not983word evaluation or a conclusion about the other architectures.
Independent Hungarian matching agrees on all300image/candidate cases checked.
The shallow arm subsequently completed80epochs/10,320updates: calibrationOCR
was499at40 and505at80, below525source at every checkpoint, so selected off.
Its training loss fell.17581→.14360 without OCR improvement. This does not
establish the outcome of the remaining models; the dilated arm is running and
the four U-Net variants are queued. Full six-arm efficacy is pending.
An additional51research-navigation tests pass;90focused tests passed in total.
