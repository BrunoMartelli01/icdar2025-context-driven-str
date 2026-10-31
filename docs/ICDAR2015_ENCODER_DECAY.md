# ICDAR2015: matched encoder regularization experiment

This is **approximate reproduction**, under
`configs/approximate/icdar2015_encoder_decay_v1.json`. The local paper
sections3.2/Figure3 and4.1 and the existing optimizer/training implementation
were inspected. The paper specifies BCE/Adam and weight decay0.01; applying
that penalty to an explicitly unfrozen encoder is an additional assumption.
STRICT configuration is untouched.

## Evidence motivating the experiment

Both actual Total-Text weight transfers completed100 epochs and their20-case
VAL100 operating grids. Neither beats the previous61.7608% validation emitted
F1: the epoch263 source retains fine-tuning epoch10,25.8278% raw/34.0491%
emitted; the partial-encoder source retains epoch0,33.8284% raw/50.7079%
emitted, or57.5130% after its separately calibrated operating grid. No new TEST
evaluation is justified. See
`provenance/ICDAR2015_TOTALTEXT_WEIGHTS_COMPLETION_V1.json`.

A saved-weight/optimizer diagnostic finds that, after these fine-tunings,
layer4 convolutional Frobenius norms retain24.89% and25.75% of their source
values; layer3 retains37.28% and38.22%. All encoder BN running buffers are
unchanged, as intended. Encoder Adam first-moment cosine with the current
weights is0.99672/0.99799, and the coupled L2 term's norm divided by the moment's
norm is0.99648/0.99778. These observations are consistent with regularization
dominating the encoder update and changing feature scales under frozen BN.
They do not alone prove the causal effect of removing decay: Adam moments also
contain loss gradients and history. Decoder/bottleneck norms also shrink; this
experiment isolates only encoder regularization. Preserve the complete
diagnostic and source checkpoints.

## Predeclared matched comparison

Both arms start from qualified ICDAR adapt1280 epoch50, effective epoch106,
checkpoint `c4676914dbcbf559a26fb98e51f5ae137b2a3ac66930d0191fca5b214fce5eb6`.
Keep the identical single-output architecture and all452 state tensors, masks,
original FIT900/VAL100 assignment and paired augmentation. Both encoder arms
enable only layer3/layer4 convolutions, with all encoder BN parameters/statistics
frozen. Both use BCE, separator/ignored-negative weight3, clipping1 and50
additional epochs. Decoder and deep-encoder LR each start at1e-5, with cosine
decay to1e-6. Source epoch0 remains eligible.

| Arm | Decoder Adam decay | Encoder Adam decay | State initialization |
| --- | ---: | ---: | --- |
| decay_control |0.01 |0.01 |restore decoder moments/steps; fresh encoder moments |
| no_encoder_decay |0.01 |0 |identical initialization |

Reset encoder moments in BOTH arms, because inherited moments contain the old
L2 term. Preserve every decoder moment and step. This reset, LR restart,
unfreezing,50-epoch schedule and per-group decay are ASSUMPTION choices. The
two arms differ only in encoder weight decay. Negative-only padding behavior,
loss denominator and sample order remain unchanged. The already-qualified
training-pixel wrapper only omits unused owner replay; validation still measures
owner topology. Do not use the OHEM loss in this experiment.

This is FP32 RGB736x1280/native368x640, physical4/accumulation2 on the actual
RTX3090, a HARDWARE_DEVIATION. Qualify both full-size backward smokes and measure
allocated/reserved peaks and OOM before full training. No automatic precision,
shape or batch substitution is allowed. Full initial VAL100 raw metrics must
exactly reproduce the pinned source before the first fitting epoch.

Use frozen CPU PARSeq at epoch0, every5 epochs, new geometry bests and final
epoch. Select by raw spatial symbol-free word F1, then emitted F1, geometry and
earliest epoch. All430 validation care words and all emitted predictions remain.
Run the same20-case operating grid after complete training. Every selection
uses VAL100 only; preserve the original40.7556% TEST baseline and the subsequent
54.7782% emitted/43.5836% raw TEST result. Before final TEST evaluation, freeze
the selected weights/policy and qualify the OCR device on all selected validation
crops. See ambiguity `ICDAR2015-ENCODER-DECAY-001`.

## Completed result

Both arms completed all50 epochs and retained epoch0. Raw word F1 remains
49.6454%; emitted61.3333%, or61.7608% after the full20-case operating grid.
All900 fitting samples appeared in each epoch, with identical sample-order
hashes across the two arms. Selected tensors exactly reproduce c467, so no
duplicate TEST evaluation is warranted. Removing encoder decay preserves final
layer3/layer4 norms (ratios1.00018/1.00021); the matched decay control reduces
them to0.58255/0.46994. This confirms the norm effect but does not establish a
validation benefit. See `provenance/ICDAR2015_ENCODER_DECAY_COMPLETION_V1.json`.

The same frozen c467 model was then evaluated on allFIT900 without augmentation:
3492/4018 geometric matches,3859 proposals, localization F1 88.6632%, foreground
IoU87.9567%. Existing VAL100 gives282/430 matches,557 proposals, localization
F1 57.1429%, foreground IoU60.0849%. All predictions, including ignored regions,
count in both populations. This is a pronounced fitting/generalization gap;
neither value is correct-word F1. The FIT diagnostic saved all900 probability
maps and per-image outputs. An initial external-runner import-path error was
recorded and corrected before inference. See
`provenance/ICDAR2015_FIT_GAP_V1.json` and the subsequent
[data-diversity experiment](ICDAR2015_SYNTHTEXT_PRETRAINING.md).
