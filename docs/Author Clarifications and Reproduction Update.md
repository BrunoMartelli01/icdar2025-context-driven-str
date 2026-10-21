# Author Clarifications and Reproduction Update

## 1. Background

During reproduction of the segmentation results, several inconsistencies were found between the paper, the released repository, and the executable behavior of `models/unet_adv.py`.

The most important issue was the attention gate: the released implementation produced a multi-channel attention tensor whose channel count did not match the encoder skip tensor, making the network fail at runtime. We therefore investigated the intended architecture experimentally while keeping all reproduction choices explicit.

Subsequent clarification from the paper author resolves several of these ambiguities.

## 2. Work performed before author clarification

The following reproduction experiments were carried out:

- Identified the channel mismatch in the released attention gate.
- Verified that the upstream `AdvancedUNet` could not execute as written.
- Tested a scalar spatial attention gate in which a single-channel attention map is broadcast over the encoder skip channels.
- Explored a projected channel-wise attention variant as an ablation.
- Tested bottleneck compression variants, including `2048 -> 512`.
- Froze the ImageNet-pretrained ResNet-50 encoder during training.
- Implemented masked BCE so ICDAR13 Don't Care regions contribute no gradient.
- Compared 256×256 and 512×512 training.
- Performed validation threshold sweeps without using the test set for threshold selection.
- Observed that increasing training resolution substantially improved validation macro IoU/F1, although a large validation-to-test gap remained.

These experiments were diagnostic and should not be confused with the now-clarified author protocol.

## 3. Architecture clarification from the author

### Decoder and output resolution

The network used for the reported results terminates at a `128×128` decoder feature map with 64 channels.

The segmentation loss is computed at `128×128`.

The additional repository upsampling stage producing `256×256` logits was added later for block-level localization and was not part of the network used for the reported segmentation results.

For reproduction of the reported architecture, that additional stage should therefore be removed.

### Bottleneck

The bottleneck is intended to reduce the channel dimensionality of the deepest ResNet-50 feature map using convolution followed by BatchNorm.

The repository's `2048 -> 2048` bottleneck does not reflect this intent.

The author recalls using approximately:

`2048 -> 1024`

but cannot confirm the exact compressed width.

Therefore `2048 -> 1024` should be recorded as an author-recalled but unresolved architectural parameter rather than treated as fully verified.

### Attention gate

The intended gate is additive spatial attention:

1. project the encoder feature and decoder gating signal to an intermediate width;
2. sum them;
3. apply ReLU;
4. apply a `1×1` convolution producing exactly one channel;
5. apply sigmoid;
6. broadcast the resulting `H×W` attention map across all encoder channels;
7. multiply it with the encoder feature.

The attention map is therefore shared across channels.

The encoder skip tensor is not projected to the decoder output width before multiplication.

The multi-channel attention tensor in the released `models/unet_adv.py` is a defect.

## 4. Training clarification from the author

### Dataset

All 229 ICDAR 2013 training images were used.

No validation subset was held out for the reported training protocol.

### Encoder

The ImageNet-pretrained ResNet-50 encoder remained frozen for the entire training run.

Only the bottleneck, attention gates, and decoder were trained.

### Training duration

The model was trained for 100 epochs with batch size 8.

### Checkpoint selection

Because all 229 training images were used, checkpoint selection was performed using foreground IoU on the training set.

For scientific reporting, this protocol must be distinguished from our earlier reproduction experiments that used a held-out validation subset.

### Don't Care regions

ICDAR Don't Care regions were excluded from BCE using a per-pixel weight/validity mask.

They contribute zero gradient and are neither foreground nor background supervision.

### Target resolution

Ground-truth masks are reduced to `128×128` for the segmentation loss.

The author's procedure is:

1. bilinear interpolation to `128×128`;
2. re-binarization using threshold `0.5`.

Nearest-neighbour target downsampling was not used.

### Augmentation

Horizontal flip probability:

`0.5`

Random rotation was also used.

The exact range is no longer known. The author suggested approximately ±15° as a reasonable reproduction choice, but this must remain explicitly marked as an assumption rather than an established paper parameter.

## 5. Required implementation update

The repository reproduction path should now be updated to:

- restore scalar single-channel additive attention;
- preserve encoder skip channel dimensionality through attention;
- remove the extra segmentation upsampling stage used only for later localization;
- produce `128×128` segmentation logits;
- compute BCE against bilinearly downsampled and re-binarized `128×128` masks;
- exclude Don't Care pixels through a per-pixel validity mask;
- freeze ResNet-50 parameters and BatchNorm state;
- use all 229 ICDAR13 training images;
- train for 100 epochs;
- select the reported checkpoint by training foreground IoU;
- use horizontal flip probability 0.5;
- record rotation ±15° as an assumption;
- test `2048 -> 1024` bottleneck compression while clearly recording that the exact width remains unconfirmed.

Previous 256×256 and 512×512 experiments remain useful as ablations but must not be presented as the clarified author protocol.