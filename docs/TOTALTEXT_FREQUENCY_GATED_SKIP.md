# Total-Text feature-wavelet gated-skip ablation

This is a `PRACTICAL_APPROXIMATION`, not the paper's AG-UNet. It leaves the
strict model and all completed letterbox/word-F1 runs untouched. The earlier
RGB-Haar additive /2 skip performed poorly, so this one tests a distinct
feature-level mechanism: fixed Haar detail bands from the frozen ResNet-50
shallow feature are projected and gated by decoder context at /4 resolution.
Only the gated residual is upsampled to /2, where the original attention gate
and skip remain in place. The residual gain begins at zero. The architecture
adds 24,641 trainable parameters to the 58,155,081-parameter baseline.

The frozen protocol is in
`configs/approximate/totaltext_frequency_gated_skip_100ep_v1.json`.
It uses the qualified Total-Text v4 word masks, letterbox, 1130/125 TRAIN
split, seed 202503, batch 8, separation-weighted BCE and exactly **100
epochs**. The best checkpoint is selected only by held-out TRAIN macro
foreground IoU, matching the original letterbox training protocol. After
training, the host launcher evaluates PARSeq on the same 125 TRAIN-validation
images with the already TRAIN-selected probability threshold 0.65, native
contours, no area/top-ten filtering and pad 0. The old letterbox threshold-0.65
reference is 552/983 spatial IoU>=0.5 and 437/983 PARSeq casefold exact.
Neither model nor threshold is selected from TEST.

Run on PAPER in the foreground (preflight, epochs, native crops and PARSeq
progress are printed; interrupted checkpoints/crops are validated and reused):

```bash
cd /extra1/bmartelli/projects/icdar2025-context-driven-str && bash scripts/run_totaltext_frequency_gated_skip_100ep_host.sh
```

The training and OCR reports are stored outside the repository under
`/extra1/bmartelli/context-str-runs/totaltext-frequency-gated-skip-100ep-v1/`.
The paper-faithful localization/recognition pipeline is not modified.
