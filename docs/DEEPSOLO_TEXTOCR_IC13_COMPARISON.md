# DeepSolo TextOCR comparison and IC13 adaptation

All local experiments here are **approximate reproduction**. No new checkpoint
has completed inference. The original qualified runtime and all prior results
remain intact.

## Evidence and current status (2026-10-16)

The [pinned official model zoo](https://github.com/ViTAE-Transformer/DeepSolo/blob/dbadae995035246bad3376c7a44c015c69e9b313/DeepSolo/README.md)
reports Total-Text E2E None **F1** of 79.7 for R50, 82.5 for R50+TextOCR,
and 83.6 for ViTAEv2-S+TextOCR. These are official polygon/word-spotting scores,
not the local casefold word-multiset metric.

The new runner audits the same TEST300/2204 care words and optionally TRAIN125/983
care words, in separate tables. It retains empty predictions, original ignore
rules, native preprocessing, inclusive0.4 threshold, no lexicon, raw responses,
strict checkpoint compatibility, per-image TP/FP/FN, F1, TN0 accuracy, input
shape, allocated/reserved CUDA peaks and error receipts. Each model uses its
native upstream configuration: R50 shortest1000/max1892, ViTAE shortest1024/max1892
plus bottom/right divisor32 padding. This is a comparison of released variants,
not an experiment isolating only the backbone.

CPU model construction passed in the existing legacy environment for both
architectures:42,588,150 parameters for R50 and33,748,854 for ViTAE. This is an
architecture/import check only: **no checkpoint compatibility, GPU-memory or
accuracy qualification for the new variants**. The GPU was subsequently occupied
by an independent segmentation training task; it was left running.

Attempts to download both official Total-Text weights returned HTTP401 via the
public content endpoint; direct downloads redirected to Microsoft login. The
official model zoo lists no IC13-finetuned checkpoint and no GitHub release assets.
A search found an IC15-only community mirror, which was not substituted for any
requested variant. Upstream issue81 concerns unavailable DeepSolo++ links; it is
corroborating context only, not proof about our two DeepSolo links.

External evidence, CPU build reports and test logs are under:
`/extra1/bmartelli/context-str-artifacts/deepsolo-textocr-ic13-preflight-20261016/`.

## Run Total-Text after obtaining the official files

Download the exact official variants and save them outside Git:

- [R50+TextOCR Total-Text](https://1drv.ms/u/s!AimBgYV7JjTlgcd138p8HaXViFk-tw?e=r15pMR)
  as `/extra1/bmartelli/context-str-artifacts/deepsolo-host/checkpoints/r50_textocr.pth`.
- [ViTAEv2-S+TextOCR Total-Text](https://1drv.ms/u/s!AimBgYV7JjTlgcd6XGlbZ-I7WvGslQ?e=rrkXLx)
  as `/extra1/bmartelli/context-str-artifacts/deepsolo-host/checkpoints/vitae_textocr.pth`.

The following pins the locally obtained bytes; the authors publish no reference
SHA256 for these alternatives. A locally computed hash does not independently
establish which training recipe produced a checkpoint. Preserve download origin.
Run once the separate segmentation training has released the GPU:

```bash
cd /extra1/bmartelli/projects/icdar2025-context-driven-str
set -euo pipefail
for TASK_MODEL in r50_textocr vitae_textocr; do
  TASK_CHECKPOINT="/extra1/bmartelli/context-str-artifacts/deepsolo-host/checkpoints/${TASK_MODEL}.pth"
  TASK_DIGEST=$(sha256sum -- "$TASK_CHECKPOINT")
  .venvs/context-str-main-py311/bin/python -u -m scripts.run_deepsolo_textocr_comparison \
    --mode approximate --model "$TASK_MODEL" \
    --checkpoint "$TASK_CHECKPOINT" --checkpoint-sha256 "${TASK_DIGEST%% *}" \
    --output-root /extra1/bmartelli/context-str-artifacts/deepsolo-textocr-comparison-v1
done
```

Each variant runs both splits,425 images total, with no router refitting.
Add `--splits test` for only300 TEST images, `--limit 1` for a separately labeled
partial smoke, or `--check` for input qualification without model inference.
Interrupted runs resume only matching sealed shards. Failures abort with a
failure receipt and are never removed from the final denominator.

The existing weights contain around180MB (R50) and144MB (ViTAE) of state tensors
according to CPU architecture inspection; actual serialization/optimizer states
can change download size. Budget **3GB extra disk** for both weights and 425 raw
responses per model, conservatively estimated from existing raw-response sizes;
this is an estimate, not a measured upper bound. No training checkpoints or new
dependencies are created by this inference command. New GPU-memory requirements
remain unmeasured.

## ICDAR2013: recovered author configuration

The [DeepSolo supplement, Appendix A](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Ye_DeepSolo_Let_Transformer_CVPR_2023_supplemental.pdf)
specifies IC13 fine-tuning1000 iterations and test resize1000/max1824.
In [upstream issue15](https://github.com/ViTAE-Transformer/DeepSolo/issues/15),
author `ymy-k` additionally supplies threshold**0.25**, batch8, LR1e-5,
backbone LR1e-6, warmup0, no LR decay, rotation disabled and the37-character
training annotations. The author explicitly says IC13 TEST files were not
uploaded and refers to SPTS for evaluation resources.

These settings are recorded in
`configs/approximate/deepsolo_ic13_author_recipe_v1.json`. It is explicitly a
blocked recipe, **not an executable training configuration**. Missing artifacts:
the shared R50 pretrained checkpoint (before Total-Text fine-tuning), official
IC13 `train_37voc.json`, pinned Generic vocabulary/pair list and IC13 correction/
evaluation protocol. Existing IC13 images and segmentation annotations do not
resolve the missing upstream-format training targets. No IC13 fine-tuning or
Generic evaluation has been run, and no None/Generic improvement is claimed.

None and Generic must use the same adapted checkpoint and raw detections, with
lexicon postprocessing only in Generic. The official90.1 score is Generic E2E F1;
the old local88.84 image-membership score is a different metric. The historical
Total-Text-checkpoint IC13 control is81.19 local word F1 (233 images,1095 care
words); its own metric must be retained for paired local improvement claims.

## Validation

The baseline CPU replay on all300 TEST images exactly reproduces TP1595,FP413,
FN609,2008 care predictions,32 ignored predictions, local word F1=75.7359924%
and TN0 accuracy=60.9476500%. This replay does not constitute new GPU inference.

The final focused run yielded **32 passes and one pre-existing failure** in
`test_host_source_revision_verification_rejects_wrong_revision`: the expected
error text `source revision mismatch` differs from the current
`Declared DeepSolo source revision does not match Git HEAD`. The rejection itself
works. Neither the legacy worker nor that test was modified to hide the failure.
New-runner tests also cover candidate/checkpoint/config identity, strict loading,
threshold, device, lexicon, resume integrity, EXIF resize and ViTAE padding.
All **10 new-runner tests passed**. `git diff --check` passed.

Final baseline replay:
`/extra1/bmartelli/context-str-artifacts/deepsolo-textocr-comparison-v1/baseline-audit/decb78f0ac41/results.json`.
