# Total-Text word-level U-Net ablation (practical)

This is a `PRACTICAL_APPROXIMATION`, not a Table 3/5 paper reproduction.
The paper/author-protocol U-Net, existing letterbox checkpoint and threshold-0.50
TEST report are unchanged.

Run on PAPER:

```bash
cd /extra1/bmartelli/projects/icdar2025-context-driven-str && bash scripts/run_totaltext_word_f1_separator_suite_host.sh /extra1/bmartelli
```

If only the threshold job fails while the TRAIN jobs continue, restart that
job alone (do not relaunch the two training branches):

```bash
cd /extra1/bmartelli/projects/icdar2025-context-driven-str && bash scripts/run_totaltext_letterbox_threshold065_test_host.sh /extra1/bmartelli
```

The historical epoch-71 checkpoint pins the original `models/unet_adv.py`
SHA256. The loader accepts the later additive separator-head class only after
reconstructing and checking the exact original source bytes; unrelated source
changes remain fatal. A failed threshold pre-inference run may refresh only
`protocol.json`'s evaluator-source hash when no mask, crop, OCR or result
exists. Completed outputs can never be relabeled this way.

The launcher verifies frozen artifact hashes and the 1130/125 TRAIN split,
then starts three independent jobs concurrently when GPU 0 has at least
16 GiB free (otherwise serially with at least 9 GiB free):

1. Existing epoch-71 letterbox model, fixed threshold 0.65 on TEST-300,
   selected from the completed TRAIN-only sweep; PARSeq crop OCR is audited.
2. Matched 100-epoch letterbox control, selected by validation word-proposal F1.
3. The same 100-epoch U-Net with a 65-parameter separator output channel,
   selected by the same word-proposal F1 rule.

The two TRAIN jobs share a frozen split and augmentation, but have separate
checkpoints, optimizer states and output directories. Every epoch reports
proposal F1, recall, fused-word count and pixel IoU. Selection uses F1 at
IoU 0.50, then fewer GT words touched by fused components, then recall,
pixel IoU and earlier epoch. The separator branch uses the existing annotated
separator target and suppresses separator-positive pixels before connected
components. No TEST image or PARSeq output is used to train or choose a
checkpoint. Final TRAIN reports contain 125 per-image geometry rows.

Outputs (outside the repository):

- `context-str-runs/totaltext-letterbox-word-f1-separator-v1/{word_f1_control,word_f1_separator_head}/`
- `context-str-runs/totaltext-letterbox-threshold065-parseq-test-v1/`

The TEST threshold check is exploratory: previous ablations have examined
the same 125 TRAIN validation images. Do not tune another threshold or the
separator model after reading this TEST result. The new models are initially
judged on TRAIN-validation geometry; their PARSeq OCR and official TEST
performance are not claimed until separately qualified.
