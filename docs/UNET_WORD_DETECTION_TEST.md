# Text+Kernel and Hybrid384 word detection on TEST

Run both frozen recipes on all **300 official Total-Text TEST images**:

```bash
cd /extra1/bmartelli/projects/icdar2025-context-driven-str
bash scripts/test_unet_word_detection.sh
```

The default device is `cuda:0`. For explicit CPU execution:

```bash
UNET_TEST_DEVICE=cpu bash scripts/test_unet_word_detection.sh
```

Check the dataset, source code and checkpoint hashes without inference:

```bash
bash scripts/test_unet_word_detection.sh --preflight-only
```

Results are printed as percentages and saved in a new timestamped directory
under `artifacts/diagnostics/unet-word-detection-test-*/results.json`.
The terminal prints the exact path. Each model reports TP, FP, FN, precision,
recall, word F1 and `(TP+TN)/(TP+TN+FP+FN)` with **TN=0 as an assumption**.
Metrics use original RGB bounding boxes, maximum one-to-one IoU>=0.50,
global micro aggregation and all2204 care words. There is no OCR requirement
for a detection match. Saved predictions, instance maps, per-image evaluations,
probability maps, resolved configuration and memory receipts support replay.

This is **approximate reproduction**, with a local bbox detection diagnostic.
It is not official polygon DetEval or paper recognition accuracy. Thresholds
and checkpoints remain those discussed on the TRAIN holdout: Text+Kernel
control epoch2/text.65; Hybrid384 Tversky768 epoch10/text.90 plus Text+Kernel
epoch61 kernels; kernel.50 and seeded-only filtering for both. The three source
networks load sequentially in FP32/batch1. GPU memory and OOM reporting are
implemented, but only CPU smoke execution was verified in this session.

The launcher derives storage paths from the repository location. Optional
environment overrides are `UNET_TEST_PYTHON`, `UNET_TEST_RUNS`, `UNET_TEST_DATA`,
`UNET_TEST_DEVICE`, and `UNET_TEST_OUTPUT`. Output directories must be new;
existing results are never overwritten. A failure exits nonzero and prevents
complete metrics. A rerun uses a new output directory and restarts inference.
`--limit N` is only an explicitly labeled partial-dataset smoke.

Verification:14 new/relevant tests, asset preflight for all300 TEST inputs,
and real CPU inference of all three checkpoints/both arms on one TEST image
passed. The two corrected smoke failures and scientific assumptions are
recorded in [UWD-TEST-001](PAPER_AMBIGUITIES.md#uwd-test-001).
