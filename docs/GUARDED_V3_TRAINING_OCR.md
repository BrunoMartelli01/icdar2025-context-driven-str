# Guarded V3 training and PARSeq — approximate reproduction

The user explicitly authorized training V3 and then counting localized/correct
words with PARSeq. The experiment is isolated under:

`/extra1/bmartelli/context-str-runs/guarded-v3-training-20261017/`

## Training

- Same V2 initialization, frozen pretrained ResNet50 AG-U-Net, RGB256/native128,
  seed202503,500 epochs, FP32/batch8, masked BCE and Adam1e-4/weightdecay.01.
- Same1130/125 split of official TRAIN; no TEST enters optimization/selection.
- Targets use the qualified dynamic V3 recipe after augmentation. Unresolved
  target conditions are retained, not removed from word denominators.
- Native Word-F1 validation uses final V3 owners, including absent original IDs.
  ForegroundIoU and the previous operational bbox Word-F1 remain recorded.
  Native Word-F1 values across different GT definitions are not directly
  interchangeable; original TEST OCR provides the downstream comparison.
- Keep unique winners for native Word-F1, bbox Word-F1, foregroundIoU and last:
  at most4 complete checkpoints plus a temporary atomic write. No old run is
  pruned. Save per-sample/per-word guard evidence compressed each epoch.

The resolved configs are `configs/approximate/guarded_v3_training_v1.json` and
`configs/approximate/guarded_v3_training_ocr_v1.json`. Source assumptions and
hardware deviations are in V3-TRAIN-OCR-001, `docs/PAPER_AMBIGUITIES.md`.

## Automatic post-training evaluation

The detached pipeline waits for successful completion of all500 epochs, then:

1. Resolves the distinct validation-selected checkpoints and cryptographic hashes.
2. Evaluates all300 original TEST images /2204 care words with pinned genuine
   PARSeq, native padding0/1/2/4/6/8, area filters0 and150 and top10 proposals.
3. Records geometric localization counts and exact Unicode-casefold OCR counts,
   retaining punctuation and all GT/predictions in denominators. Primary IoU>=.5
   uses unpadded detection boxes; padded-box sensitivity stays separate.
4. Measures predicted mask foreground/component sizes and independently audits
   every image/condition with NumPy/SciPy matching.
5. Writes `ocr_summary.json` and `ocr_summary.csv` at the run root.

Crops remain in RAM; the evaluator exports only numeric proposals, OCR and GT
evidence, not crop images or duplicate model weights. TEST padding analysis is
exploratory and never changes checkpoint selection. This is not a Table5,
context-routing or DeepSolo reproduction claim; strict PARSeq-S identity and
official recognition-accuracy protocol remain unresolved.

The V2 evaluator gained an explicit V3 namespace/contract guard. Historical V2
parameters and behavior remain unchanged. A V3 run cannot be mislabeled by a
V2 evaluation plan.

## Execution and reproducibility

Training and OCR run from the executable source snapshot in `code_snapshot/`.
Its607 source/config/provenance files are hashed in `code_snapshot_manifest.json`.
Datasets, source photos and weights are referenced explicitly, not copied into
the snapshot. This protects the long job against later repository code edits.

Existing main Python:
`/extra1/bmartelli/projects/icdar2025-context-driven-str/.venvs/context-str-main-py311/bin/python`.
Actual GPU: PAPER RTX3090/24GiB. FP32, deterministic algorithms, no AMP/TF32,
explicit `CUBLAS_WORKSPACE_CONFIG=:4096:8`; sequential model loading for OCR.

The real CUDA preflight performs one batch8 optimizer step and a two-image
validation smoke in a separate process, then verifies genuine PARSeq decoding.
Smoke weights never initialize the real run: the training process reseeds and
initializes afresh. No package installation or model download is performed.

Inspect:

```bash
cat /extra1/bmartelli/context-str-runs/guarded-v3-training-20261017/pipeline_status.json
tail -n 5 /extra1/bmartelli/context-str-runs/guarded-v3-training-20261017/training.log
```

The active training directory is
`training/total-text-guarded-v3-word-selection-500ep-v1/` under the run root.
It holds `run_contract.json`, `run_manifest.json`, `epochs.jsonl`, validation
reports, compressed GT audits, retention receipts and selected checkpoints.
Full failures/OOM stop the pipeline and preserve explicit failure metadata/logs;
OCR cannot silently proceed after failed/incomplete training.

The frozen pipeline supports rerunning the training stage from its retained
last checkpoint with identical contract/RNG. Do not launch a second copy while
the original PID/lock is active. A partially completed OCR stage needs explicit
inspection before retrying into an existing output directory.

## Launch qualification

49 relevant tests pass. CUDA training preflight passes with batch8:
allocated1166638592/reserved2248146944 bytes. Separate PARSeq smoke passes:
allocated131752448/reserved153092096 bytes. No OOM observed; no artificial OOM
test was performed. Runtime peak measurements continue in the training manifest.

The first snapshot launcher invocation used a script filename and failed at
Python imports before CUDA/training. It was corrected to `python -m` with the
snapshot as working directory; no scientific code/config substitution occurred.

PID1489202 launched the durable chain; the first full epoch completed with
loss0.42071, native Word-F1.19372, bbox Word-F1.19392 and foregroundIoU.38628.
These are early validation measurements, not final OCR results. Do not infer a
V3 improvement until training and the complete PARSeq/audit stages finish.
