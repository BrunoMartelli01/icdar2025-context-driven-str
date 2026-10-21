# ViTAEv2-S+TextOCR GPU evaluation — 2026-10-19

**Approximate reproduction.** User request: start the GPU tests with the uploaded
checkpoints. This run preserves the original R50 baseline and all router gates.

The complete ViTAEv2-S Total-Text TextOCR checkpoint supplied by the user has
135,515,957 bytes and SHA256
`54d3a454ecc704ef3168ebf9f23fc8390a3b2637d0ec83d5650e8df45e2aca4a`.
The earlier attachment transfer receipt verified identical source/destination
hashes and all ZIP-member CRCs. The R50+TextOCR attachment is still truncated
(12,136,448 bytes) and is excluded. The complete shared R50 pretraining weights
are available but do not constitute an IC13-adapted model.

## Completed engineering qualification

The existing isolated legacy host environment ran the repository DeepSolo adapter
on the original720x1280 qualification image. CUDA inference succeeded on the
current **RTX3090**, FP32 batch1, official shortest1024/max1892 preprocessing and
divisor32 padding (model input CHW3x1024x1824), threshold0.4, no lexicon.

- Checkpoint compatibility: **zero missing keys, zero unexpected keys, zero
  incorrectly shaped tensors**;33,748,854 parameters; eval mode.
- Recognized strings: `test`, `deepsolo`, `2026`; reading-order output
  `deepsolo test 2026`.
- Peak allocated CUDA memory: **1,930,212,352 bytes** (about1.80GiB).
- Peak reserved CUDA memory: **3,779,067,904 bytes** (about3.52GiB).
- Model-load time0.659s; inference0.424s. These are single engineering-smoke
  measurements including first-inference effects, not benchmark timing.
- No OOM was observed. Worker failures retain structured tracebacks and abort;
  there is no precision/model/CPU fallback.

The focused runner/native-preprocessing tests passed: **16 tests**. Three
additional subprocess/worker failure-propagation tests also passed (19 total).

## Full evaluation started

Configuration: `configs/approximate/deepsolo_textocr_gpu_20261019.json`.
The historical configuration is preserved; this new configuration records that
the ViTAE checkpoint is now available and that the R50 TextOCR file is invalid.
The full preflight verified all **125 TRAIN** and **300 TEST** inputs before GPU
evaluation. TRAIN and TEST are aggregated separately, with original care
denominators983 and2204. TRAIN is diagnostic because upstream fine-tuning used
Total-Text TRAIN. Matching, ignore handling, empty outputs and threshold treatment
are identical to the frozen local baseline; official polygon F1 stays separate.

Execution is detached from the terminal and uses a new isolated output directory:

```text
/extra1/bmartelli/context-str-artifacts/deepsolo-textocr-gpu-20261019/
  launch-vitae.json
  vitae-full.log
  qualification/runtime.json
  qualification/vitae-smoke.json
  full/<plan_hash>/resolved_configuration.json
  full/<plan_hash>/runtime.json
  full/<plan_hash>/train/*.json
  full/<plan_hash>/test/*.json
```

`launch-vitae.json` records the actual command, PID, start time and configuration
hash. The runner writes `results.json` only after all requested samples succeed;
failures create `failure.json` and are never omitted from the denominator. Its
per-image receipts preserve raw detections, input shapes, memory, timings and
word/geometric metrics. No accuracy claim is made before completion. The smoke
peak is not a bound on the full-dataset memory peak.

The GPU was idle during the engineering smoke. After full evaluation started,
an independent `run_word_evidence_suite` process also used CUDA. It was preserved.
Full-run allocator peaks refer to our worker; full-run timing is consequently
not an isolated-GPU benchmark. Resource observations are recorded in
`gpu-co-residency.json` alongside the launch receipt.

## Scientific classifications

- `PAPER_SPECIFIED`: Algorithm2 uses full-image DeepSolo for uncertainty.
- `REPOSITORY_SPECIFIED`: pinned upstream native ViTAE configuration, checkpoint
  architecture and preprocessing; existing qualified legacy worker and local
  word/geometry evaluators. Original R50 outputs remain the paired control.
- `ASSUMPTION`: identify the user-provided complete bytes with the requested
  upstream zoo variant using its filename/download origin, pinned hash, archive
  checks and exact architectural compatibility. Authors publish no reference
  checksum. Preserve the frozen425-image population without router refitting.
- `HARDWARE_DEVIATION`: existing isolated host subprocess on RTX3090, FP32 batch1,
  one model at a time; preserve qualified cuDNN enabled/benchmarkFalse/
  deterministicFalse policy. Record per-image allocator peaks and errors.
- `UNRESOLVED`: context-driven-paper checkpoint selection and exact accuracy
  protocol; R50 TextOCR complete bytes; IC13 fine-tuning targets and Generic
  resources. This run cannot establish strict reproduction or IC13 performance.

See ambiguity `DEEPSOLO-VITAE-GPU-20261019` and
`docs/DEEPSOLO_TEXTOCR_IC13_COMPARISON.md`.
