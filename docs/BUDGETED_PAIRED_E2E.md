# Frozen InstructBLIP paired routing: TRAIN and TEST

This opt-in **approximate reproduction / PRACTICAL_APPROXIMATION** replay keeps
tau=0.23055392728092478 frozen from the TRAIN image-budget calibration. It does
not rerun segmentation or PARSeq, retune a threshold, change faithful defaults,
or choose model outputs with GT. The TEST set has been inspected in earlier work;
this is not a claim of untouched, independent final validation.

## Metrics and output policies

- Main word accuracy: casefold, whitespace-token multiset intersection within
  each image, capped by GT multiplicity, divided by **all GT words** (TRAIN849,
  TEST1095). Missing words count as errors. No punctuation stripping or spatial
  matching is used. This is a declared diagnostic, not Table 5 accuracy.
- Extra wrong tokens and word precision are separately reported. Complete-image
  accuracy requires equality of predicted and GT word multisets: no omissions or
  extras, with reading order ignored. It is not “any correct word in the image”.
- The primary DeepSolo word-level policy uses **all recognized strings**. The
  original `highest_score` selected output is also evaluated in a separate
  control. Raw detections and their scores remain unchanged; no selection uses GT.
- Report `deepsolo_all_strings` for DeepSolo alone, `routed_all_strings` for the
  selected PARSeq text on accepted images and all DeepSolo strings on rejected
  images. `routed_highest_score` preserves the historical single-output reduction
  as an explicitly separate comparison. Do not mix the two denominators/policies.
- TRAIN DeepSolo is exported on all 229 images for the requested standalone
  control. These offline control calls are not online fallback calls. The online
  routing rate is counted separately from the frozen decisions (TRAIN68/229).
- TEST fallback is whatever this unchanged tau produces; no 30% cap or quantile
  is recalibrated on TEST. Accuracy remains null until all required outputs exist.

## Run on PAPER, foreground

From the repository root:

```bash
bash diagnostics/run_budgeted_paired_e2e_host.sh /extra1/bmartelli
```

Append `--check` for CPU-only validation without model inference. The default
persistent output is `context-str-runs/instructblip-budgeted-paired-e2e-v1` below
the supplied storage root. `BUDGETED_ROUTING_OUTPUT` can select another persistent
output directory. Existing source reports and results are never overwritten.

Stages: validate cached inputs -> export missing TRAIN DeepSolo -> evaluate
TRAIN CPU -> export missing TEST InstructBLIP captions -> evaluate TEST CPU.
GPU models run in separate stages/processes; MPNet always runs on CPU. InstructBLIP
uses the exact pinned TRAIN revision, checkpoint hashes, prompt and decoding.
Captions are generated once per image, not per crop. All 233 TEST DeepSolo
responses and all TEST v1/PARSeq crops are reused from the verified routing suite.
The old five-image visual InstructBLIP JSON has no complete revision/image-hash
contract and is not silently treated as a qualified TEST caption cache.

Outputs include `train-results.json`, `test-results.json`, per-image captions,
raw DeepSolo responses, CPU route decisions, and input/protocol manifests.
Writes are atomic and sealed; rerunning resumes completed shards, rejects
tampering or changed source code, and propagates failures. Cache keys include
TRAIN/TEST since numeric sample IDs overlap across splits. No large crop/image
copies or weight downloads are created.

## Verified CPU checks (2026-09-16)

All 229 TRAIN routes and 233 TEST v1/PARSeq/DeepSolo inputs pass verification.
Missing: TRAIN DeepSolo229 and qualified TEST InstructBLIP captions233. No CUDA
was executed by Codex.

Cached TEST DeepSolo control, all outputs: 861/1095 correct words (78.6301%),
1026 emitted tokens, 165 extra/wrong tokens; 111/233 exact word-multiset images.
The separate historical highest-score control recovers only 207/1095 words.
This illustrates why single-detection metrics cannot represent whole-image word
coverage. It is not a measured result of the new routed system.

Control JSON:
`artifacts/diagnostics/instructblip-budgeted-paired-e2e-v1/test-deepsolo-control.json`

SHA256: `2c622882ad3d7a08e92788f76469579f803f133005d5a3e3f1987f9faf23884d`.
