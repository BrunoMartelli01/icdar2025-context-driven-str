# Text+Kernel: three controlled word-evidence tests

**Approximate reproduction / experimental extension**, WORD-EVIDENCE-001.
User authorized all three tests on2026-10-17. Inspected local paper3.2/3.3/4.1,
existing Text+Kernel/glyph datasets, loss/partition/evaluation implementations,
V1/V2 completed OCR and audits before implementation. No STRICT default changes.

## Frozen source and common protocol

Use the qualified V1 control best epoch2, SHA25647a941e1..., whose OCR is520/983,
and the V2 isolated glyph best epoch50, SHA256ca8ec43d.... All original word
weights must equal the control tensor-by-tensor. Glyph is binary stroke evidence,
not individual character IDs. Existing validation probability maps must be
bit-identical and replay the original control instance labels exactly.

TRAIN1130 predictions fit two small16-hidden CPU MLPs. Validation125/983 words
is held out from fitting, but repeatedly reused historically: exploratory, not
independent TEST. No GT/validity/transcription is passed to feature extraction,
association or correction inference. Word-owner IDs are used only for TRAIN
supervision; annotation GT enters evaluation after proposals are frozen.

The word network is not retrained. This is the smallest controlled test of the
proposed associations and crop objective. Full shared-decoder training is a
separate, unmeasured alternative, not implied by these fitted modules.

## 1. Crossed Text/Kernel oracle diagnostic

Evaluate predicted/predicted, GT-Text/predicted-Kernel, predicted-Text/GT-Kernel,
GT/GT, plus glyph-GT/GT-Kernel through the existing frozen watershed. Regenerate
GT kernels from the same transformed word-owner maps and distance fraction.
All GT-containing rows are **oracle diagnostics**, never inference results or
an upper bound on PARSeq. This isolates support/separation and explicitly tests
the incompatible disconnected-glyph foreground with continuous word kernels.

## 2. Learned fragment association

Predict glyph connected components and existing kernel-guided word instances.
Components are fragments, not assumed characters. For each fragment consider
four nearest predicted instances with actual kernel support. Twelve geometric/
probability features include normalized gaps/centers/spans, overlap, kernel
pixel distance and Text/Kernel/Glyph probability summaries. Source predictions,
not annotations, define the entire candidate population.

TRAIN positive pairs have the same majority GT-word owner. Background pairs
are negative; ambiguous-majority targets are explicitly counted and unsupervised.
Balanced positive/negative pair BCE fits a small MLP;50 full-batch updates,
Adam1e-3/decay1e-4, seed20261017. Choose minimum measured TRAIN loss, with state
saved before the associated update; no holdout family/cut selection. Threshold.8
is an initial untuned conservative choice, not a calibrated correctness guarantee.
Each fragment attaches once. No existing pixel/ID is overwritten or merged;
conflicting fragments are preserved as diagnostics. No orphan new word IDs are
invented. This tests word completeness, not arbitrary split/merge repair.

## 3. Crop-completeness supervision

Twelve GT-free features of each existing predicted instance feed a separate
16-hidden MLP producing four signed bbox-side corrections. Each correction is
bounded to25% of the corresponding source width/height; zero initialization
replays the original boxes. Majority GT-owner matches supervise TRAIN proposals.
Unmatched/ambiguous proposals are counted and unsupervised, not removed at eval.

Loss: normalized SmoothL1 + missing GT-bbox coverage +.25 excess area +.25
foreign bbox overlap beyond the overlap already present in the own GT bbox.
This uses **bbox proxies**, not polygon/foreground contamination. Actual crop
bounds are clipped and outward-rounded; boxes remain exclusive xyxy. All source
instance identities and evaluation-ignore flags remain fixed for both new methods.

## Metrics, hardware and artifacts

Primary localization: maximum-cardinality one-to-one original-RGB bbox IoU>=.50.
OCR additionally requires exact Unicode casefold; punctuation retained.
All983 GT care words remain denominators; all care proposals count as predictions,
including empty OCR. Preserve wholly-invalid source-instance ignore, keeping full
partially-invalid boxes. Pad0/no area filter/no cap are inherited practical
assumptions, not paper Algorithm1. Report localized count, OCR count, precision,
recall,Hmean and per-GT gains/losses against the freshly replayed baseline.

Existing RTX3090, FP32 batch4 frozen-network export; CPU head training and PARSeq
FP32 batch16, AR/refine1, strict qualified checkpoint/runtime. These are
**HARDWARE_DEVIATION**. The independent GT-V3 GPU training is preserved; external
GPU processes/free VRAM are recorded before/after export. At least8GiB free is
required. No new network training competes with that job. Timing is not a latency
benchmark; own allocated/reserved peaks do not measure external job peaks.

Resolved configs, source hashes, dataset/split/checkpoint hashes, input ordering,
per-image frozen maps/proposals/OCR, head weights/loss history, CUDA peaks/OOM
failure receipts and machine-readable results remain outside Git. A snapshot
freezes code/config for execution despite concurrent repository edits. Completed
stages can resume only under the same contract and hash-verified evidence.
Benefits and independent generalization remain **UNRESOLVED** until measured.
