# Total-Text word-kernel U-Net experiment

`PRACTICAL_APPROXIMATION` / approximate reproduction. This is neither strict
Algorithm 1 nor a PSENet reproduction. No existing model, v4 target, localizer,
checkpoint, PARSeq configuration or completed result is changed.

## Motivation and evidence

The authoritative local paper `papers/2503.15639v2.pdf`, Sections 3.2/3.3,
specifies an attention-gated ResNet50 U-Net, binary text segmentation and
component crops. It does not specify instance nuclei or expansion. The new
experiment is inspired by [PSENet](https://arxiv.org/abs/1903.12473), which
separates nearby text instances with smaller kernels and progressive expansion.
Here we deliberately use only two maps and one expansion stage, not PSENet's
multiscale polygon-offset targets, backbone, losses or full expansion algorithm.

The frozen letterbox comparison on held-out TRAIN125 / 983 care words is:
threshold 0.65, geometry IoU>=0.5 552, exact casefold 437, exact after removing
Unicode punctuation/symbols 469. Its report is
`context-str-runs/totaltext-letterbox-threshold-sweep-train-v1/results.json`
(SHA256 `613957db6019d0762a4d890495e3c1c0af5f91a92ced9c232c099a103328a334`).

## Frozen intervention

- Same qualified v4 full word masks, validity/Don't Care, separation maps,
  ImageNet normalization, paired flip/rotation and aspect-preserving letterbox.
- Same input512/native256, frozen pretrained ResNet50/BN, attention skip
  floor0.25, optimizer, seed202503 and frozen TRAIN1130 / validation125 split.
- Two output channels: full word foreground and internal word nucleus.
  Channel0 initialization matches the old one-channel model under the same seed.
  Added parameters: **65**; total58,155,146; trainable32,598,114. No frequency module.
- Full loss remains separation-weighted BCE (weight3). Nucleus loss is
  equally positive/negative balanced masked BCE, added with weight1. Ignore
  pixels have zero contribution to both losses. There is no OCR loss.
- Exactly100 epochs. Checkpoint ordering: held-out word proposal F1@0.5,
  fewer GT words touched by fused instances, recall, macro foreground IoU,
  earlier epoch. These are validation signals, never inference inputs.

### Nucleus targets (`ASSUMPTION`)

Owner IDs are replayed through exactly the same augmentation/letterbox and
nearest native resize as the qualified preprocessing, then clipped to the
UNCHANGED full target and validity. For each surviving native owner, compute
its Euclidean distance to the zero-padded background and keep pixels with
distance >=0.5 times that owner's maximum distance. Compare integer squared
distances to avoid floating-point boundary drift under exact ROI optimization.
Keep the largest8-connected
core, deterministic component-order tie breaking. This is a native distance
core, **not an original-polygon offset or a letter/glyph mask**. A one-pixel
word keeps a nucleus; a word already lost in native preprocessing is reported,
not invented. Disconnected core fragments discarded by this rule are counted.

Preflight qualifies all1255 official TRAIN images, including the validation
subset: one separate connected nucleus per surviving owner, no nuclei outside
valid full foreground. Touching nuclei fail closed. This is a target-integrity
check, not parameter selection. Existing full target pixels are not changed.

### GT-blind inference (`ASSUMPTION`)

Freeze full sigmoid>=0.65 from the previous TRAIN-only study; nucleus
sigmoid>=0.5 is declared before this experiment. Clip both to the RGB canvas,
not GT validity. Intersect predicted nuclei with predicted full foreground.
Find8-connected nucleus components and expand all seeds simultaneously by
8-neighbour breadth-first search **inside full foreground only**. Equal-hop
ties use deterministic initial pixel and neighbor queue order. Labels never
merge. A full foreground component with no seed is retained as one fallback
instance, preventing automatic deletion of unseeded text. A fragmented full
mask can still produce multiple instances; this method cannot hallucinate
missing foreground or guarantee one seed per predicted word.

Final boxes are per-instance extrema, then inverse-letterbox mapped to original
RGB. **Do not run contours on the binary union of instance labels**: adjacent
labels would fuse again. Pad0, no area filter and no top10 remain the existing
practical choices (different from paper Algorithm1). We do not add grouping,
OCR-based grouping, morphology, or a threshold grid.

## Execution and outputs

On PAPER, foreground with target/epoch/mask/PARSeq progress:

```bash
cd /extra1/bmartelli/projects/icdar2025-context-driven-str && bash scripts/run_totaltext_word_kernel_100ep_host.sh
```

The launcher reuses `.venvs/context-str-main-py311`, installs nothing, runs
CPU target qualification, one CUDA training, then PARSeq on held-out TRAIN125.
It does NOT evaluate TEST. Segmentation and PARSeq are loaded sequentially.
Sequential loading and the configurable PARSeq batch are recorded in the
resolved evaluation protocol as `HARDWARE_DEVIATION`. Predictions may depend
on numerical device/batch behavior; cache reuse is not proof of equivalence.
Identical RGB crops reuse checksum-verified earlier OCR. No crop PNG copies are
saved: metadata and lossless native instance/nucleus PNGs suffice to reproduce
the RGB crops. A runner lock prevents concurrent writes.

Persistent outputs under `context-str-runs/totaltext-word-kernel-100ep-v1/`:

- `target-qualification.json`: full TRAIN integrity and per-image hashes.
- `word_kernel_100ep/{run_contract.json,epochs.jsonl,best.pt,run_manifest.json}`.
- `word_kernel_100ep/validation_word_geometry.json`: native instance geometry.
- `validation-pad0-parseq-v1/results.json`: original-coordinate one-to-one
  geometry@0.25/0.5/0.75, exact casefold, punctuation/symbol-stripped exact,
  proposal precision/recall/F1, baseline and per-word recoveries/losses.
- `validation-pad0-parseq-v1/{samples,masks,kernels,ocr}/`: auditable caches.

Only best/latest are kept during training; latest is removed after verified
success. Rerunning the same command resumes the latest atomic checkpoint,
optimizer and RNG states; a mismatched contract/checksum fails. A process
interrupted between checkpoint and epoch-log writes is recovered explicitly.
OOM and other training failures remain visible and create `failure.json`.

Repeated use of this validation population makes conclusions exploratory.
Assess recognition and fusion/split errors, not only pixel IoU. No improvement
or GPU-memory qualification is claimed before the user executes the new run.

## Implementation qualification (CPU only)

26 focused/regression tests passed: nuclei containment/connectivity, border and
small-word survival, curved/disconnected targets, deterministic fused-region
partition, unseeded fallback, instance-aware matching, paired owner mapping,
Don't Care gradients, both training heads, strict checkpoint/optimizer/RNG
roundtrip, native thresholds and lossless instance caches. `py_compile`, shell
syntax and `git diff --check` passed. No real training, CUDA or OCR ran locally.

All1255 TRAIN images passed target qualification:9,284 surviving native word
targets and9,284 separated nuclei. One word in TRAIN image `total_text:121`
was already lost in the existing preprocessing (not in the validation125);
the new method does not alter the full mask to recover it. Largest-core
selection discards disconnected core pieces where present; this is explicit,
not proof of perfect predicted nuclei.

An initial float-distance ROI comparison exposed threshold-boundary drift on
66 images. Integer squared-distance comparison fixed it without changing the
declared half-distance threshold. Full-canvas versus ROI regression passed on
all66 affected images; the final full TRAIN qualification was rerun afterwards.
The persistent local audit is
`artifacts/diagnostics/totaltext-word-kernel-implementation-v1/target-qualification.json`.
