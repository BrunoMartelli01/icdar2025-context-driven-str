# Strict Reproduction Status

## 1. Status boundary

External evidence freeze date: **2026-08-19**. Direct segmentation-protocol author
clarification was incorporated on **2026-08-29**.

This document distinguishes an identifiable upstream artifact from an artifact shown to
have produced the paper's results. The former is not sufficient for a strict claim. The
machine-readable evidence is frozen in
[`provenance/SOURCES.yaml`](../provenance/SOURCES.yaml) and
[`provenance/ARTIFACTS.yaml`](../provenance/ARTIFACTS.yaml).

No complete paper experiment or evaluation has been run. A PARSeq runtime-compatibility
qualification, an approximate synthetic segmentation smoke, an M5
recognition/context interface smoke, M6 fixture-only process/fallback integration
smokes, and a physical-host Docker DeepSolo candidate qualification were completed on
2026-08-18/19, but none is a reproduction result. Therefore, no paper result is
currently reproduced—not even approximately.
“Approximation branch” below means that the evidence is sufficient to build an
explicitly labeled future branch; it is not a result claim.

On 2026-09-03 one checksum-verified ICDAR13 image completed the full APPROXIMATE
segmentation-to-DeepSolo path. This is a real-data engineering smoke, not an accuracy
evaluation or paper result. Its result JSON SHA256 is
`72099c04b754ac4dc3dc6084dd974c9201365a9bf5b1980e6c9306589906b4b7`.

## 2. Evidence-freeze findings

### Newly resolved source-discovery questions

- The authors' institutional record points to the already-audited public repository. Its
  frozen public `main` is `363cdeec0eda8999bf10d14c8f4ca6036dd422a2`; the inspected refs
  contain no tag, and the repository exposes no release assets. No author AG-UNet checkpoint,
  training code, evaluator, or second author repository was found in the inspected public
  evidence. This resolves where the available author code is, but the missing artifact remains
  a strict blocker (`ART-AGUNET-AUTHOR-CODE`, `ART-AGUNET-AUTHOR-CHECKPOINT`).
- Official PARSeq defines P-S as PARSeq-S and publishes a base `parseq` checkpoint at tag
  `v1.0.0`, commit `315d19be8ef473a864950ab497a649a69e37c6a4`. Its public filename,
  architecture, charset, preprocessing, and default decoding are recoverable
  (`ART-PARSEQ-V100-PARSEQ-CANDIDATE`). Its unchanged checkpoint, full SHA256
  `bb5792a68e367476abca029cbf8699abc805f3d3dc7e57aae45c8ec4f7b7cd00`, was
  qualified with official PyTorch-2-compatible runtime commit `1902db043c...` under
  Python 3.11/PyTorch 2.4.1+cu124. Strict loading, tokenizer IDs, preprocessing,
  decoding, CUDA inference, and six-image v1.0.0 behavioral comparison passed
  (`ART-PARSEQ-PYTORCH2-RUNTIME-CANDIDATE`). This cross-revision pairing is an
  `ENGINEERING_CHOICE` for `APPROXIMATE` reproduction; the paper still does not select
  either artifact or their pairing.
- Official Salesforce LAVIS makes the BLIP-2 discrepancy concrete: the Flan-T5-XL config
  defaults to `eva_clip_g`, while the alternate public variant is explicitly `clip_L`. No
  ViT-B BLIP-2 checkpoint is present in the inspected official BLIP-2 registry. The repository
  default `Salesforce/blip2-flan-t5-xl` is fully pin-able, including safetensor hashes, but is
  repository evidence only (`ART-BLIP2-PAPER-VITB-SELECTION`,
  `ART-BLIP2-HF-FLANT5XL-REPOSITORY-CANDIDATE`).
- `sentence-transformers/all-mpnet-base-v2` is fully pin-able at a revision and weight hash,
  with 768-dimensional mean pooling. Its use by the paper remains unsupported
  (`ART-ALL-MPNET-BASE-V2-REPOSITORY-CANDIDATE`).
- Official DeepSolo code, the legacy dependency stack, and ResNet-50/ViTAEv2-S model-zoo
  candidates for Total-Text and ICDAR15 are identifiable. Detectron2 `v0.6` resolves to commit
  `d1e04565d3bec8719335b88be9e9b961bf3ec464`. The paper does not select among the materially
  different DeepSolo checkpoint variants (`ART-DEEPSOLO-*`).
- One ResNet-50 Total-Text config/checkpoint pairing and its Docker runtime passed
  physical-host load and real inference qualification. This resolves an APPROXIMATE
  execution gate, not the paper's checkpoint selection.
- The COCO count conflict is no longer merely numerical: official COCO-TS has **14,690**
  pixel-level supervisions, whereas **43,686 train / 10,000 validation** are the official
  COCO-Text split counts printed in the paper's COCO-TS row. This resolves the origin of the
  numbers, but not the experiment (`ART-COCOTS-OFFICIAL-DATASET`,
  `ART-COCOTEXT-COUNT-ORIGIN`).
- Official sources identify the normal ICDAR13 FST split as 229 train / 233 test, ICDAR15 as
  1,000 train / 500 public test, and Total-Text as 1,255 train / 300 test. This confirms rather
  than resolves the paper's ICDAR13 229-image recognition discrepancy.

### Segmentation questions narrowed by author clarification

Direct author clarification resolves the reported 128x128 decoder/logit/loss endpoint,
the scalar one-channel additive attention defect, preservation of encoder skip width,
full-run encoder/BatchNorm freezing, all-229/no-validation ICDAR13 training population,
100 epochs, flip probability 0.5, zero-weight Don’t Care BCE, bilinear target resize plus
0.5 re-binarization, and best training-foreground-IoU checkpoint selection. These facts
are registered under `SRC-AUTHOR-SEGMENTATION-CLARIFICATION-2026-08-29`.

The exact bottleneck remains unresolved: `2048->1024` is author-recalled, not confirmed.
Exact decoder operations, attention intermediate widths/normalization, pretrained-weight
revision, image preprocessing, rotation distribution, validity resize, Adam defaults and
schedule, seed, and checkpoint-IoU aggregation/threshold also remain unresolved. The
runnable author-protocol config classifies each such choice and remains APPROXIMATE.

### Questions not resolved by the freeze

No authoritative evidence resolves the exact remaining AG-UNet decoder operations,
bottleneck width, pretrained ResNet revision, segmentation normalization/threshold,
remaining optimizer/checkpoint-metric mechanics, localization `Amin`/padding/connectivity/
ties/crop-coordinate convention, multiple-crop
aggregation, Figure 2/prose/Algorithm 2 scoring conflict, fuzzy preprocessing, caption prompt
and generation settings, MPNet selection, PARSeq selection, DeepSolo selection/output
aggregation, recognition-accuracy definition, segmentation aggregation, FLOP protocol, or
ablation sampling. These remain as recorded in
[`PAPER_AMBIGUITIES.md`](PAPER_AMBIGUITIES.md).

## 3. A. Strictly reproducible

The following evidence and paper-specified invariants can be frozen strictly:

- the exact local paper bytes and arXiv v2 identity;
- the public author repository identity and inspected revision;
- the explicitly stated high-level components and constants: ResNet-50 Attention-Gated
  U-Net family, connected-component area filtering/padding/area sort/top 10, PARSeq-S,
  BLIP-2, MPNet cosine, fuzzy/Levenshtein lexical similarity, `alpha=0.6`, `beta=0.4`,
  `tau=0.8`, and DeepSolo fallback;
- official upstream source identities and exact revisions where listed as `RESOLVED` in the
  provenance registries;
- official dataset facts that do not depend on the paper's conflicting protocol.

These are specification/provenance facts, not a complete executable configuration. **No
complete pipeline component and no paper table is currently strictly reproducible.**

## 4. B. Reproducible with documented assumptions

The following future approximation branches are evidence-backed enough to implement only if
their `ASSUMPTION` or `REPOSITORY_SPECIFIED` classification is retained in configuration and
results:

1. **PARSeq candidate branch** — unchanged official PARSeq `v1.0.0`
   `parseq-bb5792a6.pt` paired with official runtime commit `1902db043c...`, 32x128
   bicubic/0.5 normalization, 94-character training charset, and explicitly selected
   AR greedy decoding plus one refinement iteration. The pairing passed compatibility
   qualification but remains an `ENGINEERING_CHOICE`; these are upstream defaults, not
   paper proof.
2. **Repository BLIP-2 branch** — pinned
   `Salesforce/blip2-flan-t5-xl@0eb0d3b46c14c1f8c7680bca2693baafdb90bb28`, the repository
   prompt, and repository beam/length settings. This is an EVA-CLIP-g-family/Flan-T5-XL branch,
   not the paper's unresolved ViT-B configuration.
3. **Repository MPNet branch** — pinned
   `sentence-transformers/all-mpnet-base-v2@e8c3b32edf5434bc2275fc9bab85f82640a19130`
   with the frozen safetensor hash and explicit mean pooling/cosine behavior.
4. **Repository lexical branch** — RapidFuzz 3.11.0 and raw
   `rapidfuzz.fuzz.ratio`, with preprocessing and edge cases declared. Named diagnostic
   preprocessing modes remain separate. No such pin or policy is paper-supported.
5. **DeepSolo candidate branches** — isolated official DeepSolo at
   `dbadae995035246bad3376c7a44c015c69e9b313`, Detectron2 `v0.6`, and one explicitly named
   model-zoo checkpoint per benchmark/backbone/training-data variant. Each selection is an
   assumption until author evidence exists.
6. **Dataset-protocol branches** — official ICDAR13, ICDAR15, Total-Text, and official
   COCO-TS manifests/evaluators, with checksums added after acquisition. COCO-Text may be used
   only as a separately named diagnostic branch; it must not be mislabeled COCO-TS.
7. **AG-UNet reconstruction branches** — one figure-derived architecture and, separately,
   one repository-derived repaired architecture. Neither may be called the exact paper model
   while architecture/training ambiguities remain.

On RTX 3070 8 GB, BLIP-2 offload, reduced precision, quantization, sequential loading, or a
separate process would be `HARDWARE_DEVIATION`. Quantization can change captions and therefore
scores/fallback decisions. A CPU/offload branch can be reproducible, but not silently equivalent
to the paper hardware/protocol.

Physical profiling confirmed that the pinned approximate Flan-T5-XL candidate cannot be
placed wholly on this GPU in FP16: it OOMed before inference at 7,171,370,496 allocated and
7,656,701,952 reserved bytes. The unquantized FP16 CPU-offload profile succeeded with
6,938,498,560 allocated and 7,220,494,336 reserved bytes peak GPU memory and
10,865,696,768 bytes peak CPU RSS. This is a qualified approximate
`HARDWARE_DEVIATION` and does not change strict status.

## 5. C. Currently blocked

### Segmentation path

- No author AG-UNet checkpoint exists in the inspected public evidence.
- The author code is internally broken and differs from Figure 3.
- Decoder channels, bottleneck compression, attention gates, output resolution, normalization,
  thresholding, exact ResNet weights, training duration, dataset mixing, and checkpoint
  selection remain unresolved.
- The COCO-TS row cannot be instantiated from the paper's counts.
- The paper's fgIoU/F1 aggregation is undefined and Table 3 values contain an internal
  consistency concern already recorded as `E-002`.

### Recognition/context path

- The paper does not identify a PARSeq-S checkpoint, charset, preprocessing revision, code
  revision, or decoding mode. The qualified modern pairing removes only an approximate-runtime
  dependency blocker; it does not resolve this strict scientific blocker.
- The claimed BLIP-2 ViT-B model is not mapped to a public official BLIP-2 checkpoint; prompt,
  decoding, and length unit remain unknown.
- The paper does not identify an MPNet checkpoint or fuzzy-string protocol.
- Scoring/selection differs among Figure 2, prose, and Algorithm 2, including the undefined
  Algorithm 2 variables and the mapping from up to ten crops to one `T3`.
- No DeepSolo checkpoint is selected per benchmark; no policy maps multiple text-spotting
  outputs to singular `Tfinal`.
- Recognition “accuracy” and its normalization/matching/denominator are undefined. Official
  dataset evaluators cannot be substituted without evidence.

### Ablations and efficiency

- Wrong-description sampling, no-description representation, detector-assisted comparison,
  seeds/repetitions, and statistical aggregation are not specified.
- FLOP and latency protocols are not specified; BLIP-2's reported latency is not credible as a
  target on this hardware without a frozen model/precision/protocol.

## 6. Paper-table claim matrix

| Paper target | Strict reproduction claim now | Approximate reproduction claim now | Current status |
|---|---:|---:|---|
| Table 3 — segmentation fgIoU/F1 | No | No experiment run | **Not reproducible yet**; architecture, checkpoint/training, COCO-TS protocol, and metric aggregation block it |
| Table 4 — efficiency | No | No experiment run | **Not reproducible yet**; artifact set, FLOP method, precision, timing, and hardware differ/are unspecified |
| Table 5 — recognition accuracy | No | No; one-image real-data smoke only | **Not reproducible yet**; recognizer/caption/embedder/fallback selections and accuracy definition block it |
| Table 6 — alpha/beta/tau ablation | No | No experiment run | **Not reproducible yet**; inherits Table 5 blockers and scoring ambiguity |
| Table 7 — context-description ablation | No | No experiment run | **Not reproducible yet**; inherits Table 5 blockers plus caption/ablation sampling ambiguity |

No smoke test, subset run, standard-evaluator result, or candidate-artifact result may be
presented as one of these paper tables.

## 7. Implementation boundary after this milestone

The following work can safely proceed when requested without pretending an ambiguity is solved:

- qualify additional provenance-defined DeepSolo candidates only without silently
  modernizing the frozen runtime or treating them as paper-selected;
- provenance/config schema validation and hard separation of `configs/strict/` from
  `configs/approximate/` and `configs/demo/` branches;
- dataset acquisition/manifests for ICDAR13, ICDAR15, Total-Text, and the official 14,690-mask
  COCO-TS dataset, retaining archive checksums and terms after download;
- parameterized connected-component localization mechanics that exactly preserve component
  area filtering, padding, area sorting, and top 10, while leaving `Amin`, `p`, connectivity,
  tie behavior, and crop-coordinate convention explicit and unresolved;
- interfaces for official PARSeq and DeepSolo candidates, provided candidate selection stays
  outside strict paper configuration until evidence exists.

The AG-UNet reconstruction, BLIP-2 choice, MPNet choice, fuzzy protocol, score aggregation,
DeepSolo checkpoint mapping, and recognition evaluator require explicit approximation branches
unless new author evidence is obtained.

Macro-Milestone 4 adds a separate figure-derived `ApproximateAGUNet`, paired transforms,
training/checkpoint infrastructure, Algorithm 1 localization, and explicit macro/micro
foreground metrics. This establishes an approximate synthetic smoke path only. It does not
change any STRICT blocker or the paper-table claim matrix above. Batch-size-8 full-model CUDA
training and peak-VRAM measurement passed on the physical RTX 3070 at batch size 8
(approximately 1.99 GB allocated and 2.49 GB reserved); the sandbox still does not
expose CUDA.

Macro-Milestone 8 freezes that same figure-derived model as
`approximate-ag-unet-figure3-v1`; it does not create a competing reconstruction or modify
the broken repository `AdvancedUNet`. The static contract fixes named topology/tensors,
48,454,733 parameters, four attention-skip interfaces, input/output geometry,
initialization policy, and a 456-tensor state-schema fingerprint. Architecture-aware
checkpoint loading is strict and rejects metadata/key/shape/dtype incompatibilities.
These are conformance properties of the APPROXIMATE implementation, not author-model
evidence. The M8 batch-1 float32 CUDA forward passed on the physical NVIDIA GeForce RTX
3070 with `[1,3,256,256]→[1,1,128,128]`, 264,071,680/289,406,976 bytes peak
allocated/reserved memory, and no OOM; the physical full suite passed 142/142 in 30.68
seconds. The lazy-CUDA profiler initialization and deterministic-test isolation fixes are
engineering qualification changes only. A-001 through A-006 remain unresolved.

Macro-Milestone 6 adds an explicit JSON subprocess contract, provenance-named candidate
registry, non-silent errors, separately named multi-output reduction assumptions, a
one-shot sequential lifecycle, and no-fallback/constructed-fallback fixture smokes.
The frozen ResNet-50 Total-Text source/config/checkpoint and Docker runtime passed real
physical-host CUDA inference with cuDNN enabled. The final repository-owned staged
worker boundary also passed with three detections and 869,216,256/1,270,874,112 bytes
peak allocated/reserved CUDA memory; the physical full suite passed 122 tests. Other
candidates and the paper's artifact mapping remain unresolved. These engineering
results do not change any Table 5/6/7 claim.

Macro-Milestone 7 makes the Algorithm 1 mechanics auditable without promoting missing
paper values: true foreground-pixel component area is independent of bbox area, padding
is clamped, sorting is area-descending with an explicit approximate spatial tie rule, and
the top-10 cap records overflow rejections. Tiny golden JSON/PNG fixtures preserve boxes,
areas, ordering, rejection reasons, and final crops. L-001/L-003/L-005 and upstream mask
threshold/resolution remain unresolved, so this is an APPROXIMATE implementation boundary,
not a STRICT reproduction result.

Physical-host validation on 2026-08-19 passed the focused localization suite
(12 tests in 0.31 seconds), the full repository suite (129 tests in 18.76 seconds),
and `git diff --check`. The golden JSON/PNG files are deterministic test fixtures only,
not dataset samples or experiment outputs. These results do not change the STRICT status.

## 8. Current handoff

The main environment and M4-M8 approximate foundations are implemented to the limits
recorded above. M9 adds a real-manifest-only APPROXIMATE trainer around those foundations:
registered upstream dataset artifacts, checksum verification, explicit disjoint
train/validation sources, deterministic multi-epoch ordering, validation BCE hooks, exact
model/optimizer/epoch/step/RNG continuation, architecture-aware fail-closed resume, JSONL
epoch logs, and checkpoint/run/failure manifests. Synthetic fallback is absent from the M9
entry point. The full configured APPROXIMATE M9 training run is complete; no Table 3
evaluation has started.

M10 now has an explicitly frozen APPROXIMATE-only ICDAR13 evaluation contract and a
checksum-complete 233-sample official-test manifest. It thresholds native 128x128 sigmoid
outputs inclusively at 0.5 before nearest binary backprojection, evaluates exact-white GT
at original geometry with Don't Care retained as foreground/all pixels valid, reports
per-image macro as primary and global micro as diagnostic, and assigns 1.0 to both-empty
cases. This policy is separate from M9 training semantics. Physical CUDA evaluation has
not run, so there is no Table 3 result; all listed STRICT blockers remain.

The APPROXIMATE ICDAR13 source now has 229 checksum-verified image/raw-BMP pairs and frozen
206/23 manifests under the adopted SHA-256 ranking assumption. RRC remains the official
upstream identity; the exact archives came from separately recorded public mirror URLs and
remain `SECONDARY` acquisition evidence. Split generation is byte-deterministic, portable,
disjoint, and complete. This resolves only the repository's approximate split choice.

M9 now enforces exact RGB white as background and every other RGB value as foreground before
nearest-neighbor mask resize. Don't Care remains foreground and TXT is audit-only under an
explicit APPROXIMATE `ASSUMPTION`; BCE is unchanged. The two-epoch, transform, Adam-default,
trainable-encoder, dataset-specific, pretrained-artifact, and broader protocol assumptions
also remain non-STRICT.

The short physical M9 engineering qualification passed on an NVIDIA GeForce RTX 3070
8192 MiB with driver 595.84, torch 2.4.1+cu124, CUDA float32, 48,454,733 parameters,
batch 8, image shape `[8,3,256,256]`, and mask shape `[8,1,128,128]`.
`qualification-max-steps=1` yielded one train step per configured epoch and two total steps
(`global_step=2`). Epoch train/validation losses were
`0.9355729818344116`/`0.2795393764972687` and
`0.9342144727706909`/`0.3452925980091095`; `success=true`, `oom=false`, with
2,373,442,560/2,552,233,984 bytes peak allocated/reserved CUDA memory. The external final
checkpoint SHA-256 is `f06db536167ccd3d881adababaf00b2d6cefbb8cf6524bd22d303a908540e45e`.
The physical suite passed 171 tests in 38.56 seconds and `git diff --check` passed. This
qualifies the APPROXIMATE engineering path only; it is not full scientific training and not
a Table 3 result. DS-003 remains unresolved for STRICT. STRICT execution continues to fail
on A-001 through A-008, T-001 through T-004, DS-001, and DS-003.

Separately from that short qualification, the full configured M9 APPROXIMATE run trained
`approximate-ag-unet-figure3-v1` (48,454,733 parameters) on 206 samples with 23 validation
samples using RTX 3070 CUDA float32, batch 8, two epochs, and 26 steps per epoch
(`global_step=52`). Adam used learning rate `1e-4`, weight decay `0.01`, no scheduler, and
last-checkpoint selection. Epoch train/validation losses were
`0.8646002301892031`/`0.8293485304583674` and
`0.7025007738650424`/`0.6749140412911124`. The run reported `success=true`, `oom=false`,
with 2,373,967,360/2,552,233,984 bytes peak allocated/reserved CUDA memory. The external
`checkpoints/epoch-0002.pt` SHA-256 is
`335a003ae00cf21d159151b9568e10eafde12d6ae524bf7505e1d62ae4c3bfe2`; its bytes remain
outside Git. It is recorded for future APPROXIMATE evaluation only. This is not a STRICT
checkpoint or a Table 3 result, and all existing blockers and assumptions remain unchanged.

The separate author-protocol APPROXIMATE branch completed 100 epochs on an RTX 3090. Its
frozen downstream checkpoint is epoch 98, global step 2842, SHA-256
`b450f19e79f649f8873a8b8db0d9a25ab0a7f144ac5f8676a2a80d5c1d8fd2f2`, selected by
training per-image macro foreground IoU. It has 58,155,081 total and 32,598,049 trainable
parameters because the ResNet-50 encoder and BatchNorm state are frozen; logits remain
`128x128` and the inclusive threshold remains `0.5`. The Task-1-derived validity mask is
evaluation-only: the standard and validity-aware evaluation runs saved byte-identical
prediction masks. STRICT remains blocked by the unconfirmed bottleneck, incomplete
decoder/attention operations, weight/preprocessing and optimizer/augmentation/metric
details, COCO-TS protocol, and Table 3 evaluator ambiguities.

On host `paper`, Docker is administratively inaccessible and must be treated as read-only;
therefore the earlier qualified DeepSolo Docker runtime is operationally `BLOCKED` here.
A repository-owned non-Docker Python 3.8 subprocess reconstruction path now exists, using
the frozen legacy versions and immutable source/config/checkpoint checks. On 2026-09-02 it
passed exact environment validation, strict checkpoint compatibility, the staged worker
boundary, and real CUDA inference on an RTX 3090. Three detections reduced to
`deepsolo test 2026`; inference took 0.147001147037372 seconds and peak CUDA memory was
869,216,256/1,270,874,112 bytes allocated/reserved. It is qualified for APPROXIMATE use
as a `HARDWARE_DEVIATION`; it does not inherit the Docker qualification or resolve the
existing DeepSolo/Table 5 scientific blockers.

The pinned CPU/float32 MPNet and raw RapidFuzz scoring branch also passed physical-host
qualification on 2026-09-03. Repeated batched embeddings were bit-identical,
batch-versus-single encoding differed by at most `8.195638656616211e-08`, and all three
named score strategies ran with the paper-specified `alpha=0.6`, `beta=0.4`, and
`tau=0.8`. This establishes only deterministic operation of the APPROXIMATE branch;
the exact MPNet checkpoint, fuzzy preprocessing, aggregation, accuracy definition, and
evaluation protocol remain unresolved, so it is not a Table 5 result.
