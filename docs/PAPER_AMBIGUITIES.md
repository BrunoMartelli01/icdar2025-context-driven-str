# Paper Ambiguities and Reproduction Blockers

## 1. Purpose

This is the permanent evidence and ambiguity ledger required by `AGENTS.md`. It records contradictions, omissions, and repository assumptions that can materially change scientific results. No entry is resolved merely because the repository contains a default.

Status vocabulary:

- `UNRESOLVED`: no authoritative evidence currently determines the strict behavior.
- `PARTIALLY_EVIDENCED`: some behavior is explicit, but material details remain unknown.
- `BLOCKING`: a strict result must not be claimed until this entry is resolved or the result is labeled approximate.

Any future resolution must add the evidence source, date, immutable revision or correspondence where possible, decision classification, and experiments needed to test sensitivity.

## 2. Architecture and segmentation

### A-001 — Repository AG-UNet final attention skip defect

- **Status:** `RESOLVED` by direct author clarification for the reported protocol.
- **Paper evidence:** Figure 3 shows a 64-channel encoder feature at `128×128` and a 64-channel decoder feature at that resolution.
- **Repository evidence:** `models/unet_adv.py` defines `up0` with a 64-channel encoder skip and `out_ch=128`. `AttentionGate` creates a 128-channel attention map and returns `x * att`, where `x` has 64 channels.
- **Conflict:** The multiplication is channel-incompatible. The current code is not a verified implementation of Figure 3.
- **Scientific impact:** No existing checkpoint architecture or segmentation result can be trusted against this class until the intended channel contract is established.
- **Resolution:** The intended coefficient is one-channel additive spatial attention, broadcast over the unprojected encoder skip. The historical multi-channel coefficient is a repository defect. Exact intermediate projection widths remain under A-004.

### A-002 — Decoder operations remain incomplete after endpoint clarification

- **Status:** `PARTIALLY_EVIDENCED`, `BLOCKING` for STRICT topology identity.
- **Paper evidence:** Figure 3 visually labels decoder features `512@16×16`, `256@32×32`, `128@64×64`, and `64@128×128`.
- **Repository evidence:** `AdvancedUNet` uses decoder outputs `(1024, 512, 256, 128, 64)` and therefore produces `1024@16×16` before continuing.
- **Conflict:** The repository inserts an additional/wider decoder stage relative to the apparent paper diagram.
- **Scientific impact:** Parameter count, FLOPs, VRAM, feature capacity, checkpoint compatibility, and segmentation quality all change.
- **Required evidence:** Exact layer/channel specification or original trained checkpoint with inspectable tensor shapes.
- **Author clarification:** The reported network terminates at `64@128x128`; the later 256x256 upsampling stage is not part of it. Figure 3 supplies `512/256/128/64`, but exact upsampling and decoder-block operations remain unconfirmed.

### A-003 — Bottleneck compression is described but not implemented

- **Status:** `UNRESOLVED`, `BLOCKING`.
- **Paper evidence:** Section 3.2 says the deepest layer is compressed by convolution + BatchNorm to reduce channel dimensionality; Figure 3 suggests a transition from `2048@8×8` toward `512@16×16`.
- **Repository evidence:** The bottleneck contains two `2048→2048` 3×3 convolutions and does not reduce channels. Channel reduction happens later in an up block to 1024.
- **Conflict:** “Compression” and the depicted 512-channel decoder are inconsistent with the source.
- **Scientific impact:** The repository bottleneck is extremely large and materially affects optimization and 8 GB feasibility.
- **Required evidence:** Intended bottleneck output channels, number of convolutions, order of upsampling, kernel size, activation, and relationship to the first decoder block.
- **Author clarification:** Compression is required; `2048->1024` is recalled but explicitly unconfirmed. The runnable width and block remain `ASSUMPTION`, so A-003 stays blocking.

### A-004 — Attention-gate implementation is underspecified

- **Status:** `PARTIALLY_EVIDENCED`, `BLOCKING` for STRICT topology identity.
- **Paper evidence:** Attention gates appear in every skip path and receive encoder feature `x` plus gating signal `g`; a cited attention approach is referenced.
- **Missing detail:** Projection channels, normalization, interpolation, scalar versus channel-wise attention, gate placement, activation, and whether every shown skip is gated.
- **Repository assumption:** Additive attention with Conv/BN/ReLU, channel-wise `psi`, nearest interpolation for `g`, and bilinear interpolation for a mismatched skip.
- **Scientific impact:** Changes feature suppression, parameter count, numerical behavior, and checkpoint layout.
- **Required evidence:** Original module definition or complete equations and dimensions.
- **Author clarification:** Scalar additive attention, ordering, one-channel `psi`, broadcast multiplication, and preservation of skip channels are confirmed. Intermediate widths and normalization details remain unresolved.

### A-005 — Mask/logit resolution is contradictory

- **Status:** `RESOLVED` for the reported network/loss endpoint by direct author clarification.
- **Paper evidence:** Algorithm 1 defines a mask at original `H×W`; Section 4.1 says images are resized to `256×256` and masks are downsampled to `128×128`; Figure 3 shows decoder features at `128×128` but does not label final mask dimensions.
- **Repository evidence:** The model produces logits at input resolution (`256×256`), then the inference wrapper resizes probabilities to the original image resolution.
- **Missing detail:** Whether training loss is evaluated at `128×128`, logits are upsampled to 256, masks are upsampled, or the paper’s “down-sampled” statement is erroneous.
- **Scientific impact:** Affects loss values, thin-character preservation, localization components, fgIoU/F1, and checkpoint shapes.
- **Resolution:** Reported logits and BCE target are 128x128 for 256x256 input. The later 256x256 stage is localization-only and historical 256/512 experiments remain ablations.

### A-006 — ResNet-50 pretrained weights and encoder training state are unknown

- **Status:** `UNRESOLVED`.
- **Paper evidence:** Encoder is “pre-trained ResNet50”; other modules remain frozen while the U-Net is trained.
- **Repository evidence:** Defaults to torchvision `IMAGENET1K_V2` and leaves encoder trainable unless configured otherwise.
- **Missing detail:** Weight revision and framework remain unknown. The author confirms the full encoder is frozen for the entire run, including BatchNorm state.
- **Scientific impact:** Initialization and BatchNorm statistics can materially alter Table 3.
- **Required evidence:** Checkpoint provenance and training parameter groups.

### A-007 — Image normalization and resize geometry are unknown

- **Status:** `UNRESOLVED`, `BLOCKING` for faithful preprocessing.
- **Paper evidence:** Only a resize to `256×256` is stated.
- **Repository evidence:** Direct aspect-ratio distortion, RGB scaling to `[0,1]`, no ImageNet mean/std, and antialiased tensor resize when torchvision is available.
- **Missing detail:** Mean/std, color order, aspect preservation, interpolation, antialiasing, and training/inference parity.
- **Scientific impact:** Particularly important for pretrained ResNet features and small text.
- **Required evidence:** Original transform pipeline.

### A-008 — Segmentation threshold and mask postprocessing are absent

- **Status:** `UNRESOLVED`.
- **Paper evidence:** A binary mask is consumed, but conversion from probability/logit to binary is not specified.
- **Repository evidence:** Sigmoid followed by threshold 0.5; no morphology.
- **Scientific impact:** Directly controls connected components and segmentation metrics.
- **Required evidence:** Threshold selection and any morphological/filtering steps.

## 3. Segmentation training protocol

### T-001 — Training duration is resolved; remaining optimizer/selection mechanics are incomplete

- **Status:** `PARTIALLY_EVIDENCED`, `BLOCKING` for STRICT training identity.
- **Specified:** Adam, learning rate `1e-4`, weight decay `0.01`, batch 8, BCE, resize, flips, rotations.
- **Author clarification:** 100 epochs, batch 8, no held-out validation, and best training-set foreground IoU selection over all 229 images.
- **Still missing:** Schedule/warmup, Adam betas/epsilon, IoU aggregation/threshold/tie policy, initialization, seed, and repetitions.
- **Scientific impact:** Prevents deterministic or statistically comparable training.
- **Required evidence:** Training config/logs or author clarification.

### T-002 — Dataset training regime is unknown

- **Status:** `UNRESOLVED`, `BLOCKING`.
- **Paper evidence:** ICDAR13 FST, TotalText, and COCO-TS are used “for training and testing” the U-Net, and Table 3 reports each dataset.
- **Missing detail:** One joint model versus three dataset-specific models; cross-dataset mixing; validation splits; sampling balance; pretraining/fine-tuning order.
- **Scientific impact:** Dataset size imbalance would dominate a joint training regime and change all results.
- **Required evidence:** Per-experiment dataset/config records.

### T-003 — Augmentation is incompletely defined

- **Status:** `UNRESOLVED`.
- **Specified:** Horizontal flips and random rotations.
- **Author clarification:** Horizontal flip probability is 0.5. Random rotation was used, but its range is unknown; +/-15 degrees is only a suggested `ASSUMPTION`. Rotation probability, fill, interpolation, order, and seed remain unknown.
- **Scientific impact:** Text orientation and thin masks are highly sensitive to these choices.
- **Required evidence:** Original transform code or full parameter description.

### T-004 — Paper hardware statement is inconsistent with the named GPU

- **Status:** `UNRESOLVED`.
- **Paper evidence:** “single RTX 2080 Ti (15 GB).” A standard RTX 2080 Ti has 11 GB.
- **Target:** Single RTX 3070 with 8 GB.
- **Scientific impact:** Unknown original memory optimizations; batch 8 may not transfer exactly.
- **Rule:** Any mixed precision, smaller batch, accumulation, checkpointing, or offload differing from author evidence must be a documented `HARDWARE_DEVIATION`.

## 4. Localization

### L-001 — Numeric `Amin` and padding `p` are missing

- **Status:** `UNRESOLVED`, `BLOCKING` for exact crop reproduction.
- **Paper evidence:** Both parameters exist but have no numeric values.
- **Repository evidence:** `Amin=150`, `p=6` pixels.
- **Scientific impact:** Determines region count, crop extent, recognizer input, and fallback rate.
- **Required evidence:** Author configuration or a disclosed calibration protocol. Repository values remain `REPOSITORY_SPECIFIED`.
- **TRAIN diagnostic (2026-09-24):** `configs/approximate/totaltext_train_native_padding_sweep_v1.json` predeclares native128 padding values 0, 2, 4, 6, 8, 10, 12 around the repository value 6. It reuses one frozen Total-Text v1 mask per TRAIN image, disables the bbox-area filter to match the preceding isolated no-area diagnostic, and compares proposal geometry and PARSeq OCR. The sweep values and native coordinate frame are `ASSUMPTION`; the experiment is `PRACTICAL_APPROXIMATION_DIAGNOSTIC` and cannot resolve the paper's missing numeric padding.

### L-002 — Repository uses bounding-box area instead of component area

- **Status:** `UNRESOLVED` as to historical intent; paper behavior itself is clear.
- **Paper evidence:** Filter and sort using `A(Ci)`, the component/contour area.
- **Repository evidence:** Filter by unpadded rectangle `w×h`; sort by padded rectangle area.
- **Conflict:** These are not equivalent for sparse, curved, or irregular components.
- **Scientific impact:** Changes retained top-10 components and their order.
- **Strict requirement:** Future paper path must use component area unless new authoritative evidence contradicts Algorithm 1.

#### L-002 implementation audit, 2026-09-16 — isolated repository-contours-v1

- **REPOSITORY_SPECIFIED:** `localize/repository_contours.py` reimplements the author file from frozen commit `363cdeec0eda8999bf10d14c8f4ca6036dd422a2`, git blob `d9359417120c09a9730ddc529c59caa4c0690bb0`, source SHA256 `3e0d779ae761e038f48525c4a169e3e3be7e1849ddf85c73f309ce3d0cad7dbc`. Current online `main` was not accessible; no claim is made that it is unchanged.
- **REPOSITORY_SPECIFIED:** external contours, `boundingRect`, unpadded bbox area filtering, padding clipped to original-image bounds, descending **clipped padded** bbox area, stable OpenCV contour order for ties. Values 150/6/10 are explicit in `configs/approximate/localization_repository_contours_v1.json`.
- **PAPER_SPECIFIED:** padding, top-10 cap, original-image crops. Rectangle-area filtering/sorting is not equivalent to the paper's component-area rule. The existing Algorithm 1 implementation and pipeline defaults remain unchanged.
- **UNRESOLVED:** this method does not group disconnected glyph fragments, and external-only retrieval may omit nested foreground islands. Source parity is not evidence of better word localization or reproduced Table 5 accuracy. Differential tests use the exact hash-verified frozen source on binary inputs; the wrapper additionally rejects invalid masks/parameters.
- **ASSUMPTION:** the optional CPU saved-mask comparison uses the existing approximate 8-connectivity/spatial-tie baseline on the identical original-size mask. It does not resize native masks, rerun segmentation, change thresholds, run OCR, or use GT to construct boxes.

#### Native128 repository-contour diagnostic, 2026-09-16

- **ASSUMPTION:** the user explicitly requested contour extraction, Amin=150 and padding=6 entirely in native128 coordinates, without any mask resizing. This is isolated in `localize/native_repository_contours.py` and `configs/approximate/localization_repository_contours_native128_v1.json`; no baseline is replaced.
- **REPOSITORY_SPECIFIED:** Amin still filters unpadded bounding-rectangle area, not foreground pixels. Sort uses clipped padded rectangle area at native resolution and preserves contour ties. Applying the same numeric parameters at native resolution changes their effective original-image scale; it is not equivalent to the original repository's original-size mask input.
- **ASSUMPTION:** remap only retained padded boxes with floor for lower edges and ceil for exclusive upper edges, then crop the original RGB. No native-image crop is fed to PARSeq. No downsampling, interpolation, morphology or grouping is introduced.
- **PAPER_SPECIFIED:** maximum ten original-image crops and padding are retained. The paper does not establish this native parameter coordinate frame. All outputs remain approximate diagnostics, not Table 5 accuracy.
- **ASSUMPTION:** compare all 229 TRAIN images / 849 GT words post-hoc with the existing geometry/wordness definitions. The previous report and native mask shards are hash-verified before reuse. Optional GPU PARSeq is run manually on PAPER on new crops only, reusing source-report OCR for identical image/bbox; no new segmentation is permitted in this diagnostic.

### L-003 — Connectivity, contour hierarchy, and ordering ties are unspecified

- **Status:** `UNRESOLVED`.
- **Repository evidence:** OpenCV `RETR_EXTERNAL` and `CHAIN_APPROX_SIMPLE`; Python stable sort over returned contour order.
- **Missing detail:** 4/8 connectivity, hole treatment, touching components, and deterministic tie-breaking.
- **Scientific impact:** Can change blocks on dense text scenes.
- **Required evidence:** Original implementation or a documented assumption with sensitivity tests.

### L-004 — Multiple crops do not map unambiguously to singular `T3`

- **Status:** `UNRESOLVED`, `BLOCKING` for recognition.
- **Paper evidence:** Up to 10 crops are returned, while the scorer defines one `T3`.
- **Repository evidence:** Score every crop and retain the largest paired `C3`.
- **Missing detail:** First/largest crop, per-crop outputs, concatenation, best recognizer confidence, best context score, or per-instance evaluation.
- **Scientific impact:** Directly changes selected text and reported accuracy.
- **Required evidence:** Original inference loop and output schema.

### L-005 — Bounding-box maximum-coordinate convention is ambiguous

- **Status:** `UNRESOLVED`, `BLOCKING` for exact crop reproduction.
- **Paper evidence:** Section 3.3 defines extrema named `xmax`/`ymax`, clamps them to `W`/`H`, and then uses Python-style half-open crop notation. It does not state whether one is added to inclusive foreground extrema.
- **M7 APPROXIMATE decision:** Use integer `xyxy` boxes with exclusive maxima, consistent with OpenCV connected-component width/height and NumPy slicing.
- **Scientific impact:** Inclusive versus exclusive maxima can change the rightmost/bottommost crop pixels and padding extent.
- **Required evidence:** Original coordinate extraction and crop code.

## 5. BLIP-2 captioning

### C-001 — BLIP-2 checkpoint and claimed backbone do not align

- **Status:** `UNRESOLVED`, `BLOCKING` for strict captions and VRAM planning.
- **Paper evidence:** Pretrained BLIP-2; implementation details say a “ViT-B backbone.”
- **Repository evidence:** `Salesforce/blip2-flan-t5-xl`, unpinned, loaded fully in FP32.
- **Conflict:** The paper does not name FLAN-T5-XL, and the repository selection is not evidence for the paper’s ViT-B statement.
- **Scientific impact:** Caption text, latency, model size, and 8 GB feasibility all depend on checkpoint identity.
- **Required evidence:** Exact model identifier, immutable revision/hash, and architecture/config from the authors.
- **M5 hardware feasibility (2026-08-18):** The pinned approximate candidate in FP16,
  `single_device`, and without quantization OOMed while moving the model to the RTX
  3070. Peak CUDA allocated/reserved memory was 7,171,370,496/7,656,701,952 bytes;
  inference did not begin. Unquantized FP16 Accelerate CPU offload subsequently
  succeeded: 6,938,498,560/7,220,494,336 bytes peak allocated/reserved GPU memory,
  10,865,696,768 bytes peak CPU RSS, 14.12 s load, 51.65 s generation, and 65.78 s
  total latency, with 17 CUDA and 17 CPU modules. This qualified approximate path is a
  `HARDWARE_DEVIATION`, not evidence resolving this entry.
- **M11 RTX-3090 feasibility (2026-09-03):** The same pinned candidate and repository
  protocol completed unquantized FP16 `single_device` inference on ICDAR13 `img_1.jpg`.
  It used 9,035,619,840/9,726,590,976 bytes peak allocated/reserved GPU memory and
  22,351,937,536 bytes peak CPU RSS; load/generation/total times were
  9.90/3.13/13.14 seconds. Only 15,227,813,888 of 25,326,911,488 visible GPU bytes were
  free before the profile, but inference still completed without OOM. The output's
  repeated phrasing is retained rather than post-processed. This is a 24 GB
  `HARDWARE_DEVIATION` and does not resolve the paper's checkpoint, backbone, precision,
  prompt, decoding, or latency protocol.

#### C-001-LAVIS-VISUAL-001 — Lightweight caption candidates (2026-09-14)

TRAIN routing follow-up: `configs/approximate/caption_routing_five_models_train_v1.json`
freezes five candidate captioners plus the Flan baseline in two profiles
(six model families, seven profiles), with unchanged 1,203 practical-v1 crop transcriptions and a shared T1
per image. `diagnostics/caption_routing_train.py` first freezes all captions,
reading order and decisions, then opens TRAIN GT for evaluation. No TEST inputs
are used. Florence's separate OCR is not appended to T2. The large Florence
checkpoint is now explicitly pinned to revision
`4a12a2b54b7016a48a22037fbd62da90cd566f2a`, safetensors SHA256
`8b4e610c952eef90a836c56cda0f398a672a3a6ca7b4d96b0e09a86dee42e2c3`.
Classification `ASSUMPTION`: independent original-block gates versus joined T3
in the existing geometric reading order; empty selected strings always request
fallback. `PAPER_SPECIFIED`: alpha=.6, beta=.4, tau=.8 and the comparison of
T1/T3 to T2. `REPOSITORY_SPECIFIED`: max-paired-C control kept separate from
the semantic-winner paired score. Existing local-caption neighbor-length matching
is an `ASSUMPTION`, reported separately from complete-caption matching.
InstructBLIP keeps its previously tested prompt and decoding, pinned to
`bf9bd58cb3bb06e88c03b80f9cc346b90094bc6d`; both safetensors shard hashes are
recorded in the configuration and verified before loading.
`HARDWARE_DEVIATION`: sequential FP32 LAVIS/Florence, FP16 Flan/InstructBLIP, CPU FP32 MPNet;
the new Florence-large precision differs from the earlier FP16 visual run.
Reported fallback counts are requests, not executed DeepSolo calls: qualified
TRAIN DeepSolo results are absent from the supplied inputs. Post-fallback and
Table 5 accuracy therefore remain null. Per-image casefold word-multiset metrics
and spatial IoU>=.5 crop exact metrics are explicitly distinct diagnostics.

- **Classification:** `ASSUMPTION`; experiment mode `APPROXIMATE`, result class
  `PRACTICAL_APPROXIMATION` / five-image visual diagnostic, not recognition accuracy.
- **Scope:** Only frozen ICDAR13 image IDs 1,25,26,27,28. No segmentation,
  localization, PARSeq, routing, model fitting, or full TRAIN execution.
- **Official evidence:** LAVIS `blip_caption/base_coco` config specifies BLIP
  ViT-Base (not BLIP-2); `blip2/pretrain_vitL` specifies CLIP ViT-L + Q-Former.
  Neither resolves the paper's claimed BLIP-2 ViT-B identity. Sources:
  https://github.com/salesforce/LAVIS/blob/main/lavis/configs/models/blip_caption_base_coco.yaml
  and https://github.com/salesforce/LAVIS/blob/main/lavis/configs/models/blip2/blip2_pretrain_vitL.yaml.
- **Reproducible implementation:** Official `salesforce-lavis==1.0.2` wheel
  SHA256 `489782efc2d98d8e878112b2758f82ac2e6c10dfcfa087db6656465c7784d49f`;
  dedicated Python 3.10 / Torch 1.13 / Transformers 4.26 environment avoids
  changing the modern main pipeline. Package compatibility still requires actual
  import/model qualification on the host. Official checkpoint URLs are verified
  against installed configs, downloaded atomically, and sealed by SHA256.
- **Generation assumption:** Native deterministic 3-beam caption generation,
  min_length=10/max_length=80; no instruction prompt or manual text edits.
  This diagnostic does not claim to reproduce Fig.6 caption lengths/decoding.
  Florence-base uses the frozen detailed-caption and separate OCR tasks with
  512 max new tokens, matching the historical large diagnostic's task settings.
  Separate OCR must not be silently incorporated into T2.
- **Composite checkpoint audit (2026-09-14):** The official LAVIS 1.0.2
  `blip2/pretrain_vitL` Q-Former checkpoint intentionally omits the frozen
  `visual_encoder` initialized separately by `models/clip_vit.py` from
  `https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/clip_vit_L.pth`.
  Rejecting that omission as an incomplete generation checkpoint was an audit bug.
  The diagnostic now verifies every used visual tensor against that separate
  checkpoint (including documented upstream FP16 rounding), shapes, frozen flags,
  and SHA256. The pinned implementation uses transformer blocks 0..20;
  checkpoint blocks 21/22 are explicitly recorded as unused upstream keys.
  Other missing/mismatched generation tensors still fail closed.
  Classification: `ASSUMPTION` (official composite-loader audit, not a paper
  checkpoint identity resolution). An explicit runtime-repair option preserves
  the original protocol and all successful outputs byte-for-byte, archives the
  new exact runtime source, and records its SHA in newly executed results.
- **Hardware deviation:** Sequential FP32 inference. CPU timings include
  preprocessing/generation/decode; historical Florence-large GPU timings exclude
  preprocessing and cannot establish relative speed. New reports record dtype,
  parameter counts, checkpoint bytes and any requested GPU allocator peaks.
- **Unresolved legacy provenance:** Historical Florence-large report lacks image
  hashes, pinned revision and measured parameter count. Preserve it verbatim;
  snapshot its SHA256 and disclose these limitations when comparing.
- **Artifacts/runner:** `diagnostics/lavis_captioner_candidates.py`, host helper
  `diagnostics/run_lavis_captioner_candidates_host.sh`; persistent outputs under
  `artifacts/diagnostics/lavis-captioner-candidates-20260914/`.
- **Decision:** Hold full TRAIN until all three candidates produce valid outputs
  and receive visual review against the frozen RGB images. Missing results or
  a download failure are not evidence of poor model quality.
- **Host packaging repair:** Official decord 0.6.0 Linux wheel filename declares
  `py3-none-manylinux2010_x86_64`, but internal WHEEL declares `cp36-cp36m`.
  PyPI wheel SHA256 is
  `51997f20be8958e23b7c4061ba45d0efcd86bffd5fe81c695d0befee0d442976`.
  The native library uses ctypes, has x86-64 ELF architecture and no CPython
  PyInit entry point; CPU import succeeds in Python 3.10. `ASSUMPTION`: rebuild
  only WHEEL/RECORD to match the official filename, retaining hashes of every
  runtime payload member and requiring the ordinary dependency check to pass.
  This does not modify any model/checkpoint/inference parameter. Implementation:
  `diagnostics/repair_decord_wheel.py`; original/repaired wheels and receipt are
  cached persistently. Source: https://pypi.org/pypi/decord/0.6.0/json.
- **Avoid duplicate modern runtime:** Florence reuses the qualified existing main
  py311 environment; any missing einops dependency is placed in an isolated
  pure-Python target directory selected only for Florence process execution.
  Main packages and the main frozen dependency snapshot are not changed.
  A runtime-only protocol update archives empty failed preflights. Existing
  generated captions cause an explicit stop rather than being discarded.

### C-002 — Caption length is described as both tokens and words

- **Status:** `UNRESOLVED`.
- **Paper evidence:** Section 3.4 says fixed token constraints and gives `min_length/max_length`; Figure 4 says medium descriptions are limited to 80 words.
- **Repository evidence:** Uses `min_new_tokens=40`, `max_new_tokens=80`.
- **Conflict:** Words, total sequence tokens, and newly generated tokens are different controls.
- **Scientific impact:** Caption verbosity changes both semantic and lexical scores.
- **Required evidence:** Exact generation call and tokenizer-level interpretation.

### C-003 — Prompt is incomplete versus repository shorthand

- **Status:** `UNRESOLVED`.
- **Paper evidence:** Figure 4 shows a bullet prompt covering environment, text-bearing objects, placement/arrangement, and contextual cues, ending with an ellipsis. Prose also mentions occlusions.
- **Repository evidence:** A shorter one-sentence prompt.
- **Missing detail:** Whether Figure 4 is verbatim and whether omitted bullets exist.
- **Scientific impact:** Prompt wording directly changes `T2` and routing.
- **Required evidence:** Exact prompt string used for reported experiments.

### C-004 — Decoding parameters are not reported

- **Status:** `UNRESOLVED`, `BLOCKING` for deterministic `T2`.
- **Paper evidence:** Refers to fixed token/temperature constraints but supplies no temperatures.
- **Repository evidence:** Three-beam generation, length penalty 1.0, no explicit sampling/temperature.
- **Missing detail:** Temperature, sampling flag, beams, top-p/top-k, repetition penalty, seed, stopping, and generation library/version.
- **Scientific impact:** Captions and every downstream score may change.
- **Required evidence:** Exact generation config and model/library revision.

### C-005 — Reported 120 ms caption latency lacks a protocol

- **Status:** `UNRESOLVED`.
- **Missing detail:** Precision, warmup, synchronization, image size, included preprocessing, number of trials, and GPU identity correction.
- **Scientific impact:** Cannot reproduce the efficiency claim or compare RTX 3070 adaptations.
- **Required evidence:** Timing procedure and raw measurements.

## 6. PARSeq-S recognition

### R-001 — Exact PARSeq-S artifact is unidentified

- **Status:** `UNRESOLVED`, `BLOCKING` for Tables 5–7.
- **Paper evidence:** Evaluations use the small variant PARSeq-S.
- **Repository evidence:** A `baudm/parseq` generic image-classification stub; README defaults to TrOCR.
- **Qualified approximation evidence (2026-08-18):** The unchanged official v1.0.0
  `parseq-bb5792a6.pt` checkpoint (SHA256
  `bb5792a68e367476abca029cbf8699abc805f3d3dc7e57aae45c8ec4f7b7cd00`) loads
  strictly under official runtime commit `1902db043c029a7e03a3818c616c06600af574be`
  with PyTorch 2.4.1+cu124. This pairing is `APPROXIMATE` and an
  `ENGINEERING_CHOICE`; it is not evidence of the paper's selection.
- **Missing detail:** Official checkpoint name/hash, code revision, pretraining corpus, charset, case/punctuation mode, image transform, maximum length, and decoding configuration.
- **Scientific impact:** Recognition strings and all context/fallback results vary.
- **Required evidence:** Immutable official PARSeq artifact and exact invocation used by the authors.

### R-002 — Current repository adapter is not PARSeq sequence decoding

- **Status:** Historical defect fixed for the `APPROXIMATE` branch on 2026-08-18;
  the strict artifact-selection blocker remains under `R-001`.
- **Repository evidence:** Loads `AutoModelForImageClassification`, takes `argmax`, and returns `id2label`; comments call it a stub.
- **Conflict:** PARSeq produces a character sequence with EOS/tokenizer decoding, not one classification label.
- **Scientific impact:** No result from this adapter can be labeled PARSeq-S.
- **Qualification result:** The approximate candidate preserves the 94-character
  charset, EOS/BOS/PAD IDs, 32x128 bicubic/0.5 transform, AR greedy decoding, one
  refinement iteration, and `1x26x95` output contract. All six official example images
  matched the v1.0.0 decoded strings. This qualifies a future approximate adapter but
  does not repair the repository stub or resolve strict selection.
- **M5 implementation:** `recognition/parseq.py` now validates the qualified runtime
  revision and checkpoint SHA256, loads all tensors strictly, applies the official
  transform, performs real sequence forward/decode, and preserves token/confidence
  metadata. A CPU smoke decoded an official example successfully. This is not evidence
  that the pair is the paper artifact.

### R-003 — Full-image `T1` semantics are unclear for a single-instance recognizer

- **Status:** `UNRESOLVED`, `BLOCKING`.
- **Paper evidence:** `T1` is PARSeq-S output on the source image, potentially containing multiple words.
- **Missing detail:** Resizing, which word is expected, whether text is concatenated, and how PARSeq’s fixed-length single crop maps to full-scene output.
- **Scientific impact:** `T1`, scoring, and accuracy are not well-defined.
- **Required evidence:** Example-level author outputs or original preprocessing/inference code.

## 7. Context scoring

### S-001 — Figure 2, prose, and Algorithm 2 define different aggregation

- **Status:** `UNRESOLVED`, `BLOCKING` for Tables 5–7.
- **Figure 2:** Defines `S=max(S1,S3)` and `L=max(L1,L3)`, then `C=alpha*S+beta*L`. Maxima may come from different candidates.
- **Prose:** Selects `Tfinal` by higher semantic similarity, then describes confidence as a weighted combination of “its” semantic and lexical scores, suggesting candidate-paired scores.
- **Algorithm 2:** Computes `S1`, `S3`, then uses undefined `T` in `fuzz_ratio(T,T2)` and undefined `S` in `C=0.6S+0.4L`; it selects output by semantic score.
- **Repository:** Author commit `87b90dbb51f4e0462a5403040fad06985a474a8b`
  computes `C1` and `C3`; runner commit
  `2e8b544ef2dd559a46e2ac9e0a0f205eee328e9d` scores every crop, keeps the first
  maximum `C3`, thresholds `max(C1,C3)`, then independently returns the semantic winner.
- **Scientific impact:** Different candidates can trigger confidence and be returned; fallback counts and accuracy can change substantially.
- **Required evidence:** Original scoring code or author clarification with worked examples.

### S-002 — Fuzzy comparison implementation and preprocessing are unknown

- **Status:** `UNRESOLVED`, `BLOCKING`.
- **Paper evidence:** “Levenshtein fuzzy string matching ratio,” divided by 100 in Figure 2.
- **Repository evidence:** RapidFuzz `fuzz.ratio` on raw whole strings.
- **Missing detail:** Library/version, Levenshtein versus normalized Indel similarity, `ratio` versus `partial_ratio` or token variants, case folding, punctuation, Unicode normalization, whitespace, token extraction, and empty-input behavior.
- **Scientific impact:** The lexical term contributes 40% and controls `tau=0.8` routing.
- **Required evidence:** Exact function call and preprocessing pipeline.

### S-003 — Whole-word versus long-caption fuzzy scores appear incompatible with the threshold

- **Status:** `UNRESOLVED`; investigative warning, not a resolved contradiction.
- **Evidence:** For `C=0.6S+0.4L` with cosine `S≤1`, reaching 0.8 requires `L≥0.5` even when `S=1`. Whole-string similarity between a short PARSeq word and a 40–80-token caption is normally much lower.
- **Paper result:** Reports 54% context routing on ICDAR15 and 73% on ICDAR13.
- **Possible explanations:** Unreported partial matching, token extraction, longer recognizer strings, different scaling, or different score aggregation. None is established.
- **Scientific impact:** Literal repository scoring may route almost everything to DeepSolo and fail Tables 5–7.
- **Required evidence:** Raw `T1/T2/T3/S/L/C` examples from reported runs or original scorer code.

### S-004 — Exact MPNet checkpoint and embedding behavior are missing

- **Status:** `UNRESOLVED`, `BLOCKING`.
- **Paper evidence:** MPNet embeddings with cosine similarity.
- **Repository evidence:** `sentence-transformers/all-mpnet-base-v2`, normalized embeddings, unpinned revision.
- **Missing detail:** Checkpoint, pooling, token limit, normalization, truncation, framework, precision, and revision.
- **Scientific impact:** Cosine values close to the decision threshold can change routing.
- **Required evidence:** Exact model identifier/hash and embedding invocation.

### S-005 — Tie and empty-input behavior are unspecified

- **Status:** `UNRESOLVED`.
- **Repository evidence:** Semantic ties choose `T1`; empty strings are still embedded and fuzz-scored.
- **Missing detail:** Paper defines no ties, failed recognition, no crops, empty caption, NaN, or all-negative similarity behavior.
- **Scientific impact:** Failure cases affect denominators, fallback counts, and reproducibility.
- **Required evidence:** Original error/edge-case policy or an explicit future `ASSUMPTION` with tests.
- **M5 approximate assumption:** exact semantic and paired-confidence ties choose `T1`;
  missing text is represented as an empty string; `C == tau` does not request fallback.
  These rules are tested and remain forbidden in `STRICT` mode without evidence.

## 8. DeepSolo fallback

### D-001 — Exact DeepSolo model and checkpoint are unknown

- **Status:** `UNRESOLVED`, `BLOCKING` for Table 5.
- **Paper evidence:** Pretrained DeepSolo fallback; Tables 4–5 report ResNet-50 and ViTAEv2-S variants.
- **Missing detail:** Dataset-specific config/weights, training-data variant, revision, thresholds, lexicon, vocabulary, input resize, and postprocessing.
- **Repository evidence:** The original stub contained only MMOCR-like pseudocode. M6
  replaces scientific use of that hook with an isolated subprocess contract; this does
  not identify the paper checkpoint.
- **Scientific impact:** DeepSolo handles 27–46% of reported benchmark images and substantially determines final accuracy.
- **Required evidence:** Exact config/checkpoint pair per benchmark and immutable source revision.
- **M6 status:** The ResNet-50 Total-Text candidate using
  `finetune_150k_tt_mlt_13_15.yaml` is runtime-qualified for APPROXIMATE use at config
  SHA256 `0da913...702d` and checkpoint SHA256 `ac2a01...da68`. Strict selection remains
  blocked because the paper does not identify these bytes or its per-benchmark mapping.

### D-002 — Multiple DeepSolo outputs cannot map directly to singular `Tfinal`

- **Status:** `UNRESOLVED`, `BLOCKING`.
- **Paper evidence:** Algorithm 2 assigns `Tfinal ← DeepSolo(I)` as though it were one string.
- **Model behavior:** DeepSolo is an end-to-end spotter capable of multiple detected text instances.
- **Missing detail:** Ordering, filtering, concatenation, per-instance evaluation, confidence threshold, and treatment of geometry.
- **Scientific impact:** Recognition accuracy cannot be defined without the mapping.
- **Required evidence:** Original output serialization and evaluator.
- **M6 approximate assumptions:** `highest_score` (source order wins exact score ties),
  `source_order_space_join` (single spaces), and `reading_order_space_join` (minimum
  boundary y then x, source order ties) are separately named. None is paper-exact or
  a STRICT default; raw outputs, scores, and geometry are always retained.

### D-003 — DeepSolo dependency integration is incompatible with the main scaffold

- **Status:** Process-boundary design and one approximate runtime are qualified in M6;
  scientific model choice still must match D-001.
- **Evidence:** Official legacy Detectron2 stack conflicts with the modern root requirements;
  the original stub incorrectly suggested a generic MMOCR text-recognition call.
- **Scientific impact:** A forced single environment risks changing versions or model behavior.
- **M6 decision:** A versioned JSON request/response boundary is an `ASSUMPTION` and
  engineering isolation choice. Errors and tracebacks propagate; Detectron2/DeepSolo
  are never imported by `context-str-main`. A Docker-isolated Python 3.8,
  PyTorch 1.9.0+cu111, Detectron2 0.6 runtime and `adet._C` sm_86 build passed on the
  physical host. Separate-process execution remains a `HARDWARE_DEVIATION`, not paper
  evidence. The final repository boundary smoke passed on the physical host. The worker
  avoids the frozen upstream `adet.data` import-time argument parser and implements its
  ViTAEv2 divisor-32 bottom/right padding equivalently with Detectron2 `PadTransform`;
  this is an engineering isolation adaptation, not a scientific preprocessing change.
  See `docs/DEEPSOLO_RUNTIME.md`.

## 9. Datasets and splits

### DS-001 — COCO-TS counts do not match the named dataset

- **Status:** `UNRESOLVED`, `BLOCKING` for the COCO-TS Table 3 result.
- **Paper evidence:** Table 1 gives 43,686 training and 10,000 test images, totaling 53,686.
- **Known dataset evidence:** COCO-TS is commonly described as 14,690 pixel-level supervisions derived from COCO-Text.
- **Possible explanation:** The paper may use broader COCO-Text images, generated masks, a different split/version, or incorrect counts. None is established.
- **Scientific impact:** Training data quantity and mask provenance differ by tens of thousands of images.
- **Required evidence:** Exact image IDs, mask files/generation code, split manifests, and checksums used by the authors.

### DS-002 — ICDAR13 recognition uses 229 where the paper lists 233 test images

- **Status:** `UNRESOLVED`, `BLOCKING` for the ICDAR13 Table 5 claim.
- **Paper evidence:** Table 1 lists 229 training and 233 test images. Section 4.2 says 168/229 images avoid DeepSolo on ICDAR13.
- **Conflict:** 229 is the stated training count, not test count.
- **Possible explanation:** Typo, use of the training split, filtered test subset, or different task granularity. None is established.
- **Scientific impact:** Potential split leakage or incompatible denominator.
- **Required evidence:** Exact evaluation image manifest and per-image outputs.

### DS-003 — Don’t Care loss behavior clarified; conversion mechanics partly unresolved

- **Status:** `PARTIALLY_EVIDENCED`.
- **Missing detail:** Pixel-mask version, polygon rasterization, text versus non-text labels, illegible/don’t-care regions, overlapping polygons, multilingual regions, and TotalText/ICDAR annotation revisions.
- **Scientific impact:** Affects segmentation labels and recognition denominators.
- **Required evidence:** Conversion scripts, official archive versions, and checksummed manifests.
- **Author clarification:** Don’t Care pixels receive zero BCE weight. The official Task 2.2 TXT/BMP color mapping identifies those pixels. Validity-mask resize/augmentation interpolation remains an explicit runnable `ASSUMPTION`.

### DS-004 — Additional benchmark acquisition and granularity boundary

- **Status:** `UNRESOLVED`, non-blocking for dataset ingestion and blocking for new strict recognition claims.
- **Paper evidence:** ICDAR2015 and Total-Text are named recognition benchmarks; Total-Text is also named for segmentation. CTW1500 is not named by the paper.
- **Implemented boundary:** `recognition.scene_text_datasets` preserves official word quadrilaterals for ICDAR2015, variable/fixed word polygons for Total-Text, and text-line polygons for CTW1500. CTW1500 is classified `ASSUMPTION` / out-of-paper practical extension. Dataset manifests freeze bytes but explicitly forbid a Table 5 claim.
- **Still missing:** downloaded archive hashes and exact revision identity; physical validation of every official annotation; the paper's recognition matching, normalization, and denominator; exact Total-Text mask-conversion protocol.
- **Scientific impact:** treating CTW1500 line annotations as words, or silently selecting a different Total-Text annotation revision, would change both segmentation targets and recognition denominators.
- **Required evidence:** official archive bytes, SHA256 manifests, and an authoritative evaluation protocol resolving E-001.

## 10. Metrics and reported results

### E-001 — Recognition “accuracy” is undefined

- **Status:** `UNRESOLVED`, `BLOCKING` for Tables 5–7.
- **Paper evidence:** Reports one accuracy percentage per dataset but supplies no formula.
- **Missing detail:** Image-level versus word-level versus spotting metric; exact match versus normalized edit distance; case/punctuation normalization; geometric matching; care regions; lexicons; multiple predictions; empty/failure handling.
- **Scientific impact:** Table 5 cannot be independently recomputed or compared to standard spotting baselines.
- **Required evidence:** Evaluator code and sample predictions/ground truth with expected scores.

### E-002 — Table 3 values conflict with the stated IoU/F1 relationship

- **Status:** `UNRESOLVED`, `BLOCKING` for exact Table 3 validation.
- **Paper definitions:** `IoU=|P∩G|/|P∪G|`; `F1=2|P∩G|/(|P|+|G|)`.
- **Reported examples:** COCO-TS Vanilla U-Net has fgIoU 67.38 and F1 62.60; DeepLabV3+ has fgIoU 72.07 and F1 64.10. Under the same binary prediction/ground-truth pair, F1 cannot be below IoU.
- **Possible explanation:** Column mislabeling, different averaging/subsets, metric implementation error, or transcription error. None is established.
- **Scientific impact:** Exact target validation may be mathematically impossible under the printed definitions.
- **Required evidence:** Metric implementation and raw confusion counts or per-image values.

### E-003 — Segmentation aggregation and edge cases are unspecified

- **Status:** `UNRESOLVED`.
- **Missing detail:** Global micro versus per-image macro aggregation, threshold, original versus resized resolution, empty foreground cases, ignored pixels, and averaging precision.
- **Scientific impact:** Reported percentages can shift without model changes.
- **Required evidence:** Evaluation code and output resolution.
- **Native-resolution diagnostic:** The author-protocol evaluator compares native
  `128x128` predictions with GT resized bilinearly to `128x128` and thresholded at `0.5`.
  This aligns the tensors with the clarified training/loss endpoint but is still an
  `ASSUMPTION` for Table 3 evaluation.
- **Don't Care blocker:** The evaluator implements paired-validity masking, but the frozen
  233-image test manifest/archive has no TXT validity annotations. Training colors cannot
  be reused as a lookup: 327 RGB values occur in both care and Don't Care training records.
  The current all-pixel test convention remains separately labeled and must not be
  described as author-confirmed Don't Care exclusion.
- **Diagnostic boundary:** Repeated evaluation of the 233-image official test set across
  training checkpoints is explicitly a leaky `ASSUMPTION` for investigation only. It may
  show the test trajectory, but it cannot select a checkpoint, tune the protocol, or support
  an unbiased Table 3 claim.

### E-004 — FLOP protocol is absent

- **Status:** `UNRESOLVED`, `BLOCKING` for Table 4.
- **Paper evidence:** Reports 35 G and 50 G for two conditional pipelines and claims roughly 60% reduction.
- **Missing detail:** Profiler, multiply-add convention, image/crop dimensions, number of crops, captioner/segmenter inclusion, expected versus worst-case fallback, and averaging over which data.
- **Scientific impact:** Conditional pipelines cannot have one reproducible FLOP value without workload/routing rules.
- **Required evidence:** Profiling code, input manifest, branch counts, and component-level totals.

### E-005 — False-positive and failure accounting are undefined

- **Status:** `UNRESOLVED`.
- **Paper evidence:** Tables 6–7 report false-positive counts and fallbacks.
- **Missing detail:** Definition of a false-positive recognition, relationship to incorrect confident predictions, skipped/failed samples, and whether fallback errors count.
- **Scientific impact:** Ablation tables cannot be reconstructed.
- **Required evidence:** Per-image decision records and aggregation code.

## 11. Ablation protocol

### AB-001 — Wrong-description sampling is unspecified

- **Status:** `UNRESOLVED`, `BLOCKING` for Table 7.
- **Paper evidence:** ICDAR15 is rerun with random irrelevant descriptions.
- **Missing detail:** Source pool, whether self-captions are excluded, permutation seed, fixed mapping, replacement, and handling of accidental relevance.
- **Scientific impact:** Semantic scores and false positives depend heavily on the random mapping.
- **Required evidence:** Exact description-to-image mapping and seed.

### AB-002 — No-description representation is unspecified

- **Status:** `UNRESOLVED`.
- **Missing detail:** Empty string, omitted semantic term, zero vector, recognizer-only score, or another baseline.
- **Repository behavior:** Empty strings are passed through MPNet and fuzzy scoring.
- **Scientific impact:** These choices are not equivalent and directly affect Table 7.
- **Required evidence:** Original ablation code.

### AB-003 — Detector-assisted comparison has no reproducible protocol

- **Status:** `UNRESOLVED`.
- **Paper evidence:** Figure 7 claims recognition changes marginally when comparing traditional detector boxes to block localization.
- **Missing detail:** Detector identity/checkpoint, datasets, thresholds, matching, and numerical results.
- **Scientific impact:** The “skipping detection” claim cannot be quantitatively reproduced.
- **Required evidence:** Detector configuration and raw comparison table.

### AB-004 — Ablation seeds, repetitions, and uncertainty are absent

- **Status:** `UNRESOLVED`.
- **Missing detail:** Whether captions are regenerated, stochastic decoding, random-description seeds, repeat count, and variance/confidence intervals.
- **Scientific impact:** Small reported accuracy differences may not be stable.
- **Required evidence:** Run logs or a documented future repeated-measures protocol labeled as an assumption if author evidence remains unavailable.

## 12. Repository-level blockers that preserve prior assumptions

These are not paper ambiguities, but they can silently contaminate future reproduction if not tracked:

1. `init.py` is used instead of `__init__.py`; `from pipeline import set_global_seed` cannot obtain the function from `pipeline/init.py`.
2. `PipelineRunner._get` does not correctly recognize an OmegaConf `DictConfig`, causing silent hard-coded defaults.
3. Required `configs/` files are absent; README asks the user to create them.
4. Scientific paths can silently fall back to TrOCR, dummy recognition, empty strings, or disabled DeepSolo.
5. The CLI reads optional ground truth but never evaluates it.
6. `strict=False` checkpoint loading can hide incompatible AG-UNet states.
7. Hugging Face revisions are unpinned.
8. There are no dataset/checkpoint manifests, tests, training code, evaluators, or dependency locks.

These must be addressed in small tested milestones, but repairing them must not be mistaken for resolving the scientific ambiguities above.

## 13. Provenance cross-reference after evidence freeze

Macro-Milestone 2 froze external evidence in
[`SOURCES.yaml`](../provenance/SOURCES.yaml) and
[`ARTIFACTS.yaml`](../provenance/ARTIFACTS.yaml). The table below links every ambiguity to
the most relevant registry entry. A registry entry that is `RESOLVED` may resolve only an
upstream source or a conflicting count; it does not resolve the paper's implementation choice.

**No ambiguity status above is upgraded to `RESOLVED` by this freeze.** New primary evidence
narrows `C-001`, `R-001`, `S-004`, `D-001`, `D-003`, `DS-001`, and `DS-002`, but each still
lacks the author-to-artifact/protocol link needed for a strict claim.

| Ambiguity | Corresponding provenance entries | Freeze outcome |
|---|---|---|
| A-001 | `ART-AGUNET-AUTHOR-PROTOCOL-V1` | Author confirms scalar broadcast attention and preserved skip width; resolved |
| A-002 | `ART-AGUNET-AUTHOR-PROTOCOL-V1` | 128x128/64-channel endpoint resolved; decoder operations remain blocking |
| A-003 | `ART-AGUNET-AUTHOR-PROTOCOL-V1` | Compression required; recalled 1024 width remains unconfirmed/blocking |
| A-004 | `ART-AGUNET-AUTHOR-PROTOCOL-V1` | Scalar equation resolved; intermediate widths/normalization remain blocking |
| A-005 | `ART-AGUNET-AUTHOR-PROTOCOL-V1` | 128x128 reported logits/loss endpoint resolved |
| A-006 | `ART-AGUNET-AUTHOR-PROTOCOL-V1`, `ART-RESNET50-IMAGENET1K-V2-CANDIDATE` | Frozen encoder/BN resolved; exact weights remain unknown |
| A-007 | `ART-RESNET50-IMAGENET1K-V2-CANDIDATE`, `ART-ICDAR13-FST-OFFICIAL-SPLIT` | No paper normalization/geometry evidence found |
| A-008 | `ART-AGUNET-AUTHOR-CODE`, `ART-ICDAR13-SEGMENTATION-EVALUATOR` | Threshold/postprocessing remain unspecified |
| T-001 | `ART-AGUNET-AUTHOR-PROTOCOL-V1` | 100 epochs/no validation/training-fgIoU selection resolved; remaining mechanics unknown |
| T-002 | `ART-ICDAR13-FST-OFFICIAL-SPLIT`, `ART-TOTALTEXT-OFFICIAL-SPLIT`, `ART-COCOTS-OFFICIAL-DATASET` | Dataset sources identified; mixing regime remains unknown |
| T-003 | `ART-AGUNET-AUTHOR-PROTOCOL-V1` | Flip p=0.5 resolved; rotation and transform details remain unknown |
| T-004 | `ART-PAPER-PDF-V2` | Paper hardware inconsistency remains internal to the paper |
| L-001 | `ART-AGUNET-AUTHOR-CODE` | Repository numbers remain secondary defaults |
| L-002 | `ART-AGUNET-AUTHOR-CODE` | Paper/repository area-definition mismatch remains |
| L-003 | `ART-PAPER-PDF-V2`, `ART-AGUNET-AUTHOR-CODE` | No connectivity/hierarchy/tie evidence found |
| L-004 | `ART-PAPER-PDF-V2`, `ART-PAPER-RECOGNITION-EVALUATOR` | Crop-to-`T3` aggregation remains undefined |
| C-001 | `ART-BLIP2-PAPER-VITB-SELECTION`, `ART-BLIP2-LAVIS-FLANT5XL-CANDIDATE`, `ART-BLIP2-HF-FLANT5XL-REPOSITORY-CANDIDATE` | Narrowed: official Flan-T5-XL is EVA-CLIP-g, not identifiable ViT-B; still blocking |
| C-002 | `ART-BLIP2-PAPER-PROMPT-AND-DECODING`, `ART-BLIP2-HF-FLANT5XL-REPOSITORY-CANDIDATE` | Repository uses token limits; paper word/token conflict remains |
| C-003 | `ART-BLIP2-PAPER-PROMPT-AND-DECODING` | Exact repository prompt frozen; no paper evidence binds it |
| C-004 | `ART-BLIP2-PAPER-PROMPT-AND-DECODING`, `ART-BLIP2-LAVIS-FLANT5XL-CANDIDATE` | Upstream/repository defaults recovered; paper decoding remains unknown |
| C-005 | `ART-BLIP2-HF-FLANT5XL-REPOSITORY-CANDIDATE` | Artifact size strengthens the hardware concern; timing protocol remains unknown |
| R-001 | `ART-PARSEQ-PAPER-SELECTION`, `ART-PARSEQ-V100-PARSEQ-CANDIDATE`, `ART-PARSEQ-PYTORCH2-RUNTIME-CANDIDATE` | Official checkpoint/runtime pairing qualified for approximate main; paper selection remains blocking |
| R-002 | `ART-AGUNET-AUTHOR-CODE`, `ART-PARSEQ-V100-PARSEQ-CANDIDATE`, `ART-PARSEQ-PYTORCH2-RUNTIME-CANDIDATE` | Official sequence decoder and compatibility test confirm repository classifier adapter is not conformant |
| R-003 | `ART-PARSEQ-PAPER-SELECTION`, `ART-PAPER-RECOGNITION-EVALUATOR` | Full-image multi-word behavior remains undefined |
| S-001 | `ART-PAPER-RECOGNITION-EVALUATOR` | No authoritative scoring implementation found |
| S-002 | `ART-FUZZY-PAPER-PROTOCOL`, `ART-RAPIDFUZZ-RATIO-REPOSITORY-CANDIDATE` | Library candidate identified; paper preprocessing/protocol remains blocking |
| S-003 | `ART-FUZZY-PAPER-PROTOCOL`, `ART-BLIP2-PAPER-PROMPT-AND-DECODING` | Threshold-feasibility concern remains unresolved |
| S-004 | `ART-MPNET-PAPER-SELECTION`, `ART-ALL-MPNET-BASE-V2-REPOSITORY-CANDIDATE` | Repository model fully pin-able; paper selection remains blocking |
| S-005 | `ART-FUZZY-PAPER-PROTOCOL`, `ART-PAPER-RECOGNITION-EVALUATOR` | Tie/empty-input behavior remains unspecified |
| D-001 | `ART-DEEPSOLO-PAPER-SELECTION`, all `ART-DEEPSOLO-*-CANDIDATE` entries | Official candidates per backbone/benchmark found; author selection remains blocking |
| D-002 | `ART-DEEPSOLO-PAPER-SELECTION`, `ART-PAPER-RECOGNITION-EVALUATOR` | Multi-output reduction remains undefined |
| D-003 | `ART-DEEPSOLO-RUNTIME-CANDIDATE`, `SRC-DETECTRON2-V06` | Official legacy stack/revision recovered; integration and paper revision remain unresolved |
| DS-001 | `ART-COCOTS-OFFICIAL-DATASET`, `ART-COCOTEXT-COUNT-ORIGIN` | Narrowed: printed counts are COCO-Text counts; experiment mapping remains blocking |
| DS-002 | `ART-ICDAR13-FST-OFFICIAL-SPLIT`, `ART-PAPER-RECOGNITION-EVALUATOR` | Official 229/233 split confirmed; paper's 229 recognition denominator remains blocking |
| DS-003 | `ART-ICDAR13-AUTHOR-PROTOCOL-MASKED-TARGET` and official evaluator entries | Zero-weight Don’t Care behavior resolved; exact validity transforms remain open |
| E-001 | `ART-PAPER-RECOGNITION-EVALUATOR`, `ART-ICDAR15-OFFICIAL-EVALUATOR`, `ART-TOTALTEXT-DETEVAL` | Official metrics recovered but none matches the undefined paper scalar by evidence |
| E-002 | `ART-ICDAR13-SEGMENTATION-EVALUATOR`, `ART-PAPER-PDF-V2` | No raw metric implementation/counts found; contradiction remains blocking |
| E-003 | `ART-ICDAR13-SEGMENTATION-EVALUATOR` | Official protocol exists; paper aggregation/edge cases remain unknown |
| E-004 | `ART-PAPER-PDF-V2`, `ART-BLIP2-HF-FLANT5XL-REPOSITORY-CANDIDATE` | No FLOP profiler/workload evidence found |
| E-005 | `ART-PAPER-RECOGNITION-EVALUATOR` | Failure/false-positive accounting remains undefined |
| AB-001 | `ART-PAPER-RECOGNITION-EVALUATOR`, `ART-BLIP2-PAPER-PROMPT-AND-DECODING` | No wrong-description mapping/seed found |
| AB-002 | `ART-MPNET-PAPER-SELECTION`, `ART-FUZZY-PAPER-PROTOCOL` | No authoritative no-description representation found |
| AB-003 | `ART-PAPER-RECOGNITION-EVALUATOR` | No detector/checkpoint/protocol artifact found |
| AB-004 | `ART-BLIP2-PAPER-PROMPT-AND-DECODING`, `ART-PAPER-RECOGNITION-EVALUATOR` | No seeds/repeats/uncertainty evidence found |

## 14. Macro-Milestone 4 approximate segmentation decisions

These decisions make a synthetic segmentation path runnable; they do not resolve the
ambiguities above or support a Table 3 claim. `FIGURE_INFERRED` is recorded as an
evidence basis while the decision classification remains `ASSUMPTION`, consistent with
the classification vocabulary in `AGENTS.md`.

### M4-SEG-001 — Figure-derived AG-UNet reconstruction

- **Classification:** `ASSUMPTION`; evidence basis `FIGURE_INFERRED`, except the two-convolution decoder blocks and 1×1 head retained from the author repository, which are `REPOSITORY_SPECIFIED`.
- **Decision:** Separate `ApproximateAGUNet` with ResNet-50 feature stages, one 3×3 Conv-BN-ReLU `2048→512` compressed bottleneck, bilinear upsampling plus 1×1 projection, additive one-channel spatial attention, decoder outputs `512/256/128/64`, and 128×128 logits for a 256×256 input.
- **Alternatives considered:** Repairing the historical five-stage model; retaining its uncompressed 2048-channel bottleneck; producing 256×256 logits. These were rejected for this branch because they conflict more directly with Figure 3 or the stated 128×128 masks.
- **Scientific impact:** Architecture, capacity, memory, and checkpoint schema differ materially across alternatives. The selected model has 48,454,733 parameters with an unfrozen encoder.
- **Strict status:** A-001 through A-006 remain `UNRESOLVED`/blocking. Full decisions are in `AG_UNET_IMPLEMENTATION_DECISIONS.md`.

### M8-SEG-001 — Frozen approximate architecture conformance identity

- **Classification:** `ASSUMPTION` for the figure-derived topology, input-size boundary,
  initialization, and 128×128 logit endpoint; retained `REPOSITORY_SPECIFIED` choices for
  the decoder blocks, head, and `IMAGENET1K_V2` candidate. No architectural choice is
  promoted to paper evidence.
- **Decision:** Freeze the existing branch as `approximate-ag-unet-figure3-v1` with
  architecture-contract schema version 1, 48,454,733 total/trainable parameters for the
  selected unfrozen-encoder branch, 456 state tensors, and schema fingerprint
  `bc146d1e3742a63f9b64ad8aded6ff92fbde656f8386e1a88d0e8a109ab04b5f`.
  Supported inputs are positive floating-point BCHW tensors matching the model dtype/device,
  with three channels and each spatial dimension at least 32 and divisible by 32; logits
  are `H/2×W/2`. The selected qualification branch is float32.
- **Retained repository behavior:** The registered full torchvision ResNet-50 retains
  `avgpool` and `fc` parameters, but the encoder feature path does not execute them. They
  remain in the frozen count/state schema; removing them would create a new architecture
  identity rather than an M8 conformance repair.
- **Checkpoint policy:** Identity/schema metadata, exact keys, shapes, and dtypes are
  mandatory. Missing/unexpected/mismatched states and non-strict loading are rejected.
- **Scientific impact:** This makes the selected approximation auditable and prevents
  silent checkpoint/topology drift. It does not show that this architecture or state
  schema produced the paper results.
- **Physical status:** Batch-1 float32 forward profiling with the repository-selected
  `IMAGENET1K_V2` candidate passed on `cuda:0` / NVIDIA GeForce RTX 3070 with
  `[1,3,256,256]→[1,1,128,128]`, 264,071,680 peak allocated bytes, and 289,406,976 peak
  reserved bytes. The physical full suite passed 142/142. This is an engineering
  qualification of the approximate branch, not paper architecture evidence.
- **Strict status:** A-001, A-002, A-003, A-004, A-005, and A-006 all remain
  `UNRESOLVED`/blocking. In particular, the half-resolution logit contract records rather
  than resolves A-005, and the candidate weights remain repository evidence only.

### M4-SEG-002 — Approximate paired transforms

- **Classification:** `ASSUMPTION`.
- **Decision for synthetic smoke:** direct aspect-ratio stretch; flip then rotate then resize; flip probability 0.5; rotation probability 0.5 with uniform angle in [-10°, 10°]; bilinear antialiased RGB interpolation; nearest mask interpolation; zero fill; ImageNet mean/std normalization; deterministic seed 202503 with epoch/sample-derived streams.
- **Paper-specified subset:** 256×256 images, 128×128 masks, horizontal flips, and random rotations.
- **Alternatives considered:** letterboxing/cropping, other probabilities/ranges/order/fill/interpolation, and unnormalized [0,1] inputs. None is authoritatively selected.
- **Strict status:** A-007 and T-003 remain unresolved and block faithful preprocessing.

### M4-SEG-003 — Approximate smoke-training protocol

- **Classification:** `ASSUMPTION` for BCE-with-logits mean reduction, Adam betas (0.9, 0.999), epsilon 1e-8, one smoke epoch, no schedule, last-checkpoint selection, seed 202503, deterministic algorithms, and trainable encoder. Random encoder weights are used only to avoid an implicit download in the synthetic smoke.
- **Paper-specified subset:** Adam, learning rate 1e-4, weight decay 0.01, batch size 8, and BCE.
- **Checkpoint decision:** Versioned checkpoint payload, SHA-256 return value, resolved configuration, claim/result labels, and strict state-dict loading. No `strict=False` path exists in the new engine.
- **Strict status:** T-001/T-002/T-004 and A-006 remain unresolved. The earlier batch-size-8 full-model CUDA feasibility check passed on the physical RTX 3070; this establishes hardware feasibility only and supplies no training-protocol evidence.

### M4-SEG-004 — Algorithm 1 unspecified mechanics

- **Classification:** `ASSUMPTION` for 8-connectivity and deterministic equal-area ordering top-to-bottom then left-to-right. Synthetic-only `Amin=4` and `p=2` are `ASSUMPTION` values and are not real-data defaults.
- **Paper-specified subset:** component-area filter and descending sort, padded bounds, original-image crop, and `nmax=10`.
- **Repository value status:** Historical `Amin=150` and `p=6` remain `REPOSITORY_SPECIFIED`; they were not promoted or used by the synthetic smoke.
- **Strict status:** L-001/L-003 remain unresolved. L-002's paper-specified component-area behavior is implemented and tested in the new path.

### M7-LOC-001 — Algorithm 1 conformance boundary

- **Classification:** `PAPER_SPECIFIED` for connected-component extraction, true foreground-pixel component-area filtering and descending ordering, padding/clamping, original-image crops, and `nmax=10`.
- **Retained APPROXIMATE choices:** Callers must explicitly supply `Amin`, `p`, connectivity, and tie policy. The synthetic smoke retains `Amin=4`, `p=2`, 8-connectivity, and top-to-bottom then left-to-right equal-area ordering; component label is used only as a final deterministic fallback. M7 golden-test-only values are `Amin=3` and `p=1`, not experimental defaults.
- **Coordinate decision:** Exclusive `xyxy` maxima are an `ASSUMPTION`, now exposed in the machine-readable golden record.
- **Auditability:** The result record distinguishes true component area from boxes, records unpadded and padded/clamped boxes, and records both below-area and top-10 rejections before returning retained crops.
- **Strict status:** L-001, L-003, and L-005 remain unresolved; A-005/A-008 continue to block a paper-exact upstream binary mask. M7 is APPROXIMATE only.

### M4-SEG-005 — Threshold and metric aggregation

- **Classification:** Threshold 0.5 with no morphology is `REPOSITORY_SPECIFIED`; empty-foreground handling and aggregation are `ASSUMPTION` choices.
- **Decision:** Expose inclusive probability thresholding separately; resize the thresholded 128×128 smoke mask to original-image coordinates with nearest interpolation; implement and label both `per_image_macro` and `global_micro`; require one of `perfect_if_both_empty`, `zero_if_both_empty`, or `exclude_both_empty`; accept an explicit valid-pixel mask without inferring ignore labels.
- **Strict status:** A-008, E-002, E-003, and DS-003 remain unresolved. No metric mode is selected as the paper's Table 3 mode.

### M4-SEG-006 — Dataset manifest boundary

- **Classification:** `ASSUMPTION` for the generic binary-mask pair schema used by synthetic smoke.
- **Decision:** Ordered sample IDs, relative paths constrained to an explicit data root, SHA-256 verification, explicit reproduction mode/claim label, explicit annotation-policy metadata, RGB images, and rejection of non-binary masks.
- **Strict status:** The loader does not infer official splits, rasterize benchmark polygons, or invent care/ignore behavior. DS-001/DS-003 and missing immutable official manifests continue to block real strict training.

### M9-SEG-001 — Real dataset orchestration and validation

- **Classification:** `ASSUMPTION` for dataset-specific training and `icdar13-task2.2-sha256-ranked-holdout-v1`: seed 202503, SHA-256 ranking, 206/23 cardinalities, and metrics-only held-out validation from the official 229-image training source. Dataset identity remains upstream RRC evidence; the acquired keras-ocr mirror bytes are `SECONDARY` acquisition evidence only.
- **Decision:** Numeric canonical IDs are ranked by raw bytes of `SHA256(UTF8("202503:" + sample_id))` with numeric fallback; first 23 are validation, remaining 206 training, and both manifests serialize numerically. The policy fingerprint is `3a7c3d4226e735b2e4b57fef5f8c84fa9522ad1b60852d86229a975ff07d6124`. Per-file SHA-256, disjoint IDs/paths, complete union, relative paths, archive hashes, and byte-identical regeneration are frozen.
- **Alternatives considered:** Official test data as validation, implicit split generation, and best-validation checkpoint selection were rejected because the paper supplies no such protocol and they could leak evaluation data or invent selection semantics.
- **M9 training-mask decision:** Exact RGB `(255,255,255)` is background and every other RGB value is foreground, with no tolerance or remapping. This is upstream Task 2.2 semantics. Don't Care remains foreground and TXT is audit/provenance only: that training behavior is an `ASSUMPTION`; BCE has no ignore path. Policy fingerprint: `ca318685a8fea2a0c162d80986acb88d60ec950fdec440652ff4d57bf4c0adf3`.
- **Strict status:** The split is not paper evidence and T-001/T-002 remain blocking for STRICT. DS-003 remains unresolved for paper-exact Don't Care training behavior; this APPROXIMATE training policy defines no evaluation semantics.

### M9-SEG-002 — Multi-epoch, resume, checkpoint, and logging choices

- **Classification:** `ASSUMPTION` for two epochs in the supplied APPROXIMATE ICDAR13 path, seed 202503, Adam betas `(0.9,0.999)`/epsilon `1e-8`, no scheduler, deterministic algorithms, `num_workers=0`, `drop_last=false`, and last-checkpoint semantics. `PAPER_SPECIFIED` values remain Adam, learning rate `1e-4`, weight decay `0.01`, batch size 8, BCE, 256×256 images, and 128×128 masks.
- **Decision:** Epoch permutations are derived from seed plus epoch; paired augmentation is derived from seed, epoch, and sample index. Checkpoints restore model, optimizer, completed epoch/global step, and Python/NumPy/PyTorch CPU/all-visible-CUDA RNG state. Resume rejects architecture/schema/state, training-config, resolved-config, dataset-manifest, seed, validation-protocol, or qualification-contract drift. JSONL epoch logs and JSON run/failure manifests have no wall-clock fields.

- **Alternatives considered:** Scheduler, early stopping, best checkpoint, AMP, gradient accumulation, smaller batch, activation checkpointing, and alternate optimizers are absent from this branch rather than guessed.
- **Validation evidence:** Focused tests compare uninterrupted and resumed three-epoch model tensors, optimizer state, global step, per-epoch ordering hashes, and train/validation metrics exactly on CPU. Physical RTX 3070 M9 real-data qualification remains pending; no CUDA result is inferred from sandbox execution.
- **Strict status:** T-001 through T-004 and A-006/A-007 remain unresolved. The M9 trainer is an APPROXIMATE engineering path, not evidence that the paper training protocol or checkpoint has been reproduced.

### M10-SEG-001 — APPROXIMATE ICDAR13 Table 3 evaluation protocol

- **Classification:** The 0.5 threshold is `REPOSITORY_SPECIFIED`; GT/Don't Care treatment, backprojection ordering, aggregation, and empty handling are `ASSUMPTION` choices authorized for M10 only.
- **Decision:** Exact RGB white is background and every other RGB value, including Don't Care, is foreground; every pixel is valid. Stretch/Imagenet-preprocessed 256x256 RGB produces 128x128 logits; sigmoid is inclusively thresholded at 0.5 before nearest binary backprojection to original GT geometry. Primary metrics are arithmetic means of per-image fgIoU and F1 over all 233 samples; global micro is diagnostic. Both-empty scores are 1.0. Failures remain in the denominator/records and suppress aggregate claims.
- **Separation:** This evaluation policy is independently frozen and does not inherit the M9 training-mask policy. It does not implement the official atom-based ICDAR evaluator.
- **Strict status:** APPROXIMATE only. A-005, A-007, A-008, DS-001, DS-003, E-002, and E-003 remain unresolved, along with upstream architecture/training blockers.

### M10-SEG-002 — Author-protocol native-128 evaluation diagnostic

- **Classification:** `ASSUMPTION`; the paper states 128x128 training masks but does not
  specify Table 3 evaluation resolution or aggregation.
- **Decision:** Keep predictions at native `128x128`; bilinearly resize the binary GT to
  `128x128`, re-binarize at `0.5`, and emit both macro and micro metrics. No prediction
  backprojection is performed.
- **Don't Care boundary:** Nearest-resized validity masking is implemented and fail-fast
  when requested without annotations. The acquired test artifact has 233 RGBA masks but
  no paired TXT labels, so exact training-style Don't Care exclusion cannot yet run.
- **Strict status:** APPROXIMATE diagnostic only; it cannot resolve E-003 or DS-003.

### M10-SEG-003 — Task 1 care-box conservative Don't Care diagnostic

- **Classification:** `ASSUMPTION`; Task 1 TXT files identify care word boxes but contain
  no explicit Task 2 `#RGB` Don't Care records.
- **Decision:** A foreground color is invalid only when none of its pixels overlaps any
  Task 1 care box. If one or more pixels overlap, the complete color remains valid. The
  resulting validity mask is resized with nearest interpolation for native-128 metrics.
- **Evidence:** The supplied archive has 233 ordered TXT files, 1,095 valid boxes, complete
  ID and geometry agreement with the 233 evaluation images/masks, and SHA256
  `71a7d78d57822ec0376f3f73dd808e2d662f86fb1ef2f85e09a45f26e1417018`.
  On the analogous labeled-training check the zero-overlap rule recovered 589/608 known
  Don't Care colors with 100% precision and 96.875% recall; 19 ignored colors remained
  conservatively valid.
- **Strict status:** APPROXIMATE diagnostic only. It improves on treating all Don't Care
  pixels as foreground but does not resolve DS-003 or establish the author's test mapping.

### M11-DOWN-001 — Frozen binary-mask backprojection and device sensitivity

- **Classification:** `ASSUMPTION` for nearest-neighbor backprojection after native-logit
  thresholding; `HARDWARE_DEVIATION` for any non-CUDA diagnostic execution.
- **Decision:** Downstream localization receives the standard native `128x128` prediction,
  inclusively thresholded at `0.5`, then nearest-neighbor backprojected to the source-image
  geometry. Evaluation validity masks are never loaded or applied on this path.
- **Observed sensitivity:** On ICDAR13 `img_1.jpg`, a strict-load CPU diagnostic produced
  3,557 foreground pixels versus 3,558 in the frozen RTX-3090 CUDA prediction. The saved
  standard and validity-aware CUDA predictions were byte-identical. No threshold adjustment
  or post-hoc correction is permitted; scientific downstream execution must record device
  and mask hash.
- **Strict status:** The output endpoint in A-005 is resolved, but A-008 does not specify
  the paper's downstream threshold/backprojection procedure; this remains APPROXIMATE-only.

### M11-DEEPSOLO-HOST-001 — Non-Docker legacy runtime on `paper`

- **Classification:** `HARDWARE_DEVIATION` for replacing the qualified Docker process
  with a separate unprivileged host venv/subprocess; `ASSUMPTION` for CPython 3.8.10
  because the qualified Docker record established only the 3.8 major/minor version.
- **Evidence:** The repository implements a versioned JSON-file subprocess adapter. The
  host exposes `/usr/bin/python3.8`, CUDA Toolkit 11.1, and an RTX 3090 without requiring
  privileged changes, while Docker is administratively inaccessible. The installed venv,
  source archive, checkpoint, environment, runtime manifest, and smoke records were frozen
  by SHA-256 on 2026-09-02.
- **Decision:** Preserve DeepSolo commit `dbadae995035246bad3376c7a44c015c69e9b313`,
  torch/torchvision `1.9.0+cu111`/`0.10.0+cu111`, Detectron2 `0.6+cu111`, the qualified config
  and checkpoint hashes, and an isolated process. Never install these dependencies in
  `context-str-main`.
- **Validation result:** The exact environment audit, `adet._C` import, strict checkpoint
  compatibility, staged subprocess boundary, CUDA inference, three-detection smoke, memory
  metadata, and error propagation passed. The fixture decoded to `deepsolo test 2026` in
  0.147001147037372 seconds inference time with 869,216,256/1,270,874,112 bytes peak
  allocated/reserved CUDA memory. Preserving the venv Python symlink and checking the full
  Detectron2 local version are engineering corrections, not scientific substitutions.
- **Strict status:** QUALIFIED for APPROXIMATE use only. RTX 3090 execution, host-process
  isolation, and Python 3.8.10 remain deviations/assumptions; `D-001` through `D-003` and
  the paper's model/output ambiguities remain blocking for STRICT/Table 5 claims.

### M11-PARSEQ-ARCHIVE-001 — GitHub commit-archive source transport on `paper`

- **Classification:** `ASSUMPTION` for using a GitHub-generated commit archive rather
  than a clean Git checkout; upstream commit contents remain evidence for the qualified
  APPROXIMATE runtime only.
- **Evidence:** The codeload tarball for commit
  `1902db043c029a7e03a3818c616c06600af574be` has SHA256
  `f57289cb1fc7fe41f488a52ea48f8a7f114252ef405b0b64efc1d528c29b48ef` and contains
  111 regular files beneath the expected revision-qualified root.
- **Decision:** Freeze the archive hash and fail closed unless every extracted file and
  executable bit matches the tarball and no unexpected files, directories, or symlinks
  are present. Python bytecode caches are ignored because imports may create them.
- **Strict status:** This resolves source-transfer reproducibility for the selected
  APPROXIMATE candidate only. It does not resolve `R-001`, `R-003`, or the paper's
  unknown PARSeq checkpoint/runtime pairing.

### M11-MAIN-HOST-001 — Recreated main runtime on RTX 3090 host

- **Classification:** `HARDWARE_DEVIATION` for physical execution on an RTX 3090 rather
  than the primary RTX 3070 target; the project-local environment location is an
  engineering transport choice and does not change package identities.
- **Evidence:** On 2026-09-03 the environment matched all 70 entries in the frozen
  installed snapshot, `uv pip check` passed, CUDA 12.4/cuDNN 90100 initialized, and
  CUDA tensor operations passed. The archive-verified PARSeq runtime strictly loaded
  all 175 tensors and decoded the qualification sample as `SALMON` with
  107,701,760/119,537,664 bytes peak allocated/reserved memory.
- **Record:** The qualification JSON has SHA256
  `23f4e739966ef686b8ef3646e77164f4410d41bec94423bbba9bf9e70e9ed685` and records
  Linux/glibc, executable, package versions, GPU name/VRAM, CUDA runtime, source archive,
  checkpoint, load keys, inference shape, decode, and memory.
- **Strict status:** QUALIFIED for APPROXIMATE use only. This does not resolve any
  paper-selected artifact, preprocessing, scoring, evaluation, or hardware ambiguity.

### M11-BLIP2-HOST-001 — FP16 single-device feasibility on RTX 3090

- **Classification:** `HARDWARE_DEVIATION` for FP16 execution and the RTX 3090/24 GB
  device; the paper does not specify either for Table 5.
- **Decision:** On this host, use the already frozen unquantized FP16 `single_device`
  branch rather than CPU offload. Keep the exact repository prompt and deterministic
  three-beam generation configuration; do not repair repeated caption text.
- **Evidence:** The pinned two-shard checkpoint at revision `0eb0d3b4...` completed a
  batch-one caption of checksum-verified ICDAR13 `img_1.jpg`. Peak GPU allocated/reserved
  memory was 9,035,619,840/9,726,590,976 bytes; peak CPU RSS was 22,351,937,536 bytes;
  load/generation/total latency was 9.90/3.13/13.14 seconds. The profile JSON SHA256 is
  `0c65c3ffd3bd1d0105a28181a4a2071a54a2f10b4228d18237db0ffe3d235ccc`.
- **Strict status:** QUALIFIED for this APPROXIMATE host path only. `C-001` through
  `C-005` remain unresolved and block strict caption and efficiency claims.

### M11-MPNET-HOST-001 — MPNet and score-path host qualification

- **Classification:** `REPOSITORY_SPECIFIED` for the pinned MPNet/RapidFuzz candidates
  and `ASSUMPTION` for the qualification strings, tolerances, candidate-pair aggregation,
  and exact-tie policy.
- **Decision:** Qualify the repository candidate on CPU in float32 with normalized mean
  pooling. Freeze a three-string fixture only as a runtime check; it is not an evaluation
  sample and cannot contribute to a Table 5 claim.
- **Evidence:** On 2026-09-03 the frozen MPNet checkpoint produced normalized
  768-dimensional embeddings. Repeated batched encoding was bit-identical and the
  batch-versus-single maximum absolute difference was
  `8.195638656616211e-08`, below the declared `1e-6` tolerance. The run exercised raw
  RapidFuzz and the three named aggregation strategies with `alpha=0.6`, `beta=0.4`,
  and `tau=0.8`; all requested the fallback for the qualification fixture. The JSON
  record has SHA256
  `32ce49e73b12a0e57ba1ec91cdf6b4dc29e33716679fe51bfb0d9a13da09a1db`.
- **Strict status:** QUALIFIED for the configured APPROXIMATE path only. `S-001` through
  `S-005` remain unresolved and block strict scoring and Table 5 claims.

### M11-E2E-HOST-001 — Complete ICDAR13 image-1 pipeline smoke

- **Classification:** `ASSUMPTION` for the candidate artifacts, component/crop
  selections, scoring, and `highest_score` DeepSolo reduction; `HARDWARE_DEVIATION` for
  RTX-3090 execution, BLIP-2 FP16, sequential release, and the legacy subprocess.
- **Evidence:** On 2026-09-03 the checksum-verified ICDAR13 `img_1.jpg` passed the full
  segmentation, top-10 localization, PARSeq, BLIP-2, MPNet/RapidFuzz, threshold, and
  DeepSolo path. The resolved configuration SHA256 is
  `9fe238088c745fd9cc96badffcda03d8e1ae9293130c84ec0c0f88faaeca0096`; the result
  JSON SHA256 is
  `72099c04b754ac4dc3dc6084dd974c9201365a9bf5b1980e6c9306589906b4b7`.
- **Observed behavior:** `T1="C"`, selected `T3="il"`, and the literal paired confidence
  was `0.013727791244697354`, so fallback was invoked. DeepSolo returned `break`, `your`,
  `could`, `life`, `kills`, `tiredness`, `a`, `save`, and `short`: this is exactly the
  nine lowercased Task 1 GT words as an unordered multiset. The configured
  `highest_score` reduction nevertheless selected only `could` as `Tfinal`.
- **Second-sample replication:** ICDAR13 `img_2.jpg` also completed with four lossless
  crop artifacts. `T1="H"` and selected `T3="8"` are both exact Task 1 GT strings, but
  `C=0.09558176629645362` again forced fallback. DeepSolo returned only `31` for GT
  `319` and omitted GT `8`, `H`, and `10`. The result JSON SHA256 is
  `27d1b58704ccd1f0665f846a5c35bb153aaae34682d0b128f50c4c62b0ee13ca`.
  Thus zero of the first two samples avoided fallback, versus the paper's unresolved
  73%/168-of-229 statement; two samples are not an estimator, but they expose the
  predicted S-003 failure mode without changing the matcher.
- **Lifecycle:** BLIP-2 used about 9.04/9.83 GB allocated/reserved before release; the
  main process returned to about 0.105/0.138 GB before the isolated DeepSolo subprocess.
  DeepSolo inference used about 0.696/0.984 GB peak allocated/reserved.
- **Scientific consequence:** The model boundary works, but the same successful trace
  proves that singular-output reduction and undefined recognition accuracy materially
  change the result. It cannot be aggregated under an invented metric.
- **Strict status:** Successful `approximate reproduction smoke test`; explicitly not a
  Table 5/6/7 result. `R-003`, `S-001`–`S-005`, `D-001`–`D-002`, `DS-002`, and `E-001`
  remain blocking.

### M11-ICDAR13-REC-001 — Frozen 233-image Task 1 population

- **Classification:** `ASSUMPTION` for selecting the complete official 233-image test
  population as the APPROXIMATE branch while the paper's later denominator is conflicting.
- **Decision:** Freeze numeric IDs 1–233, the already frozen per-image identities, and all
  233 Task 1 annotation-file identities without copying transcriptions into git. Decode
  UTF-8 with optional BOM, preserve CSV-decoded transcription text exactly, and validate
  all integer boxes against the decoded image geometry. Define no matching or metric.
- **Evidence:** Physical verification found 233 image/annotation pairs, 1,095 non-empty
  word records, no malformed rows, and no out-of-bounds boxes. Manifest SHA256 is
  `d8ab8863fa2dd00cdbf0667fb36c20a28432ba070164b85d9dbff0827dfb20bd`; ordered
  annotation identity SHA256 is
  `89747036f9dc09b22b3cd35887be0e6f0d694122c099f23f9a6034b8996abd41`.
- **Strict status:** Dataset identity is ready for APPROXIMATE protocol development only.
  `DS-002`, `E-001`, and `D-002` remain unresolved; no Table 5 denominator or accuracy
  has been defined.

### M11-ICDAR13-TRACE-001 — Physical multi-image trace probe, IDs 3--5

- **Classification:** `ASSUMPTION` for the current scoring and singular DeepSolo
  reduction; `HARDWARE_DEVIATION` for RTX-3090 execution, BLIP-2 FP16, sequential model
  release, and the isolated legacy subprocess.
- **Evidence:** On 2026-09-03 the resumable runner completed ICDAR13 test IDs 3--5 with
  no failed samples. The frozen batch contract SHA256 is
  `abb0cd72dea6a3694841c96e835f1e7da52a2a5dd3cfa0b196048e16ba81ff81`;
  the post-range index snapshot SHA256 is
  `3eebc687c396e672512ec5c0607b54410268b5b4394cef86b533f76b2d3575e0`.
  Result JSON SHA256 values for IDs 3, 4, and 5 are respectively
  `73d75d93a1d84b528a5c23eea29c9893005160f9b4bc9ca213b67f3146304385`,
  `0e266c3ab92048edd99c0869eaa33a1c02e2c8725158b0a576e5b60f3c88e7c4`,
  and `7680ba6a677685d81b2f85a4785e6d08ac86b1ed97d2adbbd24b225fca54c16a`.
  Rechecking all three lossless masks and nineteen lossless crops found no file-hash
  mismatch.
- **Observed behavior:** Confidence values were `0.14002422210022375`,
  `0.08306812561017808`, and `0.14359836940305787`, all below `tau=0.8`.
  DeepSolo returned `[london, royal]`, `[produkt, saeco, wehlen, eatrieaberst]`, and
  `[falling, on, hungry]`; `highest_score` reduced these to `london`, `saeco`, and
  `hungry`. For ID 3 both raw detections match the two GT words under case folding, yet
  one is discarded. IDs 4 and 5 likewise contain multiple exact or near-exact GT words,
  while their final ledgers retain only one string. This comparison is a diagnostic
  observation, not a normalization or accuracy definition.
- **Scientific consequence:** Across the independent IDs 1--2 and this IDs 3--5 probe,
  zero of five samples avoided fallback. Five samples cannot estimate the paper's routing
  proportion, but the observation is consistent with the previously predicted literal
  whole-caption S-003 failure mode and confirms that D-002 changes auditable outputs.
- **IDs 6--10 extension:** A second physical range completed all five samples without
  failure. Its index snapshot SHA256 is
  `a498462c36cf773e2d22d828869d4ec758f4168a908d2629b6f7812bc553b4f3`;
  result SHA256 values in numeric order are
  `6f0d46c3f4f560ad7b5f091131b8a229ac3fb84bb1939e4776d9ee073310f932`,
  `0babebe86c518327ea7bd315a3d0568143ba5335f3b560fce7a1bb2c2c04e233`,
  `c92101ac42bcbd2c6e160d7ed580a33f4a166afd54ae22afefb4c89ed5a7d356`,
  `b83b8760cb4190e749b02f22647bf13edd66149b783e07a29055223be13d7b4f`,
  and `dd1b73d0ced503b0f648f0aadddc785d282bfd214bcc87006ccdd8ecaf731c8c`.
  Rechecking five masks and 43 crops found no hash mismatch. Confidence ranged from
  `0.0904451835913498` to `0.26995353443669656`; all five again invoked DeepSolo.
  Raw DeepSolo lists contained multiple case-insensitive exact GT words for every image,
  but `highest_score` kept only `campus`, `clacton`, `cookery`, `intelligent`, and
  `informatikforschung`.
- **Named diagnostic observations:** Over the uniform IDs 3--10 records, literal final
  string membership in any GT is 2/8 case-sensitive and 8/8 under Unicode case folding.
  One-to-one literal multiset intersection of the 44 raw fallback strings with 53 GT words
  is 8/53 case-sensitive and 32/53 case-folded. These are deliberately separate
  `ASSUMPTION`-classified observations, not candidate values to relabel as accuracy.
  The diagnostic refuses Table 5 claims, leaves `accuracy=null`, verifies every source
  ledger/intermediate hash, and suppresses all fractions if any recorded sample failed.
- **Routing consequence:** Including the two independent smokes, zero of the first ten
  exercised samples avoided fallback. This still is not a representative routing-rate
  estimate, but it strengthens the observed incompatibility with the paper's 73% claim.
- **Uniform IDs 1--10 ledger:** Re-running IDs 1--2 into the same immutable batch contract
  produced a ten-record index at SHA256
  `ef93fd51c3679260eef67698553fd486da4cae8e39ce8c4238177184c2a777f9`.
  All ten succeeded, all ten invoked fallback, and the diagnostic verified 86 mask/crop
  PNGs. The diagnostic JSON SHA256 is
  `6a1f4fe6ef06b75dc6b84d19a108a0813fdceb5fe66e65dd3d1e4b5f56c1e091`.
  Its literal final-string membership is 3/10 case-sensitive and 9/10 case-folded;
  one-to-one fallback/GT multiset overlap is 15/66 and 41/66 respectively. These remain
  diagnostic observations, not recognition accuracy.
- **Repository-aligned comparison branch:** Local author-repository history confirms that
  the rank-1 crop assumption used above is not the public runner behavior. A separate
  APPROXIMATE configuration now preserves all other frozen candidates but selects the
  first maximum paired `C3` across every crop and uses `repository_paired_threshold`.
  Its resolved configuration SHA256 is
  `3965fe5d17116ab4d6eded0b2b543661929ffeef13cb884bb70f7a8db721202d`.
  The separate physical IDs 1--10 run completed successfully with batch-contract/index/
  diagnostic SHA256 values `d90c0f07fdc1277b4c1d0e38333078e9a7f00232a8d0f3d6080ccb9ed1ea56f4`,
  `5ae43ce81da3b556cabed258858d80050a655040029c4bcb4d08517b4a80a4e3`, and
  `92455c95edf7214fcb4765166503a421a8b0a60c88c888851ed842031bf2785f`.
  All 86 intermediate hashes passed. Relative to rank-1, selected T3 changed on 7/10
  samples and `C` changed on 5/10, while T1, T2, every PARSeq crop output, every raw
  DeepSolo output, routing, and final output remained identical. Confidence stayed in
  `0.06308199671169296--0.26995353443669656`, so fallback remained 10/10. Selected-T3
  literal GT membership fell from 3/10 to 1/10 under both named casing conventions;
  final and raw-fallback diagnostics were unchanged. This comparison qualifies the
  repository control flow only and strengthens S-003. Because lexical similarity is
  bounded by one, `C >= 0.8` requires semantic similarity `S >= 2/3`; the largest
  semantic score among T1 and every recorded crop was only `0.39617995924870714`.
  Even replacing every lexical score with one would therefore leave all ten below the
  threshold (largest upper bound `0.6377079755492243`). This does not define accuracy or
  justify changing `tau`, score scaling, caption text, or normalization.
- **PARSeq/DeepSolo alignment diagnostic:** The hash-linked alignment report at SHA256
  `93195baebca1106a6eb8ca78dce7cbd007e7471e58da95c4b89805dfa3bfa935`
  retains all 76 PARSeq-crop by 54 DeepSolo-detection pairings for the same ten traces.
  Its spatial comparison is an `ASSUMPTION`: each DeepSolo polygon is reduced to its
  axis-aligned envelope and compared with the exact padded source-coordinate crop box.
  Sixty-six of 76 crops overlap a detection and their median positive best crop coverage
  is `0.9864153811332046`, but only 31/54 detections overlap any crop, only 23/54 have
  their center inside a crop, and only 6/54 have at least half their envelope covered.
  Among the 31 spatially best positive pairs, only 2 are case-fold exact and 5 reach a
  case-folded RapidFuzz ratio of at least `0.8`; the median ratio is `0.5`. These
  asymmetric results show that many retained PARSeq inputs are small subregions or
  fragments inside larger word-shaped DeepSolo envelopes, while many DeepSolo detections
  have no retained segmentation crop. The first-ten outputs are therefore not generally
  aligned at word-instance level. This is a diagnostic conclusion, not an official
  matching rule, recognition accuracy, or authority to change either recognizer.
- **Localization-parameter sensitivity diagnostic:** A predeclared 24-point grid keeps
  the paper-specified foreground-component area, area-descending order, padded original-
  image crops, clipping, and `nmax=10` fixed. It varies only `Amin={50,150,300}` and
  padding `{0,6,12,24}` around the repository values, plus the unresolved 4/8
  connectivity. Every generated crop is associated with ICDAR13 Task 1 GT by maximum
  box IoU before its PARSeq text is inspected. The diagnostic ran on CPU as a
  `HARDWARE_DEVIATION`; its baseline gate reproduced all 76 CUDA-trace crop boxes,
  crop pixels, and decoded strings exactly. At the repository `150/6/8` point, 41/66
  GT words had any crop overlap, 7/66 had best IoU at least `0.5`, and the geometry-first
  PARSeq text was case-fold exact for 5/66 GT words. Changing `Amin` to 50 or 300 changed
  crop count from 76 to 80 or 70 but changed none of those three counts. Connectivity 4
  reduced positive-overlap GT words to 38 and IoU-at-least-0.5 words to 5. Padding 24
  raised overlap to 44, case-fold exact text to 6, and case-fold RapidFuzz-at-least-0.8
  text from 8 to 13, but left IoU-at-least-0.5 at 7 and reduced exact text within that
  strong-geometry subset from 3 to 2. No parameter point is selected: the first-ten
  exploratory grid shows that padding can trade tight geometry for permissive overlap,
  while the tested area/connectivity choices do not repair word-level localization.
  The candidate report SHA256 is
  `185230e4cb893f64c437b2e3532753ae638e281388a50e5ed535c5e1936dea3a`;
  it requires promotion from temporary storage before provenance registration.
- **Strict status:** Successful `approximate reproduction trace collection`; explicitly
  not a recognition evaluation or Table 5 result. `R-003`, `S-001`, `S-003`, `D-002`,
  `DS-002`, and `E-001` remain blocking.

## L-APP-001 — Appearance-guided local remapping (separate prototype)

- **Classification:** `ASSUMPTION`. The user authorized implementation after the
  segmentation/localization logic audit. Paper section 3.3 / Algorithm1 supplies
  the component/crop framework but does not specify appearance-guided grouping.
- **Implementation:** `localize/appearance_word_remap.py`, explicit configuration
  `configs/approximate/localization_appearance_guided_local_remap_v1.yaml` and the
  CPU-only `scripts.diagnose_appearance_remap_train` entry point. Scientific mode
  is `approximate`, result class is `PRACTICAL_APPROXIMATION`, Table5 accuracy null.
- **Choices:** Keep v1 groups immutable; consider only mutual immediate neighbours
  and at most three base groups per rescue. Compare foreground Lab, local contrast
  polarity/magnitude and distance-transform foreground-thickness proxy. Require
  whole-group compatibility, use each base once and never feed merged outputs
  back into neighbour search. Geometry/areas are in original-image pixels.
- **Provisional values:** Appearance DeltaE76<=20, |contrast L|>=5, contrast
  ratio<=2.5, thickness ratio<=2, background-ring/height=.25, minimum3 native
  foreground pixels and9 background samples. Geometry .4/1.5/.5/.5/2.5 is
  transferred from the earlier TRAIN rescue candidate. No new values were chosen
  from TRAIN/TEST/OCR performance. They are engineering hypotheses, not qualified
  thresholds or proof that appearance differentiates adjacent words.
- **Controlled alternatives:** Preserve frozen v1, add a matched-postprocessing
  v1 control, then compare the new groups. Area150 and pad6 reuse repository
  values (`REPOSITORY_SPECIFIED`); applying area filtering after grouping is part
  of this explicit approximation. Area means original foreground pixels, never
  filled-contour interiors. The paper's top10 mechanics are retained.
- **Expected effect and risks:** Additional RGB evidence may reject some false
  geometric links. Same-font/colour neighbouring words remain ambiguous; low-res
  thickness and low contrast can veto correct merges. Existing v1 overmerges are
  not split. No improvement or promotion is claimed without evaluation.
- **Boundary:** No existing Algorithm1/v1/v2, segmentation/checkpoint, recognizer
  or contextual-routing behaviour is changed. No GPU, training, OCR, new grid or
  TEST selection is part of this implementation. See
  `docs/APPEARANCE_REMAP_DIAGNOSTIC.md` for the complete deterministic protocol.
- **Execution guard correction:** TRAIN sample245 is102x422, below128 on one
  axis, but the frozen nearest mapping preserves all33 components one-to-one.
  Rejecting every image smaller than128 was an overly restrictive implementation
  guard. The separate prototype now verifies the actual native/original bijection
  instead; actual component loss/split/merge still causes failure. Mapping,
  thresholds, frozen-v1 behaviour and sample denominators are unchanged.
- **Frozen TRAIN result:** The complete CPU-only run covers229 images /849 words
  and reproduces frozen v1 exactly. Best-IoU>=.5 counts are568 (v1),567 (v1 with
  matched area filter),566 (appearance); F1 is .553606/.622393/.627147. Appearance
  WORD_GOOD is550 vs551 for the matched control; MULTIWORD is40 vs37. Most of the
  apparent F1 gain over frozen v1 comes from area filtering, not new grouping.
  No promotion or OCR improvement is claimed; no post-result parameter change.
  JSON SHA256 `770ce5b13a7ddcb09c84985e1dbc9ed49e70190676d633bf248035714032df06`;
  CSV SHA256 `5a83eb5a6241ada7115e698564e18db4c609ca720a0a1078a90a9277c08f2fc7`.

## L-PERSIST-001 — Multiscale component persistence (separate TRAIN diagnostic)

- **Classification:** `ASSUMPTION`; result class `PRACTICAL_APPROXIMATION`.
  The paper specifies connected components, component-area ordering, padding and
  top10, but not a morphology scale-space, merge tree or persistence selection.
- **Frozen inference choices:** Raw native128 components; existing vertical line
  gates `0.7/0.5/2.5`; requested normalized scales
  `0,.05,.10,.15,.20,.25,.30,.35,.40,.50,.60`; symmetric horizontal dilation
  with half-up integer radius. Dilation determines connectivity only. Final boxes
  union original raw-component boxes and receive six original-pixel padding.
- **Parameter-free selection interpretation:** A node's normalized persistence is
  `(death-birth)/0.60`; weighted interval DP maximizes its product with raw CC
  count. Overlap excludes ancestor/descendant co-selection. Exact ties cover more
  components, use fewer groups, then deterministic spatial/ID order. This
  interpretation was fixed before aggregate GT metrics were read.
- **Boundary:** One CPU-only execution used the pinned229 TRAIN mask cache. All
  proposals were finalized before loading GT. No CUDA, TEST, OCR/PARSeq, training,
  grid or post-result threshold change occurred. Frozen v1 was reproduced exactly.
- **Result:** v1 versus persistence: IoU>=.5 `568 -> 357`, proposal F1
  `.553606 -> .403390`, WORD_GOOD `552 -> 322`, undermerge `92 -> 95`,
  overmerge `36 -> 254`, and MULTIWORD `37 -> 158`. It recovered40 v1
  failures but lost251 v1 successes. This is `NO-GO`; no promotion or TEST/OCR
  evaluation is authorized by this result.
- **Artifact:** `/tmp/multiscale-component-persistence-train-v1.json`, SHA256
  `cc4ed54d91855388d550c4408e865095d8ec67666de103a2793e0c2cbd53d119`.

## L-SPAN-001 — Contiguous component-span oracle (separate TRAIN diagnostic)

- **Classification:** `ASSUMPTION`, `DIAGNOSTIC_ONLY`; result class
  `PRACTICAL_APPROXIMATION_DIAGNOSTIC`. Paper section 3.3 / Algorithm 1 does not
  specify text-line formation, component-span enumeration, or candidate reranking.
- **Frozen inference-safe construction:** Use the pinned native128 TRAIN masks,
  the already frozen v2 vertical line/fragment geometry, raw components ordered
  left-to-right, every contiguous span of at most12 raw components, and maximum
  span width / line-median raw-component height of15. Final boxes are unions of
  unmodified native raw-component boxes, remapped to the source image and padded
  by six original-image pixels. Exact final padded-box equality is the only
  deduplication rule; no near-IoU threshold was introduced. The oracle pool is
  deliberately not capped at10 because this task measures candidate-generation
  feasibility rather than runnable pipeline precision.
- **Scientific boundary:** Candidate generation completed for all229 TRAIN images
  before Task2.2 GT was opened. GT was used only to select the best candidate per
  word and classify failures post hoc. No CUDA, TEST, OCR/PARSeq, training,
  parameter selection, or pipeline modification occurred. Frozen v1 retained
  proposals reproduced the established `679/568/324` counts at IoU
  `.25/.50/.75`.
- **Result:** The16,325 unique spans average71.29/image (median46, p95 234.6).
  Oracle coverage is747/849 at IoU>=.25,659/849 at>=.50,366/849 at>=.75,
  and73/849 at>=.90; median best IoU is.72044. The `.50` ceiling is77.62%,
  below the predeclared85% feasibility gate, hence
  `CANDIDATE_GENERATION_INSUFFICIENT`. Of190 failures,76 have no foreground
  component,63 fail union geometry/remapping,27 require multiple generated
  lines,12 exceed the12-component span cap, and12 reflect line assignment.
- **Artifact:** `/tmp/contiguous-span-oracle-coverage-train-v1.json`, SHA256
  `b18ff8ed9735c9031e08d41880996e2217efbcd202cee7c328331cd9850aca35`.

## L-MHYP-001 — Multi-hypothesis word-proposal oracle (separate TRAIN diagnostic)

- **Classification:** `ASSUMPTION`, `DIAGNOSTIC_ONLY`; result class
  `PRACTICAL_APPROXIMATION_DIAGNOSTIC`. The paper specifies one binary mask and
  Algorithm1 component boxes; it does not specify a probability-threshold
  ensemble, contiguous span hypotheses, cross-line recovery, bbox variants or
  oracle reranking.
- **Frozen candidate protocol:** Native sigmoid thresholds
  `.20/.30/.40/.50`, all retained simultaneously; maximum20 contiguous raw
  components; width/median-height pruning15 inherited from `L-SPAN-001`;
  frozen line geometry; and cross-line unions only when raw components from
  distinct lines pass the already frozen v1 gates `.35/.7/.5/2.5`. Each span
  yields standard six-pixel padding, a tight box and adaptive original-coordinate
  padding `max(6, round(.15 * median component height))`. Greedy stable
  deduplication uses bbox IoU>=.95 and frozen precedence `.50,.40,.30,.20` so
  neither GT nor an oracle score chooses representatives. Stage5 and FULL are
  intentionally identical because the requested cumulative definition adds no
  operation after bbox variants.
- **Boundary and provenance:** The pinned229-map probability cache has manifest
  SHA256 `9eb8f44ad939fdc72d4a4d97f18c56a674b1525b843cac44752f9a0f69d6f7ba`;
  every map/file hash and its threshold-.50 mask identity was reverified on CPU.
  All candidate pools for all229 images were finalized before GT was opened.
  No segmentation rerun, CUDA, TEST, OCR/PARSeq, training or result-driven
  parameter choice occurred. Algorithm1 and practical-remap-v1 were unchanged.
- **Result:** The old `.50` span pool gives659/849 (`77.62%`) oracle recall at
  IoU>=.5. Threshold ensembling raises this to705 (`83.04%`); span20 and
  line-robust additions do not raise the `.5` count. Bbox variants raise FULL to
  735 (`86.57%`), with median best IoU `.91269`, at a cost of285.12 candidates
  per image (median164, p95 `889.2`). Of114 FULL failures,57 have no predicted
  foreground in the word at threshold.20,40 fail component-union geometry,
  13 fail line/topology and4 hit pruning/span limits. This satisfies only the
  predeclared `BORDERLINE` band (`.85 <= recall < .90`), not GO.
- **Artifact:** `/tmp/multihypothesis-word-proposal-oracle-train-v1.json`, SHA256
  `501fb6deeac0e0741d5e77103e0ba1144664cf064db015b4f7957791df00df57`.

## R-RANK-001 — PARSeq candidate-confidence ranking (separate TRAIN diagnostic)

- **Classification:** `ASSUMPTION`, `DIAGNOSTIC_ONLY`; result class
  `PRACTICAL_APPROXIMATION_DIAGNOSTIC`. Neither the paper nor its Algorithm2
  defines PARSeq confidence as a word-proposal ranking signal.
- **Frozen proposal input:** Use the native probability ensemble
  `.20/.30/.40/.50`, same-line contiguous spans capped at12, no line-robust
  candidates, the tight/standard-six/adaptive bbox variants, and stable greedy
  bbox deduplication at IoU>=.95. This is separate from and does not modify
  practical-remap-v1 or the paper-faithful Algorithm1 path.
- **Three independent scores:** (A) upstream PARSeq's product of tokenizer-
  filtered greedy-path probabilities, including EOS when present; (B) mean
  character log-probability obtained stably from `log_softmax`; (C) minimum
  decoded-character probability. B/C exclude EOS. Empty B/C values rank after
  finite values; exact ties use deterministic candidate ID. No weights, score
  combinations, classifier training or threshold selection are permitted.
- **Boundary:** Candidates and crops are generated and frozen without GT. PARSeq
  runs in deterministic batches using the qualified checkpoint/runtime. Only
  after all OCR shards exist may Task2.2 TRAIN GT be opened for positive-box-
  intersection association, IoU success, case-fold exactness and rank metrics.
  TEST, segmentation inference, routing, BLIP2, MPNet and DeepSolo are excluded.
- **Status:** Implementation prepared; no result or promotion is claimed until
  the manual PAPER GPU command completes and the final report is hash-verified.

## S-CTX-RANK-001 — BLIP/context proposal ranking (separate TRAIN diagnostic)

- **Classification:** `ASSUMPTION`, `DIAGNOSTIC_ONLY`; result class
  `PRACTICAL_APPROXIMATION_DIAGNOSTIC`. The paper defines semantic and lexical
  alignment between recognized text and one full-image BLIP-2 description, but
  does not specify using those scores to rank overlapping word-box hypotheses.
- **Frozen diagnostic protocol:** Reuse exactly the 58,707 candidate boxes and
  PARSeq transcriptions from `R-RANK-001`. Generate exactly one caption per each
  of the 229 TRAIN images with the pinned repository BLIP-2 candidate, prompt,
  decoding configuration, unquantized FP16 and the qualified RTX-3090
  single-device path. Reuse that caption for every proposal in its image. Rank
  candidates independently by normalized MPNet cosine `S`, raw whole-string
  RapidFuzz ratio `L`, fixed paper weights `C=0.6S+0.4L`, or the frozen PARSeq
  mean-token-log-probability baseline. Do not apply `tau`, combine context with
  PARSeq confidence, select weights, train, or access TEST.
- **Evaluation boundary:** BLIP captioning reads only full RGB images. Candidate
  S/L/C values are computed from frozen OCR artifacts and the caption cache
  before the already frozen GT-derived report rows are accessed. GT is used only
  post hoc for association and ranking metrics. `GO` is predeclared to require
  fixed `C` to have positive net top-1 flips over mean-token-log-probability for
  both IoU>=.50 and PARSeq exactness; otherwise the result is `NO_GO`.
- **Status:** Implementation prepared; no diagnostic result is claimed until the
  manual PAPER BLIP-2 export and subsequent CPU ranking finish and their report
  hashes are verified.

## S-CTX-LOCAL-001 — Anti-repetition caption and local-context matching (TRAIN diagnostic)

- **Classification:** `ASSUMPTION`, `DIAGNOSTIC_ONLY`; result class
  `PRACTICAL_APPROXIMATION_DIAGNOSTIC`. This is a separate practical branch and does
  not alter the repository/paper-faithful caption or whole-caption scorer.
- **Caption decision:** Keep the pinned BLIP-2 model, revision, prompt, deterministic
  three-beam decoding, unquantized FP16 placement and one caption per image. The
  practical profile changes only generation termination/repetition controls:
  `min_new_tokens=20`, `max_new_tokens=80`, `early_stopping=true`,
  `no_repeat_ngram_size=3`, and `repetition_penalty=1.1`. The frozen repository profile
  remains byte-for-byte contract compatible. This choice does not resolve C-002 through
  C-005 and requires a new manually executed PAPER GPU caption cache.
- **Matching decision:** Tokenize each caption deterministically into auditable contiguous
  windows sized either exactly like the candidate transcription or at candidate length
  plus/minus one token. Compute MPNet cosine `S` and raw RapidFuzz `L` against the same
  window and maximize the paired paper-weighted score `C=0.6S+0.4L`. Preserve
  `tau=0.8`. Do not average across unrelated caption windows: that would reintroduce the
  length dilution observed for a word compared with a full scene description.
- **Frozen repository-caption evidence:** In TRAIN-229, 187 captions contained a repeated
  four-gram and the median duplicate-four-gram fraction was `0.5862`. Whole-caption
  matching sent all `229/229` images to fallback. On the fixed
  `BALANCED_ANCHOR_MEAN` proposal set, exact-length local matching accepted `101/229`
  images, of which `83` were spatially and textually exact (`18` false accepts);
  neighbor-length matching accepted `103/229`, with `83` spatially exact (`20` false
  accepts). Report SHA256:
  `88bb97e9d90043dd8c76d4da71ab900179d8ceb41947c3e269e76c38fee55974`.
- **Safety boundary:** The unpruned 58,707-candidate pool produced too many false accepts,
  so local scores may be evaluated only after a frozen proposal-selection stage. TRAIN
  GT is loaded only after proposals and context scores exist. No TEST, OCR rerun,
  segmentation rerun, parameter training or Table 5 claim is permitted here. Promotion
  remains pending evaluation with the new anti-repetition caption cache.
- **Persistent anti-repetition cache:** The PAPER host completed all 229 TRAIN captions
  under shared `/extra1` storage. The manifest SHA256 is
  `d427d18e435b314c4407906d86d1558c4908a6c57a55d670937262cdc54a1497`;
  all 229 per-image shard hashes verify. Post-hoc quality checks find zero captions with
  a repeated four-gram (repository control: 187), median/mean token counts `16/15.63`,
  and a total cache size of about 1.4 MB. A compact persistent v1 evaluation path stores
  at most ten lossless crops and one candidate/OCR shard per image; segmentation and
  PARSeq are loaded sequentially and all completed shards are verified when resuming.

## L-PROB-THRESH-001 — Probability-threshold word-connectivity sweep (TRAIN diagnostic)

- **Classification:** `ASSUMPTION`, `DIAGNOSTIC_ONLY`; result class
  `APPROXIMATE_DIAGNOSTIC`. The paper specifies thresholding a segmentation
  prediction and applying connected-component localization, but does not establish
  that lowering the frozen author-protocol threshold from `.50` is valid.
- **Frozen diagnostic protocol:** Reuse the verified 229-image TRAIN native
  probability cache; evaluate the predeclared thresholds
  `.50,.475,.45,.425,.40,.375,.35,.325,.30`; at each threshold use native 128x128
  connected components followed by the frozen component-area filter, original-image
  coordinate remap, six-pixel original-coordinate padding, component-area ordering,
  and top-10 cap. No grouping, morphology, segmentation rerun, OCR, TEST access, or
  training occurs in the geometry sweep. GT is opened only after proposals have
  been generated and is used post hoc for diagnostic topology and geometry metrics.
- **Selection rule:** Freeze the threshold with the most `WORD_GOOD` proposals;
  break an exact tie by fewer `MULTIWORD_MERGE`, then fewer overmerges, then the
  threshold closest to `.50`. This selects `.30` with 220 `WORD_GOOD` proposals.
- **Result:** Lowering `.50` to `.30` changes 57 initially fragmented words to one
  component and 42 of those also become `WORD_GOOD`; it introduces three new unique
  GT-word fusion pairs. Geometry remains far below practical-remap-v1: 222/849
  versus 568/849 GT words at IoU>=.50, and 220 versus 552 `WORD_GOOD` proposals.
  The result supports a modest connectivity effect, not threshold replacement or a
  paper-reproduction claim.
- **Artifacts:** `/tmp/probability-threshold-word-connectivity-train-v1.json`,
  SHA256 `49038549de0ba4ce1b90e9c09e78cb261deb1b1ba8c467997010973aabc09913`;
  selected crop manifest
  `/tmp/probability-threshold-word-connectivity-train-v1-crops/manifest.json`,
  SHA256 `f8cd3b710e3a93e4aabbfb4d6ecb095b292501c03c7eec8d94fa95659e70e193`.

## R-CAND-SEL-001 — Practical candidate selection (separate TRAIN diagnostic)

- **Classification:** `ASSUMPTION`, `DIAGNOSTIC_ONLY`; result class
  `PRACTICAL_APPROXIMATION_DIAGNOSTIC`. The paper does not define generation of
  multiple word-box hypotheses, candidate-family consensus, PARSeq-confidence
  proposal selection, or global interval optimization. Algorithm1,
  practical-remap-v1 and practical-remap-v2 remain unchanged.
- **Frozen inputs and leakage boundary:** Reuse the 58,707 same-line span-12
  candidates, their qualified PARSeq outputs, the four native probability maps
  (`.20/.30/.40/.50`) and the already computed fixed BLIP/MPNet context score.
  practical-remap-v1 boxes are inference-time anchors. Each anchor admits its
  closest hypothesis plus at most two line-compatible alternatives having bbox
  IoU>=.50 to that anchor. GT is absent from selector APIs and is first accessed
  only after every TRAIN selection is frozen. No CUDA, model inference,
  training, TEST access or parameter fitting occurs.
- **Predeclared variants:** `OCR_FIRST`, `CONSENSUS_FIRST` and
  `EQUAL_RANK_FUSION` use fixed within-image percentile-rank weights over mean
  PARSeq token log-probability, threshold-family persistence, line-relative
  boundary isolation, native foreground density and v1 support. Context is an
  exact-tie-break only. Per-line weighted interval scheduling removes overlap,
  followed by deterministic cross-line deduplication and top-10.
- **Fail-closed correction:** An initial implementation admitted every positive
  anchor intersection and allowed a `.50` utility floor to remove the base
  hypothesis. It lost 227--295 v1 successes and is retained only as a rejected
  audit artifact. The accepted implementation corrects the already declared
  anchor-local contract; this is not a result-selected scientific parameter.
- **Result:** The best corrected variant is `OCR_FIRST`, but it remains below
  frozen v1: IoU>=.50 is516/849 versus568/849; proposal F1 is.51859 versus
  .55361; `WORD_GOOD` is498 versus552; exact PARSeq is326/849 versus502/849;
  fuzzy>=.8 is501/849 versus578/849. It recovers15 v1 geometry failures but
  loses67 v1 geometry successes and raises selected multiword crops from37 to
  47. Therefore this selector is `NO_GO` and must not be promoted or evaluated
  on TEST merely to seek a favorable result.
- **Artifacts:** Rejected initial report
  `/tmp/practical-candidate-selection-train-v1.json`, SHA256
  `37b985f4780e6f279c7b7c34277d6b834799054a184a4a8a106ea11fe204a516`;
  corrected report
  `/tmp/practical-candidate-selection-train-anchor-safe-v1.json`, SHA256
  `a3997573c41bfda3165c4078446c2f4ef6b0ecc0705fec6ea95f1a817d34aa66`.

## R-CAND-SEL-002 — Anchor-complete candidate selection (TRAIN diagnostic)

- **Classification:** `ASSUMPTION`, `DIAGNOSTIC_ONLY`; result class
  `PRACTICAL_APPROXIMATION_DIAGNOSTIC`. This is a correction experiment for
  `R-CAND-SEL-001`, not part of the paper recognition algorithm.
- **Selection contract:** Every exact practical-remap-v1 crop is a feasible
  singleton fallback. Alternative boxes may cover only a contiguous span of
  same-line v1 anchors and must contain each anchor's unpadded box. Dynamic
  programming partitions every anchor line completely. Its fixed-denominator
  objective is the candidate arithmetic-mean score multiplied by the number of
  covered anchors; therefore two fragments cannot beat a whole-word candidate
  merely because two proposal scores are summed.
- **Deduplication provenance:** Candidate boxes and ordering remain identical
  to the frozen 58,707-candidate pool. Every hypothesis suppressed at bbox
  IoU>=.95 now contributes its threshold, bbox variant, line and raw-component
  provenance to the retained candidate. CPU verification reproduced all 58,707
  frozen boxes in order and found 15,935 candidates supported by all four
  thresholds.
- **Variants:** `OCR_ONLY_ANCHOR_MEAN`,
  `OCR_CONSENSUS_ANCHOR_MEAN`, and `BALANCED_ANCHOR_MEAN` are predeclared.
  Context is used only for deterministic exact-score ties. No GT enters feature
  generation or selection.
- **Qualified TRAIN control:** The manual qualified PARSeq pass reproduced the
  frozen v1 control exactly:568/849 GT words with proposal IoU>=.50,
 502/849 exact casefold and578/849 RapidFuzz casefold>=.8. This closes the
  fail-closed prerequisite for interpreting the selector variants.
- **Result:** `BALANCED_ANCHOR_MEAN` is the best predeclared variant. Relative
  to v1 it raises IoU>=.50 from568 to593, IoU>=.75 from324 to434,
  proposal F1 from.55361 to.61008, `WORD_GOOD` from552 to577, exact
  PARSeq from502 to534, and fuzzy>=.8 from578 to598. It recovers40 v1
  geometry failures while losing15 v1 geometry successes; it recovers47
  exact OCR failures while losing15 exact successes. The selected proposal
  count falls from1,203 to1,095.
- **Decision:** `REJECT` under the promotion rule frozen before evaluation:
  exact must improve, IoU>=.50 recall must not decrease, and `MULTIWORD_MERGE`
  must not exceed v1. The first two conditions pass, but multiword proposals
  rise from37/1,203 (3.08%) to55/1,095 (5.02%). This TRAIN diagnostic must
  therefore not proceed to TEST as a promoted replacement without a new,
  separately predeclared TRAIN-only experiment.
- **Artifact:** `/tmp/practical-candidate-selection-v2-train.json`, SHA256
  `8db4dac11c9d08e956d1f11c982b97c5e538e009f3dff9d0b5bd0e393b7a94e5`.

## R-E2E-COMP-001 — Algorithm1 versus rejected balanced selector on TEST

- **Classification:** `ASSUMPTION`, `DIAGNOSTIC_ONLY`; result class
  `PRACTICAL_APPROXIMATION_DIAGNOSTIC`. This is an explicit user-requested
  descriptive TEST comparison, not promotion of the TRAIN-rejected selector and
  not a Table 5 reproduction.
- **Frozen comparison:** Run the official233-image TEST population once through
  the frozen epoch-98 segmentation checkpoint. The exact same segmentation
  output feeds (A) Algorithm1 with the repository/assumption parameters
  `Amin=150`, original-pixel padding6, connectivity8 and top10; and (B) the
  unmodified TRAIN-frozen `BALANCED_ANCHOR_MEAN` selector. T1, one full-image
  BLIP-2 caption, MPNet embeddings, and any required DeepSolo output are shared
  per image. The repository scorer selects the first maximum-C3 crop, thresholds
  `max(C1,C3)` at `.8`, and returns the semantic winner before fallback.
- **Leakage boundary:** The balanced selector remains `REJECT`; TEST outcomes
  cannot change its weights, gates, generator or promotion status. GT is loaded
  only after segmentation, proposals, OCR, caption, selection, routing and
  fallback shards are complete. All stages are resumable and provenance-bound.
- **Metric boundary:** E-001, DS-002 and D-002 remain unresolved. The report keeps
  Table5 `accuracy=null` and exposes best-geometric-crop OCR, singular final-text
  membership, and per-image one-to-one case-folded multiset diagnostics. No
  numeric delta from the paper's88.8% is computed because those diagnostics are
  not known to be the paper metric.
- **Physical TEST result:** All five stages completed for233/233 images using
  shared segmentation, T1, captions and fallback outputs. Algorithm1 versus
  balanced proposal recall at IoU>=.50 is174/1,095 (`15.89%`) versus528/1,095
  (`48.22%`); best-geometric-crop PARSeq exact is102 (`9.32%`) versus453
  (`41.37%`). Before fallback, singular final-string GT membership is71/233
  (`30.47%`) versus152/233 (`65.24%`). Balanced changes selected T3 on201
  images and the pre-fallback output on151.
- **Routing discrepancy:** Both branches remain below the paper-specified
  `tau=.8` on every image: median/max C are `.13846/.46005` for Algorithm1 and
  `.21110/.46005` for balanced. Therefore DeepSolo is invoked233/233 times,
  unlike the paper's reported61/229 fallbacks, and both branches have identical
  final outputs. DeepSolo produces1,026 strings, with861 per-image one-to-one
  case-folded multiset matches (`78.63%` recall, `83.92%` precision, `81.19%`
  F1). Its singular highest-score string is a GT member on207/233 images,
  `88.841%`, which rounds to the paper's`88.8%`; this is strong diagnostic
  evidence about a possible metric interpretation, not authoritative resolution
  of E-001 because the denominator and reduction are still undocumented and the
  paper reports229 routing images.
- **Artifact:** `/tmp/icdar13-e2e-faithful-vs-balanced-v1.json`, SHA256
  `e6db8027e1d0f6ec3bf5e3b574c431ad1779857330825cb14bf766f505747eee`.

## R-ROUTING-SUITE-001 — Persistent frozen TEST routing comparison

- **Classification:** `ASSUMPTION`. The user requested a descriptive comparison
  of the existing techniques and conditional fallback counts. Result class is
  `PRACTICAL_APPROXIMATION_DIAGNOSTIC`, labeled approximate reproduction.
- **Evidence:** The earlier full TEST experiment recorded 207/233 final-string
  GT membership with 233/233 fallbacks. Its original temporary JSON is now
  missing; the record is preserved, not claimed to have been recomputed.
- **Frozen choices:** Reuse Algorithm1, v1 and BALANCED_ANCHOR_MEAN unchanged;
  compare repository routing and the previously TRAIN-tested exact-length,
  T3-only local caption gate. Original and anti-repetition captions are separate
  predeclared branches. No thresholds/weights are selected on TEST. Balanced
  retains its historical TRAIN `REJECT` status.
- **Source classifications:** alpha=.6/beta=.4/tau=.8 are `PAPER_SPECIFIED`;
  repository crop/scoring interpretation is `REPOSITORY_SPECIFIED`; local
  caption windows, grouping, anti-repetition and single-output membership
  evaluator are `ASSUMPTION`; FP16 BLIP2, sequential loading and isolated legacy
  DeepSolo on RTX3090 are `HARDWARE_DEVIATION`. E-001/DS-002/D-002 remain
  `UNRESOLVED`; Table 5 comparable accuracy and gap remain null.
- **Reporting:** Standalone DeepSolo is computed once per image for controls.
  These physical executions are distinct from each branch's logical fallback
  count. Accepted predictions are cross-tabulated with the shared DeepSolo
  output. See `docs/ICDAR13_ROUTING_AUDIT.md` for metric definitions, frozen
  branches, persistent paths and invocation. No new GPU results are claimed.

## L-004-JOIN-001 — User-requested concatenation of retained block texts into T3

- **Classification:** `ASSUMPTION`. The user explicitly requests that T3 contain
  all retained block transcriptions, joined together. This supplies the practical
  protocol; it does not establish how the paper authors resolved L-004.
- **Implementation:** `diagnostics.joined_t3` forms T3 using the existing frozen
  geometric line-order helper on retained crop boxes, left-to-right within each
  line, joining nonempty stripped strings with one space. Duplicate words and
  punctuation are preserved. No boxes are regenerated, merged, filtered or
  selected using text, confidence or GT. Empty blocks remain in provenance.
- **Comparison:** T1 remains the full-image PARSeq string. Both complete T1
  and complete joined T3 are compared to the same full caption T2 using MPNet
  and raw RapidFuzz. Two already implemented scoring interpretations are
  reported separately: `repository_paired_threshold` and
  `selected_candidate_paired`. Neither is selected from TEST outcomes. The
  latter computes confidence from the same semantic-winning candidate, following
  the prose in paper section 3.5; the former preserves the repository control.
- **Frozen settings:** alpha .6 / beta .4 / tau .8 (`PAPER_SPECIFIED`), with
  T1 winning exact semantic ties (existing `ASSUMPTION`). Original and existing
  anti-repetition captions are reused as separate profiles. Cached DeepSolo's
  `highest_score` reduction is unchanged. This is image-level T1-versus-T3
  routing, not independent per-word gating.
- **Execution boundary:** CPU MPNet replay only, official TEST-233, checksummed
  artifacts of the completed routing suite. No new segmentation, PARSeq,
  BLIP2, DeepSolo inference or training. All GT access follows scoring.
- **Output:** `artifacts/diagnostics/t3-joined-test-v1/results.json` is an ignored
  persistent local report, with assembled T3 blocks, line IDs, all decisions,
  fallback counts and explicit word-token diagnostics. Those metrics split the
  serialized final string on whitespace, match casefold tokens one-to-one within
  each image and are not a spatial or official Table 5 evaluation. The retained
  single-output DeepSolo reduction must not be confused with all raw detections.
- **Isolation:** The separate `diagnostics` namespace does not change the source
  files or source-hash contract of the previously completed suite. Original
  reports and configurations are retained. `table_5_accuracy` remains null.
- **CPU replay result:** All 233 TEST images completed. For both scoring
  interpretations, original captions yield 233/233 fallbacks in Algorithm1,
  v1 and Balanced; anti-repetition captions yield 233/233, 232/233 and 232/233,
  respectively. Concatenating T3 alone therefore does not remove the near-total
  fallback under whole-caption matching at tau .8. No parameters were changed
  after these observations. Report SHA256:
  `85ba5f70c6662837937a0c1703a95efecd5d4bcad912ad777913e71094671690`.

### S-006 — Opt-in InstructBLIP paired image-budget calibration (2026-09-16)

- `PAPER_SPECIFIED`: section 3.4 prose selects the semantic winner and its paired
  lexical score with alpha=.6/beta=.4. Full T2 is retained; the Figure 2 conflict
  remains unresolved for strict claims. Existing faithful tau=.8 is unchanged.
- `ASSUMPTION`: user-selected InstructBLIP/practical-v1 baseline, largest observed
  TRAIN C threshold allowing at most floor(.30*229)=68 image fallbacks. Equal C
  ties are not split; empty outputs always fall back. Selection has no access to
  GT or OCR correctness. A new distribution need not have the same fallback rate.
- `ASSUMPTION`: retain cached T1 semantic-tie policy (Algorithm 2 uses strict >).
  No segmentation, OCR, caption or embedding inference was rerun; no TEST loaded.
- Frozen tau: 0.23055392728092478. TRAIN: 161 accepted images, 68 fallback images.
  Post-hoc image-local casefold token-multiset matching yields 398 correct words
  among 560 GT words on accepted images; 289 GT words occur on fallback images.
  Even perfect fallback caps this route at 687/849 (80.9187%). At least 282/289
  fallback words must be correct to reach ceil(.8*849)=680 words. This is an upper
  bound, not an observed end-to-end accuracy; actual fallback and Table5 accuracy
  remain null. Extra incorrect tokens are separately reported (303).
- Approximate configuration: `configs/approximate/instructblip_budgeted_paired_train_v1.json`.
  CPU replay: `diagnostics/budgeted_paired_train.py`; opt-in scorer:
  `scoring/budgeted_paired.py`. Existing pipeline defaults are not changed.
- Persistent report: `artifacts/diagnostics/instructblip-budgeted-paired-train-v1/results.json`;
  SHA256 `2ee15ee137fdcf68605f001fa05f523a4c17b42673ff9848172dc25668f1a9c2`.

### S-006 / D-002 / E-001 — Frozen-tau TRAIN/TEST end-to-end extension

- `PAPER_SPECIFIED`: retain semantic winner's paired S/L under section 3.4 prose,
  alpha=.6/beta=.4 and image-triggered DeepSolo; the Figure 2 conflict persists.
- `ASSUMPTION`: extend the user-selected practical InstructBLIP/v1 configuration
  with the fixed TRAIN-budget tau, unchanged on TEST. This is not a new strict
  default, and the TEST fallback rate need not equal 30%.
- `ASSUMPTION`: for the requested all-word accuracy, serialize **all DeepSolo raw
  recognized strings**, not just the highest-confidence detection. Preserve and
  separately evaluate the original highest-score reduction as a control. No GT
  word is used to select detections or choose between reductions at inference.
- `ASSUMPTION`: within-image exact casefold whitespace-multiset matching, with
  duplicate counts capped by GT and all missing words counted as errors. Report
  extra tokens/precision and exact whole-image multiset equality separately.
  Spatial matching, official Table 5 equivalence, and the author's exact output
  reduction remain unresolved. `table5_accuracy` remains null.
- `HARDWARE_DEVIATION`: preserve sequential FP16 InstructBLIP on PAPER GPU,
  CPU FP32 MPNet and isolated qualified legacy DeepSolo subprocess. No new
  segmentation or PARSeq inference; reuse original sealed image outputs.
- Diagnostics and executable protocol: `docs/BUDGETED_PAIRED_E2E.md`,
  `diagnostics/budgeted_paired_e2e.py`, and its host wrapper. TRAIN standalone
  DeepSolo control calls are explicitly distinguished from conditional fallback
  calls. GT is used only for evaluation. No TEST-based selection is performed.

### MASK-OCR-REGION-001 — Frozen TEST mask quality versus v1 OCR errors

- `ASSUMPTION`: post-hoc per-word associations use the existing v1 crop of highest
  bbox IoU, with a positive intersection required. No GT enters inference.
  This diagnostic is distinct from T3 token-multiset accuracy and routed accuracy.
- `ASSUMPTION`: reuse the native TEST evaluator's exact-white Task2 foreground,
  bilinear binary-target resize to 128 and threshold >=.5. Exact Task2 TEST ignore
  annotations remain unavailable; the existing all-colors-foreground treatment
  is retained and disclosed rather than presented as exact training Don't Care.
- All 233 cached uint8 native prediction hashes match the actual routing suite's
  frozen epoch98 masks. No model, segmentation or OCR inference was executed.
- `ASSUMPTION`: diagnostic severity bins recall<.5, recall .5-.8, then recall>=.8
  split by pixel precision .8 and crop IoU .5. These bins are descriptive only,
  neither tuned thresholds nor a causal attribution of recognition failures.
- Existing recognition GT maxima are inclusive. The audit explicitly adds one
  before half-open geometry/mapping; it separately reproduces the legacy control
  (which omitted that increment): exact411 and IoU>=.5 count473. Corrected geometry
  gives483, with exact411 unchanged. No pipeline outputs or older reports changed.
- Among 684 wrong/missing best-crop words: 311 have pixel recall<.5; 98 have
  recall .5-.8; 68 lose all GT foreground during native target resizing despite
  positive source-resolution foreground; 84 have high recall but precision<.8;
  83 have high recall/precision but poor crop IoU; 40 have high recall/precision
  and IoU>=.5 but wrong OCR (residual crop quality versus recognizer unresolved).
- Median native pixel recall is .9481 for correct words, .4585 for wrong/missing
  words with nonempty native GT. This supports both segmentation/scale loss and
  downstream grouping/crop problems; it does not justify blaming all errors on
  either U-Net or PARSeq. No inference tuning follows these TEST observations.
- Output: `artifacts/diagnostics/parseq-mask-region-test-audit-v2/results.json`,
  SHA256 `1538847ae8e008dcdd70ad34537679b62c6a123014528a1c8fe9ab9925ebba32`;
  per-word CSV SHA256 `b4aad275c6bef6273e9db8d8d5616630f42e4be9c98aeeaf47f90659a64e5024`.
  v1 is preserved; v2 adds verification of source foreground for vanished targets.

### GT-MASK-LOCALIZATION-001 — frozen TRAIN localizers on perfect pixel targets

- `ASSUMPTION`: a TRAIN-only counterfactual geometry diagnostic, not a new
  localizer, tuning exercise, OCR experiment, or Table 5 accuracy claim.
  Run the same frozen algorithms on cached predictions, qualified native128 GT,
  and original-resolution Task 2.2 GT. Preserve every algorithm parameter.
- `PAPER_SPECIFIED`: section 3.3 / Algorithm 1 uses components, component-area
  filtering, padded boxes and top10. Perfect **glyph/pixel** foreground is not
  the same object as a connected, filled word-box mask. Do not silently replace
  one target definition with the other.
- `REPOSITORY_SPECIFIED`: retain upstream bbox-area contour semantics as a
  separate comparison, never as an equivalent to true component-area Algorithm1.
- `ASSUMPTION`: native GT is the qualified bilinear resize to128 followed by
  >=.5. Original-coordinate algorithms use the unchanged binary nearest
  backprojection. The native128 contour variant retains native area/padding and
  runs only on native inputs, not on a silently rescaled full-resolution branch.
- `ASSUMPTION`: generate proposals from the whole target foreground, including
  Don't Care; do not apply GT validity downstream to either predicted or oracle
  masks. Evaluate care words from the existing Task2.2 word-group protocol.
  Ignored-region outputs are not uniquely supervised: perfect-target inference
  here is a disclosed counterfactual, not an attainable-model guarantee.
- `ASSUMPTION`: practical-remap-v1 and user-frozen v1.1-rescue are grouping
  controls. No grid, threshold selection, TEST, segmentation inference or OCR.
  Better masks need not monotonically improve these frozen grouping rules; the
  comparison is not a formal mathematical upper bound on localization.
- `UNRESOLVED`: no verified frozen selected v2 configuration was located in the
  checked artifacts. Exclude it rather than invent parameters or repeat a grid.
- Protocol: `configs/approximate/gt_mask_localization_oracle_train_v1.json`.
  Output: `artifacts/diagnostics/gt-mask-localization-oracle-train-v1/`.
  Verify all TRAIN image/mask/annotation hashes and all229 cached mask identities;
  require exact reproduction of historical Algorithm1/repository/v1 boxes and
  geometry/wordness aggregates before accepting the new report. Report top10
  losses and PRED→GT recovered/lost words separately. No defaults are changed.
- Completed TRAIN result: Algorithm1 IoU>=.5 coverage171/849 on prediction,
  172 on native GT,42 on source GT; v1 gives568,548,626 respectively; frozen
  rescue578,564,629. Original GT has752 words intersecting>=3 raw components.
  v1 native GT still misses301 words (35 lack foreground,23 top10,243 remaining
  geometry failures even before cap). On source GT it misses223 (40 top10,
  183 remaining geometry failures). These are not OCR accuracies or rigorous
  attainable-model ceilings. Even without Don't Care regions, v1 prediction
  coverage375/481 exceeds native-GT coverage362/481; ignored pixels alone do
  not explain the deficit. Result SHA256
  `cec8fbc12b3c246daecfda25e722970b0cc8ed2890621a8088435e69daa57564`.

### L-DIRECTION-001 — component-neighbour priority and orientation audit

- `ASSUMPTION`: separate TRAIN229/849 diagnostic on exactly the prior prediction,
  native-GT and source-GT masks. No training, OCR, TEST or segmentation inference.
  Three predeclared cumulative variants: horizontal only; horizontal→vertical;
  horizontal→vertical→+45°→135°. No geometry threshold grid or reselection.
- `ASSUMPTION`: reuse all four frozen v1 gates (.35/.7/.5/2.5) in a projected
  direction frame. Only mutual nearest compatible forward/backward neighbours
  are linked, ordered by normalized longitudinal bbox gap, along-axis centroid
  distance and raw ID. This differs from v1's all-compatible-pairs graph.
- `ASSUMPTION`: multi-component groups are locked after each stage. Subsequent
  directions consume only unused singleton components; they cannot merge whole
  horizontal words into vertical blocks. This priority can itself miss a correct
  vertical/diagonal group when its components were consumed horizontally; do
  not claim it is uniformly better. Diagonal axes are fixed±45°, not arbitrary.
- `PAPER_SPECIFIED`: preserve original component union boxes, original pixel
  foreground-area ordering, padded source-image crops and maximum10. The
  neighbour/direction procedure is not established by paper Algorithm1.
- `ASSUMPTION`: estimate each care word's orientation from principal-axis PCA
  of its source GT glyph-pixel centroids, strictly post-hoc. At least3 glyphs
  and eigenvalue ratio>=4 are required; otherwise UNKNOWN. Horizontal/vertical
  tolerance15°; remaining reliable axes are DIAGONAL. These are diagnostic
  bins fixed before the run, not inference parameters or official annotations.
- `UNRESOLVED`: raw connected components are not necessarily letters; single
  glyphs can fragment and different glyphs can touch. Neither component IDs
  nor GT character identity is used to choose groups. Orientation of short,
  ambiguous or single-character strings cannot reliably be inferred this way.
- Frozen baseline reports are read-only; original Algorithm1/v1/v2 and rescue
  remain unchanged. Main comparison includes per-orientation bbox coverage,
  recovered/lost words and top10 losses. Geometry does not establish PARSeq
  correctness, especially for rotated words. User-authorized new code lives in
  `localize/directional_component_neighbors.py`, with runner
  `diagnostics/directional_neighbors_train.py` and config
  `configs/approximate/directional_neighbors_train_v1.json`.
- Completed frozen comparison: prediction v1/H/HV/HVD IoU>=.5 counts are
  568/567/559/560; HVD recovers3 but loses11, all11 H-estimated. Every loss is
  present before top10. HVD MULTIWORD44 vs v1 37; no promotion. On GT128 counts
  are548/546/541/536; on source GT626/626/627/630. UPGRADE is a genuine diagonal
  rescue (.2711→.7491), but V incorrectly merges separate word-scale CCs such
  as Break/glass/case across lines. Connected components are not letters by
  definition. Result SHA256
  `1b35d100def10f2ac6c56a70bda3310888f89e2ed8f73a18cd0693e1c130ffb9`.
- Orientation estimator limitation found during visual QC:750 H,7 putative D,
  0 confident V,92 UNKNOWN. Of the7 putative D, only UPGRADE is clearly diagonal;
  5 horizontal strings have punctuation/shape-induced PCA bias and stylized
  The is ambiguous. Separate `orientation-visual-audit.json` records the review.
  Existing labels/results are preserved; never report7 confirmed diagonal GT
  words or interpret no confident V as proof that UNKNOWN contains no vertical
  text. This post-hoc check does not change inference or selection parameters.

### L-BILINEAR-001 — Binary native-mask upsampling control

- User-authorized TRAIN-only diagnostic; `ASSUMPTION`, not a strict paper
  interpolation requirement. Section3.3 specifies binary-mask components but
  does not establish this resize convention (`UNRESOLVED`).
- Freeze all localizer parameters. Replace only binary128 nearest backprojection
  with CPU float32 bilinear to original image H/W (`align_corners=False`,
  `antialias=False`), then binarize with `>=0.5` before connected components.
  This is NOT interpolation of pre-sigmoid logits or native probabilities.
- Cached prediction hashes and qualified GT128 preprocessing must match the
  prior experiment. GT_FULLRES and native128-only contours are unchanged controls.
  No segmentation inference, OCR, TEST, training, or parameter selection.
- Separate runner `diagnostics/bilinear_mask_backprojection_train.py`, config
  `configs/approximate/bilinear_mask_backprojection_train_v1.json`, artifacts
  `artifacts/diagnostics/bilinear-mask-backprojection-train-v1/`.
  Existing nearest pipeline and reports remain unchanged. Geometry coverage
  must not be called recognition accuracy or Table5 reproduction.

### L-BILINEAR-002 — Higher post-interpolation thresholds

- `ASSUMPTION`: user-requested TRAIN-only sensitivity diagnostic after
  L-BILINEAR-001. Predeclare thresholds .50/.55/.60/.65/.70 on bilinearly enlarged
  cached binary128 masks. This does not change the frozen native U-Net sigmoid
  threshold .5 or interpolate native network probabilities. No new model inference.
- Preserve all localizers/parameters, data and hashes. Reuse .50 proposals only
  after verifying all229 resized mask hashes; reproduce .50 aggregate metrics.
  GT affects evaluation only, not thresholding or grouping. No TEST, OCR, training,
  or automatic promotion. Paper support for this threshold remains `UNRESOLVED`.
- Separate config `configs/approximate/bilinear_threshold_train_v1.json`, runner
  `diagnostics/bilinear_threshold_train.py`, persistent artifacts
  `artifacts/diagnostics/bilinear-threshold-train-v1/`.

### L-BILINEAR-003 — Lower post-interpolation thresholds

- `ASSUMPTION`: follow-up requested by user; predeclared TRAIN-only grid
  .50/.45/.40/.35/.30 on the same bilinearly enlarged cached binary128 masks.
  No change to native probability threshold, segmentation, localizers or OCR.
  Same frozen .50 control, verified hashes, GT-only evaluation and no TEST.
- `UNRESOLVED`: paper section3.3 supplies no post-resize threshold convention.
  This is approximate geometry sensitivity, not recognition accuracy.
- Config `configs/approximate/bilinear_lower_threshold_train_v1.json`;
  same diagnostic runner extended to accept only the two explicitly declared
  grids. Previous higher-threshold runner preserved byte-for-byte in its
  artifact directory as `executed-runner.py`, matching recorded source SHA.
  Existing higher-threshold JSON/CSV and production baseline remain unchanged.
- New outputs `artifacts/diagnostics/bilinear-lower-threshold-train-v1/`.

### L-GROUP-COHERENCE-PROBABILITY-001 — Two independent group diagnostics

- User-authorized approximate TRAIN229/849 diagnostics, not changes to Algorithm1,
  V1, V2, segmentation, PARSeq or routing. Paper section3.3 inspected; the new
  grouping rules are `ASSUMPTION`, not `PAPER_SPECIFIED`.
- Proposal1: V1 pair-edge pool, sorted strongest normalized gap then center-y/IDs.
  Each proposed union is checked for global vertical overlap, centroid spread and
  height compatibility with frozen V1 limits. Core excludes height below
  median/V1 height-ratio; this does not override V1's pairwise small-fragment gate.
  With >=3 positive internal X gaps, largest must be <=1.5 times median of others;
  otherwise no spacing test. Parameters declared before evaluation, no grid.
- Proposal2: independently filter V1 edges using original-resolution nearest
  projection of *real cached native probabilities*. In the positive horizontal
  gap and shared vertical bbox interval require mean>=.2 and minimum column
  maximum>=.2. X-overlap/touch retains the geometric edge. This column statistic
  does not prove a connected path and is NOT trained character affinity.
- `ASSUMPTION`: frozen nearest mask baseline retained to isolate grouping changes;
  no adoption of recent bilinear sweeps. Group union bbox, area sum, pad6/top10.
  GT word annotations never enter inference. GT128 preprocessing/Don't Care and
  GT_FULLRES controls match the prior qualified oracle. GT corridor support is
  binary foreground, not network confidence; no fake probabilities for predictions.
- Local probability artifacts unavailable in the inspected persistent/cache paths.
  CPU report explicitly omits prediction Proposal2 metrics until verified export.
  PAPER exporter is manual CUDA-only, atomic per-image cache/resumable, hashes
  probabilities AND binary masks against historical metadata, records VRAM and
  failures, never accepts mismatched model outputs. No CUDA executed by Codex.
- Config `configs/approximate/group_coherence_probability_train_v1.json`; code
  `localize/coherent_probability_groups.py`; CPU evaluation
  `diagnostics/group_coherence_probability_train.py`; optional manual export
  `diagnostics/run_group_coherence_probability_host.sh`. Persistent outputs under
  `artifacts/diagnostics/group-coherence-probability-train-v1-cpu/` (partial)
  or `group-coherence-probability-train-v1-complete/` (after PAPER export).

### TRAIN-V1-OCR-FAILURE-001 — Frozen OCR errors versus mask/group geometry

- User-requested CPU TRAIN-only post-hoc analysis. `ASSUMPTION`: preserve the
  historical best-IoU crop assignment, require positive intersection for OCR
  availability, exact casefold and RapidFuzz on raw casefold strings. Missing
  crops stay in the849-word denominator. Freeze229 images/1203 crops/502 exact/
  578 fuzzy>=.8. No new segmentation, OCR, threshold/grid, or routing.
- Reconstruct V1 geometry from exact cached mask hashes; require every retained
  bbox and per-word cached text/metric to match the source report. Native pixel
  metrics use qualified GT128 with floor/ceil word-box mapping and true TP/FP/FN,
  not predicted-foreground-count/GT-count. Preserve existing Don't Care rules.
- Report independent axes: observed crop category, native mask recall/precision
  bins, existing original-resolution geometric failure taxonomy. These labels
  describe associations, not proven unique causes. A native word rectangle can
  also contain neighboring glyph foreground. GT component unions and alternative
  crop exact-text checks are strictly diagnostic, not usable inference decisions.
- `UNRESOLVED`: per-word GT_BOX OCR controls are unavailable in accessible cached
  artifacts. Do not fabricate which13 historical GT_BOX errors occurred or label
  an error at IoU>=.5 an intrinsic PARSeq failure. No replacement OCR inference.
- Persist per-word evidence/CSV and at most2 ascending-ID distinct-image examples
  per observed error category. Visual panels show actual RGB crops and native
  GT/prediction masks, no image generation or edits to original data.
- Runner `diagnostics/train_v1_ocr_failure_audit.py`; config
  `configs/approximate/train_v1_ocr_failure_audit_v1.json`; output
  `artifacts/diagnostics/train-v1-ocr-failure-audit-v1/`.
- `ASSUMPTION`: additional post-hoc crop-completeness diagnostic counts original
  annotated glyph-color pixels retained by the actual crop. The .8 descriptive
  retention limit is not an inference parameter. No mask resizing is used for
  this measurement. Among66 non-multiword errors with IoU>=.5,49 omit at least
  one entire annotated glyph;62 clip some glyph pixels. These observations do
  not prove sole causality: clipping also occurs in correctly recognized words.
- Sample108 has care transcription `arlboro`, whereas the crop contains visible
  `Marlboro` including Don't Care foreground. Preserve annotations and scores;
  do not reinterpret this discrepancy as necessarily an OCR hallucination.
- Secondary runner `diagnostics/train_v1_crop_glyph_audit.py`; immutable output
  `glyph-clipping.json` in the same diagnostic directory. No baseline changed.

### TRAIN-CROP-COMPLETENESS-001 — Completeness versus foreign foreground

- `ASSUMPTION`: CPU TRAIN-only post-hoc original glyph-color pixel retention;
  count other care-word pixels separately from Don't Care. An any-pixel foreign
  intersection is a literal diagnostic, not proof of a meaningful word merge.
  Full retention means all annotated pixels, not OCR success; slight clipping
  may be harmless. Measure cached binary prediction support on original glyph
  pixels using the frozen nearest backprojection, not new inference.
- `ASSUMPTION`: compare actual cached V1 crop to stored GT-associated raw-
  component union plus frozen V1 padding. This uses GT and is DIAGNOSTIC_ONLY,
  not a deployable crop selector. Positive glyph support is not proof of adequate
  glyph reconstruction. No threshold selection, parameter change, or OCR run.
- `UNRESOLVED`: no counterfactual crop OCR or per-word GT_BOX control; report
  geometric recovery only, never infer improved recognition accuracy.
- Paper section3.3 inspected: component boxes,padding,area filtering,top10.
  Nothing in this diagnostic changes that path or practical-remap-v1/v2.
  Config `configs/approximate/train_crop_completeness_v1.json`; runner
  `diagnostics/train_crop_completeness.py`; separate persistent output directory.
- Follow-up manually launched PARSeq diagnostic: freeze exactly80 incomplete/
  absent failed V1 words whose diagnostic union retains all glyphs without other
  care/Don't Care pixels. Run only union and GT_BOX crops, reuse V1 text. Selection
  uses GT and prior failures: report conditional counts out of80, never general
  accuracy, pipeline promotion, or a realizable routing improvement. No segmentation.
  Qualified PARSeq weights/runtime verified; per-crop atomic resumable cache;
  standalone batch1 is `HARDWARE_DEVIATION`. Codex executes CPU tests only.
  Launcher `diagnostics/run_recovered_train_crops_host.sh`; outputs in
  `artifacts/diagnostics/train-recovered-crops-parseq-v1/`.
- Follow-up evidence verified locally: selected80 V1 errors yield78 exact with
  recovered crops and78 with GT_BOX; both controls fail on the same two GT words.
  This supports crop-induced errors for78 selected cases, NOT a deployable GT-free
  recovery rule or new Table5 accuracy. Full notes and report hashes are in
  `docs/TRAIN_GROUPING_FAILURES_AND_RECOVERY.md`. Per-word GT_BOX controls remain
  unavailable for the other769 TRAIN words in this diagnostic.

### CONTEXTUAL-OCR-CORRECTION-001 — Frozen small-model post-OCR comparison

- `ASSUMPTION`: user-authorized practical diagnostic, not paper routing. Hold V1,
  cached OCR and full InstructBLIP captions fixed; compare FLAN-T5-base and
  Qwen2.5-0.5B/1.5B under OCR-only,OCR+caption,caption-only. No training or TEST.
- `ASSUMPTION`: immutable revisions,prompts,greedy generation,strict crop IDs and
  explicit invalid-output abstention preserving OCR. GT only in separate CPU
  evaluation; all849 words remain in denominators. Caption-only is lexical-only.
- `HARDWARE_DEVIATION`: separate sequential model loads,batch1,FLAN FP32/Qwen BF16,
  no quantization; VRAM and latency recorded. No model/CUDA run by Codex.
- `UNRESOLVED`: no guaranteed correction accuracy or full-system cost benefit;
  schema validity does not establish truth. No automatic model promotion and
  no substitution into strict/paper-faithful modes.
- Config `configs/approximate/contextual_ocr_correction_train_v1.json`; protocol
  and commands in `docs/CONTEXTUAL_OCR_CORRECTION_TRAIN_V1.md`.

### CONTEXTUAL-OCR-PLAINTEXT-002 — Full TRAIN per-target output protocol

- `ASSUMPTION`: user-authorized practical extension, not the paper's scoring or
  routing. Reuse frozen V1 OCR and captions; request one corrected target as plain
  text with other OCR readings, with/without full caption. No GT in generation.
  Evaluate all 229 TRAIN images / 849 words, including missing predictions, with
  the prior frozen geometric association. No TEST or automatic promotion.
- `ASSUMPTION`: malformed output abstains on that target only; preserve punctuation
  and score any valid one-line narration as produced, without hidden correction.
  Sequential per-target requests may cost more than the previous image-level run.
- `HARDWARE_DEVIATION`: inherit qualified weights, revisions and FP32/BF16 from
  the previous diagnostic; standalone sequential GPU loads and batch1.
- `UNRESOLVED`: correction quality and compute savings remain unmeasured. Scripts
  and CPU unit tests prepared, not executed at the user's request. Previous
  experiment artifacts remain untouched. See `docs/CONTEXTUAL_OCR_PLAINTEXT_TRAIN_V2.md`.

### CONTEXTUAL-OCR-FEWSHOT-003 — TRAIN GT in held-out demonstrations

- `ASSUMPTION`: explicitly authorized supervised in-context examples. Four fixed
  OCR/GT demonstrations (two keep, two correct) per held-out image fold, selected
  deterministically from other folds only. No weight updates; not annotation-free
  training-free reproduction. Paper3.4/3.5 do not prescribe an LLM corrector.
- `ASSUMPTION`: five image-content-grouped folds, no TEST, full849-word denominator,
  same frozen crop association, OCR and captions. Geometric demo eligibility and
  seeded ordering frozen before GPU execution; no manual example tuning. Same
  Qwen1.5B zero-shot instruction/decoding; input budget2048 accommodates examples.
- `HARDWARE_DEVIATION`: standalone BF16 Qwen1.5B batch1 on PAPER; cached revisions
  and weights verified; latency/VRAM recorded. CPU preparation and evaluation.
- `UNRESOLVED`: held-out demonstration folds do not undo previous TRAIN-informed
  design choices; near-duplicate scene leakage is not ruled out by image hashing.
  No guaranteed gain or automatic promotion. See
  `docs/CONTEXTUAL_OCR_FEWSHOT_TRAIN_V1.md`.

### CONTEXTUAL-OCR-EDIT-GATE-004 — Offline conservative caption-output gates

- `ASSUMPTION`: explicitly requested TRAIN-only offline comparison, Levenshtein
  edit distance<=1 and<=2 on casefold strings. No other normalization; no weights,
  generation, segmentation, crop or routing changes. Reject any output whitespace
  or empty correction, preserving original PARSeq exactly. Case-only changes can
  pass at distance0 but cannot improve casefold accuracy.
- `ASSUMPTION`: controls reproduce V1, unfiltered caption correction, and the
  previously reported single-word fallback (493 lexical matches,269 changed
  crops,303 whitespace rejections). Fail closed on provenance/control mismatch.
- `ASSUMPTION`: exact/recovered/damaged use frozen audit GT-to-best-IoU-crop
  assignment; lexical scores use per-image casefold multisets. No oracle fragment
  association. Difference503-502 is a net difference of distinct metrics, not a
  count of individual assignment errors. GT never enters the edit gate.
- `UNRESOLVED`: TRAIN GO is diagnostic only, not independent generalization or
  automatic promotion. Paper3.5 specifies semantic/lexical routing, not this gate.
  Config `configs/approximate/conservative_ocr_edit_gates_train_v1.json`;
  runner `diagnostics/conservative_ocr_edit_gates_train.py`.

### CONTEXTUAL-OCR-PROMPT-005 — Fixed prompt ablation on cached TRAIN OCR

- `ASSUMPTION`: four predeclared instruction variants versus the cached
  conservative control, OCR-only/with-caption. Same model, examples, fold
  separation, input fields and decoding; no manual rewording after results.
  Per-crop correction, not multi-fragment reconstruction or paper3.5 routing.
- `ASSUMPTION`: apply the same single-word parser/fallback to every variant,
  including reinterpreting cached control outputs. GT used in existing other-fold
  demonstrations and post-generation log/evaluation only. Preserve all849 words.
- `HARDWARE_DEVIATION`: identical standalone Qwen1.5B BF16 batch1; user launches
  GPU generation on PAPER, Codex checks CPU-only. No environment changes.
- `UNRESOLVED`: negative outcomes do not rule out all LLMs; positive TRAIN prompt
  selection is not independent validation. No auto-promotion. See
  `docs/OCR_PROMPT_COMPARISON_TRAIN_V1.md`.

### CONTEXTUAL-OCR-GEOMETRY-006 — Non-oracle ALL_WORDS geometry/caption ablation

- `ASSUMPTION`: compare bbox-only vs explicit pairwise geometry, crossed with
  no caption, cached InstructBLIP, Florence-base, BLIP2-ViTL. Same frozen V1
  observations and zero-shot prompt; no GT grouping, demonstrations or designated
  correct primary reading. No new localization or caption generation.
- `ASSUMPTION`: ALL_WORDS accepts spaces/newlines; invalid/truncated responses
  fall back to joined V1. Preserve raw output; narration detection is heuristic.
  Evaluate image-level lexical multisets, not spatial exact or Table5 accuracy.
- `HARDWARE_DEVIATION`: qualified pinned Qwen1.5B BF16 batch1, manual PAPER GPU;
  CPU-only preparation/evaluation. Record timing, memory and runtime.
- `UNRESOLVED`: filtered/top10 input cannot restore missing observed fragments;
  TRAIN-informed design does not establish generalization. LLM reconstruction
  is not paper section3.5 routing. Older experiments and production unchanged.
  See `docs/GEOMETRY_CONTEXT_RECONSTRUCTION_TRAIN_V1.md`.

### PARSEQ-VISUAL-MARGIN-007 — Offline conservative candidate reranking

- `ASSUMPTION`: exact-string cached AR probability for V1 KEEP; absent/case-only
  reference forces KEEP, not a fabricated rank penalty. Same AR top10 alternatives,
  two cached Qwen mean-logprob conditions, user-declared128 margin/beta/tau combinations.
  The post-refinement V1 can have lower AR probability than an alternative.
- `ASSUMPTION`: one selection per crop, strict gain>tau, KEEP ties, no GT in
  selection. GT only for frozen spatial metrics on745 eligible rows out of849.
- `UNRESOLVED`: AR summed and Qwen mean likelihoods are not calibrated; saved
  candidate export lacks full token paths and cryptographically bound runtime.
  TRAIN-selected improvements would require independent validation. Paper3.5 does
  not specify this selector; no routing/default changes and no Table5 claim.
  See `docs/PARSEQ_VISUAL_MARGIN_CONTEXT_GATE_TRAIN_V1.md`.

### OCR-EVIDENCE-BATTERY-008 — Generation with context budgets and beam evidence

- `ASSUMPTION`:40 fixed TRAIN conditions, one output per existing V1 crop. Caption
  token budget, number of context crops, per-crop AR1/3/5/10/20 hypotheses, genuine
  cached confidence, synthetic examples, structured response and stress controls.
  No GT in prompts/neighbor selection; no new boxes, OCR, captions or model training.
- `ASSUMPTION`: prefix truncation, quoted spans and OCR-overlap sentence views are
  GT-free but change context content. They do not identify a pure length effect.
  Strict single-token response schemas with original-V1 fallback; fixed spatial
  and lexical metrics remain distinct. Whole-image beam input still targets ONE crop.
- `HARDWARE_DEVIATION`: pinned Qwen1.5B BF16 batch1, manual GPU; new long-context
  inputs require user-run memory qualification. CPU preflight never launches CUDA.
- `UNRESOLVED`: only700 of1203 crops have cached AR from a previously GT-associated
  subset; probability is not correctness confidence. TRAIN-informed comparison of40
  variants is exploratory, not independent validation or strict paper3.5 routing.
  Original Task2.2-derived recognition GT may omit ignored visible glyphs.
  See `docs/OCR_EVIDENCE_GENERATION_BATTERY_TRAIN_V1.md`.

### OCR-TARGET-BOUND-BATTERY-009 — Target identity and verifiable LLM edits

- `ASSUMPTION`:80 fixed TRAIN-only conditions (8 evidence packages ×10 correction
  mechanisms), same V1 crops/OCR. Paper §3.5 specifies semantic/fuzzy routing, not
  generative OCR correction; these experiments are isolated practical extensions.
  Exact target IDs, original-copy checks, indexed patches, context agreement and
  proposal verification are inference-safe heuristics, not paper-specified rules.
- `ASSUMPTION`: one-token output, casefold edit≤1 external guard, patch distance≤2,
  original character uncertainty<.9 and strict JSON schemas. All gates are frozen
  before generation. Invalid/truncated outputs preserve V1 exactly. Separate raw
  proposal and accepted metrics to expose parser-driven apparent improvements.
- `ASSUMPTION`: greedy output cap128 allows patch schemas (previous battery48).
  Caption budget32, cached target AR10, shared prompt deduplication and no-context
  identity controls. Not every comparison is an orthogonal factor intervention.
- `HARDWARE_DEVIATION`: inherited pinned Qwen1.5B BF16 batch1, manual PAPER GPU
  only. Record per-request peak allocated/reserved VRAM and OOM errors; CPU
  preparation and evaluation must not run model inference.
- `UNRESOLVED`: TRAIN-informed 80-way exploratory selection is not independent
  validation. Historical AR availability covers only700/1203 GT-associated crops.
  A GT-independent prompt does not remove this selection bias. No Table5 claims.
  See `docs/OCR_TARGET_BOUND_BATTERY_TRAIN_V2.md`; original scientific paths unchanged.

### DS-005 — Additional-dataset segmentation target construction

- `PAPER_SPECIFIED`: Section 4.1 trains/evaluates segmentation on ICDAR13 FST,
  COCO-TS and Total-Text. It uses ICDAR2015 only as a recognition benchmark and
  does not use CTW1500.
- `ASSUMPTION`: the practical extension rasterizes ICDAR2015 word quadrilaterals
  and CTW1500 text-line polygons as filled foreground. These targets are not
  equivalent to ICDAR13/Total-Text glyph-level pixel masks.
- `REPOSITORY_SPECIFIED`: the acquired Total-Text mirror includes pixel masks;
  JPEG values are canonically thresholded at 128, while official ignore polygons
  define zero-weight validity regions.
- `ASSUMPTION`: out-of-image polygon coordinates are clipped to the valid pixel
  extent and counted. EXIF orientation is normalized only when raw JPEG geometry
  disagrees with the annotation geometry.
- `UNRESOLVED`: Table 3 does not establish whether the authors trained one model
  per dataset or a joint model. Three separate runs mirror the ICDAR13 protocol;
  a fourth joint run is separately labeled `PRACTICAL_APPROXIMATION` and uses
  deterministic domain-balanced oversampling. None may be relabeled strict.

## M-SCENE-V1-001 — Cross-dataset practical-remap-v1 recognition diagnostic

- **ASSUMPTION / PRACTICAL_APPROXIMATION:** The ICDAR13-TRAIN-frozen
  `practical-remap-v1` graph parameters are applied unchanged to predictions
  from the separately trained CTW1500, ICDAR2015, Total-Text and balanced
  three-dataset checkpoints. No TEST data or OCR output selects parameters.
- **Evaluation boundary:** ICDAR2015 and Total-Text use post-hoc care-word
  annotations, exclusive axis-aligned polygon envelopes, Unicode casefold-only
  exact comparison, and deterministic maximum one-to-one matching requiring
  bbox IoU >= 0.50. These choices are not established by the paper and results
  must not be labeled Table 5 reproduction.
- **UNRESOLVED / fail-closed:** The frozen local CTW1500 mirror contains
  detection-only text-line polygons. A separate qualifier now accepts only the
  upstream `ctw1500_e2e_annos.zip`, requires its 1,000/500 transcription files,
  and cross-checks every physical instance against the frozen mirror geometry.
  Physical inspection found eight stale TRAIN count headers/eight absent
  recognition geometries, six missing transcriptions, and 403 TRAIN `###`
  flags absent from the detection mirror. No missing text/geometry is invented;
  the six unverifiable text instances are excluded and all discrepancies are
  persisted in qualification metadata. Until
  that physical archive passes qualification, CTW1500 may report proposal
  geometry and PARSeq output counts but no recognition accuracy. After it
  passes, recognition remains text-line-level rather than word-level and is an
  out-of-paper `ASSUMPTION`, not Table 5.
- **Implementation:** `scripts/evaluate_scene_text_v1_parseq.py` freezes
  segmentation/localization shards before loading GT for scoring, runs PARSeq
  only after releasing the segmenter, and supports restart without repeating
  completed image or crop inference.

## M-SCENE-ALG1-002 — Algorithm 1 crops on additional TEST datasets

- **PAPER_SPECIFIED:** Section 3.3 and Algorithm 1 require connected text
  regions, true component-area filtering and descending ordering, padded
  bounding boxes, original-image crops, and at most ten retained blocks.
- **REPOSITORY_SPECIFIED:** `Amin=150` and padding `p=6` are the only frozen
  upstream numeric values. The paper does not state either value.
- **ASSUMPTION:** This isolated additional-dataset diagnostic applies both
  numeric values in the native `128x128` prediction frame, uses foreground
  pixel count rather than bounding-rectangle area, 8-connectivity, a
  top-to-bottom/left-to-right equal-area tie break, and floor-lower/ceil-upper
  remapping of the already padded box to the original RGB image. This follows
  the requested native-mask interpretation but is not established by the
  paper, whose Algorithm 1 denotes `M` as `H x W`.
- **Evaluation boundary:** CTW1500 remains text-line level; ICDAR2015 and
  Total-Text are word level. GT is loaded only after masks, boxes, and crops
  are frozen and is used solely for a geometry diagnostic. No PARSeq, caption,
  context routing, or DeepSolo inference occurs in this milestone.
- **Classification:** `APPROXIMATE_PAPER_ALGORITHM1`; never strict reproduction
  or a Table 5 result. Blocking ambiguities remain `A-008`, `L-001`, `L-003`,
  and `L-005`.
- **Recognition follow-up:** `scripts/evaluate_scene_text_algorithm1_parseq.py`
  runs the qualified PARSeq checkpoint on these frozen crops. Identical
  image/bbox outputs from the matching V1 run may be reused after provenance
  checks. Case-sensitive and casefold exact counts use one-to-one bbox matching
  at IoU>=0.50 with all care GT instances in the denominator. This is an
  explicit diagnostic matching rule, not an established Table 5 protocol;
  CTW1500 remains text-line level. Every OCR result is stored per crop so a
  stopped GPU run can resume without repeating completed recognitions.

## M-TOTALTEXT-WORD-REGIONS-001 — Curved word-level target and fusion penalty

- **PAPER_SPECIFIED:** The paper trains an attention-gated U-Net for pixel-level
  text segmentation and localizes connected mask components. It does not
  prescribe filled, word-level polygon targets or an inter-word fusion loss.
- **REPOSITORY_SPECIFIED:** The prepared Total-Text baseline uses the distributed
  `groundtruth_pixel` mask. That baseline and its checkpoints are unchanged.
- **ASSUMPTION / PRACTICAL_APPROXIMATION:** A separate Total-Text TRAIN target
  fills each official word polygon as a curved ribbon, resolving overlapping
  polygons by nearest exclusive-pixel ownership (deepest-interior seed if a
  polygon has no exclusive core). A black 8-neighbour contact separator keeps
  different word labels disjoint in the source raster. The per-word owner map
  is transformed alongside the target; a second 8-neighbour separator is
  enforced after native `128x128` resizing, because bilinear thresholding
  otherwise reconnects different words. The ordinary variant
  uses masked BCE; the fusion-penalty variant gives separator pixels four
  times the ordinary BCE weight, normalized by total effective weight. The
  auxiliary separator also includes short inter-word background gaps within
  one native-pixel radius, without changing the foreground target. No GT
  transcription enters target generation or training.
- **Risk:** Annotation polygons overlap in many source images, so enforcing a
  gap removes some annotation foreground. Resize to `128x128` and bilinear
  thresholding can also erase narrow words or change their topology. Source
  and transformed-target disjointness are enforced, but prediction-level
  separation is not guaranteed. Evaluate target coverage/topology and held-out word localization
  before claiming any improvement. The weight 4.0 is a fixed diagnostic
  choice, not a paper parameter or a tuned optimum.

## M-TOTALTEXT-WORD-REGIONS-002 — Pixel-guided curved boundary refinement

- **PAPER_SPECIFIED:** The paper uses pixel-level text targets but does not
  specify a word-level curved-ribbon construction or a target-contour fitting
  procedure.
- **ASSUMPTION / PRACTICAL_APPROXIMATION:** An isolated `refined-v2` target
  keeps the v1 disjoint word-owner map and the official Total-Text source pixel
  GT. For each owner, all source pixel-GT foreground inside its region is
  retained. A Euclidean-distance envelope around those pixels is grown within
  the owner's polygon, using at least one native-128 pixel of source-scale
  radius and then the smallest radius yielding one connected 8-neighbour
  region. If no pixel-GT seed exists or the owner's region is disconnected,
  it falls back to the full v1 owner and records the case. The resulting
  raster boundary has many contour points and follows curved text more tightly
  than densifying unchanged polygon edges would. A deterministic native-128
  topology guard restores a v1 word owner if refinement increases its
  connected-component count or erases it; if neighbouring owners' bilinear
  interactions prevent a per-word restoration, the complete image falls back
  to its v1 target. This guard uses only TRAIN targets, never TEST or OCR.
- **Invariant:** Refined regions are subsets of v1 per-word owners; different
  words cannot overlap. The same post-resize owner separation, validity masks,
  architecture, training split and optimizer are preserved. Neither GT text
  nor TEST annotations enter target construction. Evaluate native-128 word
  connectivity and pixel-GT coverage before promoting this target for GPU
  training. It must never overwrite the v1 or paper-faithful configurations.

## M-TOTALTEXT-WORD-REGIONS-003 — Filled adaptive word-box target

- **PAPER_SPECIFIED:** Algorithm 1 localizes connected foreground regions with
  contours, area filtering, padding and a ten-block cap. The paper does not
  prescribe filled word-level TRAIN targets or a method to reshape Total-Text
  annotation polygons.
- **ASSUMPTION / PRACTICAL_APPROXIMATION:** A separate v3 TRAIN target starts
  from the already disjoint v1 word owners. Within each elongated owner, broad
  longitudinal strips (4–12) use the official TRAIN pixel mask to set their
  transverse bounds; every strip is filled solid, with at least 70% of the
  owner's typical thickness. Candidates retaining under 65% of owner area,
  losing any original pixel-GT seed, or worsening source/native-128 component
  topology revert to v1. Near-square/no-seed words also remain v1. This is
  deliberately a multi-corner bounding ribbon, not the v2 letter-distance
  envelope. The v1 owner/ignore/separator rules, training network and paper
  localization path are untouched. No OCR/TEST data influence construction.
- **Evidence:** TRAIN qualification in
  `artifacts/totaltext-adaptive-word-boxes-v3/qualification.json` and
  `qualification_native128.json` covers all 1255 images / 9285 care words;
  5999 boxes change before ten native-topology restorations, all 39,942,201
  covered original pixel-GT seeds survive, and no native word topology
  worsens. This demonstrates label-shape safety, **not** model improvement.
- **UNRESOLVED:** Whether training on these fuller adaptive boxes improves
  held-out word proposals and PARSeq recognition over v1 is untested; strict
  paper fidelity remains blocked by the paper's target-construction ambiguity.
- **ASSUMPTION / PRACTICAL_APPROXIMATION:** The user's subsequent 500-epoch
  v3 training request deliberately departs from the author-clarified 100
  epochs. It has its own protocol name and run directory; the interrupted
  earlier v3 run is retained. No 500-epoch result should be presented as an
  exact paper training run or as evidence of better held-out recognition until
  it is evaluated on the same frozen TEST protocol.

## M-TOTALTEXT-WORD-REGIONS-004 — Curved filled-box correction after v3 rejection

- **PAPER_SPECIFIED:** The paper uses a binary segmentation target and
  contour/bounding-box localization, but does not specify curved word-level
  TRAIN target construction.
- **ASSUMPTION / PRACTICAL_APPROXIMATION:** The user's requested replacement
  keeps a filled word polygon rather than the stepped v3 strip mask or the
  letter-like v2 envelope. The two long sides of each existing Total-Text
  curved annotation define a centerline and local transverse direction.
  Shape-preserving cubic interpolation and a low-frequency upper/lower
  envelope of the official TRAIN pixel GT produce two continuous curved
  boundaries; the entire space between them is filled. If needed, a uniform
  geometric buffer expands the whole polygon to retain source text, never
  individual glyph contours. The v1 owner/validity/native-topology checks
  trigger a per-word or whole-image v1 fallback when necessary.
- **Boundary:** This is a separate TRAIN-only target; no TEST/OCR data or
  transcription is used to fit curves. V1, v2 and v3 artifacts remain intact.
  The rejected v3 500-epoch runner must not be repurposed for v4. A new model
  should be trained only after visual and native-target qualification, then
  evaluated on the held-out recognition protocol before any accuracy claim.
- **ASSUMPTION / PRACTICAL_APPROXIMATION:** The isolated v4 configuration now
  permits the user-requested 500-epoch training, using the qualified v4 TRAIN
  manifest rather than v3's. This is a duration deviation from the clarified
  100-epoch protocol. The chained TEST diagnostic freezes the selected checkpoint
  before native-128 contour extraction and PARSeq. Its bbox-area filter in native
  coordinates, one-to-one IoU>=0.5 word matching, and casefold metric are
  explicit implementation/evaluation assumptions, not resolved Table 5 rules.
- **TRAIN-only geometric oracle (2026-09-25):** Replacing predictions with the
  exact no-augmentation native-128 v4 TRAIN target gives 8,935/9,285 words with
  IoU>=0.5 from *all unpadded raw contour boxes*, but only 3,972 after the
  frozen bbox-area filter, 3,931 after top-10, and 847 after native padding=6.
  The padded candidate has 2.595× the tight box area at the median. This is
  `PRACTICAL_APPROXIMATION`/`DIAGNOSTIC_ONLY`, not U-Net accuracy or Table 5.
  The paper does not resolve whether `Amin` and padding were intended in the
  native-128 or original-image coordinate frame; these numbers do not justify
  changing the strict path or choosing a new setting on TEST.
  Report: `artifacts/diagnostics/totaltext-v4-perfect-mask-localization-train-v1/results-full-audit.json`.

## M-TOTALTEXT-CURVED-CROP-001 — Component-shaped RGB cutout with zero padding

- **PAPER_SPECIFIED:** Section 3.3 / Algorithm 1 crops the original image with
  padded rectangular component boxes. It does not specify a curved RGB cutout.
- **ASSUMPTION / PRACTICAL_APPROXIMATION:** In a separate TEST diagnostic, each
  retained native-128 contour is mapped to its single 8-connected component.
  Its binary support is bilinearly enlarged to original RGB size and thresholded
  at 0.5. The tight, zero-padding rectangular RGB crop preserves pixels inside
  that curved support and replaces off-component pixels with white, as required
  for rectangular PARSeq input. An unchanged tight RGB box is the paired control.
  There is no geometric dewarping, morphology, grouping, or change to PARSeq.
- **REPOSITORY_SPECIFIED / ASSUMPTION:** The isolated experiment keeps the
  repository-style native-128 unpadded *bbox-area* filter `Amin=150`, stable
  bbox-area sorting and top-10. `pad=0` is the user's explicit diagnostic
  request. These are not the strict paper component-area semantics.
- **GT use:** The TEST GT-v4 branch is an oracle built from official TEST
  annotations using the frozen v4 target builder and the qualified TEST
  pixel-extent clipping/Don't Care protocol. It is not deployable inference.
  The predicted branch reuses verified epoch-0458 TEST native masks; both use
  identical proposals for curved and tight-box crop comparisons.
- **CPU preparation:** 300 TEST images / 2,204 care words. GT-v4 oracle:
  1,030 retained zero-pad proposals and 1,030 one-to-one IoU>=0.5 matches.
  Frozen predicted masks: 875 proposals and 518 IoU>=0.5 matches. PARSeq
  recognition requires the separate PAPER CUDA command; geometry is not OCR.
  Prepared artifact: `artifacts/diagnostics/totaltext-v4-curved-component-crops-test-v1-qualified/prepared.json`.

## M-TOTALTEXT-V1-CONTOURS-TEST-001 — Contour localization with the trained v1 target

- **PAPER_SPECIFIED:** Section 3.3 describes connected regions/contours,
  component filtering, padding, area ordering, top-10 and original-image crops.
- **REPOSITORY_SPECIFIED:** The frozen repository contour helper uses
  `RETR_EXTERNAL`, `CHAIN_APPROX_SIMPLE`, unpadded bounding-box area for the
  minimum-area test, clipped padded-box area for sorting, `Amin=150`, and
  padding 6.
- **ASSUMPTION / PRACTICAL_APPROXIMATION:** The diagnostic applies these
  numeric values in the native 128×128 frame, then maps boxes to original RGB;
  segmentation is the separately trained Total-Text v1 word-polygon model.
  PARSeq runs on each retained crop. Word-level one-to-one matching at
  bbox-IoU>=0.50 is used only for post-hoc TEST reporting. No result from this
  diagnostic is a Table 5 claim or a strict paper reproduction.
- **State:** Runner is prepared and provenance-checked; PAPER CUDA execution
  is required to produce the TEST recognition report.

## M-SCENETEXT-GT-NOFILTER-001 — User-requested no-filter GT oracle diagnostics

- **PAPER_SPECIFIED:** Section 3.3 / Algorithm 1 requires component-area
  filtering and a maximum of 10 text blocks. Disabling either changes the
  scientific protocol, so the strict paper path and prior reports remain intact.
- **ASSUMPTION / PRACTICAL_APPROXIMATION:** For future GT-oracle diagnostics on
  Total-Text, ICDAR2015, and CTW1500, the user requested no area filter and no
  top-10 cap. This is recorded in
  `configs/approximate/scene_text_gt_oracle_no_filters_v1.json`. Current
  Total-Text v4 mask-resolution tests also retain the established zero padding.
- **Evidence:** On Total-Text TEST-300, GT-v4 native128 has 2,204 raw contours
  for 2,204 care words. With no area filter/cap, 2,119 GT words have a
  one-to-one box IoU>=0.5; repository-style bbox-area `Amin=150` retains 1,033,
  and top-10 retains 1,030. Thus the filter, not absent GT foreground, is the
  main reason the filtered GT-v4 crop coverage is near 47%.
- **Alternative:** Keep frozen area150/top10 to compare with earlier runs; that
  remains available as the separate `frozen` mode. No result from no-filter
  GT-oracle work may be relabeled a strict Table 5 reproduction.

## M-SCENETEXT-PRED-BIN-UPSCALE-NOFILTER-001 — Predicted-mask resolution diagnostic

- **PAPER_SPECIFIED:** Section 3.3 / Algorithm 1 filters components by area and
  limits crops to ten; the paper does not establish binary-mask upscaling as an
  alternative native segmentation output.
- **ASSUMPTION / PRACTICAL_APPROXIMATION:** For the frozen Total-Text v4 TEST-300
  prediction cache, compare contours at 128/256/512/1024 by bilinearly
  upscaling the *already thresholded* native-128 binary mask and re-thresholding
  at 0.5. Disable area filtering and the top-ten cap; retain zero padding.
  This follows the user's requested diagnostic and is isolated in
  `configs/approximate/totaltext_v4_predicted_binary_resolution_no_filters_v1.json`.
- **Scientific limitation:** No new high-resolution U-Net features or logits
  are produced; this measures interpolation and contour-coordinate effects,
  not the effect of retraining a 256/512/1024-output segmentation model.
  The strict paper path and previous filtered diagnostics are unchanged.

## M-TOTALTEXT-V4-HIGHRES-TRAIN-001 — Native-256 practical segmentation ablation

- **PAPER_SPECIFIED:** Sections 3.2 and 4 describe an attention-gated U-Net with
  pretrained ResNet-50, Adam at `1e-4`, weight decay `0.01`, batch size eight,
  BCE loss, flips and rotations; inputs are 256×256 and masks are 128×128.
- **ASSUMPTION / PRACTICAL_APPROXIMATION:** The isolated Total-Text v4 study
  instead uses 512×512 RGB and 256×256 native targets/logits. The target is
  bilinearly reduced from the qualified full-resolution v4 word-region mask
  and binarized at 0.5. Its purpose is to test whether *new native detail*,
  unlike interpolating an existing 128×128 prediction, improves word-region
  separation. It does not change the strict or existing v4 model.
- **ASSUMPTION:** A SHA256-determined 125-image subset of the official 1,255
  TRAIN images is held out for checkpoint selection, using validation macro
  foreground IoU and earliest-epoch tie-break. TEST is not read for training or
  selection. All three arms use the same split and 500-epoch budget.
- **ASSUMPTION:** Three predeclared arms isolate boundary weighting and skip
  information: baseline separation weight `1` and attention floor `0`,
  separation weight `3` and floor `0`, then weight `3` and floor `0.25`.
  The floor adds a residual contribution from a low-attention skip without new
  parameters; it is not a paper-specified gate. Alternative weights/floors and
  validation rules remain untested, rather than chosen from TEST results.
- **Diagnostic-only validation:** Held-out TRAIN word-owner maps may measure
  predicted component splits/fusions and bbox overlap after checkpoint
  selection; they are not used as model input, grouping rules, or supervision
  for variant selection. GPU memory and the complete resolved run contract are
  recorded. See `configs/approximate/totaltext_v4_highres_ablation_v1.json`.

## M-TOTALTEXT-V4-HIGHRES-PADDING-TEST-001 — Native-256 contour/padding TEST diagnostic

- **PAPER_SPECIFIED:** Section 3.3 / Algorithm 1 extracts connected contours,
  pads their boxes, crops original RGB, filters by component area, sorts by
  component area and limits to ten blocks. The paper does not settle the
  coordinate frame in which padding is measured when mask and RGB resolutions
  differ.
- **ASSUMPTION / PRACTICAL_APPROXIMATION:** The isolated diagnostic uses the
  three completed Total-Text v4 native-256 models, a fixed `sigmoid>=0.5`
  threshold and `cv2.findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)` directly
  on the true native 256×256 binary output. Padding `{0,2,4,6,8,12}` is
  measured in native pixels before floor/ceil remapping to original RGB.
  There is no mask upscaling, morphology, grouping or curved crop.
- **ASSUMPTION / user-requested deviation:** Total-Text area filtering and
  top-ten truncation are disabled, consistent with earlier no-filter oracle
  diagnostics but not with strict Algorithm 1. Every contour remains a crop.
- **Evaluation only:** Official TEST-300 GT is used after predictions and OCR
  are frozen for one-to-one bbox-IoU>=0.5 word matching. Exact-word identities
  are intersected across padding values to measure complementarity. Because
  this is a multi-padding TEST comparison, it MUST NOT be used to choose a
  deployment padding or claim a paper Table 5 result; a final choice requires
  separate TRAIN selection. Configuration:
  `configs/approximate/totaltext_v4_highres_contour_padding_test_v1.json`.

## M-TOTALTEXT-V4-IMAGE-FREQUENCY-TRAIN-001 — Isolated practical input ablations

- **PAPER_SPECIFIED:** Section 4.1 specifies horizontal flips and random
  rotations during segmentation training, with 256×256 RGB input and 128×128
  target. Section 3.2 specifies the attention-gated ResNet-50 U-Net, but no
  photometric augmentation, aspect-preserving resize, Fourier augmentation or
  wavelet features.
- **ASSUMPTION / PRACTICAL_APPROXIMATION:** Four independent arms reuse the
  qualified Total-Text v4 512→256 target, the frozen 1130/125 TRAIN split,
  500-epoch optimizer/loss, separation weight 3 and skip floor 0.25. The arms
  are: moderate RGB-only photometric perturbation; paired centered letterbox
  preserving aspect ratio; TRAIN-donor low-frequency Fourier-amplitude mixing;
  and a fixed Haar-detail adapter on the first decoder skip. These are not
  paper-faithful segmentation configurations. The Haar adapter adds 192
  trainable parameters, leaves the pretrained RGB encoder unchanged and starts
  from a zero residual contribution.
- **Reproducibility / evaluation:** Validation never receives image-only
  perturbations. Fourier donors are restricted to the 1130 TRAIN samples;
  masks, validity, separation and word-owner maps are never perturbed by color
  or Fourier operations. Letterbox geometry is identical on all paired maps,
  with the padded target area invalid. Checkpoint selection remains held-out
  TRAIN foreground IoU, matching the prior high-resolution ablation; per-word
  proposal IoU, one-to-one precision/recall/F1 and component split/fusion
  counts are separate decisive diagnostics. Frozen pad-0 PARSeq exact is
  evaluated on the same held-out TRAIN 125 only after checkpoint selection.
  No TEST result may be used to choose an arm. Exact parameters and their
  scientific classifications are in
  `configs/approximate/totaltext_v4_image_frequency_ablation_v1.json`.
- **ASSUMPTION / user-requested shorter budget:** The frozen configuration and
  checkpoint contracts remain at 500 epochs, preserving the completed
  photometric run and the resumable letterbox checkpoint. Future execution
  explicitly stops at epoch 100 and records both configured and actual budgets
  in its result manifest. This saves compute but makes the 100-epoch arms
  non-comparable as a controlled 500-epoch intervention ablation; it is never
  a strict paper result. No TEST data or OCR result determined the stop epoch.
- **ASSUMPTION / numerical diagnostic:** Training-stage word geometry uses
  eight-image validation batches, whereas the practical pad-0 PARSeq crop
  audit performs single-image segmentation inference. At a hard probability
  threshold, these may produce different connected-component geometry even
  for identical input tensors. The crop audit retains its immutable batch-one
  masks/OCR and only accepts a difference if a deterministic batch-eight
  checkpoint replay reproduces the frozen training geometry exactly per image.
  Any unreproduced difference remains an explicit failure, not a relaxed
  tolerance or paper-reproduction claim.
- **ASSUMPTION / frozen practical TEST follow-up:** The TRAIN-125 pad-0 PARSeq
  diagnostic selected the letterbox arm before consulting official Total-Text
  TEST-300 labels. The selected 100-epoch practical checkpoint, zero native
  padding, disabled component-area filter and disabled top-ten cap are pinned
  in `configs/approximate/totaltext_letterbox_pad0_test_v1.json`. Those crop
  choices differ from Algorithm 1 in the paper and must be reported as a
  `PRACTICAL_APPROXIMATION`, never as Table 5 recognition accuracy. TEST is
  used only for post-hoc one-to-one word evaluation, not further selection.
- **ASSUMPTION / threshold sensitivity after TEST:** The paper specifies a
  binary segmentation mask but not the sigmoid cutoff for this practical
  Total-Text letterbox network. A predeclared seven-value cutoff sensitivity
  study runs only on the frozen TRAIN-125 validation split and must reproduce
  the original threshold-0.50 masks and scores exactly. Other cutoffs are
  post-hoc diagnostics of crop and word-identity changes, not a new strict
  parameter choice; the official TEST-300 result remains fixed at 0.50.
- **ASSUMPTION / matched letterbox 500-epoch continuation:** The completed
  100-epoch letterbox run retained only its best epoch-71 checkpoint, including
  Adam and RNG; epoch 100 cannot be resumed exactly. Two new branches restore
  the identical epoch-71 state and continue to epoch 500, with separation
  weights 3 (control) and 5 (one predeclared stronger boundary penalty). Weight
  5 is a user-requested practical intervention, not paper-specified or tuned on
  TEST. The original 100-epoch result remains unchanged. Concurrent use of one
  RTX 3090 is a `HARDWARE_DEVIATION`; per-process VRAM peaks and errors must be
  recorded before comparison.
- **ASSUMPTION / post-hoc word audit of fixed 500-epoch checkpoints:** The
  completed weight-5 branch selected its original epoch-71 checkpoint, whose
  model tensors were verified identical to the source; its PARSeq OCR may be
  reused exactly. The weight-3 branch selected epoch 263 and requires a new
  TRAIN-125 batch-one crop/PARSeq audit. The audit compares per-word recovered
  and lost GT identities with fixed threshold 0.5, pad 0, and no contour
  filtering, without loading TEST or using OCR to choose a checkpoint.
- **ASSUMPTION / practical threshold transfer:** A separate threshold-0.65
  Total-Text TEST-300 diagnostic uses the already completed TRAIN-125 sweep
  (`613957db...`) to freeze the cutoff before that run. It uses the unchanged
  epoch-71 letterbox checkpoint, no area/top-ten filter, and pad 0. The
  existing threshold-0.50 TEST report remains untouched. Because the same
  TRAIN validation population had already been inspected in other ablations,
  this is an exploratory practical transfer check, not an unbiased strict
  paper result or a new TEST-tuned parameter search.
- **ASSUMPTION / word-F1 checkpoint selection:** Two isolated 100-epoch
  Total-Text TRAIN-only runs use the frozen 1130/125 split and the same
  letterbox images and word-v4 targets. Unlike the paper's unspecified
  checkpoint rule, they rank epochs by held-out native-256 proposal F1 at
  IoU 0.5, then fewer GT words touched by fused components, then word recall,
  pixel IoU, and earlier epoch. Pixel IoU remains reported, not optimized as
  the sole selection criterion. TEST is not loaded during model development.
- **ASSUMPTION / separator-head ablation:** One of those runs retains the
  ResNet-50 attention U-Net and adds a 65-parameter second output channel for
  the existing TRAIN inter-word separator mask. Balanced positive/negative
  separator BCE is added with weight 0.25 to the unchanged weight-3
  foreground BCE. At inference, pixels with separator probability >=0.5 are
  removed from foreground probability >=0.5 before native connected
  components. The matched control has no second head and uses the identical
  word-F1 checkpoint rule. This is `PRACTICAL_APPROXIMATION`, not the
  paper-specified single-channel binary U-Net. Concurrent execution on the
  RTX 3090 is a `HARDWARE_DEVIATION` and requires memory-peak reporting.

## TOTALTEXT-CROP-CLIP-TRAIN-001 — Crop variants and frozen CLIP selector

- **Status/classification:** `ASSUMPTION`, approximate Total-Text TRAIN-125
  diagnostic only. This is separate from the paper's Algorithm 1, MPNet/BLIP-2
  scoring and DeepSolo fallback and cannot enter `STRICT` configuration.
- **Paper evidence:** Section 3.3/Algorithm 1 specifies connected components,
  component-area filtering, padding, area sorting and at most ten original-image
  crops. It does not fix the mask threshold, padding amount or a CLIP selector.
  The frozen letterbox baseline and its threshold 0.5/pad 0/no-filter contour
  path are `REPOSITORY_SPECIFIED`; its omission of area filtering/top ten means
  the result is a practical approximation of Algorithm 1.
- **Experiment choice:** Reuse the seven previously declared TRAIN thresholds
  0.35–0.65 and test native padding 0/2/4/8. Match alternative threshold
  contours to baseline anchors one-to-one at tight-native IoU 0.3. CLIP ViT-B/32
  receives each original RGB crop and its PARSeq text either directly or in a
  quoted-word prompt. Compare cosine scores with 0/0.02 margin gates, PARSeq
  confidence, a shuffled-text control and a GT-assisted one-choice-per-anchor
  oracle. Grids, matching, prompts, gates and selectors are `ASSUMPTION`.
- **Metric choice:** One-to-one bbox IoU at least 0.5 plus casefold exact word
  matching over all 983 care GT words is `REPOSITORY_SPECIFIED` for this frozen
  Total-Text TRAIN baseline. It does not establish the paper's Table 5 metric.
- **Partial check:** All 125 TRAIN validation samples passed input SHA and
  frozen-baseline crop checks. There are 1,173 baseline anchors and 19,392
  distinct candidates. All 4,878 pad-0 candidates have existing PARSeq OCR.
  On these pad-0 threshold variants, baseline is 368/983; PARSeq-confidence
  selection is 403/983 (47 recovered, 12 damaged); the GT-assisted anchor
  ceiling is 440/983 (74 recovered, 2 damaged). This is a threshold-only
  diagnostic and says nothing yet about padded-crop accuracy or CLIP selection.
  A one-image Total-Text CPU PARSeq smoke decoded 84 candidate crops, including
  63 newly read padded variants; its result class is `SMOKE_TEST`.
- **Full TRAIN check:** The pinned ViT-B/32 snapshot
  `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` scored all 19,392 crops.
  Best predeclared CLIP arm (quoted-word prompt, threshold-only, margin 0.02)
  reached 374/983, recovering 20 baseline errors while damaging 14 correct
  words. PARSeq confidence on threshold variants reached 403/983. The
  GT-assisted one-choice-per-anchor ceiling with threshold and padding
  variants reached 462/983; adding padding raises the threshold-only ceiling
  by 22. Every padding-inclusive CLIP arm was far below the baseline. Full
  per-candidate report SHA256:
  `101baf43e5758aa309af6059b10b6c5c792128eb7f9219fc2a2eb4b3d5ba6bc4`.
- **Unresolved scientific question:** Frozen CLIP showed only a +6-word gain
  in its best predeclared TRAIN arm and a large loss when padding candidates
  were included. Whether that small gain transfers to a frozen independent
  TEST rule is `UNRESOLVED`. CLIP4STR trains recognition branches and CLIPTER
  fine-tunes fusion; neither establishes this zero-shot selector.
- **Hardware:** Sequential PARSeq and float32 CLIP loading is a
  `HARDWARE_DEVIATION` for the target 8 GB RTX 3070. The full run used one RTX
  3090 with batch size 16: peak CLIP allocated/reserved VRAM was 724.6/792.0
  MiB; PARSeq was 135.9/164.0 MiB. The two models loaded sequentially. These
  peaks do not qualify execution or output equivalence on the target RTX 3070.
- **Protocol:** `configs/approximate/totaltext_crop_clip_battery_train_v1.json`,
  `docs/TOTALTEXT_CROP_CLIP_BATTERY.md`. The runner hashes images, probability
  arrays, model artifacts, code and crop pixels. GT is only used after candidate
  generation and model scoring, apart from the labeled oracle ceiling.

## TOTALTEXT-FEATURE-WAVELET-SKIP-001 — Decoder-gated frequency skip

- **Status/classification:** `ASSUMPTION`, isolated `PRACTICAL_APPROXIMATION`.
  Paper Section 3.2 specifies attention-gated encoder skips, but not a wavelet
  transform or feature-frequency fusion. The strict AG-UNet remains unchanged.
- **Evidence and alternative:** The earlier fixed Haar-on-RGB /2-skip adapter
  degraded Total-Text TRAIN validation word geometry and PARSeq. This new
  ablation instead takes three fixed Haar detail bands from the frozen ResNet-50
  /2 feature map and gates their projected residual using decoder features.
  A zero-initialized scalar gain preserves the unmodified skip at initialization.
  This may suppress background edges, but that benefit is `UNRESOLVED` until
  measured; frequency detail alone does not establish word separation.
- **Frozen experiment:** Same qualified v4 word target, letterbox transform,
  1130/125 TRAIN split, seed, optimizer, loss and macro-foreground-IoU
  checkpoint selection as the earlier letterbox run; exactly 100 epochs. The
  pre-existing TRAIN-selected threshold 0.65 is a comparison point, not a
  parameter selected using TEST. Native contours, pad 0, no area filter and no
  top-ten cap are practical choices, not paper Algorithm 1.
- **Protocol:** `configs/approximate/totaltext_frequency_gated_skip_100ep_v1.json`.
  The run contract pins model/trainer/source hashes and the split; checkpoint
  loading is strict. The next OCR validation must compare on the same 125 TRAIN
  images before any TEST evaluation.

## TOTALTEXT-WORD-KERNEL-001 — Instance nuclei and seeded expansion

- **Status/classification:** `ASSUMPTION`, isolated `PRACTICAL_APPROXIMATION`.
  Paper Sections3.2/3.3 specify a binary AG-UNet foreground and component crops,
  not nuclei, instance identities or expansion. The paper-faithful path stays
  unchanged; no Table3/5 exact-reproduction claim is made.
- **Evidence:** Current letterbox geometry and PARSeq failures include merged
  adjacent words. [PSENet](https://arxiv.org/abs/1903.12473) establishes the
  kernel/expansion approach for arbitrary-shaped scene text. Our two-map
  distance-core target and simultaneous BFS are adaptations, not PSENet.
- **Declared choices:** Unmodified qualified v4 full targets and ignore masks;
  native per-owner half-max interior-distance core, largest connected fragment;
  one additional output channel; existing foreground loss plus balanced core
  BCE weight1; full threshold0.65, nucleus0.5; deterministic8-neighbour expansion
  restricted to full prediction; unseeded full-CC fallback; per-instance boxes
  with pad0/no area/no top10. All are frozen before execution and GT-blind at
  inference. GT owner IDs are only targets and held-out evaluation references.
- **Alternatives/risk:** Multi-kernel polygon offsets would be closer to PSENet
  but add target/output choices. A separator head already failed to improve OCR.
  Predicted missing/multiple nuclei and fragmented foreground can still lose or
  split words; small-word target survival and cross-word separation are checked
  on all TRAIN images before training. Final instance labels must NOT be
  collapsed into a binary mask before crop extraction.
- **Protocol:** `configs/approximate/totaltext_word_kernel_100ep_v1.json`,
  exactly100 epochs on frozen TRAIN1130/validation125. Selection is validation
  word-F1, with fusion/recall/IoU/earliest tie-breaks. No TEST or OCR is used in
  training/checkpoint selection. Benefit remains `UNRESOLVED` pending the GPU
  run and matched held-out PARSeq evaluation. Details: `docs/TOTALTEXT_WORD_KERNEL.md`.

## TT-CROP-RECOVERY-POOL-V1 — Unrestricted crop hypotheses and oracle recall

- **Scope:** User-authorized approximate Total-Text crop recovery suite on the
  qualified held-out TRAIN-125 population. Paper sections 3.2/3.3 and Algorithm
  1 establish binary masks and connected localization (`PAPER_SPECIFIED`).
  The existing native-256 letterbox checkpoint, split, caches and qualified
  PARSeq runtime are `REPOSITORY_SPECIFIED` for this diagnostic, not sufficient
  evidence for STRICT reproduction.
- **New choices (`ASSUMPTION`):** Multi-threshold components and seed-guided
  geodesic partitions, erosion at thickness-relative radii, RGB/probability
  minimum-cost seams, overlapping-window inference, short adjacent partition
  unions, asymmetric/symmetric border shifts, median-color margins and PCA
  orientation. Parameters are explicit in the isolated approximate config.
  No GT transcription or geometry enters generation or recognition. No
  baseline-anchor restriction, global area/top10 cap or NMS is applied.
- **Evidence and alternatives:** The earlier TRAIN mask diagnostic found word
  fusions to dominate localization failures. GT separator removal was an oracle
  causal probe and must not enter this deployable generator. Increasing a global
  threshold traded fewer fusions against missing foreground. The new suite
  preserves multiple alternatives for later selector research; it does not
  assume each seed, eroded core or seam segment is a complete word.
- **Metric (`ASSUMPTION`):** Exact Unicode casefold plus P/S removal, source
  bbox IoU >= .5, maximum one-to-one matching over the entire crop pool. Each
  care word counts once, and all original care words remain in the denominator;
  normalized nonempty GT is reported separately. Rotated crops use enclosing
  source bboxes and artificial margins do not enlarge their spatial footprint.
  This is oracle pool recall, not final recognition precision/accuracy. The
  strict IoU gate may reject some correctly read unusual crops.
- **Hardware (`HARDWARE_DEVIATION`):** Sequential float32 segmentation/PARSeq
  stages, segmentation batch one, PARSeq batch 16, optional CPU execution.
  Per-stage peaks and propagated OOM errors are recorded. Reused qualified OCR
  can originate from different device/batch settings; crop/model/runtime hashes
  and upstream execution identity are preserved. Device-output equivalence is
  not established by cache reuse.
- **Unresolved:** Independent-data transfer, optimal crop budget, curved
  rectification, numerical equivalence on RTX 3070 and the eventual GT-blind
  selector remain `UNRESOLVED`. No Table 5 or strict reproduction claim.
- **Protocol:** `configs/approximate/totaltext_crop_recovery_suite_train_v1.json`,
  `docs/TOTALTEXT_CROP_RECOVERY_SUITE.md`; entry point
  `bash scripts/test_totaltext_crops.sh`. Generated artifacts stay outside the
  repository and source/model/data contracts are hash checked on resume.

## TT-CROP-WORD-GROUPS-V1 — Group threshold/window alternatives by word hypothesis

- **Scope:** User-authorized approximate Total-Text held-out TRAIN-125 grouping
  of original baseline/threshold-tree/tiled-segmentation crops. Inspected paper
  sections 3.2/3.3 and Algorithm 1 establish mask/connected localization
  (`PAPER_SPECIFIED`) but do not specify cross-view crop grouping. Qualified
  source probability/crop/PARSeq caches are `REPOSITORY_SPECIFIED` diagnostic
  inputs; no new model inference or strict localization change.
- **New choices (`ASSUMPTION`):** Global-RGB bbox IoU and scale-relative center
  compatibility; deterministic complete-link core grouping; ordering by distinct
  threshold/view support; median-box-nearest references; containment/disjointness
  tests for sharing a composite crop across multiple groups under one original
  candidate ID. Transitive graph connectivity is an explicit negative control.
  Parameters live in `configs/approximate/totaltext_crop_word_groups_train_v1.json`.
  No OCR text, confidence or annotation enters group construction. Every original
  candidate is retained; choosing the final reading is deferred.
- **Evaluation (`ASSUMPTION`):** Frozen normalized exact IoU>=.5 witnesses,
  maximum flow with unit group/crop/GT capacities, geometric core-pair purity
  only on uniquely GT-labelled candidates, and fragmentation of correct witnesses
  across groups. GT enters after group artifacts are saved. Shared membership
  cannot multiply a single recognition output in the oracle. Group counts include
  background fragments and parent alternatives, not just true words.
- **Evidence/alternatives:** On 18,912 cached threshold/window crops (782/983
  unrestricted oracle), graph connectivity at IoU .45 preserves only 698 under
  one-reading-per-group capacity. Complete link at .45 preserves 780 with 7,711
  groups; at .55 it preserves 781 with 8,683 groups. These are same-TRAIN
  comparisons, not independent tuning validation. Precise details and failures:
  `docs/TOTALTEXT_CROP_WORD_GROUPS.md`.
- **Unresolved:** Overlapping curved/vertical boxes (including TOMATO/KETCHUP
  in TRAIN image45), geometry-indistinguishable alternatives, fragmented seeds,
  high group counts from background hypotheses, exact-mask branch associations,
  independent-data transfer and the eventual GT-blind selector. Containment is
  only a hypothesis about multiword crops. No Table 5 or strict claim.
- **Execution:** Deterministic CPU float64 geometry/cache evaluation, no new
  model execution or hardware-driven numerical change. Upstream GPU/precision
  provenance remains frozen. Entry point: `bash scripts/test_totaltext_groups.sh`.

## TOTALTEXT-PARTIAL-ENCODER-001 — matched late-encoder fine-tuning

- **Classification:** ASSUMPTION, APPROXIMATE/PRACTICAL_APPROXIMATION only.
  Paper section3.2/Fig3 uses pretrained ResNet50 but does not establish this
  schedule; the current author protocol explicitly freezes encoder/BN.
- **Intervention:** Separate class enables gradients only for layer3/4 non-BN
  parameters. AuthorProtocolAGUNet and all prior practical variants stay unchanged.
- **Frozen comparison:** Both branches start at qualified letterbox epoch71,
  identical strict weights/source RNG, fresh Adam, 50 additional epochs, decoder
  LR1e-5 and encoder LR1e-6. BN affine/statistics frozen. Split1130/125, native256,
  unchanged v4 targets/loss/attention and threshold0.65/pad0/no area/top10.
- **Selection:** Predeclared validation native word-F1 with fusion/recall/fgIoU
  tie-breaks, unchanged checkpoint as epoch0 eligible. No TEST/OCR in epoch selection.
  Final held-out PARSeq125 reports original-coordinate geometry and exactness.
- **Hardware:** Sequential FP32 GPU branches/recognizer, recorded batch8 versus
  batch1 evaluation and peak allocated/reserved VRAM (HARDWARE_DEVIATION).
- **UNRESOLVED:** Does adapting high-level features improve word separation and
  correct PARSeq words over frozen-control and letterbox baseline? No benefit claimed.
- **Protocol:** `docs/TOTALTEXT_PARTIAL_ENCODER.md`; isolated practical training
  config validates new LR without relaxing the existing paper-valued validator.

## CROP-GROUP-WORD-SELECTION-V1 — Frozen evidence selectors and CLIP

- **Scope:** User-authorized selection experiments on original threshold/window
  crops and previously frozen word-hypothesis groups. Explicit APPROXIMATE TRAIN
  diagnostic; no strict reproduction or Table 5 claim. Paper sections 3.3/3.5
  and Algorithm 2 were inspected with the current recognition/CLIP runtime.
- **Paper evidence (`PAPER_SPECIFIED`):** Localized crops and PARSeq-S recognition
  are paper components. These selectors, cross-crop likelihoods, grouping and
  CLIP are not specified by the reproduction paper. Its caption/MPNet/DeepSolo
  path and strict configuration are unchanged.
- **Existing inputs (`REPOSITORY_SPECIFIED`):** Qualified original crop/PARSeq
  caches, per-token probabilities, whole/tile probability contracts, pinned
  model artifacts, image manifests and existing complete-link groups supply
  diagnostic inputs. The default adapter uses the completed TRAIN-125 fixture.
  Upstream grouping .55 was compared on that fixture and is not independently
  validated; no new grouping tuning occurs in this experiment.
- **Selectors (`ASSUMPTION`):** Length-normalized/minimum/balanced confidence;
  consensus capped per originating view; probability-grid interval stability;
  soft-mask coverage/purity relative to a fixed geometric group reference;
  equal geometric-mean combinations. No fitted weights, confidence threshold,
  dictionary, preferred word length or GT-dependent selection is introduced.
  Zero mask support has an explicit geometric area fallback. Confidence is not
  interpreted as calibrated correctness probability.
- **Cross-PARSeq (`ASSUMPTION`):** Candidate-prefix causal teacher forcing with
  frozen qualified weights; raw text plus EOS, PAD excluded, length normalized;
  equal representative-crop contributions. Five distinct readings and three
  original crops per group are resource budgets, fixed before evaluation.
  No-EOS greedy strings beyond scoring capacity receive UNSUPPORTED_LENGTH
  records and zero cross score, never truncation; other selectors retain them.
  This architectural case caused an initial full-run failure, whose artifact
  remains preserved. Other runtime/contract/OOM errors still fail explicitly.
- **CLIP (`ASSUMPTION`):** Four separate arms compare each normalized candidate
  with the same full-image or fixed group-reference embedding, using raw text
  or `a photo of the written word "{text}"`. Median-RGB square canvases preserve
  the field of view before pinned official preprocessing. Raw cosine is saved;
  `(cosine+1)/2` supplies a dimensionless common assignment score. Model revision
  and every local snapshot file are recorded and incomplete loading is rejected.
- **Assignment/evaluation (`ASSUMPTION`):** Maximum total score, one output per
  group, one use per original crop ID, repeated spellings allowed on distinct
  crops, zero-score abstention, deterministic exact-score ties. All inference
  artifacts are frozen before GT evaluation. Exact Unicode casefold/P-S removal,
  IoU>=.5 and one-to-one matching retain all care GT in the denominator. Predictions,
  noise/unmatched counts, baseline recoveries/losses and frozen group/shortlist
  oracles are reported; group count is not word count.
- **Hardware (`HARDWARE_DEVIATION`):** Sequential model loading, float32 batch16,
  optional CPU execution. CPU real-model smoke/full stages are instrumented.
  CUDA allocated/reserved peaks and device/model/shape/batch metadata are logged
  when CUDA runs; synthetic OOM propagation is tested. Actual RTX3070/OOM
  qualification and numerical equivalence across devices remain UNRESOLVED.
- **Alternatives/evidence:** Native probability product without length
  normalization, one vote per crop and candidate-dependent CLIP images were
  rejected because of length/correlation/image-comparison confounds. Primary
  sources are PARSeq (arXiv:2207.06966), STR calibration (arXiv:2012.12643),
  CLIP (arXiv:2103.00020), plus pinned official PARSeq runtime code. They support
  model/scoring mechanisms and cautions, not efficacy of these experimental
  selectors. No selector parameter changes follow measured evaluation results.
- **Unresolved:** Independent-dataset transfer, upstream grouping transfer,
  calibrated confidence, threshold stability on incomplete grids, inaccurate
  masks/references, semantic versus literal CLIP matching, correlated views,
  background/duplicate group rejection and final end-to-end precision.
- **Protocol:** `configs/approximate/crop_group_word_selection_v1.json`,
  `docs/CROP_GROUP_WORD_SELECTION.md`; command
  `bash scripts/test_crop_group_selectors.sh`. Full per-image evidence/results
  and resolved software/code/model/data provenance stay outside the repository.
- **Measured evidence:** Complete CPU TRAIN-125 run
  `/tmp/crop-group-word-selection/0d7113261cc3/results.json`; all 14 selectors
  executed, 59 unit/regression tests passed. Baseline 400/983, soft-mask/confidence
  732, cross-PARSeq 691, best full-image CLIP 640, best group-crop CLIP 670; group
  oracle 781, five-word shortlist oracle 778. Eight of 27773 cross pairs were
  explicitly unscorable-length cases. Highest word count still emits 7522 outputs
  (exact precision 9.73%); noise rejection is unresolved. Full negative comparisons
  are preserved in the protocol document. No post-evaluation parameter change.

## CROP-GROUP-WORD-SELECTION-TEST-V1 — Frozen rules on official TEST

- **Scope:** User-authorized full TEST launcher, with original threshold/window
  crop generation, PARSeq, fixed word groups and all 14 existing selectors.
  Explicit APPROXIMATE_TEST_DIAGNOSTIC, or SMOKE_TEST when limited; no strict
  reproduction or Table 5 claim. Paper section 3.3/Algorithm 1 was re-inspected.
- **Evidence (`REPOSITORY_SPECIFIED`):** Existing qualified TEST manifest
  (300 images, 2204 care words), selected TRAIN letterbox checkpoint, strict
  checkpoint loading/qualification, PARSeq source/checkpoint, pinned local CLIP
  snapshot and original annotation parser/coordinate clipping.
- **Protocol (`ASSUMPTION`):** Transfer frozen TRAIN threshold-tree/window grids,
  reference grouping complete_link/.55 and generic selector rules unchanged.
  No grouping/selector fitting or parameter search on TEST. Disable auxiliary
  erosion/seam generators and final crop variations in this dedicated path;
  original TRAIN generator defaults remain unchanged. Generate actual float32
  whole/window probabilities, never reconstruct probabilities from binary masks.
  Reuse only hash-checked TEST source shards and their crop OCR cache.
- **Evaluation (`ASSUMPTION`):** All 2204 care words retained in the full
  denominator; casefold/P-S removal, IoU>=.5, one-to-one GT-instance matching.
  Freeze original source crop/OCR artifacts before source annotations are opened,
  and all selector predictions before selector evaluation. Report per-method
  correct/prediction counts, baseline losses/recoveries, noise and group oracle.
- **Hardware (`HARDWARE_DEVIATION`):** Sequential float32 segmentation, PARSeq
  and CLIP, segmentation batch one and recognition batch16; optional CPU. Record
  shapes, device/software/backend flags, model/code/data hashes and CUDA peaks.
  Actual RTX3070 execution/OOM behavior remains UNRESOLVED.
- **Verification:** 63 focused/regression tests and real pinned-model CPU TEST
  smoke through every stage. The first TEST image supplies 92 crops/207 cross
  pairs; its zero-recovery oracle is not a full TEST accuracy result. Initial
  grouping-summary failure on an undefined mean was corrected to explicit n/a;
  inference, model-contract and OOM errors still propagate. Full TEST performance
  remains UNRESOLVED until the complete official evaluation is run.
- **Protocol:** `bash scripts/test_totaltext_test.sh`;
  `configs/approximate/totaltext_crop_group_test_suite_v1.json`,
  `configs/approximate/totaltext_crop_word_groups_test_v1.json`,
  `docs/CROP_GROUP_WORD_SELECTION.md`. Outputs remain outside the repository in
  `/tmp/totaltext-crop-group-test-suite/<hash>/selectors/`.
- **CLIP channel correction (`ASSUMPTION`/correctness):** The official image
  processor inferred a 3×3×3 PIL RGB group crop as channels-first. The GPU TEST
  attempt failed at `total_text:999` after all 300 source/OCR/group/cross-score
  shards and 239 CLIP image shards were completed. The runtime now declares
  `channels_last`. Real pinned-model CPU scoring of that exact image completed
  95 groups/129 readings with finite scores; 65 focused/regression tests passed.
  All prior CLIP scores are discarded because tiny crops can be misprocessed
  without an exception. A pinned source-runner hash plus source/input/model/
  device/shard contracts qualify reuse of the complete TEST crop/OCR source;
  exact old/new runtime hashes and per-pair contracts qualify the 70,657 frozen
  cross-PARSeq pairs. The failed run remains intact. Twelve TRAIN references
  also have tiny canvases, so historical TRAIN CLIP counts need a corrected
  rerun before exact comparison. Selection parameters were not changed. Full
  corrected TEST/GPU results remain UNRESOLVED until executed.

## TT-WORD-ABLATION-001 — four practical word-training interventions

- **Evidence (`PAPER_SPECIFIED`):** Local paper section3.2/Fig3 and implementation
  details establish ResNet50 attention U-Net and BCE/Adam. They do not establish
  pixel links, instance-balanced loss, word-anchored crop augmentation or decoder
  deep supervision.
- **Decision (`ASSUMPTION`):** Separate practical branches and matched control,
 50 additional epochs from qualified best frozen-control epoch23; same1130/125
  TRAIN split, native256, threshold.65 and pad0. All new values are declared in
  `configs/approximate/totaltext_word_ablation_50ep_v1.json`. Strict/default
  segmentation, Algorithm1 and prior practical experiments remain unchanged.
- **Targets (`ASSUMPTION`):** Existing v4 owner maps supply per-word positive
  weights, same-word neighboring pixel labels and TRAIN crop anchors. Cut words
  become invalid during zoom augmentation. Owner/GT inputs never enter inference.
  Auxiliary heads supervise unchanged native targets via upsampled logits.
- **Hardware (`HARDWARE_DEVIATION`):** FP32/batch8 training, batch1 segmentation
  OCR validation, sequential models/branches on one GPU; memory peaks and errors
  recorded. No dependency installation or Codex CUDA execution.
- **Evaluation (`ASSUMPTION`):** Native validation word-F1 selection with source
  epoch0 eligible; OCR only after selection, original bbox IoU>=.5 one-to-one,
  all983 validation words retained, casefold and Unicode P/S-stripped rates.
- **Outcome (`UNRESOLVED`):** GPU effectiveness and runtime/OOM behavior have
  not yet been measured. No TEST tuning, automatic promotion or Table5 claim.
  See `docs/TOTALTEXT_WORD_ABLATION.md` for algorithm/parameter provenance.

## CROP-GROUP-POSTFILTERS-V1 — Frozen output suppression and abstention

- **Scope/evidence:** User-authorized two-method experiment on the completed
  crop-group TEST selections. Paper section 3.3 and Algorithm 1 specify padded
  contour crops, component-area filtering and a top-10 cap, but no cross-crop
  grouping, nonmaximum suppression or group abstention. This remains an
  `APPROXIMATE_TEST_DIAGNOSTIC`, not a strict paper or Table 5 result.
- **REPOSITORY_SPECIFIED inputs:** Original selected `mask_completeness_confidence`
  outputs, source crop IDs/origins/readings, complete-link group core/shared
  memberships, and hash-checked whole/tile soft probability maps are reused
  without model inference or changed PARSeq transcriptions.
- **ASSUMPTION / duplicate rule:** Greedy score ordering suppresses a lower
  ranked output only when both exclusive-box IoU and soft foreground-mass IoU
  are at least 0.5. The standard geometric halfway overlap is declared in
  approximate config, not fitted to TEST. An explicitly shared parent crop
  preserves groups with different readings; zero mask support falls back to
  geometric IoU. Scores are for crop choice, not calibrated group validity.
- **ASSUMPTION / abstention rule:** A selected group survives if a core crop is
  from the original baseline or two distinct core RGB hashes agree on its
  normalized reading. Shared multiword parents do not provide independent core
  evidence. There is no word-length rule, lexicon, score cutoff, count-per-image
  budget, or annotation input. Correlated near-identical crops and false baseline
  regions can still survive; valid isolated words can be rejected.
- **Evaluation boundary:** Every postfilter selection shard is frozen before any
  GT is opened. The existing one-to-one casefold/P-S-normalized exact matcher
  with axis-aligned IoU>=0.5 reports a local precision/recall/Hmean proxy only.
  Official polygon spotting evaluation and cross-dataset transfer are
  `UNRESOLVED`. Earlier TEST diagnostics motivated the structural abstention
  study, so this is exploratory. No numerical overlap cutoff was fitted and
  neither rule was revised after its full TEST postfilter result.
- **Measured full TEST result:** On 300 images/2,204 care words, input 18,052
  predictions/1,722 correct; duplicate suppression 15,462/1,637; abstention
  6,509/1,600; combination 5,667/1,548. Proxy Hmean respectively 17.00%,
  18.53%, 36.73%, 39.33%. The combination loses 174 of the frozen input's
  correct words, so it is not a free precision gain.
- **Protocol/artifact:** `configs/approximate/crop_group_postfilters_v1.json`,
  `docs/CROP_GROUP_POSTFILTERS.md`, `bash scripts/test_crop_group_postfilters.sh`;
  `/tmp/totaltext-crop-group-test-suite/30643625ace4/postfilters-d7fee644accb/results.json`.

## GT-WORD-GROUP-SELECTOR-V1 — One oracle group per annotated word

- **Scope/evidence:** User-authorized counterfactual TEST diagnostic with exactly
  one group per GT word. The reproduction paper's section3.3/Algorithm1 specifies
  contours, area filtering, padding and a top-10 cap (`PAPER_SPECIFIED`), not
  GT-defined group membership. This is an `ORACLE_TEST_DIAGNOSTIC`, never a
  deployable, strict or Table5 result.
- **REPOSITORY_SPECIFIED inputs:** Frozen 47,538 original crop/reading candidates,
  aligned probability maps and the existing `mask_completeness_confidence`
  selection rule. Each source/image/mask contract is hash checked; no model
  inference or new OCR occurs.
- **ASSUMPTION / oracle grouping:** Each care GT yields one group. A crop enters
  every group whose original-coordinate axis-aligned GT box has IoU>=0.5 with
  the crop box. The group reference is chosen from member crop geometry by
  median-box medoid, so GT coordinates determine membership but do not enter
  the selector's completeness score. One original crop and one group each have
  capacity one; an empty GT group remains in the denominator. The fixed IoU
  comes from the existing local metric, not a TEST-fitted cutoff.
- **Measured 300-image TEST:** 2,204 GT groups, 1,997 with a spatial candidate,
  1,823 with an exactly correct crop, and a one-per-group/crop oracle ceiling
  of 1,823. The unchanged completeness/confidence selector chooses 1,997
  outputs with 1,659 correct for their assigned GT; PARSeq-confidence control
  chooses 1,525. There are 207 groups without a crop, 174 with crops but no
  correct OCR witness, and 164 reachable groups where the primary selector
  misses the correct crop. Global one-to-one matching yields 1,661 rather than
  1,659 because two outputs cross-match nearby GTs.
- **Limitations (`UNRESOLVED`):** Predicted grouping and official polygon
  spotting evaluation are different tasks. Previous TEST diagnostics motivated
  this oracle question, so transfer is untested. The experiment isolates crop
  choice under GT-supplied membership; it does not repair the 381 words lacking
  an exact crop witness in the frozen pool.
- **Protocol:** `configs/approximate/gt_word_group_selector_v1.json`,
  `docs/GT_WORD_GROUP_SELECTOR.md`, `diagnostics/gt_word_group_selector.py`,
  `bash scripts/test_gt_word_group_selector.sh`;
  `/tmp/totaltext-crop-group-test-suite/30643625ace4/gt-word-groups-420282ed5f05/results.json`.

## TT-STRUCTURAL-GROUPS-V1 — Proposal lineage, mask footprint, boundaries and admission

- **Scope (`ASSUMPTION`):** User-authorized grouping experiment on frozen
  Total-Text threshold/window crops. Paper section 3.3 and Algorithm 1 specify
  connected contours, component-area filtering, padding, sorting and a ten-block
  cap (`PAPER_SPECIFIED`); they do not specify cross-threshold lineage,
  cross-window grouping, seed-run boundaries or group admission. These four
  ablations are approximate diagnostics and cannot support a strict/Table 5
  reproduction claim.
- **Source (`REPOSITORY_SPECIFIED`):** Reuse the qualified frozen 300-image TEST
  source with 47,538 original crops, 1,823/2,204 local correct-word oracle,
  original PARSeq outputs and saved whole/tile probability maps. Regenerate
  proposal masks using an instrumented generator, verifying each original crop
  ID, bbox, method and origin before evaluation. Instrumentation exposes
  component/seed ancestry without changing crop recipes or model outputs.
- **Rules (`ASSUMPTION`):** Complete-link merges of geometrically corresponding
  source masks; 256-cell common image projection for whole/tile footprints;
  local principal-axis seed-gap boundary with a frozen PARSeq concatenation
  check; sharing a multiword parent among child groups; and an image-relative
  foreground-mass admission gate. All comparison bounds are stored in the
  resolved plan. They are structural hypotheses, not paper parameters or
  learned/calibrated dataset constants. The first two stages use masks only;
  the third and fourth also use frozen PARSeq readings. No GT enters grouping.
- **Alternatives considered:** Plain connected-box transitive merging lost
  23/782 recoverable TRAIN words at IoU 0.55 in the earlier grouping diagnostic;
  a baseline-only anchor omits many recoverable TEST words; unconditional
  singleton removal would discard some valid isolated word hypotheses. These
  motivate complete-link constraints, tiled views and the isolated-word escape.
- **Evaluation (`ASSUMPTION`):** Same frozen casefold/Unicode P-S-stripped exact
  text and axis-aligned IoU>=0.5 one-to-one metric as prior crop/group
  diagnostics. Report group oracle retention, fragmentation, mixed core groups,
  selected correct words and outputs for every stage. Save per-image group and
  evaluation artifacts before aggregate claims.
- **Independence (`UNRESOLVED`):** A three-image TEST smoke run exposed a generic
  proximity-only false-merge risk; the OCR concatenation guard was added before
  the complete run. Numerical comparison bounds were not fitted to TEST, but
  the full TEST outcome is exploratory, not a clean independent validation of
  the guard. Transfer to another benchmark and official polygon spotting
  metrics remain unmeasured.
- **Measured full TEST:** Original complete-link grouping: 21,284 groups,
  1,823 one-per-group oracle words, 1,722 selected correct from 18,052 outputs.
  Lineage: 18,360 / 1,822 / 1,546 / 15,243. Add cross-window footprints:
  17,343 / 1,820 / 1,507 / 14,298. Add boundary/parent handling: 16,596 /
  1,751 / 1,396 / 13,572. Add image-relative admission: 9,629 / 1,732 /
  1,379 / 9,533. Order in each tuple is groups / one-per-group oracle /
  selected correct / selected outputs. Geometrically mixed core groups rise
  from 16 initially to 238 after the boundary stage; its false merges are a
  material failure. The first two stages largely preserve crop recoverability
  but require a selector designed for larger hypothesis groups.
- **Protocol:** `docs/TOTALTEXT_STRUCTURAL_WORD_GROUPS.md`,
  `localize/structural_word_groups.py`,
  `scripts/analyze_totaltext_structural_groups.py`,
  `bash scripts/test_totaltext_structural_groups.sh`;
  `/tmp/totaltext-crop-group-test-suite/ba82c525eadb/source/structural_groups/443e094771fb/results.json`.

## TT-CROP-GROUP-REFINEMENTS-V1 — Robust reference and conservative output grouping

- **Paper evidence (`PAPER_SPECIFIED`):** Section 3.3 and Algorithm 1 specify
  connected-component localization, padding, area ordering and the ten-block
  cap; section 3.5 specifies PARSeq-S for recognition. The paper does not
  specify grouping threshold/window crops, a representative crop, or duplicate
  output consolidation. These refinements cannot be used as strict reproduction.
- **Frozen inputs (`REPOSITORY_SPECIFIED`):** Reuse the existing TRAIN-125 and
  TEST-300 original crop/PARSeq/whole-and-tile probability artifacts. The TEST
  control is the min-native-side-3 filtered group/selector run. No new model
  inference or checkpoint substitution occurs.
- **New choice (`ASSUMPTION`):** Replace the single geometric medoid as the
  mask-support target with a coordinate-wise inner-quartile envelope of core
  candidate boxes. Shared multiword parents do not define the envelope. The
  unchanged balanced PARSeq confidence and equal geometric-mean combination
  provide the final group selection score. No GT or OCR text enters the
  reference construction.
- **New choice (`ASSUMPTION`):** The guarded variant accepts a robust-reference
  crop switch only when its original PARSeq balanced character confidence does
  not decrease. Conflicting original-crop assignments revert to baseline until
  each crop belongs to at most one chosen group. The rule uses no GT, language
  list or fitted confidence margin; its score is the robust-reference score.
- **New choice (`ASSUMPTION`):** Consolidate final outputs only if bbox
  IoU>=0.9 and their normalized readings are equal or one character apart
  for words of at least four characters. Keep the higher-scoring output.
  This is a second-stage grouping hypothesis; the original core groups and
  group oracle are unchanged. Similar text alone never merges locations.
- **Alternatives considered:** The full core-box envelope yielded 728/983
  correct TRAIN words versus 740 for the inner quartile. Wider output grouping
  at identical-reading IoU 0.75 retained 740/983 TRAIN and 1,742/2,204 TEST
  from 7,255 and 16,162 outputs, respectively: one TEST word was lost. An
  IoU 0.5 bound retained 1,722/2,204 from 15,158 TEST outputs. The sweep is
  in `/tmp/totaltext-crop-group-refinements-v1/output_merge_probe.json`.
  An image-median
  probability admission gate with baseline/two-crop rescue yielded 719/983
  correct from 5,522 outputs, so it was not included in the primary method.
  Applying the robust selector to previously frozen, min-side-3 filtered
  lineage groups produced 16,365 groups and 1,597/2,204 correct TEST words
  (former selector: 1,546); adding cross-window footprints produced 15,364
  groups and 1,552 correct. Both remain below the original-group robust result
  of 1,743. This exploratory cross-run probe is saved in
  `/tmp/totaltext-crop-group-refinements-v1/structural_selector_probe.json`.
  Voting over selected original-group readings inside the lineage groups gives
  1,558 correct TEST words from 14,086 outputs; footprint groups give 1,478
  from 13,156. This negative control is recorded in
  `/tmp/totaltext-crop-group-refinements-v2/lineage_vote_probe.json`.
- **Measured approximate diagnostics:** TRAIN baseline 732/983 correct from
  7,522 outputs; robust reference 740, conservative consolidation 7,456 outputs
  with 740 correct; guard+consolidation 738 correct from 7,462 outputs. TEST
  min-side-3 baseline 1,722/2,204 from 16,846 outputs; robust reference
  1,743, then conservative consolidation 16,659 outputs with 1,743 correct.
  Guard+consolidation yields 1,745 correct from 16,678 outputs. TEST robust
  selection gains 50 and loses 29 individual words versus baseline, net +21;
  guard gains 31 and loses 8, net +23. The group oracle remains 1,823.
- **Unresolved:** Output count remains far above the number of actual words;
  independent-dataset transfer, calibrated word admission, official polygon
  spotting metrics and whether a stronger grouping method can retain the
  1,823-word oracle are unproven. TEST was previously inspected in this project,
  so these are exploratory comparisons, not a clean independent validation.
- **Protocol:** `docs/CROP_GROUP_REFINEMENTS.md`,
  `configs/approximate/crop_group_refinements_v1.json`,
  `recognition/crop_group_refinements.py`,
  `scripts/evaluate_crop_group_refinements.py`;
  `/tmp/totaltext-crop-group-refinements-v2/{train-full,test-full}/results.json`.

## TT-INDEPENDENT-DETECTOR-SLOTS-V1 — DeepSolo word-slot feasibility

- **Paper boundary (`PAPER_SPECIFIED`):** Paper section 3.3/Algorithm 1 localizes
  padded connected components from a segmentation mask. DeepSolo is a fallback,
  not a specified independent detector for grouping PARSeq crops. This is an
  isolated APPROXIMATE feasibility test, not a strict or Table 5 reproduction.
- **Inputs (`REPOSITORY_SPECIFIED`):** Reuse the frozen 300-image Total-Text TEST
  guarded selector with 16,678 outputs and 1,745 local correct words, plus the
  qualified isolated ResNet-50 DeepSolo candidate and its SHA-bound checkpoint.
- **Association (`ASSUMPTION`):** DeepSolo detection polygon envelopes define
  spatial slots. The 0.05 inference score threshold probes recall; the existing
  0.4 config threshold is a post-hoc detector-score control. Crop/slot IoU
  controls 0.1/0.25/0.5 are declared before output inspection. A crop may be
  eligible for multiple slots, while maximum-weight final assignment emits
  at most one crop per slot and uses each original crop once. Equal geometric
  mean of crop score and box IoU is a GT-blind ranking assumption. DeepSolo
  transcription never replaces the frozen PARSeq reading.
- **Hardware (`HARDWARE_DEVIATION`):** The qualified DeepSolo Python 3.8 CUDA
  subprocess runs sequentially and records per-image peak allocated/reserved
  VRAM. The target paper hardware and exact fallback checkpoint are unresolved.
- **Coordinate mapping (`ASSUMPTION`, TT-INDEPENDENT-DETECTOR-SLOTS-EXIF-V1):**
  OpenCV honors EXIF orientation on JPEG decode; the frozen PIL crops and GT
  use the stored raster. Inverse EXIF pixel-index transforms map detector
  polygons back to that raster before box clipping. A finite polygon with an
  empty clipped envelope is counted and excluded; malformed polygons fail.
  One of 300 TEST images has orientation 6. All its 100 detections mapped to
  positive boxes. Across the complete run, 12,469/12,469 raw detections yield
  slots and zero were rejected. The v2 detector shards remain immutable; the
  v3 analysis records the source plan and every shard hash.
- **Orientation effect (`UNRESOLVED`):** Inverse mapping restores the frozen
  crop/annotation coordinate frame while preserving detector outputs. Inference
  on the EXIF-oriented raster is not claimed to be equivalent to inference on
  the stored raster; its effect on predictions for image 651 is unmeasured.
- **Execution (`HARDWARE_DEVIATION`):** The qualified RTX 3090 host completed
  300 DeepSolo GPU inference shards. The CPU-only v3 analysis completed 300
  selection and 300 evaluation shards. Its 2,204-GT-word diagnostic reports
  1,499 correct among 1,886 outputs for score >=0.4 and crop/slot IoU >=0.5;
  local bbox/exact-text Hmean proxy 0.733, not official Total-Text polygon
  Hmean or paper Table 5. The available DeepSolo candidate is a Total-Text
  model-zoo artifact, so cross-dataset generalization remains `UNRESOLVED`;
  CRAFT/DBNet checkpoints are not locally present.
- **Protocol:** `docs/TOTALTEXT_INDEPENDENT_DETECTOR_SLOTS.md`,
  `configs/approximate/totaltext_independent_detector_slots_v1.json`,
  `localize/independent_word_slots.py`,
  `scripts/evaluate_totaltext_detector_slots.py`,
  `bash scripts/test_totaltext_detector_slots.sh`.

## TT-YOLO-IMPROVEMENTS-V1 — exploratory detector/crop/admission battery

- `PAPER_SPECIFIED`: Sections3.3/3.5 crop localized components with padding and
  use PARSeq-S. Inspect these sections and existing direct-YOLO implementation;
  this external detector battery does not enter STRICT mode.
- `REPOSITORY_SPECIFIED`: Reuse the pinned YOLOv8s TextOCR checkpoint, genuine
  qualified PARSeq adapter, frozen original-raster crop receipts and local
  casefold/P-S-stripped one-to-one bbox metric, keeping all2,204 care GT.
- `ASSUMPTION`: Baseline TEST failures motivate families (joint confidence,
  NMS, crop views, resolution/windows). Fixed generic controls are declared
  before new outcomes: joint0.5/0.6; NMS0.5; views-5%/+5%/+15% for score>=0.25;
  whole960; four640 windows overlap20%, tile floor0.25. These are exploratory
  TEST results, not untouched validation or parameters optimized from TEST.
  No labels enter inference, ranking, filtering or view selection.
- `ASSUMPTION`: OCR-confidence view selection retains the original detector
  bbox for evaluation. Sliced inference is a fixed small control inspired by
  SAHI, not an exact SAHI implementation. Joint model scores are uncalibrated.
- `HARDWARE_DEVIATION`: Explicit CPU/GPU choice; sequential YOLO/PARSeq, FP32,
  batch32, threads4 and seed0. CPU measurement cannot qualify target GPU.
- `UNRESOLVED`: Official polygon/ignore-aware Hmean, transfer, one-character
  detection training gaps and erroneous historical transcriptions. The baseline
  counts270 unmatched boxes overlapping ignored annotations; these cannot be
  filtered at inference using GT. Preserve historical metric for comparison.
- Protocol: `configs/approximate/totaltext_yolo_improvements_v1.json`,
  `recognition/yolo_word_refinement.py`,
  `scripts/evaluate_totaltext_yolo_improvements.py`,
  `docs/TOTALTEXT_YOLO_IMPROVEMENTS.md`.

Measured approximate TEST result: all300 images/2,204 care GT complete;
21,991 new genuine PARSeq readings;12 focused tests and genuine smoke pass;
independent matching reproduces all39 controls and checks35,228 pixel hashes.
Original640 raw0.40 gives1,574/2,367, Hmean68.87%; joint0.60 plus identical-word
NMS gives1,597/2,205, P72.43%, R72.46%, Hmean72.44%. Whole960/windows controls
do not improve Hmean. Fixed shrink5%/padding5%/padding15% give1,550/1,565/1,404
correct versus1,574. Confidence-chosen views yield1,616/2,319, Hmean71.46%.
CPU recorded detector/OCR processing is131.89/453.30seconds; no GPU claim.

Additional `ASSUMPTION`: after the first exploratory TEST battery, separately
declare `totaltext_yolo_scale_agreement_v1.json`. Exact normalized text and
IoU0.5 across640/960, one-to-one matching, pair geometric-mean joint score,
threshold0.5/0.6, original640 coordinates, identical-word NMS0.5. This new method
also remains an exploratory TEST analysis and is not independent validation.
Its genuine cached outputs give1,629/2,290, Hmean72.50% at0.50;1,533/1,989,
P77.07%, R69.56%, Hmean73.12% at0.60. Focused unit and independent score/pair/
geometry/matching audit pass. No new neural inference for this extra analysis.

The `joint<=sqrt(detector)` bound derives a pre-OCR cutoff `threshold²`: for
joint0.60,2,564 original detections at>=0.36 yield exactly the same selected
IDs as all8,318 in a cached300-image audit. This is an algebraic necessary
condition, not threshold tuning or a measured optimized-runtime result.
Outputs and evidence: `/tmp/yolo-failure-improvements/v1/cpu/full`.

## TT-YOLO-DIRECT-PARSEQ-V1 — direct object boxes as recognizer inputs

- `PAPER_SPECIFIED`: Sections 3.3/3.5 crop localized image regions and use
  PARSeq-S recognition. Their component localization and padding differ from
  this user-requested detector experiment.
- `REPOSITORY_SPECIFIED`: Reuse the existing qualified official PARSeq artifact
  `bb5792a6...`, source archive `f57289cb...`, autoregressive decoding and one
  refinement iteration, 32x128 RGB preprocessing, frozen Total-Text TEST care
  GT, and casefold/P-S-stripped exact-text one-to-one bbox metric.
- `ASSUMPTION`: Recognize each frozen YOLO box once, with no extra padding;
  evaluate the same predeclared YOLO score0.05/0.25/0.40 controls. No mask,
  original crop groups, recognizer-confidence filter, empty-reading filter,
  or GT selects the new crops. All emitted boxes count as predictions.
- `HARDWARE_DEVIATION`: The restricted session has no CUDA driver, so genuine
  CPU FP32 recognition is explicitly selected. Record device, batch32,
  CPU threads4, seed0 and memory metadata. A separate explicit GPU launch
  remains available and cannot silently reuse CPU shards.
- `UNRESOLVED`: The qualified checkpoint is not proven to be the paper's
  PARSeq-S. The metric is a local axis-aligned bbox Hmean proxy, not official
  Total-Text polygon Hmean. Strict reproduction and target GPU qualification
  remain separate.
- Protocol: `configs/approximate/totaltext_yolo_direct_parseq_v1.json`,
  `scripts/evaluate_totaltext_yolo_direct_parseq.py`,
  `docs/TOTALTEXT_YOLO_DIRECT_PARSEQ.md`.

Measured approximate result: all300 TEST images/2,204 care GT completed on CPU
FP32, reading8,318 detector crops in171.39seconds of recorded OCR processing.
At YOLO scores0.05/0.25/0.40, correct words are1,833/1,688/1,574; output
counts8,318/3,224/2,367; precision22.04%/52.36%/66.50%; recall83.17%/76.59%/71.42%;
local Hmean34.84%/62.20%/68.87%. Empty normalized readings430/77/36 remain
in the denominator. The full independent sparse-matching and pixel-hash audit
passes; eight focused tests, genuine smoke and cached smoke replay pass. No
failed sample is omitted. CPU recognition is a recorded hardware deviation;
official polygon Hmean and GPU comparison remain unresolved. Results and source
snapshot: `/tmp/yolo-direct-parseq/full-cpu-v1`; report SHA256
`cea2c557adcede16d0141d99dc5ced381ebd2f8215631b3c68cf7becb968d3b5`.

## TT-YOLO-WORD-SLOTS-V1 — lightweight object-detection candidate

- `PAPER_SPECIFIED`: Paper section 3.3/Algorithm 1 uses padded segmentation
  components, with DeepSolo only as a fallback later in the pipeline. A YOLO
  object detector assigning one box to each image word is outside STRICT mode.
- `REPOSITORY_SPECIFIED`: Reuse the 300-image frozen Total-Text TEST PARSeq crop
  predictions, one-crop-per-slot GT-blind assignment, exact normalized-text
  bbox metric, and frozen source/manifest contracts from the DeepSolo slot
  diagnostic.
- `ASSUMPTION`: Try the pinned third-party YOLOv8s TextOCR word-box checkpoint
  as a lightweight external-transfer candidate. Use its single `text` class,
  saved Ultralytics `8.1.0` runtime, BGR NumPy input, 640 input, FP32 batch1,
  0.05 recall-oriented confidence floor, NMS IoU0.7,
  maximum300 detections, and predeclared score0.05/0.25/0.4 with crop-slot
  IoU0.1/0.25/0.5. No TEST outcome selects a setting. Git blob SHA1 and file
  size identify exact checkpoint bytes; record local SHA256 on download.
- Measured approximate result: a later GPU-host run completed the frozen 300-image
  Total-Text TEST diagnostic: 2,204 GT words, 8,318 low-score detections,
  2,036 GT box matches at score0.05, maximum 1,690 correct final readings,
  and best predeclared local Hmean proxy0.683 (1,449/2,042 outputs). The
  DeepSolo reference reaches local Hmean0.733 (1,499/1,886 outputs). All
  300 detector/selection/evaluation shard totals match the final report at
  `/tmp/yolo-word-slots/evaluation/full/results.json`.
- `HARDWARE_DEVIATION`: The completed run used RTX 3090, not the target 8 GB
  RTX 3070. FP32 batch1, size640 inference recorded peak allocated
  115,851,776 bytes and reserved144,703,488 bytes. This does not by itself
  qualify the target GPU.
- `UNRESOLVED`: Official polygon Hmean, transfer to other datasets, target
  RTX3070 execution, and short/one-character recall require separate tests.
  Published TextOCR mAP50 is a different metric. The third-party training
  conversion excludes one-character words.
- Protocol: `configs/approximate/totaltext_yolo_word_slots_v1.json`,
  `scripts/evaluate_totaltext_yolo_word_slots.py`,
  `scripts/run_totaltext_yolo_word_slots.sh`,
  `docs/TOTALTEXT_YOLO_WORD_SLOTS.md`. The restricted session's initial
  package attempt failed because DNS resolution failed; the later GPU-host
  run completed with Ultralytics8.1.0 in the separate environment.

## TT-BERT-CORRECTION-V1 — Frozen latest-letterbox PARSeq post-correction

- **Paper evidence (`PAPER_SPECIFIED`):** Local PDF sections 3.3–3.5 and
  Algorithms 1/2 specify component localization, PARSeq-S, BLIP-2 descriptions,
  MPNet/Levenshtein scoring and DeepSolo fallback. They do not specify BERT
  word correction. The user requested this additional experiment on Total-Text
  and explicitly selected the latest letterbox variant. It is an approximate
  extension and cannot enter STRICT or support a Table 5 reproduction claim.
- **Frozen source (`REPOSITORY_SPECIFIED`):** Latest completed single-head
  letterbox word-F1 control, selected epoch 68, checkpoint
  `a3308015aa1b898b1a8ceb03128d6a113bd9c01a01726fba930dd65794c3a552`.
  Reuse 125 TRAIN-holdout images, 949 crops, 983 care words, native256 external
  contours, pad0, no area/top10 filtering. Its report SHA256 is
  `92169d7338e6d5302bdf4ecd96981d2e29c3bcd543d9f23ef8e13320fbf0a1b1`.
  Report/label/crop/checkpoint hashes are checked before preparation. The
  historical OCR cache is an explicit input and must exactly agree where
  duplicated; it does not replace a missing model or modify strict defaults.
- **Model (`ASSUMPTION`):** Use frozen English `google-bert/bert-base-uncased`,
  immutable revision `86b5e0934494bd15c9632b12f734a8a67f723594`, verified in the
  official Hugging Face model API. This is pretrained BERT, not OCR-fine-tuned
  BERT. Salazar et al. (ACL 2020) motivate frozen MLM rescoring; Kauf/Ivanova
  (2023) motivate within-word left-to-right masking. ABINet motivates separate
  visual/language evidence but is not implemented by this component.
- **Alternative investigated (`ASSUMPTION`):** NeuSpell's released
  `murali1996/bert-base-cased-spell-correction` is spelling-fine-tuned, but its
  authoritative snippet uses pooled subword encodings, a 100002-label word
  classifier and an external vocabulary mapping. Loading it as an MLM would
  silently initialize a different prediction head. Initial choice: qualify
  genuine BERT MLM decoding first. NeuSpell requires a separate head/mapping
  adapter and actual OCR transfer qualification; its published English
  spelling results are not Total-Text evidence.
- **Candidates/gates (`ASSUMPTION`):** Preserve PARSeq KEEP; ASCII single-word
  eligibility; protect native minimum token probability >=0.95; maximum 24
  bounded vocabulary/context candidates; Levenshtein budget 1/2 and relative
  budget 0.3. Candidate pools are identical across context conditions. Use
  mean target-only PLL-word-l2r gain minus edit penalty 1.0; require gain>1.5,
  KEEP on ties. These are config hypotheses frozen before BERT inference, not
  paper parameters, fitted thresholds, calibrated confidence or a PARSeq
  posterior over alternatives. No GT-derived lexicon is used.
- **Context/scoring (`ASSUMPTION`):** Three geometric line neighbours per side;
  vertical overlap 0.5 and gap 8 heights. Use frozen original OCR throughout.
  Caption arm requires full image-ID/hash-bound coverage; first80 whitespace
  words are a declared budget. BERT scores target tokens only, masks current
  and future within-word subwords, retains their log scores/sum/count, and uses
  their mean for the gate. The published scoring paper uses sentence sums;
  this mean/target-only adaptation and residual length/domain bias are explicit.
- **Runtime (`HARDWARE_DEVIATION`):** Standalone FP32 BERT with cached upstream
  inference, explicit CPU or CUDA:0, mask-task batch16, maximum256 input tokens,
  seeded deterministic settings. No quantization/offload or automatic device
  substitution. Record model file SHA256, complete loading-key audit, software,
  placement, batch/input bounds, loading/inference times and GPU allocated/
  reserved peaks. Missing MLM/backbone keys or runtime/OOM failures are fatal;
  token overflow/unknown target is a declared, auditable KEEP abstention.
- **Evaluation (`ASSUMPTION`):** Reuse frozen axis-aligned bbox IoU>=0.5
  one-to-one exact Unicode casefold metric with punctuation retained; also
  case-sensitive/fuzzy metrics. Micro denominator includes all983 care GT words;
  unmatched words stay in it. Report recovered and damaged GT sets, per-crop
  decisions and unchanged geometry. This is not official polygon spotting.
- **Verified preparation (`REPOSITORY_SPECIFIED`):** 384/983 exact casefold,
  372 case-sensitive, 475 fuzzy, 503 geometric matches. Of949 outputs,555 pass
  word-shape eligibility and 137 remain after the native-probability gate.
  No BERT output contributed to these counts. 46 focused CPU/regression tests
  pass; tiny random BERT is an execution fixture, not scientific inference.
- **Completed real inference (`REPOSITORY_SPECIFIED`):** The user completed
  the host launcher; pinned BERT weights are now cached. On all125 TRAIN-holdout
  images, word_only obtains381/983 exact (0 recovered,3 damaged,10 changed
  crops); local_ocr obtains383/983 (0 recovered,1 damaged,8 changed crops).
  Original baseline is384/983. All250 shards and aggregate metrics have been
  independently checked. CPU FP32,four threads,batch16; correction-only times
  4.41/4.88s. No failures or sample exclusions. MLM/backbone keys match; only
  the explicitly allowed unused pooler/NSP keys are excluded. Model safetensors
  SHA256:`68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3`.
  Result SHA256:`a7ae0a603323c4d2d6696b7383dbc6c6dd2646432f8200f4e52e8f674ee788ef`.
- **Remaining questions (`UNRESOLVED`):** Benefit of different BERT/spelling
  checkpoints, caption context, broader proper-name/non-English behavior,
  GPU memory and independent TEST transfer. The initial missing-model blocker
  is resolved; these negative TRAIN results do not qualify useful correction
  or reproduce Table5. Caption caches examined are ICDAR13 and must not be
  attached to Total-Text images.
- **Protocol:** `docs/TOTALTEXT_BERT_CORRECTION.md`,
  `configs/approximate/totaltext_v4_bert_correction_v1.json`,
  `recognition/bert_correction.py`,
  `scripts/evaluate_totaltext_bert_correction.py`,
  `scripts/run_totaltext_bert_correction_host.sh`.

## TT-BERT-CONFIDENCE-ABLATION-V1 — Native-confidence protection audit

- **Rationale (`ASSUMPTION`):** After v1 failed to recover any word, a
  post-hoc GT-assisted audit identified36 recoverable GT words already in the
  fixed candidate pools;29 were protected by native token probability>=0.95
  and7 were scored but kept. The one-candidate-per-crop matching oracle is
  420 exact versus baseline384. This oracle diagnoses a constraint; it never
  supplies words to inference or proves a deployable correction rule.
- **Controlled change (`ASSUMPTION`):** A separate approximate config changes
  only `policy.protect_min_token_probability` from0.95 to1.0, retaining KEEP
  for exactly saturated values. Source crops, code, candidate pools, model,
  revision, precision, all edit/score/context bounds and denominator stay
  fixed. This is a post-hoc TRAIN ablation, not predeclared v1 validation or
  TEST tuning. The original result/config are not overwritten.
- **Hardware (`HARDWARE_DEVIATION`):** Same standalone cached BERT CPU FP32,
  four threads,batch16. No additional model, quantization, dependency install
  or precision substitution. GPU peak fields are null because no GPU is used.
- **Measured outcome (`REPOSITORY_SPECIFIED`):** All125 images per condition
  complete and verify. word_only:341/983,1 recovered,44 damaged,62 changed
  crops,13.31s correction. local_ocr:363/983,3 recovered,24 damaged,42 changed,
  16.38s. Geometric503 matches unchanged. Local recoveries are EVERY,Open,
  TROPICAL. Result SHA256:
  `6919040e863c2b18f0727a0458957ed7e9e0bf4c244ba5545dc3c9f3534de7d5`.
  Native confidence indeed hides some missing-character errors, but relaxing
  its gate exposes substantially more incorrect language substitutions.
- **Interpretation (`UNRESOLVED`):** A reliable lightweight BERT correction
  rule is not established. No best threshold is selected, no configuration
  is promoted, and no strict/TEST/generalization claim follows. Caption and
  a spelling-fine-tuned BERT head require their own qualification.
- **Protocol:** `configs/approximate/totaltext_v4_bert_confidence_ablation_v1.json`,
  `docs/TOTALTEXT_BERT_CORRECTION.md`,
  `/tmp/totaltext-letterbox-bert-confidence-ablation-v1/results.json`;
  CPU audits `/tmp/totaltext-letterbox-bert-completed-audit-v1.json` and
  `/tmp/totaltext-letterbox-bert-confidence-ablation-audit-v1.json`.

## TT-NEUSPELL-CORRECTION-V1 — Spelling-trained BERT after frozen PARSeq

- **Scope (`ASSUMPTION`):** User-authorized extension after generic MLM
  correction damaged more words than it recovered. NeuSpell's pretrained
  English spelling head is frozen; the reference paper sections3.3–3.5 do not
  specify BERT correction. Approximate TRAIN diagnostic only, isolated from
  STRICT, existing MLM results and official TEST/Table5 claims.
- **Head/provenance (`REPOSITORY_SPECIFIED`):** Author NeuSpell snippet retains
  the noisy word, averages final subword states and applies its trained100002
  class head. HF checkpoint revision
  `d2a5bbccc41a0f4ff5e7c16e1c3b8d96ba8883b2`; NeuSpell repository revision
  `aa3782c2ea25d286b487ead3d99389dfc6801907`. Authenticate raw vocabulary Git
  blob `e0a3ce1814cd9656209b8524cc4b61d9828235d0` and snippet blob
  `a564cbce9e659f16717d31c7765b8cd2391ec585`. Downloaded weight SHA256 must
  match authoritative pinned HF LFS metadata; preserve complete file manifests
  and exact label order. Data-only vocabulary conversion rejects executable
  pickle objects. Load genuine backbone/head strictly after auditing only
  explicitly allowed unused legacy pooler keys; no random head substitution.
- **Source (`REPOSITORY_SPECIFIED`):** Reuse the same 125-image latest completed
  letterbox single-head `word_f1_control`,949 proposals/983 care words, native
  256 masks/512 input, padding0/no area/top10 filters and SHA-bound PARSeq outputs.
  No segmentation, crop, recognition checkpoint or source text is regenerated.
- **Input/selection (`ASSUMPTION`):** Original visible ASCII word>=3; preserve
  punctuation, numbers, mixed-case and multiword outputs. No native PARSeq
  confidence gate. Compare word_only/local_ocr, optionally SHA-bound captions
  (prefix80 words), with native-head and conservative arms. Conservative edit
  limits1/2 at length6, relative<=0.3, predicted probability>=0.5; require
  log-gain>1.5 versus summed casefold KEEP-class probability when representable.
  OOV originals cannot be scored by this word head: reference=null and use only
  edit/probability gates, never invent a score from UNK. Special labels pass
  through; unknown target/over256 tokens causes an explicit KEEP abstention.
  Thresholds are uncalibrated hypotheses. Fast offset target extraction and
  caption prefixes adapt the author decoding and require independent validation.
- **Evaluation (`ASSUMPTION`):** Inherit frozen bboxIoU>=0.5 deterministic
  one-to-one exact Unicode casefold/retained punctuation, micro983 care GT
  denominator including unmatched words; also case-sensitive/fuzzy metrics,
  recovered/damaged/changed traces and unchanged geometry. GT isolated from
  inference; no failed-sample omissions. Not official polygon spotting.
- **Runtime (`HARDWARE_DEVIATION`):** Standalone cached upstream inference,
  explicit CPU or single CUDA:0, FP32/batch8/max256 tokens/four CPU threads,
  deterministic seed20261006 and TF32 disabled. Record model set, actual input
  shapes, loading audit/dtypes, software and GPU allocated/reserved peaks.
  OOM/nonfinite/loading errors propagate. Time is correction-only; selection
  arms share the same neural timing. GPU memory remains unqualified.
- **Qualification (`REPOSITORY_SPECIFIED`):** 77 focused CPU tests pass,
  including31 new tests with genuine tiny BERT word-head fixtures and46 MLM,
  metric and PARSeq regressions. Real-source preparation confirms125/949/983,
  baseline384 exact/372 case-sensitive/475 fuzzy/503 geometric matches;
  555 eligible crops. Engineering fixtures do not estimate pretrained accuracy.
- **Outstanding (`UNRESOLVED`):** Actual pretrained NeuSpell assets/inference,
  OCR-domain transfer, rare/non-English names, calibrated gates, caption benefit
  and GPU memory. Real inference failed explicitly for missing assets; HF
  network probe produced ConnectionError. No useful correction claim is
  supported until the host downloads authenticated weights/vocabulary and runs
  paired evaluation. No Total-Text captions were generated in this milestone.
- **Protocol:** `docs/TOTALTEXT_NEUSPELL_CORRECTION.md`,
  `configs/approximate/totaltext_letterbox_neuspell_correction_v1.json`,
  `recognition/neuspell_correction.py`, `scripts/evaluate_totaltext_neuspell.py`,
  `scripts/run_totaltext_neuspell_correction_host.sh`;
  preparation `/tmp/totaltext-letterbox-neuspell-preflight-v1/`.

### Total-Text letterbox .50 frozen failure examples (post-hoc diagnostic)

- `scripts/package_totaltext_letterbox_failure_examples.py` audits the existing
  TEST300/2204-word letterbox epoch71 outputs; it runs no segmentation or OCR.
  This is APPROXIMATE / PRACTICAL_APPROXIMATION, not Table 5 reproduction.
- **ASSUMPTION / diagnostic only:** exclusive failure categories use bbox IoU
  .50, word-box coverage .80, and significant fusion coverage .50 for each of
  at least two GT words. Raw CC incidence is measured in the GT word bbox;
  components are not asserted to be letters. A component-union witness uses
  GT only post-hoc and is never used to change inference or select crops.
- **REPOSITORY_SPECIFIED:** casefold and Unicode P/S removal reuse the frozen
  evaluation, including deterministic maximum one-to-one matching. Original
  glyph GT and qualified V4 word GT are shown separately; the latter replays
  square letterboxing, native256 target resize, validity and contact suppression.
- **UNRESOLVED:** acceptable bbox IoU does not establish that the crop includes
  every glyph or that the recognizer alone caused the failure. Categories
  describe observed evidence, not exclusive causal proof. Three deterministic
  examples per nonempty category illustrate, but do not estimate, prevalence.
- **Normalized visual subset (`REPOSITORY_SPECIFIED`):** offline
  `reselect-normalized` excludes the 100 normalization-only failures and
  selects five examples for each remaining category from the frozen audit.
  It neither reruns inference nor changes matching. Small crop thumbnails may
  be enlarged for display only; the original crop hash is still checked.
- **Lexical position-free audit (`ASSUMPTION`, post-hoc only):** exact multiset
  overlap is computed separately within each image; every prediction and GT
  can contribute at most once. It does not match across images, tokenize
  characters, or award credit for substrings; empty normalized strings do not
  match. Casefold and Unicode P/S removal preserve the frozen normalization.
  Presence of the GT text in another crop is an additional diagnostic, not
  a recovery count, because duplicate words may reuse already-counted outputs.
  English visual panels retain OCR/GT strings exactly as originally stored.

## TT-NEUSPELL-LEGACY-TOKENIZER-V2 — Export normalization and position buffer

- **Observed v1 failure (`REPOSITORY_SPECIFIED`):** The user downloaded and
  authenticated the pretrained NeuSpell artifacts. The v1 host run failed
  before neural inference because `AutoTokenizer` resolved lowercase=True
  versus the configuredFalse expectation. A real checkpoint compatibility
  check then found the legacy `bert.embeddings.position_ids` entry in addition
  to the two already declared unused pooler keys. Both failures were retained.
- **Tokenizer choice (`ASSUMPTION`):** The pinned export omits `do_lower_case`;
  the installed transformers4.47.1/tokenizers0.21.4 loading call resolvesTrue.
  v2 recordsTrue explicitly, matching the published HF snippet call in this
  environment, with no flag override or suppression of the mismatch check.
  InferringFalse from the model name in v1 was incorrect. Raw OCR strings and
  conservative output case stay separate from encoder normalization. True is
  an explicit approximate protocol choice, not a strict paper component.
- **Training preprocessing (`UNRESOLVED`):** The original toolkit uses the
  named `bert-base-cased` tokenizer, while the exported custom HF tokenizer
  lacks its casing setting. The author HF loading path and training-intended
  casing may differ. The v2 result qualifies only the recorded loading variant;
  no silent claim of full NeuSpell training/preprocessing reproduction follows.
- **Legacy position entry (`REPOSITORY_SPECIFIED`):** The actual authenticated
  checkpoint buffer [1,512]/int64 exactly equals the buffer regenerated by
  modern BERT. Exclude it only when explicitly allowed by the v2 config and
  after exact shape/dtype/value equality. Audit that proof, exclude only the
  already allowed unused pooler keys, then strictly load every learned tensor.
  Non-equivalent positions, unapproved extras, missing/mismatched weights fail.
- **Evidence (`REPOSITORY_SPECIFIED`):** Checkpoint SHA256
  `c5b3c5ab535fc3782b2d02f301662d0106dd419e36c09626443366c9eb6f1bb5`;
  tokenizer configSHA256
  `359fdd360be3de14228b56e2ca9c6c940ff0b1a3dc88a5bf32bdeeb21ec7be9c`;
  raw vocabularySHA256
  `0bdbeed9f99760810b51f27ea609f8f8366c3038b3de99e7603a935018929a41`.
  Model/source revisions and Git blob mappings remain pinned as in v1.
- **Experiment isolation (`REPOSITORY_SPECIFIED`):** A separate approximate
  v2 config/output retains the original v1 config, failed host preparation and
  errors. Mask/contours/PARSeq outputs,125 images/949 proposals/983 care words,
  all gates/context bounds and503 geometric matches stay fixed. No TEST,
  captions or threshold tuning is introduced. STRICT remains untouched.
- **Runtime (`HARDWARE_DEVIATION`):** Cached standalone CPU FP32, four threads,
  batch<=8, seed20261006, input lengths3–27 under the declared256-token bound.
  Loading4.09s; correction4.61s word_only/5.28s local_ocr. Shared neural timings
  across selection arms. No GPU execution/VRAM qualification or package install.
- **Completed outcome (`REPOSITORY_SPECIFIED`):** All250 shards complete
  without failed/excluded samples. Baseline384/983; word_only/native331
  (19 recovered,72 damaged), word_only/conservative379 (18/23), local/native325
  (19/78), local/conservative380 (19/23). More words are recovered than with
  the generic MLM, but the aggregate remains negative. No OCR improvement,
  calibrated gate, TEST transfer or Table5 claim is supported.
- **Verification (`REPOSITORY_SPECIFIED`):** 83 focused tests pass, including
  omitted tokenizer flags and exact/non-equivalent legacy buffers. An independent
  audit rechecks250 shards/metrics/deltas/crop identities and fast/slow encoder
  parity on1110 scored inputs. ResultSHA256
  `56aa6e980c35db6cc1a8b4b17e200d47516f21d8e6a848b61579fe4bae415920`.
- **Protocol:** `configs/approximate/totaltext_letterbox_neuspell_correction_v2.json`,
  `docs/TOTALTEXT_NEUSPELL_CORRECTION.md`,
  `/tmp/totaltext-letterbox-neuspell-correction-v2/results.json`,
  `/tmp/totaltext-letterbox-neuspell-correction-v2-audit.json`.

## TT-BERT-METHODS-V1 — Selective edits, fixed-pool BERT ranking and Florence context

- **Scope (`ASSUMPTION`):** User-requested additional BERT methods and scene
  context, developed after negative NeuSpell v2 TRAIN results. Same125-image
  holdout is reused development data, not independent validation. APPROXIMATE
  only; no strict, official TEST or Table5 result. Three methods/two casing
  controls/three contexts were fixed before their additional outputs, with no
  threshold sweep after measurement.
- **Frozen evidence (`REPOSITORY_SPECIFIED`):** Same letterbox mask/contours/
  PARSeq source hashes,949 crops/983 care words/503 geometric matches and
  baseline384 exact. Old sealed model/code/config/results remain unchanged.
- **Casing controls (`ASSUMPTION`):** Compare published lowercase=True and
  explicitFalse following the original toolkit cased encoder. The latter
  `training_cased` name denotes a hypothesis; training-time intended export
  preprocessing remains unresolved. Default accent stripping is linked to the
  casing choice. Identical learned weights and all other bounds stay fixed.
- **Selective edits (`ASSUMPTION`):** Original source length>=4 must be OOV in
  the pretrained NeuSpell word-label vocabulary. Require head probability>=0.8
  and only<=2 missing-character insertions, preserving all source characters
  in order and original casing. Known words, punctuation, numbers and mixed-case
  output shapes remain protected. No GT-derived lexicon, special-class text or
  invented OOV KEEP score. Inspired by local-edit work/LevOCR, not their trained
  architecture reproduction; probability is not calibrated correctness.
- **Contextual candidate ranking (`ASSUMPTION`):** Up to6 candidates include
  immutable exact OCR KEEP and bounded word_only NeuSpell top10 labels with
  probability>=0.01. OOV-only; existing1/2 edit budgets and relative<=0.3 bound.
  Pool is frozen across contexts within each casing control. Frozen BERT MLM
  target-only mean PLL-word-l2r accepts gain minus0.5*edits strictly>1.5; ties
  KEEP. Mean/target-only scoring retains length/tokenization bias. Unknown/overflow
  abstentions are explicit; model/runtime/nonfinite errors propagate.
- **Scene observations (`ASSUMPTION`):** Actual Florence-2-large-ft captions on
  all125 original SHA-bound RGB images, with pinned revision/model/source files,
  fixed detailed-caption task/decoding and exact input coverage. Substitution
  for strict BLIP-2 is confined to this approximate extension. Current TRAIN
  manifestSHA05c41d87cf83e6d3d25cfc465eea9345ebd19b7cb0034030b87438486ff9be9b
  is RGB lookup only; each image must match the frozen source, and annotations
  are not opened by caption inference. Prefix for NeuSpell/separate segment for
  MLM; budget80 words. Captions are fallible, not new GT or candidate lexicons.
- **Runtime (`HARDWARE_DEVIATION`):** Existing main py311 plus existing isolated
  einops0.8.1, reused without installation/upgrade. CPU FP32/four threads,
  Florence batch1/greedy beam1/max64 new tokens,768-square pixels; separate model
  processes and cached stages. Caption cost845.29s; correction-only timings,
  actual shapes, hashes, loading audits and placement are saved. GPU peaks=null,
  no GPU/VRAM qualification. Inherited early_stopping=True is ignored with
  beam1 and warns; pinned generation config preserves the behavior.
- **Measured outcome (`REPOSITORY_SPECIFIED`):** All18 full-data arms complete.
  Best word-only OOV insertions396/983 (14 recovered,2 damaged,+12); local and
  local+Florence insertion arms392 (10/2,+8). Fixed-pool lowercase MLM reranking
  word383/local382/scene388 (scene5 recovered,1 damaged,+4 versus baseline).
  Broad conservative scene head377; cased scene PLL386 (2/0,+2). Context helps
  that reranker but is not uniformly beneficial. No arm excludes failed samples
  or changes geometry/denominator; all full results and per-image outputs saved.
- **Literal transcription (`UNRESOLVED`):** Best-arm damages RESIDENTAL->RESIDENTIAL
  and INTEGRIT->INTEGRITY show ambiguity between language repair and visible
  OCR. Image802 literally prints RESIDENTAL. Image1274 has a pole across the word
  region; intended complete text/occlusion policy is unresolved. Supplied GT
  remains unchanged, and no sample-specific exception or relabeling is added.
  Independent transfer/calibrated gates remain unresolved.
- **Qualification (`REPOSITORY_SPECIFIED`):** 103 focused tests pass, including
  genuine tiny head+MLM execution/resume with GT files absent and caption/RGB/
  corruption/selection boundaries. Real pretrained smoke/full caption generation
  and every head/PLL arm pass. Independent audit verifies750 head+750 PLL+125
  caption shards, all18 metrics/deltas/crop identities, fixed pools and3330
  fast/slow encoding matches. No-caption controls reproduce exactly in the full
  scene experiment. No model/runtime/loading failure is suppressed.
- **Protocol:** `configs/approximate/totaltext_bert_methods_v1.json`,
  `docs/TOTALTEXT_BERT_METHODS_V1.md`, `recognition/bert_method_selection.py`,
  `scripts/generate_totaltext_bert_captions.py`,
  `scripts/evaluate_totaltext_bert_methods.py`,
  `scripts/run_totaltext_bert_methods_host.sh`.
  Full resultSHA256:
  `1c8539835200573ef50ccc2bbb5147f3b249ec54e6f8e73b5a9d319bf5733620`.
  CaptionsSHA256:
  `06b50d6c3a6b48a160cd208472397307620f4081ae1f952e287776cdb571ffa2`.
  Audit `/tmp/totaltext-bert-methods-full-v1-audit.json`.


## TT-BERT-VISUAL-V1 — broader candidates and visual/contextual BERT correction

**Scope:** user-requested80–90% recovery target, with denominator still
`UNRESOLVED`. Report all983 care GT,503 geometrically localizable words and119
baseline localized errors separately. This is reused Total-Text TRAIN development
in APPROXIMATE, never independent validation/TEST/Table5. Original letterbox
segmentation, contours and949 originalRGB crop identities remain frozen.

- `PAPER_SPECIFIED`: local paper sections3.3–3.5 motivate crop recognition and
  scene context, with BLIP-2/MPNet/lexical matching and DeepSolo in the strict path.
  They do not specify BERT correction, lexicon expansion or this fusion rule.
- `REPOSITORY_SPECIFIED`: authenticated PARSeq-S weights/runtime and NeuSpell
  vocabulary/word head; retain strict learned-weight loading. Existing sealed
  BERT v1/v2/methods outputs and implementations are preserved unchanged.
- `ASSUMPTION`: extend candidates using original KEEP, previous insertion anchor,
  six frozen head top10 lists, Florence caption tokens,32 external pretrained
  vocabulary neighbours and canonical PARSeq AR beam10. No Total-Text-derived
  lexicon or GT enters inference. Alphabetic single-word sources2–25 characters
  only; protect internal mixed-case/numeric/punctuation/multiword shapes.
  Newly generated lexical alternatives allow1 edit for source length<6, else3,
  relative edit<=0.5. The immutable insertion reference is retained regardless
  of that new edit bound. Beam alternatives may exceed lexical bounds. Pool
  maximum96; ordered provenance and excluded/protected cases remain auditable.
- `ASSUMPTION`: causal teacher-forced PARSeq sum over characters plusEOS is
  image-conditioned evidence, distinct from native post-refinement confidence.
  Confirm each complete beam path against independent full teacher forcing.
  Keep exact original and insertion anchors plus the strongest AR alternatives
  in a shared24-candidate shortlist. English vocabulary, caps and AR length
  bias may exclude a correct word; measure candidate ceilings post-hoc only.
- `ASSUMPTION`: six predeclared controls: insertion KEEP, visual-only, and visual
  plusBERT local/Florence with original/insertion anchors. Gain=dVisual+0.5*dBERT,
  require dVisual>=-1 and gain>0.1; ties KEEP, unknown BERT reference abstains,
  nonfinite/runtime/loading failures fatal. BERT uses the existing target-only
  meanPLL-word-l2r, fixed original OCR neighbours and separate scene segment.
  Candidate availability is identical across score-context conditions and can
  include caption tokens even in the local-score control. Scores are not a
  calibrated posterior or the strict paper alpha/beta/tau formula.
- `ASSUMPTION`: paired exact Unicode casefold, punctuation retained, bboxIoU>=0.5
  maximum one-to-one matching, all983 care GT and no failed-sample exclusion.
  Oracle candidate coverage may read GT during evaluation and is explicitly a
  diagnostic upper bound; it must never be reported as achieved accuracy or used
  to choose per-sample output. Thresholds are fixed before new model scores.
- `HARDWARE_DEVIATION`: separate CPU FP32 PARSeq and BERT processes, cached
  earlier heads/Florence, deterministic seeds, fourthreads, AR/BERT scorebatch16,
  shortlisted context candidates. Reuse the existing isolated einops dependency
  without installs. No GPU execution/VRAM qualification or paper timing claim.
- `UNRESOLVED`: intended legacy head casing, literal misspellings/occlusion
  transcription versus linguistic completion, evidence calibration, proper-name
  recovery, the exact80–90% target denominator and independent transfer.

Config: `configs/approximate/totaltext_bert_visual_correction_v1.json`. Report:
`docs/TOTALTEXT_BERT_VISUAL_CORRECTION_V1.md`. Evidence/inspiration: original
[PARSeq paper](https://arxiv.org/abs/2207.06966) describes external-language-model
rectification errors when image evidence is absent;
[LevOCR](https://arxiv.org/abs/2209.03594) fuses visual and textual features during
sequence refinement; [ABINet](https://arxiv.org/abs/2103.06495) models character
probabilities and iterative correction. This experiment reuses frozen pretrained
models and does not reproduce their newly trained architectures.

## CLIP-WORD-VERIFICATION-V1 — Fixed-image PARSeq/caption verifier

- `PAPER_SPECIFIED`: Sections3.3–3.5 and Algorithms1/2 of local
  `papers/2503.15639v2.pdf` specify original-image localized crops, PARSeq-S,
  BLIP-2 descriptions, MPNet/lexical scoring and DeepSolo fallback. They do not
  specify a CLIP verifier. This user-authorized extension remains isolated in
  APPROXIMATE configuration and does not resolve the strict checkpoint,
  architecture, scoring or evaluation ambiguities.
- `REPOSITORY_SPECIFIED`: reuse the frozen TRAIN125 crop/PARSeq battery plan
  SHA256 `63e5b7542b45ecbc72acba16152e4456bdd29af4adf907fb097282b867ed64b5`,
  1173 threshold0.5/pad0 anchor boxes, and threshold-only/pad0 raw PARSeq
  alternatives. Check prepared/OCR/image/checkpoint/source contracts. Reuse
  original-image Florence-2-large-ft descriptions with SHA-bound RGB identities
  and caption artifact
  `06b50d6c3a6b48a160cd208472397307620f4081ae1f952e287776cdb571ffa2`.
- `ASSUMPTION`: freeze output boxes to the original anchors and compare every
  alternative against identical image features per anchor/view. Use full RGB,
  baseline crop, doubled surrounding crop and target-covered surrounding
  control, preserving field of view with the existing median-RGB square canvas.
  This separates transcript selection from crop selection. The existing paired
  crop/text battery and existing shared-image group selector were inspected;
  neither is overwritten. Masking can introduce synthetic visual artifacts.
- `ASSUMPTION`: word raw/quoted prompts and a normalized three-template mean;
  word-first scene-caption prompts; normalized feature residual against the
  same caption prompt with empty word; cyclically shuffled captions. Shared
  caption prefix is shortened to model context, never candidate text. A
  candidate-independent scalar caption subtraction would not change rankings;
  the normalized feature residual is a separate, unvalidated hypothesis.
- `ASSUMPTION`: use existing Florence descriptions in this experimental arm,
  rather than regenerating BLIP-2; captioner substitution is confined to this
  extension. Captions can contain independent text guesses; evaluation records
  literal target occurrences and shuffled-caption controls. Caption benefit
  cannot establish the paper's BLIP-2 checkpoint or description ablation.
- `ASSUMPTION`: native PARSeq confidence control, equal midrank fusion and a
  fixed0.02 strict visual-improvement gate requiring PARSeq/crop/surround
  agreement; baseline-preserving ties. No GT word, dictionary or caption-derived
  word enters the candidate pool. Scores are not calibrated probabilities.
- `ASSUMPTION`: micro casefold exact, punctuation retained, deterministic
  maximum one-to-one bboxIoU>=0.5 matching and all983 care GT denominator;
  empty outputs never match. Annotations open only after saved selections.
  Fixed-geometry pool oracle is evaluation-only. Correctness AUROC uses baseline
  boxes overlapping exactly one care GT, with explicit missing/ambiguous counts;
  PARSeq baseline AUROC uses its original native confidence, while candidate
  ranking uses per-word maxima across source crops. Pair discrimination also
  reports Levenshtein-one alternatives. These reused TRAIN images are development
  data, not independent validation; anchor/candidate pairs are correlated.
- `HARDWARE_DEVIATION`: separate sequential CLIP loading, explicit CPU or
  single-GPU execution, float32/batch16/four CPU threads, fixed seeds,
  deterministic algorithms and disabled TF32; source PARSeq/caption execution
  remains separately recorded. Allocated/reserved VRAM is null on CPU. OOM is
  logged and re-raised; its simulated unit test is not GPU qualification.
- `REPOSITORY_SPECIFIED`: the initial missing-snapshot/DNS/CUDA preflight
  failures are preserved. The later host run completed B/32,B/16,L/14 on
  TRAIN125, with declared immutable revisions, upstream weight receipts and
  loading-key checks. B/16 upstream SHA256 is recorded in the suite receipt,
  without mutating the historical config. Results are linked in
  `docs/CLIP_VERIFICATION_RESULTS.md`.
- `UNRESOLVED`: target RTX3070 memory qualification remains unperformed;
  the completed host run used an RTX3090. CUDA remains unavailable in Codex.
- `UNRESOLVED`: absolute correctness calibration, prompt/context and spelling
  sensitivity transfer, safe correction gates, caption leakage and independent
  TEST benefit require evidence. No Table5/Table7 reproduction or new strict
  fallback policy follows from this exploratory battery.

Protocol: `configs/approximate/clip_word_verification_train_v1.json`; paths are
separate in `clip_word_verification_local_paths_v1.json`. Implementation:
`recognition/clip_word_verifier.py`, `scripts/run_clip_word_verification.py`.
Research, runnable commands, artifacts and measured outcomes are documented in
`docs/CLIP_WORD_VERIFICATION.md` with primary CLIP/CLIPTER/CLIP4STR/spelling
sources. Existing scientific paths and user changes are preserved.

Measured B/32 CPU approximate TRAIN diagnostic: complete125/1173/983,
baseline368, fixed-geometry pool oracle405, native-confidence selection383,
crop written362, crop caption355, shuffled-caption359, normalized
context-residual369, equal-rank fusion378, conservative agreement370
(2 recovered/0 damaged/9 changed). On482 unambiguous-overlap baseline anchors,
native PARSeq correctness AUROC0.823 versus crop-ensemble0.622 and
context-residual0.646. One-edit pair ranking uses67 pairs: PARSeq82.1%,
crop-ensemble59.7%, context-residual67.2%; these are correlated development
observations. Caption literal-target occurrence160/482 is a measured risk,
not evidence of generic context benefit. Scores and selections for the first
image are identical between independent real CPU smoke/full runs.43 focused
unit tests pass; GPU OOM handling was simulated only. Full result SHA256
`e9a6db80ffee2c33c3d2120d42ce12dcc0b87c43029647917ff4bb045c0c3e2e`,
under `/tmp/clip-word-verification-train-v1-final/vit_b32/train125/`.
No CLIP checkpoint/version transfer or improved paper recognition is claimed.


## TT-BERT-PADDED-V1 — explicit auxiliary padded RGB evidence

`PAPER_SPECIFIED`: section3.3/Algorithm1 of the local paper pads localized boxes
on every side and clamps to image boundaries. `ASSUMPTION`: one fixed additional
experiment uses p=max(1,ceil(0.25*original_bbox_height)), with all125 TRAIN images
and949 original localization boxes preserved. This changes recognition pixels;
it is not an identical-crop comparison, strict reproduction or TEST evaluation.
Record padded bbox, padding pixels, padded cropSHA and original cropSHA per block.
No annotation-derived coordinates or per-sample choices enter inference.

`ASSUMPTION`: native padded PARSeq is an explicit visual control. In contextual
controls, reuse the previous24-candidate pool and append the padded native word
when absent. Score all these candidates on the padded RGB using genuine causal
PARSeq likelihood, and evaluate BERT local/Florence with the previous fixed gain
formula. Reuse BERT likelihood only for exact candidate/context strings and the
same pinned model/files/scoring/precision; newly appended words require actual
fresh BERT inference. An absent reference is never assigned a fabricated score.
Five controls are declared before padded outputs, without a threshold sweep.

`HARDWARE_DEVIATION`: isolated CPU FP32 processes, fourthreads, cached previous
BERT/Florence/head stages, AR/BERT batch<=16. No installs, GPU qualification,
training or paper timing claim. `ASSUMPTION`: report exact casefold/983, exact/503
and recovered/damaged separately under the unchanged bboxIoU>=0.5 one-to-one
metric. GT-dependent candidate ceilings are diagnostic only, never predictions.

`UNRESOLVED`: clipping versus language-model rectification is a hypothesis;
padding can expose adjacent words/background and reduce accuracy. Independent
transfer, literal misspellings/occlusion transcription and the user target's
80–90% denominator remain unresolved. Mask architecture/checkpoint, binary masks,
contours and the strict namespace are untouched. Config:
`configs/approximate/totaltext_bert_padded_evidence_v1.json`. Final combined report:
`docs/TOTALTEXT_BERT_VISUAL_CORRECTION_V1.md`.


### TT-BERT-PADDED-V1 measured development result

Both full original/padded experiments and an independent artifact audit completed.
The unchanged-crop fusion gives378/503 under scene context and fails to improve
396/503 insertion control. Padded native reaches403/503; padded BERT+Florence
reaches406/503 (80.7%),35 recovered/13 damaged, net+22 from384 baseline.
Local BERT without Florence reaches401, so paired context contributes+5 on this
fixed candidate pool. All-GT accuracy remains406/983=41.3%; localized error
recovery is35/119=29.4%. This does not establish80–90% error recovery or all-GT
accuracy. The padded pool's GT-dependent diagnostic ceiling is444/503=88.3%.
No parameters or annotations changed after these outcomes. All outputs remain
APPROXIMATE TRAIN development, not independent validation or a strict default.

## CLIP-INNOVATIVE-VERIFICATION-V2 — Region dependence and synthetic glyph probes

- `PAPER_SPECIFIED`: local paper sections 3.3–3.5 specify original RGB crops,
  PARSeq-S, scene descriptions and the MPNet/lexical/DeepSolo pipeline. These
  innovative CLIP rules are user-authorized APPROXIMATE extensions and do not
  modify or resolve the strict method/checkpoint/scoring/evaluation choices.
- `REPOSITORY_SPECIFIED`: retain all raw PARSeq v1 candidate pools, original
  threshold0.5/pad0 anchor boxes, caption/image/source provenance and all 28 v1
  control selections. Verify base configuration, code hashes, score shard
  hashes and input contracts before deriving any new rule. Completed base
  reports are hashed as provenance; their GT fields are not read by selection.
- `ASSUMPTION`: word-conditioned score difference between surrounding crop
  with target present and the same crop with the target box covered. Test
  raw/written/ensemble/context-residual variants and a fixed0.02 gate requiring
  PARSeq agreement. This intervention removes background inside the box too;
  its effect is not certified character-level causal grounding.
- `ASSUMPTION`: synthesize only already available complete printable-ASCII
  PARSeq strings as glyph images in three checksum-bound DejaVu fonts and two
  polarities. Average normalized image features, with square-canvas and
  bicubic anisotropic-square variants; compare real crops in the image tower.
  Font, word length and visual-semantic biases can influence this query.
  Undrawable/non-ASCII strings fail explicitly, never render surrogate glyphs.
- `ASSUMPTION`: reversed-string glyph queries preserve raw characters while
  testing order sensitivity; reversed strings never enter the OCR candidate
  pool. Palindromes contribute zero order evidence. Reversal may change meaning
  and kerning. Record render strings/font hashes/shape/RGB hash to reproduce
  every probe without committing generated images.
- `ASSUMPTION`: stretch the complete crop to declared CLIP resolution and
  compare image/text and glyph retrieval; equal-weight two-view scores and
  fixed0.02 glyph gate requiring PARSeq/region-effect/glyph agreement. All GT
  annotation reading is deferred until saved selections. Native v1 metrics
  retain all care GT and exact casefold/punctuation matching at bboxIoU>=0.5.
  Empty readings remain in all relevant method denominators.40 full v1/v2
  arms versus33 counterfactual-only arms are identified explicitly.
- `HARDWARE_DEVIATION`: sequential standalone frozen float32/batch16 models,
  four CPU threads, deterministic seeds/backends, explicit CPU or one CUDA
  device, cache image/template features within each model process. Preserve
  source execution provenance separately. Record CUDA allocated/reserved peaks,
  model set, processed shapes and propagated OOM/failures. CPU and simulated
  OOM tests do not qualify actual GPU memory.
- `ASSUMPTION`: host launcher downloads only declared immutable official
  B/32,B/16,L/14 model/tokenizer files, checks full commit and upstream LFS
  SHA256, and saves per-model download receipts. It installs no dependencies,
  uses separate result trees per code/config/device/offline protocol, propagates
  subprocess failures and preserves existing configs/results. B/16's newly
  obtained upstream SHA256, when available on host, is recorded in the receipt;
  no unverified value is invented or silently written into STRICT configuration.
- `UNRESOLVED`: pretrained spelling-subspace projection availability/provenance
  and compatibility, patch-to-character matching, font/length shortcuts,
  probability calibration, gate transfer and independent TEST performance.
  Target RTX3070 qualification remains unresolved; the completed host run
  used an RTX3090. CUDA preflight still fails in the Codex session.

Completed host approximate TRAIN diagnostic (`REPOSITORY_SPECIFIED`): all
three checkpoints, all125 images/1173 fixed boxes/983 care GT and all40 arms
per checkpoint. Official weight receipts are upstream-verified, including B/16
SHA256 `ec89c7b09c749a60aae3c9cd910516f24b58214a7df060b48962d14c469cfbf0`.
L/14 rank fusion obtains386 exact (19 recovered/1 damaged versus baseline368),
native confidence383 (26/11), synthetic-glyph square383 (19/4), conservative
PARSeq/crop/surround agreement376 (8/0). Direct caption prompts lose7/5/4 exact
readings versus crop-written for B/32,B/16,L/14 respectively. L/14 glyph-square
AUROC is0.725 versus native0.823; no absolute correctness probability is
calibrated. All outcomes are exploratory development results, not independent
TEST, significant gains, or new strict defaults. A read-only CPU audit checked
750 score shards, identical model-independent pools/boxes, preserved v1
selections, code/input/checkpoint hashes and independently recalculated all
six reports. The original artifacts remain unchanged.
Report `/tmp/clip-verification-host-suite/6920d0b0c030/comparison.json`, SHA256
`2cdbf5b858ad181eff5322a512d41264e613af109a2ef5287e34fcbcf6eb828a`;
audit `/tmp/clip-verification-analysis-6920d0b0c030/verified_results.json`;
human interpretation `docs/CLIP_VERIFICATION_RESULTS.md`.

Measured approximate TRAIN development: counterfactual-only v2 reads all125
images/1173 boxes/983 care GT from complete B/32 scores. Baseline368;
confidence383; region-effect written362 (14 recovered/20 damaged);
region-effect context-residual366 (19/21); PARSeq-agreement region-effect gate
372 (4 recovered/0 damaged/23 changed).67 one-edit pairs give region-effect
written68.7% versus prior crop-written59.7% and native PARSeq82.1%; these
correlated reused development observations are not independent validation.
Report `/tmp/clip-innovative-verification-v2/vit_b32/counterfactual/results.json`,
SHA256 `0901ef57307bac7b0b361ad0b4844906cfc5d4a1d3e273609b5448075ab138c6`.
The initial syntax error in the new reversal expression was caught and fixed
before model inference.53 focused tests pass, including download corruption,
strict candidate/box preservation, rendering/cache controls, and propagated
simulated OOM. A real one-image B/32 CPU host-suite smoke completed all40 arms;
this does not qualify full rendering accuracy or checkpoint transfer.

Protocol: `configs/approximate/clip_innovative_verification_train_v2.json`,
paths in `clip_innovative_verification_local_paths_v2.json`; implementation in
`recognition/clip_innovative_verifier.py` and
`scripts/run_clip_innovative_verification.py`. Host command:
`bash scripts/run_clip_verification_suite_host.sh`. Full description/primary
FDP, T-MARS and spelling-subspace sources: `docs/CLIP_INNOVATIVE_VERIFICATION.md`.

## TT-BERT-CROP-SELECTION-V1 — BERT selection/admission of native crop groups

`PAPER_SPECIFIED`: inspected local paper sections3.3–3.5 specify padded component
localization, PARSeq-S, BLIP-2 scene descriptions and MPNet/lexical/DeepSolo routing.
They do not specify BERT, cross-threshold many-crop groups or the admission rule
below. This user-requested extension belongs exclusively to APPROXIMATE mode;
no strict configuration, fallback, localization limit or paper score changes.

`REPOSITORY_SPECIFIED`: reuse125 frozen TRAIN images, original crops/masks and
PARSeq readings from the crop-group/refinement pipeline. The source baseline is
`guarded_consolidated`:738 correct /7,462 outputs /983 care GT under its local
punctuation/symbol-removing bbox metric. Authenticate source plans, image/mask
contracts, original OCR records, selections and caption RGB identities. Existing
spelling-correction metrics use a different normalization and are not combined.

`ASSUMPTION`: shortlist baseline KEEP plus<=2 distinct native readings per group,
bboxIoU>=0.5 and robust visual score>=0.85*baseline. Frozen high-confidence
same-line OCR anchors exclude the target and groups sharing parent candidates.
The caption control reuses pinned, fallible Florence descriptions in a separate
BERT segment. No dictionary-generated alternatives or corrected recursive
context. General pretrained BERT-base-uncased uses the existing target-only mean
PLL-word-l2r; it is not fine-tuned for text/background classification. Salazar2020
and Kauf/Ivanova2023 motivate scoring, not a claim of visual grounding.

`ASSUMPTION`: within-group ranking uses log(visual ratio)+0.15*BERT meanPLL delta,
gain>0.1 and KEEP on ties/unavailable reference. Assign each group/original crop
at most once; collisions revert involved groups to baseline. Admission is a
separate decision: geometric mean of PARSeq balanced confidence, absolute
letterbox mask average and min(1,height_in_mask_cells/2). Relative mask
completeness is not an absolute probability that a region contains text.
Joint admission multiplies physical by sigmoid((PLL+6)/2)^0.25; unavailable
BERT likelihood explicitly abstains to physical. No hard lexical/numeric ban.

`ASSUMPTION`: seven fixed families and thresholds0,0.05,...,0.95 are predeclared
before new scores. Select each TRAIN operating point for maximum precision while
retaining>=95% of the original baseline-correct GT identities, rather than using
newly recovered words to conceal lost baseline words. Report all curves and
unfiltered ranking controls. GT is separate from inference; evaluation opens
labels only after complete inference/payload checks. Preserve all983 care GT,
maximum-cardinality one-to-one bboxIoU>=0.5 exact normalized word matching,
micro precision/recall/Hmean proxies and explicit empty-output handling. This
is TRAIN development/calibration, not independent validation, Table5 or official
Total-Text polygon evaluation.

`HARDWARE_DEVIATION`: standalone CPU FP32 BERT, fourthreads/batch16, cached PARSeq
and Florence outputs. Optional single CUDA placement has explicit preflight,
allocated/reserved peak logging and recorded/propagated OOM; no fallback or
installation. CPU and simulated OOM tests do not qualify RTX3070 GPU memory.

`UNRESOLVED`: generic BERT is not a text detector; language can discard real names
or favor fluent hallucinations. Independent transfer, official polygon metrics,
calibration, alternative generator schemas and80–90% target recovery remain
unresolved until actual measured evidence. Preserve visual-only controls and
export their operating point if they outperform the BERT controls.

Implementation/protocol: `recognition/bert_crop_selection.py`,
`scripts/evaluate_totaltext_bert_crop_selection.py`,
`configs/approximate/totaltext_bert_crop_selection_v1.json`.
Details and measured evidence: `docs/TOTALTEXT_BERT_CROP_SELECTION_V1.md`.

### TT-BERT-CROP-SELECTION-V1 measured TRAIN result and final protocol

All125 images/983 care GT completed:9,595 shortlisted native readings,19,190
valid genuine BERT crop/context scores,2,036 frozen OCR anchors and4,826 groups
with local context. Labels were withheld during the entire real neural process.
`ASSUMPTION`: a configuration-only final revision exposes original anchor length2
and duplicateIoU0.5 as explicit required fields; all numeric decisions are
unchanged, and missing policy parameters now fail without implicit defaults.
`REPOSITORY_SPECIFIED`: reuse historical neural scores only after exact observable
sample equality, immutable model/masking/runtime checks, full payload/receipt
validation and scoring-function AST equality against an authenticated original
source snapshot. Source/destination labels were withheld during cache replay.
All143 final controls/operating points exactly reproduce the original neural
experiment. This is recorded cache reuse, not new neural inference or a
scientific parameter retuning after outcomes.

Negative BERT result: local crop reranking yields651 correct(10 recovered/97
lost); Florence reranking689(13 recovered/62 lost), versus738 baseline. Neither
retains95% of baseline-correct GT even before admission. BERT/Florence admission
without changing crops retains718/6,632 outputs at threshold0.25, precision10.8%,
recall73.0%. Physical-only admission gives721/4,439 at0.70, precision16.2%.
Best eligible family is native balanced PARSeq confidence>=0.95:704/2,806,
precision25.09%, recall71.62%,34 baseline words lost,95.4% baseline words retained,
4,656 outputs removed(62.4%). The exported recommendation therefore uses no BERT
at inference. Language/context comparisons remain available; the80–90% target
and stronger visual text/background verification remain `UNRESOLVED`.

`HARDWARE_DEVIATION`: actual CPU FP32 neural scoring took502.5seconds, with no
GPU memory/timing claim.74 focused tests pass; finalized genuine smoke completed
all stages. Genuine batched-vs-original token likelihood error<=3.8147e-6, tolerance
1e-5. An independent sparse maximum-matching/normalization/IoU audit confirms
704 correct /2,806 predictions /983 GT and proves native-confidence-only filtering
reproduces every exported crop ID without model inference. Initial smoke
checksum rejection after code updates and a corrected auxiliary audit that
mistook provenance text for GT fields are documented, not bypassed.

Final report:`/tmp/totaltext-bert-crop-selection-v2/results.json`,
SHA256:`66159fa62e6f8d987c400abd4d6282de03a8832d082c267fd06033df8e74a1e9`.
Native export:`/tmp/totaltext-bert-crop-selection-v2/selected-crops-confidence@0.95.json`,
SHA256:`940c933e5b3088ab8f25aa37c1cc27161b04d06b59add3e14e1f295b24472160`.
Evidence/limitations and model-free integration:
`docs/TOTALTEXT_BERT_CROP_SELECTION_V1.md`.

## CLIP-HIGH-RECALL-ADMISSION-V1 — Visual admission of many native PARSeq crops

- `PAPER_SPECIFIED`: inspected local PDF sections3.3–3.5/Algorithm1 specify
  padded component crops, PARSeq-S and contextual scoring/fallback. CLIP crop
  admission and cross-threshold groups are outside the strict paper method.
- `REPOSITORY_SPECIFIED`: frozen TRAIN125 groups have18,912 native candidates,
  unrestricted oracle782/983 and group oracle781. Guarded_consolidated yields
  738 correct/7,462 outputs. Read sealed observable inputs prepared for the
  crop-selection experiment, never its BERT scores; authenticate IDs/geometry,
  raw native PARSeq confidence, RGB hashes, original selection and provenance.
  The whole unmodified pool has40,380 crops/oracle811 and the transformed union
  472,358/oracle854; those require different crop-reconstruction/group adapters
  and are not silently substituted in this first test.
- `ASSUMPTION`: confidence>=0.5 is a cost-reducing first stage, motivated by
  the previously observed TRAIN control738/738 words in5,410 outputs. It was
  selected with this development evidence, not asserted to transfer safely.
  Record all2,052 exclusions; evaluate every method against the original738
  baseline-correct GT identities and all983 care GT. Native confidence0.95
  reproduces704/2,806 and is retained as the primary non-CLIP control.
- `ASSUMPTION`: six native-word signals: written/three-prompt ensemble,
  three-font/two-polarity square glyph cosine, positive-text minus negative-
  background prompt cosine, word-ensemble minus background cosine, and glyph
  query minus a black/white uniform-image query. The residual is an admission
  comparison across different crop images, not evidence that subtracting a
  common score changes candidate ordering within one image. No new spelling,
  caption, dictionary, word-length blacklist, GT crop or artificial crop margin.
- `ASSUMPTION`: fit label-free TRAIN empirical mid-CDF ranks, then equal-weight
  geometric means of PARSeq/CLIP ranks. Freeze all331 controls and save the CDF.
  The seed20261008 is a recorded reproducibility choice. Select per-family
  TRAIN precision subject to >=95% of the same baseline GT identities retained
  (>=702/738), not merely the same aggregate correct count. Threshold selection
  uses TRAIN labels; inference/admission-score generation does not. Exported
  per-crop decisions contain no GT, with this threshold-selection use explicit.
- `REPOSITORY_SPECIFIED`: retain the high-recall source metric, exact casefold
  after removing Unicode P/S, maximum-cardinality one-to-one bboxIoU>=0.5.
  All samples and983 care GT remain in the denominator. This differs from the
  prior punctuation-retaining368/405 CLIP fixture and official polygon Hmean.
- `HARDWARE_DEVIATION`: frozen single FP32 CLIP, batch16/four CPU threads,
  explicit CPU or CUDA:0, sequential loading, disabled TF32/deterministic
  backends. Record loaded model, input shape, device/GPU, latency, allocated/
  reserved peaks and propagated failures/OOM; no automatic model/device change.
- `UNRESOLVED`: robust precision benefit on independent data, L/14 GPU qualification
  for the new admission workload, independently held-out transfer, rank/score
  calibration, font/length/semantic confounds, source-pool expansion and target
  RTX3070 execution. Current Codex has no CUDA. B/32 CPU full run completed.

Verification:59 focused tests pass. A genuine B/32 CPU smoke scores15 of26
original outputs and finishes all331 controls on image14 (two care GT).
The best smoke point gives2/2 from2 outputs versus2/2 from3 for native0.95;
this is execution evidence, not a full-pool benefit claim. Payload hashes,
cache/config mismatch, original identities/boxes, threshold-zero exclusions,
ties, protected GT retention, tiny RGB channel order and a recorded/propagated
simulated memory failure are tested. Previous configs, models and result
artifacts remain separate; no packages or weights are installed or changed.

Measured approximate TRAIN development (`REPOSITORY_SPECIFIED`): all125 images,
7,462 source outputs,5,410 scored crops and331 controls. B/32 FP32 CPU scoring
took605.15s with no GPU-memory claim. Best retained point is native rank plus
word/background contrast rank geometric mean>=0.45:714 correct/2,798 outputs,
precision25.52%, recall72.63%,24 protected baseline words lost. Native>=0.95
gives704/2,806, precision25.09%, recall71.62%,34 lost. This is a small gain
selected on reused development data, not independent validation. A post-hoc
equal-budget diagnostic gives703 native versus714 combined correct at2,798
outputs; that budget is not exported as a per-image deployment rule.
All125 score/selection shards and331 aggregate metrics passed an independent
audit. L/14 also completed a genuine CPU one-image smoke; full L/14 GPU remains
pending. Source pool, models, original experiments and STRICT files stay intact.
Report `/tmp/clip-high-recall-admission-v1/vit_b32/7423fe6400b8/results.json`,
SHA256 `baa1ba9bf9bf5e862e9c6ec05f57e204a72abae64647e3924d0f13a978e8ca80`.

Protocol: `docs/CLIP_HIGH_RECALL_ADMISSION.md`,
`configs/approximate/clip_high_recall_admission_train_v1.json`,
`recognition/clip_crop_admission.py`,
`scripts/run_clip_high_recall_admission.py`.
Host L/14 command: `bash scripts/run_clip_high_recall_admission_host.sh`.

### WM-001 — Experimental word multi-task extension

- **Status:** `UNRESOLVED` for independent TEST performance and STRICT scientific
  claims; completed exploratory v1 evidence is recorded in the follow-up below.
- **Scope/date:** User-authorized word-separation research extension, 2026-10-08.
  This does not resolve or replace the paper's binary segmentation/localization
  protocol, existing architecture ambiguities or Table3/Table5 evidence gates.
- **Paper evidence:** Local `papers/2503.15639v2.pdf`, section3.2/Fig.3 specifies
  pretrained ResNet50, attention skips and a compressed bottleneck; section3.3
  specifies component-area crops with padding/top10. Implementation Details
  specifies BCE, RGB256/native128, Adam1e-4/decay.01/batch8. It does not specify
  kernels, centers, Dice, watershed or instance-detection metrics.
- **Repository inspection:** Existing `models/totaltext_word_kernel.py` and
  `segmentation/totaltext_word_kernel.py` already implement a two-channel model,
  connected native distance cores, paired owner mapping, simultaneous expansion,
  fallback components and per-instance boxes. `localize/crop_recovery.py` already
  contains a deterministic minimax watershed. Completed control, kernel,
  separator and word-training ablation reports were inspected before editing.
  Their losses/thresholds/budgets are not all matched. No historical run is repeated.
- **`PAPER_SPECIFIED`:** Preserve the existing ResNet50 attention-U-Net family.
  The selected runnable decoder remains an approximate implementation with its
  existing architecture assumptions; no new authoritative architecture evidence
  has been established.
- **`ASSUMPTION`:** Reuse qualified v4 polygon-derived full targets, validity,
  separation and owner maps, frozen TRAIN1130/validation125, seed202503,
  letterbox512/native256, frozen encoder/BN and skip floor. Add only a65-parameter
  center output to the existing kernel model (130 additional parameters versus
  one output;58,155,211 total). Copy common output tensors at initialization and
  reseed after model creation so branch-specific head RNG consumption cannot
  change sampler/worker streams.
- **`ASSUMPTION`:** Kernel construction remains the existing half-max interior
  distance/largest connected core rule. For Gaussian centers, project each owner
  centroid onto its kernel, use owner covariance scaled by.5² with eigenvalue
  floor1², clip to valid owner support, and retain one peak1 per surviving word.
  These are declared choices, not polygon-offset PSENet targets. Lost native
  words are counted, never invented. A raw-word-polygon helper reuses the existing
  overlap/ignore rasterizer; the matched runner's qualified v4 targets are unchanged.
- **`ASSUMPTION`:** Configurable text/kernel BCE+Dice and sigmoid Gaussian MSE,
  default task weights1/1/1, BCE/Dice weights1/1, smoothing1, text separator
  weight3. Center MSE balances Gaussian support and background within each image.
  All-invalid terms have zero gradient. A fresh text-only BCE+Dice control is
  scientifically necessary; historical BCE control/kernel reports remain separate
  references instead of being misrepresented as matched controls.
- **`ASSUMPTION`:** GT-blind center NMS radius3/threshold.50 splits multi-peak
  kernels and adds distant rescue markers. Compare foreground CC, kernel
  shortest-path expansion and the existing minimax watershed, with configured
  landscape weights.50/.25/.25. Preserve every unseeded foreground component.
  Native threshold.65 comes from pre-existing TRAIN evidence, not TEST tuning.
  Per-instance IDs remain separate through native-to-original box mapping.
- **`ASSUMPTION`:** Native bbox maximum-cardinality one-to-one IoU>=.50 word
  precision/recall/F1; any-pixel merge/split diagnostics divided by all expected
  care words, including lost words. Fully invalid predicted instances are ignored;
  partially invalid instances retain their complete boxes. Pixel IoU/F1 use the
  existing macro/micro valid-pixel metrics and perfect-if-both-empty policy.
  These are exploratory TRAIN-holdout geometry metrics, not official polygon-IoU
  detection evaluation or strict recognition accuracy.
- **`HARDWARE_DEVIATION`:** Sequential single-GPU segmentation-only branches;
  VRAM allocated/reserved peaks, FP32/batch8/input shape and failures are recorded.
  Explicit worker overrides are recorded separately. The restricted CPU sandbox
  requires num_workers0 because PyTorch worker IPC sockets are unavailable;
  there is no automatic fallback or hidden precision/batch reduction.
- **Alternatives considered:** Repeating old control/kernel runs would duplicate
  completed experiments without resolving their loss mismatch. Building a new
  decoder, polygon rasterizer or watershed would duplicate existing code and
  confound the head comparison. Plain MSE and CC/BFS remain selectable diagnostic
  alternatives. Raw polygon offsets and official polygon metrics require a
  separately qualified protocol; they are not silently substituted here.
- **Remaining questions:** Center/kernel supervision may increase false splits;
  NMS can suppress nearby small words, and native preprocessing cannot recover
  lost words. New100-epoch training, RTX3070 memory qualification, independent
  TEST validation and PARSeq recognition benefit have not been measured. Existing
  holdout reuse makes any future gains exploratory. STRICT remains blocked.
- **Qualification:** New target construction qualified all1255 TRAIN images:
  9284 surviving words with centers; one word was already lost in existing native
  preprocessing. A complete three-head real-batch CPU forward/backward/Adam step
  passed with local checksum-verified pretrained encoder, FP32/batch8/RGB512,
  output3x256x256 and explicit zero workers. This is a smoke-test result; GPU peaks
  are null. Focused tests cover shapes/common initialization, parameter budget,
  polygon/curved/small/empty/ignored targets, soft Gaussian losses, zero ignore
  gradients, fused-kernel separation, fallback, PARSeq coordinate mapping,
  visual exports, strict checkpoints and interrupted optimizer/RNG recovery.
- **Implementation/protocol:** `docs/WORD_MULTITASK.md`,
  `configs/approximate/word_multitask_v1.json`, `models/word_multitask.py`,
  `segmentation/word_multitask.py`, `segmentation/word_multitask_evaluation.py`,
  `localize/word_multitask.py`, `scripts/run_word_multitask.py`,
  `scripts/compare_word_multitask.py`, `scripts/run_word_multitask_host.sh`.

#### WM-001 operational follow-up — JSON contract repair

- **Evidence:** All three user GPU smoke artifacts completed, but training failed
  before model creation because serialized Adam betas were a JSON list while the
  regenerated in-memory contract contained a tuple. A recursive comparison found
  no other difference; normalizing through JSON made every contract equal.
- **Decision (`ASSUMPTION`):** Canonical JSON contracts plus an explicit migration
  limited to the exact defective runner SHA256
  `298b9023dfb7fe2fb6d9e8cf4234da515fffe953c7595341c63197f453bff38c`
  and complete smoke-only branches. All scientific code/parameters/artifacts
  remain exact checks. Original smoke JSONs/contracts are archived intact and
  transition receipts record hashes. Existing training artifacts, unknown
  revisions and changed scientific inputs fail closed. Smoke weights are not
  reused as training initialization. No target, model, loss or metric changes.
- **Observed hardware (`HARDWARE_DEVIATION`):** Preserved artifacts record RTX3090,
  FP32, batch8, RGB512/native256, all three variants successful without OOM;
  peak allocated3,967,820,800/3,969,918,464/3,972,017,152 bytes and peak
  reserved5,349,834,752 bytes. These are one-batch smoke measurements on RTX3090,
  not qualification of RTX3070, full training or paper reproduction.
- **Verification:** Replay on copies of all three actual artifacts preserves
  original smoke bytes and archives the original contracts. CPU regressions
  cover canonical serialization, smoke/train/evaluate/checkpoint/export stages,
  reuse, rejected scientific changes/training artifacts, checksum tampering and
  interrupted archival. Complete training/validation gains remain `UNRESOLVED`.
- **Implementation:** `segmentation/word_multitask_provenance.py` and
  `tests/segmentation/test_word_multitask_provenance.py`; protocol and recovery
  instructions remain in `docs/WORD_MULTITASK.md`.

#### WM-001 completed v1 experiment — 2026-10-08

- **Observed evidence:** All three branches finished 100 epochs and selected
  checkpoint replay on the same TRAIN1130/validation125 split, with 983 care
  validation words. Text / Text+Kernel / Text+Kernel+Center word F1 is
  .589620/.621272/.584130; merge rate .298067/.173957/.266531 and split rate
  .048830/.096643/.108850. This supersedes the earlier unmeasured 100-epoch
  training statement for this run. Selected epochs are 56/61/66.
- **Interpretation (`ASSUMPTION`):** Native bbox IoU>=.50 and existing any-pixel
  diagnostics support a kernel benefit with increased fragmentation under the
  frozen v1 protocol. Center supervision/post-processing does not improve word
  F1 here. Pixel IoU alone would favor the center model; the training and
  post-processing contributions of its center map have not been isolated.
  Single-seed, validation-selected results from a reused holdout remain
  exploratory approximate reproduction. Historical BCE runs remain references.
- **Hardware (`HARDWARE_DEVIATION`):** Actual RTX3090, FP32/batch8/RGB512/native256,
  sequential segmentation-only training, no OOM. Peak allocated
  4,233,335,296 / 4,236,481,536 / 4,237,008,384 bytes; reserved
  5,354,029,056 bytes each. RTX3070 qualification is still unmeasured.
- **Verification:** Exact comparison rebuild, checkpoint/source/encoder/export
  hashes, all100 sample-order hashes shared between variants, 1,500 visual/map/
  proposal files, per-image geometry and global/pixel metrics recomputed from
  saved maps, and original-coordinate boxes verified. All35 focused tests pass.
  The stale pre-training failure artifact is preserved. No new inference or
  training was performed during this audit.
- **Remaining `UNRESOLVED`:** Independent TEST performance, seed variability,
  official polygon metrics, PARSeq recognition benefit, center calibration and
  isolated center-map effects, target-GPU qualification and STRICT blockers.
- **Records:** `docs/WORD_MULTITASK_RESULTS_V1.md`,
  `provenance/word_multitask_v1_completion_audit.json`; original artifacts remain
  under `/extra1/bmartelli/context-str-runs/totaltext-word-multitask-v1`.
  The user-requested offline gallery ZIP is generated separately and preserves
  the original probability visualizations and report hashes.

### WM-002 — User-requested kernel/center-guided contour crops

- **Scope/evidence:** User requests both real recognition on existing findContours
  crops and optional kernel/center-guided splitting/union. Inspected local paper
  section3.3/Algorithm1, frozen repository contour semantics, multi-task map
  exports, existing watershed, crop recovery and genuine qualified PARSeq adapter.
  Original contours, strict localization, checkpoints and old results are preserved.
- **`REPOSITORY_SPECIFIED`:** Reuse RETR_EXTERNAL/CHAIN_APPROX_SIMPLE auditing,
  label-aware boxes, native-to-original letterbox mapping, original RGB crop
  hashing, pinned real PARSeq AR/refine1, and existing maximum-cardinality geometry/
  casefold/symbol-normalized matching. Existing seeded labels are a frozen control.
- **`ASSUMPTION`:** Refine IDs using a common kernel through text>=.50 support
  between distinct original high-threshold text components within foreground
  Chebyshev distance3. A connected center>=.50 region with exactly one retained
  peak may recombine kernel-supported partitions and its center rescue partition.
  Two distinct retained centers veto all unions, including transitive unions;
  kernel-only paths cannot undo splits of the same original component. No new
  foreground pixels. One crop per ID may include multiple disconnected contours.
- **`ASSUMPTION`:** Predeclare settings rather than sweep them. Reuse all125
  validation maps/983 care words and the existing selected checkpoints and pad0/
  no-filter/no-cap crop protocol. Compare contours, existing seeded watershed and
  refined crops. Freeze crop recipes before GT scoring, recognize even don't-care
  crops, then exclude only wholly invalid instances from metrics. Native merge/
  split and original bbox IoU>=.50 plus exact-casefold P/R/F1 remain exploratory
  holdout metrics. No threshold/checkpoint selection in this crop evaluation.
- **`ASSUMPTION`:** Exact RGB crop hashes allow existing OCR reuse under pinned
  weights/runtime; legacy decode omissions are reconstructed from immutable
  runtime defaults AR=True/refine1. Preserve provider protocol/record hashes,
  reject contradictory transcriptions and replay the first16 cached hashes on
  the selected device. Limited replay does not prove complete output equivalence.
- **`HARDWARE_DEVIATION`:** Cached segmentation maps avoid further GPU inference.
  PARSeq executes alone on an explicit CPU/GPU device with FP32/batch8 and memory
  metadata. CPU execution and reuse of historical GPU cache outputs are recorded
  separately; no inferred GPU-memory qualification from CPU measurements.
- **Remaining `UNRESOLVED`:** Benefit of unions, reliable center calibration,
  multi-seed/independent TEST behavior, official polygon metrics and the paper's
  exact PARSeq-S artifact. Strict claims and strict defaults remain unchanged.
- **Implementation/protocol:** `localize/word_guided_contours.py`,
  `scripts/evaluate_word_multitask_crops.py`,
  `configs/approximate/word_contour_refinement_v1.json`,
  `docs/WORD_GUIDED_CONTOURS.md`. Focused tests and actual frozen-map/crop
  evaluation qualify the new variant; no previous run is repeated or overwritten.

#### WM-002 completed crop/recognition qualification — 2026-10-09

- **Measured evidence:** All125 images /983 care words processed for seven
  frozen-map crop pipelines. Simple findContours exact-casefold IoU>=.50 matches
  are425/463/428 for Text/Text+Kernel/Text+Kernel+Center. Existing seeded methods
  give480/465 for two/three heads; refined methods give481/465. The best two-head
  refined original-coordinate bbox F1 is.616152 and exact end-to-end F1.479561,
  with exact all-GT recall481/983=.489318. Source checkpoints remain unchanged.
- **Measured refinement:** Seven shared-kernel unions for two heads and one for
  three, across six images. Zero center-region unions on these maps. Two-head
  split GT words decrease95→93; merged GT words remain171. Extra exact matches
  versus existing seeded cropping: gained1/lost0; the newly qualifying word is
  ELIZABETH from prior ELI/ZABETH crops. Three-head exact matches are unchanged.
  This is a small additional improvement; existing seeded cropping supplies
  most of the gain. Counts of unions are not counts of recovered words.
- **Paired regressions:** Two-head refined versus simple contours gains28/losses10
  exact GT words, net+18; three heads gain56/losses19, net+37. Preserve these
  losses and every image in the published per-image outputs.
- **Execution (`HARDWARE_DEVIATION`):** Actual CPU PARSeq FP32/batch8/threads2,
  segmentation inference not repeated.3,207 distinct RGB crops:2,949 new OCR
  records and258 from qualified historical GPU caches; all16 deterministic
  current-device cache replays reproduce strings. No OOM; GPU peaks null.
  The restricted environment does not qualify new CUDA behavior.
- **Verification:**111 focused/related tests pass, including25 new tests.
  Real125-image evaluation and completed-phase reuse pass. Independent audit
  checks875 sample/pipeline pairs,2,625 crop/instance/visual files and3,207 OCR
  records, recomputing native/original geometry and RGB hashes. An initial
  synthetic test exposed an unmerged center-only rescue partition; the guarded
  sole-center partition inclusion fixed it before the real experiment.
- **Remaining `UNRESOLVED`:** Independent TEST/generalization, multi-seed
  variance, actual new CUDA-memory qualification, center calibration/isolated
  contributions, official polygon scoring and exact paper PARSeq-S artifact.
- **Records:** `docs/WORD_GUIDED_CONTOURS.md`,
  `provenance/word_contour_refinement_v1_results.json`,
  `artifacts/word-multitask-crop-refinement-v1/recognition_results.json`;
  all8 union groups have a lossless user-review archive, without GT-based selection.

### ETD-001 — External lightweight text detectors with the qualified PARSeq decoder

- **Scope/date:** User-authorized CRAFT/similar model comparison on Total-Text
  TEST,2026-10-08. **APPROXIMATE only**; no Table3/5 or strict reproduction claim.
- **Evidence inspected:** Local paper sections3.2/3.3/3.5 specify attention
  segmentation, component crops and PARSeq-S. Existing practical U-Net,
  qualified PARSeq adapter, direct YOLO evaluation, balanced OCR admission,
  annotation preparation and their frozen results were inspected first.
- **`PAPER_SPECIFIED`:** Recognition from localized image patches is part of the
  paper. CRAFT, DBNet and FAST are not specified components of that pipeline.
- **`REPOSITORY_SPECIFIED`:** Reuse qualified official PARSeq source/checkpoint,
  casefold/punctuation/symbol normalization, legacy annotation clipping,
  one-to-one word+bbox matching and the prior balanced OCR confidence rule.
- **`ASSUMPTION`:** Versioned EasyOCR1.7.2 CRAFT general checkpoint and
  docTR0.11 DBNetResNet50/FAST-Tiny detector checkpoints; detector-only interfaces,
  native preprocessing, configurable upstream thresholds, strict state loading.
  Upstream MD5/filename SHA256 prefix checked before freezing full artifact SHA256.
  Installed implementation source hashes accompany experiment configuration.
- **`ASSUMPTION`:** Compare original RGB envelopes against long-axis quadrilateral
  rectification, no added padding. Original envelopes remain matching geometry.
  Pick one crop by balanced OCR confidence, with first-view tie breaking; test
  the already declared joint.6/same-reading-NMS.5 controls. CRAFT's native maximum
  Gaussian heatmap regression score is retained and explicitly clamped to[0,1]
  only for admission. Scores across detector families are not calibrated alike.
  Degenerate/outside polygons become explicit rejected detector records, not
  swallowed errors or fabricated crops. Malformed/nonfinite predictions fail.
- **`ASSUMPTION`:** Micro bbox word precision/recall/Hmean, IoU>=.5 and exact
  normalized text, all care words/images retained. Ignored GT is excluded but
  predictions on ignored regions remain FP, preserving the earlier local
  comparison. This is not the official polygon spotting protocol. All controls
  are reported; no TEST-fitted default or generalization claim is introduced.
- **`HARDWARE_DEVIATION`:** Sequential detector/PARSeq residency, FP32, detector
  batch1, PARSeq batch32, single explicitly selected CPU/CUDA device. FAST uses
  docTR's inference reparameterization; before/after counts are recorded. Stage
  CUDA allocated/reserved peaks, shapes, device and OOM reports are saved. CPU
  execution has null CUDA measurements, not inferred memory estimates.
- **`UNRESOLVED`:** docTR checkpoint training-data overlap with Total-Text;
  exact paper PARSeq-S artifact; official polygon protocol comparability;
  actual new detector inference/accuracy and target-GPU resource qualification.
  The practical U-Net58,155,081-parameter reference was counted locally; model
  size alone does not establish FLOPs or speed.
- **Qualification:** Focused local tests and annotation audit are completed:
  TEST300/2,204 care words match prior frozen GT exactly;343 ignored words and20
  clipped polygons are explicit. Local isolated dependency installation failed
  because PyPI DNS is unavailable; CUDA is unavailable in this process. No new
  pretrained detector result is reported. Host command runs checked downloads,
  real one-image smoke tests and the complete TEST evaluation with no fallback.
- **Alternatives considered:** EasyOCR DBNet needs legacy DCN compilation;
  docTR provides a pure PyTorch path on the qualified main stack. Full EasyOCR
  or docTR OCR would confound detector and recognizer differences. The original
  FAST paper's polygon postprocessing and curved rectification are not silently
  substituted for docTR's actual implementation or the generic rectangle crop.
- **Implementation:** `docs/TOTALTEXT_TEXT_DETECTORS.md`,
  `configs/approximate/totaltext_text_detectors_v1.json`,
  `localize/external_text_detectors.py`,
  `scripts/prepare_totaltext_text_detectors.py`,
  `scripts/evaluate_totaltext_text_detectors.py`,
  `scripts/test_totaltext_text_detectors.sh`.
- **Runtime correction/qualification update:** Host-provided pinned packages
  and weights are now available locally. An incomplete `UNKNOWN.egg-info`
  exposed a metadata sorting error: missing Name/Version values are now recorded
  explicitly under `package_metadata_anomalies`, with valid package/version pairs
  stored as JSON-native lists for stable cached configuration comparison.
  Required runtime version checks remain fail-closed. **`REPOSITORY_SPECIFIED`:**
  inspection of installed docTR0.11 DBNet/FAST postprocessors establishes rotated
  outputs as four normalized vertices plus `(0, score)`; the adapter now reads
  that confidence column and scales only the vertices. The initial duplicated
  score-row assumption was incorrect and is superseded by this source evidence.
  27 focused tests pass, including both real postprocessor interfaces; one
  dependency deprecation warning remains. Real CPU FP32 one-image smoke tests
  and a cached rerun complete for CRAFT, DBNetResNet50 and FAST-Tiny with PARSeq,
  under a new output profile preserving the earlier diagnostic failure.
  The corresponding inference parameter counts are20,770,466/25,412,162/8,521,920.
  New detector CPU smoke inference is therefore qualified; complete TEST
  accuracy, target-GPU memory and other listed ambiguities remain unresolved.
  Smoke report:
  `/tmp/totaltext-text-detectors-v1/metadata-fix-qualification-v3/cpu/smoke-1/results.json`.
- **Full CPU qualification update:** All three pretrained detectors completed
  Total-Text TEST300/2,204 care words with genuine PARSeq, CPU FP32/threads4,
  detector batch1/PARSeq batch32 and the frozen predeclared controls. Common
  best-OCR-crop/joint.6/same-reading-NMS.5 results: CRAFT1,612 correct/2,443
  outputs/Hmean69.38%; DBNetResNet501,134/1,603/59.57%; FAST-Tiny1,347/3,874/44.32%.
  All60 controls, per-image readings and provenance are retained. An independent
  SciPy matcher verified every control, all GT denominators and11,393 original
  RGB crop checksums. One outside FAST region is explicitly rejected and
  reported; no care image/word is omitted. No TEST-specific parameter change is
  introduced. Earlier full CPU accuracy uncertainty is now resolved for this
  exact approximate local-bbox experiment; target-GPU memory/outputs, official
  polygon protocol, checkpoint training overlap and strict paper fidelity remain
  unresolved. Results:
  `/tmp/totaltext-text-detectors-v1/metadata-fix-qualification-v3/cpu/full/results.json`.

## YTR-001 — Frozen YOLO/PARSeq transfer to additional TEST datasets

- **Evidence inspected:** authoritative `papers/2503.15639v2.pdf`, sections
  3.3/3.5; existing YOLO direct-crop and confidence selection code; qualified
  ICDAR13 Task1 / prepared ICDAR2015 / CTW1500 E2E manifests and dataset parsers.
- **`PAPER_SPECIFIED`:** the paper localizes segmentation components and uses
  PARSeq-S; object-detector localization is an external approximation experiment.
- **`REPOSITORY_SPECIFIED`:** reuse the pinned one-class YOLOv8s TextOCR weights,
  qualified genuine PARSeq artifact, BGR detector/RGB recognizer conventions,
  exclusive integer crop geometry and existing confidence/NMS selection.
  ICDAR13 has 233 TEST images/1,095 care words; ICDAR2015 has 500 TEST images;
  qualified CTW1500 has 500 TEST images/2,687 care text lines.
- **`ASSUMPTION`:** freeze original 640 input, raw score controls0.05/0.25/0.4,
  joint0.6 and same-text NMS0.5 before new TEST evaluation; no dataset-specific
  tuning. Micro normalized exact transcription + bbox IoU>=0.5, maximum
  one-to-one matching; casefold and remove Unicode punctuation/symbols, preserve
  whitespace. Ignore GT excluded; predictions on ignored regions remain FP.
  CTW1500 is scored separately against complete lines, without fabricated word
  boxes or GT-driven prediction grouping. This is a local bbox proxy.
- **`HARDWARE_DEVIATION`:** explicit CPU/GPU, FP32, detector batch1/PARSeq batch32,
  CPU threads4 and sequential model residency. Per-stage GPU peaks and failures
  are recorded. This session cannot access CUDA; CPU smoke qualification and
  user-run GPU/full TEST results must be distinguished.
- **`UNRESOLVED`:** strict PARSeq-S identity, official polygon/ignore handling,
  external checkpoint training overlap and new full TEST/GPU performance until
  executed. The supplied command uses already-qualified local packages/weights;
  no substitution, dependency installation or strict default change occurs.
- **Alternatives:** adapting Total-Text-specific frozen source scripts would
  couple other datasets to unrelated segmentation/GT shards; the isolated runner
  instead reuses their validated helpers. Splitting CTW GT into fictitious word
  boxes or using GT to combine predictions would not evaluate the deployed
  method; both alternatives are excluded from this transfer experiment.
- **Implementation:** `docs/SCENE_TEXT_YOLO_TRANSFER.md`,
  `scripts/evaluate_scene_text_yolo_parseq.py`, `scripts/test_scene_text_yolo.sh`,
  `configs/approximate/scene_text_yolo_parseq_transfer_v1.json` and its local path
  profile. Complete new TEST results remain pending until the GPU command runs.
- **Local qualification:** 18 focused tests pass after correcting an initial
  CSV bracket syntax error. Real CPU smoke runs completed all three datasets;
  a cached rerun computed zero new predictions. Independent SciPy matching
  verified 21 controls and45 RGB crop hashes. ICDAR2015's first test image has
  no care labels, so an additional two-image/65-crop partial run explicitly
  verified matching on image2's ten care words. Full identities are qualified
  as ICDAR13:233/1,095 care words; ICDAR2015:500/2,077 care words/3,153 ignored;
  CTW1500:500/2,687 care lines/388 ignored. The upstream Ultralytics load warning
  remains visible. These checks do not qualify complete TEST or target GPU
  metrics. Artifacts: `/tmp/scene-text-yolo-transfer-qualification-v1/cpu/`.
- **Completed GPU qualification supersedes pending accuracy above:** all1,233
  images completed on RTX3090, FP32/YOLO batch1/PARSeq batch32 with no failures.
  Frozen joint0.60 gives ICDAR13:862 correct/1,038 outputs/Hmean80.83%;
  ICDAR2015:1,401/2,336/63.49%; CTW1500 lines:1,106/5,245/27.89%. Independent
  SciPy matching checked8,631 controls, all25,992 RGB crop hashes and all GT
  denominators. Maximum recorded allocation/reservation288.52/390MiB does not
  qualify RTX3070 behavior. Official polygon evaluation and strict reproduction
  remain unresolved. Artifacts: `/tmp/scene-text-yolo-transfer-v1/cuda-0/full/`.

## YTS-001 — Cached confidence thresholds and CTW1500 line incompatibility

- **Evidence inspected:** paper sections3.3/3.5; qualified YOLO transfer outputs,
  original confidence/NMS/matching implementations; qualified PARSeq runtime
  `configs/main.yaml`, `configs/charset/94_full.yaml` and model construction.
- **`REPOSITORY_SPECIFIED`:** retain all completed TEST proposal boxes, direct
  PARSeq predictions, normalized text and local bbox IoU0.5 micro matching.
  PARSeq's94-character vocabulary has no whitespace; source data processing
  removes whitespace and its max label length is25. CTW GT remains text-line
  granularity; no word GT geometry is invented.
- **`ASSUMPTION`:** analyze detector confidence and existing joint-confidence /
  same-text NMS with one common grid0.05..0.99/step0.01. Search TEST maxima by
  Hmean, then correct count, then lower threshold. A shared control must improve
  all baselines and maximizes the smallest absolute Hmean gain. These are
  TEST-selected diagnostics, not new validation or dataset-specific deployed
  defaults; the existing transfer config remains frozen at0.60.
- **`ASSUMPTION`:** diagnose missed multi-token lines using proper contiguous GT
  token fragments, requiring prediction-area overlap>=0.5 with the line's axis
  bbox. GT informs diagnostics only. No GT-derived prediction grouping or ignore
  filtering is introduced; predictions on ignored regions remain FP.
- **Measured findings:** CTW care GT comprises1,599 single-token lines,1,086
  multi-token lines and2 normalized-empty labels. Baseline exact counts are
 1,106/0/0. Of the missed multi-token lines,1,007 have at least one correct
  fragment and858 have two distinct fragments. Strings such as HARBOR POINT are
  read as separate HARBOR and POINT outputs; a single decoded PARSeq output
  cannot express their required space. The2,687 lines contain4,546 whitespace
  tokens, a descriptive count without word geometry. Scalar confidence cannot
  resolve this word/line output mismatch.
- **`REPOSITORY_SPECIFIED`:** deterministic descending-score NMS admits a nested
  threshold filter after NMS equivalent to thresholding first. All baseline IDs
  and matching counts are checked. Since OCR_quality<=1, sqrt(detector_score) is
  an upper bound on joint confidence: at0.60, only9,678 saved proposals remain
  eligible before OCR, preserving every currently admitted output. This bound
  is a cached-score budget diagnostic; a new GPU batching run is not claimed.
- **Full cached results:** joint Hmean maxima: ICDAR13 threshold0.69/81.96%;
  ICDAR2015 threshold0.68/65.06%; CTW lines threshold0.62/27.93%. Shared0.62 gives
  81.31/64.32/27.93% and loses5/22/18 correct units relative to0.60 while removing
  25/125/140 outputs. Lower0.50 instead recovers21/85/62 correct units and lowers
  Hmean. No TEST-selected setting replaces the earlier baseline automatically.
- **Qualification:** 16 focused tests pass. Independent confidence, greedy NMS,
  normalization, IoU and SciPy matching validate all234,270 per-image controls,
 25,992 candidate scores, denominator counts and aggregate P/R/Hmean. No new
  model inference occurs; original GPU provenance and per-image source hashes
  are retained. Runtime and vocabulary identities are verified against source.
- **`UNRESOLVED`:** independent validation of selected thresholds, GT-blind
  line grouping, actual pre-OCR batch rerun, detector floor below0.05, proposal
  NMS changes and official polygon/ignore protocol remain separate experiments.
- **Implementation:** `docs/SCENE_TEXT_YOLO_THRESHOLDS.md`,
  `scripts/analyze_scene_text_yolo_thresholds.py`,
  `scripts/test_scene_text_yolo_thresholds.sh`,
  `configs/approximate/scene_text_yolo_threshold_sweep_v1.json` and local profile.
  Results: `/tmp/scene-text-yolo-thresholds-v1/results.json` and independent audit.

## YFC-001 — Add Total-Text to confidence comparison; estimate per-image FLOPs

- **Evidence inspected:** paper sections3.3/3.5/Table4, current confidence code,
  all frozen detector/OCR/GT source plans, qualified PARSeq transformer/decoder
  code, installed Ultralytics8.1.0 preprocessing and source checkpoint identities.
- **`REPOSITORY_SPECIFIED`:** add only Total-Text's original640/direct/no-added-pad
  crops to the three existing datasets. Reuse normalized exact text/bbox IoU0.5,
  maximum one-to-one matching and unchanged ignore policy. Canonical border
  clipping reproduces every physical Total-Text care transcription/bbox exactly:
 300 images/2,204 care words/343 ignored/20 clipped polygons. Highres, tiling and
  alternate crop views are not mixed into this comparison.
- **`HARDWARE_DEVIATION`:** Total-Text recognition is stored CPU FP32 with prior
  RTX3090 detector inference; other completed recognition runs are RTX3090 FP32.
  New read-only compute profiling is CPU FP32/thread4 with sequential model
  residency, detector batch1 and stored per-image OCR batch32 ordering. CPU
  profiling is not target GPU latency/memory qualification.
- **`ASSUMPTION`:** transfer the same95-value detector/joint threshold grid to
  Total-Text. Per-dataset and shared maxima use TEST labels and remain exploratory.
  Shared0.62 improves all four baselines; Total-Text Hmean72.44->72.78%, with
  its individual0.64 maximum73.08%. Original0.60 configuration is not overwritten.
- **`ASSUMPTION`:** count2 FLOPs/MAC for dense convolutions/linears, full MHA
  projections and ViT/decoder QK+AV products. Exclude all pointwise, bias,
  activation, normalization, softmax, embedding, preprocessing and postprocessing
  work. This convention differs from the earlier THOP-supported scope; fused
  square640 YOLO is28.4327424G versus earlier28.65G. The comparison is a dominant
  arithmetic estimate, not a complete operator total or exact Table4 reproduction.
- **`REPOSITORY_SPECIFIED`:** use recorded detector tensor shapes and qualified
  auto LetterBox/stride32 for Total-Text shapes, checking all1,233 recorded inputs.
  Profile genuine unchanged AR/refine1 PARSeq using original RGB pixels/crop
  checksums and official transform. Functional MHA projections and fused ViT
  attention must be explicitly counted; MHA child hooks are excluded to prevent
  double counting.
- **`ASSUMPTION`:** estimate OCR compute from all-corpus crop density and measured
  mean per-crop cost on four evenly spaced source images/dataset, chosen without
  GT. All16 images/264 crop readings match cached strings. A separate forced26
  step profile is a conservative counted-arithmetic budget8.207660544G/crop,
  not production early EOS or an accuracy result. Saved ranges are not confidence
  intervals. With current all-proposal OCR, estimated means are ICDAR13:93.8G,
  ICDAR2015:181.4G, Total-Text:224.8G, CTW:197.3G per original image.
- **`ASSUMPTION`:** hypothetical pre-OCR detector-score bound at joint0.60 gives
  estimated52.6/60.7/83.6/100.8G per image, retaining frozen admitted candidates
  while holding sampled OCR cost fixed. New batching/EOS behavior and numerical
  output equality require execution qualification; no cheaper production run is
  claimed.
- **Qualification:** 13 focused tests pass. Independent SciPy audit checks all
 57,000 new Total-Text per-image controls/8,318 candidate scores; prior3-dataset
  audits cover234,270 controls. The comparison covers1,533 images/34,310 crops.
  Independent TorchDispatch dense operator accounting matches both genuine
  checkpoints exactly: YOLO14,216,371,200 MACs, PARSeq full-length4,103,830,272.
  Initial validations exposed out-of-image annotation borders, profiler scope
  mismatches involving attention scaling, and an independent audit that omitted
  atomic inference-mode Linear/Conv2d/SDPA operators. Canonical clipping and
  matching the declared dense scope resolve those issues; no error is suppressed,
  no original GT/output is changed. The partial v1 artifact is preserved, and
  the completed profile uses v2. Upstream dependency/load warnings remain.
- **`UNRESOLVED`:** exact conditional GPU FLOPs for every crop, all non-MAC
  operators, RTX3070 hardware behavior, actual pre-OCR optimized batch rerun,
  official polygon evaluation, strict PARSeq-S identity and Table4 profiler
  compatibility remain outside this approximation experiment.
- **Implementation:** `docs/YOLO_PARSEQ_FOUR_DATASET_COMPARISON.md`,
  `scripts/compare_yolo_parseq_four_datasets.py`, `scripts/neural_macs.py`,
  `scripts/test_yolo_four_datasets.sh`, approximate four-dataset scientific and
  local path profiles. Results and independent audits:
  `/tmp/yolo-parseq-four-dataset-comparison-v2/`.

## CWR-001 — User-requested CTW1500 word recovery instead of exact full lines

- **Evidence inspected:** authoritative paper sections3.5/4.1/Table5; frozen
  CTW1500 TEST care/ignore line transcriptions and geometry; source YOLO/PARSeq
  selection and the earlier line-fragment diagnostic. The paper does not
  evaluate CTW or specify this protocol.
- **`ASSUMPTION`:** evaluate whitespace-delimited occurrences after Unicode
  casefold and punctuation/symbol removal. Preserve repetitions and derive the
  word denominator from all 500 images. Two normalized-empty care lines yield
  zero tokens and are explicitly counted; ignored lines are excluded. Empty
  predictions count as one FP. Failed samples are never omitted.
- **`REPOSITORY_SPECIFIED`:** reuse the existing line-fragment support rule:
  intersection area divided by predicted rectangle area must be at least0.5
  within the GT line axis bbox. This permits a word-sized box inside a long line
  without inventing word boxes or applying word-to-line IoU as word IoU.
- **`REPOSITORY_SPECIFIED`:** maximum one-to-one matching and unchanged frozen
  YOLO/PARSeq confidence selection. GT enters only the evaluator, with each
  prediction and target word occurrence matched at most once. Report an image
  lexical multiset control separately. This does not group or improve outputs.
- **`ASSUMPTION`:** micro token precision/recall and harmonic F1/Hmean measure
  line-supported word recovery. Reuse the previous prefilter controls at fixed
  joint0.60/NMS0.5 without fitting thresholds to the new TEST score. Keep
  ignored-region predictions as FP to preserve the earlier ignore handling.
- **Alternatives considered:** exact full-line evaluation is unsuitable for a
  single-word recognizer when the desired target is word recovery; image-wide
  bag-of-words matching loses spatial evidence; proportional GT word boxes
  would fabricate unknown character positions. Preserve a separately versioned
  line-supported diagnostic and the historical line results.
- **`UNRESOLVED`:** exact word positions inside lines and the ability to tell
  repeated real words from duplicated detections; full polygon support, official
  word-level protocol and ignore-region prediction treatment. No official
  CTW spotting or state-of-the-art claim follows from this word proxy. Strict
  reproduction is unchanged.
- **Implementation:** `recognition/line_token_evaluation.py`,
  `scripts/evaluate_ctw1500_word_recovery.py`, `scripts/test_ctw_words.sh`,
  isolated approximate configs and `docs/CTW1500_WORD_RECOVERY.md`. The runner
  verifies source identities and baseline ID equivalence, then checks every
  word count with independent Unicode/rectangle code and SciPy matching.
- **Qualification/results:** 14 focused tests and a real one-image smoke pass;
  all 500 TEST images are rescored with no failures. Independent matching checks
  all3,000 controls. The complete care denominator is4,546 words from2,687 lines,
  including two explicitly reported normalized-empty lines;388 ignored lines
  are excluded. Joint0.60 at floor0.05 or safe0.36 recovers3,508 words, with
  5,245 outputs, precision66.88%, recall77.17%, recovery F1/Hmean71.66%. The
  safe floor retains5,691 of12,594 proposals and exactly the same selected IDs.
  The lexical multiset control without geometry counts3,530. All original
  models/outputs are unchanged. Artifacts: `/tmp/ctw1500-word-recovery-v1/`.

## CWA-001 — Audit CTW word recovery and distinguish an analytic cutoff from TEST F1

- **Evidence inspected:** paper sections3.5/4.1 and current metric/selection code;
  all500 physical CTW TEST annotations, original detector/OCR shards, cached GT,
  historical word controls and IEEE floating-point boundary examples.
- **`REPOSITORY_SPECIFIED`:** retain joint0.60, same-reading NMS0.5 and the
  line-supported word-occurrence protocol. Derive the necessary pre-OCR bound
  from OCR quality<=1, using `sqrt(detector_score)>=joint_threshold` directly.
  A nominal squared cutoff is0.36, but its rounded comparison can reject a
  floating-point tie. Unit proofs at joint0.10/0.60/0.64 reproduce this numerical
  issue; the direct bound retains those admissible candidates. The historical
  dataset's output IDs and metrics are unchanged.
- **`REPOSITORY_SPECIFIED`:** validate cached care/ignore GT against physical
  transcriptions and canonical geometry, and all OCR candidates against original
  detector slot IDs/boxes/confidences and crop shapes. Mirror the source parser's
  clipping convention;16 raw polygons are clipped at parse time, before the
  source's post-load count. Record raw clipping separately. Reject contradictory
  metric metadata, duplicate IDs and noninteger occurrence counts explicitly.
- **`ASSUMPTION`:** audit the same95-value detector-floor grid at fixed joint0.60,
  retaining the analytic bound as the primary method. Identify the largest
  tested cutoff preserving every reference output separately from the highest
  TEST F1. Maxima use TEST labels and are exploratory; no deployment or strict
  default is changed using those labels.
- **Measured results:**0.36 is the largest tested output-preserving cutoff:
  5,691 crops,3,508 correct/4,546 words,5,245 outputs, F1/Hmean71.66%. TEST F1
  peaks at0.41:5,343 crops,3,448 correct,5,068 outputs, precision68.03%,
  recall75.85%, F1/Hmean71.73%. Saving348 additional crops loses60 correct
  words; the F1 gain is only0.071 percentage points. The source baseline's
  minimum admitted detector score is0.360975056886673, consistent with the
  output-preserving grid result; it does not establish a new universal bound.
- **Qualification:**26 focused tests and real smoke pass. All48,000 word
  controls across500 images agree with independent Unicode/rectangle code and
  SciPy matching. Every one of the prior3,000 controls has identical counts,
  selected IDs, matching edges and GT matches. No model inference is repeated,
  no failed sample omitted and no scientific error suppressed.
- **`UNRESOLVED`:** official word geometry/ignore evaluation, positions of
  repeated words, ambiguous line support and independently validated F1 tuning.
  The baseline has13 predicted tokens eligible for several line boxes and26
  within-line repeated GT occurrences,15 of which are matched. One-to-one
  caps counts without supplying the missing word geometry. The method remains
  an approximate recovery diagnostic, with the limitations recorded in CWR-001.
- **Implementation:** audited word-recovery runner/module, separate audit v2
  approximate profiles, existing `scripts/test_ctw_words.sh` command and
  `docs/CTW1500_WORD_RECOVERY.md`. Completed artifacts and review audits:
  `/tmp/ctw1500-word-recovery-audit-v2/`.

## DCE-001 — Neutralize don't-care predictions after frozen selection

- **Evidence inspected:** authoritative paper sections3.5/4.1; current frozen
  four-dataset YOLO/PARSeq and CTW word evaluation, physical TEST annotations,
  and native benchmark functions in DeepSolo's `text_eval_script.py` and
  `text_eval_script_ic15.py`. The paper does not specify ignore-region semantics.
- **`ASSUMPTION`:** this user-authorized external YOLO experiment adopts the
  inspected benchmark's ignore-neutralization rule in APPROXIMATE mode. It
  updates evaluation only and supplies no strict/Table5/official benchmark claim.
- **`REPOSITORY_SPECIFIED`:** the inspected native evaluator neutralizes a
  prediction if intersection with any single ignored GT polygon divided by
  prediction area is strictly greater than0.5. Use native Polygon3, original
  annotation vertices, no polygon union, no enclosing-box approximation, and
  no per-vertex clamping. Ignore exclusion precedes care matching, including
  ambiguous care/ignore overlaps. Exactly0.5 is not excluded.
- **`REPOSITORY_SPECIFIED`:** keep original detector boxes, crop RGB hashes,
  PARSeq readings, confidence selection, analytic pre-OCR bound and NMS. Apply
  ignore decisions only to final predictions in the scorer; do not change image
  pixels, crop geometry or recognition inputs using GT. Do not rerun NMS or
  resurrect suppressed candidates after GT-dependent neutralization. Original
  source runs and previous metrics remain separately preserved.
- **`ASSUMPTION`:** retain prior normalized exact-word bbox matching for
  ICDAR13/ICDAR2015/Total-Text and line-supported word occurrences for CTW
  (CWR-001/CWA-001). Micro F1 and Hmean coincide. The standalone ignore update
  cannot convert these care-matching proxies into official end-to-end scores.
- **`ASSUMPTION`:** run pure CPU polygon geometry in the existing isolated
  Python3.8 environment with Polygon3 3.0.9.1/Shapely2.0.0 and extension hashes.
  Shapely provides an independent audit; `make_valid` is restricted to that
  explicitly logged audit, never primary Polygon3 inputs. Reject differing
  areas/strict-threshold decisions. Degenerate ignored polygons are retained and
  logged: Total-Text has one single-point ignored region with zero coverage.
  No DeepSolo model/environment is modified and no dependencies are installed.
- **Alternatives considered:** GT-based pixel masking or pre-OCR cropping would
  leak annotation knowledge into inference; ignored polygon bboxes/unions and
  vertex clamping can change exclusion geometry; a full official evaluator
  would additionally change care matching/text rules and CTW granularity, beyond
  this isolated update. Preserve unchanged readings and implement exclusion in
  metrics using original ignored polygons. Eight Total-Text and five CTW ignore
  polygons contain out-of-frame vertices, which native intersections handle.
- **Measured full TEST results:** at joint0.60 and analytic prefilter
  (nominal0.36), ICDAR13 stays862/1,095 correct, P83.04/R78.72/H80.83%;
  ICDAR2015 stays1,401/2,077 correct, neutralizes541 of2,336 outputs,
  P78.05/R67.45/H72.37%; Total-Text stays1,597/2,204 correct,
  neutralizes155 of2,205, P77.90/R72.46/H75.08%; CTW word proxy
  stays3,508/4,546 correct, neutralizes507 of5,245,
  P74.04/R77.17/H75.57%. All baseline output IDs are preserved, no correct
  words lost at the primary setting; the improvement removes former FPs only.
- **Qualification:**55 focused tests, real four-dataset smoke and complete
  1,533-image rescoring pass. All145,083 intersections agree with independent
  geometry. All147,168 per-image cutoff controls are checked, with21,080
  distinct before/after selected-set scores audited using SciPy matching.
  The48,000 preceding CTW word controls remain unchanged before exclusion.
  No failed sample is omitted, model inference repeated or error suppressed.
- **`UNRESOLVED`:** full official care-polygon IoU/matching, official
  transcription/word-spotting/lexicon rules and exact GT revisions; CTW word
  positions and official word-level protocol. These limitations remain. The
  common selection cutoff is unchanged; maxima on the retained95-cutoff TEST
  grid are exploratory, not threshold validation or strict defaults.
- **Implementation/artifacts:** `scripts/evaluate_yolo_dont_care.py`, pure
  `scripts/dont_care_polygon_worker.py`, `recognition/dont_care_evaluation.py`,
  isolated approximate profiles and `docs/YOLO_DONT_CARE_EVALUATION.md`.
  Short command: `bash scripts/test_yolo_evaluation.sh`. Full artifacts under
  `/tmp/yolo-dont-care-word-evaluation-v1/`; prior word/audit runs are preserved.

## UCR-001 — Same U-Net resegmentation of RGB contour crops

- **Evidence inspected:** local paper sections 3.2–3.3, Figure 3 and Algorithm 1;
  existing `models/unet_adv.py`, `models/segmenter.py`, frozen segmentation,
  repository/native contour localizers and qualified Total-Text highres/
  letterbox preprocessing/checkpoint-loading implementations. Algorithm 1 has
  no second U-Net inference on localized crops. This user-requested procedural
  experiment is isolated in APPROXIMATE mode; existing scientific/default
  pipelines and STRICT profiles are unchanged.
- **`PAPER_SPECIFIED`:** foreground component-area filtering/ordering, cropping
  RGB from localized regions and a maximum of ten final blocks. Rectangle or
  contour-polygon area is not substituted for foreground component pixels.
- **`ASSUMPTION`:** exactly one additional forward per retained seed on an
  expanded original RGB crop, using the same model/weights/preprocessing.
  Expand each axis by `ceil(0.5 * padded_seed_size) + 4` original RGB pixels
  per side, clipped to image boundaries. These exploratory context values are
  neither paper values nor validated word-boundary thresholds.
- **`ASSUMPTION`:** threshold native sigmoid probabilities before nearest
  binary projection to original RGB dimensions; invert original square-canvas
  letterbox padding. This explicit new baseline localizes in original RGB
  coordinates, not the previous native128/native256 coordinate frame.
- **`ASSUMPTION`:** external contours, 8-connectivity, exact foreground-pixel
  areas excluding holes, deterministic xyxy tie ordering, minimum area zero,
  output padding zero, ten seeds/children/final blocks, exact padded-box
  deduplication only. Other overlapping hypotheses remain; neighboring text
  can enter a crop. Do not relocalize the union of overlapping ROI predictions.
  Output padding zero is an experimental departure, not a strict paper choice.
- **`REPOSITORY_SPECIFIED`:** the default experimental profile replays the
  existing highres letterbox input (512 RGB → 256 logits), fill (124,116,104),
  ImageNet normalization, skip floor 0.25 and threshold 0.5. The runner verifies
  checkpoint/config/run hashes and contracts and loads weights strictly with
  the existing loader. Author-protocol use instead requires the exact saved
  resolved training config and matching architecture/preprocessing. No
  checkpoint is downloaded and no scientific component is replaced.
- **`HARDWARE_DEVIATION`:** one-model, batch-one sequential inference; float32,
  no quantization/offload/mixed precision. The real engineering smoke uses CPU
  because CUDA is unavailable in this session, with four CPU threads recorded.
  GPU execution and physical RTX 3070 peak allocated/reserved memory remain
  unqualified. The runner records both peaks on CUDA and propagates/logs OOM;
  simulated OOM tests do not constitute physical GPU qualification.
- **`ASSUMPTION`:** compare full-image and crop masks using ROI component
  counts, seed-associated predicted components (any shared seed foreground
  pixel), retained seed fraction, mask IoU, added/removed pixels and border
  contact. These are review signals only. No heuristic acceptance threshold,
  automatic baseline substitution or word-correctness claim is introduced.
  Empty crop predictions stay empty; inference/load errors propagate. The
  raw ROI union is an artifact with unobserved zeros outside crops, not a full
  segmentation mask for evaluation or replacement of the baseline.
- **Alternatives considered:** immediate morphological merging could join
  distinct words; tight crops cannot recover absent neighboring fragments;
  automatic success claims from fewer components confuse recovery with fusion
  or missed text. Joint neighboring-fragment crops or a second context-margin
  agreement test are separate possible experiments, not enabled defaults.
- **`UNRESOLVED`:** whether zoomed RGB improves word-instance boundaries,
  which context margin preserves neighboring fragments without word fusion,
  how to resolve nonidentical overlapping hypotheses, and whether component
  disagreement predicts actual mistakes. A missed full-image component has no
  seed; text outside its crop cannot be recovered. TRAIN/validation word-level
  evaluation is required before tuning a policy or reporting improvement.
- **Real smoke evidence:** epoch-71 letterbox checkpoint SHA256
  `5fca2ffc15b9689abbd5d75e80710ddd38d6419d9610fed840b2c54bb01f3991`,
  official TRAIN RGB images 11 and 147, no annotations/OCR loaded. Image 11
  retains ten final proposals; image 147 changes from three to seven and
  visually fragments “MOTORS” further. This is a smoke-test observation, not
  a recognition/segmentation metric or a strict reproduction result.
- **Implementation:** `localize/unet_crop_refinement.py`, standalone
  `scripts/diagnose_unet_crop_refinement.py`, explicit
  `configs/approximate/unet_crop_refinement_v1.json`, focused tests and
  `docs/UNET_CROP_REFINEMENT.md`. Outputs include native probabilities, paired
  masks/RGB crops, mapped boxes, decisions, artifact/source/input/checkpoint
  hashes and complete resolved configuration; existing outputs are preserved.

## WCO-001 — Auxiliary dense word-center attraction

- **Evidence inspected:** local paper sections 3.2/3.3, Fig.3; existing practical
  U-Net, completed word/kernel/Gaussian-center implementation and result record.
  The paper supports foreground shape and component localization, but specifies
  neither this auxiliary signal nor an offset loss/voting protocol.
- **`ASSUMPTION`:** the user-authorized extension uses a separate practical
  frozen-ResNet50 attention U-Net to predict dense normalized x/y displacement
  toward each visible valid word centroid. Keep the original text U-Net as the
  shape source. Reuse its decoder; replace the output with a zero-initialized
  two-vector head. An optional strict warm start explicitly resets only the
  original prediction head and preserves the source. Checkpoint identity is
  distinct from two-channel text/kernel outputs.
- **`ASSUMPTION`:** construct vectors after exact paired owner-map transforms
  and clipping, without kernels or Gaussian targets. Use tanh-bounded offsets
  and SmoothL1, equal word means per image then supervised-image means. Ignore
  pixels/background have zero gradient; empty batches explicitly skip updates
  without dropping samples from counts. Record beta and the actual loss name.
- **`ASSUMPTION`:** predicted base foreground supplies voters, boundaries and
  flood cost. Component-local probability-weighted bilinear votes supply
  candidate centers, projected markers and dense agreement seeds. Smoothing,
  density cutoff, NMS and agreement radius are configurable exploratory choices.
  Invalid endpoint votes are rejected/counted; unseeded components remain intact.
  Preserve the exact base foreground union. The signal itself does not detect
  crops; optional downstream instance boxing is a separate existing operation.
- **Alternatives considered:** another Gaussian-center/kernel head would repeat
  an existing experiment; point-only geometric expansion failed the ideal fused
  rectangle test, so add dense agreement rather than weaken the test. A direct
  crop/box detector changes the user's requested role. Component-local voting
  avoids borrowing unrelated text cues but cannot reconnect disconnected word
  fragments. Concave centroids remain valid location cues and project to support.
- **`HARDWARE_DEVIATION`:** optional sequential execution/saved aligned maps on
  the 8GB single GPU differs from the paper runtime; record actual placement,
  precision and model set. CPU synthetic smoke has no real-dataset claims.
- **`UNRESOLVED`:** offset checkpoint training, thresholds/radius calibration,
  real-data merge/split and recognition benefit, official evaluation, physical
  RTX3070 VRAM/OOM qualification and simultaneous model feasibility. This task
  implements/tests the auxiliary signal, not a complete reproduction result.
- **Implementation:** `models/word_center_offsets.py`,
  `segmentation/word_center_offsets.py`, `localize/word_center_offsets.py`, isolated
  approximate profile, synthetic forward/backward profiler and focused tests.
  Protocol and qualification history: `docs/WORD_CENTER_OFFSETS.md`. STRICT
  configurations, existing models and previous experiment artifacts are preserved.

## WCO-002 — Total-Text offset-training launcher

- **Evidence inspected:** local paper sections 3.2/3.3 and Implementation Details;
  qualified Total-Text v4 TRAIN manifest, frozen split/preprocessing helpers,
  completed matched text-control run/checkpoint, existing optimizer/checkpoint
  infrastructure and the WCO-001 auxiliary-vector implementation.
- **`ASSUMPTION`:** train the auxiliary model independently for 100 epochs on
  the existing TRAIN1130/validation125 split with letterbox512/native256,
  seed202503 and a SHA-bound trained practical text-model warm start. Reset
  only its output head. Use the existing Adam LR1e-4/weight-decay.01/batch8 and
  record Adam betas/epsilon. This offset task, loss, spatial protocol, duration
  and warm start are exploratory choices, not strict paper specifications.
- **`ASSUMPTION`:** choose minimum word-balanced validation offset SmoothL1,
  earliest exact tie. Also report native-pixel endpoint error with equal word
  means per image then supervised-image means. Keep empty images in sample
  counts, with null objectives; report expected/surviving/lost words. No TEST,
  base-mask geometry or recognition score selects this auxiliary checkpoint.
- **`ASSUMPTION`:** strict architecture/state loading plus source-checkpoint,
  dataset-manifest, split, source-code and resolved-config hashes guard the
  run. Preserve best/last checkpoints, optimizer/RNG state and contiguous epoch
  history; recover a checkpoint-before-log interruption without retraining.
  Partial execution is IN_PROGRESS, not completion of the configured budget.
  Export selected validation vectors/targets with per-sample checksums.
- **`HARDWARE_DEVIATION`:** release the CPU-loaded base before auxiliary GPU
  training. Explicit CPU, batch-size and worker overrides are recorded; no
  automatic precision, batch, alternate model or CPU retry follows a failure.
  Log actual model set, precision, device, shape, batch and CUDA peaks/OOM.
- **Alternatives considered:** a command pointing to the old Gaussian/kernel
  runner would train a different task. Requiring an already-trained auxiliary
  checkpoint would not initialize this new signal. A mutable/downloaded encoder
  default would lose the selected baseline's provenance. Validation geometry
  would additionally require loading/predicting base masks and defining a new
  joint evaluation protocol; offset-only selection keeps this launcher focused.
- **`UNRESOLVED`:** full auxiliary training and physical RTX3070 qualification,
  calibrated vote settings, separation benefit using predicted base masks and
  complete official evaluation. Real-dataset smoke/preflight and synthetic
  resume tests do not resolve these scientific questions.
- **Implementation:** isolated approximate Total-Text training profile,
  `scripts/train_word_center_offsets.py`, wrapper word-count metadata and tests
  for validation aggregation, endpoint scaling, partial/resume equivalence,
  interruption recovery, strict provenance, earliest ties and OOM propagation.
  The command is in `docs/WORD_CENTER_OFFSETS.md`.


## TRR-001 — Text coverage loss and input resolution

- **Scope/evidence:** user requested recall-oriented text-mask training and
  higher input resolution, with separation supplied by the existing separately
  trained offset network. Inspected local paper sections3.2/3.3/4.1 and the
  qualified model, transforms, dataset, multi-task loss/evaluation, historical
  word-balanced/zoom/deep-supervision/partial-encoder experiments and saved maps.
- `PAPER_SPECIFIED`: ResNet50 attention segmentation backbone and BCE in the
  paper's256/128 training protocol. Those scientific paths remain unchanged.
- `REPOSITORY_SPECIFIED`: reuse qualified full-resolution v4 artifacts,
  frozen1130/125 TRAIN split, single-text epoch56 checkpoint, paired letterbox,
  validity/owner mapping, visualization and strict checkpoint/RNG helpers.
- `ASSUMPTION`: a2x2 continuation with BCE+Dice/BCE+Tversky and512/768 RGB
  inputs, native256/384 masks, no added parameters. Tversky usesFP=.3,FN=.7
  and smooth1; every other loss term stays matched. All branches use fresh
  Adam, LR1e-5 and50 additional epochs, with matched restarted augmentation.
- `ASSUMPTION`: score on identical canonical256 owner/validity targets by CPU
  bilinear sampling sigmoid probabilities; retain every expected word. Select
  >=90% coverage words at >=90% micro pixel precision over five predeclared
  validation thresholds; retain unqualified points explicitly, earliest epoch
  ties and epoch0. Component geometry/merges and OCR do not select checkpoints.
  The reused TRAIN holdout is exploratory, not independent TEST evidence.
- `HARDWARE_DEVIATION`: FP32/batch4 and sequential single-model execution,
  rather than paper batch8. Actual device, model set, input shape and CUDA
  allocated/reserved peaks are recorded; there is no automatic OOM fallback.
- **Alternatives:** existing per-word BCE/zoom/partial encoder were already
  implemented and tried, so are not duplicated or combined with Tversky here.
  Larger input was chosen before tiling to isolate scale with the same model.
  Native-grid scoring was rejected because it would change scoring targets.
- **Qualification:**157 focused/related tests, full1255-image target preflight
  at both sizes and four real FP32/batch4 CPU forward/backward/validation
  smokes pass. One training word lost at256 survives at384; validation remains
  983/983. No code/checkpoint of the auxiliary offset network was modified.
- `UNRESOLVED`: benefit after50 epochs, execution/VRAM/OOM behavior on the
  target RTX3070, independent TEST behavior, and combined offset/PARSeq results.
  CUDA is inaccessible in this session; synthetic OOM tests only qualify error
  propagation. No training benefit, official paper result or GPU fit is claimed.
- **Implementation/profile:** `segmentation/text_recall.py`,
  `scripts/run_text_recall_resolution.py`,
  `configs/approximate/text_recall_resolution_v1.json`;
  protocol and launch command in `docs/TEXT_RECALL_RESOLUTION.md`.


## WCO-003 — Joint shape and center-guided crops

- **Scope/evidence:** user explicitly requested combining the trained offset
  network with the base U-Net to estimate word positions/counts and partition
  or expand predicted masks for better RGB crops. Inspected both existing
  implementations, completed selected-checkpoint exports and local paper
  sections3.2/3.3. This is an experimental extension, not a paper claim.
- `PAPER_SPECIFIED`: segmentation supplies foreground; localization uses
  components/contours, component-area filtering/order, padding and top10 blocks.
- `REPOSITORY_SPECIFIED`: qualified base epoch56 and offset epoch82, matched
 512/256 preprocessing, frozen125 TRAIN holdout/983 care words, hash-verified
  native exports, existing exclusive-box geometry and real pinned PARSeq.
- `ASSUMPTION`: global endpoint voting with sigma3/density0.5/NMS8/support8,
  radius12 agreement, dense geodesic seeds and explicit unseeded fallback.
  Shared center identities can join disconnected fragments; multiple centers
  partition fused shapes. These settings were frozen before full scored fusion;
  center count remains a hypothesis, not guaranteed word count.
- `ASSUMPTION`: preserve every high-confidence0.65 core pixel; optionally grow
  into probability>=0.35 within3 Euclidean native pixels through three bounded
  connected steps, requiring endpoint agreement and RGB canvas membership.
- `ASSUMPTION`: compare baseline, previous local voting, global identities and
  growth using identical pad0/no area filter/no cap diagnostic cropping. Freeze
  recipes before GT scoring. Native any-shared-pixel merge/split, original box
  IoU>=0.5 maximum matching, count MAE and exact-casefold PARSeq/all983 words
  are exploratory metrics; ignore only fully invalid instances after inference.
- `HARDWARE_DEVIATION`: CPU fusion of previously generated RTX3090 FP32
  network outputs and CPU PARSeq instead of new simultaneous RTX3070 inference.
  Preserve source precision/memory records; no new joint GPU qualification.
  Qualified OCR reuse requires exact RGB hash and16-sample current-device replay.
- **Alternatives:** raw local peaks often duplicate nearby evidence and cannot
  reunite disconnected fragments; unconstrained dilation can merge words or
  include background. Annotation-derived counts/centers or oracle selection
  were rejected because the requested crop signal must operate at inference.
- `UNRESOLVED`: calibrated word-count accuracy, independent TEST benefit,
  official polygon/recognition results and simultaneous target-GPU execution.
  The reused TRAIN holdout and approximate auxiliary task cannot resolve these.
- **Implementation:** `localize/center_guided_word_crops.py`, isolated profile
  `configs/approximate/center_guided_word_crops_v1.json`, frozen-export runner
  `scripts/evaluate_center_guided_word_crops.py`; full protocol and limitations
  in `docs/CENTER_GUIDED_WORD_CROPS.md`. Focused synthetic tests plus complete
  paired real-validation scoring are required before reporting an improvement.
- `HARDWARE_DEVIATION`: optional fresh-image runner strictly loads both trained
  checkpoints sequentially in FP32/batch1 on an explicit device, records each
  model's memory, and replays matched qualified RGB preprocessing. No desired
  counts or annotations enter inference. CPU/GPU/batch numeric differences may
  affect threshold boundaries. See `scripts/predict_center_guided_word_crops.py`.
- **v1 qualification/result:**75 focused/related tests and fresh CPU inference
  with both selected checkpoints pass. Complete125-image/983-word scoring gives
  geometry F1 baseline59.069% versus joint growth46.840%, count MAE2.248 versus
 3.544, and real exact-casefold PARSeq correct words425 versus335. No aggregate
  benefit is established. All source/crop hashes and16/16 cached OCR replay
  checks pass; no sample failed or disappeared from the denominator. Fresh CPU
  versus archived GPU image14 counts match, with measured boundary differences.
  See `docs/CENTER_GUIDED_WORD_CROPS.md` and local v1 reports. These exploratory
  observations do not resolve calibration, TEST generalization or GPU residency.

## UWR-001 — Glyph/word and mask-quality experiments

- **Scope/evidence:** the user requested applying the researched mask-improvement
  ideas and jointly using the original letter-shaped GT and modified word GT.
  Inspected local paper sections3.2/3.3, original distributed pixel manifest,
  qualified v4 word-owner artifacts, model/augmentation/loss/localization code,
  completed Text/Kernel/Center and PixelLink/zoom/deep-supervision reports.
- `PAPER_SPECIFIED`: ResNet50 attention segmentation and connected-component
  localization remain the architectural family. Original/STRICT paths are not
  changed; these new heads are not specified by the target paper.
- `ASSUMPTION`: pair original binary glyph foreground and modified full-word
  regions by sample ID/RGB hash and exact indexed augmentation. Original GT
  lacks individual character IDs; connected components are not fabricated as
  character supervision. A shared decoder with a separate glyph head tests
  cooperation without loading two full networks. Glyph association never joins
  distinct word IDs; orphan/ambiguous components remain explicit diagnostics.
- `ASSUMPTION`: five branches warm-start strictly from qualified Text + Kernel
  epoch61. Four segmentation branches use matched BCE+Dice, fresh Adam LR1e-5,
  50 additional epochs, seed202503, frozen1130/125 TRAIN split, RGB512/native256,
  FP32/batch8, and epoch0-inclusive word-F1 selection. No TEST/OCR selects weights.
- `ASSUMPTION`: PAN-inspired4-dimensional embedding with pull/push losses and
  local4-neighbor kernel growth. No missing foreground is reconstructed;
  unassigned foreground becomes reported fallback instances. This is not a
  full PAN reproduction. DB-inspired learned threshold uses a declared native
  border target and soft binary loss adapted to full-word probability, rather
  than silently importing DBNet targets/backbone. All weights/margins/thresholds
  are isolated in `configs/approximate/unet_word_research_v1.json`.
- `ASSUMPTION`: quality branch freezes segmentation weights and BN, trains ROI
  features+predicted-mask regression for best-word bbox IoU, coverage and purity,
  selects minimum held-out quality MSE, and compares area10/quality10 ranking on
  identical candidates. GT labels enter only training/post-hoc scoring. TRAIN
  predictions are in-sample for the source segmenter; out-of-fold calibration
  is preferable but not supplied by this exploratory branch.
- `ASSUMPTION`: cached-map instability across.55/.65/.75 ranks suspicious
  instances without changing masks. Per-proposal best bbox IoU<.50 defines this
  diagnostic's bad-bbox label; it is not one-to-one word/OCR correctness. Compare
  against low mean pixel confidence, rather than assuming uncertainty wins.
- `ASSUMPTION`: main metrics reuse native bbox one-to-one IoU>=.50, all983 care
  words, any-pixel merge/split and wholly-invalid-only prediction exclusion;
  pad0/no area/no cap except separate ranking arms. OCR uses original RGB crops,
  the qualified approximate PARSeq pairing, AR/refine1, exact casefold and
  Unicode P/S-stripped alternatives, recording both gains and losses. These
  diagnostics do not establish official polygon metrics or Table3/Table5.
- `HARDWARE_DEVIATION`: sequential single-network branches and separate PARSeq
  process. Explicit CPU, batch/worker overrides and actual device, backend/TF32
  flags, shapes, allocated/reserved peaks and OOM are recorded. No automatic
  precision/device/model/batch change follows an error.
- **Alternatives:** blindly replacing every crop worsened the earlier51-image
  experiment. Local PixelLink was already tried and selected epoch0; Gaussian
  center had lower word F1 than Text + Kernel. Two independent full U-Nets would
  add another58M backbone and require separate residency qualification. Morphology
  alone cannot provide learned word ownership. A binary union would undo instance
  IDs. Global disconnected embedding grouping is a separate future adaptation.
- **Initial qualification:** full1255-image paired-target preflight and five real
  FP32/batch2 CPU forward/backward/validation smokes pass. Frozen125-image stability
  diagnostic reproduces all source labels, scores1029 care proposals and obtains
  AUC.8361 versus.8699 for low pixel confidence; top206 unstable proposals have
  bad-bbox precision87.86%, compared with39.26% overall. This diagnoses errors;
  it does not correct them or establish new-head performance.
- `UNRESOLVED`: full50-epoch gains, physical RTX3070 memory/OOM qualification,
  quality calibration, selective crop-reinference benefit, one-seed variability
  and independent TEST generalization. CUDA is inaccessible in this session.
- **Implementation:** `models/unet_word_research.py`,
  `segmentation/unet_word_research.py`, `localize/unet_word_research.py`, paired
  runner/stability/OCR/comparison scripts and host wrapper. Protocol and launch
  command are in `docs/UNET_WORD_RESEARCH.md`. Strict/RNG checkpoint resume,
  interruption recovery, warm-transfer compatibility, coordinate pairing,
  auxiliary gradients, empty/ignored predictions and OCR provenance are tested.


### TRR-001 follow-up — CUDA allocator initialization

The host first CUDA smoke failed before allocation/model transfer with
`RuntimeError: Invalid device argument` while resetting memory peaks. Inspected
the existing qualified allocator helper, current runner, recorded failure and
all run artifacts. There are no checkpoints, epoch logs or completed smokes.

- `REPOSITORY_SPECIFIED`: reuse
  `segmentation.evaluate._prepare_cuda_memory_tracking`, which selects the
  device, initializes/releases one allocation and resets integer-device peaks.
  No model/optimizer/precision/batch/targets or RNG samples change.
- `ASSUMPTION`: permit only the exact known original and corrected launcher
  hashes to transition a failed, pretraining contract; every scientific source,
  artifact and setting remains exact. Archive original metadata byte for byte,
  retain the original failure/preflight evidence, record transition hashes and
  commit the root contract last. Successful results/training artifacts, changed
  settings, unknown sources and other failures cannot use this path.
- Qualification:185 related tests pass; allocator-order/int-index and error
  propagation tests are mocked, not GPU execution. The actual failed host run
  was copied read-only into/tmp and its preflight/metadata recovery executed
  successfully. No external run directory is modified by this session.
- `UNRESOLVED`: physical CUDA execution and VRAM/OOM qualification remain
  host-only because CUDA is inaccessible here. No trained-model improvement or
  strict reproduction claim follows from this launcher fix.


## WCO-004 — Coherent word-center vote loss

- **Scope/evidence:** user requested implementing the first proposed improvement
  after a complete125-image failure analysis: change auxiliary vector learning
  before adding reliability heads or another architecture. Inspected existing
  v1 targets/loss/trainer, selected exports, local paper sections3.2/3.3 and
  prior Gaussian/embedding experiments. This is an approximate extension.
- `PAPER_SPECIFIED`: the paper describes foreground segmentation/localization,
  not this dense vector task, collective vote loss or word-centroid objective.
- `REPOSITORY_SPECIFIED`: preserve the qualified separate two-offset U-Net,
  frozen encoder/BN, paired targets/preprocessing, TRAIN1130/validation125,
  strict warm start, deterministic checkpoint/RNG and export interface. Keep
  both branches' historical minimum validation offset-loss selection.
- `ASSUMPTION`: word-level robust endpoint consensus plus squared-distance
  own-versus-nearest-rival hinge. Normalize new geometry by max(H,W), with
  margin0.1 of nearest GT spacing squared divided by spacing. Weights1/.5/.5
  and beta.01 are predeclared; equal word/image aggregation preserves v1.
- `ASSUMPTION`: coincident centers within.001 native pixels cannot be separated
  by xy; exclude only those rival pairs and count ambiguous words. Owner/target
  centroid alignment tolerance.001 and exact-target smoke-loss tolerance1e-6
  reject corrupted geometry without inventing or moving centers. One-word
  cases have no rival penalty; background/ignore retains zero gradient.
- `ASSUMPTION`: log per-word/image endpoint RMS scatter and strictly-nearer-rival
  vote rates along with all expected/surviving/lost words. These are GT-assisted
  validation diagnostics, not production word counts or official evaluation.
- `ASSUMPTION`: fresh matched control/coherent100-epoch branches start from the
  same text checkpoint and zero vector head; do not continue auxiliary epoch82.
  The only branch intervention is the training loss, with otherwise matched
  selection, data, seed, optimizer, duration and voting settings.
- `HARDWARE_DEVIATION`: one auxiliary resident at a time, CPU warm-start base
  released before training, explicitly recorded CPU/batch/worker overrides.
  Rival argmin uses2048-pixel chunks without candidate autograd retention;
  selected distances retain gradients. Record actual precision/model/input and
  allocated/reserved CUDA peaks/OOM without automatic fallbacks.
- **Alternatives:** a new Gaussian head or generic learned embedding would
  repeat existing components and conflate this intervention. Jointly changing
  backbone, vote reliability, postprocessing or selection would prevent a
  loss-only attribution. Unsquared distance contrast was rejected after a
  restoring-gradient test exposed its flat region beyond rival centers.
- `UNRESOLVED`: full trained center/crop/OCR benefit, vote reliability head,
  calibrated count selection, independent TEST behavior and physical GPU fit.
  Implementation/target/gradient/resume/CPU smokes cannot resolve these.
- **Implementation:** `segmentation/word_center_votes.py`, isolated v2 profile
  `configs/approximate/word_center_votes_totaltext_train_v2.json`, runner
  `scripts/train_word_center_votes.py`; protocol/commands in
  `docs/WORD_CENTER_VOTE_LOSS.md`. Historical v1 code/weights remain unchanged.
- **Qualification:**104 focused/relevant tests and both real FP32/batch2 CPU
  forward/backward smokes pass, with matched RGB order/regression loss. Full
 1255-image target preflight passes; validation retains983/983 words. Initial
  unsquared-distance gradient and field-name collision failures were corrected
  and rerun. Historical sources are hash-identical. See lightweight receipt
  `provenance/word_center_vote_loss_v2_qualification.json`. Full training/new
  physical GPU VRAM/OOM execution and downstream benefit remain unresolved.


## WCO-005 — Word-level failure attribution and conservative crop selection

- **Scope/evidence:** user requested a detailed explanation of degraded correct
  word counts and proposals for improvement. Inspected the existing vector
  targets/loss, global voting/flood/cropping, exact PARSeq matching, and local
  paper sections 3.2–3.3/Algorithm 1. No training, model, localization API,
  recognizer or strict configuration is changed. All new artifacts are isolated
  approximate diagnostics on the reused TRAIN holdout, 125 images/983 words.
- `REPOSITORY_SPECIFIED`: completed coherent v2 training reaches 100 epochs;
  selected epoch 84 checkpoint SHA256
  `6ad12c0f725f1abbe8344373a9429f3334f974c51346340ccf5236ef94cb8eeb`.
  Source/export hashes, original word/ignore alignment and common V1 training
  settings/all 100 sample orders are verified. Earlier WCO-004 smoke receipts
  describe their historical stage; they are not rewritten as training receipts.
  Current matched crop/OCR sources recognize 425 base words versus 348 with V2
  global centers/expansion, while historical V1 recognized 335.
- `ASSUMPTION`: attribute lost baseline-correct GT words after frozen inference
  using any original bbox eligible at IoU>=0.5 and exact casefold text. Of 152
  lost words, 113 lack eligible geometry and 39 have only wrong readings on
  eligible crops. Eligibility does not prove full glyph coverage; native target
  coverage measures qualified polygon-derived owner support, not character GT.
- `ASSUMPTION`: form families from the bipartite overlap graph of predicted
  baseline/refined native instances. Choices use only this topology and cached
  raw OCR likelihoods; care/ignore enters scoring after raw recipe selection.
  Split, merge, mixed and one-to-one operations account for 73/30/48/1 lost
  baseline-correct words respectively, with 54/2/12/7 new correct words.
- `ASSUMPTION`: union exactly duplicate-free bbox recipes as a candidate-pool
  diagnostic. It yields 500 correct words among 1753 care proposals; exact
  text+crop F1 decreases to 36.55%. GT-assisted best-image/family choices and
  perfect cared-word-offset replacement are explicitly nondeployable oracles.
  Family oracle yields 490 correct words, retaining all 425 base successes.
  Perfect cared-word offsets yield geometry F1 68.52%, with base probabilities
  and predicted background vectors held fixed; no oracle OCR is run.
- `ASSUMPTION`: post-hoc policy ablations retain the base for all multi-parent
  families. Accepting single-parent splits unconditionally still yields only
  412 correct words; allowing only one-to-one refinements yields 431. Requiring
  improved mean per-crop mean token log probability over all raw proposals for
  a single-parent change yields 451 words, gaining 34 and losing 8. No parameter
  search is performed; recipes are archived. These policies follow observed
  failure diagnosis and are not an independently validated final method.
- `REPOSITORY_SPECIFIED`: a separate existing Tversky768/.90 source has 443
  correct contour words, 358 with V2 vectors, and 525 with transferred learned
  word kernels. Source receipts, record hashes and per-image scoring are
  reverified with identical sample IDs/original GT. Its different text source
  prevents treating 525 versus 425 as a loss-only ablation. Existing kernel,
  generic embedding and learned-link components should not be reimplemented
  as novel proposals; the new diagnostic prioritizes crop selection.
- `HARDWARE_DEVIATION`: CPU analysis of archived RTX3090 FP32 predictions and
  qualified CPU PARSeq outputs. No model is resident, no new GPU memory or
  recognition measurement, training, installation or dependency change occurs.
- `UNRESOLVED`: independent transfer/generalization, calibrated crop selection,
  a genuinely trained vote-reliability/scale head, direct center/crop checkpoint
  selection and a lightweight single-model deployment. Report measured gains
  and lost baseline successes together; do not guarantee zero-loss selection.
- **Records:** `artifacts/diagnostics/center-guided-v2-word-loss-audit-v1/`
  contains complete per-word/family outputs, post-hoc selected raw recipes,
  original and alternative source hashes, scripts and the detailed report.


## UWR-002 — Corrective word experiments

- **Evidence:** local paper2503.15639v2 sections3.2–3.3/Algorithm1 specifies the
  ResNet50/attention family and component localization, not these extra heads.
  Completed v1 held-out TRAIN125/983 experiments found glifi+word F1.64198 versus
  control.64257; adaptive fusion248 words versus195; embedding selected epoch0.
  Post-hoc epoch50 CPU extraction recovered F1.50727 to.61103 by replacing hard
  aggregation with the original watershed. These are approximate exploratory
  results, not strict reproduction or independent generalization evidence.
- `PAPER_SPECIFIED`: backbone architecture family remains unchanged; all other
  inherited decoder/preprocessing ambiguities retain their existing status.
- `ASSUMPTION`: newly trained branches strictly initialize v1 control best epoch2
  (SHA47a941e17472d26c4fff259d0a81c061e23040059c49d97cf2a89a9430045cd8),
  with a frozen reference and matched50-epoch continuation. Private head LR1e-4,
  shared1e-5, fresh Adam/weight decay.01; epoch0 is eligible. TRAIN1130/125 is reused.
- `ASSUMPTION`: original glyph binary foreground supplies no character IDs.
  A private two3x3 head sees decoder+early features; glyph-only source is frozen.
  Staged glyph warmup10 then40 joint epochs ramps lambda to.1 over10 epochs.
  The isolated branch selects glyph pixelF1; joint/control select word bboxF1.
  Shared-gradient norms/cosine are measured, not inferred from loss magnitude.
  Conservative glyph association preserves word IDs; it cannot itself repair
  arbitrary word splits/merges. Two full independent U-Nets remain untested.
- `ASSUMPTION`: v1 embedding last epoch50 is held fixed for hard/watershed/soft
  extraction, rather than its selected epoch0. Soft8-neighbor competitive minimax
  cost adds bounded embedding distance with weight.25/scale1.5 and no rejection;
  original seed IDs, all foreground, and genuinely unseeded components survive.
  This is a greedy adaptation, not exact PAN or globally optimal labeling.
- `ASSUMPTION`: eight pixel links are supervised by same-word owners with valid
  foreground anchors, inverse-owner-area weights and separate positive/negative
  BCE; reciprocal inference links>=.5 form components with no area/crop cap.
  These settings differ from full PixelLink/VGG16/OHEM and can fragment words;
  raw failure counts and downstream OCR must be reported.
- `ASSUMPTION`: a frozen-probability threshold branch isolates extraction from
  negative transfer. The DB-inspired branch uses A(1-r²)/L with r=.4, raster4-edge
  perimeter/Euclidean shrink instead of polygon clipping; coherent core and binary
  targets plus boundary L1(weight10), threshold range.3–.7. A farthest-pixel core
  rescue is counted for native words removed by shrink. Core residual starts from
  frozen kernel logits and expands within frozen full-word support. BCE+Dice and
  this nucleus expansion differ from DBNet OHEM/Vatti/unclip; no full DBNet claim.
  Original GT files are never modified; new targets are exported separately.
- `HARDWARE_DEVIATION`: one FP32 model at a time, recognition in a later process;
  explicit CPU batch/worker/thread overrides are separately recorded. v1 source
  hardware was RTX3090, not a target RTX3070 qualification. GPU peaks/OOM must be
  tested on the host; caches reject a changed OCR execution backend.
- **Alternatives considered:** more training alone already lowered wordF1;
  hiding fragments with an area filter would confound evaluation; two full U-Nets
  introduce a larger architecture/memory change before testing private heads;
  full DBNet replacement would abandon the U-Net experiment. Retaining the
  original aggregation, source probabilities, and frozen/continued controls keeps
  interventions measurable. Each source loads strictly and is checksum-verified.
- `UNRESOLVED`: independent TEST benefit, calibration/generalization, optimal
  gradient balance, robustness to noisy kernels/links, and physical8GB CUDA fit.
  Local training-free/CPU results and smokes cannot resolve these questions.
- **Implementation:** isolated `*_unet_word_research_v2*` modules/runners,
  `configs/approximate/unet_word_research_v2.json`; full protocol and metric
  aggregation/ignore/empty/matching rules in `docs/UNET_WORD_RESEARCH_V2.md`.
  Every output is approximate; STRICT configuration and historical v1 are intact.

## TRC-001 — Frozen text-recall crops and PARSeq

- **Scope/evidence:** user requested actual PARSeq evaluation of the four trained
  text-recall variants and alternatives to FindContours. Inspected local paper
  sections3.3/3.5, frozen training/validation outputs, existing contour/PARSeq
  evaluator and completed global/local offset studies. The previous word Hmean
  was localization only. Old offset consensus degraded recognition and is not
  assumed to improve new masks.
- `PAPER_SPECIFIED`: connected components/contours supply localized padded text
  blocks; component-area filtering/order and top10 limit; PARSeq-S recognition.
- `REPOSITORY_SPECIFIED`: qualified original RGB/GT, canonical256 maps, unchanged
  125-image983-word TRAIN holdout, four fixed selected checkpoints and offset82,
  pinned local PARSeq artifact hashes, strict crop hashes, reusable OCR/metrics.
- `ASSUMPTION`: compare frozen selected thresholds with common0.50 separately;
  never reselect checkpoints using OCR. Identical512 outputs and identical RGB
  crops are computed once and reused. Retain pad0/no area/no cap diagnostic
  cropping for all methods; no strict-path parameter changes.
- `ASSUMPTION`: distance maxima per component (Gaussian sigma1, relative peak
  height0.6, minimum radius1, spacing8) seed the existing foreground-constrained
  minimax flood; tiny components retain one fallback maximum. All foreground
  pixels survive; irregular or curved words may over-split. The OpenCV distance
  transform/watershed tutorial motivates the alternative, not these text settings:
  https://docs.opencv.org/4.x/d2/dbd/tutorial_distance_transform.html
- `ASSUMPTION`: transfer existing offset-global consensus sigma3/density0.5/
  spacing8/support8/radius12 to aligned256 text maps, with growth disabled. No
  retraining or768 offset inference is implied. Inference recipes take no GT,
  validity, word-count or OCR input and are frozen before annotations are scored.
- `ASSUMPTION`: global one-to-one original exclusive-bbox IoU>=0.50 and exact
  Unicode casefold text, punctuation retained; Hmean2TP/(proposals+983). Ignore
  only wholly invalid instances after inference, preserve complete partial
  overlaps and every GT word. Symbol-normalized scores remain secondary; these
  exploratory bbox metrics are not official polygon or Table5 accuracy.
- `HARDWARE_DEVIATION`: CPU postprocessing of frozen RTX3090 FP32 segmentation
  outputs and CPU PARSeq FP32/batch8. Verify immutable cache provider/result and
  per-record hashes and pinned decoding, reject conflicting identical-crop text,
  and require existing16-crop replay. No two-model GPU or RTX3070 claim.
- **Alternatives:** replacing contours with components alone cannot divide a
  fused region. A new trained detector, rectification, padding, learned kernel
  retraining and threshold sweeps introduce additional factors and are deferred.
  GT-informed marker counts, oracle crop selection and OCR checkpoint tuning
  were rejected. Existing offset failures and older better kernel results remain
  separate historical evidence, not matched new-checkpoint comparisons.
- `UNRESOLVED`: whether these frozen alternatives improve OCR; independent TEST
  generalization, robust marker calibration, curved-word rectification and
  target8GB qualification. Record full scored results before claiming benefit.
- **Implementation:** `localize/text_recall_crops.py`,
  `scripts/evaluate_text_recall_crops.py`, isolated approximate config and
  `docs/TEXT_RECALL_CROPS.md`. Training/STRICT paths and previous results intact.

## TRC-002 — Existing trained kernel and coherent-vote transfer

- **Evidence/scope:** the active user requested real OCR and better word
  identification. Further repository inspection found Text+Kernel epoch61
  already improves historical OCR, and coherent-vote v2 completed100 epochs with
  selected epoch84. Inspected their implementations, source manifests/exports,
  `WORD_CENTER_VOTE_LOSS.md`, `WORD_GUIDED_CONTOURS.md`, and paper3.3/3.5.
  These previously trained cues are reused; no new training is introduced.
- `REPOSITORY_SPECIFIED`: original trained kernel, its0.50 threshold and
  watershed weights, selected coherent-vote model, normalized256 vector schema,
  frozen125/983 split, existing strict source/checkpoint hash checks and GT-blind
  foreground-constrained `word_instances`/global-center flood.
- `ASSUMPTION`: transfer the kernel map to each selected text-recall mask while
  preserving all foreground and unseeded fallback. The auxiliary predictions
  come from another trained decoder; they are not outputs of a new joint model.
- `ASSUMPTION`: evaluate coherent votes with unchanged TRC-001 sigma3/density0.5/
  spacing8/support8/radius12, no growth; vary vector source only. Preserve both
  frozen selected and common0.50 thresholds, existing pad0/no-filter/no-cap
  crops and all care words. Never tune checkpoints/thresholds using OCR.
- `HARDWARE_DEVIATION`: CPU fusion of immutable RTX3090 FP32 source maps and
  CPU-only pinned PARSeq FP32/batch8. Exact-RGB-hash OCR reuse requires source
  record verification and16-crop replay. Segmentation is not rerun and no
  simultaneous/backbone GPU footprint or lightweight deployment is claimed.
- **Alternatives:** distance-only markers already show excessive splitting in
  TRC-001 geometry; trained kernels provide a different cue already available.
  Old offset failure does not establish that a newer trained cue fails. New
  networks, retraining and an OCR-informed checkpoint search were deferred.
- `UNRESOLVED`: actual OCR benefit, robust transfer/generalization, official
  TEST behavior, and a jointly trained lightweight head implementation. Even a
  useful cached two-model transfer is not proof of lightweight joint execution.
- **Implementation:** `scripts/evaluate_text_recall_learned_crops.py`, isolated
  `configs/approximate/text_recall_learned_crops_v1.json` and shared existing
  validation/localization/OCR helpers. Scientific modes and old results intact.

### TRC-001/002 qualification — completed frozen-map OCR

- **Observed approximate evidence:** all32+16 pipelines finish on125 images/983
  care words, using genuine pinned CPU PARSeq FP32/batch8. At selected
  Tversky768 epoch10/threshold0.90, FindContours recognizes443 words/Hmean45.25%;
  transferred kernels recognize525/Hmean49.39%. Kernel gains104 and loses22 GT
  identities. Native merge31.74%->22.38%, split5.70%->13.33%; individual failures
  remain visible. All four selected text variants benefit from kernel transfer.
- **Negative evidence:** distance watershed345/Hmean28.87%, v1 offsets355/33.21%
  and coherent v2 offsets358/33.40% at that same Tversky768 point. At common0.50,
  broad text alone yields320/Hmean34.43%; kernel transfer523/47.83%. Extra mask
  coverage alone and ideal-cue smoke tests cannot establish better actual crops.
- **Qualification:**96 tests pass after correcting a test-file name collision;
  actual full OCR,16-replay cache checks for both studies and independent
  per-sample scoring pass. Audit6,000 image/pipeline rows,54,335 RGB crop hashes,
 13,000 export hashes, unchanged GT/annotation hashes and500 exact native
  localization replays. Old results/checkpoints and strict configuration intact.
- `UNRESOLVED`: independent generalization, transfer into a single shared
  decoder/head model, target RTX3070 footprint and improved noisy-center
  calibration. Two cached source networks are not a lightweight deployment
  result; observed holdout benefit does not resolve official paper accuracy.
- **Records:** `docs/TEXT_RECALL_CROPS.md`, full JSON reports and the two
  `provenance/text_recall*_crops_v1_results.json` receipts; representative offline
  OCR review includes both improved and worsened cases, with every48 aggregate
  pipeline results preserved.

## TRF-001 — Frozen Text+Kernel and native384 crop refinements

- **Authorization/evidence:** user requested execution of the proposed crop,
  native-resolution, fallback and GT-crop diagnostics. Inspect local paper
  sections3.2-3.3/Algorithm1 and the existing text-recall, research-v1 control,
  word-guided-contours and PARSeq evaluation implementations before changes.
- `PAPER_SPECIFIED`: ResNet50/attention backbone and component localization.
  Strict Algorithm1, models, weights and historical evaluation are preserved.
- `REPOSITORY_SPECIFIED`: selected research-v1 control epoch2, original selected
  two-head kernels epoch61 and existing conservative kernel fragment unions.
- `ASSUMPTION`: use native384 Tversky768 probabilities with bilinearly resized
  predicted256 kernels; regenerate384 target/ignore/owners from original data.
  Scale spatial radii; compare primary geometry/OCR in original RGB coordinates.
  Native384 topology is not the same raster metric as canonical256 topology.
- `ASSUMPTION`: freeze five recipes before scoring: unchanged baseline, existing
  refinement, bounded text.50 expansion preserving IDs/radius3 at256, near-kernel
  fallback maximum.35, and seeded-only recall-sacrificing diagnostic. No GT,
  target count, OCR string, area cap or GT-based crop selector enters inference.
- `ASSUMPTION`: separate fixed-identity GT bbox OCR on all983 words identifies
  crop versus recognizer limitations; oracle performance is never claimed as
  a detector result. Original bbox IoU.50/casefold, care handling and denominators
  follow TRC-001. All interventions use separate APPROXIMATE configuration.
- `HARDWARE_DEVIATION`: CPU FP32/PARSeq batch8, no new segmentation inference.
  Reference crop hashes all execute afresh; other exact RGB hashes may reuse
  complete pinned caches with16-crop replay. GPU footprint remains unqualified.
- **Initial failure:** frozen256-only instance PNG writer rejected384; failure
  metadata is retained and no OCR/aggregate was emitted. New isolated export
  adapter preserves uint16 IDs and rejects changed cache contents; corrected
  execution uses a separate result directory.
- `UNRESOLVED`: effectiveness, independent generalization, a single-model
  realization of the transferred768 text/kernel maps, and pending research-v2
  branches. Reused125-image TRAIN holdout is exploratory, not official TEST or
  strict Table5 reproduction. No additional training is attributed to this run.
- **Records:** `docs/TEXT_RECALL_REFINEMENTS.md`, frozen configuration
  `configs/approximate/text_recall_refinements_v1.json` and isolated result JSONs.
- **Completed qualification:**104 focused/related tests pass. All15 inference
  arms plus separate GT diagnostic complete125/983 on CPU. Independently audit
 2,000 rows,18,049 RGB crops,3,875 export hashes and7,528 OCR record hashes.
  Exact replay of125 reference masks/native metrics and125 canonical hybrid
  masks/crop receipts; fresh CPU agrees with historical CUDA on all1,037 care
  crop texts.16/16 qualified historical-cache replays pass; no OOM.
- **Measured outcome:** joint512 fallback filtering retains the same520 correct
  GT words and removes31 incorrect proposals, Hmean51.49→52.29%. Hybrid256
  seeded-only reaches52.91% but loses11 correct words (525→514). Native384
  reads528/Hmean48.09%, bounded384 growth530/47.81%; extra geometric coverage
  is not better overall OCR. GT bbox diagnostic838/983, not a detector result
  or absolute ceiling;325 incorrect joint512 words are readable from GT crops.
- **Interpretation:** no broad mask-recall improvement is established. The
  single-model observed gain is precision without held-out recall loss; require
  independent confirmation. Results, frozen independent auditor and12-example
  offline review are bound by `provenance/text_recall_refinements_v1_results.json`.

## UCL-001 — learned crop word-instance links

Authorized experimental extension: the user requested word-instance connections
and training the second pass on predicted crops, and reported93% PARSeq accuracy
on GT crops. Existing implementation, local paper sections3.2–3.3/Algorithm1,
and the qualified V1/V2 protocols were inspected before changes.

- `PAPER_SPECIFIED`: ResNet50/attention-gated U-Net family and component-based
  first-pass localization. The existing approximate architecture ambiguities
  remain; learned links and a second segmentation pass are not paper components.
- `ASSUMPTION`: strictly initialize independent crop models from v1 control
  best epoch2. Reuse completed V2 global links without retraining or modifying its
  sources/results; require its exact50-epoch receipt, source/config/map hashes.
- `ASSUMPTION`: context windows derive only from frozen first-pass predictions,
  expanded50% per side with minimum4 RGB pixels. One uniform deterministic crop
  per TRAIN image/epoch, twenty epochs, paired horizontal flips only. A source
  image with no proposals has an explicitly reported full-image window.
- `ASSUMPTION`: less than95% visible valid original owner pixels makes that word's
  crop fragment invalid for all losses. Original GT is immutable; no per-character
  IDs are inferred from binary glyph masks. Qualified paired resize/contact
  handling and regenerated word nuclei are reused.
- `ASSUMPTION`: eight reciprocal pixel-neighbor predictions, existing same-owner
  supervision and owner/class-balanced BCE, coefficient ramp5 epochs to.25;
  threshold.5. Sparse graph CC is tested equivalent to V2's BFS. Joint crop
  decoder training is compared with an otherwise matched text/kernel crop arm.
- `ASSUMPTION`: accept child supports intersecting the predicted seed, explicitly
  retain seeds when no child survives, and merge repeated supports with
  intersection/min-support-area>=.6. GT/OCR never enter extraction. Projected
  empty components and fallback counts remain visible. This can still produce
  incorrect merges and cannot recover text outside all context windows.
- `ASSUMPTION`: all four arms use original RGB exclusive bbox coordinates and
  original care annotation boxes, maximum one-to-one IoU>=.5, all983 words in
  the frozen1130/125 TRAIN split, seed202503. Select earliest best word bboxF1,
  with existing merge/recall/IoU tie-breakers; include epoch0. Repeated holdout
  selection remains exploratory. V2 native256 geometry is not an identical
  scoring control. Pixel and any-pixel instance overlap diagnostics use the
  original valid raster; wholly ignored candidates are excluded after extraction.
- `ASSUMPTION`: final PARSeq uses the existing qualified approximate checkpoint,
  original RGB/pad0, AR=True/refine_iters1, exact casefold plus bbox IoU>=.5;
  punctuation stripping is a separate secondary metric. Per-word gains/losses,
  all source windows, predicted supports, crop hashes and failures are retained.
- `HARDWARE_DEVIATION`: frozen encoder, FP32/batch4, four CPU threads and
  sequential model processes. GPU peaks and OOM must be measured on the host;
  CPU engineering smoke does not establish target RTX3070 fit. No automatic
  batch, precision, model or dependency substitution.
- `UNRESOLVED`: exact provenance/protocol of the user's93% GT-crop result; it is
  not remeasured or used as an independently verified metric here. Improvement
  after full crop training, independent generalization and physical GPU memory
  fit remain unmeasured. Official TEST and strict paths are untouched.

Alternatives considered: repeat the already completed GT-crop diagnostic (rejected
after the user's clarification); duplicate V2 link training (reuse its qualified
completed artifact); infer character IDs from glyph connected components (no
authoritative character supervision); deploy untrained crop replay (previously
degraded recall); merge by bbox overlap alone (can fuse adjacent words).

Records: `configs/approximate/unet_crop_links_v1.json`,
`docs/UNET_CROP_LINKS.md`, `provenance/unet_crop_links_v1_qualification.json`.
Qualification:107 focused/related tests and two real CPU forward/backward/second
pass smokes pass. Full training, final PARSeq comparison and GPU execution are
pending. Primary instance-link source: https://arxiv.org/abs/1801.01315.

### UCL-001 completion analysis — 2026-10-11

The qualification above describes the implementation-time state. The subsequent
host suite `/extra1/bmartelli/context-str-runs/unet-crop-links-v1` completed both
20-epoch crop trainings and final PARSeq, with its comparison written at
2026-10-11 04:16:28 Europe/Rome. Read-only verification confirms free suite and
runner locks, SUCCESS receipts, contiguous epoch logs, checkpoint/config/source
hashes, all125 RGB/annotation populations and983 care words. Source control is
V1 epoch2, not the different425-correct baseline in the older center study.

Measured exact-casefold PARSeq plus original bbox IoU>=.5: control521, crop_word524
(75 gained/72 lost), global_links490, crop_links476 (63 gained/108 lost). The
crop-links graph yields18,851 care proposals and6.43% word bboxF1. Across its1,115
saved native crop maps,65,222 of75,207 components are single pixels; components
of up to4 pixels occupy0.94% of foreground. Any-owner-pixel split diagnostics
must not be interpreted as complete destruction of the whole annotated word.

- `ASSUMPTION`: a post-hoc fixed-checkpoint decoder ablation applies the existing
  configured text/kernel watershed to crop_links epoch4 saved probabilities.
  No new inference, threshold tuning or training; original first-pass anchors,
  remapping, duplicate-overlap rules and all125/983 scoring inputs stay fixed.
  Seventeen known watershed maps replay exactly before the full diagnostic.
  Geometry changes to1,387 care proposals,676 matches and57.05% word bboxF1.
  This ablation has **no OCR result** and does not establish a correct-word gain.
- `ASSUMPTION`: lost exact GT identities are classified by availability of a
  bbox with the existing IoU>=.5 criterion, then exact OCR, then one-to-one
  competition. Crop-links loses54 words without an eligible bbox and54 with an
  eligible bbox but wrong OCR. GT enters this diagnosis and illustrative crop
  selection only after predictions are frozen; it never routes inference.
- `HARDWARE_DEVIATION`: completed host training actually used RTX3090,
  FP32/batch4,512 RGB/256 native, sequential U-Net/head and PARSeq processes.
  Crop-word allocated/reserved peaks2.21/2.58GiB; crop-links2.52/2.94GiB; no OOM.
  PARSeq alone FP32/batch8,32x128 input:141.3/168MiB; no OOM. CPU-only saved-map
  analysis loads no neural models. Physical RTX3070 qualification stays open.
- `UNRESOLVED`: independent generalization, the user's93% GT-crop OCR protocol,
  OCR after the fixed decoder ablation, and the best OCR epoch among checkpoints
  selected only by geometricF1. One seed/reused TRAIN holdout does not support
  official TEST or paper-table claims. OCR recorded deterministic algorithms
  disabled and cuDNN TF32 enabled; bit-for-bit replay is not established.

Full analysis, resolved approximate diagnostic configuration, per-word losses,
connectivity counts, scripts and crop examples:
`artifacts/diagnostics/unet-crop-links-v1-completion-analysis-v1/REPORT.md`.
The audit re-scores all rows and verifies40,586 unique care OCR cache records;
28 relevant crop/link and runner tests pass. Existing checkpoints, training
outputs, pipeline sources and STRICT configuration remain unchanged.

## WCL-001 — Word centerline and radius extension

Date: 2026-10-11. Active scope: the user requested analysis of the agents' work,
online research on segmentation/crop improvements, an experiment learning a
longitudinal line per word, and runnable test commands. This entry authorizes
only the isolated APPROXIMATE extension, not a change to STRICT defaults.
The authoritative local paper sections3.2/3.3 were inspected along with the
existing Text+Kernel model, paired word-owner targets and crop evaluators.

- `PAPER_SPECIFIED`: the paper uses a ResNet50 attention-aided segmentation
  architecture and connected-component localization with padding, component
  area filtering/order and at most10 crops. It does **not** specify a learned
  centerline, radius, private auxiliary head or the diagnostic crop protocol.
- `ASSUMPTION`: reuse the hash-qualified V1 Text+Kernel control epoch2, with all
  original tensors and BatchNorm buffers frozen. A private128→32→32→2 head
  predicts line logits and positive local radius. This tests available frozen
  features, not the potential of a fully retrained centerline U-Net. The exact
  source architecture remains the existing practical approximation.
- `ASSUMPTION`: create targets after the exact paired augmentation/resize from
  **filled word-owner regions**, not disconnected glyph masks. Zhang–Suen
  thinning preserves all components; if thinning erases a tiny component, retain
  its row-major farthest-interior pixel and audit singleton skeleton components.
  Local radii use zero-padded Euclidean distance to background. The line ribbon
  has radius/5 half-width and is clipped to the same valid owner, with nearest
  skeleton radius assigned to its pixels. Original GT/owner artifacts are not
  mutated. Expected absent owners still remain in evaluation denominators.
- `ASSUMPTION`: this is inspired by TextSnake's geometric representation, but
  does not reproduce its polygon side-pair centerline, endpoint shortening,
  orientation heads or striding algorithm. Raster branches/fragments are
  retained and audited, not silently removed through a longest-path shortcut.
- `ASSUMPTION`: valid-pixel BCE+Dice on the line plus SmoothL1 on log radius;
  radius weights give each surviving owner equal contribution within an image,
  followed by an equal image mean. Adam1e-4, decay0.01,20epochs, fixed threshold0.5,
  initial line logit−2/radius2. Geometry-F1 selection for `line_radius`, with
  declared tie-breaks including epoch0; no OCR-driven checkpoint selection.
- `ASSUMPTION`: compare the exact source kernel watershed with line markers
  expanded within frozen foreground (explicit unseeded fallback), and separate
  line/radius disk reconstruction outside that foreground. Disk ownership uses
  minimum normalized distance and row-major ties. Canvas comes only from RGB
  geometry; no GT enters either decoder. Radius floor0.5/cap128 are declared
  diagnostic bounds with clipping counts. All methods use pad0/no area filter/
  no crop cap, and therefore are not strict Algorithm1 reproduction.
- `ASSUMPTION`: real qualified PARSeq, original RGB boxes, AR/refine1, exact
  Unicode casefold plus original bbox IoU≥0.5, maximum-cardinality one-to-one
  matching. Punctuation stripping remains secondary. Every held-out care word
  stays in recall; only wholly invalid predictions are excluded after inference.
  Gains/losses retain sample and GT identities as well as full crop hashes.
  The public checkpoint/runtime are the existing ART-PARSEQ-V100-PARSEQ-CANDIDATE
  and ART-PARSEQ-PYTORCH2-RUNTIME-CANDIDATE selections; they do not resolve the
  authors' exact PARSeq-S artifact (R-001 remains blocking for STRICT mode).
- `HARDWARE_DEVIATION`: one FP32 model with TF32 disabled, frozen source features,
  then PARSeq in another process. Batch8, indexed augmentation and deterministic
  settings are recorded; the CPU engineering smoke explicitly overrides batch2/
  workers0. Source/config/environment hashes, CUDA backend and CUBLAS workspace
  configuration are recorded. Target GPU fit and peak allocated/reserved memory
  must be qualified on the host; OOM is propagated without implicit adaptation.
- `UNRESOLVED`: learned benefit after20epochs, suitability of frozen features,
  actual RTX3070/8GB memory fit, seed stability and independent generalization.
  The repeatedly reused TRAIN125/983 holdout supports exploratory comparisons
  only, not official TEST or Table3/5/6/7 reproduction claims.

Alternatives considered: a single-pixel line alone (too brittle and does not
determine crop height); an orientation/striding TextSnake reimplementation
(larger scope and additional assumptions); full backbone retraining (deferred
until the isolated representation test); repeat already completed glyph/link/
center-offset/DB/recrop experiments (rejected as duplication); add clDice in
the first run (deferred to a separate loss ablation); treat existing rotated/
Bezier rectification as new (already implemented). TextField's nearest-boundary
direction field remains a distinct unimplemented future alternative.

Qualification: focused engineering tests and a real CPU forward/backward smoke
pass. On all125 held-out images the target oracle retains983 owners with no
mixed line components and reconstructs968/983 native bbox matches atIoU≥0.5
(F1=98.47%). This uses GT lines/radii, is a **smoke-test result**, and is not
learned performance or OCR accuracy. The first preflight failed because raw
historical reference owners were compared before clipping to target/validity;
the corrected comparison and a regression test now retain exactly the prior
evaluation semantics. Training, final PARSeq and physical GPU qualification
remain pending. See `docs/WORD_CENTERLINE.md`,
`configs/approximate/word_centerline_v1.json` and
`artifacts/word-centerline-v1-qualified/target_qualification.json`.

Primary sources: TextSnake ECCV2018 sections3.1–3.5,
https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Shangbang_Long_TextSnake_A_Flexible_ECCV_2018_paper.pdf;
TextField, https://arxiv.org/abs/1812.01393;
clDice CVPR2021, https://arxiv.org/abs/2003.07311.


## USS-001 — Nucleus-guided links and selective second-pass replacement

- **Evidence inspected:** local paper2503.15639v2 sections3.2–3.3/Algorithm1;
  completed UCL-v1 selected exports, word/kernel/link extraction, original RGB
  coordinate remapping and PARSeq protocol. The user requests both procedures
  after crop_word524/983 versus control521/983 and crop_links476/983.
- `PAPER_SPECIFIED`: attention/ResNet50 foreground segmentation supplies localized
  RGB crops. Paper Algorithm1 uses component area, padding and top10; the new
  isolated diagnostic retains UCL-v1's explicit no-filter/pad0/no-cap extension.
- `ASSUMPTION`: freeze UCL-v1 word epoch16 and link epoch4 checkpoints and all
  already exported logits/windows. No retraining, new checkpoint selection or
  new U-Net inference. Pin checkpoint/validation hashes and verify source code,
  prepared manifest receipt, original RGB and every consumed evidence checksum.
  Three original reference arms replay their per-image geometry exactly.
- `ASSUMPTION`: 8-connected predicted kernels are immutable nuclei. Minimax
  priority flood uses equal soft reciprocal-link and existing word/kernel
  landscape barriers; float64 costs; ties by path length/ID/coordinate. Low links
  never introduce markers. Every unseeded foreground component survives once.
  Foreground and nucleus identities are asserted unchanged. This cannot separate
  a fused nucleus or connect disconnected foreground islands.
- `ASSUMPTION`: select second passes using only first-pass mean word probability
  below.90, kernel count !=1, kernel/support fraction below.05, or bbox fill
  below.35. Thresholds declared before evaluation; no GT, transcription or OCR
  confidence supplies a gate. These proxies may miss high-confidence errors.
- `ASSUMPTION`: require original-anchor coverage>=.85, support growth<=1.75,
  at most3 children, and other-seed foreground inside any new RGB box<=.10.
  Reject to exact original support/box. Selective arms have no cross-seed merge,
  which could otherwise mutate retained seeds. Conservative guards may reject
  genuine recoveries. Save per-seed decisions, reasons and metrics.
- `ASSUMPTION`: six-arm original RGB bbox matching at IoU.5, exact casefold
  fixed PARSeq and punctuation-stripped secondary score; .25/.75 geometry
  secondary; care/ignore protocol and any-pixel merge/split from UCL-v1.
  All983 GT words/125 reused TRAIN images remain in denominators. Matched
  ablations separate soft decoding from gating using identical source logits;
  word and link models have different selected epochs.
- `HARDWARE_DEVIATION`: CPU frozen-map replay followed by PARSeq alone in a
  separate FP32 process, batch8/four threads. Record software/backend, GPU
  allocated/reserved peaks, failures/OOM, and source OCR-cache execution context.
  External cache is read-only and must match device/batch/precision/backend.
  RTX3070 qualification requires host GPU execution; no automatic fallback.
- `UNRESOLVED`: repeated TRAIN holdout, one seed, gate generalization, crop
  readability versus geometry, and reported93% GT-crop OCR provenance. Official
  TEST and strict configuration remain untouched; no automatic promotion.
- **Alternatives considered:** hard threshold links alone caused extensive
  fragments; dropping tiny components can discard small real words; rerunning
  training would confound extraction; unconditional replacement loses previously
  correct crops. A learned gate needs independently supervised training and
  evaluation before it can replace the explicitly assumed heuristic.
- **Implementation:** `localize/unet_seeded_selective.py`, dedicated replay/OCR/
  summary/host runners, isolated approximate config and focused tests. Procedure,
  parameters, limitations and command: `docs/UNET_SEEDED_SELECTIVE.md`.


## UDM-001 — Text domain adaptation, scale fusion, progressive kernels and rectification

User-authorized APPROXIMATE experimental extensions; details, schedules, limits
and commands in `docs/UNET_DETECTION_METHODS.md` and the isolated
`configs/approximate/unet_detection_methods_v1.json`. Local paper sections3.2–3.3
and Algorithm1 plus existing implementations/results were inspected first.

- `PAPER_SPECIFIED`: ResNet50 and attention-gated U-Net backbone family. The
  existing practical decoder/strict ambiguities are not resolved by this work.
- `REPOSITORY_SPECIFIED`: qualified v1 control best epoch2; SHA-pinned word
  targets, TRAIN1130/125,seed202503,RGB512/native256,paired transforms and losses;
  existing component/watershed, checkpoint/RNG, matching, PARSeq and Bézier helpers.
  Completed V2/crop-link experiments are retained rather than retrained.
- `ASSUMPTION`: full encoder-feature convolution adaptation with frozen BN and
  a matched continuation control; compact zero-residual spatial decoder fusion
  inspired by DBNet++ with exact nearest scale expansion, not original ASF;
  three connected nested distance-fraction nuclei and confidence growth inspired
  by PSENet, not original polygon/BFS protocol; masked nesting loss. Added
  parameters16,835 fusion/130 kernels. Source initialization copies all original
  names/shapes/dtypes and loads the complete resulting dictionary strictly.
- `ASSUMPTION`: optional50000-image/five-epoch SynthText domain pretraining
  warms the qualified v1 model before matched50-epoch Total-Text adaptation. It
  is not ImageNet-only initialization/full-corpus training. imnames/wordBB/txt,
  declared0-based coordinates; SHA-bound deterministic subset; exact target RGB
  and duplicate synthetic bytes excluded/logged. Last synthetic checkpoint is
  used with no target-holdout selection during that stage. Extra data/updates
  differ from controls. Missing/malformed external data fail explicitly.
- `ASSUMPTION`: reuse geometric PCA/Bézier envelopes on predicted Total-Text
  instance support reconstructed within its mapped original box; pad0; explicitly
  logged exact axis fallback for invalid fits. This is not learned ABCNet. Each
  view retains the same original bbox for GT matching; GT never selects a crop.
  Exact casefold with punctuation retained,maximum bboxIoU>=.5 matching/all983
  words,care output precision/Hmean,native merge/split and pixel metrics are
  local diagnostics, not official polygon or strict Table5 metrics.
- `HARDWARE_DEVIATION`: FP32/batch2,single GPU,sequential model processes,explicit
  CPU smoke/device/batch overrides and worker settings. Actual allocated/reserved
  peaks/model set/shapes are logged; OOM propagates without automatic substitution.
- `UNRESOLVED`: end-to-end80% success, independent generalization, actual
  SynthText availability/effect, perceptual near-duplicate overlap, physical8GB
  target fit, and association of the qualified PARSeq artifact with paper PARSeq-S.

Alternatives considered: repeat completed private-link/DB/crop experiments,
threshold/padding sweeps or generic CLIP/BERT selection (already studied); replace
the backbone with a larger detector (would change scope/parameter budget); import
a Total-Text-pretrained detector trained on allTRAIN1255 (would contaminate our
held-out125). They are not silently introduced. No STRICT defaults, original
dataset/weights/results or unrelated user changes are overwritten.

## WDA-001

### User-requested U-Net word-detection accuracy with TN=0 (2026-10-11)

- `ASSUMPTION`: the user explicitly requested `(TP+TN)/(TP+TN+FP+FN)`
  for word detection and authorized setting TN=0. Saved detection protocols
  do not define a finite population of negative regions, so zero is an assumed
  value rather than a measured true-negative count. The requested metric then
  equals `TP/(TP+FP+FN)` and `F1/(2-F1)` for global micro counts. It is not
  pixel accuracy, OCR accuracy, or an official paper recognition metric.
- `ASSUMPTION`: compare preserved U-Net word bbox results at IoU>=0.50,
  separating single-checkpoint models from two-checkpoint hybrids and native
  raster diagnostics from original RGB coordinates. Exclude GT oracle and
  smoke-only results from the observed detection ranking. The 51-image
  IoU0.25 crop proxy is a different protocol and is not pooled into this ranking.
- `REPOSITORY_SPECIFIED`: preserve existing maximum-cardinality one-to-one
  geometry matching, care/ignore handling, predictions, and denominators.
  The best saved single-checkpoint result found on the 125-image/983-word
  Total-Text TRAIN holdout is reference512 Text+Kernel control epoch2 with
  seeded-only or tied soft-fallback extraction. Native counts are TP649,
  FP357, FN334: F1=65.258924%, requested TN0 accuracy=48.432836%.
  Original RGB counts are TP647, FP359, FN336: F1=65.057818%,
  requested TN0 accuracy=48.211624%. Hybrid384 combines two U-Net
  checkpoints and is retained separately in the report.
- `UNRESOLVED`: measured TN under a defined negative-region protocol,
  independent generalization, and official TEST/STRICT performance remain
  unavailable from this recalculation. No training or model inference ran.

Evidence: local paper section4.1 specifies foreground pixel F1, not this
word-detection accuracy. Existing `segmentation/word_multitask_evaluation.py`,
`segmentation/totaltext_word_kernel.py`, and
`recognition/scene_text_v1_evaluation.py` define the saved bbox/count protocol.
Source results remain in
`artifacts/text-recall-refinements-v1-qualified/geometry_results.json`.
The new `artifacts/diagnostics/unet-word-detection-accuracy-tn0-v1/` contains
the resolved source protocol, input hash, aggregate results, and per-image counts.
All six F1/count/accuracy checks and 375 original-coordinate image/arm matching
replays passed. Alternatives considered: define negative candidates to measure
TN, report recall alone, or use pixel accuracy; none was silently substituted
for the explicitly requested TN0 calculation.

## UWD-TEST-001

### Frozen Text+Kernel and Hybrid384 transfer to official Total-Text TEST

The user requested a Bash command to evaluate the two discussed U-Net recipes
on TEST. The existing refinements evaluator is restricted to the TRAIN holdout;
it cannot become a TEST evaluator by changing its output directory.
`scripts/test_unet_word_detection.sh` and
`scripts/evaluate_unet_word_detection_test.py` provide an isolated detection-only
entry point using `configs/approximate/unet_word_detection_test_v1.json`.

- `ASSUMPTION`: transfer frozen `seeded_only` extraction to TEST without
  threshold/checkpoint tuning. Text+Kernel uses control epoch2, input512/native256,
  text threshold.65. Hybrid384 uses Tversky768 epoch10, input768/native384,
  text threshold.90, plus Text+Kernel epoch61 kernels interpolated256 to384.
  Both retain kernel threshold.50, watershed, pad0/no area/no proposal-count cap.
- `REPOSITORY_SPECIFIED`: reuse the preserved checkpoint SHA256s, strict model
  loading, preprocessing configuration, bilinear kernel alignment and inference
  localization functions. Historical source hashes must match before execution.
  TEST manifest SHA256 is
  `da2f87e7cae765f426af39c9641535138d6657d2e9332e6fdf9e80f2ba6b69bd`.
- `ASSUMPTION`: original RGB exclusive bounding-box maximum one-to-one matching
  at IoU>=.50 measures word detection without OCR constraints. This local bbox
  diagnostic is not the official Total-Text polygon DetEval protocol or Table5.
  Full execution requires all300 images and2204 care words; a failed/missing
  sample prevents aggregate results. Empty precision/recall/F1 are zero;
  accuracy is null when TP+FP+FN=0. TN=0 remains the user-authorized WDA-001
  assumption, not a measured background population.
- `REPOSITORY_SPECIFIED`: persist both prediction arms before loading annotation
  or ignore semantics for scoring. Exclude only instances whose entire native
  support is invalid; keep partially ignored boxes complete. TEST validity uses
  the preserved binary mask, letterbox fill0, and nearest resizing independently
  at256/384. Intermediate probabilities, instance IDs, boxes, ground truth,
  matching indices, ignore IDs, hashes and resolved protocol remain auditable.
- `HARDWARE_DEVIATION`: explicit CPU or single CUDA GPU, FP32/batch1,
  deterministic seed202503, TF32 disabled, one source network resident at a time.
  CPU execution, batch differences and disabling historical cuDNN TF32 may change
  borderline predictions. GPU peak allocated/reserved memory and OOM failure
  records include model/input/batch/precision information; errors propagate.
- `UNRESOLVED`: full TEST results, CPU/GPU output equivalence, GPU execution/OOM
  qualification and RTX3070 memory fit have not been established by this work.
  A successful1-image CPU smoke is a partial-dataset result, never a full TEST
  evaluation. Earlier unrelated TEST explorations in this repository mean this
  run must not be described as a previously untouched benchmark evaluation.

Paper sections3.2–3.3 and4.1 establish segmentation, component localization and
pixel metrics; the existing kernel/watershed experiment is an approximate
extension. No scientific model, original result, checkpoint, dataset or STRICT
configuration was modified. All14 new/relevant unit tests and the300-input asset
preflight passed. CPU smoke v1 failed explicitly on a wrong nested constructor
configuration key. Smoke v2 failed because the reused immutable JSON writer
correctly rejected overwriting the memory receipt. Both failures remain under
`/tmp/unet-word-detection-test-command-smoke-v1/` and `-v2/`; corrected smoke v3
strict-loaded and executed all three networks and both detection arms successfully.
No full TEST or GPU performance claim follows from this qualification.

## UWC-001

### Dense neural compute for the evaluated Text+Kernel and Hybrid384 recipes

- `ASSUMPTION`: report executed convolution/linear MACs per RGB image at batch1,
  using two FLOPs per MAC. The count includes the frozen ResNet encoder; it
  excludes unexecuted registered layers, bias additions, BatchNorm, activations,
  pointwise attention, sigmoid, resizing, concatenation and CPU localization.
  Watershed/filter/box work is input dependent and has no measured total here.
  This dense neural arithmetic protocol does not resolve paper Table4 E-004.
- `REPOSITORY_SPECIFIED`: reuse `scripts/neural_macs.py` forward hooks and the
  actual pinned TEST checkpoints with strict loading. All three actual CPU
  FP32 forwards were independently checked against torch.profiler dense
  conv2d/matrix FLOP events; exact equality and hook cleanup passed.
  Text+Kernel512 is66.060025856 GMAC /132.120051712 GFLOP per image.
  Hybrid384 sums Tversky768148.625620992 GMAC and kernel51266.060025856 GMAC:
  214.685646848 GMAC /429.371293696 GFLOP, or3.249857 times Text+Kernel.
- `REPOSITORY_SPECIFIED`: the preserved full TEST CUDA stage receipts show
  18.464209 seconds/300 images for Text+Kernel and40.150512 seconds/300 for
  the two Hybrid384 source stages. These averages are61.547363 and133.835039
  ms/image. Timers include device transfer, RGB disk reads, preprocessing,
  forward, CPU copy, compressed map writing and unloading/garbage collection;
  they exclude initial checkpoint loading and later CPU localization/evaluation.
  They are not isolated neural latency or complete detector FPS.
- `HARDWARE_DEVIATION`: fresh arithmetic profiling uses CPU FP32/batch1;
  timing/memory figures are inherited from the user's CUDA FP32/batch1 TEST run.
  Sequential allocator peaks are474.390137/620.386230 MiB allocated and
  538/698 MiB reserved for Text+Kernel/Hybrid384. Saved stage receipts report
  no OOM. These are measured allocator values, not full device-process memory.
- `UNRESOLVED`: total detector FLOPs including localization, pure GPU latency,
  complete pipeline throughput, repeatability/error bars, actual TEST GPU model,
  cross-device output equality, and the paper's FLOP convention remain unresolved.
  The historical training report's RTX3090 identity is not evidence of TEST GPU
  identity; the TEST runtime only records CUDA availability/software version.

Results, checkpoint/input/source hashes, per-layer executed MACs, independent
profiler totals and timing limits are stored in the evaluated TEST directory's
`compute_cost.json`. The first attempt failed before arithmetic profiling because
the checksum helper received a string instead of Path; explicit conversion fixed
it, and the successful profile preserved all original TEST outputs and models.

## WORD-GT-001 — User-requested separated full-word supervision

- `PAPER_SPECIFIED`: authoritative local paper section 4.1 uses ICDAR13,
  COCO-TS and Total-Text for text segmentation; Algorithm 1 consumes text pixels.
  The paper does not specify this new filled-word target or artificial gaps.
- `REPOSITORY_SPECIFIED`: downloaded ICDAR13 has 229 TRAIN character TXT/BMP
  pairs and 233 TEST Task1 word-box TXT files; ICDAR2015 has 1000/500 word
  quadrilateral annotations; Total-Text has 1255/300 word polygons, character
  masks and full text-region masks. Physical inspection of all1555 distributed
  Total-Text region PNGs confirms binary0/255 encoding and matching image sizes.
  Reuse existing parsers and nearest-exclusive-pixel polygon ownership from
  `segmentation/totaltext_word_regions.py` without changing existing paths.
- `ASSUMPTION`: fulfill the user's full-word/separated-region request using
  source-resolution polygon fills; overlapping source regions receive unique
  owners and 8-neighbour contact pixels are removed on both sides. Preserve
  the largest 8-connected region per word if contact removal fragments it,
  tie-breaking by first component in raster order. Record discarded pixels;
  fail explicitly if any care word disappears. Ignore geometry receives zero
  foreground and zero validity. All-ignore images remain in the manifest.
  These changes affect foreground areas/boundaries and are approximate targets,
  unsuitable for claiming the paper's pixel-segmentation Table3 protocol.
- `ASSUMPTION`: ICDAR13 TRAIN blank-separated character groups define care
  words; their character-box corner convex hulls define filled regions. Parse
  each ignored character row separately. In the physical mixed group257,
  retain its care character instead of dropping a group containing ignored
  rows. This grouping is supported by physical structure but has not been
  established as exactly equivalent to official TRAIN Task1 word annotations.
  TEST uses acquired Task1 word rectangles, preserving source transcripts.
- `UNRESOLVED`: CTW1500 local GT remains line-level. Online research found
  VimTS `gt_ctw1500_word.zip` and ArT Latin word-polygon annotations as candidate
  sources. VimTS archive format, TRAIN coverage, image identity, annotation
  revision and completeness require physical qualification. The official CTW
  ID-vs-ArT list does not by itself establish a bijective sample mapping.
  Shell download of ArT failed with DNS resolution error; web retrieval of
  the JSON and VimTS OneDrive archive failed. No approximate equal-width
  split or model prediction is promoted to genuine GT.
- `ASSUMPTION`: original-resolution export alone is qualified. Any later
  resize/augmentation needs an owner-map topology audit and renewed separation;
  source-resolution separation does not guarantee native128 word survival.
  Source SHA256s, resolved mode/configuration, code hashes, all instance IDs,
  care/ignore counts and per-image checks are exported. No training or GPU
  model inference runs as part of this artifact generation.

Sources, commands, artifact schema and scientific limits:
`docs/SEPARATED_WORD_GT.md`. Existing strict configs and source GT are preserved.

### WORD-GT-001 update — V4 reference and wider gaps (user steering)

- `REPOSITORY_SPECIFIED`: inspected `segmentation/totaltext_curved_box_v4.py`,
  its preparation script, approximate500-epoch configuration, qualified source
  manifest and image147 comparison. V4 consists of filled curved word regions.
  Reuse its checksum-frozen1255 TRAIN owner maps, masks and validity directly;
  verify source image and annotation identity. No V4 model weights change.
- `ASSUMPTION`: remove unprotected boundary pixels near another source word
  using Euclidean distance with anisotropic sampling expressed in a128x128
  reference image. Clearance2 reference pixels and maximum per-word area
  loss0.35 are explicitly selected, untuned approximate construction inputs.
  They are not paper parameters or guaranteed minimum final word distances.
  Keep filled interiors and existing curves; never uniformly erode every word.
  Source stroke GT is protected within starting owners wherever available.
  ICDAR2015 has no downloaded stroke GT and relies on topology/area guards.
- `ASSUMPTION`: restore a word if the gap operation erases it, exceeds the
  area-loss bound, or disconnects protected text. Restoration and protected
  close pixels are reported rather than hidden. Remove secondary disconnected
  islands only when they contain no protected source text. Physical V4 audit
  found18 original-resolution fragmented words;16 have only unseeded secondary
  islands, while samples228/1111 contain389 stroke pixels in secondary pieces.
  Preserve these latter instances with explicit inherited-fragmentation flags.
- `ASSUMPTION`: for newly rasterized non-V4 sources, explicit care word regions
  take precedence over overlapping coarse ignore geometry. The first full
  export with ignore-box precedence failed on ICDAR13 sample170 after70 images:
  a broad ignore box completely covered care word `semi`. Preserve that failure
  record. Full-word supervision includes inter-character background, so broad
  character-level ignore boxes are unsuitable as unconditional exclusions.
  Record every care/ignore overlap pixel. This policy does not alter the frozen
  V4 validity or the existing pipeline's ignore policy.
- `REPOSITORY_SPECIFIED`: ICDAR13 source192 includes a zero-height ignored
  character box, which is line geometry, not an invalid care word. The second
  export failed explicitly on that row. Accept point/line ignore geometry as
  other existing parsers already do; continue to reject degenerate care boxes.
  Preserve the failed export instead of suppressing either failure.
- `ASSUMPTION`: only original-resolution survival and cross-word separation
  are enforced for export. Nearest-resized reference128 owner audits are saved
  before/after and can expose disappearing small words or remaining contacts.
  Future training transforms need separate qualification. No accuracy claim
  follows from making target words more separated.


## CLIP-IMAGE-ROUTER-V1

- `PAPER_SPECIFIED`: section3.5/Algorithm2 calls DeepSolo(I) when the context score is low.
- `REPOSITORY_SPECIFIED`: use existing Text+Kernel epoch61/seeded PARSeq outputs, native geometryF1 .621272 and original bboxF1 .613320. Saved recognition report SHA25690740316514031ac4fba44dafe4e48f150ec3f806fb98abd3a9ae1b094bca392; 125 images,1109 predictions,80 source-protocol ignored predictions,1029 care outputs,480 exact-casefold matches,983 GT.
- `ASSUMPTION`: user-requested CLIP extension routes whole images by minimum over all predicted words; no text/crop selection or ignore masks in routing. Empty outputs/readings force fallback. Compare crop prompts, rendered glyphs, text/background contrasts, native confidence and full-image controls. Embedding scores are not calibrated probabilities.
- `ASSUMPTION`: pin80 calibration/45 validation images with SHA256(seed:ID), seed20261012. Fit CDF references only on calibration features. Select each threshold and winning family only on calibration labels, requiring>=99% correct/detected and zero erroneous accepted images; maximize image coverage then words. Never retune on validation. Separately label all125 scans post-hoc.
- `ASSUMPTION`: correct/detected accuracy isTP/(TP+FP); also expose TN0 accuracyTP/(TP+FP+FN) over accepted scenes, missing words, scene completeness, correct images unnecessarily routed and accepted-word counts. Use the original casefold-exact/IoU>=.50 matching and evaluation-only source ignore policy.
- `HARDWARE_DEVIATION`: sequential CLIP FP32/batch16/four CPU threads, explicit CPU or CUDA, deterministic TF32-disabled settings and per-shard memory/error receipts; DeepSolo stays in its qualified separate legacy runtime.
- `UNRESOLVED`:99% population reliability, independently acquired TEST routing, selection bias of the repeatedly used TRAIN holdout, ignored-text safety, and DeepSolo end-to-end accuracy remain unestablished. A small zero-error sample or all-image fallback cannot demonstrate a useful safe router.
- Implementation and commands: [CLIP_IMAGE_ROUTER.md](CLIP_IMAGE_ROUTER.md).


## UDM-STORAGE-001 — Retaining the declared SynthText subset

The local paper section4.1, the UDM-001 extension, its frozen configuration and
existing SynthText selector were inspected before this storage-only change.
The paper does not specify this external pretraining; it remains the existing
APPROXIMATE extension and does not enter STRICT configuration.

- `REPOSITORY_SPECIFIED`: reuse the configured50000 eligible images, seed202503,
  coordinate origin0, exact target-RGB/duplicate exclusions, original gt.mat,
  unchanged source RGB bytes, image and annotation validation. Training/target
  losses, architecture, splits, epochs, checkpoint selection and PARSeq are
  unchanged. Acquisition calls the same prepare_synthtext selector using an
  optional callback immediately before the normal RGB checksum calculation.
  Full extraction and replay without the callback must select identical IDs,
  labels, hashes and exclusions. Retain rejected candidates visited before the
  selected subset is complete, so later ordinary preparation makes identical
  skips. Existing missing/malformed inputs still fail rather than being filtered.
- `ASSUMPTION`: optional subset extraction and explicit deletion of the helper's
  verified cached ZIP after successful preparation. This changes retained storage,
  not selected training examples. Record torrent info hash, archive SHA, original
  annotation SHA, config/target-manifest SHA, subset manifest/selection, actual
  retained bytes and CRC scope. The entire ZIP still downloads and its piece
  checks/full archive SHA are preserved. CRC checks cover retained members only
  in subset mode; do not claim validation of unused members. Deletion rejects
  a changed/unbound/symlink ZIP and never deletes GT, images or checkpoints.
- `ASSUMPTION`: reduced-mode initial space check is one archive size plus28GiB,
  comprising16GiB extraction headroom and the existing12GiB experiment reserve.
  Actual per-file space is checked during extraction. This is an operational
  budget rather than a measured corpus size or accuracy parameter. Full-mode
  extraction/reservation remains selectable. No environment packages are installed.
- `UNRESOLVED`: actual retained corpus size, live external transfer, end-to-end
  accuracy and GPU qualification. Only offline fixtures establish equivalence;
  no external pretraining result or80% claim is made. The optional callback changes
  a tracked scientific source SHA, requiring a new experiment output directory;
  historical manifests/checkpoints/qualification receipts remain untouched.

Alternatives considered: retain the complete extracted corpus; train directly
from ZIP (would require dataset/worker I/O changes); resample a smaller training
subset (would change the scientific intervention); download individual internal
images via this torrent (it publishes one ZIP, not independent image entries).
Implementation and commands: docs/SYNTHTEXT_DOWNLOAD.md.

### WORD-GT-001 update — recovered CTW word GT and explicit source correction

- `REPOSITORY_SPECIFIED`: the pinned CTW dataset author's VimTS repository
  supplies `evaluation/gt_ctw1500_word.zip` at commit
  `929dacbee21f5be22eb2091360933da4d3fba122`. The GitHub connector retrieved its
  exact342143 bytes; Git blob SHA1 is
  `2c53ce6f0e54ed5856e4252d3f73db65f585e497`, independently recomputed, and
  archive SHA256 is
  `c93f48f99d11eb3d5acc8618ec9b5001a69277ee905184ee7c1c26af33011cbf`.
  The500 annotation IDs match all downloaded CTW TEST images. Physical parsing
  finds5698 rows,4915 care/783 ignore,4..17 polygon vertices, UTF-8 text after
  the literal comma/four-hash delimiter; ignore transcripts are exactly###.
  This resolves TEST artifact acquisition, not the paper's word protocol.
- `ASSUMPTION`: use this external TEST annotation variant as approximate
  supervision. Visual inspection showed DOUGLASTON split into DOUGL/ASTON and
  E-313 into E/313 in1001. When an original care line is a single whitespace
  token and spatially contained source parts reconstruct its normalized text
  exactly, use the original polygon/transcription for that single word. Parts
  are ordered along the original annotated centerline. Multiple-token lines
  are never merged. Casefold plus Unicode letter/number filtering is used only
  for verification of merges; original source text is not changed.
- `ASSUMPTION`: curate source redundancy with explicit audit records. Exact
  duplicate rows are removed. A complete smaller polygon may be removed only
  if its normalized text is a strict substring of a containing larger word,
  and every raster pixel is inside that larger polygon. Equal-text nested
  alternate polygons retain the tighter source boundary, with source-order
  tie-breaking for raster equivalence. Keep unrelated overlaps and fail when
  resulting ownership cannot preserve all curated word IDs. These corrections
  change the GT variant/word count and cannot be substituted into historical
  CTW recognition denominators or scores.
- `REPOSITORY_SPECIFIED`: the first CTW export stopped at1027 after26 images
  on a duplicate64 region. The subsequent export stopped at1239 after238 images
  on nested A alternatives. Preserve failure receipts and partial artifacts;
  add focused tests rather than catch-and-ignore or silently omit images.
- `UNRESOLVED`: complete linguistic correctness of the VimTS word labels,
  original transcription spacing errors, human review of every boundary, and
  source word-GT geometry for all1000 CTW TRAIN images. The recovered archive
  is TEST-only. No equal-width split or detector pseudo-label becomes GT.

The final artifacts remain approximate reproduction GT preparation outputs,
not segmentation/recognition evaluation results. See `docs/SEPARATED_WORD_GT.md`.

### WORD-GT-001 qualification — completed separated-word artifacts

- `REPOSITORY_SPECIFIED`: complete exports contain 4017 masks: Total-Text
  1255/300, ICDAR2015 1000/500, ICDAR13 229/233 and CTW1500 TEST 500.
  Independent OpenCV 8-connected component verification found no missing
  curated owner IDs and no components shared by distinct words. All archive
  members match delivered files; CTW original-line/source-word pairs and
  generation-code hashes also passed supplementary verification.
- `ASSUMPTION`: preserve the two documented inherited V4 fragmented words,
  rather than erase their 389 secondary stroke pixels. All 40038178 protected
  source-stroke pixels within starting V4 owners survive. Total-Text TRAIN
  changes 4977 word regions, removing 2015089 foreground pixels. These are
  artifact topology observations and do not establish recognition improvement.
- `ASSUMPTION`: CTW curation produces 4776 care words after 95 single-token
  line merges, one exact duplicate removal, nine contained lexical-fragment
  removals and three nested equal-text removals. This curated denominator
  cannot replace the original source benchmark denominator in evaluation.
- `UNRESOLVED`: CTW TRAIN word geometry, exhaustive linguistic/manual source
  qualification and future resized/augmented target topology remain unresolved.
  Requested clearance is limited by protected strokes and restoration guards.

Artifacts and machine-readable qualification are under
`artifacts/separated-word-gt-v1/v4-wider-gaps/`. All outputs retain the
`approximate reproduction` label. Focused GT/curation/word-region tests pass
(27); existing dataset parsing/ICDAR13 conversion tests pass (32). Source-split
and corrected CTW smoke exports pass; early failures remain documented outside
the final delivery tree. No model training or paper-result claim is made.

### WORD-GT-TRAIN-001 — new-target training and PARSeq command integration

- `PAPER_SPECIFIED`: inspected paper section4.1 and Algorithm1 again before
  integrating training. The paper specifies Adam1e-4/weight-decay0.01, batch8,
  BCE, input256 and mask128, and localization through components with padding
  and top10. Filled word masks and 500 epochs are not established there.
- `REPOSITORY_SPECIFIED`: reuse the qualified V4 AG-U-Net architecture,
  frozen ResNet50, FP32/batch8, paired transforms including resized owner-contact
  separation, 500 epochs, and TRAIN foreground-IoU checkpoint selection.
  Preserve frozen V4 protocol/config validation and strict configuration.
- `ASSUMPTION`: introduce explicit approximate protocol
  `practical_totaltext_separated_words_500_v1`, artifact
  `ART-TOTALTEXT-SEPARATED-WORDS-V1`, and checksum-bound adapter manifest for
  all1255 new TRAIN targets. Only the target source changes in the V4 training
  comparison. The common data root is an explicit runtime argument; originals
  are read in place and no GT pixels/photos are rewritten.
- `REPOSITORY_SPECIFIED`: reuse the existing TEST-300 native128 contour/PARSeq
  evaluator, local pinned PARSeq artifacts and sequential model loading. Its
  bbox-area>=150 filter/sort, padding6 and top10 are approximate repository
  behavior, not the paper's component-area protocol. TEST labels are used
  only for post-hoc matching; no GT-derived crop is passed to PARSeq.
- `ASSUMPTION`: report exact Unicode-casefold text with one-to-one box IoU>=0.50:
  matches/all2204 care words, and separately matches/geometry matches. The first
  is recall-like and the second conditional OCR accuracy; neither is the
  undefined paper Table5 accuracy. Ignored GT are excluded and predictions are
  not subjected to an official ignore-overlap filter. Preserve case-sensitive
  scores, detection metrics, raw crops/masks/OCR and denominators in JSON.
- `UNRESOLVED`: new-target physical GPU training success, allocated/reserved
  memory, OOM behavior, resized/augmented survival of small words, TEST accuracy
  and any improvement over V4. Preparing commands does not resolve these.
  Existing approximate PARSeq weights do not establish PARSeq-S fidelity.

The new entry point is
`scripts/run_totaltext_separated_words_train_and_parseq_host.sh`; `--check`
performs CPU checks only. Commands, metric definitions, output locations and
resumption are documented in `docs/SEPARATED_WORD_GT.md`.

Qualification: CPU preflight passed twice with all1255 input checksums,
three transformed target examples, TRAIN/TEST ID separation and pinned PARSeq
artifact hashes. All46 focused tests passed. Initial worker8 CPU integration
attempts hung with AF_UNIX `EPERM` in the managed sandbox and were interrupted;
this is an environment failure, not successful multiprocessing qualification.
The integration test now explicitly uses a tiny CPU fixture and a test-only
single-process DataLoader, preserving production worker8 and failure handling.
No real GPU training or TEST inference was launched. Physical runtime and
accuracy remain `UNRESOLVED` until the user runs the command on the server.

### CLIP-IMAGE-ROUTER-V1 fixed-rule follow-up

- `ASSUMPTION`: after inspecting B32 validation, select native_word_advantage as an exploratory fixed family. Its original calibration threshold .6214129671585633 accepts15/15 calibration words and10/10 validation words:8/125 images,25/25 words,117 fallback images,5 initially correct images unnecessarily routed. No omitted care words in accepted images. Family selection is post-hoc; threshold selection itself does not use validation labels.
- `ASSUMPTION`: an efficient fixed-rule runner computes only the exact word-template/background contrast and native confidence; byte-check identical model/device/config/code and crop/OCR rows when explicitly reusing full-battery scores. No absent signal is replaced with a dummy. Preserve original full-battery outputs.
- `UNRESOLVED`: calibration coverage winner native_glyph_textness fails validation safety (16/22=72.73%). Pure scene/crop CLIP-only calibration families mostly cannot accept any safe image.8 safe images cannot establish99% population reliability; zero-error exact95% lower image precision bound is68.77%. The fixed rule and additional models require independent testing.

## CLIP-IMAGE-ROUTER-V1 completed fixed-rule qualification

- `REPOSITORY_SPECIFIED`: all3 model runs completed125 source images/1109 unchanged predictions. Baseline480/1029 correct/detected=46.65%; TN0 includingFN is480/(1029+983-480)=31.33%.
- `ASSUMPTION`: calibration-only thresholds for the post-hoc chosen fixed family: B32 .6214129671585633, B16 .6192174372838609, L14 .6801702981222083. Respectively accept25/25,22/22,5/5 correct words in8,7,2 images; send117,118,123 to DeepSolo. Calibration/validation splits are15/15+10/10,12/12+10/10,3/3+2/2 words. All accepted scenes are complete under the local care metric; initially correct images unnecessarily routed are5,6,11. These tiny development counts do not certify future safety.
- `HARDWARE_DEVIATION`: actual CPU/FP32/batch16/four threads per process; independent CPU processes overlap, so elapsed times are not isolated performance comparisons. B32/B16/L14 explicitly reused125/88/28 byte-qualified full-battery shards; others were real inference. All GPU peaks are null; no CUDA execution or RTX3070 memory qualification is claimed.
- `REPOSITORY_SPECIFIED`:115 focused tests pass; all3 independent audits replay125 images/1109 OCR identities, CDFs, thresholds, metrics and routing actions using manual bboxIoU/SciPy cardinality assignment. The first auditor failed min([]) on an empty image; actual routing already correctly rejected it. Corrected audits pass and source/results are unchanged.
- `UNRESOLVED`: independent official TEST validation, probability guarantees, combined post-DeepSolo accuracy and matched DeepSolo polygon/EXIF/ignore evaluation. DeepSolo runtime/checkpoint/source and117 B32 requests pass actual artifact preflight, but GPU fallback itself is not executed. The optional extended L14 all-signal battery is separate and remains in progress.

Completed comparison: `/tmp/clip-image-router-word-advantage-v1/comparison.json`, SHA2562236c469e7faa7b9864f5e23b3aa3c7860b489e764dccd2832517dae8e21973e. See [CLIP_IMAGE_ROUTER.md](CLIP_IMAGE_ROUTER.md) for per-model receipts and executable commands.

## CLIP-ROUTER-E2E-001 — split-separated CLIP/PARSeq/DeepSolo evaluation

- `PAPER_SPECIFIED`: section3.5/Algorithm2 uses `DeepSolo(I)` for fallback on
  the original full image. The local paper was inspected before this extension.
- `REPOSITORY_SPECIFIED`: reuse the qualified Text+Kernel epoch61 source,
  seeded watershed including unseeded components, pinned PARSeq and CLIP B16
  prompts, and the qualified isolated DeepSolo Total-Text candidate. Existing
  TEST maps match its checkpoint; the separate epoch2/Hybrid384/seeded-only
  aggregate is not reused as this model's predictions.
- `ASSUMPTION`: TRAIN means the already analysed held-out125/983care words;
  official TEST is300/2204care words. Report separately. B16/family selection
  is exploratory after TRAIN labels; the prior TRAIN coverage threshold .146906
  and TRAIN80 CDF references stay frozen on TEST. No TEST fitting is allowed.
- `ASSUMPTION`: exact casefold, punctuation retained, exclusive bbox IoU>=.5,
  maximum-cardinality one-to-one micro matching; detected-word accuracy and
  TN0 accuracy are separate. Evaluation-only U-Net ignore uses exact native
  support; DeepSolo ignore uses original polygon rasterization (nearest-even
  vertices), letterbox nearest256, entirely-invalid nonempty support only.
  Inverse EXIF precedes geometric evaluation. Empty clipped envelopes and tiny
  empty native footprints stay care/unmatchable rather than being discarded.
  This metric is not official polygon DetEval and does not resolve Table5.
- `REPOSITORY_SPECIFIED`: qualified pinned upstream `text_spotter.inference`
  performs its inclusive score threshold solely after network outputs. Cached
  TEST .05 raw responses can be filtered>=.4 without rewriting their actual
  inference options. Both original artifacts and conversion receipts persist.
  This establishes equivalence by code inspection, not GPU repeatability.
- `ASSUMPTION`: cached kernel TEST maps have historical run/checkpoint/input/code
  evidence but no historical individual map hashes. Seal current digests and
  retain that evidence limit; no claim of stronger provenance is made.
- `HARDWARE_DEVIATION`: sequential cached segmentation, FP32 batch8 PARSeq,
  batch16 CLIP CPU/explicit CUDA, isolated single-image FP32 DeepSolo CUDA;
  cross-device/batch numerical equivalence remains unproven. Save memory/error
  records and propagate failures. The main sandbox cannot access CUDA; missing
  TRAIN DeepSolo requests require the already-qualified GPU host.
- `UNRESOLVED`: exact paper PARSeq-S/DeepSolo artifacts, recognition metric and
  official ignore/matching conventions still block strict/Table5 claims. CLIP,
  kernel head and the half-coverage gate remain explicit experimental extensions.

See [CLIP_ROUTER_E2E.md](CLIP_ROUTER_E2E.md), isolated approximate configurations
and `scripts/run_clip_router_e2e.py`. No strict configuration is changed.

Qualification/result:97 focused tests and real two-image CPU/cached-CUDA smoke
pass. Full independent TEST300 (no fitting) accepts149/300 images, with151
DeepSolo routes:1459 TP/610 FP/745 FN,70.5172% accuracy on2069 detected care
words and51.8479% TN0 accuracy. Independent CDF/EXIF/polygon/PIL-ignore/SciPy
matching audit passes all300 samples. Same-protocol all300 DeepSolo control:
1588 TP/420 FP/616 FN,79.0837% detected-word accuracy,60.5183% TN0. TRAIN125
remains separate with62 accepted/63 fallback requests; no combined TRAIN
accuracy is claimed until those63 qualified GPU outputs are available.


## UDM-SYNTHTEXT-TARGETS-001 — Visible and thin synthetic word targets

Before this change, the existing SynthText selector/loader, Total-Text ownership
and corridor implementation, configured pretraining protocol, saved run
contracts/checkpoints, and local paper sections3.2/4.1 were inspected. SynthText
pretraining remains an external approximate extension, not a paper-specified
training stage. No STRICT defaults are modified.

- `REPOSITORY_SPECIFIED`: retain the declared50000 image cohort, seed202503,
  original integer wordBB coordinates/transcripts, exact RGB overlap/duplicate
  exclusions, pretrained v1 source, five synthetic epochs and later matched
  Total-Text training/PARSeq protocol. Preserve the existing bilateral corridor
  behavior by default and on Total-Text. The old run has last.pt/best.pt at
  epoch0/global_step0; no completed epoch can be recovered. The failure record
  measures2,427,747,840 peak allocated/2,642,411,520 reserved bytes before failure;
  this does not qualify a complete epoch or full hardware fit.
- `REPOSITORY_SPECIFIED`: actual failure reproduced at shuffle position75,
  dataset index5757, sample synthtext:348698, word10 `--`, image171/sun_6_10.jpg
  (600x600). Its GT polygon raster has39 pixels, but the contact corridor erased
  the complete owned region. Therefore simply treating this as corrupt data or
  skipping the sample would discard an observable label. A raw-raster census
  of all50000 images counted422616 annotations and6 wholly off-image boxes.
- `ASSUMPTION`: only in the new explicit preserve_visible_words_v1 policy,
  restore pre-corridor owned pixels for each word erased entirely by contact
  removal. Every restored pixel remains in that word's original GT support;
  none is fabricated or added outside the polygon. Restored pixels are removed
  from the auxiliary separator map. Very thin adjacent words can retain touching
  owner labels when binary separation is geometrically impossible. This choice
  may weaken the separation penalty locally; it does not guarantee improved
  accuracy. Alternatives considered: skip thin labels (loses real supervision),
  dilate/clamp polygons (manufactures support), globally disable separators
  (changes all synthetic targets), or remove the guard (hides label loss).
- `ASSUMPTION`: a box with its entire bounding extent outside the image is an
  unobservable synthetic annotation. Retain its original annotation/transcript
  in the manifest and record its original1-based word index in the per-image
  audit; do not clamp it into the image or remove the image from the cohort.
  All remaining visible labels require nonempty ownership. Visible empty or
  conflicting unrepresentable labels, including duplicate geometry, still fail
  with sample/word diagnostics. This is not a silent corrupt-sample exception.
- `ASSUMPTION`: validate all source-resolution targets during preparation and
  bind per-image target audits/aggregate counts to the immutable manifest.
  Compare the audit on every runtime access after RGB SHA checking. Report
  annotation/visible/outside/restored counts separately and count native-size
  resampling losses in epoch metrics. Original Total-Text evaluation words,
  matching rules, ignore handling and the983-word denominator remain unchanged.
- `UNRESOLVED`: whether this target correction or synthetic adaptation improves
  localization-and-recognition recall toward80%, and whether tiny words survive
  native resampling in a complete GPU experiment. Source-resolution raster
  qualification is separate from training/inference and recognition results.

Implementation/configuration and restart command are documented in
[UNET_DETECTION_METHODS.md](UNET_DETECTION_METHODS.md). Use a fresh output root;
never mutate the old run contract or overwrite its checkpoints/failure record.
Validation receipts distinguish numerical target tests from the restricted
runner's inability to perform PyTorch POSIX shared-memory worker transfers.

## CLIP-ROUTER-SWEEP-001 — descriptive image-acceptance threshold curves

- `PAPER_SPECIFIED`: section3.5/Algorithm2 supplies confidence-based full-image
  DeepSolo fallback. Paper Table6 uses MPNet/fuzzy alpha/beta/tau; this CLIP
  extension is separate and claims neither Table6 nor strict reproduction.
- `REPOSITORY_SPECIFIED`: freeze B16 native/word-advantage minimum scores,
  TRAIN80 CDF references, epoch61 Text+Kernel/PARSeq outputs and qualified
  DeepSolo .4 instance-score handling. Reuse immutable per-image evidence and
  replay both previous .146906 full-pipeline aggregates exactly.
- `ASSUMPTION`: report a17-point threshold grid plus all observed score
  breakpoints as a descriptive curve. TEST labels support reported metrics;
  no new threshold is chosen on TEST, and a future choice from these curves
  would no longer have an independent selected-threshold TEST evaluation.
- `ASSUMPTION`: TRAIN125/983care GT and TEST300/2204care GT stay separate.
  Distinguish care words retained from PARSeq from final care predictions after
  entire-image fallback. Preserve raw/ignored counts, TP/FP/FN, precision,
  TN0 accuracy, recall and F1. Missing fallback outputs invalidate the complete
  point, leave final metrics null, and never disappear from denominators.
- `HARDWARE_DEVIATION`: cached recombination is CPU-only with no new main-model
  inference. Optional62 additional TRAIN DeepSolo requests use the existing
  isolated FP32 single-image CUDA runtime with original failure/memory records.
  No package installation, Docker change or strict configuration change occurs.
- `UNRESOLVED`: currently only63 TRAIN DeepSolo images are available, so high
  TRAIN thresholds remain incomplete until further GPU execution. Exact paper
  checkpoints, metric/ignore ambiguities and end-to-end latency/FLOPs remain
  unresolved; invocation counts alone do not establish computational savings.

Qualification:39 focused tests pass. All425 source image/branch populations and
310 TEST/140 TRAIN curve points pass independent SciPy matching/geometry/count
audit, including63 original TRAIN DeepSolo polygon frames. TEST .146906 gives
2069 predicted care words,70.5172% precision,51.8479% TN0; .5 gives2002 words,
78.5215% precision,59.6811% TN0, with259/300 DeepSolo routes. Complete all-DeepSolo
TEST gives2008 words,79.0837% precision,60.5183% TN0. These are descriptive points,
not a TEST-tuned recommendation. TRAIN .146906 now gives876 words,75.7991%
precision,55.5649% TN0; higher points preserve explicit missing-output status.

See [CLIP_ROUTER_THRESHOLD_SWEEP.md](CLIP_ROUTER_THRESHOLD_SWEEP.md), isolated
approximate configurations, `scripts/sweep_clip_router_e2e.py`, CSV/JSON results
and offline SVG/HTML visualizations outside Git. Original artifacts are unchanged.


### CLIP-ROUTER-PROMPT-002: updated encoders and complete-transcript prompts

This user-authorized extension inspects the existing crop/image router and local
paper section3.5/Algorithm2. It is **approximate reproduction**, separate from
strict MPNet/fuzzy/BLIP2 behavior. No strict config or frozen prior scientific
implementation is changed.

- `PAPER_SPECIFIED`: Algorithm2 sends uncertain outputs to whole-image DeepSolo;
  it does not specify CLIP, SigLIP2, these prompts or a50% acceptance budget.
- `REPOSITORY_SPECIFIED`: preserve epoch61 Text+Kernel proposals, original RGB
  crop hashes, raw PARSeq and balanced character confidence, all words in routing,
  the original TRAIN80/45 partition and previously audited per-engine metrics.
  Frozen branch metrics contain425 images, full TEST DeepSolo and63 TRAIN outputs.
- `ASSUMPTION`: five crop prompts and four scene-transcript prefixes, deterministic
  character replacement/reversal/deletion controls, weakest-word native/CLIP rank
  fusion and two crop/global combinations. Reading order uses bbox top then left.
  Context overflow preserves all characters through prefixed chunks: weakest
  scene cosine, normalized mean for crop prompt chunks. Phrases are text
  embeddings, not generative instructions or proof of exhaustive text detection.
- `ASSUMPTION`: fit GT-free rank references/cut nearest50% coverage on TRAIN80;
  choose prompt family/model by retained-word precision on TRAIN45, then retained
  words/name tie breaks. Transfer unchanged cuts to TEST. Descriptive exact image
  budgets62 TRAIN/149 TEST use score/ID ranking without correctness labels.
  These are exploratory comparisons because prior TEST results were inspected.
  Report care precision TP/(TP+FP), TN0 accuracy TP/(TP+FP+FN), recall/F1, emitted
  words, safe and complete images, and missing fallback IDs separately by split.
- `HARDWARE_DEVIATION`: sequential FP32 batch8 on CPU here/CUDA host, deterministic
  kernels and TF32 off, with model/shape/batch/peak allocated/reserved/error records.
  SigLIP2 FixRes uses its native compatible SiglipModel plus fast-tokenizer JSON,
  fixed64-token padding and no attention mask. No package install, NaFlex
  substitution, non-strict checkpoint loading or DeepSolo runtime change.
- `UNRESOLVED`: no real SigLIP2 weights can be downloaded inside the sandbox;
  pinned224/384 model revisions and published SHA256 are configured, but real
  checkpoint/runtime/memory qualification requires the provided GPU-host run.
  Existing weights can qualify real CLIP prompt inference here; mocked/tiny random
  SigLIP API tests do not establish pretrained-checkpoint compatibility or gain.
  A newly rejected TRAIN image without DeepSolo leaves the entire combined TRAIN
  metric null. No missing sample is dropped and no output is substituted.

Evidence and alternatives: SigLIP2 official2025 paper/model cards motivate updated
multilingual/dense features; official HF documentation distinguishes compatible
FixRes from NaFlex and specifies fixed text padding. CLIPTER and CVPR2025 scene
text retrieval use learned adaptations, so their gains are not assumed to apply
here. MetaCLIP2 needs a newer class/runtime and is deferred instead of upgrading
shared dependencies. Scene-caption, text-occlusion and crop-contact-sheet controls
are future proposals, not executed variants. See
[CLIP_ROUTER_PROMPT_ABLATION.md](CLIP_ROUTER_PROMPT_ABLATION.md), the isolated v2
configs and per-image experiment outputs outside Git.

Qualification:52 focused tests pass, including real tiny SiglipModel API shapes
(random weights, structural compatibility only) and TRAIN-only model selection
when TEST metrics favour the opposite model. Full CPU FP32/batch8 old-model
experiments complete125 TRAIN +300 TEST images each. Independent scalar/CDF/
partition/chunk/count audits pass425 images and52 points per model (104 total).
B/16 baseline3865-word contrast replay has maximum difference8.20e-8; B/32 TRAIN
baseline is independently replayed against its original frozen125-image run.
At matched62 TRAIN/149 TEST image budgets, no tested model/prompt improves full
TEST TN0 over the old B/16 baseline51.8479%. Its TRAIN retained-word precision
is66.4723% and TEST62.2699%; B/16 all-text English crop prompt gives TRAIN57.4866%,
TEST61.4496% and final TEST TN051.5493%; Italian gives TRAIN63.2153%, TEST59.9802%,
final TEST TN049.7371%. Values are exploratory, not probability guarantees.
The TRAIN45 winner B/32/plain-scene (99/158=62.6582%) has final matched-budget
TEST TN051.3324%, reinforcing that tiny validation gains do not qualify a better
router. Original router remains unchanged. Real SigLIP2 weights and GPU memory
qualification still require the supplied host command; complete combined TRAIN
metrics still require missing fallback predictions rather than omitted samples.

## WORD-GT-002 — User-requested smaller word targets after v1 training

- `PAPER_SPECIFIED`: re-read local paper section3.3/Algorithm1 and section4.1.
  Foreground text pixels yield component boxes, which are padded before cropping.
  The paper does not prescribe eroding GT, word-core targets, an erosion ratio
  or interpreting padding as evidence for smaller masks. The filled word target
  and proposed shrink remain outside STRICT reproduction.
- `REPOSITORY_SPECIFIED`: inspected v1 target construction, its completed
  training bridge, native128 paired owner/separator transforms, existing
  `segmentation/totaltext_word_kernel.py` distance-based nuclei and the saved
  TEST reports. The existing two-head kernel/expansion model is not introduced;
  the user requests a new target for the same single-output AG-U-Net.
- `ASSUMPTION`: create a separate compact-v2 delivery from checksum-qualified
  v1 owners. Per-word interior distance uses anisotropic128-reference units;
  requested inset is min(0.55 times maximum interior distance,2 reference pixels).
  Keep the largest connected candidate, with stable raster-order ties. This
  intentionally supervises smaller filled interiors and may remove letter pixels;
  it is not character-only GT and no longer promises v1 stroke protection.
- `ASSUMPTION`: back off the inset by halves (up to8 reductions) when the
  candidate retains less than20% of source area, less than80% of longest bbox
  span, or the unchanged six-reference-pixel crop padding cannot cover the
  original owner bounds. Restore the source owner if no safe candidate exists.
  These are explicit untuned construction parameters, not paper values.
- `ASSUMPTION`: replay effective nonaugmented training targets (bilinear mask
  resize/threshold0.5, nearest owners/validity/separators, cross-owner contact
  removal). Restore any additionally lost word, iterating until no new losses
  remain. Already absent source-native words are reported, not fabricated.
  Random-augmentation survival remains unresolved. Restored inherited source
  fragments are reported rather than silently discarded.
- `REPOSITORY_SPECIFIED`: retain V4/v1 architecture,500 epochs, batch8, FP32,
  optimizer, seed, augmentation and TRAIN-IoU checkpoint selection; use new
  protocol `practical_totaltext_tighter_words_500_v2`, artifact
  `ART-TOTALTEXT-COMPACT-WORDS-V2`, config/run/output paths. The shared trainer
  explicitly rejects the wrong target source. Previous runs cannot be reused
  as v2 checkpoints merely because they have the same model architecture.
- `ASSUMPTION`: retain the prior native128 contour filter150, padding6, top10
  and PARSeq for a target comparison. Smaller components may be filtered more
  often; the existing padded-box matching issue is not solved by altering GT.
  Actual improvement, and whether a different crop/metric protocol is warranted,
  remain `UNRESOLVED`; no TEST recognition score selected shrink parameters.

New source/target image hashes, instance identities, per-word removed areas,
geometry/native restorations, resolved parameters and independent binary8-CC
checks are saved. Original GT and source photos are unchanged. Generation runs
on CPU; no GPU training is launched while preparing the requested rerun command.

WORD-GT-002 qualification: regenerated and independently verified4017 masks
with24734 preserved care owner IDs, zero interword component sharing, and zero
additional nonaugmented native128 disappearances relative to v1. The same two
inherited TRAIN fragments remain recorded. Total-Text foreground decreases
37.04% TRAIN/42.80% TEST; other split reductions are documented in
`docs/TIGHTER_WORD_GT_V2.md`. All four ZIP members and generation fingerprints
match delivered files. Seven split smoke exports and47 focused tests pass;
tiny CPU integration fixtures use explicit single-process loading as before.
Old-v1 and compact-v2 host preflights pass with all input checksums and pinned
PARSeq artifacts; original v1 config/manifest hashes remain unchanged. No new
physical GPU training or TEST recognition score is produced here.

### CLIP-ROUTER-LITERATURE-003 — DataComp, DFN and larger SigLIP 2 router tests

User-authorized follow-up to CLIP-ROUTER-PROMPT-002. Inspected the existing
CLIP/PARSeq router, frozen v2 runtime/evaluation and local paper section3.5 /
Algorithm2. This is **approximate reproduction**, with an isolated v3 config and
runner; strict scoring and historical scientific source files are unchanged.

- `PAPER_SPECIFIED`: local Algorithm2 motivates uncertain-image DeepSolo fallback.
  It specifies MPNet/fuzzy scoring, not CLIP, the new encoders or these thresholds.
- `REPOSITORY_SPECIFIED`: reuse the checksum-qualified epoch61 Text+Kernel/PARSeq
  source, all predicted words and original crop hashes, v2 score families, original
  TRAIN80/45 partition and audited425-image engine metrics. Keep all983 TRAIN /
  2204 TEST care words, original matching/ignore rules and frozen reference gate.
- `ASSUMPTION`: test DataComp-L14 and DFN-H14 foundation encoders motivated by
  CLIP4STR TableIII, plus SigLIP2 L16/384 FixRes motivated by its2025 pretraining.
  CLIP4STR's trained decoders/recognition gains and NaFlex OCR gains are not assumed
  for these frozen encoders. Expected effect is stronger text-sensitive features;
  whether that improves reliable whole-image routing is an empirical question.
- `ASSUMPTION`: use the publisher-repository HF conversions at immutable revisions,
  not mutable defaults. Validate native/HF dimensions, layers, vision/text heads
  and GELU/QuickGELU, and reject every missing/unexpected checkpoint key. DataComp
  uses its published HF processor; DFN's missing HF processor is reconstructed
  only from its explicit native squash/bicubic/RGB mean/std profile. DataComp's
  explicit vision head-width64 comes from OpenCLIP v2.30.0's published default;
  DFN's80 comes directly from its native config. No package install is performed.
- `ASSUMPTION`: the pinned DFN HF conversion has49409 embedding rows, whereas its
  native config/tokenizer have49408 tokens. Permit only the documented unused row,
  preserve all weights, verify actual BOS49406/EOS49407/EOS-padding and masks, and
  reject a tokenizer sequence using the extra row. HF modern EOS pooling and
  DataComp's legacy eos=2/first-argmax pooling must select actual EOS. Do not modify
  model configs, discard rows or use non-strict loading to hide incompatibility.
- `ASSUMPTION`: enforce SigLIP2 paper2.1 lowercasing before Gemma tokenization during
  both chunk-length checks and embedding; retain raw PARSeq transcription in
  artifacts. Use published right padding, EOS insertion/no BOS, fixed native text
  context and no attention mask. Fast tokenizer JSON needs no SentencePiece install.
- `ASSUMPTION`: when the web page does not establish a weight digest (DFN HF file),
  resolve publisher Hub LFS metadata at the pinned full revision before execution,
  verify downloaded/existing bytes and seal a receipt. Record its digest and actual
  model SHA in resolved metadata; missing/foreign receipts and changed bytes fail.
  The native OpenCLIP file checksum must not be used for the different HF file.
- `ASSUMPTION`: select family/model using TRAIN45 retained-word precision, with the
  inherited tie breaks; transfer TRAIN80 50%-coverage cut unchanged to TEST. Compare
  GT-free exact62 TRAIN/149 TEST image budgets descriptively. Report separate
  exploratory threshold curves on a configured0..1 grid in0.05 steps plus the TRAIN
  cut and all observed score breakpoints. Never select a threshold using TEST GT.
  Report words, precision TP/(TP+FP), requested TN0 accuracy TP/(TP+FP+FN), recall/F1,
  accepted/fallback IDs and missing outputs. A score is not a correctness probability.
- `HARDWARE_DEVIATION`: sequential FP32/batch2 with deterministic kernels and TF32
  off on the8GB-target GPU or explicitly selected CPU. Persist device/model/shape/
  batch/software metadata, CUDA allocated/reserved peaks and propagated OOM errors.
  Sequential models and a lower batch can affect numerical outputs; existing-weight
  smoke tests qualify execution, while GPU memory/output equivalence is pending.
- `UNRESOLVED`: all three new checkpoints are absent and sandbox DNS/CUDA access
  prevents their local inference. Real converted-checkpoint compatibility, native
  numerical equivalence, newer-model routing accuracy and GPU peaks need the host
  run. Structural random-model tests are not pretrained-checkpoint qualification.
  Newly rejected TRAIN images without saved DeepSolo keep complete TRAIN metrics
  null; full TEST fallback is available. No failed/missing image is dropped.
- `UNRESOLVED`: RN50x16 needs an absent native ResNet CLIP runtime; NaFlex and
  MetaCLIP2 need distinct newer classes/environments. They are deferred rather than
  silently represented by ViT/FixRes encoders. Past TEST inspection makes all v3
  comparisons exploratory; no strict reproduction or Table5 claim is supported.

Evidence, pinned artifacts, commands and alternatives are in
[CLIP_ROUTER_LITERATURE_MODELS.md](CLIP_ROUTER_LITERATURE_MODELS.md), including
CLIP4STR, SigLIP2, CVPR2025 scene-text retrieval and publisher model/config pages.
67 focused tests pass, including real tiny HF CLIP pooling/pad invariance, native
pixel preprocessing, Gemma fast JSON/SigLIP API compatibility, weight receipts,
preserved source callbacks, TRAIN-only selection and threshold denominators/missing
fallback. Initial synthetic fixtures lacked image_size in two tests; corrected,
both pass. Real cached B16/L14 CPU FP32/batch2 smoke inference passes2 TRAIN+2 TEST
images/model; no full accuracy or new-model gain is inferred from this partial run.
Final smoke artifacts verify current code/plan/checkpoint hashes; B16 feature replay
against frozen v2 on the4 images differs by at most1.49e-7. CPU peak CUDA fields
are null; actual new-checkpoint/GPU qualification remains pending.

<a id="padgt-001"></a>

## PADGT-001 — separated-v1 cached-mask padding diagnostic (2026-10-13)

The user requested a padding comparison of the previous separated-v1 network
while compact-v2 trains. This is an **approximate reproduction**, exploratory
post-hoc analysis of the complete Total-Text TEST300 split, not an independently
validated hyperparameter choice or a paper Table5 result.

- `PAPER_SPECIFIED`: section3.3 of the authoritative local v2 PDF applies padding
  to localized text boxes and limits crops to ten per image.
- `REPOSITORY_SPECIFIED`: preserve existing RETR_EXTERNAL/CHAIN_APPROX_SIMPLE,
  unpadded **bbox** area150 and clipped padded-bbox-area sorting. Bbox-area
  filtering/sorting differ from the paper's component-area path and remain
  confined to this approximate diagnostic.
- `ASSUMPTION`: reuse the frozen separated-v1 epoch485 native128 predictions
  (checkpoint `dc85f29b…87eb7c`) on all300 TEST images. Compare the predeclared
  native-pixel padding grid0,1,2,4,6,8 with baseline6, without regenerating masks,
  changing the filter, retraining, or consulting GT when generating proposals.
  Padding cannot change the retained count under this filter/cap; sorting can
  change selected identities in images with more than ten candidates.
- `ASSUMPTION`: preserve registered one-to-one bbox-IoU>=0.50 casefold-exact
  matching on all2204 care words, including empty/missed outputs and punctuation.
  Report separately matching on padded crop boxes and unpadded detection boxes,
  plus image-level lexical multiset counts. Crop-box IoU improvement alone does
  not establish OCR improvement. Predictions on ignored regions are not
  neutralized by this inherited diagnostic metric.
- `HARDWARE_DEVIATION`: rerun qualified PARSeq on CPU FP32/batch16/two threads
  to leave the training GPU available. Use deterministic algorithms and seed
  202503. Rerun baseline6 too and audit text/confidence parity against saved GPU
  inference; numerical differences are possible. CUDA memory fields are null
  because this experiment does not use CUDA.
- `UNRESOLVED`: the exact paper padding and PARSeq-S artifact remain unresolved.
  Use the existing provenance-qualified PARSeq checkpoint/runtime only with the
  approximate label. TEST inspection cannot establish generalization of the
  best observed padding; a future independently evaluated choice needs TRAIN
  calibration and an untouched evaluation protocol.

The isolated plan is
`configs/approximate/totaltext_separated_v1_cached_padding_test.json`; execution
is `scripts/evaluate_cached_totaltext_padding.py`. It verifies source-report,
manifest, mask, source-crop and PARSeq hashes; freezes GT-free proposals and OCR
before reading original annotations; preserves crop pixels, per-sample outputs,
source provenance, CPU/GPU parity and both metric families. Identical RGB crops
are deduplicated using shape and pixel hashes.

Full execution and independent saved-output audit completed: all300 images,
all2204 care words,937 retained crops per padding and5622 distinct RGB crops.
For padding0,1,2,4,6,8, unpadded-detection-box casefold-exact counts are
401,401,395,380,309,253; registered padded-crop-box counts are
401,396,361,219,113,54. Padding0/1 yield18.1942% versus14.0200% for6 using
unpadded-box matching. The original padded-box criterion includes a geometric
effect and must not be conflated with this OCR improvement. Only image547
changes top10 identities for0,1,8; the count remains937 in every variant.

- `ASSUMPTION`: an additional post-hoc control intersects retained contour IDs
  between each padding and6 before scoring. This GT-blind intersection controls
  for sorting/capping differences. Padding0 and1 still produce401 exact matches
  versus309 for6 on the same936 components: the net gain92 remains entirely in
  the paired-component comparison. The0/1 tie is descriptive, not independently
  validated padding selection.

CPU/GPU baseline6 text parity is937/937; maximum absolute sequence-confidence
difference is9.5367e-6. CPU FP32/batch16 execution took902.7s; no failed image or
sample was excluded and no CUDA allocation was used.55 focused tests and a real
3-image/42-crop smoke test passed. An independent SciPy sparse bipartite-matching
audit agrees for every padding, metric family and paired-component control,
and verifies all crop/OCR/mask/annotation/code hashes. Results, per-sample
outputs, audit, CSV and visual examples are under
`artifacts/diagnostics/totaltext-separated-v1-padding-test-20261013/`.
The running compact-v2 experiment and its configuration were not changed.

## PAPER-IMAGE-ROUTER-001 — joined-T3 paper baseline against frozen CLIP

The user requests the paper router with T3 constructed from **all** detected
words, Figure4-style captions, and a baseline for the existing CLIP comparison.
Inspected the authoritative local v2 PDF sections3.3–3.5/Figure4/Algorithm2,
existing whole-caption MPNet/fuzzy scoring, BLIP2/PARSeq adapters and the frozen
CLIP population/branch protocol. This is an **approximate reproduction**,
isolated from STRICT and from earlier completed CLIP experiment code.

- `PAPER_SPECIFIED`: use a BLIP2 image description, MPNet sentence cosine,
  semantic T1/T3 selection, alpha0.6/beta0.4/tau0.8 and DeepSolo(I) fallback.
  Algorithm2's strict S1>S3 comparison selects T3 on a semantic tie. Figure4
  describes environment, text-bearing objects, spatial arrangement and semantic
  cues; section3.4 favors medium descriptions.
- `ASSUMPTION`: user-authorized T3 concatenates every frozen crop reading in the
  same bbox top/left/instance order as CLIP. Preserve repetitions, ignored-region
  outputs and low confidence; empty readings force fallback. L-004's undefined
  multiple-crop aggregation remains unresolved; concatenation is not asserted
  to be a uniquely established paper procedure.
- `ASSUMPTION`: follow the prose's selected-semantic-candidate confidence.
  Preserve S1/S3/L1/L3/C1/C3, literal winner/text and literal fallback. Do not
  silently use the diagram's independent maxima or the existing max-paired-C
  interpretation. Algorithm2 uses underdefined S/T variables; S-001 remains
  unresolved for strict claims.
- `ASSUMPTION` (superseded): the first localized-word comparison vetoed every
  T1 winner because it lacks word boxes. The user's subsequent clarification
  explicitly requires accepting a winning T1 as one prediction **only when C1
  passes the threshold**, matching the selected-candidate interpretation.
  Current primary gate `paper_selected` thresholds C1 for a T1 winner and C3
  for a T3 winner; `joined_t3` remains a separate ablation. T1 is never
  represented by the crop outputs or given a fabricated bbox.
- `REPOSITORY_SPECIFIED`: reuse the qualified pinned approximate PARSeq runtime,
  FlanT5-XL BLIP2 and all-mpnet-base-v2; raw case-sensitive RapidFuzz ratio and
  the epoch61 crop outputs/one-to-one care metrics. Exact PARSeq-S, BLIP2
  ViT-B, MPNet checkpoint and fuzzy preprocessing remain unresolved. DeepSolo
  remains the fallback via unchanged, qualified, frozen outputs; its runtime
  documentation and strict-status record were read before integration.
- `ASSUMPTION`: restate the visible, truncated Figure4 prompt, request a coherent
  40–80-word description, and use explicit deterministic three-beam generation.
  Additional structured/long/six-question profiles are diagnostic alternatives,
  not strict paper decoding. Preserve native answers, repetition and out-of-range
  lengths; no OCR/GT in prompts, filler, truncation or authored scene repair.
  The displayed description is72 words; exact full prompt, token-vs-word limits
  and decoding remain C-002/C-003/C-004 ambiguities.
- `ASSUMPTION`: InstructBLIP FlanT5-XL is an explicitly named instruction-caption
  ablation using its native processor/model and independently pinned revision,
  shard and tokenizer hashes. It does not substitute for the primary BLIP2
  `figure4_single` reference. A TRAIN image-only probe motivates its style;
  neither TEST scores nor recognition correctness select the caption profile.
- `ASSUMPTION`: compare the same125 held-out TRAIN/300 TEST images and983/2204
  care words, frozen ignore flags, TRAIN80 coverage calibration and
  descriptive62/149 image budgets. Report fixed threshold/breakpoint curves,
  accepted words, FP=0 nonempty safe images and FP=FN=0 complete images.
  Semantic coherence is not evidence that all scene words were detected. Metrics
  are micro precision TP/(TP+FP) and requested TN0 TP/(TP+FP+FN), never accuracy
  inferred from embedding magnitude.
- `ASSUMPTION`: MPNet context overflow beyond384 tokens and any empty reading
  force whole-image fallback rather than silently excluding/truncating words.
  Missing DeepSolo outputs leave full combined metrics unavailable with explicit
  IDs. TEST300 fallback is complete; TRAIN has63 outputs and62 missing, so a new
  router may require further qualified fallback inference before full TRAIN
  accuracy is available.
- `HARDWARE_DEVIATION`: sequential models, batch1, explicit CPU FP32 T1/MPNet;
  CPU FP32 or CUDA FP16 captions; optional CPU offload with6GiB GPU/48GiB CPU
  weight budgets, deterministic kernels and TF32 disabled. Record placement,
  input/processor shape, timing, CPU RSS and allocated/reserved GPU peaks;
  propagate OOM/failures. No quantization or dependency installation. Actual
  CUDA/offload and8GB peak qualification remain pending, not inferred from CPU.
- `REPOSITORY_SPECIFIED`: reuse the newly generated425 native CPU T1 results via
  an immutable manifest and original source-plan/shard hashes. Verify RGB IDs,
  checkpoint/source/preprocessing/strict loading, recognizer code and identical
  software before reuse; retain original metadata instead of relabeling results
  after caption-profile changes.
- `UNRESOLVED`: score/aggregation and caption checkpoint/prompt/decoding
  ambiguities block STRICT and Table5 claims. One-image caption probes do not
  establish population caption quality, full-image completeness, selector
  precision or improved recognition. Full425 captions/new-router accuracy and
  missing TRAIN fallbacks are not declared complete.

- `ASSUMPTION` (user T1 clarification): primary evaluation matches raw word
  predictions one-to-one within each image by Unicode casefold exact text,
  retaining punctuation and duplicates, without a bbox-IoU criterion. Apply
  this same rule to T1, T3, DeepSolo and replayed frozen CLIP decisions; never
  combine unlocalized T1 word counts with the old geometric comparison.
  T1 is exactly one raw sequence, is not split on whitespace, and is care
  because no location permits ignore assignment. Other GT words remain FN.
  Preserve prior T3/DeepSolo ignore and invalid-envelope rules. Original report,
  GT and independent TEST prediction artifacts are SHA-pinned, and geometric
  counts replay against frozen branches before word scoring. Record each
  accepted source and separate T1/T3 image counts. Preserve the old geometric
  results/curves as a separate diagnostic; the paper's metric remains E-001
  unresolved.

Initial integrated CPU smoke before the T1 clarification: TRAIN total_text:14, native T1=`1`, joined
T3=`TOKU TOKU 38h PERANIA`. BLIP2 reference returns7 words, S3=0.125569,
C3=0.140648 and semantic winnerT1. InstructBLIP returns51 words/3 sentences,
S3=0.254959/C3=0.168611 and winnerT3. Both request DeepSolo at0.8. Visual review
finds the51-word style closer to Figure4, but the quoted sign reading "Toyotu"
is wrong and a busy street is speculative. Forced-long BLIP2 returns47 words
with absent cars/parking; four-question34-word and six-question26-word probes
also remain generic/incorrect. These negative observations are preserved, not
repaired or hidden by length checks.

The initial71 focused tests passed, including complete-transcript overflow, T1 reuse,
native InstructBLIP family metadata, inclusive thresholds, frozen CLIP
source/split checks, care denominators and missing-fallback accounting. An
initial synthetic test fixture had inconsistent engine GT denominators; it was
corrected to preserve the same GT population and the test passed. Final real
CPU smoke verifies current code/config/weights and raw outputs; CPU lifetime RSS
peaks16,575,545,344/16,886,480,896 bytes, with CUDA fields null. Implementation,
commands, probe evidence and qualifications:
[PAPER_IMAGE_ROUTER_BASELINE.md](PAPER_IMAGE_ROUTER_BASELINE.md).

The T1-acceptance update passes83 focused tests and a new real CPU two-caption
smoke under `/tmp/paper-image-router-baseline-v2-t1-smoke/d2c52b5444b3/` with
current code/config/weights and sealed annotation/word-branch receipts.
Full425-image word-source replay verifies the unchanged983/2204 care word
denominators and original geometric branch counts; caption/router qualification
remains one-image smoke. On that image both selected confidences remain below0.8.
The lower-cut smoke confirms accepted T1 contributes exactly one `1` prediction
(FP=1, FN=2), with no crop-word substitution or dropped GT. Complete new-router
accuracy and CUDA/offload memory qualification remain pending the GPU host run.


## CKPT-WORD-001 — disk-limited separated-v1 checkpoint diagnostic (2026-10-13)

User-authorized checkpoint comparison; no deletion is authorized by this diagnostic.
The /extra1 filesystem was full at the initial check; later3.7GiB free were observed.
compact-v2 stopped after343 logged epochs;
epoch344.pt.tmp is incomplete and no successful run manifest/TEST report exists.
Separate-v1 holds500 checkpoints (~230GiB), compact-v2 ~158GiB.

- `PAPER_SPECIFIED`: Table3 foreground pixel IoU/F1 remain reproduction metrics.
- `REPOSITORY_SPECIFIED`: original strict model loading, validation preprocessing,
  source care-owner maps and frozen practical contour policy are reused.
- `ASSUMPTION`: compare epochs50,100,200,226,300,375,447,485,500 on32 deterministic
  TRAIN samples, then up to3 shortlisted models on disjoint96 TRAIN samples.
  All images were used in training; these subsets are not independent validation.
  Matching is maximum one-to-one native owner-mask IoU>=0.5; original bbox matching
  with IoU>=0.5 separately measures operational crops before adding padding.
  Eight-connected prediction components; every source owner stays in the denominator,
  including128-resolution disappearance. Ignore pixels excluded only for native
  metrics. Touching multiple GT owners and fragmented GT words are separately counted.
  Empty-word precision/recall/F1 use0; empty pixel union uses IoU/Dice1.
- `HARDWARE_DEVIATION`: CPU FP32/batch1/eight threads, contiguous tensors;
  CUDA is unavailable in the sandbox. No crops, masks or weights are saved;
  only numeric reports under /tmp, with a10MiB output budget. CPU/GPU numerical
  equivalence beyond existing cached baseline is not asserted.
- `UNRESOLVED`: the best checkpoint among all500 and independent validation
  generalization are unknown. Word-instance F1 is a task-specific diagnostic,
  not a replacement for the paper's Table3 metric or a strict reproduction claim.

Plan/code/tests/results: /tmp/word-checkpoint-selection-v1-20261013/.
Retain recommendations must preserve the original485 for existing result provenance,
latest500 for resumability, and any tested native/operational word-F1 winners.
Weights are read in place; nothing is deleted or copied. Full qualification pending.

CKPT-WORD-001 qualification completed:9 checkpoints on32 TRAIN images/199 GT,
then485/447/226 on disjoint96 TRAIN images/701 GT. Native word-mask F1 is
43.8247%/51.9838%/50.1548%; operational original-box TP is179/195/192 and
box F1 is37.2141%/40.2062%/39.5469%. The447 wins both confirmed F1 rankings.
Retain plan preserves447/485/500;497 other v1 weights account for228.608GiB.
No weights were deleted, copied or cropped.7 focused tests and a strict-load
CPU inference/aggregate/hash audit pass; initial smoke API and audit syntax
errors were corrected and documented. All scalar outputs occupy about1MiB.
This is neither an exhaustive500-checkpoint search nor independent validation.
Durable code/config/tests/per-image results/audit/retain plan:
artifacts/diagnostics/checkpoint-word-selection-v1-20261013/.

## V2-WORD-PAD-001 — compact-v2 Word-F1 and streaming padding diagnostic (2026-10-13)

The user authorized checking the v2 word-F1 selection signal, deleting unnecessary
v1 checkpoints while retaining the previously selected447/485/500, and comparing
v2 padding without saving crops. These are approximate diagnostics.

- `REPOSITORY_SPECIFIED`: compact-v2 resolved config selects full-TRAIN macro
  foreground pixel IoU; validation is disabled. Its343 completed epoch records
  contain no word-instance F1. The original full-TRAIN pixel-IoU winner is239.
  epoch344.pt.tmp is incomplete; no successful500-epoch run manifest exists.
- `PAPER_SPECIFIED`: section3.3/Algorithm1 uses padding and at most10 blocks;
  Table3 pixel IoU/F1 remains separate from this word-instance metric.
- `ASSUMPTION`: reuse the qualified CKPT-WORD-001 native owner-mask one-to-one
  mask-IoU>=0.5 metric, all original care-owner denominators and fixed32/96 disjoint
  TRAIN subsets. Screen50/100/150/200/239/250/300/319/343, then confirm the original
  IoU winner and shortlisted native/operational Word-F1 winners on96 images.
  These images participated in training and are not independent validation.
- `ASSUMPTION`: freeze the confirmed TRAIN native-mask Word-F1 winner before TEST
  and explore padding0/1/2/4/6/8 in128-resolution coordinates on official TEST300.
  Report one-to-one original unpadded-box IoU>=0.5 plus casefold-exact OCR, padded
  crop-box matching separately, and lexical multiset results. The unchanged
  repository bounding-box-area150 filter and padded-box-area sorting are reused
  for v1 comparability; they differ from paper-specified component-area behavior.
  GT annotations are loaded only after proposals/OCR are frozen. Padding can
  change top10 identities through the inherited padded-area sort. Small compact
  targets can reduce unpadded-box IoU and the number admitted by the area filter.
- `HARDWARE_DEVIATION`: CPU FP32 segmentation/batch1/eight threads followed by
  qualified PARSeq/batch16/two threads, because CUDA is unavailable in this session.
  Crop pixels are streamed in memory. Only JSON, texts, coordinates and pixel
  hashes are saved; no rasters/weights are copied. Scratch outputs use /tmp with
  explicit small-output budgets. CPU/GPU equivalence for new v2 predictions is
  not claimed.
- `UNRESOLVED`: exact PARSeq-S artifact and paper padding remain unavailable;
  the qualified official PARSeq checkpoint is an approximate substitution. The
  sweep omits caption/context/fallback and cannot support Table5 claims. Best
  checkpoint over all343, independent validation generalization and complete
  500-epoch v2 training remain unresolved.

v1 pruning preflight validates all500 planned weights/sidecars against source
records and the SHA256 of retained447/485/500. Application failed on the first
unlink with EROFS: /extra1/bmartelli/context-str-runs is mounted read-only in this
session. No checkpoint or sidecar was removed; reclaimed bytes0. The safe host
launcher preserves the three weights and all run metadata; eligible497 weights
represent228.608GiB. This is a filesystem access failure, not pending user consent.

Code/config/tests and deletion audit:
artifacts/diagnostics/tighter-v2-word-padding-20261013/.
Qualification/results are pending at this entry's initial creation.

V2-WORD-PAD-001 checkpoint qualification completed:9 screened epochs on32 TRAIN
images/199 care words;239/300/319 confirmed on disjoint96 TRAIN images/701 care
words. Native mask Word-F1 is49.7151%/37.2671%/51.4673%, respectively. Native
matched-word counts are349/270/342; predicted component counts703/748/628.
Operational localized-word counts are161/174/148; bbox Word-F1 is
35.1913%/36.1746%/32.5633%. Thus319 wins native mask Word-F1,239 raw native TP,
and300 operational Word-F1/word count among the confirmed models. These are
different objectives; none is independent validation or an exhaustive343-model
search. The TEST padding checkpoint is explicitly frozen to319 from TRAIN
native mask Word-F1, SHA2566d85b9db02d964d52c739f16d2410dde70fc63a149f7034e96db0ad6685d1106.
32 focused tests pass; real strict-load checkpoint239 smoke and24-crop PARSeq
smoke pass. The initial TEST smoke used the wrong dataset root and failed with
FileNotFoundError; corrected to the qualified TEST root and rerun successfully.
The original failure is recorded in qualification.json. Full padding is pending.

V2-WORD-PAD-001 full approximate TEST qualification completed on300 images/2204
care words with319.741 proposals at each padding;4446 distinct crops processed
only in RAM. Primary unpadded-box/casefold-exact correct counts for0/1/2/4/6/8 are
349/389/400/381/352/282; padding2 is highest (18.1488% of all care GT).
Padded crop-box correct counts are349/406/423/384/271/147; these additionally
change matching geometry. Lexical multiset counts are367/412/427/415/391/319.
Unpadded geometric matches remain501 for every pad; no candidate identity or
top10 truncation changes occurred, so the primary count gain is OCR-related.
The original TEST GT exactly matches the earlier v1 diagnostic on all300 images.
V1 primary best401 versus v2 best400 is descriptive: training completion,
checkpoint selection and supervision differ; no causal GT superiority is claimed.
Independent NumPy box-IoU/SciPy maximum matching audit, all12 checkpoint
aggregate checks, source annotation/code hashes and32 focused tests pass.
CPU padding elapsed834.6s, peak total process RSS1.158GiB; no GPU memory is
fabricated. No rasters/weights were saved or copied; durable numeric outputs
are approximately10MiB. V1 pruning remains blocked by EROFS with zero deletions;
the supplied host launcher executes the authorized447/485/500 retention plan.

## WORD-REVIEW-20261014 — review of supervision and trained-model experiments

The user requested restudying generated masks and trained networks. This
authorizes a lightweight review and cached-coordinate analysis; no training,
new GT, crops, weights or changes to original scientific paths are introduced.

- `REPOSITORY_SPECIFIED`: inspect V4/v1/v2 resolved configs, current epoch logs,
  original GT construction and source paper3.2/3.3. All share frozen ResNet50,
  one-output AG-U-Net,256 RGB/128 masks, masked BCE/separation weight1, Adam
  and the same training hyperparameters. All343 shared epoch order hashes agree.
  No independent validation exists for those three original models. The separate
  V4 native256 three-arm ablation has a125-image validation split and different
  TEST localization policy (no area filter or cap); it must remain separate.
- `PAPER_SPECIFIED`: padding surrounds localized components. It supplies no
  inference that supervised word masks must be eroded. Component-area filtering
  and top10 are specified; inherited bbox filtering remains approximate.
- `ASSUMPTION`: replay all cached external contour bboxes from retained/rejected
  metadata for V4 epoch458, v1 epoch485 and v2 epoch319. Check source image hashes,
  identical original TEST300/2204 care GT and exact150/pad6/top10 replay before
  comparing the existing area150/no-area policies at common native128/pad2
  bbox sorting/top10. Score unpadded-box maximum one-to-one IoU>=0.5. This is
  post-hoc TEST geometry exploration, not threshold calibration or new OCR.
- `UNRESOLVED`: component pixel areas are unavailable from bbox metadata;
  area-policy effects on PARSeq accuracy, best validation/deployment settings,
  causal superiority of v2 targets and the true best OCR checkpoint are unknown.
  Additional geometric matches cannot be promoted to correct recognized words.

V1 host pruning audit now reportsCOMPLETE:497 weights/497 sidecars deleted,
447/485/500 retained,228.608GiB reclaimed. Current /extra1 has about233GiB free.
Compact-v2 still records343/500 complete epochs and incomplete344 temporary file.
Code/plan and resulting review are isolated under
artifacts/diagnostics/word-mask-experiment-review-20261014/.

WORD-REVIEW-20261014 qualification: two focused tests,900 exact cached proposal
replays and1800 separately audited NumPy/SciPy per-image matching graphs pass.
Original TEST GT agrees for all300 images/2204 care words. Bbox-area150 versus0,
both with native128/pad2 ranking/top10 and unpadded IoU>=0.5, yield V4/458
517→726, v1/485551→811 and v2/319501→813 geometric matches. V2 precision drops
67.61%→40.61%, proposals741→2002. Extra geometric matches are not extra OCR.
V4 cached pad6 OCR re-scored against unpadded boxes gives294 correct words;
the previous padded-box criterion130 was reproduced first. No new inference.
The initial metadata inspection assumed object-list shard records and raised
TypeError; corrected to the actual path-list schema before scoring.
The separate native256 TEST record reports562/636/739 exact words atpad0 for
baseline/separator/separator+residual, with no area/top10 filtering and a125-image
validation split. It must not be conflated with original native128 runs or used
to attribute the gain solely to GT or resolution. Full review and per-image
results are stored in the isolated package; total output remains below1MiB.

## HIGHRES-ROUTING-001 — frozen router transfer to residual native256 (2026-10-14)

The user requests an actual routing test on the best high-resolution model.
This authorizes a small separate runner, approximate configurations and results;
original model, crop, recognition and routing implementations remain preserved.
Paper section3.5/Algorithm2 and existing CLIP/context/DeepSolo boundaries were
inspected before implementation. The complete qualified TEST300 DeepSolo cache
and original-image BLIP-2/T1 caches permit an end-to-end cached evaluation.

- `PAPER_SPECIFIED`: BLIP-2 caption, MPNet cosine/fuzzy context confidence,
  alpha=.6/beta=.4/tau=.8 and DeepSolo original-image fallback.
- `REPOSITORY_SPECIFIED`: completed residual high-resolution epoch435, checkpoint
  SHA25699620c7962578fecc5c33101a490ca63c2a056b0ba2551cc5e89cba7b4e22346;
  RGB512/native256, pad0 external contours, existing GPU PARSeq batch32 outputs.
  Existing host-isolated DeepSolo source/config/checkpoint/response contracts and
  inclusive post-inference score>=.4 reuse are checked by the canonical preflight.
- `ASSUMPTION`: transfer the frozen CLIP B16 native_word_advantage TRAIN80 CDFs
  and TRAIN125 threshold .146906 from Text+Kernel epoch61 without fitting again.
  This does not establish calibration on the residual model. Every prediction
  enters the minimum; blank/empty OCR forces fallback, no GT-dependent gate.
- `ASSUMPTION`: preserve prior high-resolution no-filter/no-top10 pad0 proposals,
  all outputs count against care GT; no prediction ignore exclusion. This differs
  from the prior CLIP experiment's native-support ignore and is explicitly
  reported. Pad0/model selection already explored TEST; this is a transfer
  diagnostic, not an independent final setting or official DetEval protocol.
- `ASSUMPTION`: paper prose confidence of the semantic winner, Algorithm2 strict
  semantic tie selects T3. All crop outputs form reading-order T3. T1 has one raw
  unlocalized sequence; a combined spatial score is unavailable if T1 is accepted.
  Lexical multiset and localized bboxIoU>=.5 exact casefold metrics stay separate.
- `HARDWARE_DEVIATION`: sequential CPU FP32 CLIP/MPNet; reuse historical CUDA
  PARSeq, FP16 CPU-offloaded BLIP-2 and isolated CUDA FP32 DeepSolo outputs with
  their actual runtime metadata. No new GPU/OOM/equivalence claim. No new masks,
  crops, checkpoints, downloads or dependency changes.
- `UNRESOLVED`: exact paper PARSeq-S, BLIP-2 and DeepSolo identities/metric,
  optimal residual-model TRAIN calibration and pad/filter settings. All results
  are approximate reproduction, without strict/Table5 claims.

Runner: scripts/evaluate_totaltext_highres_routing.py. Configurations:
configs/approximate/totaltext_highres_routing*_v1.json. Read-only preflight
checks all300 source contour reconstructions, images, masks, annotations, OCR,
T1/captions, model hashes and frozen routing references before inference.
An initial preflight failed because a string path reached a Path-only SHA helper;
the boundary now normalizes to Path and a regression test checks both path types
and modified bytes. No inference/results were produced by that failed preflight.
The first focused test run reported1 failure/50 passes: the blank-crop fixture
declared the wrong crop hash. The fixture was corrected; the actual hash check
and failure behavior remain enforced. The real two-image CPU routing smoke
completed with both full-image DeepSolo reuse and new CLIP/MPNet scoring.

HIGHRES-ROUTING-001 qualification:51 focused tests and the final-code two-image
CPU smoke pass. Complete TEST300/2204 succeeds with0 omitted samples,0 new
crop/mask/weight files. Spatial exact TP: residual-PARSeq739, transferred CLIP
plus DeepSolo1286, DeepSolo-only1588. CLIP accepts119 and routes181 images;
net gain547 versus PARSeq, net loss302 versus DeepSolo-only. Paper BLIP-2/MPNet
tau.8 accepts0/300; maximum selected confidence .6250795576, so the combined
output is entirely DeepSolo. Lexical-only counts780/1300/1595 stay separate.
All predictions count: no prediction ignore exclusion,2040 DeepSolo outputs
versus2008 in the prior native-support-ignore protocol; GT denominator unchanged.
Independent SciPy/Counter/bisect/polygon/EXIF audit passes every per-image and
aggregate count, with0 maximum CLIP score difference. Results/configuration,
source router contracts/CDFs and per-sample metadata are in
artifacts/diagnostics/totaltext-highres-routing-20261014/. New CPU FP32 inference
took267.13s after preflight; peak process RSS1,727,512,576byte. No new GPU
allocator/OOM/equivalence qualification or whole-pipeline latency claim.

## V1-447-OCR-001 — complete OCR of Word-F1-selected v1 (2026-10-14)

The user requests full447 OCR before old-v2 deletion/restart commands to assess
the checkpoint metric. Authoritative paper3.2/3.3/3.5/4.1 and the existing v1/v2
checkpoint/padding/model/training implementations were inspected.

- `PAPER_SPECIFIED`: RGB256/native128, pretrained frozen ResNet50 AG-U-Net,
  padding/top10, PARSeq recognition; paper4.1 pixel foregroundIoU/F1 remain
  distinct from custom object Word-F1. Exact paper artifacts remain unresolved.
- `REPOSITORY_SPECIFIED`: reuse qualified streaming v2 evaluator geometry/code,
  preserving external contours, native bbox-area150 filtering, padded-box-area
  ordering/top10, floor/ceil remap and casefold exact matching/punctuation.
  Epoch447 SHAe9146a514adc91bb39a36b394a646710cf5028539c2e2bec8a82fdc55b0ddc30
  was selected previously from9 screened/3 confirmed checkpoints on128 TRAIN-used
  images; epoch485 is the original complete-TRAIN pixel-IoU winner.
- `ASSUMPTION`: fixed six native128 pads0/1/2/4/6/8, complete official TEST300/2204;
  primary unpadded detection bboxIoU>=.5 and one-to-one exact text, separate
  padded-box/lexical metrics. Original care GT only, all predictions count.
  Padding comparison is post-hoc exploration, not independent tuning.
- `HARDWARE_DEVIATION`: CPU FP32 segmentation/batch1/eight threads; sequential
  CPU PARSeq/batch16/two threads. Source485 masks were CUDA, so a real2-image
  smoke checked parity and found1 changed pixel on one image. Consequently all
  300 segmentations were re-run for485 in the same CPU conditions as447.
  CPU OCR5550 crops was reused only on identical RGB hashes/pinned recipe and
  equality with frozen previous predictions;78 changed crops were re-recognized
  in RAM. No crop/mask/weight export. No full CPU/GPU equivalence claim.
- `UNRESOLVED`: causal/general predictive validity of Word-F1, exhaustive best
  OCR checkpoint, independent padding, exact paper architecture/PARSeq-S/metric.

Qualification:447 completes300/2204 with914 proposals/pad,5484 unique crop arrays,
correct primary counts420/420/421/397/341/282. Controlled CPU485 completes the
same300/2204 with938 proposals/pad,5628 unique crop arrays, counts
401/401/395/380/307/252. The original cached-GPU-mask485 result was
401/401/395/380/309/253; the known numerical difference is preserved, not silently
relabelled. Geometry447579 versus CPU485550 (549 atpad8). Best OCR421 versus401
supports the proposed Word-F1 choice for these two candidates; it is not a
proof over all500 checkpoints or a strict/Table5 reproduction.
Separate NumPy IoU/SciPy maximum-matching audits pass all per-image criteria,
all2204 GT/300 samples and12 historical checkpoint aggregates. Full source and
output provenance are in artifacts/diagnostics/v1-447-ocr-and-v2-restart-20261014/.
The initial485 control was interrupted before a complete result to correct
copied explanatory notes naming447; its initial sources/configuration and
interruption receipt are retained. Only the verified control enters the final
comparison. No numerical parameter was selected using those partial outputs.

## V2-WORD-RESTART-001 — held-out metrics and bounded storage (2026-10-14)

The user authorizes preparing commands to remove all old compact-v2 weights
and restart with best-checkpoint retention, after447 OCR. No new500-epoch run
or real old-weight deletion has been executed in this task.

- `REPOSITORY_SPECIFIED`: compact-v2 masks/owner/validity/separation manifests,
  original author-clarified model, frozen ResNet50 ImageNet1K_V2, RGB256/mask128,
  paired transforms, masked BCE/separation_weight1, Adam1e-4/weightdecay.01,
  FP32/batch8, seed202503/500 epochs remain the source protocol. Encoder bytes
  are supplied locally and strictly loaded; SHA256
  11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca.
- `ASSUMPTION`: withhold the frozen125 official TRAIN IDs and optimize1130.
  They were used in earlier exploratory work; this is validation excluded from
  this new optimizer, not a fresh blind benchmark. TEST is not accessed.
  Monitor native mask Word-F1 (valid pixels,8-connected, maskIoU>=.5, original
  care-owner denominator including disappeared native words), operational bbox
  Word-F1 and macro foregroundIoU, retaining per-sample/statistical evidence.
- `ASSUMPTION`: keep unique epochs for the three best signals plus last,
  maximum4 completed weight files/approximately1.84GiB, plus one transient
  atomic write. Metric ties use earliest epoch. Preserve optimizer/RNG, complete
  epoch/per-sample logs and strict resume/source-contract validation. Selected
  and obsolete checksums are checked before any retention unlink.
- `HARDWARE_DEVIATION`: real CPU/batch1 engineering preflight only; source
  host GPU not visible to this session. CUDA preflight explicitly failed before
  training, with null allocator peaks. Prepared host launcher qualifies real
  FP32/batch8 GPU train/validation and records allocated/reserved/OOM behavior
  before500 epochs; later epoch records retain actual GPU/memory metadata.
- `UNRESOLVED`: GPU/batch8 qualification on the executing host, strict model/
  artifact/metric identity and general OCR predictiveness of each retained signal.

41 focused tests pass, including discarded-checkpoint integrity checks and
synthetic weights-only pruning that preserves run metadata. Final-code real
CPU smoke passes one optimizer step and two validation images. New trainer is
isolated in scripts/train_totaltext_compact_v2_word_selection.py and
configs/approximate/totaltext_compact_v2_word_selection_v1.json; previous trainer
and strict defaults are untouched. Snapshot recipe/checks/source provenance and
commands are in the diagnostic package. The old-v2 weights-only inventory is
read-only PREFLIGHT_PASS:343 weights,343 sidecars and incomplete344 temporary,
169,510,218,530 logical bytes (157.87GiB). No actual old weights or GT deleted;
GT, original epoch/config metadata and v1 keep447/485/500 are preserved.

### V2-WORD-RESTART-001 execution qualification (2026-10-14)

The subsequent user instruction explicitly authorizes executing both the reviewed
old-v2 cleanup and the prepared training launcher with full host access. The
earlier preparation-only and GPU-unavailable statements above describe the
historical session, not this execution.

- `REPOSITORY_SPECIFIED`: the frozen weights-only cleanup plan was applied after
  confirming no active old-v2 training. It removed343 complete weights,
  343 SHA sidecars and the incomplete epoch344 temporary file, reclaiming
  169,510,218,530 logical bytes (157.87GiB). The old checkpoint directory is
  empty. Original epochs.jsonl/resolved_config.yaml and retained v1 weights
  447/485/500 were verified against their frozen SHA256 values after cleanup.
  GT was not deleted. Available space on /extra1 increased to about391GiB.
- `HARDWARE_DEVIATION`: actual single NVIDIA GeForce RTX3090,24GiB,
  driver560.28.03, Torch2.4.1+cu124. This differs from the documented RTX3070
  target and the paper hardware; no equivalence claim is made. The prepared
  model, FP32/batch8, augmentation, optimizer, deterministic settings and
  TF32-disabled protocol are unchanged. No AMP, offload or reduced batch size
  was introduced.
- `ASSUMPTION`: use the already documented1130/125 held-out split and retain
  the three metric winners plus the last epoch, with earliest-epoch tie handling.
  Native-mask Word-F1 and operational bbox Word-F1 are custom validation
  signals; neither is the paper's recognition accuracy or pixel F1.
- `UNRESOLVED`: exact strict architecture/artifact/metric identity and general
  OCR predictiveness remain unresolved. GPU/batch8 execution and allocator
  qualification are now resolved for this approximate experiment only.

The real CUDA preflight passed one optimizer step on8 TRAIN images with input
8x3x256x256 and native output8x1x128x128, followed by two validation images.
Peak allocated CUDA memory was1,166,638,592 bytes; peak reserved memory was
2,248,146,944 bytes. No OOM occurred. The explicit OOM/error receipt handler and
launcher failure propagation remain enabled; no artificial OOM was injected.
The fresh500-epoch run was launched in a detached process at17:47:08 UTC and
completed its first epoch with all1130 TRAIN and125 validation images. Epoch
records, validation outputs, selected checkpoints and allocated/reserved peaks
are saved under the separate external run directory. No TEST images, OCR crops,
new GT, downloads or dependency changes enter this launch.

Actual cleanup audit, launch receipt and GPU qualification are recorded under
artifacts/diagnostics/v1-447-ocr-and-v2-restart-20261014/execution-20261014/,
with the timestamped deletion audit in its parent directory. The previous
publication manifest and sealed preparation snapshots are preserved unchanged.
The training remains in progress; starting it is not completion of500 epochs
or a strict/Table3/Table5 reproduction claim.

## V2-SELECTED-OCR-SIZE-001 — selected checkpoints, OCR padding and mask areas (2026-10-15)

The user requests correct recognized word counts at different paddings for all
four retained compact-v2 checkpoints, and the sizes of their predicted masks.
Paper sections3.2/3.3 and the existing streaming447 OCR, native128 localization,
strict checkpoint loader and qualified PARSeq adapter were inspected first.

- `PAPER_SPECIFIED`: localize segmentation output, pad text blocks and cap at10
  crops before recognition (paper3.3). Mask dilation is not part of crop padding.
- `REPOSITORY_SPECIFIED`: preserve the previous approximate diagnostic's
  RETR_EXTERNAL, native bbox-area150 filtering, clipped padded-box-area ordering,
  floor/ceil original-coordinate remapping and qualified official PARSeq
  autoregressive decoding with one refinement. This bbox-area filter differs
  from paper component-area filtering and is not promoted to STRICT.
  Completed training500 and frozen validation125 select native-Word-F1 epoch382,
  bbox-Word-F1 epoch361, pixel-IoU epoch126 and last500. Their source contracts,
  full epoch history, weights and current scientific source hashes are checked.
- `ASSUMPTION`: compare pads0/1/2/4/6/8 in native128 pixels on all TEST300/2204
  care words. Primary metric is maximum one-to-one unpadded-box IoU>=.5 plus
  Unicode casefold exact text, punctuation retained; all predictions count and
  no failed sample leaves the denominator. Padded-box and lexical metrics remain
  separate. All proposals and OCR are fixed before annotations are read.
- `ASSUMPTION`: report actual foreground pixels per128x128 image, per-image
  foreground fraction, pooled8-connected foreground-component areas including
  rejected fragments, and pooled retained component/bbox dimensions before and
  after crop padding. Holes do not count as foreground. Per-image and per-object
  aggregation are explicitly separate; no averaging of component medians.
- `HARDWARE_DEVIATION`: single RTX3090 CUDA FP32, segmentation batch1 and then
  sequential PARSeq batch32; deterministic settings, TF32/AMP disabled, explicit
  CUBLAS workspace configuration. Crops exist only in RAM; outputs are numeric
  JSON, OCR text and pixel hashes, within128MiB. No packages or weights are
  downloaded/copied. Record allocated/reserved peaks and propagate OOM/errors.
  Earlier447/485 CPU results are not a controlled device-matched comparison.
- `UNRESOLVED`: exact paper architecture/PARSeq-S/recognition metric, independent
  padding choice and general predictive validity of the selection metrics. This
  is post-hoc TEST exploration, approximate reproduction only, not Table5.

Implementation is isolated in scripts/evaluate_totaltext_compact_v2_selected_ocr.py
and configs/approximate/totaltext_compact_v2_selected_ocr_padding_v1.json.
Results and qualification are recorded separately under
artifacts/diagnostics/compact-v2-selected-ocr-20261015/; old results and training
artifacts remain unchanged.

V2-SELECTED-OCR-SIZE-001 qualification:77 focused tests pass. The real two-image
CUDA smoke covers all four models and six paddings, including PARSeq batch32.
Complete TEST evaluation succeeds for300 images/2204 care words per checkpoint,
with16,742 unique RGB crop arrays recognized in RAM. Primary exact spatial
counts at pads0/1/2/4/6/8 are epoch126:365/388/387/373/344/295;
epoch361:419/437/432/425/380/313; epoch382:366/392/394/381/361/299;
epoch500:316/355/360/345/331/271. Epoch361, the validation bbox-Word-F1 winner,
leads all six paddings among these four models; this limited result does not
establish a universal checkpoint metric. Best437/2204 is19.83%, with pad1 chosen
post-hoc on TEST. Padded-box and lexical counts are retained separately.

Mean foreground fractions are7.183%/7.964%/7.127%/5.847% for epochs126/361/382/500.
Retained component median foreground areas are208/212/216/208 native pixels;
all-component medians are43/43/34/32, including fragments rejected by the crop
filter. Retained proposal counts are797/864/730/659. No padding changes the
top10 identities in this evaluation; the binary masks are never padded/dilated.
Independent NumPy IoU/SciPy maximum-matching/Counter auditing passes14,400
per-image evaluations, all reported integer metric totals and size distributions.
Eight real CUDA mask replays reproduce exact native array hashes; independent
SciPy8-connected labeling agrees with OpenCV component areas and boxes.

The full run takes125.55s, with peak CUDA allocated458,178,048 bytes and reserved
522,190,848 bytes; peak process RSS1,828,651,008 bytes. No OOM occurs and no
artificial OOM is injected. The error handler records OOM/peaks and re-raises;
there is no model/device/precision fallback. Outputs contain no new raster or
weight files. This remains approximate post-hoc segmentation/PARSeq evaluation,
without context routing or strict/Table5 reproduction claims.

## NATIVE128-GT-REVIEW-001 — matched comparison and visual review (2026-10-16)

The user requests comparison against all native128 model versions, actual GT
and prediction views, and analysis of further target shrinking. Local paper3.2,
3.3/Algorithm1 and the existing training, preprocessing, streaming OCR and
native contour implementations were inspected before this isolated diagnostic.

- `PAPER_SPECIFIED`: segmentation, component localization, padding and maximum10
  blocks; the paper leaves numeric padding unspecified. It does not prescribe
  eroding word GT until a padding of6 becomes optimal.
- `REPOSITORY_SPECIFIED`: p6 and bbox-area150, external contours and padded-box
  sorting are upstream choices. The original-coordinate and later native128
  adaptations are distinct. This diagnostic preserves the established native128
  bbox-area150/top10 path for the primary comparison, rather than relabelling
  that path as strict component-area localization.
- `ASSUMPTION`: compare all15 retained full-training selected native128 weights
  on identical TEST300/2204 care annotations, CUDA FP32 preprocessing, official
  PARSeq AR/refine1 and six native paddings0/1/2/4/6/8. Separate cross-dataset
  models from Total-Text-trained variants. Exclude qualification-only models;
  failed/no-weight runs and deleted historical compact-v2 checkpoints remain
  explicitly unavailable. Two ICDAR13 full runs may have identical tensor states;
  calculate state hashes instead of treating serialization hashes as differences.
- `ASSUMPTION`: a controlled additional crop uses6 original RGB pixels per side
  on the frozen pad0 selected components. It changes only crop padding, not the
  native filter or ranking. It is not a replay of the complete original pipeline.
  For V4/458, separated-v1/447 and compact-v2/361, also disable the bbox-area
  threshold while preserving top10, to measure the interaction between shrinking
  targets and the existing crop filter. Report all proposal counts and precision.
- `ASSUMPTION`: primary correct words use maximum one-to-one unpadded original
  bbox IoU>=.5 plus Unicode-casefold exact, punctuation retained; all predictions
  count, care-word denominator complete, no ignored-prediction exclusion. TEST
  exploration cannot independently select deployment parameters or identify a
  causal GT-only gain when training duration/split/checkpoint selection differ.
- `ASSUMPTION`: show actual unaugmented TRAIN supervision through its paired
  native transform, with validity/owner/separator rules preserved, alongside real
  predictions. Fixed common TRAIN IDs11/427/1085 are visual examples, not a
  quantitative evaluation subset. Compute target-size statistics on all1255
  TRAIN images for five Total-Text target versions; other-domain model cards
  must identify their own supervision/source images without calling reference
  Total-Text labels their training GT. No new training masks are generated.
- `HARDWARE_DEVIATION`: RTX3090 CUDA FP32, sequential segmentation batch1 and
  PARSeq batch32, deterministic algorithms, TF32/AMP disabled, pinned local
  artifacts, explicit CUBLAS workspace. Crops remain in RAM. Small requested
  composite visualizations may be saved; no weights/crop dataset is copied.
  Record allocated/reserved peaks and propagate inference/OOM errors.
- `UNRESOLVED`: exact strict artifacts/architecture/recognition metric, original
  paper crop calibration and independent controlled benefit of shrinking alone.
  All outputs are approximate reproduction, without strict/Table5 claims.

Plan: configs/approximate/native128_gt_review_v1.json. Evaluation:
scripts/review_native128_ocr.py. New outputs are isolated under
artifacts/diagnostics/native128-gt-review-20261016/; historical results are kept.

Qualification (2026-10-16):98 focused tests pass. An initial two-image smoke
stopped because the current training validator rejects an archived ICDAR13
num_workers0 configuration. The failed partial attempt is preserved. An isolated
inference-snapshot reader now checks mode, shapes, normalization, interpolation,
geometry and architecture without imposing current training-worker policy or
mutating historical bytes; a regression test covers this boundary. Strict
checkpoint loading remains enforced. Final15-model/18-pipeline GPU smoke passes.
Two ICDAR13 files have identical tensor states:15 checkpoints,14 distinct states.

Complete matched TEST300/2204 succeeds; independently audited NumPy IoU/SciPy
matching/Counter totals pass37,800 per-image evaluations. At the best native
padding, original pixel-TT/curved/V4/separated447/compact361 give98/355/361/421/437
correct words with bbox-area150. V4/separated447/compact361 without the area
filter, still top10, give525/648/708 at native pad1. For compact361 this changes
437/864 correct/proposals to708/2146: OCR precision50.58%→32.99%, exact spatial
F128.49%→32.55%. Separated447 without filter has F132.69%, slightly higher than
compact's32.55%, despite fewer correct words. Count/precision tradeoff is explicit.
Compact361 with6 ORIGINAL pixels yields434, compared with380 at6 NATIVE pixels
and437 at1 native. It is not meaningful to force GT erosion to make native6 best.
The four compact models exactly reproduce all24 prior native-padding counts.
Historical V4 native6 was294; the new uniform runtime gives295. Both are kept,
without silently conflating historical device/backend/batch settings.

All1255 TRAIN targets were replayed for five GT versions with actual paired
nonaugmented preprocessing. Native foreground mean: pixel3.11%, curved8.36%,
V48.94%, separated8.82%, compact5.70%. Separated→compact reduces total native
foreground35.39%. Importantly, full-resolution owner connectivity is insufficient:
native-word topology on all9285 original TRAIN words gives separated68 fragmented
words/24 absent, compact275 fragmented/8 absent. There are245 newly fragmented
words and38 repaired; no newly absent words. These are observed target properties,
not predicted-word counts. Direct owner replay matches every one of2510 actual
native-target array hashes before the connectivity comparison. The reduction can
improve survival while increasing fragmentation. Additional blind shrinking is
not supported; native connectivity, filter/padding calibration on validation and
a matched1130/125 GT-only training comparison remain appropriate next analyses.
No further training or replacement GT was executed here.

Full OCR uses86,316 unique RAM crops,501.94s, CUDA FP32 on RTX3090; allocated peak
458,178,048 bytes, reserved522,190,848 bytes, RSS2,067,902,464 bytes. No OOM or
fallback; no artificial OOM injection. Error receipts retain exceptions/peaks.
Requested figures use exact real arrays with nearest-neighbour mask display;
three common TRAIN examples and each checkpoint's own training example are
exported, with explicit provenance and no TEST-performance implication. The
optional metric chart initially failed because matplotlib is absent; no install
was attempted and complete tables/data-derived mask figures replace that chart.
No original GT, weights, strict defaults or historical results are changed.

## V3-TRAIN-OCR-001 — guarded V3 training and automatic PARSeq evaluation (2026-10-17)

The user explicitly requests starting training with V3 and then measuring
localized/correct words with PARSeq. Paper3.2/3.3/4.1, V2 training/selection,
native metrics, the V3 runtime adapter and existing selected-checkpoint OCR
were inspected before implementation. This is approximate reproduction.

- `PAPER_SPECIFIED`: attention-aided ResNet50 segmentation followed by padded
  connected-component text localization, top10 reference and recognition.
- `REPOSITORY_SPECIFIED`: matched V2 author-clarified model, pretrained encoder
  artifact, initialization/seed202503, FP32 batch8, Adam1e-4/weightdecay.01,
  masked BCE and500 epochs. Reuse existing validation and safe checkpoint
  retention primitives. The only target modification is the pinned V3 recipe.
- `ASSUMPTION`: same1130/125 TRAIN holdout and selection signals as V2; native
  Word-F1 uses FINAL V3 owners, all original word IDs including missing ones,
  maskIoU>=.5; operational bbox Word-F1 and foregroundIoU remain recorded.
  These125 images were explored before and are not a fresh blind evaluation.
- `ASSUMPTION`: retain unique winners of native Word-F1, bbox Word-F1,
  foregroundIoU and last (at most4 weights plus an atomic transient), with
  earliest exact ties. Record compressed per-image/per-word guard diagnostics
  for every training epoch; unresolved V3 targets remain explicit.
- `ASSUMPTION`: after complete TRAIN-only selection evaluate all retained
  models on complete TEST300/2204 with fixed native padding0/1/2/4/6/8,
  bbox-area0 and150, top10. Report primary one-to-one unpadded bboxIoU>=.5
  Unicode-casefold exact words, separate padded-box sensitivity and mask sizes.
  Independent NumPy/SciPy audit follows each area arm. TEST never selects
  checkpoints or changes training; padding analysis remains exploratory.
- `REPOSITORY_SPECIFIED`: genuine locally pinned PARSeq AR/refine1, with
  artifact/revision checks; the evaluator's existing V2 path stays unchanged
  and an explicit V3 plan validates the V3 contract/recipe namespace.
- `HARDWARE_DEVIATION`: actual PAPER RTX3090, single GPU FP32, no AMP/TF32,
  batch8 training then sequential segmentation batch1/PARSeq batch32. Run real
  CUDA optimizer/validation and PARSeq preflights, record allocator peaks and
  propagate OOM/failures. Freeze an executable source snapshot for the durable
  training->OCR subprocess chain; no crop/weight copies in the repository.
- `UNRESOLVED`: V3 OCR benefit, strict artifacts/metric and generalization
  across seeds. No performance conclusion is drawn before the training and
  full OCR/audit stages complete.

Configs: guarded_v3_training_v1.json and guarded_v3_training_ocr_v1.json in
configs/approximate/. Training: scripts/train_guarded_v3.py; durable chain:
scripts/run_guarded_v3_training_ocr.py. New outputs are isolated from all old
runs. No packages are installed and no old checkpoint is deleted.

## WORD-GT-003 — post-transform guarded word targets V3 (2026-10-17)

The user explicitly requests the three proposed safeguards: connectivity after
resize/separation/augmentation, per-word backoff for fragile words, and joint
separation/crop-coverage verification. Paper3.2/3.3, current paired transforms,
V1/V2 generation, dataset loading and NATIVE128-FAIL-001 were inspected first.

- `PAPER_SPECIFIED`: binary word localization with connected components and
  padding; no explicit GT erosion or native connectivity repair recipe.
- `REPOSITORY_SPECIFIED`: pin existing V1 reference and compact-V2 TRAIN1255
  sources; preserve legacy RGB transform, augmentation RNG/epoch/index mapping,
  validity, source GT files, old training protocols and STRICT configuration.
- `ASSUMPTION`: opt-in dynamic V3 adapter after the complete native transform.
  Keep satisfactory V2 cores; grow defective ones within the same transformed
  V1 owner support. Accept only one8-connected component satisfying area, both
  bbox spans and crop coverage, while forbidding contact with other retained
  words. Preserve every original valid V2 foreground pixel; do not select only
  a largest fragment or erase a neighbour to claim success.
- `ASSUMPTION`: initial untuned parameters retain V2 maxgrowth2,8 steps,
  minarea.2/minspan.8, with span checked on both axes. Padding2 for the coverage
  guard comes from existing approximate localization config, not a numeric
  paper prescription or new TEST optimization. A reference bbox short side<=2
  protects the complete permitted source shape. This is a candidate policy,
  not an independently selected optimum.
- `ASSUMPTION`: nearest-owner reference restoration may replace the old local
  bilinear-threshold core only when guards require it. It remains inside the
  same valid source owner, explicitly departs from old target rasterization,
  and does not change RGB interpolation or the historical transform code.
- `ASSUMPTION`: qualify all1255 TRAIN images unaugmented and under actual
  deterministic epoch0/1 flip/rotation; include all9285 source word IDs in each
  denominator, even if rotation/clipping/native rasterization loses them.
  Export unaugmented native previews plus a runtime recipe; previews must not
  be passed through the old source-space loader and resized/rotated again.
- `UNRESOLVED`: impossible native-grid topology, coverage/clearance tradeoffs,
  behavior across every future augmentation and causal OCR improvement.
  Unrepairable targets retain all original fragments and explicit per-word
  failure reasons. No blanket connectivity guarantee or OCR gain is claimed.

This is approximate reproduction, CPU-only GT construction/qualification.
No new training, GPU job, package installation, strict default or old GT change.
Plan: configs/approximate/guarded_word_gt_v3.json. Implementation is isolated in
segmentation/guarded_word_gt.py; ordinary SegmentationDataset calls its adapter
only when the explicit make_guarded_dataset factory is selected.

Completed WORD-GT-003 evidence (2026-10-17):42 focused tests pass; real3-image
smoke covers47 source words under all3 views. Complete qualification covers
TRAIN1255/9285 words per view, with3765 independent OpenCV image checks.
Unaugmented V2->V3:275->23 fragmented and8->5 missing words, zero cross-owner
contacts;6641 changed words and240 words with at least one unresolved guard.
Of those,31 fail crop coverage and217 fail the both-axis span constraint
(overlapping reasons). Foreground1172351->1544731 native pixels; this explicitly
restores31.76% more area than V2, remaining smaller than the previous V1 target
population. Thus this is not a claim that targets become uniformly smaller.
Actual augmented epoch0:300->28 fragmented,30->19 missing,267 unresolved;
epoch1:278->22 fragmented,25->18 missing,248 unresolved. All source IDs count;
unrepairable words and fragments are retained and individually reported.
The deterministic greedy order is ascending source owner ID; growth uses8
linear radii0.25..2 then the permitted full source fallback. This ordering and
schedule are ASSUMPTION, not a globally optimal separation/coverage solution.
The training-facing AuditedGuardedWordDataset transports native owners, recipe
hashes and per-word JSON evidence through DataLoader workers. Real epoch1 CPU
batch2/2-worker smoke is byte-identical to direct indexing and reproduces the
complete audit hashes for all3 selected images. RGB and validity are unchanged.
Runtime390.15s CPU, peakRSS1173991424 bytes; no GPU inference or training.
Unaugmented native128 previews, per-word audits and comparison figures are in
artifacts/diagnostics/guarded-word-gt-v3-20261017/. Runtime usage and candidate
policy limits: docs/GUARDED_WORD_GT_V3.md. OCR benefit remains UNRESOLVED.

## NATIVE128-FAIL-001 — failure diagnosis and native GT audit (2026-10-16)

The user requests analyzing failures and how to improve GT. Inspect local
paper3.2/3.3, existing mask generation/native transforms, frozen cap experiment,
matching and previous native GT topology before implementing this diagnostic.

- `PAPER_SPECIFIED`: binary segmentation, component localization, padding and
  max10 blocks. No numeric word-shrink recipe or padding6 is established.
- `REPOSITORY_SPECIFIED`: qualified frozen V4/458, separated-v1/447 and
  compact-v2/361 native128 predictions/PARSeq; previous primary unpadded bbox
  IoU>=.5 Unicode-casefold metric; existing owner/validity/separator/contact
  native-training transform. Keep sources and existing GT unchanged.
- `ASSUMPTION`: fixed padding1/area0 cap10 versus unlimited diagnosis on complete
  TEST300/2204. Seed exact top10 matching, augment unlimited exact, then geometry
  to produce additive categories while preserving matched GT vertices; check
  every cardinality independently with SciPy. Geometry minus exact OCR is a
  joint crop/recognizer failure, not proof that recognition alone is responsible.
- `ASSUMPTION`: characterize merge/fragment candidates using bbox-coverage
  proxies (coverage>=.5, fragment GT fraction>=.05); never equate these with
  pixel-mask topology. Report padded-box sensitivity separately. A per-contour
  padding oracle uses GT only for diagnostic upper bounds, never inference;
  one contour cannot supply multiple words through multiple padding reads.
- `ASSUMPTION`: actual unaugmented TRAIN1255 owner survival/connectivity at128
  versus a target-only256 counterfactual. Replay actual128 target hashes and
  compare existing separated/compact GT. No new model, target regeneration or
  production configuration is authorized by this diagnostic. Proposed native
  topology guards must subsequently be tested under augmentation and matched
  training before claiming OCR gain.
- `UNRESOLVED`: exact strict artifacts/metric, pixel-topology failure attribution
  from bbox-only caches, causal benefit of GT edits and independent cap/padding
  calibration. TEST findings are exploratory, not hyperparameter selection.

CPU-only read-only source analysis; no GPU inference, weights, crop exports,
installations or training. Numerical outputs and a few scientific comparison
figures are isolated in artifacts/diagnostics/native128-failure-analysis-20261016/.
Plan: configs/approximate/native128_failure_analysis_v1.json.

Completed NATIVE128-FAIL-001 evidence (2026-10-16):34 focused tests pass;
all300 TEST images and2204 care GT retained, every matching cardinality checked
independently with SciPy. Fixed V2/361 padding1, area0:708 exact words within
top10;71 additional exact words beyond10;190 geometrically localized but not
exact OCR;1235 fail unpadded bboxIoU>=.5 even without cap. These four categories
partition the2204 words via nested maximum matchings. Bbox proxies indicate
possible split/merge cases but do not establish pixel-mask topology.
All2510 actual native128 TRAIN target hashes replay exactly. Of9285 words,
separated/compact targets have68/275 fragmented and24/8 missing words;
592/1443 surviving word boxes have short side<=2 pixels. At256, target-only
compact replay has191 fragmented and0 missing words, not a new trained model.
245 fragmented compact words have connected separated-v1 counterparts; this
is a fallback opportunity count, not a validated regenerated target set.
Under the same frozen OCR, padded-box scoring changes V2 top10 exact count
708->777 and unlimited779->891; these are metric sensitivity comparisons,
not improvements of model outputs. Of190 localized OCR failures,56 best-IoU
texts differ only after removing non-alphanumeric characters;102 are one edit
away. Official GT and primary normalization remain unchanged. Three illustrative
TRAIN words have connected full-resolution compact owners yet fragment/vanish
under the real native conversion. Source review confirms the generator checks
native owner presence, not one connected native component per word.
Proposed post-transform connectivity/coverage guards and adaptive shrink remain
ASSUMPTION; no GT or models regenerated and no OCR gain claimed. CPU-only
numeric outputs, two small figures and provenance are in the isolated package.

## NATIVE128-CAP-001 — crop-count cap ablation (2026-10-16)

The user asks whether changing the maximum mask/crop count improves or worsens
results. The preceding experiments still used top10 after disabling area. This
task evaluates5/10/20/unlimited without changing any trained model or GT. Local
paper3.3/Algorithm1, existing contour ranking, frozen cache and matching code
were inspected first.

- `PAPER_SPECIFIED`: maximum10 localized text blocks is the paper reference.
- `REPOSITORY_SPECIFIED`: reuse complete frozen external native128 contours,
  stable clipped padded-bbox-area ranking, existing bbox-area0/150 policies,
  PARSeq AR/refine1 readings and six native pads0/1/2/4/6/8. Check all source
  publication/config/result/shard/RGB hashes and replay every top10 proposal.
- `ASSUMPTION`: change only the cap to5/20/unlimited as a user-requested
  counterfactual research diagnostic, explicitly outside the paper procedure.
  This is not a resolution of an ambiguous parameter. Neither strict nor
  existing approximate production defaults are changed. Compare V4/458,
  separated-v1/447 and compact-v2/361 on all TEST300/2204, with unchanged weights.
- `ASSUMPTION`: freeze one reading for each exact RGB crop across caps; reuse
  compatible prior GPU OCR only after RGB-hash/recipe checks, infer missing
  below-top10 crops in RAM. GT only scores outputs and never selects proposals.
  Micro exact-casefold text plus maximum one-to-one unpadded bboxIoU>=.5,
  punctuation retained, all predictions and care GT count. Report emitted crops,
  exact precision/recall/F1; nested caps cannot lose exact TP with fixed readings,
  but precision and F1 can fall. Parameters remain exploratory TEST choices.
- `HARDWARE_DEVIATION`: RTX3090 CUDA FP32 PARSeq batch32 only for new crops,
  deterministic algorithms, TF32/AMP disabled, CUBLAS workspace explicitly set.
  No new segmentation inference, rasters, crops, weights or installations.
  Record allocator allocated/reserved peaks and OOM/exception receipts.
- `UNRESOLVED`: strict checkpoint/metric identities, independently calibrated
  cap/padding and production speed/accuracy tradeoff. Changing the paper cap is
  not a strict/Table5 reproduction.

Plan: configs/approximate/native128_crop_limit_v1.json; runner:
scripts/evaluate_native128_crop_limit.py. Separate results:
artifacts/diagnostics/native128-crop-limit-20261016/.

Completed NATIVE128-CAP-001 evidence (2026-10-16): all300 TEST images and2204
care words,144 conditions (3 weights x2 area policies x4 caps x6 native pads).
70 focused tests and the real2-image CUDA smoke pass. Independent NumPy/SciPy
matching verifies all43,200 per-image scores;10,800 top10 proposal replays and
all36 top10 aggregate baselines match the previous uniform run exactly.
At fixed native padding1 without area filtering, cap10->unlimited exact words
are V4/458:525->560 (F1:27.1318%->27.5591%), separated-v1/447:648->715
(32.6942%->33.5052%), compact-v2/361:708->779 (32.5517%->31.3418%).
Compact-v2 emits2146->2767 crops, so additional correct words do not imply
improved precision/F1. With bbox-area150, compact-v2 already has no image above
10 eligible crops and all cap>=10 results are identical. The best tested native
padding remains1 for compact-v2 at every cap. These are exploratory TEST
diagnostics, not independent model/hyperparameter selection evidence.
38,574 unique RGB readings,32,157 qualified reused and6417 new GPU crops;
allocated205,955,584/reserved247,463,936 bytes, no OOM, no saved crops/weights.
Numeric outputs, per-image audit, resolved configuration and code snapshots are
stored in the separate result package. Strict/production defaults are unchanged.

## CLIP-ROUTER-LITERATURE-CONTEXT-001 — host alternatives and valid prompt boundaries (2026-10-14)

User-authorized verification of the alternative CLIP-family selectors. Inspected
the existing v3 runtime/runner, frozen v2 prompt chunker, completed host results,
and local paper section3.5/Algorithm2. This remains **approximate reproduction**.

- `PAPER_SPECIFIED`: uncertain recognition invokes whole-image DeepSolo in local
  Algorithm2; its context scorer uses MPNet cosine and fuzzy lexical similarity
  with weights0.6/0.4 and threshold0.8.
- `REPOSITORY_SPECIFIED`: audit the sealed DataComp L14 and DFN H14 full425-image
  host runs, old B16/B32 controls, original TRAIN80/45 partition, frozen13 score
  families, native OCR inputs and saved fallback availability. The audit verifies
  source/checkpoint/config/result/feature hashes and208 comparisons on1700
  features before applying the context fix. Old source bytes are preserved in
  the external audit artifact. No cached routing outputs are rewritten.
- `ASSUMPTION`: replay the frozen selections on the current paper-router word
  metric: casefold-exact one-to-one words within images, punctuation retained,
  no bbox IoU, same frozen ignore/invalid-envelope handling for all engines.
  This gives a consistent comparison to the completed T1/T3 baseline; retain
  the old geometric metrics separately and do not identify either with the
  paper's undefined recognition accuracy. TEST was previously inspected, so
  all comparisons remain exploratory. No TEST correctness chooses a gate.
- `ASSUMPTION`: compare the GT-free descriptive62 TRAIN/149 TEST image budget,
  and separately transfer the unchanged TRAIN cuts. A paired image bootstrap
  with5000 resamples/seed20261014 reports a descriptive95% interval, not a new
  model-selection or deployment rule. Complete TRAIN accuracy stays unavailable
  when any selected fallback is missing; no failed/missing image is omitted.
- `REPOSITORY_SPECIFIED`: the real SigLIP2 run fails after58 TRAIN images at
  sample817 with `Prompt chunk exceeds context`. The existing whitespace
  preference can select a token-invalid prefix after the fitting-prefix scan.
  Fix this in an isolated v3 helper by selecting whitespace boundaries only
  from token-valid character prefixes. Validate complete formatted prompts,
  preserve every raw character, restore the process-local callback on failure,
  and record the explicit `validated_whitespace_boundaries_v1` policy in the
  approximate config. Keep historical v2 files and aggregation/scoring unchanged.
  Changed v3 source/config hashes require new experiment identities.
- `HARDWARE_DEVIATION`: completed host alternatives used RTX3090, FP32/batch2,
  sequential encoders. DataComp CUDA allocated/reserved peaks are
  1,772,411,904/1,818,230,784 bytes; DFN4,091,493,376/4,183,818,240 bytes. Corrected
  SigLIP2 qualification uses explicit CPU FP32/batch2, with null CUDA peaks;
  no output equivalence or full-run VRAM qualification is assumed from CPU.
- `UNRESOLVED`: native-versus-HF numerical conversion equivalence, exact paper
  checkpoints/accuracy, full corrected SigLIP2 routing performance and target
  RTX3070 qualification remain open. DataComp's matched-budget gain is small:
  word precision65.32% versus63.29%, final TN053.25% versus52.66%, paired95%
  difference interval[-1.71,+3.02]pp. Its transferred cut accepts161 TEST images
  and gives51.89% TN0; DFN accepts172 and gives50.64%. No selector promotion or
  near-perfect acceptance claim is supported by these results.

48 focused tests pass. Real-tokenizer checks cover all425 inputs,1700 scene
prompts and39719 unique word/mutation prompts per model: DataComp/DFN's old
successful prompts remain identical, and20 old SigLIP2 overflows are corrected
with full character preservation. All3400 DataComp/DFN and232 saved SigLIP2
scene prompt lists match. Real pinned SigLIP2 CPU inference passes2 TRAIN+2 TEST
images and six corrected prompt chunks for the failed817 sample. These are
smoke tests, not full recognition results; no OOM occurs and the failed-sample
CPU RSS peak is4,335,398,912 bytes.

Evidence, counts, both metric protocols, thresholds, bootstrap interval, source
snapshots and qualification receipts are in
`/tmp/clip-router-alternatives-audit-20261014/`. The remaining cached-weight GPU
command and model details are in
[CLIP_ROUTER_LITERATURE_MODELS.md](CLIP_ROUTER_LITERATURE_MODELS.md).

## ADASpot-GAP-001 — Text+Kernel+Gap controlled extension

User-authorized first AdaSpot implementation/training stage (2026-10-14).
Inspected original paper sections3.2–3.3/Algorithm1; existing qualified
Text+Kernel model, targets, BFS, training/checkpoint and PARSeq interfaces;
and user-provided AdaSpot proposal sections4.1–4.2/6. Proposal PDF SHA256
`7e2a5a663f5ada85c607cb5200a3e17578617e9e387ab682c10e032f25bfb059`.
This is an **approximate reproduction / research extension**, not evidence
resolving missing artifacts or protocols of the original paper.

- `PAPER_SPECIFIED`: ResNet50 attention-U-Net architecture family, as specified
  in original section3.2. Exact existing decoder/weights/preprocessing are still
  inherited approximate choices; no STRICT configuration is promoted.
- `ASSUMPTION`: continue checksum-qualified Text+Kernel epoch61 in two matched
  branches, Gap/control,50 epochs, common LR1e-5/fresh Adam, new Gap LR1e-4,
  encoder/BN frozen. Preserve TRAIN1130/125/983, seed202503,512→256 letterbox
  and original owner/text/kernel/validity/separator targets. Reused validation
  is exploratory. The proposal's full joint recognizers/gate are later work.
- `ASSUMPTION`: Gap is valid non-text within a circular radius3 native pixels
  of at least two distinct augmented/resized owners, with no M/K mutation.
  Unlike the proposal's horizontal same-line wording this handles arbitrary
  orientation without fabricating line IDs. Distant words may lack a corridor;
  false predicted gaps can remove text. Fragmented same-owner support supplies
  only one neighbor. BCE+Dice per valid image, task weights1, ignore zero gradient,
  all-invalid batches skip Adam decay. New65parameter head starts at zero/-2.
- `ASSUMPTION`: deterministic8-neighbor kernel BFS for both branches;
  thresholds M≥0.65/K≥0.5 and barrier G>0.5 before seed CC/growth. Unseeded
  components are preserved explicitly. No GT input to decoder. Padding0/no
  area filter/no top10 remain diagnostic departures from Algorithm1. Retain
  original watershed reports and decode new identical maps with Gap on/off.
- `ASSUMPTION`: native bbox one-to-one IoU0.50 micro word metrics, any-pixel
  fusion/split counts over all983 care words, fully invalid-only exclusion
  after predictions, macro valid-pixel foreground metrics after gap suppression
  with empty/empty=1. Checkpoint ranks word F1, fewer fusions, recall, fgIoU,
  earlier epoch including0. Qualified PARSeq AR/refine1 then scores frozen RGB
  bbox0.50/casefold-exact (punctuation retained), full denominator and per-word
  gains/losses. All are local proxies, not official polygon/lexicon scores.
- `HARDWARE_DEVIATION`: sequential single-GPU segmentation branches, then
  separate PARSeq process, explicit FP32/batch8/TF32off. The available GPU is
  RTX3090 rather than RTX3070/paper hardware. Actual allocated/reserved peaks,
  device/model/shape/batch and OOM records are retained; successful3090
  execution cannot qualify the primary8GB target. No automatic CPU fallback,
  reduced batch, mixed precision or alternate models.
- `UNRESOLVED`: Gap benefit, whether barriers help beyond auxiliary training,
  independent generalization, new combined OCR accuracy, target RTX3070 fit and
  full AdaSpot adaptive-recognition premise. Existing STRICT blockers, exact
  PARSeq-S identity and paper-table metrics remain unresolved.

Alternatives: reuse only existing separator labels (definition differs), shrink
the full mask instead of separate K (confounds crop coverage), frozen-only gap
head (tests feature availability but not adaptation), repeat all historical
training, and combine rectification/recognizers/gate immediately. Chosen design
retains the source backbone and a matched continuation plus same-map Gap-off
ablation to distinguish effects. No claimed target number selects hyperparameters.

Implementation/config/metrics: [ADASpot_GAP.md](ADASpot_GAP.md), new isolated
`models/adaspot_gap.py`, `segmentation/adaspot_gap.py`, `localize/adaspot_gap.py`,
GPU runner/launcher and frozen-crop PARSeq evaluator under `scripts/`,
`configs/approximate/adaspot_gap_v1.json`. 36 new and64 relevant regression
tests pass. Initial pytest collection name collisions were fixed with unique
file names, not disabled tests. Target/GPU qualification and live training
receipts are separate from unit fixtures; launching is not training completion
or a strict/Table3/Table5 reproduction.

### ADASpot-GAP-001 prior-art audit (2026-10-15)

- **Observed local evidence:** pre-existing `make_word_regions` already labels
  background reached by two distinct word-owner dilations and combines it with
  owner-contact separators, at arbitrary orientation. Existing M+separator
  head adds65 parameters and suppresses predicted separators before CC.
  Both historical matched100-epoch trainings are complete, with identical
  TRAIN sample-order hashes in every epoch and checksum-verified best weights.
  In that protocol, control/separator word F1 are52.0704%/48.8665%, fusions
  199/162 and splits77/99. Frozen PARSeq has384/362 correct of983 care words,
 949/1002 proposals and Hmean39.7516%/36.4736%. These are reused TRAIN-holdout
  local diagnostics, not official benchmark or strict results.
- `ASSUMPTION`: retain the authorized M+K+G vs M+K experiment as an incremental
  ablation of combined kernel/gap supervision and gap barriers. New native
  radius/target timing/loss/BFS/warm start differ from M+separator; they do not
  establish a novel Gap principle. The historical negative result does not
  measure the new combination; it is a reference, not its matched control.
- `UNRESOLVED`: whether the exact combination has an external prior and
  whether it helps OCR. Primary precedents establish border learning rather
  than novelty: WordFence2017, Wu/Natarajan ICCV2017 with author GitLab code,
  Repulsive Text Border2020. CM-Net's Gap Mask is an internal contour-to-core
  band, not an inter-word corridor, and its auxiliary head is training-only.
  Direct CVF PDF opens returned403; indexed primary text and available original
  sources were used without installing or executing external implementations.

See [ADASpot_GAP.md](ADASpot_GAP.md#verifica-dei-precedenti-15-ottobre-2026)
for primary links, artifact identities, differences and limitations. This audit
does not resolve any original-paper STRICT ambiguity.

### CLIP-SPECIALIZED-ROUTERS-001 — native Long-CLIP / LLM2CLIP (2026-10-15)

Authorized research extension following the user's clarification “come clipllm”.
All runs use `configs/approximate/clip_specialized_routers_v1.json`; they are
**approximate reproduction**, not a strict reproduction or a Table5 claim.

- `PAPER_SPECIFIED`: local paper §3.5 / Algorithm2 selects recognition using
  MPNet/fuzzy context scores and invokes full-image DeepSolo for uncertainty.
  These specialized CLIP encoders are experimental alternatives to that scorer.
- `REPOSITORY_SPECIFIED`: freeze the prior epoch61 Text+Kernel/PARSeq outputs,
  TRAIN holdout125 / TEST300, square median-RGB canvases,13 score families,
  TRAIN80 calibration /45 selection, seed20261012, and descriptive equal image
  budgets62 /149. Preserve ignored raw predictions as model inputs. New routers
  cannot access GT for inference. DeepSolo outputs are reused unchanged.
- `ASSUMPTION`: “clipllm” means Microsoft's LLM2CLIP; test its released
  OpenAI-L14/336 HF checkpoint with its paired caption-contrastive Llama3-8B
  encoder, alongside official Long-CLIP-B/L. Retrieval gains reported by these
  authors do not establish OCR spelling-verification gains. Select score family
  on the legacy geometric retained-word precision on TRAIN45, as in the prior
  experiment; report the same decisions on the current within-image word metric.
- `REPOSITORY_SPECIFIED`: use pinned native Long-CLIP248-position weights,
  strict builder, BPE/ftfy and preprocessing. LLM2CLIP retains its complete
  trained bidirectional Llama encoder and vision/text adapter, including four
  residual blocks and4096→1280 projection. Native LLM2Vec no-instruction role
  framing and mean pooling of caption plus EOT exclude the role header/padding.
- `ASSUMPTION`: use the native LlamaEncoderModel shipped inside the pinned
  Microsoft checkpoint with a separately pinned native mask helper, rather than
  its mutable cross-repository auto_map now requiring an incompatible runtime.
  A thin inference bridge avoids installing PEFT/LLM2Vec. Published framing,
  tokenizer IDs, attention masks and FP32/BF16 pooling were compared directly
  against five AST-extracted published methods on eight varied strings; all
  matched. A native tiny-model future-token test confirmed bidirectionality.
  These structural checks do not replace real-weight qualification.
- `ASSUMPTION`: the existing legacy runtime lacks removed `pkg_resources`.
  Replace exactly its single `from pkg_resources import packaging` line with
  `import packaging.version` in memory after checking the source SHA. The version
  guard and all model/tokenizer/preprocessing functions remain identical. Fail
  if the expected import differs; do not edit the downloaded source or install
  dependencies. A regression test checks the narrow transformation.
- `HARDWARE_DEVIATION`: sequential isolated native model processes,4 CPU
  threads,batch2,TF32off,deterministic algorithms. Long-CLIP uses existing
  Py3.10/Torch1.13 at FP32; LLM2CLIP uses Py3.11/Torch2.4 at BF16 with eager
  attention and cache disabled. Convert pooled vectors to FP32 before the
  trained adapter as published; use explicit BF16 autocast instead of the
  example's default FP16 autocast, and final FP32 cosine features. Actual GPU
  is RTX3090/24GB, not the primary RTX3070/8GB. Record allocated/reserved memory,
  model set,shape,batch,precision and errors; no automatic offload/quantization.
- `ASSUMPTION`: preserve all transcription characters until native tokenization;
  split overlong formatted prompts only at validated whitespace boundaries.
  Versioned stdio/NPY bridge checks RGB bytes,SHA,ordering,count,dtype and unit
  vectors. Primary word metric is per-image casefold-exact multiset matching,
  punctuation retained, no bbox IoU; ignore handling comes from the checksum-
  frozen word-branch artifact. Precision=TP/(TP+FP), user TN0=TP/(TP+FP+FN).
  Empty predictions keep FN; missing fallback samples make combined metrics
  unavailable rather than shrinking the denominator.
- `UNRESOLVED`: the release warns HF weights may differ from the paper's native
  PyTorch numbers. Strict equivalence and RTX3070 suitability are unqualified.
  The initial frozen source lacked62 TRAIN DeepSolo outputs; the separate
  completion below now resolves that availability. Original paper recognition-
  accuracy definition and all existing strict blockers remain unresolved.

Initial runtime qualification recorded two explicit failures: legacy CUDA
allocator statistics before CUDA initialization, then the removed packaging
re-export. Both were corrected with regression tests; no error or sample was
silently ignored. Real-model results and any further qualification failures are
recorded in [CLIP_SPECIALIZED_ROUTERS.md](CLIP_SPECIALIZED_ROUTERS.md).

Alternatives considered: interpolate ordinary CLIP positions (not Long-CLIP),
use an ordinary causal Llama or unpaired CLIP text encoder (not LLM2CLIP), install
or upgrade packages, or silently quantize/offload. None was used. CLIP4STR and
CLIPTER require their trained OCR architectures; they are distinct potential
experiments and are not represented as tested by these routers.

Completion extension: `ASSUMPTION` — run the62 previously uncached TRAIN
DeepSolo images with the existing qualified ResNet50/score0.4 protocol to allow
complete comparison of any frozen gate. Write supplemental branches outside the
original experiment; no GT, PARSeq, existing DeepSolo or TEST output is changed,
and no family/reference/threshold is refitted. `HARDWARE_DEVIATION` — sequential
existing Python3.8 FP32 CUDA workers on RTX3090 after CLIP workers exit, retaining
per-image memory/failure metadata. The executor's source, artifact and population
preflight passed, followed by all62 real inferences with strict checkpoint
compatibility. All125 TRAIN and300 TEST branches are now available. The
specialized audit preserves original missing-branch results and separately
replays completed TRAIN branches, including previous CLIP and MPNet gates.

RGB boundary correction: `REPOSITORY_SPECIFIED` — the existing bridge supplies
RGB HWC arrays, so the HF processor must receive that known format explicitly.
The first full LLM2CLIP run failed on mixed3×3/2×2 crops at TEST1543 after287
TEST images. Inferring channel order can also silently permute3×3 RGB pixels.
Do not present the failed partial run as complete or discard its small crops.
The corrected plan reruns all425 images; no old partial features are inherited.
Real-HF-processor unit tests and the actual failing input verify channel colors,
batch/single equivalence and equality of the five unaffected input batches.
The corrected TRAIN rerun changed image1189's features and left124/125 feature
sets bit-identical. Its TRAIN family/cut remain unchanged and were frozen before
any corrected TEST output. This validates rerunning the full experiment rather
than claiming that every formerly successful partial feature was unaffected.

Completed evidence: all three models produced425 final features each, with no
failed sample omitted. The independent audit verified1275 payloads and replayed
312 original word/geometric policy points, plus completed TRAIN comparisons and
the frozen MPNet baseline.102 relevant tests pass. Equal-budget whole-system
TN0 TRAIN/TEST: original B16=56.22/52.66%, Long-B=51.04/52.08%,
Long-L=54.18/52.47%, LLM2CLIP=54.03/51.86%. No tested specialized encoder
improves that baseline. The conditional exploratory bootstrap intervals include
zero; no significance claim is made. Results, native memory measurements,
threshold curves, failed-run history and persistent input snapshots are linked
from [CLIP_SPECIALIZED_ROUTERS.md](CLIP_SPECIALIZED_ROUTERS.md). This resolves
the measured comparison, not the original-paper or primary8GB STRICT blockers.

## YOLO-PAIRED-ROUTERS-001 — detector-controlled routing comparison

Active user authorization: compare the existing CLIP router with section3.5
on the working YOLO backbone and investigate whether it exceeds the paper.
Authority inspected: local2503.15639v2, section3.3/3.5, Algorithm2 and Table5;
existing YOLO direct-crop, admission/NMS, fixed CLIP, MPNet and word evaluators.

- `PAPER_SPECIFIED`: MPNet semantic T1/T3 selection, selected-candidate
  confidence0.6S+0.4L, tau0.8 and whole-image DeepSolo fallback; TotalText
  Table5 reports78.4% with ResNet50 and79.8% with ViTAEv2-S.
- `REPOSITORY_SPECIFIED`: existing YOLOv8s TextOCR640, genuine qualified
  PARSeq and joint0.60/same-reading NMS0.5. TEST300 outputs are frozen;
  image-only BLIP2 figure4_single captions, whole-image T1 and all425
  qualified DeepSolo branches already exist and must retain their provenance.
- `ASSUMPTION`: the controlled experiment replaces localization only, using
  identical final YOLO/PARSeq outputs, images and DeepSolo for both routers.
  T3 joins all final readings; T1 emits one sequence and must pass its own
  threshold, per the user's earlier clarification. No GT enters routing.
- `ASSUMPTION`: fixed original B16 native_word_advantage, with fresh empirical
  CDFs fit only to YOLO TRAIN80. Freeze six GT-blind coverage cuts before
  scoring TEST; report transferred cuts and descriptive equal-image budgets
  separately. Native confidence and YOLO/OCR joint confidence are controls.
  No new family selection. YOLO0.60 was previously studied on TEST, so the
  entire extension remains exploratory even though new router fits are TRAIN-only.
- `ASSUMPTION`: common primary metric is within-image exact-casefold multiset
  matching, punctuation retained, no bbox IoU, keeping every care GT word.
  Ignore uses the earlier native256 wholly-invalid support rule: exclusive
  rectangle for YOLO, original polygon for cached DeepSolo, unlocalized T1
  always care. It is distinct from official polygon-overlap ignore and the
  separate YOLO don't-care diagnostic; their precision/F1 must not be mixed.
  Report P, recall, F1, TN0, emitted words, safe and complete images. An
  oracle maximizing microF1/TN0 from GT is diagnostic only, including fixed
  budgets; paired-image bootstrap intervals condition on fixed fitted gates.
- `HARDWARE_DEVIATION`: RTX3090; existing isolated environment, deterministic
  TF32-disabled sequential FP32 YOLO GPU batch1, PARSeq CPU batch32 (same
  cached TEST precision/runtime), CLIP GPU batch16 and MPNet CPU. Record
  input shapes/allocator peaks and propagate failures/OOM. Reused captions
  retain their original FP16/offload provenance; no new DeepSolo inference.
- `UNRESOLVED`: the authors' U-Net-only baseline and exact Table5 accuracy
  definition are absent. Their absence does not block a paired local router
  comparison, but blocks an equivalent claim of exceeding their Table5 score.
  Checkpoint, fuzzy preprocessing, caption quality and multicrop ambiguities
  remain. TRAIN125 is a held-out subset for our segmentation; upstream
  DeepSolo was fine-tuned on TotalTextTRAIN, so it is diagnostic only.

Implementation/configuration: `scripts/run_yolo_router_comparison.py`,
`recognition/yolo_router_comparison.py` and isolated
`configs/approximate/yolo_router_comparison*_v1.json`. Generated results stay
outside Git. Pending measurement is not a completed scientific result.

Completed evidence for YOLO-PAIRED-ROUTERS-001: all125 new TRAIN YOLO/PARSeq
samples and425 CLIP/MPNet image features completed, with no omitted sample.
67 focused tests pass; genuine smoke and full independent count/selection/
RGB audit pass. Equal150-image TEST budgets give CLIP F1=73.4518%,
TN0=58.0425%, versus approximate paper router F1=73.0516%, TN0=57.5444%.
The paired conditional TN0 interval[-1.6748,+2.5913]pp includes zero; no
superiority claim. YOLO-only F1=68.8291%, DeepSolo-only=75.7360%; exact GT
oracle=79.9155% (diagnostic only). Original tau0.8 accepts zero images.
Actual half-coverage TRAIN cuts accept189 CLIP /197 paper TEST images,
distinct from the descriptive equal150 budgets. Source/model/checkpoint/
configuration hashes, allocator peaks, scientific limits, prior corrected
smoke/collection failures and output paths are recorded in
`docs/YOLO_PAIRED_ROUTER_COMPARISON.md`. The Table5 metric/checkpoint
ambiguities remain UNRESOLVED; none of these numbers is a strict paper result.

## TABLE5-SOURCE-METRIC-AUDIT-001 — recognition accuracy versus source F1

User-authorized source/protocol audit,2026-10-16, with no changes to scientific
evaluation code, thresholds, checkpoints or historical results. Inspected local
2503.15639v2 section4.1/Table5/references and original Mask TextSpotter2018,
Mask TextSpotterv3, MANGO, SPTS, TESTR and DeepSolo papers; checked author
code at pinned commits and visually inspected original table pages.

- `PAPER_SPECIFIED`: our target Table5 is labeled recognition accuracy and
  gives no corresponding recognition formula. Its pixel F1 formula in4.1
  does not define recognition F1. Preserve its literal labels and numbers.
- `REPOSITORY_SPECIFIED`: official DeepSolo and SPTS evaluators aggregate
  global matched-word TP/detection/GT counters and compute harmonic mean,
  with different geometry/care handling. DeepSolo's TotalText table explicitly
  uses F1; SPTS uses point matching, MANGO states an IoU>0.1 adaptation,
  and TESTR uses polygons. None is our multiset/native256 diagnostic.
- `ASSUMPTION`: exact correspondences of all four target TotalText baseline
  values with source end-to-end F1/Hmean strongly suggest F1 was intended by
  target recognition accuracy. This inference does not establish the metric
  used to obtain the target's own78.4/79.8 results and must not silently enter
  STRICT configuration. Source71.2 is Mask TextSpotterv3, whereas target ref42
  identifies the2018 model, whose None score is52.9. Only5/12 baseline cells
  exactly match the cited method/version; one additional cell matches v3.
- `UNRESOLVED`: all four ICDAR15 baseline cells differ from the source model
  rows examined; target TESTR IC13 is not reported in the cited TESTR paper;
  original author reruns/protocols could explain differences but are not
  documented. Source Table5 R50 TextOCR82.5 matches78.4+4.1; target ViTAE
  implied83.7 differs from source83.6. Exact checkpoint, lexicons, GT/care,
  matching, decoding, output localization and confidence selection remain
  unresolved. IC13 target denominator229 versus Table1 TEST233 also remains.
- `ASSUMPTION`: continue reporting both local F1 and TN0/Jaccard, distinctly.
  Algebraically J=F1/(2-F1) only with identical counts: conditional F178.4%
  implies J64.47%, and79.8% implies66.39%. This does not convert our local
  geometric/normalization/ignore rules into an official benchmark protocol.

Detailed source tables, pinned paper versions/commits, formulas and limitations:
`docs/PAPER_TABLE5_SOURCE_AUDIT.md`. External machine-readable evidence,
download hashes, code snapshots and table-page renders are under
`/extra1/bmartelli/context-str-artifacts/paper-table5-source-audit-20261016/`.
No strict ambiguity is resolved merely by this inferred metric interpretation.

## ROUTER-ACCURACY-TARGETS-001 — conditional accuracy and U-Net requirements

User-authorized cached numerical analysis,2026-10-16: compare our versions'
accuracy to the papers and quantify what U-Net performance would be required.
Local paper Table5, source-metric audit, frozen router implementations and
all425 completed fallback branches inspected; no new inference or gate fitting.

- `ASSUMPTION`: treat displayed78.4/79.8 as nominal target F1 based on the
  source audit, and show indicative numerical gaps only. Local multiset/care
  rules differ from official polygon/point evaluators, so no benchmark claim.
- `REPOSITORY_SPECIFIED`: reuse original149-image U-Net B16 and descriptive
  encoder comparisons,150-image YOLO comparisons, complete TEST300 /2204 GT
  and the previously tested exact microF1/TN0 branch oracle optimizer.
- `ASSUMPTION`: a GT oracle is diagnostic only, not a trained router. Frozen
  U-Net/PARSeq versus DeepSolo yields maximum78.0052% without a budget limit,
  and75.0919% with149 acceptances. GT-free routing cannot exceed that frozen
  two-branch ceiling. YOLO's prior oracle79.9155% is a separate diagnostic.
- `ASSUMPTION`: with original149 accepted images,978 accepted outputs/977 GT,
  619 accepted TP and fixed DeepSolo855 TP/1091 outputs, total F1 is
  2*(855+TP_accepted)/(2204+2069). At unchanged output count,821 accepted TP
  strictly exceed nominal78.4%, and850 exceed79.8%; these are202/231 more
  correct words. Accepted-branch F1 would need83.99/86.96%, respectively.
  This is not a global U-Net requirement; changes to gates/counts invalidate it.
- `UNRESOLVED`: no unique mapping from pixel segmentation F1 or standalone
  word F1 to hybrid performance exists without the joint error distribution,
  localization, recognition and routing behavior. Attribute no OCR error to
  U-Net alone. Exact official target protocol and unrounded author values remain
  unresolved. Latest Kernel-only/padding holdout results are not TEST scores.

Details: `docs/ROUTER_ACCURACY_TARGETS.md`. Input hashes, all version gaps,
oracle choices and arithmetic checks:
`/extra1/bmartelli/context-str-artifacts/router-accuracy-gap-analysis-20261016/`.
No strict configurations, checkpoints, source evaluators or old results changed.

## KERNEL-ONLY-PADDING-001 — single-output extraction and padding comparison

The user requested Kernel-only predictions and corrected the proposed Gap sweep
to a padding sweep. Paper sections3.2–3.3 / Algorithm1 and existing native256
target/model/padding/evaluation components were inspected before implementation.

- `PAPER_SPECIFIED`: component boxes with clamped padding (section3.3).
- `ASSUMPTION`: extract the Kernel output from the completed matched M+K
  control best checkpoint, preserving all shared tensors and slicing only
  channel1 of its last1x1 layer. Source training remains joint M/K. Verify CUDA
  output probabilities and exact native threshold masks for all125 samples.
- `ASSUMPTION`: use only K>=0.5 on the RGB canvas and8-connected components,
  with no Text gate, Gap, growth, area filter or proposal cap. Compare with the
  same source's M+K labels at fixed padding0/2/4/6/8/12 native256 pixels.
- `ASSUMPTION`: report padded-box localization and exact casefold OCR counts
  separately from unpadded detection-box localization, using maximum one-to-one
  RGB bbox IoU>=0.5 and all983 care GT. Ignore only wholly invalid instances
  after GT-blind decoding/cropping/OCR. Reproduce unchanged M+K pad0 OCR exactly.
- `HARDWARE_DEVIATION`: single RTX3090, FP32, TF32off, batch8 and sequential
  segmentation/PARSeq; measure peaks and propagate OOM without CPU fallback.
- `UNRESOLVED`: exclusive-K training benefit, independent validation and
  generalization. This extraction does not resolve these questions or establish
  official polygon evaluation / Table3 or Table5 reproduction.

Evidence, rationale and alternatives are recorded in
`docs/KERNEL_ONLY_PADDING.md` and
`configs/approximate/kernel_only_padding_v1.json`. Dedicated K-only training
is deferred because the user authorized it conditionally if existing Kernel
predictions could not be used. No training or strict configuration is altered.

Completed KERNEL-ONLY-PADDING-001 evidence (2026-10-16): single-output CUDA
probabilities exactly reproduce all125 original Kernel maps (max difference0).
17 tests pass; independent Hungarian matching confirms all1,500 image/arm
scores and983-word denominators. The unchanged M+K padding0 reference reproduces
648 localized /524 exact-OCR words. Kernel padding2 gives622 localized /
483 exact-OCR words, with976 proposals. Single-output inference and PARSeq
both complete on RTX3090 without OOM. No new training was performed; dedicated
K-only training and independent generalization remain UNRESOLVED. Results and
provenance are recorded in `docs/KERNEL_ONLY_PADDING.md` and the external run
`/extra1/bmartelli/context-str-runs/kernel-only-padding-v1-20261016/`.

## KERNEL-TRAIN-001 — explicit Kernel-only supervised fine-tuning

The user explicitly requested starting Kernel-only training after the extracted
head study. This supersedes the conditional deferral in KERNEL-ONLY-PADDING-001.
Paper sections3.2–3.3 and existing implementation/checkpoint protocols were read.

- `ASSUMPTION`: initialize from the qualified extracted single-output model,
  then optimize exclusively masked BCE+Dice(K) for50 additional epochs with
  Adam1e-5, frozen encoder/BN and trainable decoder/head. Source pretraining
  was joint M/K; do not describe this as training from scratch.
- `REPOSITORY_SPECIFIED`: reuse the checksum-qualified owner-to-kernel target
  construction after paired augmentation/native resize, same1130/125 TRAIN
  split, RGB512/native256, seed and preprocessing.
- `ASSUMPTION`: select by per-image macro Kernel IoU then F1 and earliest,
  including epoch0; both-empty scores1. No OCR, padding or TEST enters selection.
- `ASSUMPTION`: after all50 epochs compare fixed0/2/4/6/8/12 native padding
  using the existing qualified PARSeq and983-word one-to-one localization/OCR
  protocol; preserve all denominators and per-image outputs.
- `HARDWARE_DEVIATION`: single RTX3090, FP32 batch8, TF32off, deterministic
  algorithms and sequential segmentation/PARSeq. Record peaks
  and propagate errors/OOM without CPU fallback.
- `UNRESOLVED`: from-scratch K-only benefit, independent validation, multiple
  seeds and causal isolation from further optimization (no matched new M+K
  continuation). This remains exploratory approximate reproduction.

Protocol, alternatives and launch/result qualification:
`docs/KERNEL_ONLY_TRAINING.md`,
`configs/approximate/kernel_only_training_v1.json`.

Completed KERNEL-TRAIN-001: all50 exclusively Kernel-supervised epochs and
7,100 CUDA steps completed on RTX3090 without OOM, followed by the full fixed
padding OCR comparison. Best epoch4 increases Kernel macro IoU63.5231% to
63.9072%, but the descriptive best pad2 yields623 localized /478 exact-OCR
words versus622/483 for the extracted source. M+K pad0 remains648/524.
Every983-word denominator and all750 per-image/padding counts were verified
independently using NumPy IoU + SciPy Hungarian matching. No overall OCR
improvement or strict reproduction is established. Completion provenance is in
`provenance/kernel_only_training_v1_completion.json`; scientific generalization
and causal attribution remain UNRESOLVED.

## KERNEL-FRESH-001 — user-requested new U-Net on GT Kernel masks

The user explicitly deleted/rejected the prior fine-tuned Kernel and requested
a new U-Net trained on GT Kernel on GPU. The old completed run and checkpoints
were removed; historical lightweight metrics and deletion authorization remain
in `provenance/kernel_only_training_v1_deletion.json`. This supersedes the
initialization choice in KERNEL-TRAIN-001, not its historical measurements.

- `PAPER_SPECIFIED`: retain the attention U-Net / pretrained ResNet50 family
  (section3.2); strict decoder details remain unresolved elsewhere.
- `ASSUMPTION`: new random bottleneck/decoder/head with only local pinned
  ImageNet encoder initialization. No Text/Kernel/Gap checkpoint initialization.
  Encoder/BN frozen,100 full epochs, fresh Adam1e-4, only masked BCE+Dice(K).
  This is not a completely random encoder or a continuation of the deleted model.
- `REPOSITORY_SPECIFIED`: reuse qualified native256 Kernel GT derived from
  word owners after paired augmentation, RGB512 letterbox,1130/125 TRAIN split.
- `ASSUMPTION`: same predeclared Kernel macro IoU/F1/earliest checkpoint rule,
  including epoch0; six fixed-padding qualified PARSeq evaluations after the
  full budget, preserving every983 care GT and local bbox one-to-one semantics.
- `HARDWARE_DEVIATION`: RTX3090, FP32 batch8, TF32off, deterministic algorithms
  and sequential segmentation/PARSeq with measured peaks and propagated OOM.
- `UNRESOLVED`: independent validation, multiple seeds and strict reproduction;
  no result or superiority claim before complete training and evaluation.

See `docs/KERNEL_ONLY_FRESH.md` and
`configs/approximate/kernel_only_fresh_v1.json`. The reference extracted-K/M+K
run supplies comparison metrics/cache only, never segmentation weights.

## DEEPSOLO-TEXTOCR-IC13-001 — paired variants and recovered IC13 recipe

Active request (2026-10-16): evaluate official R50+TextOCR and ViTAEv2-S+TextOCR
on Total-Text, then an IC13-adapted model with None/Generic kept separate.
This work is **approximate reproduction**, not an identification of the
context-driven paper's selected DeepSolo checkpoint.

- `PAPER_SPECIFIED`: DeepSolo supplemental Appendix A specifies IC13
  fine-tuning1000 iterations, evaluation resize1000/max1824, and Generic E2E
  F1=90.1. The context-driven paper's Algorithm2 falls back on the full image.
- `REPOSITORY_SPECIFIED`: pinned DeepSolo zoo lists Total-Text R50+TextOCR
  E2E None82.5 and ViTAEv2-S+TextOCR83.6. Their native configurations differ
  in shortest test resize (1000 versus1024), with ViTAE divisor32 padding.
  Upstream author `ymy-k`, issue15, supplies IC13 threshold0.25, batch8,
  LR1e-5/backbone1e-6, warmup0,1000 iterations, no LR decay, rotation disabled,
  37-character training annotations, and shared pretrained initialization.
  The author says IC13 TEST files were not uploaded and refers to SPTS.
- `ASSUMPTION`: compare released Total-Text variants using frozen TRAIN125
  and TEST300, original local word/geometry evaluators and ignore treatment,
  inclusive threshold0.4, no lexicon, no router refitting and no TEST selection.
  Retain official polygon F1 as a separate reference. TRAIN is diagnostic:
  upstream variants have seen Total-Text TRAIN during fine-tuning.
- `HARDWARE_DEVIATION`: use the existing isolated host legacy stack,
  FP32 batch1 with sequential processes. New-runner shards bind plan/image/
  checkpoint/source hashes, preserve strict-load validation, actual allocated/
  reserved VRAM, model input shape and failures. No new runtime dependencies
  were installed. The native worker's existing cuDNN/RNG policy is preserved.
- `UNRESOLVED`: both official TextOCR weight downloads currently return401
  or redirect to login; checkpoint load, inference accuracy and new GPU peaks
  have therefore not been qualified. CPU construction alone passed for both
  architectures. No official IC13-finetuned weights are listed in the zoo;
  shared initialization, upstream-format training targets, Generic vocabulary/
  pair list and correction/evaluation protocol remain missing. Do not silently
  use a Total-Text/IC15 checkpoint as an IC13-adapted model, repair targets or
  substitute a community mirror. An IC15-only mirror was rejected as unsuitable.

Baseline CPU replay exactly recovers TEST300 word TP1595/FP413/FN609, local
F1=75.7359924%, TN0 accuracy=60.9476500%. This is a cache audit, not new
checkpoint inference. The existing revision-rejection test has an error-message
regex mismatch; it is reported, not suppressed or fixed in the frozen worker.
See `docs/DEEPSOLO_TEXTOCR_IC13_COMPARISON.md`, the two new approximate
configurations, and external evidence in
`/extra1/bmartelli/context-str-artifacts/deepsolo-textocr-ic13-preflight-20261016/`.

## REPORTING-INDEX-001 — Read-only evidence navigation and model comparisons

Classification: **ASSUMPTION** (reporting and navigation choice, not a paper
component or a new experiment). User-authorized repository refactor, 2026-10-17.

Evidence inspected: local paper sections3.3/3.5/4.1/4.2 and Tables3–5;
repository specification, author clarifications, STRICT/runtime records;
agent experiment review, ICDAR13 routing audit, Table5 source audit, provenance
receipts and saved artifact audit/review summaries. Detailed source inventory,
content hashes and Markdown section locations are in `docs/AUDIT_INDEX.json`.
The inventory states its root/depth/cache exclusions; it is not an assertion that
all files on external disks were independently requalified.

Decision: preserve historical scientific documents and results byte-for-byte
apart from this appended decision. Add a bounded document catalogue, literal
search, exact section reads and a standard-library reporting interface. Explicit
registry entries select JSON pointers from existing reports, pin SHA256, retain
source qualification labels and check known populations. No mutable latest-run
selection or cross-namespace config fallback is introduced. Missing/changed
sources, missing registered rows and inconsistent counts/rates cause failure.

Derived reporting uses micro P=TP/Npred, R=TP/NGT,
F1=2TP/(Npred+NGT), and requested TN0=TP/(TP+FP+FN), solely for the stated
one-to-one protocol. Undefined zero-denominator rates are null/UNRESOLVED. This
does not redefine the original evaluators, their handling of empty predictions,
ignore, coordinate frames, lexical preprocessing or official matching thresholds.
Source-row pointers and hashes preserve access to every original intermediate
and metadata field; exported projections are not replacement evidence.

Comparison boundaries explicitly distinguish native geometry from text+box OCR,
word multisets from spatial matching, TRAIN from TEST, repeated exploratory TEST
from untouched validation, top10 from uncapped extraction, native from RGB
padding, descriptive budgets from transferred thresholds, and GT diagnostics
from inference methods. In particular, the existing native128 review uses
bbox-area filtering and all-prediction counting; it is labeled as that historical
repository variant and is not silently promoted to paper Algorithm1.

Alternatives considered: rewriting/splitting the accumulated ambiguity record,
moving historical audits, merging all metrics into a leaderboard, automatically
collecting every run/cache, and rerunning missing results. These were not adopted:
original paths/history remain stable, protocols remain visible, searches avoid
sample caches and the report tool never starts model execution. No checkpoint,
model, configuration namespace, scientific evaluator or environment is changed.

Limits: registry source hashes validate the saved bytes, not checkpoint integrity
or a full replay of per-sample evaluation. Index statuses are historical reported
metadata, not newly issued qualifications. Where summaries lack image counts,
counts are not guessed. Registered groups are selected existing summaries, not
all research artifacts. VRAM/timing fields retain their original TRAIN/feature
export scope and do not create a common end-to-end performance protocol.
E-001, D-001/D-002, DS-002 and all other STRICT blockers retain their status.

## ADASpot-PHASE0-001 — Paired crop, oracle-separator and decoding diagnostics

- Authorization: user asked to follow the proposed staged AdaSpot diagnostic
  roadmap before committing to additional segmentation training or a gate.
- Evidence inspected: local original paper sections3.2–3.3; attached AdaSpot
  proposal sections4.1–4.8, hypotheses and Phase0; existing frozen-control exports,
  qualified PARSeq adapter/runtime and PCA/Bézier fitter/renderer.
- `REPOSITORY_SPECIFIED`: completed Text+Kernel matched control, qualified local
  PARSeq artifact, TRAIN1130/125 holdout/all983 care words and matching protocol.
- `ASSUMPTION`: original GT axis boxes and target-G substitution on unchanged
  predicted M/K isolate geometry/current-separator-target potential. They are
  explicit oracle diagnostics, never inference candidates. Axis GT is not an
  optimal-crop guarantee; oracle G bounds only the present radius3 target idea.
- `ASSUMPTION`: reuse existing predicted-instance RGB PCA/Bézier crops with fixed
  evaluation boxes; axis fallback on declared geometric failures. This is a
  proxy, not proposal feature-space Mask-Align.
- `ASSUMPTION`: NAR0/NAR1/AR1 change upstream decoding flags on the same strict
  PARSeq weights; weights/production defaults remain unchanged. They are not
  trained one/four-layer AdaSpot heads. Confidence routing uses minimum character
  probability with empty/tie rules and32 seeded equal-budget random controls;
  GT-priority selection is diagnostic, not an exact matching oracle bound.
- `HARDWARE_DEVIATION`: sequential FP32/batch8 single actual GPU, synchronized
  warm wall timing includes transform/forward/CPU decode, with allocated/reserved
  memory and OOM recording. Modes use identical unique crops and batch membership;
  these include diagnostic crops. RTX3090 does not qualify RTX3070.
- `UNRESOLVED`: actual shared-feature head gap, learned Mask-Align, gate calibration,
  full joint training, independent transfer and official benchmark benefit.
  No strict reproduction, central AdaSpot hypothesis or novelty claim follows.
- Alternatives: more unchanged kernel-only epochs (unsupported by current curves);
  treating RGB warp as feature Mask-Align or PARSeq modes as trained AdaSpot heads
  (rejected as scientifically misleading). No simultaneous major model rewrites.
- Reproducibility: isolated APPROXIMATE config, source/input hashes, fixed crops,
  original per-word denominator, recoveries/regressions, baseline replay, no TEST
  selection or threshold tuning. See `docs/ADASpot_PHASE0.md` and
  `scripts/diagnose_adaspot_phase0.py`.

## ADASpot-GAP-PROBE-001 — Balanced Gap probe on frozen Text/Kernel features

- Authorization: follow the diagnostic roadmap; Phase0 GT-Gap improved exact OCR
  from524 to600/983 (78 recoveries,2 regressions) and fused-word involvement
  from202 to34. These are oracle diagnostics, not learned gains. Independent
  Hungarian audit passed all1875 image/arm cases and192 random routing cases.
- `REPOSITORY_SPECIFIED`: same qualified Text+Kernel matched control, targets,
  native256/512RGB and TRAIN1130/125 split; source model stays in eval with all
  parameters and BN frozen. Existing M/K probabilities supply validation BFS.
- `ASSUMPTION`: smallest separate intervention is a new65parameter1x1 Gap head
  on frozen64channel features, trained as a linear probe on cached FP32 points.
  No augmentation. Retain all training positive Gap pixels; sample without
  replacement from valid non-Gap predicted-text and remaining valid background,
  each capped at max(64,image positive count). Equal total stratum weight via
  inverse sampled-class-count BCE. Targets and fixed G>.5 threshold unchanged.
  Sampling/balancing changes effective priors; probabilities may be uncalibrated
  and barriers may overcut. This is an explicit risk, not a guaranteed remedy.
- `ASSUMPTION`: predeclared20epochs Adam1e-3, zero weight decay, pixel batch16384,
  seed202503; native wordF1 then fewer fused words then earliest epoch selects
  including epoch0. No OCR or parameter sweep selects the head. A failed probe
  does not exclude spatial/new-feature or alternative-target approaches.
- `HARDWARE_DEVIATION`: single CUDA feature pass, sampled train/full validation
  FP32 caches on CPU/disk, CUDA-only head optimization/inference, sequential
  qualified PARSeq after source backbone unloading; real GPU and memory/OOM
  recorded. Linear training and1x1-convolution inference equivalence is tested.
- `UNRESOLVED`: whether the frozen features suffice, whether later joint training
  helps, independent validation and actual shared-feature recognition heads.
- Reproducibility: isolated approximate config, strict source checkpoint loading,
  hashes for source/input/cache, disjoint1130/125 split, deterministic sampling,
  explicit epoch0 geometry replay, full20epochs, per-word OCR gains/losses.
  See `scripts/train_adaspot_gap_probe.py` and `docs/ADASpot_PHASE0.md`.

## WORD-EVIDENCE-001 — Authorized Text/Kernel oracle, glyph association and crop completeness

User authorizes all three tests on2026-10-17. Existing paper3.2/3.3/4.1,
Text+Kernel training/partition, paired glyph targets, V1/V2 results and geometry/
PARSeq evaluators inspected. See docs/WORD_EVIDENCE_SUITE.md and isolated
configs/approximate/word_evidence_suite_v1.json for exact choices and limits.

- PAPER_SPECIFIED: attention-U-Net family, connected localization and original
  RGB recognition crops; extra learned modules are not paper components.
- ASSUMPTION: reuse qualified frozen control and isolated glyph states; fit
  small association and crop modules on canonical TRAIN1130 predictions,
  preserving held-out125/983, source weights, data and fixed postprocessing.
- ASSUMPTION: GT Text/Kernel cross and glyph-GT compatibility are diagnostic
  oracle rows only; a perfect target is not an inference result or OCR bound.
- ASSUMPTION: predicted fragment/kernel geometry and probability features,
  majority word-owner pair supervision, balanced BCE,16-hidden MLP, fixed.8
  conservative cut; preserve existing IDs/pixels and count conflicts/orphans.
- ASSUMPTION: bounded25percent crop-side corrections trained with SmoothL1,
  coverage/excess/foreign-bbox proxy losses, same source ignore identities.
  Bbox overlap is explicitly not polygon contamination or official evaluation.
- ASSUMPTION:50 full-batch CPU updates; best measured TRAIN-loss state selected
  without holdout tuning. New modules do not retrain the whole shared decoder.
- HARDWARE_DEVIATION: RTX3090 FP32 batch4 export, CPU fitted heads and CPU PARSeq
  batch16; sequential internal models. Preserve unrelated ongoing GT-V3 training,
  record co-resident GPU processes, check free memory, retain own allocator peaks
  and propagate OOM/errors without switching precision/model/data.
- UNRESOLVED: detection/OCR gain, losses/FP tradeoff, calibration and independent
  generalization. All outcomes remain approximate reproduction, not paper tables.

Alternatives include replacing full-word Text with disconnected glyph masks,
repeating the already negative auxiliary glyph experiment, and full-network
fine-tuning before locating the failure. The three isolated tests instead assess
existing evidence and train the smallest new grouping/crop modules. Failures,
empty OCR and all expected GT words remain auditable; no original output is changed.

## DEEPSOLO-VITAE-GPU-20261019 — real checkpoint qualification and full run

The active user request authorizes starting GPU tests with the uploaded weights.
The complete ViTAEv2-S Total-Text TextOCR file is135,515,957 bytes, SHA256
`54d3a454ecc704ef3168ebf9f23fc8390a3b2637d0ec83d5650e8df45e2aca4a`.
On2026-10-19, the existing isolated legacy worker passed real CUDA inference
on the qualified720x1280 engineering image: zero missing/unexpected/wrong-shape
keys,33,748,854 parameters, output `deepsolo test 2026`. FP32 batch1 input
CHW3x1024x1824 allocated peak1,930,212,352 bytes and reserved peak3,779,067,904
bytes on RTX3090. This is a smoke measurement, not full-dataset memory or accuracy.

- `PAPER_SPECIFIED`: full-image DeepSolo fallback in Algorithm2.
- `REPOSITORY_SPECIFIED`: pinned upstream ViTAE native config, shortest1024/
  max1892 and divisor32 padding, inclusive threshold0.4, no lexicon; existing
  legacy worker and frozen local word/geometry evaluation protocols.
- `ASSUMPTION`: identify user-supplied complete bytes with the named official
  zoo candidate using origin/filename, SHA256, archive integrity and exact model
  compatibility; there is no author-published reference checksum. Compare frozen
  TRAIN125/983care and TEST300/2204care separately, without refitting routers.
- `HARDWARE_DEVIATION`: existing Python3.8 host subprocess on RTX3090, FP32
  batch1, one model at a time, qualified cuDNN/RNG settings unchanged; preserve
  per-image actual allocator peaks, input shapes and structured errors. No OOM
  occurred in the smoke and no error-triggered substitution is permitted.
- `UNRESOLVED`: exact context-driven-paper checkpoint/metric mapping and strict
  reproduction; complete R50+TextOCR bytes (current file is truncated); IC13
  training targets and Generic protocol. Shared R50 pretraining is not IC13
  adaptation. No IC13 result or new full-dataset accuracy is claimed at launch.

Full input qualification passed for all425 images. GPU evaluation was started
with a new approximate configuration, leaving historical configurations, raw
baseline outputs and source implementations intact. See
`docs/DEEPSOLO_VITAE_GPU_20261019.md` and
`configs/approximate/deepsolo_textocr_gpu_20261019.json`. Launch, smoke, raw
responses and terminal log are stored outside Git under
`/extra1/bmartelli/context-str-artifacts/deepsolo-textocr-gpu-20261019/`.

The engineering smoke used an idle GPU. During the subsequent full run, an
independent `run_word_evidence_suite` process also used CUDA; it was preserved
and recorded in the external co-residency receipt. Own-model allocator peaks
remain per process; full-run timing must not be claimed as an isolated benchmark.

## ADASpot-GAP-CONTEXT-001 — Matched 1x1/3x3 heads with hard Gap positives

- Authorization: on 19 October 2026 the user requested starting the agreed
  comparison: frozen Text/Kernel, extra weight on Gap pixels mistaken for text,
  matched pointwise/spatial heads and word-level OCR assessment.
- Evidence inspected: original paper sections 3.2–3.3 and Algorithm 1; attached
  AdaSpot proposal sections 4.1–4.2/4.7; existing Gap probe code, config, source
  checkpoint and completed independent audits. All previously pinned source and
  input hashes were checked unchanged before the new experiment.
- `REPOSITORY_SPECIFIED`: reuse qualified Text+Kernel control, existing RGB512 /
  native256 targets, TRAIN1130/125 split, kernel BFS thresholds and PARSeq AR1.
  Reuse the old hash-verified full validation feature cache without rewriting it.
- `ASSUMPTION`: a new 3x3 linear Gap convolution (577 parameters) versus a matched
  1x1 control (65). Zero weights, bias -2, zero spatial padding; no hidden layer.
  The proposal specifies 1x1: the 3x3 head is an explicit research extension.
  Sampled patches use channel-major C/dy/dx ordering; native convolution and point
  training must agree, including edge padding and gradient routing.
- `ASSUMPTION`: retain all TRAIN Gap positives, split into predicted-M foreground
  and background. The former is the hard-positive bridge proxy; not every such
  pixel is a topological bridge. Sample valid non-Gap foreground/background
  negatives without replacement, capped at max(64, image positive count) each.
  Equal total BCE weight for the four strata comes from TRAIN counts only.
  Changed priors can overcut letters; G>.5 stays fixed, without calibration or
  validation-derived class weights/threshold searches.
- `ASSUMPTION`: both heads see exactly the same cached examples and seeded epoch
  permutations, Adam1e-3, zero decay, 20 complete epochs, no augmentation. Include
  epoch0; select native word F1, fewer fused words, earliest epoch. OCR is executed
  only after both trainings finish; it never selects checkpoints. Report original
  RGB localization+exact-casefold counts, all983 care words, gains and regressions.
- `HARDWARE_DEVIATION`: single-GPU FP32 feature extraction and CUDA head training;
  CPU/disk sampled patch cache, existing validation cache, sequential OCR loading.
  Actual GPU and allocated/reserved VRAM/OOM recorded. Concurrent GPU processes
  are preserved; no latency/GFLOPs or RTX3070 qualification is claimed.
- `UNRESOLVED`: benefit, independent validation, calibration, joint adaptation,
  feature-space Mask-Align, and actual shared recognition heads/gate.
- Alternatives: repeat the old unweighted positive mixture (does not isolate
  spatial context under new loss), nonlinear larger heads or whole-decoder
  updates (deferred to keep this comparison focused). Old runs/configs unchanged.
- Initial preflight helper was mistakenly run with system Python, which lacks
  SciPy; it exited before any experiment. Repeated in the existing main venv,
  without installation, and verified every pinned source/input hash unchanged.
- Implementation/config/docs: `models/adaspot_gap_context.py`,
  `scripts/run_adaspot_gap_context.py`,
  `configs/approximate/adaspot_gap_context_v1.json`, `docs/ADASpot_GAP_CONTEXT.md`.

### ADASpot-GAP-CONTEXT-001 follow-up — fixed epoch20 OCR diagnostic

`ASSUMPTION`: after both complete primary trainings selected epoch0 geometrically,
perform one separate post-hoc OCR evaluation of each fixed final epoch20. This
checks whether trained-head OCR differs despite no geometric promotion. No
checkpoint search, further training, threshold change, automatic promotion or
primary-result modification. Reuse the same pinned inputs, matching and
`HARDWARE_DEVIATION` (cached FP32 features, single CUDA GPU, sequential PARSeq).
Configuration and every output explicitly label the analysis post-hoc. It cannot
establish an independently validated OCR optimum.

### ADASpot-GAP-CONTEXT-001 measured outcome, 19 October 2026

Both matched heads completed20CUDAepochs/720steps, with identical580938training
examples and epoch permutations. Independent center-feature comparison against
the earlier probe is bitwise identical. Primary native-F1 selection retains
epoch0 for both arms (OCR524/983). A separate explicitly post-hoc fixed epoch20
analysis gives1x1:496/983 (8 recovered,36 damaged);3x3:476/983 (15/63). Native
fused-word involvement202→176/151 improves, while localized-and-read accuracy
worsens. Keep the original reference; no head is promoted. Equal LR across
larger fan-in, changed training priors, calibration, capacity and optimization
remain alternative explanations, not causally resolved findings.

67focused tests and both independent audits pass. A read-only helper initially
used system Python without SciPy; corrected without installing anything. The
first final-epoch diagnostic failed before inference due to a CUDA memory-stat
reset before initialization; explicit initialization fixed it. Failed diagnostic
artifacts/source snapshot are preserved, a new retry root completed, and primary
results remain unchanged. See `docs/ADASpot_GAP_CONTEXT.md` for exact roots,
metrics, memory, scope and result-class separation.

## ADASpot-GAP-NORMALIZED-001 — Normalize, separate head calibration and preserve Text support

- Authorization: user explicitly requested a better training method after repeated
  epoch0 selections. Inspect existing heads/loss/decode, original paper3.2–3.3,
  Algorithm1, and AdaSpot proposal4.1–4.2 before changes. Existing selected/last
  checkpoints and all historical experiments remain unchanged.
- Evidence: old20epoch heads cut93048/54515 valid target-Text foreground pixels;
  these are whole-word regions, not necessarily ink. Weighted BCE changes the
  precision/recall tradeoff (PyTorch official BCEWithLogitsLoss documentation).
  An explicit post-hoc restoration diagnostic on unchanged final heads gives
  spatial native matches623→656 but F1 .64031, still below original .64325;
  this is a diagnostic, not a selected replacement or independent result.
- `ASSUMPTION`: normalize sampled feature dimensions using mean/std from Gap-head
  fit images only (std floor .001). Matched normalized3x3 linear and hidden32
  ReLU heads, fresh zero output/bias-2, AdamW1e-3/decay1e-4,20epochs, identical
  pixel permutations and existing4stratum BCE weights recomputed on fit subset.
  No original segmentation parameter is changed. Compare nonlinear separability
  and stable input scales; no improvement or nonzero epoch is guaranteed.
- `ASSUMPTION`: deterministic100image calibration split inside original1130TRAIN;
  remaining1030images fit both heads and normalization. All125evaluation images
  excluded from normalization, optimization and checkpoint/decision choices.
  Frozen M/K backbone previously saw these100images: only head-optimization
  exposure is held out, not backbone exposure. Keep split/point IDs and hashes.
- `ASSUMPTION`: at fixed epochs0/5/10/15/20 calibrate decision threshold over
  predeclared .5/.75/.9/.95/.99 plus off control and cut/preserve decoder, using
  ONLY100head-calibration images and native max-one-to-one wordF1, then fewer
  fused words, earlier epoch, higher threshold, cut tie. This is decision
  calibration, not posterior calibration or temperature scaling. Selecting off
  remains legitimate if all trained candidates fail. No evaluation/OCR selection.
- `ASSUMPTION`: optional preserve decoder refills original predicted M foreground
  from already assigned Gap-cut instance IDs using existing constant-cost
  geodesic flood (length then ID ties); explicit fallback for unseeded support.
  Preserve every known ID and all original M support, never GT support. This
  separates partition from erasure but may preserve false bridges or false seeds.
  Original cut remains a matched candidate. This changes proposal postprocessing
  only in an isolated approximate extension.
- `REPOSITORY_SPECIFIED`: qualified source/checkpoint/targets, sampled580938TRAIN
  patches, RGB512/native256, immutable cached125evaluation features and qualified
  PARSeq AR1; original-RGB bboxIoU>=.5/exactcasefold, punctuation and all983care
  words retained. Existing mode boundaries/STRICT path are unchanged.
- `HARDWARE_DEVIATION`: actual single CUDA GPU, FP32 cached fit/calibration data,
  sequential loading, memory/OOM logged without automatic substitution; no8GB
  or latency qualification. Native convolutions analytically absorb fit-only
  normalization and are tested against patch prediction at image borders.
- `UNRESOLVED`: independent transfer, calibration variance/backbone exposure,
  contribution of each coupled new choice, joint feature adaptation and full
  AdaSpot recognition. New trials remain exploratory on reused evaluation125.
- Sources: https://docs.pytorch.org/docs/2.14/generated/torch.nn.BCEWithLogitsLoss.html
  (weighting and precision/recall); https://proceedings.mlr.press/v70/guo17a.html
  (distinguish learned scores from calibrated probabilities; our threshold
  selection is an explicit assumption, not their temperature-scaling method).
- Initial new unit fixture omitted existing decoder's required center_threshold;
  fixed the fixture rather than relaxing production configuration validation.
- Code/config: `models/adaspot_gap_normalized.py`,
  `localize/adaspot_gap_preserve.py`, `scripts/train_adaspot_gap_normalized.py`,
  `configs/approximate/adaspot_gap_normalized_v1.json`.

## ADASpot-GAP-DECODER-001 — Train spatial Gap features, preserve qualified M/K

User requested better training after repeated epoch0 selection. Existing code,
paper3.2–3.3/Algorithm1 and AdaSpot proposal4.1–4.2 inspected. Normalized spatial
linear/nonlinear heads both completed20epochs and retained epoch0/off524words;
this motivates testing trainable spatial features, not proof of a unique cause.

- `ASSUMPTION`: deep-copy qualified bottleneck/four attention upsampling stages,
  train copied decoder and zero Gap head/bias-2 with copied BN trainable. Original
  ResNet50 and full original M/K branch stay frozen/eval, no shared tensor storage.
  Strict source and branch checkpoint loading; exact teacher outputs and frozen
  full state including BN checked. Additional decoder is deliberately distinct
  from proposal's final shared-feature architecture; extra params/cost reported.
- `REPOSITORY_SPECIFIED`: qualified source/targets/RGB512/native256 and paired
  augmentation, source radius3 Gap target, genuine qualified PARSeq AR1 and
  full983care-word original-RGB bbox/exactcasefold protocol unchanged.
- `ASSUMPTION`: same1030fit/100head-calibration/125evaluation IDs,10full GPU epochs,
  AdamW1e-4/decay1e-4; .5four-stratum mean BCE + .5masked batch Dice + .25GT-Text
  negative BCE, Dice smooth1; absent strata zero with fixed denominator4,
  all-invalid skips optimizer/decay. No GT inference. Source teacher defines
  foreground strata detached at unchanged threshold.65.
- `ASSUMPTION`: predeclared calibration checkpoints0/1/3/5/10, thresholds
  .5/.75/.9/.95/.99 and cut/preserve/off; same wordF1/fusion/conservative ties
  as normalized trial, calibration ONLY, all10epochs executed. Evaluation/OCR
  consulted after selection, epoch0 eligible rather than forced improvement.
- `HARDWARE_DEVIATION`: actual single CUDA GPU FP32 batch4/sequential loading,
  memory/OOM recorded, explicit failure without CPU fallback. No8GB qualification.
- `UNRESOLVED`: same calibration source-pretraining exposure and reused exploratory
  evaluation, individual coupled-choice contributions, independent transfer and
  variance, full shared AdaSpot and strict reproduction. No guarantee of gain.

Details: `docs/ADASpot_GAP_DECODER.md`,
`configs/approximate/adaspot_gap_decoder_v1.json`,
`models/adaspot_gap_decoder.py`, `segmentation/adaspot_gap_decoder.py`,
`scripts/train_adaspot_gap_decoder.py`. Initial architecture test referenced
nonexistent UpBlock.conv; fixed the test's parameter access, retaining exact
teacher equality and all production validation.

## ADASpot-GAP-OCR-SELECTION-001 — Align selection with actual read words

The audited10epoch/2580CUDAstep decoder experiment chose epoch1 by calibration
native wordF1 (.83219→.83458). EvaluationRGB localized648→649, exact524→523,
2recovered/3damaged. Masks alone did not predict OCR benefit. Selected G cuts
349/3745hard Gap pixels and717valid whole-word-region pixels at threshold.9;
those pixels are word regions, not necessarily ink. These are post-hoc diagnostics.

- `ASSUMPTION`: new, explicitly post-hoc method-development trial. Before
  calibration OCR, freeze selection as maximum localized+exact-casefold word
  count on ONLY100calibration images, ties OCRHmean/earlier epoch/higher
  threshold/cut. No evaluation-derived per-image admission/gating rules.
- `REPOSITORY_SPECIFIED`: every deployable saved checkpoint: off, source best1
  and final10, fixed thresholds.5/.75/.9/.95/.99 and cut/preserve. Epochs3/5
  lack saved weights and cannot be deployed without replay; this limitation is
  explicit. Same original-RGB/casefold/care/empty-proposal rules, full983word
  evaluation denominator, source10full epochs, no retraining/source mutation.
  Same-crop qualified AR1 cache is hash-checked; missing OCR runs on GPU.
- `HARDWARE_DEVIATION`: actual single CUDA GPU FP32/batch4, sequential models,
  memory/OOM and failures reported without CPU fallback; no8GB qualification.
- `UNRESOLVED`: protocol redesigned after preceding evaluation failure, reused
  evaluation and source-pretraining calibration exposure prevent independent
  confirmation. Variance/transfer/full AdaSpot remain unresolved. No gain or
  nonzero epoch forced; baseline/off stays eligible.

See `docs/ADASpot_GAP_OCR_SELECTION.md`,
`configs/approximate/adaspot_gap_ocr_selection_v1.json`,
`scripts/select_adaspot_gap_ocr.py`. This is approximate reproduction only.

Completed outcomes for ADASpot-GAP-OCR-SELECTION-001: all21candidates,100image/
802word calibration. Off525exact, selected epoch1/.9/preserve527. Evaluation
125images/983words: exact524→526 (2recovered/0damaged), RGB localized648→649,
native650→651, proposals1038→1041, OCRHmean51.8555→51.9763%. Fusion-involved
GT words202→204: no claim of meeting the proposal's fusion-reduction target.
The improvement is small and exploratory; criterion redesign after an earlier
failure/reused evaluation prevents independent confirmation. Geometry-selected
cut523and normalized head524negative trials remain preserved. Both training
runs executed all declared epochs on GPU, no forced checkpoint replacement.

## ADASpot-GAP-FULLTRAIN-001 — All1255official TRAIN, fixed prior choices

User2026-10-20requests allimages. Inspect current model/loss,
paper3.2–3.3/Algorithm1 and AdaSpot proposal4.1–4.2. Initial1130interpretation
explicitly corrected to all1255officialTRAIN. Former100calibration/125holdout
enter optimization; old524/526 cannot evaluate new held-out performance.

- REPOSITORY_SPECIFIED: same frozen qualified ResNet50/M/K, copied32.6MGap
 decoder/zerohead, original paired augmentation/GT/radius3Gap, RGB512/native256,
 AdamW1e-4/decay1e-4, .5BCE+.5Dice+.25Textprotection, seed202503. Fresh copied
 Gap decoder, not continuation of oldGap. Source full state and BN unchanged.
- ASSUMPTION: all1255once per epoch for10epochs, save fullsampleIDs/hash;
 priorcalibrated epoch1/.9/preserve fixed before run. Save1/3/5/10. No internal
 validation. TESTcompares control,prior1030Gap,fixedfull1,final10diagnostic;
 never TEST-based winner selection or parameter revision.
- Initial preflight FAILED before training: exactRGB TEST578=TRAIN212
 (49ee079a6f9d9565f6167c4425db511d87d16aaa59f6bc87d4b2573c7bdb3d0c),
 TEST656=TRAIN81
 (3bdb83fed0aa26e9370e65113ebc3aca02eed5726d0c1e04e8743865f52d926f).
 ASSUMPTION: honor allTRAIN1255, pin ONLYthese known pairs; unknown/changed
 overlap still fails. PrimaryTEST300/2204words includes duplicates; secondary
 TEST298 excludes exactlythese2with explicit counts. No hidden exclusions or
 independentTEST300 claim. TEST298also historicallyexplored. Do not silently
 suppress overlap failure; preflight finding preserved in config/record.
- ASSUMPTION: TESTlabels frozen before nativevalidity/transcripts; onlywholly
 invalid components ignored,mixedcare retained. Nearest pairedletterboxvalidity;
 allcareGT/misses andempty/duplicatecare predictions retained. OriginalRGB
 bboxIoU>=.5 one-to-one maximumcount/exactcasefold withpunctuation. No fabricated
 TESTnativeowner/fusion metric,officialpolygon or Table5claim.
- HARDWARE_DEVIATION: actualCUDA-only FP32batch4TRAIN/batch1segmentationTEST/
 batch4PARSeq; sequentialunloading,memory/OOM with explicit failures,noCPU
 fallback or8GB/latency qualification.
- UNRESOLVED: independenttransfer/seeds,overfitting,sourceGT/architecture,
 fullsharedAdaSpot. Moreimages/latercheckpoint do not guarantee gain.

See docs/ADASpot_GAP_FULLTRAIN.md,
configs/approximate/adaspot_gap_fulltrain_v1.json and
scripts/train_adaspot_gap_fulltrain.py. Approximate reproduction only;
sourceexperiments and STRICT defaults preserved.


## V3-FAIL-001 — frozen V3/447 error attribution (2026-10-20)

User requests analysis of current failures and improvements. Paper sections3.2,
3.3 and4.1, the native contour/matching implementation, guarded V3 targets,
selected-checkpoint OCR and the prior NATIVE128-FAIL-001 audit were inspected.
This is approximate reproduction; no training, new inference or individual
crop export. The previous TEST exploration and source audits are preserved.

- `REPOSITORY_SPECIFIED`: reuse frozen V3/447 area0/pad1/top10 predictions,
  qualified PARSeq outputs, complete TEST300/2204 care words, primary unpadded
  bboxIoU>=.5 casefold exact and all predictions/GT in denominators.
- `ASSUMPTION`: additive nested matchings exact retained -> geometry retained
  -> geometry including recorded top10 rejects. Verify every maximum with
  independent SciPy. A geometry-only gain beyond cap is an opportunity, not
  measured correct OCR, because rejected crops were not recognized.
- `ASSUMPTION`: bbox coverage .5 / fragment fraction .05 proxies are inherited
  from NATIVE128-FAIL-001; area-ratio and no-overlap labels are descriptive
  geometry, not pixel-mask topology or causal attribution. Size is continuous
  GT bbox projection to128; preserve annotation orientation codes verbatim.
- `ASSUMPTION`: a best-of-six-padding oracle uses only pad1-selected contours,
  one match per contour and cached readings; label it GT-dependent diagnostic.
  Padded-box/IoU25 sensitivity never replaces the primary metric. Exact edit
  distance, alphanumeric comparison and crop bbox coverage are diagnostics;
  retain official GT and normalization unchanged.
- `REPOSITORY_SPECIFIED`: frozen training config has separation_weight=1.0;
  separator pixels receive ordinary valid-background BCE, no extra weighting.
  The existing trainer and separation_weighted_bce_with_logits already support
  a larger weight. A new architecture is unnecessary for this first ablation.
- `ASSUMPTION`: proposed follow-up ablations are separator-weighted BCE
  (or a separate separator-head experiment), true scale/resolution recovery
  for small text, fixed-budget candidate ranking and adaptive crop extent.
  Rationale: 607/1058 unlocalized GT have a larger best-IoU bbox (521 also
  meet the multi-word coverage proxy), 305/340 smallest GT are unlocalized,
  and 138 additional geometric matches occur beyond cap. These are proposed
  experiments, not demonstrated improvements or paper-specified recipes.
- `UNRESOLVED`: pixel-level merges/splits of predictions, causal gains from
  GT/loss/resolution changes, OCR beyond cap, strict identities/protocol and
  independent generalization. New tuning must use TRAIN validation.

Config: configs/approximate/guarded_v3_failure_analysis_v1.json.
Implementation and outputs are isolated in
artifacts/diagnostics/guarded-v3-failures-20261020/. No strict default changes.

Completed ADASpot-GAP-FULLTRAIN-001: all1255uniqueTRAINimages each of10epochs,
3140CUDAupdates, no skips, RTX3090FP32batch4, unchanged frozenM/Kstate/BN.
TEST300/2204words: baseline1293exact, prior1030Gap1299, fixedfull1=1311,
final10diagnostic=1324. TEST298/2187words excluding2knownRGBduplicates:
1278/1284/1296/1309 respectively. Relative to priorGap, full1recovers18/
damages6 (+12), full10recovers41/damages16 (+25) in bothcohorts. Morefinal10
proposals/lowerprecision: full1Hmean54.3082%vs54.2512% full10onprimary300.
Predeclared1/.9/preserve remains selected; no TESTbasedpromotion or newsearch.
All2897uniqueRGBcrops qualifiedPARSeqAR1GPU; primary/secondary tables,
per-sample outputs and duplicatecontributions retained.78focusedtests pass.
Source oldaudits/configs unchanged; mode remains approximate reproduction,
no claim of untouchedTEST/officialpolygon/Table5 or sourceambiguity resolution.


## MASK-CROP-FAIL-001 — Mask-to-crop research and refreshed Guarded V3 OCR failure diagnosis

User requests a new research focus on usable crops from a given U-Net mask,
first analyzing OCR failures and locating the strongest existing U-Net.
Inspected local paper3.3/Algorithm1, repository navigation/host notes, saved
model comparisons, completed GuardedV3 selection/OCR, NATIVE128-FAIL/CAP,
WORD-EVIDENCE and current localization/matching implementations.

- `PAPER_SPECIFIED`: component localization, component-area filtering, padding,
  descending component area and cap10. Changing the cap is a research extension,
  not a resolution of ambiguity or a silent STRICT/default change.
- `REPOSITORY_SPECIFIED`: GuardedV3 bbox Word-F1 selects epoch447 on TRAIN
  validation, SHA2565b44de7b88b5b1d636e8f53c586565f6491aabaae3ccea361e96e32d540c1faa.
  Completed TEST300/2204, pad1/area0/cap10, exact Unicode casefold plus unpadded
  bboxIoU.5 yields795 correct/2086 predictions/1008 geometric matches. Existing
  matched V4, separatedV1 and compactV2 results give525/648/708; identical GT
  checked. This is not a global ranking across incompatible protocols.
- `ASSUMPTION`: preserve exact matching when augmenting geometry matching;
  partition all2204 GT into795 correct,213 geometry-matched crop/OCR failures
  and1196 without a retained geometric match. Independently check every
  cardinality with SciPy. GT is exclusively diagnostic, never crop generation.
- `ASSUMPTION`: uncapped raw CC boxes establish geometry only after all300 raw
  component/external-contour counts and retained CC-ID/bboxes agree. Geometry
  grows1008->1146 with2688 instead of2086 proposals:138 additional localized
  words, not138 newly recognized words. Missing below-cap V3 OCR is not guessed.
- `ASSUMPTION`: same coverage.5/fragment.05 bbox proxies as NATIVE128-FAIL:519
  fusion-proxy/67 fragment-proxy words among1196 failures; overlapping, not
  measured pixel topology. Native-short-side bins count25/340 correct at<=4
  and565/1130 at>8. The cache lacks complete TEST masks/probabilities.
- `ASSUMPTION`: one alternative reading per fixed baseline contour supplies a
  GT-assisted padding oracle863; same-crop padded-box metric823 and positive-
  overlap exact869 are metric sensitivity, not new model performance.213
  localized failures include112 best-IoU texts one edit away and54 alnum-only
  differences; these subsets overlap and primary normalization remains unchanged.
- `HARDWARE_DEVIATION`: CPU cached-output analysis only; no neural inference,
  training, installations or GPU allocation. PAPER verified directly as paper/
  bmartelli; actual RTX3090 observed busy and left untouched. Model/RGB/proposal/
  result/checkpoint hashes, resolved config, per-word/per-image outputs retained.
- `ASSUMPTION`: next proposed work freezes V3, tests crop budget without area
  pruning, reuses bounded local split/merge generators, then learns crop
  refinement/selection with recognition supervision and coverage/neighbor
  protection. Spatial Transformer Networks1506.02025, MaskTextSpotterV3
  2007.09482 and FOTS1801.01671 are primary method inspirations, not measured
  results or imported substitutes. This proposal is not implemented or trained.
  Existing geometry-only crop correction654 vs647 geometry but502 vs520 OCR
  motivates directly supervising readability. All calibration must use TRAIN.
- `UNRESOLVED`: uncapped V3 OCR, exact pixel attribution from box-only exports,
  usefulness of each proposed module, independent calibration/generalization
  and strict artifacts/metrics. TEST was repeatedly explored and is not fresh.

23 focused/related tests and the full300-image replay pass. Original scientific
sources/weights/GT/results remain unchanged. A matplotlib-only figure attempt
failed because it is not installed; existing Pillow produced the figure, with
no installation. Full numeric report, model comparison, sources, illustrative
image/crop overlays and provenance: artifacts/diagnostics/mask-to-crop-research-20261020/.


## WORD-INSTANCE-EXT-001 — Novelty-checked local word objectives (2026-10-20)

The user explicitly requested checking previous similar repository experiments
before implementing only substantially new variants. The local v2 paper
sections3.2-3.3/Algorithm1, nearest existing crop/link/embedding/Gap/geometry/OCR
implementations, configs, audits and completed reports were inspected first.
`docs/WORD_INSTANCE_NOVELTY_AUDIT.json` preserves18source/evidence hashes, search
scope and the three implementation decisions. This establishes local experimental
differences only, not novelty against the literature or an exhaustive host search.

- `PAPER_SPECIFIED`: attention segmentation and component-based padded crops in
  paper3.2-3.3; these extra objectives are not specified by that paper.
- `REPOSITORY_SPECIFIED`: pinned completed word-evidence source contract,1130/125
  frozen maps, v4 owners/care annotation order, original RGB/manifest hashes and
  qualified PARSeq artifact/archive. Preserve original Text/Kernel postprocessing,
  checkpoint, datasets, old metrics and the entire STRICT path.
- `ASSUMPTION`: three separate fresh three-layer24channel local CNN experiments,
  not a full U-Net retrain. Prompt-selected word mask versus same-capacity
  no-prompt control; local four-edge BCE plus sampled maximin-pair BCE versus
  identical edge-BCE control; bounded four-side geometry plus frozen-PARSeq
  transcript/EOS CE versus identical geometry-only control. Never combine the
  interventions in this first study. The predicted kernel is an inference input;
  majority GT owner is consumed only for fit supervision. Tied/ignore-only/
  unowned/truncated prompt cases are counted as unsupervised, not negative GT.
- `ASSUMPTION`: local MALIS adaptation uses maximum spanning forests, at most16
  deterministic equally weighted anchors per word, separately normalized positive
  and negative pair BCE, coefficient.25. Valid background carries paths but not
  labeled endpoints; ignored pixels carry neither. It is not original all-pairs
  squared Rand loss or constrained MALIS. Strong false merges can make negative
  maximin paths penalize internal bottlenecks; this limitation is tested explicitly.
  Source: https://proceedings.neurips.cc/paper_files/paper/2009/hash/68d30a9594728bc39aa24be94b319d21-Abstract.html .
- `ASSUMPTION`: source boxes choose50%context with minimum2native pixels; RGB
  exact square extent to96square bilinear, priors bilinear and discrete maps
  nearest. BCE+Dice mask objective; bounded25%geometry changes with normalized
  SmoothL1 and.1causal PARSeq CE. Original RGB qualified bicubic inference differs
  from continuous bilinear crop training. This mismatch is declared, not qualified
  as equivalent. Unsupported/long/empty training OCR labels are counted and never
  remove evaluation care words. Frozen recognizer weights remain exact; no new
  PARSeq checkpoint, fallback or paper-S identity is claimed.
- `ASSUMPTION`: deterministic1030fit/100head-calibration/125evaluation partition,
  seed20261020,20epochs,batch1,AdamW1e-4/decay1e-4,no augmentation,one paired
  predicted window per fit image/epoch. Calibration has original-backbone exposure,
  but none to new-head optimization. Compare fixed0/5/10/20 and baseline/off using
  exact read+localized word count, then OCR Hmean and geometry Hmean. Off/earlier
  candidates win ties. No evaluation label chooses a candidate or threshold.
- `ASSUMPTION`: graph/mask extraction retains only the component anchored at the
  selected predicted kernel. Empty/no-kernel cases explicitly keep source boxes;
  no orphan words, new seeds, dedup, area filter, padding or cap. Affinities operate
  within frozen predicted Text only. These choices limit recall and retain possible
  duplicate kernels; this is not Algorithm1 or complete scene-text reproduction.
- `ASSUMPTION`: original exclusive RGB bbox maximum one-to-one IoU>=.5 and exact
  Unicode casefold, punctuation retained, every983care word and empty/duplicate
  care prediction counted. Ignore identities come from original wholly invalid
  instances after GT-free inference; partially ignored source instances stay whole.
  Recompute a common matched baseline under current device/batch1 rather than
  assume the historicalCPU/batch16score520. Keep per-image/per-word evidence.
- `HARDWARE_DEVIATION`: new local model plus frozen PARSeq on explicit device,
  source segmentation as cached maps,FP32/batch1,CPU differentiable grid sampling
  and gradient-preserving transfers to avoid CUDA grid-sampler nondeterminism.
  Actual hardware,co-resident jobs,allocator peaks and OOM propagate in receipts.
  No latency,8GBqualification or paper-equivalence claim.
- `UNRESOLVED`: benefits,capacity/window limits,unseen evaluation and variance,
  crop-transform mismatch,training target tradeoffs,strict paper artifacts and
  official recognition protocol. Existing125image evaluation is exploratory.

Implementation: `models/word_instance_extensions.py`,
`segmentation/word_instance_extensions.py`, `segmentation/word_instance_data.py`,
`localize/word_instance_extensions.py`, `recognition/parseq_crop_objective.py`,
`scripts/run_word_instance_extensions.py`; explicit namespace config
`configs/approximate/word_instance_extensions_v1.json`. Detailed protocol and
negative source results: `docs/WORD_INSTANCE_EXTENSIONS.md`.

Qualification history: an initial maximin-gradient test assumed a boundary was
the bottleneck when a stronger false bridge made internal edges the bottleneck.
Independent brute-force widest-path comparison confirmed the implementation;
corrected the fixture and added an explicit limitation test. First CUDA smoke
passed1255input hashes and real PARSeq inference, then failed before training
on NumPy-int JSON boxes; canonicalized reader-boundary integer coordinates and
added a regression test. Failed output/source retained, no fallback or failure
suppressed. Corrected CUDA smoke executes8real updates for each of sixarms,
including8OCR-supervised updates; all gradients finite/nonzero, PARSeq state
unchanged, no OOM. This is engineering smoke, not evidence of better OCR.

## YOLO-TEXT-COVERAGE-RESOLUTION-001 — completeness gate and larger detector inputs

User-authorized implementation,2026-10-21: retain a considerable light-branch
share while checking for unread text; compare larger YOLO image inputs.
Inspected local2503.15639v2 sections3.3/3.5/Algorithm2, existing direct-crop,
paired-router, U-Net export/projection and evaluation code before modification.

- `PAPER_SPECIFIED`: section3.5 routes uncertain images to DeepSolo. The
  proposed coverage rule, YOLO and resolution controls are external extensions.
- `REPOSITORY_SPECIFIED`: pinned YOLO/PARSeq/CLIP outputs and software,
  TRAIN80 calibration/45 validation IDs, original stored RGB crop pixels,
  predicted two-channel Text/Kernel256 maps and frozen425 complete
  ViTAEv2-S+TextOCR fallback predictions. Preserve640 outputs exactly.
- `ASSUMPTION`: probability-mass foreground coverage within final exclusive
  word rectangles, mapped by the existing center-pad/nearest256 geometry;
  threshold0.5, uncovered-component diagnostic minimum area4. Exclude padding,
  read only `probabilities`, keep GT/validity/owners out of inference. Empty
  masks are neutral; empty readings force fallback. An imperfect predicted
  mask cannot certify complete text recovery.
- `ASSUMPTION`: mix minimum native/CLIP word score with TRAIN-CDF coverage
  rank; select Text/Kernel and weight0/0.25/0.5/0.75 by TRAIN80 combined microF1
  at50% bypass. Equal scores retain the simpler confidence-only control.
  Freeze all fits before new TEST features; report the45 validation images
  separately. TRAIN125 is not the full training split; DeepSolo saw upstream
  TotalTextTRAIN. Historical TEST exploration remains a limitation.
- `ASSUMPTION`: primary GT-free batch rank budget50%, controls30/70%, with
  sample-ID tie breaking. This guarantees the requested batch bypass share,
  not a calibrated single-image reliability. Transfer real TRAIN scalar
  thresholds separately and report actual TEST counts, including drift/ties.
- `ASSUMPTION`: resolutions640/960/1280 with unchanged joint0.6/same-reading
  NMS0.5. New detector floor0.35 preserves all potentially eligible candidates
  because joint0.6 requires detector score at least0.36. Reuse audited960
  TEST CPU detector/OCR; obtain missing960 TRAIN and1280 TRAIN/TEST outputs.
  Keep the common exact-casefold/punctuation-retained per-image multiset
  metric and evaluation-only ignore rule, not official polygon WordSpotting.
- `HARDWARE_DEVIATION`: actual RTX3090, existing isolated environment,
  deterministic TF32-disabled sequential CUDA FP32 YOLO batch1, CPU FP32
  PARSeq batch32 and CUDA FP32 CLIP batch16. Record original image dimensions,
  actual YOLO tensor shapes, peak allocated/reserved VRAM, co-resident jobs,
  and explicit failures/OOM. Cached CPU versus new GPU detector provenance
  prevents attributing every difference exclusively to resolution.
- `UNRESOLVED`: target paper accuracy/checkpoint mapping, STRICT artifacts,
  quality of the incomplete-mask signal and target8GB qualification. No
  superiority claim from smoke or incomplete measurements.

Implementation and reproducible commands:
`docs/YOLO_TEXT_COVERAGE_RESOLUTION.md`,
`recognition/text_coverage_router.py`,
`scripts/run_yolo_coverage_resolution.py` and isolated approximate profiles.
Fourteen focused unit tests and real GPU smoke passed; full measurement and
independent audit were pending when this protocol entry was written.

Completed evidence,2026-10-21: all125 TRAIN/300 TEST images at640/960/1280;
no failed/omitted sample. Calibration selects Text/0.25 at640, confidence-only
at960 and Kernel/0.25 at1280. At exactly150/300 TEST bypass images, selected
gate TN0/F1 are63.3925/77.5954%,59.6452/74.7222%,56.6488/72.3259%.
The640 confidence control is63.0278/77.3215%; the small TN0 gain+0.3648pp
has conditional paired95% interval[-0.6091,+1.3578]pp and does not establish
superiority. Complete accepted images remain41/150. Larger inputs degrade
the tested system. Fixed TRAIN thresholds drift to different TEST bypass
counts; the50% TRAIN cut accepts186/300, distinct from the150-image budget.
Sixty-three tests, genuine GPU smoke and independent audit pass; the latter
replays1,275 feature records,10,284 RGB crop hashes and all metric/routing
counts. No real OOM occurred; a fixture verifies OOM failure propagation.
New-stage allocated/reserved peaks796,635,136/1,210,056,704bytes exclude cached
U-Net/DeepSolo inference, so no full latency/8GB or strict-paper claim.
All outputs, source/checkpoint/software hashes and scientific limits are
linked from `docs/YOLO_TEXT_COVERAGE_RESOLUTION.md`; unresolved blockers remain.


## ADASpot-MASK-ALIGN-001 — Shared-feature sampler before new OCR heads

- Authorization: user accepts positive full1255 Gap results and requests the
  next incremental proposal step. Original paper3.2–3.3 and supplied author
  proposal4.3–4.6 inspected, together with frozen models and Phase0 RGB fitter.
- `ASSUMPTION`: proposal4.3 motivates skeleton/cubic/EDT/normal8x32 and rotated
  branch fallback, but this is not specified by the original paper. Reuse local
  Zhang–Suen thinning; remove redundant axial-supported diagonal graph edges,
  fit a scaled transverse cubic over the dominant rectangle axis, extend to
  full pixel support with constant endpoint EDT radii and .5pixel floor. Sample
  projected coordinate, not arc length. Branch/cycle/disconnected/short/rank/
  fold failures fall back; reject >25% outside among64probe positions and reverse
  steps>.5nativepixel. These numeric safeguards are predeclared without TEST
  OCR tuning. Degenerate masks retain IDs via rotated fallback; empty is error.
- `REPOSITORY_SPECIFIED`: audited full1255 Gap epoch1/.9/preserve and its exact
  frozen care proposals/boxes. Strictly load original Text/Kernel source, keep
  all weights/BN immutable, replay saved sigmoid probabilities within1e-6FP32.
- `ASSUMPTION`: use existing original Text/Kernel F64/native256 from RGB512,
  whereas proposal assumes stride4. Export8x32 and16x64 from this same F; no new
  higher-resolution deep feature yet. Detached geometry, bilinear feature
  sampling with pixel-center coordinates, align_corners=false and zero padding.
- `ASSUMPTION`: separately score same frozen PARSeq AR1 on original-axis and
  matched32x128 bilinear-axis/rotated/mask RGB geometry. This re-encodes RGB and
  is not feature OCR accuracy, H2 validation, a learned head or shared latency.
  Keep all300/2204 and2624care predictions, one-to-one bboxIoU>=.5/casefold with
  punctuation, unchanged ignore admission; paired recovered/damaged IDs. Also
  report same298/2187 excluding exactly known duplicate TEST578/656. TEST was
  historically explored and primary300 contains exact TRAIN RGB duplicates.
- `HARDWARE_DEVIATION`: actual GPU checked before run, sequential FP32 CUDA
  source/PARSeq and no CPU inference fallback. CUDA grid_sample backward may
  be nondeterministic: explicitly use CPU sampling for deterministic autograd
  then transfer outputs/gradients back to CUDA; verify actual F gradients.
  Record device boundaries, memory peaks, batch/shape and OOM. No8GB claim.
- `UNRESOLVED`: new light/deep heads, feature-to-token training, learned gate,
  joint training and generalization/improvement from real feature recognition.
  No strict, official polygon, Table5 or complete AdaSpot claim.
- Alternatives: reuse RGB Bézier as if it were feature Mask-Align (rejected);
  install another skeleton library (unnecessary); silently nondeterministic
  CUDA backward (rejected); train several new major components at once (defer).
  Details, config, reproducible commands and completed receipts:
  `docs/ADASpot_MASK_ALIGN.md`, `configs/approximate/adaspot_mask_align_v1.json`.


## MASK-CROP-TEST-001 — Three frozen-V3 mask-to-crop experiments

2026-10-21. Authorized by the user to implement and launch all three proposals.
This is APPROXIMATE exploratory research, not STRICT or a Table5 reproduction.
Authoritative evidence read: local paper section3.3/Algorithm1 and section3.2;
existing native128 localization, crop-recovery primitives, qualified PARSeq
adapter and its pinned upstream encode/decode implementation.

- PAPER_SPECIFIED: component localization, padding and maximum10 reference.
- REPOSITORY_SPECIFIED: GuardedV3/447 selected on TRAIN validation bboxWordF1;
  freeze its checkpoint and qualified PARSeq AR+refine1. Retain RGB256/native128
  stretch geometry, historical padded-bbox ranking, area0/pad1, exact casefold
  with punctuation and unpadded bboxIoU>=.5 maximum one-to-one matching.
- ASSUMPTION: budgets20/unlimited are research variants; use identical crop
  pixels/recognition across nested budgets. Local threshold/erosion/geodesic
  and seam partitions plus nearby fragment unions reuse existing generators.
  Native-to-RGB mapping must remain stretch, not the generators' letterbox
  mapping. Bounded groups can replace originals, preserving identity as an
  available validation-selected policy. All parameters are explicit in
  configs/approximate/mask_crop_three_tests_v1.json.
- ASSUMPTION: train a correctness selector only on the frozen1130 TRAIN IDs;
  calibrate its group cost/margin on existing125 held-out TRAIN IDs. Negatives
  include fragmented, multiword and spatially incorrect crop readings.
  GT-free inference scores expected correct words against proposal cost.
- ASSUMPTION: train a small RGB+mask-context crop refiner, up to4096 seeded
  eligible TRAIN proposals, with bounded translation/scale. Loss combines
  frozen PARSeq causal teacher-forced CE including EOS, GT coverage, neighbor
  contamination and box regression. Training's bilinear differentiable sampler
  is an approximation of deployment's unchanged integer crop/PIL bicubic
  preprocessing; evaluate deployment crops with qualified AR+refine1. Keep
  original detection boxes fixed when measuring crop-only OCR refinement.
  Unsupported/too-long training transcripts are recorded as training exclusions
  only, never removed from evaluation denominators.
- HARDWARE_DEVIATION: actual single GPU/FP32, segmentation batch1 and recognition
  batch32, refiner training batch16. Sequential model stages. CUDA grid_sample
  backward is nondeterministic: permit this explicitly during refiner training,
  preserve seeds/noTF32/noAMP, restore deterministic inference, record memory
  peaks and OOM. Existing other jobs are not stopped.
- UNRESOLVED: TEST and the existing125 TRAIN holdout have been repeatedly
  explored; no independent generalization claim. Strict PARSeq-S identity and
  official paper recognition metric remain unresolved. More geometric matches
  and a GT-conditioned candidate-pool oracle are not achieved OCR improvements.

No original GT, trained U-Net/PARSeq weight, source result, strict setting or
production routing is overwritten. Frozen run outputs retain full per-sample
proposals, hypothesis lineage, OCR, matching, gains/losses, selections, source
hashes, checkpoints and actual hardware metadata in a fresh external run.


## WORD-INSTANCE-RUN-001 — Authorized full six-arm launch (2026-10-21)

The user explicitly requests implementing and starting the tests after the
WORD-INSTANCE-EXT-001 implementation qualification. Re-read the existing protocol,
verified the unchanged qualified739-file snapshot and passed26focused tests before
launch. No scientific architecture,loss,threshold,optimizer,seed or budget changes.

- `ASSUMPTION`: execute the already documented three matched pairs,20epochs each,
  plus the original reference;1030fit/100head-calibration/125evaluation images,
  all983evaluation care words. Fixed0/5/10/20 and off selection uses only head
  calibration OCR. This remains a reused-TRAIN exploratory comparison, not
  independent TEST or a paper-table reproduction.
- `REPOSITORY_SPECIFIED`: reuse hash-bound source maps/manifests, qualified
  v1control and genuine PARSeq artifacts. Executable source is copied byte-exactly
  from qualified_source_v3, with739hashes and an immutable resolved command.
- `HARDWARE_DEVIATION`: actual PAPER RTX3090/24GiB,FP32/batch1,CPU deterministic
  differentiable sampler, cached segmentation and sequential six-arm execution.
  GPU idle and355GiB disk free at launch; preserve unrelated processes. Original
  qualified runtime is used without installation/download or fallback.
- `ASSUMPTION`: a durable external supervisor preserves runner failure/exit code
  and, only after full success, starts a read-only independent SciPy Hungarian
  audit of3375planned image/candidate cases, full populations/denominators,
  checkpoint/selection/source hashes, pair deltas and recovered/lost word IDs.
  Its numerical matching self-test passes before launch. Failure is explicit;
  a result summary is created only after successful independent checks.
- `UNRESOLVED`: actual efficacy,completion,time and independent generalization.
  Launch or early baseline metrics do not establish a successful experiment.

Run: `/extra1/bmartelli/context-str-runs/word-instance-extensions-20261021-v1/`.
Supervisor PID1663313, runner PID1663314 at launch. `launch_receipt.json` includes
actual command/cwd,environment,hardware,source hashes and qualification pointers.
`pipeline_status.json` and `suite.log` expose progress; `results/progress.json`
and per-arm training receipts expose completed epochs. Final `comparison.md`
and `independent_audit.json` are automatic outputs, not yet available at launch.
The full1255image/map preflight passed and genuine baseline PARSeq inference is
running. Detailed operation: `docs/WORD_INSTANCE_EXTENSIONS.md`.


## MASK-CROP-TEST-001-RESULT — Completed experiments and qualification

2026-10-21, approximate reproduction only. All three authorized tests completed
with frozen V3/447 and qualified PARSeq. Full explored TEST300/2204 care GT:
baseline795/2086,F1.370629; cap20:876/2552,F1.368377; unlimited904/2688,
F1.369583 (+109/-0). Split/merge selector870/2190,F1.395995 (+92/-17).
Recognition-trained refiner768/2086,F1.358042 (+32/-59); all8epochs complete,
epoch7 selected on TRAIN holdout. Identity accepted on holdout before TEST,
so its accepted policy retains795; the negative learned result is not hidden.
Selector cost.15/margin0 chosen on holdout320→365/983; no TEST selection.

ASSUMPTION: report complete primary populations, secondary diagnostics and
accepted/raw policies separately; candidate pool1032 correct out of5537 is a
GT-conditioned diagnostic, not achieved selector accuracy. No production
change. Independent audit PASS for4800 SciPy matching cardinalities, source
hashes, complete populations, frozen selections and fixed crop-only detection
boxes.46focused unit tests and frozen-code GPU smoke PASS.

UNRESOLVED: independent generalization remains unavailable; audit additionally
finds two exact image-file SHA256 overlaps in the local TRAIN/TEST artifacts:
TRAIN212=TEST578,TRAIN81=TEST656. Do not silently change or filter the primary
300-image/2204-care denominator. Post-hoc sensitivity only,298images/2187care:
baseline782,F1.367309; selector858,F1.393488; refiner755,F1.354627. Both
conclusions persist. File identity checks are not an exhaustive near-duplicate
search or evidence about another copy of the official dataset.

HARDWARE_DEVIATION: actual RTX3090/24GiB FP32; refiner CUDA nondeterministic
backward recorded, deterministic inference restored. Refiner+frozen-PARSeq
training peakallocated591035904/reserved645922816bytes, batch16, contexts
4x64x256 and sampled recognition RGB3x32x128. No OOM. Original recognizer
tensor hash unchanged before/after2048crop-refiner optimizer steps.

Engineering issue: first isolated budget launch failed before inference because
the snapshot omitted pipeline; failure/log retained. Complete source snapshot
imports, smoke and full rerun passed. No dependency install, GT/old weight/old
result modification, or unrelated source edit. External run and code snapshots:
`/extra1/bmartelli/context-str-runs/mask-crop-three-tests-20261021-v1/`.
Human documentation: `docs/MASK_CROP_THREE_TESTS.md`.


## GRAPH-CUT-WORD-001 — V3/447 graph-cut word partition (2026-10-21)

User authorizes implementing and testing the proposed new graph-cut method.
Paper3.3, existing native contours, mask_crop_hypotheses, geodesic_partition,
frozen V3 selection/OCR and the max-flow API were inspected before changes.
Primary references: Boykov/Kolmogorov2003,
https://www.microsoft.com/en-us/research/publication/computing-geodesics-and-minimal-surfaces-via-graph-cuts/ ;
SciPy maximum_flow documentation (actual installed SciPy version recorded).

- `PAPER_SPECIFIED`: localization from segmented text, padded crops, max10.
- `REPOSITORY_SPECIFIED`: fixed V3/447 selected by TRAIN bbox Word-F1; same
  native128 probabilities, RGB256 preprocessing, pinned genuine PARSeq AR/refine1,
  pad1/area0, legacy padded-bbox ranking, primary unpadded original bboxIoU>=.5
  casefold exact with punctuation retained and all care GT/predictions counted.
- `ASSUMPTION`: smallest first extension is support-preserving binary s-t
  Potts cut with two word labels and fixed zero background. This is not free
  three-label optimization or a missing-text recovery method. Each source
  component emits at most two instances; no uniform erosion/discarded fragments.
  Threshold/erosion seeds are shared with the existing minimax geodesic control.
- `ASSUMPTION`: graph grid2x native inside each original-RGB ROI; eight-neighbor
  nonnegative RGB/probability affinities and normalized seed-distance unary.
  Integer quantization256 is explicit; hard capacity exceeds finite sum;
  overflow fails. Verify integer energy equals max-flow and tiny graphs against
  exhaustive enumeration. No claim of exactness for the unquantized energy.
- `ASSUMPTION`: fixed seed/energy/geometry/gate grid in the isolated config,
  on repeatedly explored TRAIN125/983 only. Max exact read+localized count
  subject to nondecreasing OCR F1; prefer fewer changes on ties, retain off.
  Freeze choice before TEST300. Report best active diagnostic even when off
  wins; matched geodesic control changes only partitioning algorithm/gating
  choice. No TEST parameter optimization or promotion from TEST results.
- `HARDWARE_DEVIATION`: actual RTX3090, CUDA FP32 batch1 segmentation then
  batch32 PARSeq, CPU SciPy cuts, deterministic/noTF32/noAMP. Use existing
  environment with no installation. Record peaks/OOM and software versions.
  Cache small exact-FP32 probability arrays and numeric/RLE partitions; RGB
  crops stay in RAM and no weights are duplicated.
- `UNRESOLVED`: actual OCR benefit, seed-to-word ambiguity and false character
  splits, background/absent-word recovery, independent seeds/evaluation and
  strict artifact/recognition metric identity. This is approximate reproduction,
  not the original paper Algorithm1, Table5 or context/DeepSolo validation.

Config: configs/approximate/guarded_v3_graphcut_v1.json. New implementation:
localize/word_graph_cut.py, scripts/run_guarded_v3_graph_cut.py. Old modules,
weights and all source audits remain intact. Evaluation uses frozen executable
source copies so unrelated workspace edits cannot change a running experiment.

Completed measured diagnostic: validation320->331 correct /983; fixed graph-cut
TEST795->775 /2204,50 recovered/70 lost, OCR F1 .370629->.355668. Active matched
geodesic793; its validation promotion remains off. No TEST retuning. Source/native
baseline replay, per-image OCR audit and4021 partition review pass. Support is
preserved, but60 previously correct crops are split and10 are displaced by cap;
50 losses lack geometry and10 further split losses have wrong OCR. Retain the
baseline as reference; this method's observed benefit is negative. Generalized
graph-cut efficacy and better word markers remain UNRESOLVED. Failed first smoke
(rank field) and packaging-directory error are preserved/documented, both fixed
before the successful full run. See docs/GUARDED_V3_GRAPH_CUT.md and the result
review in artifacts/diagnostics/guarded-v3-graphcut-20261021/.


## MASK-REFINER-001 — small whole-image RGB/mask/probability correction

User authorized online investigation, implementation and multiple architecture/training experiments on2026-10-21. Local paper3.2–3.3/Algorithm1 and existing crop/prompt/DB/Gap implementations inspected. Details and literature distinctions: [MASK_PROBABILITY_REFINER.md](MASK_PROBABILITY_REFINER.md); hashed local near-prior evidence: [MASK_PROBABILITY_REFINER_NOVELTY.json](MASK_PROBABILITY_REFINER_NOVELTY.json).

- PAPER_SPECIFIED: segmentation/component cropping are specified; additional corrector training is not.
- REPOSITORY_SPECIFIED: frozen SHA-bound Text/Kernel predictions, v4 full-word and distance-kernel targets, qualified original-RGB PARSeq runtime and existing matching helpers. This reuses prior assumptions without promoting them to paper evidence.
- ASSUMPTION: fresh8-input/two-residual-output shallow/dilated/tiny-context-U-Net, shared3pass U-Net, exact zero-head source identity with epsilon1e-4 stable-logit proxy, corrected probability difference, maxdelta12. 80whole-image epochs1030fit/100calibration/125evaluation, batch8, AdamW3e-4/decay1e-4/cosine1e-5, clip5, exact paired D4 augmentation, separately normalized class BCE+.5Dice+.05probability preservation; matched error/boundary4x and50%predicted-mask morphological corruption ablations. Adaptations inspired by CascadePSP/PointRend/SegRefiner; no reproduction of their full methods or literature novelty claim.
- ASSUMPTION: calibrate epochs0/5/20/40/80 by exact OCR count, then OCR Hmean, then geometry Hmean; source off wins ties; all arms selected before125evaluation. Unchanged watershed .65/.5, pad0/area0/no cap, wholly invalid predicted-support ignore, original exclusive RGB bboxIoU>=.5 one-to-one + exact casefold. Native valid foreground micro pixels .65/.5, empty-union score1. All983care words and125images retained.
- HARDWARE_DEVIATION: actual RTX3090 FP32batch8, source maps cached onCPU, one corrector plus frozen PARSeq resident; deterministic CUDA/noAMP/TF32,workers0,CPUthreads4. Actual peaks/OOM/errors recorded; not8GB or latency qualification.
- UNRESOLVED: real OCR gains, fusion/fragmentation, high-confidence missing regions, generalization from source-in-sample predictions and synthetic errors, reused holdout, independent data/seeds and inference overhead. Calibration images were seen by the source segmenter. No strict reproduction, paper metric or statistical significance claim.

39focused tests passed; six-arm16image/two-update real CUDA smoke succeeded, body/head changed and frozen PARSeq unchanged. Smoke allocator2,422,149,120bytes/reserved2,623,537,152bytes, noOOM. Artifacts: /extra1/bmartelli/context-str-artifacts/mask-refinement-battery-20261021-v1. Performance claims await complete calibration/evaluation.


## RESEARCH-SYNTHESIS-001 — Cross-workstream saved-output research synthesis

2026-10-21, authorized analysis of work to date and other agents' completed
experiments. No neural inference/training or changes to scientific components.
REPOSITORY_SPECIFIED: reuse each audited experiment's protocol and original
counts; verify source/result SHA256 and completion rather than stale launch notes.
The six local word-instance arms finished20epochs, independent audit PASS3375
cases: prompt/maximin select off, geometry522 vs geometry+OCR520 on125/983.
ASSUMPTION: conditional paired-image bootstrap,10000draws,seed20261021,
percentile95% intervals. Frozen models/policies; no inference about training
variance, tuning multiplicity or independent generalization. V3 selector
F1delta+2.5365pp, CI[+1.4084,+3.6921],60images improve/14worsen. Gapfull1
versus M/K +.3433pp,CI[-.0901,+.8489]; RGBMaskAlign -5.8409pp,
CI[-7.1221,-4.5823]. Negative results are retained.
ASSUMPTION: GT-conditioned unions of existing crops/readings with maximum
one-to-one bboxIoU>=.5/casefold matching, same-crop/bbox deduplication, no full
mutually-exclusive-group constraints. OriginalV3:795,uncapped904,selected870,
uncapped+selected oracle996,full local pool1032,raw+local oracle1141. These
are potential of candidate pools, not deployed selections or promised gains.
Descriptive cross-family GT overlap:754common,116onlyV3selected,557onlyGap;
identical GT/order verified, but protocols and runtime costs differ.
REPOSITORY_SPECIFIED: current OCR-informed selector actually needs5165unique
RGBcrop readings from5537proposals to emit2190words, versus2086baseline:
2.476x cold crop workload is not a measured latency ratio. Historical same-
crop cache reuse is not a deployment speed benefit on new images.
UNRESOLVED: Gap1's1311 versus V3selected870 cannot isolate architecture,
training, target, resolution, budget or ignore effects. Only paired internal
comparisons support effects. Feature-space OCR on Mask-Align is untrained;
negative RGB-proxy scores do not establish its eventual accuracy. Need fresh
validation and multiple seeds; TEST and298sensitivity are already explored,
with2known exact local TRAIN/TEST overlaps retained in primary denominators.

Prioritized proposals are explicitly untested hypotheses: transfer discrete
selection to M/K with matched Gap/noGap controls; adaptive budget and cheap
pre-OCR quality selection; subsequently equal-capacity feature-OCR axis versus
Mask-Align. Avoid repeating already-negative local heads or RGB warps without
a distinct controlled factor. Scientific source/paper paths and primary
literature links are in docs/RESEARCH_SYNTHESIS_20261021.md. Machine-readable
config is configs/approximate/research_synthesis_20261021_v1.json; numerical
outputs/source hashes under artifacts/diagnostics/research-synthesis-20261021-v1/.

## UOCR-001 — Auxiliary transcription gradients into the shared Text/Kernel decoder

User2026-10-21authorized implementation and test launch after completed local
crop/affinity/OCR-guided geometry experiments. Inspect paper3.2–3.3/Algorithm1,
original Text+Kernel, qualified source research-v1/control epoch2 and negative
word-instance extensions first. This is an isolated approximate research
extension, not strict paper reproduction or a replacement recognition adapter.

- REPOSITORY_SPECIFIED: source47a941e17472d26c4fff259d0a81c061e23040059c49d97cf2a89a9430045cd8,
  frozen ResNet50/encoderBN, original shared decoder/Text+Kernel, full v4 targets,
  paired512/256 letterbox and masked BCE+Dice with separator weight3. Qualified
  genuine PARSeqAR1 remains final OCR, watershed/nocap/noarea/pad0 as a diagnostic.
- ASSUMPTION: train a temporary two-conv64/BiGRU64/linear CTC reader on native
  GTowner-box8x96features, fit-only casefold character vocabulary/blank0. Compare
  identical-capacity reader learning on detached features with connected CTC
  gradients into the existing decoder. Reader discarded in inference. Shared
  detection/recognition features are inspired by FOTS1801.01671; this train-only
  adaptation, not frozen-PARSeq box-coordinate loss, is locally unmeasured.
- ASSUMPTION: common3epoch reader-only warmup without mutating source/BN, followed
  by20matched AdamW epochs, decoderLR1e-5/reader1e-3/decay.01/OCRweight.1, separate
  gradient clipping5. Disable augmentation in BOTHarms to avoid mirror-label and
  clipped-word confounds in first test. At most16words/image cycle by epoch,
  length48/CTC feasibility and all missing/unsupported/empty supervision counted.
  Loss character-normalized perword, mean per supervised image then batch.
- ASSUMPTION: original1130split into1030fit/100calibration with seed20261020;
  preserve original125evaluation/all983care words. Calibration-only epoch0/5/10/20
  selection by exact localized+read count/Hmean/earliest. Offeligible, no evaluation
  choice or official TEST access; all20epochs run. Care/ignore and bbox matching
  as documented, all misses/empty/duplicate outputs preserved in denominators.
- HARDWARE_DEVIATION: actualCUDA FP32batch4, both arms CPU bilinear feature
  sampling and CPUCTC with autograd transfers for deterministic backward. Original
  models, GRU/CNN and optimizers CUDA; PARSeq sequentially after unloading. Record
  actual GPU/allocated+reserved/memory/OOM and zero/nonzero feature gradients;
  no automatic device/precision/model fallback and no8GB/latency qualification.
- UNRESOLVED: source architectural/annotation approximations, auxiliary weight,
  axis-grid distortion, limited supervision/augmentation, single-seed variance and
  independent transfer. Calibration images already seen by source segmentation;
  evaluation125reused. No claim of80percent, independent validation, strict Table5,
  official polygons, exact recognition accuracy or qualified PARSeq-S provenance.

See docs/UNET_AUXILIARY_OCR.md and configs/approximate/unet_auxiliary_ocr_v1.json.
Preserve original runs/checkpoints; preflight/smoke status does not establish gain.


## GRAPH-CUT-ORACLE-001 — recorded parent/child selection ceiling (2026-10-21)

User asks what the completed graph-cut experiment teaches and what to improve.
Review source GRAPH-CUT-WORD-001 and its frozen comparison before this analysis.
This is approximate reproduction, GT-dependent diagnostic, no new inference.

- `REPOSITORY_SPECIFIED`: fixed V3/447 graphcut-e0-p0.65-a0.3 outputs, original
  baseline parents, actually retained/recognized graph-cut children, unchanged
  exact Unicode-casefold and original bboxIoU>=.5 matching; max10 per image.
- `ASSUMPTION`: enumerate each parent's original-versus-child branch, with
  parent/child mutual exclusion. Allow GT-based rejection of any candidate,
  including an incorrect sibling; maximize one-to-one exact matches up to10.
  For each branch, min(10, maximum matching cardinality) is attainable because
  this oracle can retain only its matched crops. This is a ceiling for the
  restricted recorded/recognized pool, not for every possible segmentation.
- `HARDWARE_DEVIATION`: CPU-only cached-output analysis using SciPy maximum
  matching. No model loading, changed precision, RGB generation or crop export.
- `UNRESOLVED`: how closely a real GT-free selector can approach the ceiling,
  action-level learned utility, independent transfer, and whether ILP selection
  improves scores relative to existing candidate-quality/greedy selectors.

Measured: baseline795, applied graph cut775, GT oracle844 on300images/2204care
words;554 branches enumerated, at most4available split parents per image. The
+49 is NOT a deployable gain or F1/precision estimate. TEST is already explored;
no test-driven retuning or original-experiment metric changes are performed.

Proposed next controlled factors (`ASSUMPTION`, unimplemented in the inspected
code): learn whole-action split utility against keep-parent, with penalties for
losing previously correct words; independently compare global budgeted ILP
parent/child selection against the existing greedy selection at the same scores.
Existing single-crop correctness MLP and parent/child hypothesis storage are
not called new methods. A quality predictor or optimizer alone does not guarantee
better words. New model/threshold tuning must use TRAIN-only calibration.

Artifacts: artifacts/diagnostics/guarded-v3-graphcut-20261021/selection_oracle.py
and selection-oracle/results.json (per-image counts, sources, code checksums).
Primary optimization reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html .


## ADASpot-JOINT-001 — Two feature OCR heads, calibrated gate and joint training

- Authorization: user explicitly requests all three remaining AdaSpot components
  and joint training/testing together. Original paper3.2–3.3/recognition and supplied
  author proposal4.1–4.8 inspected with the existing backbone, Mask-Align and
  official qualified PARSeq decoder/tokenizer/checkpoint.
- `ASSUMPTION`: implement proposal NAR1plusonecloze light8x32, AR4deep16x64,
  two-layerMLPcorrectness gate and joint DiceM/K/G+CE_L+CE_D+BCE_gate, which
  are not original-paper components. Gate cues detach; groundtruth only supervises
  losses/training matching and calibration, never prediction geometry/routing.
- `REPOSITORY_SPECIFIED`: strictly reuse qualified Text/Kernel source and pinned
  official PARSeq runtime/archive/checkpoint. Copy all26shape-compatible layer0/
  norm/textembedding/posqueries/classifier tensors to eachhead via explicit merge
  and strict loading; list every fresh key. Extra3deep layers initialize fresh.
  This does not resolve the original exact PARSeq-S artifact ambiguity.
- `ASSUMPTION`: one shared decoder replaces the earlier dedicated32.6MGap branch;
  unfreeze encoder/BN and allheads. Preserve v4RGB512/native256 M/K/G onF64stride2,
  pool2forstride4lightF; deep uses higher-resolution originalF. Segmentation
  stride2 differs from proposal stride4. NewGap1x1 zero/bias-2 warmstart; F64→384
  projections, fixed2D positions, norm, official12attentionheads/MLPratio4/dropout.1
  and25character/94charset interface. Geometry remains detached and mask-only.
- `ASSUMPTION`: initial2epochs, deterministic1155fit/100calibration TRAIN partition,
  noaugmentation, seededpredicted-only max16trainingROIs/image; no evaluationROIcap.
  Eachfit image once/epoch. SourceM/K previously saw some calibration images;
  new joint components never optimize on them. Nativeowner boxIoU>=.5 matching
  labels OCR/gate; unsupported/overlength/emptyGT is logged, excluded onlyfromCE,
  never evaluationGT. Keep unmatchedROIs asnegativegatecases. Existing radius3
  orientation-independent GapGT differs from proposal same-line horizontalGap.
- `ASSUMPTION`: lightCE averages first/refined; AR4teacherforced causalCE;
  gate targets actualrefined casefold correctness. AdamWgroups1e-5backbone/
  1e-4heads,decay1e-4,clip5,alllambda1,Dicesmooth1. These unspecified numerics
  and ROI supervision are predeclared, not inferred original-paper settings.
- `ASSUMPTION`:50TRAINimages fit scalar temperature, disjoint50select reject/
  keep/deepaccept thresholds by bboxcasefoldHmean then fewerdeepcalls; fixed
  finalepoch beforeTEST. Explicitrejections/emptyreadings notemitted, while
  allcareGT retained; allroutes saved. Alwayslight/deep retains emptycare
  proposals. Report full300/2204 and same298/2187excluding knownRGBduplicates,
  historicallyexploredTEST. AUROC undefined forsingleclass, neverinvented.
- `HARDWARE_DEVIATION`: actualGPU,FP32imagebatch1/trainingROIcap16/evalchunks16;
  CUDA jointupdates andinference, explicitCPUbilinearbackward/autogradtransfers
  fordeterminism. Measureallocated/reservedVRAM/shapes/modelset andOOM. No
  installation,mixedprecision,implicitCPUfallback,8GBorpaperlatencyclaim.
- `UNRESOLVED`: convergedtraining,SynthText/TextOCRscenepretraining,multiseeds,
  full officialpolygonevaluation,strictartifacts,FLOPs/edgequalification and
  accuracy superiority. A functional smoke is not a useful trained OCR claim.
- Alternatives: repeat frozenRGBPARSeq decoding as if it were newheads (rejected);
  preserve separateGapdecoder in claimedsharedbackbone (rejected); tuneTEST
  thresholds/forcepositiveAUROC (rejected). Keep priorruns/checkpoints immutable.
  Config, reproducible commands and all qualification results:
  `docs/ADASpot_JOINT.md`, `configs/approximate/adaspot_joint_v1.json`.


### ADASpot-JOINT-001 completed qualification and posthoc diagnostics

The fixed2epoch/1155fit-image run completed2310CUDAupdates and all300TESTimages.
Both realOCRheads and adaptive score0/2204exact words, while all7tracked branches
update and independent auditpasses900TESTimage-arm/196threshold cases. This is
an unsuccessful approximate accuracy result, not a reproduced AdaSpot benefit.
Gate has0positiveTESTlabels, undefinedAUROC; allcalibrationHmeans tie at0 and
predeclared tie picks always-light thresholds0/0/0, with0deep calls. TinyECE in
thisdegeneratecase does not establish useful discrimination. Original1311word
Gap+RGBPARSeq reference is preserved.

Additional explicit `ASSUMPTION`: after observing this fixed negative result,
perform a bounded100step fit-only single-word memorization diagnostic on the
firstsupportedmatched predictedROI in fixedfit order (TRAIN35, ATHENS). Freeze
backbone/gate, use actualcachedFgrids and optimize onlyOCRheads, seed202503,
AdamW1e-3/decay1e-4/clip5. Bothheads memorize by10steps; this is a Smoke-test
result only and does not change benchmarkweights/results or demonstrate visual
generalization. Parameters/config: adaspot_joint_overfit_check_v1.json.

Additional `REPOSITORY_SPECIFIED` verification: fresh actual calibrated selective
inference on all300TESTimages reproduces saved masks/boxes/routes/readings/scores
with300backbonecalls and0deepcalls. FP32gateprobability tolerance2e-5, exact
strings/geometry/counts. Forcedkeep/reject/escalate functional checks exercise
deepbranch separately without TESTpolicy selection. Missing scene pretraining,
recognition adaptation and joint-objective interference remain `UNRESOLVED`
causal hypotheses. Full details and commands: docs/ADASpot_JOINT.md.

## YOLO-FROZEN-CONSISTENCY-001 — image-quality routing without new network training

User-authorized tests,2026-10-21: scale/crop stability, predicted text coverage
and CLIP compatibility to keep images whose light-branch F1 is at least0.8.
Authority inspected: local2503.15639v2 sections3.3/3.5/Algorithm2; existing
YOLO640/960, PARSeq crop views, coverage, word metrics and routing code.

- `PAPER_SPECIFIED`: uncertain full images fall back to DeepSolo. The proposed
  consistency rules are external extensions, not Algorithm2 or strict results.
- `REPOSITORY_SPECIFIED`: audited125/300 paired640/960 features, original RGB
  crops, strict qualified PARSeq CPU FP32 source/checkpoint, TEST crop views,
  Text/Kernel coverage, CLIP/native reference scores and complete frozen ViTAE
  fallback. Preserve existing640 words and evaluation exactly.
- `ASSUMPTION`: exact-casefold/punctuation-retained one-to-one cross-scale
  matching at bboxIoU>=0.5, maximum cardinality then IoU. Crop probes use5%
  shrink/expansion; per-word exact agreement in both views and soft normalized
  edit agreement. Empty OCR forces fallback. Probes cannot correct final words.
- `ASSUMPTION`: thirteen predeclared confidence, scale, crop, coverage and
  geometric-mean/minimum score controls; no fitted classifier/weights. Read
  predicted probabilities/coverage only; GT never enters inference features.
- `ASSUMPTION`: quality label is per-image local word F1>=0.8. TRAIN80 alone
  selects a family by maximum accepted count meeting empirical80/90/95% good
  image fractions; ties use quality then declared order. Independently fit each
  threshold on TRAIN45 with at least5 acceptances. Freeze before new TEST
  consistency features; report all families. No qualifying threshold means
  explicit all-image fallback, not a substituted unsafe threshold.
- `ASSUMPTION`: GT-free10/20/30/40/50% batch rank budgets and transferred scalar
  coverage cuts are separate. Choose a family for each budget only on TRAIN80
  good-image count, ties by declared order. Targets and intervals are empirical;
  no formal conditional-risk guarantee or TEST-based winner selection.
- `HARDWARE_DEVIATION`: existing isolated environment, CPU FP32 PARSeq batch32
  and four threads to match TEST caches. Other networks/inference are reused
  with provenance; record actual host/GPU, shapes, state hash before/after,
  optimizer_steps0, CPU-null CUDA memory fields and all failures. No installs.
- `UNRESOLVED`: consistently wrong/missing words can remain stable; U-Net
  coverage can miss true text. Only45 calibration images, historical TEST
  inspection, upstream DeepSolo training and independent target8GB/paper
  metric artifacts limit generalization and strict claims. Per-image quality
  does not by itself optimize combined microF1 versus DeepSolo.

Protocol, reproducible commands and eventual results:
`docs/YOLO_TRAINING_FREE_CONSISTENCY_ROUTER.md`. Initial six focused tests and
genuine frozen-model smoke passed; full measurement and audit are pending.

Completed evidence,2026-10-21: all125TRAIN/300TEST;2,044 new CPU FP32 PARSeq
views and4,410 reused TEST views, unchanged model-state SHA and optimizer_steps0.
The predefined all-checks soft-crop variant retains48/60 good TEST images at20%
bypass (prior39/60), finalF1 79.9633% (prior79.8346%); at30% it retains67/90
(prior54/90), finalF1 79.3840% (prior79.0698%). Small F1 gains have paired95%
intervals crossing zero; no superiority claim. At50% bypass its finalF1 drops
to76.8910% despite91good images versus85 for the prior77.5954% gate.
Transferred90%-target all-checks thresholds retain61/63images, good48/50:
78.69/79.37% good-image precision, not the requested90%. Seven/eight clean
calibration cases do not establish reliability. The TRAIN-selected primary
family is native_clip and does not generalize (36/70good TEST); preserve that
failure rather than substituting a favorable TEST winner. All families/targets,
zero-bypass policies, source hashes and limitations are documented in
`docs/YOLO_TRAINING_FREE_CONSISTENCY_ROUTER.md`. Seventy-nine tests, genuine
smoke and independent425feature/6,454crop/676point audit pass; no omitted or
failed sample. Final-pipeline latency, new unseen validation and strict paper
protocol remain unresolved. No threshold/model was retuned from TEST labels.

### YOLO-CONTEXT-QUALITY-001 — Frozen context for per-image F1>=0.8 (2026-10-21)

- `PAPER_SPECIFIED`: inspected local §3.5/Algorithm2: image-only BLIP2 context,
  MPNet/fuzzy comparisons and whole-image DeepSolo fallback. New features are
  experimental extensions, not the paper scoring algorithm.
- `REPOSITORY_SPECIFIED`: audited425 original640 YOLO/PARSeq word sets, BLIP2
  captions, complete ViTAEv2-S+TextOCR branches, coverage/consistency controls,
  local word metric and disjoint TRAIN80/45 split.
- `ASSUMPTION`: quoted/narrow attributed caption evidence, casefold multiset
  recovery with punctuation/duplicates preserved. Absence is not contradiction;
  generated captions may hallucinate and are not exhaustive. Unquoted spans
  marked ambiguous. No annotation lexicon.
- `ASSUMPTION`: MPNet contrast vs3 unrelatedTRAIN80 captions; CLIP crop spelling
  margins with plain/own/foreign-caption prompts and lower20th percentile.
  Deterministic spelling mutants, no GT. Explicit over-context abstention,
  no silent word/caption truncation.
- `ASSUMPTION`: frozen FLAN-T5-base teacher-forced Yes/No support/coherence are
  text-only plausibility probes, with wrong-caption control. They cannot prove
  literal pixel correctness, localization completeness or F1.
- `ASSUMPTION`: no model training, fitted mixture weights, word corrections or
  filters. CDF and family selection onlyTRAIN80; thresholds onlyTRAIN45; freeze
  before newTEST inference. Primary predeclared base_clip_context; report all17
  families. Invalid empirical purity cut abstains on all images. Budget
  shortfalls explicit. No formal risk guarantee or untouched-test claim.
- `ASSUMPTION`: per-image F1>=.8 classification, accepted-image purity and final
  microF1/TN0 are distinct. Only120/300 goodTEST images imply maximum80% purity
  when accepting150. Thresholds80/90/95% refer to fraction of good images.
- `HARDWARE_DEVIATION`: actual PAPER GPU checked; sequential FP32 models in
  existing environment, no installs/downloads. State hashes before/after,
  optimizer_steps0, peak allocated/reserved memory, shapes/lengths and all
  OOM/runtime failures recorded. Historical8GB qualification is not this run.
- `UNRESOLVED`: missing-text completeness, caption hallucination,45-image
  calibration uncertainty, full latency and strict paper protocol. Better
  quality classification does not guarantee better microF1 than DeepSolo.

Protocol and source links: `docs/YOLO_CONTEXT_QUALITY_ROUTER.md`. Initial20
focused tests and genuine6-image smoke pass; full425 comparison/audit pending.
