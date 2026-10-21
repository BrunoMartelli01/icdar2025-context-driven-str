# Reproduction Specification

## 1. Purpose and authority

This document is the structured scientific specification for reproducing *A Lightweight Context-Driven Training-Free Network for Scene Text Segmentation and Recognition* (arXiv:2503.15639v2). The local paper at `papers/2503.15639v2.pdf` is the primary scientific source. Direct segmentation-protocol author clarification recorded in `docs/Author Clarifications and Reproduction Update.md` is primary reproduction evidence that can resolve intended reported behavior, while recalled/unconfirmed values remain assumptions. Existing source code and `README.md` are secondary evidence and do not override either source.

This milestone records evidence; it does not resolve contradictions or authorize implementation. Open questions are tracked in `docs/PAPER_AMBIGUITIES.md`.

## 2. Source classifications

Every parameter or behavior below has one of these classifications:

- `PAPER_SPECIFIED`: explicitly stated in paper prose, a figure, algorithm, footnote, or table.
- `REPOSITORY_SPECIFIED`: present in the existing repository or README but not established by the paper.
- `ASSUMPTION`: a necessary proposed choice not established by either source. No assumptions are adopted by this documentation milestone.
- `UNRESOLVED`: absent, contradictory, or insufficiently precise in the available evidence.

Where paper and repository values differ, both are recorded. A repository value must not enter the strict configuration until evidence promotes it to `PAPER_SPECIFIED` or it is explicitly accepted as an `ASSUMPTION`.

## 3. End-to-end data flow

| Item | Value or behavior | Source | Evidence or note |
|---|---|---|---|
| Input | RGB scene image, denoted `I ∈ R^(H×W×3)` | `PAPER_SPECIFIED` | Section 3.1. |
| Segmentation output | Binary text-foreground mask `M ∈ {0,1}^(H×W)` | `PAPER_SPECIFIED` | Sections 3.2–3.3 and Algorithm 1. |
| Localization output | Padded rectangular crops derived from mask components | `PAPER_SPECIFIED` | Section 3.3 and Algorithm 1. |
| `T1` | Recognizer output from the source/full image | `PAPER_SPECIFIED` | Section 3.5 and Algorithm 2. |
| `T2` | BLIP-2 scene description from the source image | `PAPER_SPECIFIED` | Sections 3.4–3.5. |
| `T3` | Recognizer output from a localized crop | `PAPER_SPECIFIED` | Section 3.5 and Algorithm 2. |
| Decision | Context score selects a lightweight prediction when confident; otherwise invoke DeepSolo | `PAPER_SPECIFIED` | Figure 2, Section 3.5, Algorithm 2. |
| Multiple-crop reduction to one `T3` | Existing runner evaluates all crops and retains maximum `C3` | `REPOSITORY_SPECIFIED` | `pipeline/runner.py`. Paper reduction rule is `UNRESOLVED`. |

## 4. Attention-Gated U-Net segmentation architecture

### 4.1 Paper specification

| Parameter or behavior | Value | Source | Evidence or note |
|---|---|---|---|
| Architecture family | U-Net modified for text segmentation | `PAPER_SPECIFIED` | Section 3.2. |
| Encoder | Pretrained ResNet-50 | `PAPER_SPECIFIED` | Section 3.2. |
| Encoder feature shown at input `256×256` | 64 channels at `128×128` | `PAPER_SPECIFIED` | Figure 3. |
| Next encoder feature | 256 channels at `64×64` | `PAPER_SPECIFIED` | Figure 3. |
| Next encoder feature | 512 channels at `32×32` | `PAPER_SPECIFIED` | Figure 3. |
| Next encoder feature | 1024 channels at `16×16` | `PAPER_SPECIFIED` | Figure 3. |
| Deepest encoder feature | 2048 channels at `8×8` | `PAPER_SPECIFIED` | Figure 3. |
| Skip behavior | Attention gates inserted into each skip path | `PAPER_SPECIFIED` | Section 3.2. |
| Attention inputs | Encoder feature `x` and decoder gating signal `g` | `PAPER_SPECIFIED` | Figure 3B. |
| Bottleneck operations | Convolution followed by BatchNorm and activation | `PAPER_SPECIFIED` | Figure 3 and Section 3.2. |
| Bottleneck purpose | Compress deepest representation and reduce channel dimensionality | `PAPER_SPECIFIED` | Section 3.2. |
| First decoder feature shown | 512 channels at `16×16` | `PAPER_SPECIFIED` | Figure 3; exact relation to the bottleneck is `UNRESOLVED`. |
| Subsequent decoder features shown | 256 at `32×32`, 128 at `64×64`, 64 at `128×128` | `PAPER_SPECIFIED` | Figure 3. |
| Output | Text foreground mask | `PAPER_SPECIFIED` | Figure 3 and Section 3.2. |
| Exact decoder block operations | Not completely stated | `UNRESOLVED` | Kernel sizes, convolution count, upsampling method, and post-concatenation layout are absent. |
| Exact attention-gate equations/channels | Not stated | `UNRESOLVED` | Paper cites attention gates but gives no complete implementation. |
| ResNet-50 weight revision | Not stated | `UNRESOLVED` | No ImageNet weight version or hash. |
| Encoder frozen during U-Net training | Not stated | `UNRESOLVED` | “All other modules remain frozen” may exclude or include the pretrained encoder; wording is insufficient. |

### 4.2 Existing repository choices, not strict evidence

| Parameter or behavior | Repository value | Source | Strict status |
|---|---|---|---|
| ResNet weights | `ResNet50_Weights.IMAGENET1K_V2` with legacy fallback | `REPOSITORY_SPECIFIED` | Not established by paper. |
| Decoder channels | `(1024, 512, 256, 128, 64)` | `REPOSITORY_SPECIFIED` | Conflicts with Figure 3’s apparent `512,256,128,64` path. |
| Bottleneck | Two `2048→2048` 3×3 Conv-BN-ReLU layers | `REPOSITORY_SPECIFIED` | Conflicts with paper’s stated channel reduction. |
| Upsampling | Nearest-neighbor ×2 plus 1×1 convolution | `REPOSITORY_SPECIFIED` | Paper method is `UNRESOLVED`. |
| Decoder ConvBlock | Two 3×3 Conv-BN-ReLU layers | `REPOSITORY_SPECIFIED` | Paper method is `UNRESOLVED`. |
| Output head | 1×1 convolution to one logit channel | `REPOSITORY_SPECIFIED` | Compatible in intent, not explicitly specified. |
| Encoder default | Trainable (`freeze_encoder=False`) | `REPOSITORY_SPECIFIED` | Paper behavior is `UNRESOLVED`. |
| Checkpoint loading | `strict=False` | `REPOSITORY_SPECIFIED` | Prohibited for strict work unless later justified and fully reported. |
| Current implementation viability | Final skip has 64 channels while its attention map has 128 channels | `REPOSITORY_SPECIFIED` | Known blocking defect; see ambiguity A-001. |

### 4.4 Author-clarified reported architecture (2026-08-29)

Direct author clarification establishes that the reported network ends at
`64@128x128`, produces its segmentation logits there, and computes loss there. The
later decoder-only upsampling to 256x256 belongs only to post-report localization code.
The attention coefficient is additive and scalar: projected encoder and gate features
are summed, passed through ReLU, mapped by a 1x1 convolution to one channel, sigmoid
activated, broadcast, and multiplied with the original-width encoder skip. The
ResNet-50 encoder remains frozen, including BatchNorm running state.

The bottleneck must compress 2048 channels, but `2048->1024` is only
author-recalled and unconfirmed. The runnable
`author-clarified-ag-unet-2048x1024-v1` branch therefore classifies that width, the
single 3x3 bottleneck implementation, attention intermediate widths, and unresolved
decoder operations as `ASSUMPTION`; it is an approximate reproduction. Historical
`approximate-ag-unet-figure3-v1` checkpoints/results and 256/512 input-resolution
ablations retain their original identities and protocol labels.

### 4.3 M8 approximate architecture contract

M8 does not repair or promote the repository `AdvancedUNet`. It freezes the separate
figure-derived branch as `approximate-ag-unet-figure3-v1`, contract schema version 1.
Its machine-readable source of truth is
[`provenance/AG_UNET_ARCHITECTURE_M8.json`](../provenance/AG_UNET_ARCHITECTURE_M8.json).

| Contract item | Frozen approximate value | Classification | Strict implication |
|---|---|---|---|
| Total/trainable parameters | `48,454,733` / `48,454,733` for the selected unfrozen-encoder branch | `ASSUMPTION` implementation property | Not a paper parameter count |
| Registered unused encoder tail | torchvision `encoder.avgpool` and `encoder.fc` remain in parameters/state but are not executed | `REPOSITORY_SPECIFIED` retained behavior | Not described by the paper; changing it requires a new identity |
| State schema | 456 tensors; SHA-256 `bc146d1e3742a63f9b64ad8aded6ff92fbde656f8386e1a88d0e8a109ab04b5f` over sorted names/shapes/dtypes | `REPOSITORY_SPECIFIED` qualification contract | No author checkpoint exists to validate it |
| Input | Positive floating-point BCHW matching model dtype/device, 3 channels, H/W each ≥32 and divisible by 32; selected branch float32 | `ASSUMPTION` | Paper specifies only the configured 256×256 size |
| Output | One-channel logits at `H/2×W/2`; 128×128 for canonical input | `ASSUMPTION` (`FIGURE_INFERRED`) | A-005 remains unresolved |
| Checkpoint load | Identity/schema/fingerprint plus exact keys/shapes/dtypes; strict only | `REPOSITORY_SPECIFIED` safety policy | Does not create paper compatibility |
| Selected pretrained candidate | torchvision `IMAGENET1K_V2` | `REPOSITORY_SPECIFIED` | A-006 remains unresolved; tests may explicitly use no downloaded weights |

The four canonical attention paths use compatible gate/skip/coefficient tensors:
`512/1024/1@16²`, `256/512/1@32²`, `128/256/1@64²`, and
`64/64/1@128²`. A-001 through A-006 remain strict blockers.

## 5. Segmentation training

| Parameter or behavior | Value | Source | Evidence or note |
|---|---|---|---|
| Trained component | Attention-aided U-Net | `PAPER_SPECIFIED` | Section 4.1 implementation details. “Training-free” does not mean the segmenter is untrained. |
| Optimizer | Adam | `PAPER_SPECIFIED` | Section 4.1. |
| Learning rate | `1×10^-4` | `PAPER_SPECIFIED` | Section 4.1. |
| Weight decay | `0.01` | `PAPER_SPECIFIED` | Section 4.1. |
| Batch size | 8 | `PAPER_SPECIFIED` | Section 4.1. |
| Loss | Binary cross-entropy (BCE) | `PAPER_SPECIFIED` | Section 4.1. |
| Image training size | `256×256` | `PAPER_SPECIFIED` | Section 4.1. |
| Mask training size | `128×128` | `PAPER_SPECIFIED` | Section 4.1. |
| Augmentation | Horizontal flips and random rotations | `PAPER_SPECIFIED` | Section 4.1. |
| Other modules | Frozen | `PAPER_SPECIFIED` | Section 4.1; scope relative to ResNet encoder is `UNRESOLVED`. |
| Inference batch size | 1 | `PAPER_SPECIFIED` | Chosen to accommodate frozen BLIP-2. |
| Paper training GPU | Single “RTX 2080 Ti (15 GB)” | `PAPER_SPECIFIED` | Paper statement; hardware description is internally suspect and not the reproduction target. |
| Epoch count / stopping | 100 epochs | `PAPER_SPECIFIED` via author clarification | No early-stopping protocol; full run only. |
| Learning-rate schedule | Not stated | `UNRESOLVED` | Material training parameter. |
| Adam betas/epsilon | Not stated | `UNRESOLVED` | Library defaults must not be silently treated as paper values. |
| Weight initialization beyond pretrained encoder | Not stated | `UNRESOLVED` | Material to convergence. |
| Dataset training regime | Separate per-dataset versus combined training not stated | `UNRESOLVED` | Needed to interpret Table 3. |
| Validation split/model selection | No validation holdout; best training-set foreground IoU | `PAPER_SPECIFIED` via author clarification | Exact IoU aggregation and prediction threshold remain `UNRESOLVED`. |
| Random seed/repetitions | Not stated | `UNRESOLVED` | Needed for repeatability and uncertainty. |
| Flip probability | `0.5` | `PAPER_SPECIFIED` via author clarification | Rotation details remain incomplete. |
| Rotation range/probability/interpolation | Not stated | `UNRESOLVED` | Augmentation incomplete. |
| BCE reduction / logits versus probabilities | Not stated | `UNRESOLVED` | Loss contract incomplete. |
| Mixed precision | Not stated | `UNRESOLVED` | Any use on RTX 3070 is potentially a `HARDWARE_DEVIATION` under `AGENTS.md`. |

### 5.2 Author-clarified ICDAR13 training protocol

All 229 official training images are training samples and none is held out. The
pretrained ResNet-50 encoder and its BatchNorm state remain frozen; only bottleneck,
attention, decoder, and output head parameters train for 100 epochs at batch size 8.
Don’t Care colors identified by `#`-prefixed paired Task 2.2 TXT records produce a
per-pixel validity mask and zero BCE gradient. The full-resolution binary target is
resized to 128x128 with bilinear interpolation and re-binarized at 0.5. Checkpoints are
ranked by foreground IoU measured over the unaugmented 229-image training population.

The runnable APPROXIMATE config uses +/-15 degree rotation with probability 0.5,
nearest validity-mask resize, per-image macro training IoU, threshold 0.5, ImageNet V2
weights/normalization, Adam defaults, no schedule, and seed 202503 as explicit
`ASSUMPTION` or `REPOSITORY_SPECIFIED` choices. No result from the earlier 206/23 M9
protocol is relabeled as following this clarification.

The optional post-hoc checkpoint trajectory reuses the official 233-image test set as a
validation-like diagnostic. It is deliberately labeled `leaky test-as-validation diagnostic`:
the test trajectory may be inspected, but it cannot select a checkpoint, tune a scientific
choice, or be reported as an unbiased Table 3 evaluation.

### 5.1 M9 APPROXIMATE ICDAR13 split contract

The acquired source population is the 229 physically verified ICDAR13 Task 2.2 training
image/BMP pairs. RRC remains the official upstream identity; the exact local archives were
obtained from separately recorded keras-ocr v0.8.4 mirror URLs and are `SECONDARY`
acquisition evidence, not direct RRC downloads.

`icdar13-task2.2-sha256-ranked-holdout-v1` is an `ASSUMPTION`: canonical numeric IDs are
ranked by raw bytes of `SHA256(UTF8("202503:" + sample_id))`, with numeric ID as fallback.
The first 23 ranks are validation and the remaining 206 are training; both manifests are
serialized in numeric-ID order. Validation is a metrics-only held-out subset of the official
training source and cannot choose a best checkpoint. The policy fingerprint is
`3a7c3d4226e735b2e4b57fef5f8c84fa9522ad1b60852d86229a975ff07d6124`.

For the M9 APPROXIMATE training path, Task 2.2 RGB `(255,255,255)` maps exactly to
background 0 and every other RGB value maps to foreground 1. There is no tolerance,
threshold, morphology, palette assumption, or color remapping. This white/non-white rule
follows upstream Task 2.2 semantics. Treating Don't Care regions as foreground is an
`ASSUMPTION`: TXT annotations are audit-only and do not change BCE targets; no ignore mask,
masked BCE, or `ignore_index` is introduced. The policy fingerprint is
`ca318685a8fea2a0c162d80986acb88d60ec950fdec440652ff4d57bf4c0adf3`. It makes the
APPROXIMATE manifests training-ready but does not resolve DS-003 for STRICT or define
evaluation semantics.

## 6. Segmentation preprocessing and postprocessing

| Parameter or behavior | Value | Source | Evidence or note |
|---|---|---|---|
| Image resize | Resize to `256×256` | `PAPER_SPECIFIED` | Section 4.1. |
| Mask resize | Downsample to `128×128` | `PAPER_SPECIFIED` | Section 4.1. |
| RGB normalization | Not stated | `UNRESOLVED` | Especially material for a pretrained ResNet-50. |
| Aspect-ratio handling | Not stated | `UNRESOLVED` | Stretch, pad, crop, or letterbox are not identified. |
| Image interpolation | Not stated | `UNRESOLVED` | Training and inference can differ materially. |
| Training-target interpolation | Bilinear to `128x128`, then threshold `0.5` | `PAPER_SPECIFIED` via author clarification | Validity-mask and rotation interpolation remain `UNRESOLVED`. |
| Mask binarization threshold | Not stated | `UNRESOLVED` | Required for localization and metrics. |
| Repository input range | RGB converted to float `[0,1]`, no mean/std normalization | `REPOSITORY_SPECIFIED` | `models/segmenter.py`. |
| Repository resize | Direct resize without letterboxing; antialiasing when torchvision is available | `REPOSITORY_SPECIFIED` | `models/segmenter.py`. |
| Repository threshold | `0.5` | `REPOSITORY_SPECIFIED` | `models/segmenter.py` and README. |
| Repository inference output | Logits at `256×256`, sigmoid, bilinear resize to original image, then threshold | `REPOSITORY_SPECIFIED` | Does not explain paper’s `128×128` training mask. |

## 7. Block-level localization

| Parameter or behavior | Value | Source | Evidence or note |
|---|---|---|---|
| Input | Binary mask `M` and original image `I` | `PAPER_SPECIFIED` | Algorithm 1. |
| Component extraction | `FindContours(M)` / connected-component analysis | `PAPER_SPECIFIED` | Section 3.3 and Algorithm 1. |
| Area filter | Keep `Ci` when component area `A(Ci) ≥ Amin` | `PAPER_SPECIFIED` | Algorithm 1. |
| Bounding box | Extreme component coordinates `(xmin,ymin,xmax,ymax)` | `PAPER_SPECIFIED` | Section 3.3. |
| Padding | Subtract/add `p` on all sides | `PAPER_SPECIFIED` | Section 3.3 and Algorithm 1. |
| Bounds | Clamp horizontal coordinates to `[0,W]` and vertical coordinates to `[0,H]` | `PAPER_SPECIFIED` | Section 3.3 and Algorithm 1. |
| Sort | Component area descending | `PAPER_SPECIFIED` | Algorithm 1. |
| Maximum blocks | `nmax=10` | `PAPER_SPECIFIED` | Section 3.3 and Algorithm 1. |
| Crop source | Original image using the padded box | `PAPER_SPECIFIED` | Algorithm 1. |
| `Amin` numeric value | Not stated | `UNRESOLVED` | Repository uses 150. |
| `p` numeric value | Not stated | `UNRESOLVED` | Repository uses 6 pixels. |
| Contour connectivity/retrieval mode | Not stated | `UNRESOLVED` | Repository uses OpenCV external contours. |
| M7 area filter | Foreground-pixel count for each connected component | `PAPER_SPECIFIED` | Preserves `A(Ci)` independently from bbox area. |
| M7 sort key | Descending foreground-pixel component area | `PAPER_SPECIFIED` | Equal-area behavior remains separate below. |
| M7 approximate connectivity | 8-connectivity, explicitly supplied | `ASSUMPTION` | Retains M4-SEG-004; L-003 remains unresolved. |
| M7 approximate equal-area tie | Top-to-bottom, then left-to-right, then component label only as a final total-order fallback | `ASSUMPTION` | Retains the M4 spatial policy and makes its last-resort ordering explicit. |
| M7 crop-coordinate convention | Integer `(xmin,ymin,xmax,ymax)` with exclusive maxima | `ASSUMPTION` | Matches NumPy/OpenCV slicing and permits clamping maxima to `W/H`; L-005 remains unresolved. |
| Historical repository area filter | Bounding rectangle area `w×h` | `REPOSITORY_SPECIFIED` | Conflicts with component area and is not used by the M7 path. |
| Historical repository sort key | Padded bounding-box area | `REPOSITORY_SPECIFIED` | Conflicts with component-area sorting and is not used by the M7 path. |
| Repository contour mode | `RETR_EXTERNAL`, `CHAIN_APPROX_SIMPLE` | `REPOSITORY_SPECIFIED` | Not established by paper. |

## 8. Recognition

| Parameter or behavior | Value | Source | Evidence or note |
|---|---|---|---|
| Strict recognizer | PARSeq-S (small variant) | `PAPER_SPECIFIED` | Section 3.5. |
| `T1` input | Source/full image | `PAPER_SPECIFIED` | Section 3.5. |
| `T3` input | Localized cropped text instance | `PAPER_SPECIFIED` | Section 3.5. |
| PARSeq checkpoint/revision | Not stated | `UNRESOLVED` | Exact released model is unidentified. |
| PARSeq pretraining data | Not stated for selected checkpoint | `UNRESOLVED` | Paper only generally notes synthetic and real data for compared models. |
| Charset/case/punctuation | Not stated | `UNRESOLVED` | Affects decoded strings and accuracy. |
| Input transform/size | Not stated | `UNRESOLVED` | Must match selected checkpoint. |
| Decoding | Not stated | `UNRESOLVED` | AR/NAR, refinement, EOS, confidence, and maximum length are absent. |
| Multiple crops | Selection/aggregation rule not stated | `UNRESOLVED` | Paper defines singular `T3` despite up to 10 crops. |
| Multiple text in full image | Expected behavior of single-instance PARSeq-S not explained | `UNRESOLVED` | Material to `T1`. |
| Repository default recognizer | TrOCR (`microsoft/trocr-base-printed`) | `REPOSITORY_SPECIFIED` | Demo/debug only; forbidden as a strict substitute. |
| Repository PARSeq adapter | `baudm/parseq` through a generic image-classification API | `REPOSITORY_SPECIFIED` | Explicit stub, not valid sequence decoding. |
| Repository crop rule | Evaluate all crops and retain maximum `C3` | `REPOSITORY_SPECIFIED` | Not paper-established. |

## 9. BLIP-2 scene-caption generation

| Parameter or behavior | Value | Source | Evidence or note |
|---|---|---|---|
| Caption model family | Pretrained BLIP-2 | `PAPER_SPECIFIED` | Section 3.4. |
| Caption input | Full/source scene image | `PAPER_SPECIFIED` | Figures 2 and 4. |
| Prompt focus | Visible text, setting/environment, text-bearing objects, placement/arrangement, contextual cues, and possible occlusions | `PAPER_SPECIFIED` | Section 3.4 and Figure 4. |
| Verbosity tiers | Short, medium, long | `PAPER_SPECIFIED` | Section 3.4. |
| Short limits | `min_length=20`, `max_length=40` | `PAPER_SPECIFIED` | Section 3.4 footnote. |
| Medium limits | `min_length=40`, `max_length=80` | `PAPER_SPECIFIED` | Section 3.4 footnote. |
| Long limits | `min_length=80`, `max_length=120` | `PAPER_SPECIFIED` | Section 3.4 footnote. |
| Selected tier | Medium | `PAPER_SPECIFIED` | Qualitatively selected as best trade-off. |
| Caption limit description | Figure 4 calls medium descriptions “limited to 80 words” | `PAPER_SPECIFIED` | Conflicts with “token constraints”; unit is `UNRESOLVED`. |
| Backbone in experiments | “ViT-B backbone” | `PAPER_SPECIFIED` | Section 4.1 implementation details. Exact BLIP-2 model remains `UNRESOLVED`. |
| Approximate latency | `~120 ms/image` | `PAPER_SPECIFIED` | Paper hardware and measurement protocol are not sufficient to reproduce it. |
| Exact BLIP-2 checkpoint/revision | Not stated | `UNRESOLVED` | Required for strict captions. |
| Exact prompt text | Figure gives a detailed prompt ending with an ellipsis | `UNRESOLVED` | It may be illustrative or truncated. |
| Temperature values | Not stated | `UNRESOLVED` | Paper says fixed token/temperature constraints. |
| Other decoding values | Not stated | `UNRESOLVED` | Beam count, sampling, length penalty, seed, and stopping are absent. |
| Repository checkpoint | `Salesforce/blip2-flan-t5-xl`, unpinned revision | `REPOSITORY_SPECIFIED` | Not uniquely supported by paper and presents an 8 GB VRAM risk. |
| Repository limits | `min_new_tokens=40`, `max_new_tokens=80` | `REPOSITORY_SPECIFIED` | Not equivalent to words or necessarily to `min_length/max_length`. |
| Repository decoding | Three beams, length penalty 1.0, no explicit temperature | `REPOSITORY_SPECIFIED` | Paper values are `UNRESOLVED`. |
| Repository prompt | Shortened one-sentence prompt | `REPOSITORY_SPECIFIED` | Does not reproduce Figure 4 verbatim. |

## 10. Context scoring and selection

| Parameter or behavior | Value | Source | Evidence or note |
|---|---|---|---|
| Semantic model family | MPNet sentence embeddings | `PAPER_SPECIFIED` | Section 3.5. |
| `S1` | `cosine_similarity(embedding(T1), embedding(T2))` | `PAPER_SPECIFIED` | Figure 2 and Algorithm 2. |
| `S3` | `cosine_similarity(embedding(T3), embedding(T2))` | `PAPER_SPECIFIED` | Figure 2 and Algorithm 2. |
| Lexical metric | Levenshtein fuzzy string matching ratio | `PAPER_SPECIFIED` | Section 3.5. |
| Figure lexical terms | `L1=fuzz_ratio(T1,T2)/100`, `L3=fuzz_ratio(T3,T2)/100` | `PAPER_SPECIFIED` | Figure 2. |
| Semantic weight | `alpha=0.6` | `PAPER_SPECIFIED` | Section 3.5, Algorithm 2, Table 6. |
| Lexical weight | `beta=0.4` | `PAPER_SPECIFIED` | Section 3.5, Algorithm 2, Table 6. |
| Confidence threshold | `tau=0.8` | `PAPER_SPECIFIED` | Section 3.5, Algorithm 2, Table 6. |
| Confidence form | `C=alpha*S + beta*L` | `PAPER_SPECIFIED` | Figure 2, prose, Algorithm 2. |
| Candidate choice | Higher semantic similarity selects `T1` or `T3` | `PAPER_SPECIFIED` | Section 3.5 and Algorithm 2 line 8. |
| Aggregation of `S`, `L`, and candidate | Contradictory across paper locations | `UNRESOLVED` | Figure uses independent maxima; prose implies selected-candidate scores; Algorithm 2 leaves `S` and `T` undefined. |
| Exact MPNet checkpoint/revision | Not stated | `UNRESOLVED` | Repository assumes all-mpnet-base-v2. |
| Embedding pooling/normalization | Not stated | `UNRESOLVED` | Affects cosine values and thresholding. |
| Fuzzy implementation | Not stated | `UNRESOLVED` | Library, ratio variant, and edit-cost convention absent. |
| String preprocessing | Not stated | `UNRESOLVED` | Case, punctuation, Unicode, whitespace, token extraction, and empty strings absent. |
| Repository MPNet | `sentence-transformers/all-mpnet-base-v2`, unpinned | `REPOSITORY_SPECIFIED` | Plausible but not paper-established. |
| Repository fuzzy metric | RapidFuzz `fuzz.ratio` divided by 100 | `REPOSITORY_SPECIFIED` | Whole-string comparison without normalization. |
| Repository aggregation | Compute paired `C1`,`C3`; threshold `max(C1,C3)`; return semantic winner | `REPOSITORY_SPECIFIED` | One interpretation, not a resolution of the paper conflict. |

## 11. DeepSolo fallback

| Parameter or behavior | Value | Source | Evidence or note |
|---|---|---|---|
| Fallback family | Pretrained DeepSolo | `PAPER_SPECIFIED` | Section 3.5 and Algorithm 2. |
| Trigger | Confidence `C < 0.8` | `PAPER_SPECIFIED` | Section 3.5 and Algorithm 2. |
| Fallback input | Original image `I` | `PAPER_SPECIFIED` | Algorithm 2. |
| Evaluated backbones | ResNet-50 and ViTAEv2-S | `PAPER_SPECIFIED` | Tables 4–5. |
| Exact config/checkpoint by dataset | Not stated | `UNRESOLVED` | Multiple official DeepSolo checkpoints exist. |
| Lexicon/vocabulary protocol | Not stated | `UNRESOLVED` | Material to text spotting results. |
| Multiple output handling | Not stated | `UNRESOLVED` | DeepSolo returns spotted instances while algorithm expects one `Tfinal`. |
| Repository fallback | Stub catches all failures and returns `""` | `REPOSITORY_SPECIFIED` | Cannot reproduce paper results. |
| Repository config flow | Does not pass DeepSolo config/checkpoint paths from runner | `REPOSITORY_SPECIFIED` | Missing integration. |
| M6 approximate process boundary | Version-1 JSON request/response through the isolated qualified Docker runtime | `HARDWARE_DEVIATION` | Preserves failures, tracebacks, raw detections, text, confidence, geometry, latency, and runtime metadata; it is not the historical repository stub. |
| M6 qualified candidate | ResNet-50 Total-Text config/checkpoint identified in `docs/DEEPSOLO_RUNTIME.md` | `ASSUMPTION` | Real load and inference passed on the physical host, but no evidence maps this candidate to the paper's experiments. |

## 12. Datasets and splits

### 12.1 Segmentation

| Dataset | Training images | Test images | Source | Status note |
|---|---:|---:|---|---|
| TotalText | 1,255 | 300 | `PAPER_SPECIFIED` | Table 1. Annotation version/conversion is `UNRESOLVED`. |
| ICDAR13 FST | 229 | 233 | `PAPER_SPECIFIED` | Table 1. Exact later recognition split is contradictory. |
| COCO-TS | 43,686 | 10,000 | `PAPER_SPECIFIED` | Table 1, but conflicts with known COCO-TS dataset cardinality; protocol is `UNRESOLVED`. |

### 12.2 Recognition

| Dataset | Paper characterization | Source | Missing protocol |
|---|---|---|---|
| ICDAR13 FST | Focused, high-quality, primarily horizontal scene text | `PAPER_SPECIFIED` | Exact image/word split and matching protocol are `UNRESOLVED`. |
| ICDAR15 IST | Incidental, multi-oriented, blurred/occluded scene text; paper uses 500 images in ablations | `PAPER_SPECIFIED` | Care handling, word versus image evaluation, and transcription normalization are `UNRESOLVED`. |
| TotalText | Horizontal, multi-oriented, and curved text | `PAPER_SPECIFIED` | Exact annotation/evaluator and multilingual handling are `UNRESOLVED`. |

No dataset, loader, manifest, conversion, or checksum is present in the repository (`REPOSITORY_SPECIFIED` current state).

## 13. Metrics and measurement

| Metric | Definition/value | Source | Missing detail |
|---|---|---|---|
| Foreground IoU | `|P ∩ G| / |P ∪ G|` | `PAPER_SPECIFIED` | Per-image versus global aggregation, empty masks, threshold, and ignore pixels are `UNRESOLVED`. |
| Foreground F1 | `2|P ∩ G| / (|P| + |G|)` | `PAPER_SPECIFIED` | Aggregation and edge cases are `UNRESOLVED`. |
| Recognition | “Text-recognition accuracy” in percent | `PAPER_SPECIFIED` | Definition, unit of evaluation, matching, normalization, and care/lexicon protocol are `UNRESOLVED`. |
| Efficiency | FLOPs in `10^9` operations | `PAPER_SPECIFIED` | Input size, profiler, included branches, and conditional averaging are `UNRESOLVED`. |
| Caption latency | Approximately 120 ms/image | `PAPER_SPECIFIED` | Warmup, precision, timing boundaries, and repetitions are `UNRESOLVED`. |
| False positives in ablations | Count reported | `PAPER_SPECIFIED` | Formal definition is `UNRESOLVED`. |
| Fallback count | Number of images routed to DeepSolo | `PAPER_SPECIFIED` | Failure accounting is `UNRESOLVED`. |

The author-protocol evaluation now has an explicit native-logit diagnostic: the
one-channel prediction remains at `128x128`, while the full-resolution binary GT is
bilinearly resized to `128x128` and re-binarized at `0.5`. Per-image macro remains the
primary configured aggregation and global micro remains diagnostic; the paper does not
identify either aggregation or the evaluation resolution, so these remain `ASSUMPTION`.
The evaluator can apply a nearest-resized validity mask exactly as in training, but the
acquired 233-image test GT archive contains no paired TXT labels. RGB cannot substitute for
those labels because the same colors occur in both care and Don't Care training records.
Exact training-style test-time Don't Care exclusion therefore remains blocked. A separate
APPROXIMATE diagnostic composes the 233 official Task 1 care-box TXT files with the Task 2
color masks: a complete foreground color is ignored only when none of its pixels overlaps
any Task 1 box. The analogous training check recovered 589/608 known ignored colors with
zero care-color false positives (96.875% recall), so this conservative inference remains an
`ASSUMPTION` and is never substituted into the exact author protocol.

## 14. Ablation specifications

### 14.1 Table 6: context-score parameters

Dataset is 500 ICDAR15 test images (`PAPER_SPECIFIED`).

| `alpha` | `beta` | `tau` | Accuracy (%) | False positives | DeepSolo fallbacks | Source |
|---:|---:|---:|---:|---:|---:|---|
| 0.5 | 0.5 | 0.80 | 77.1 | 19 | 237 | `PAPER_SPECIFIED` |
| 0.6 | 0.4 | 0.80 | 78.2 | 13 | 228 | `PAPER_SPECIFIED` |
| 0.7 | 0.3 | 0.80 | 76.4 | 24 | 248 | `PAPER_SPECIFIED` |
| 0.6 | 0.4 | 0.75 | 78.0 | 20 | 203 | `PAPER_SPECIFIED` |
| 0.6 | 0.4 | 0.85 | 77.8 | 11 | 301 | `PAPER_SPECIFIED` |

Exact metric, strings, checkpoints, seeds, and scorer interpretation are `UNRESOLVED`.

### 14.2 Table 7: context-description sensitivity

Dataset is the ICDAR15 evaluation set used above (`PAPER_SPECIFIED`).

| Scenario | Context-based recognitions | Fallbacks | False-positive recognitions | Source |
|---|---:|---:|---:|---|
| No description | 163 | 337 | 23 | `PAPER_SPECIFIED` |
| Wrong description | 81 | 419 | 44 | `PAPER_SPECIFIED` |
| Correct description | 272 | 228 | 13 | `PAPER_SPECIFIED` |

How wrong descriptions are sampled, whether mappings are fixed, and how empty descriptions are embedded are `UNRESOLVED`.

### 14.3 Other analyses

| Analysis | Paper statement | Source | Missing detail |
|---|---|---|---|
| Caption verbosity | Medium descriptions qualitatively outperform short and long | `PAPER_SPECIFIED` | No quantitative table, dataset, or selection metric; `UNRESOLVED`. |
| Skipping detection | Block localization changes recognition only marginally versus detector-assisted boxes | `PAPER_SPECIFIED` | Detector, numerical comparison, and evaluation protocol are `UNRESOLVED`. |
| Backbone efficiency | Pipeline evaluated with ResNet-50 and ViTAEv2-S DeepSolo fallbacks | `PAPER_SPECIFIED` | Conditional FLOP calculation is `UNRESOLVED`. |

## 15. Reported target results

Reported values are targets to investigate, not evidence that the protocol is complete.

### 15.1 Table 3: segmentation fgIoU/F1 (%)

| Model | Encoder | ICDAR13 fgIoU | ICDAR13 F1 | TotalText fgIoU | TotalText F1 | COCO-TS fgIoU | COCO-TS F1 | Source |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Vanilla U-Net | Conv. layers | 68.00 | 78.50 | 72.50 | 80.65 | 67.38 | 62.60 | `PAPER_SPECIFIED` |
| DeepLabV3+ | Xception | 69.27 | 80.20 | 74.44 | 82.42 | 72.07 | 64.10 | `PAPER_SPECIFIED` |
| TextFormer | ResNet-50 | 72.27 | 83.80 | 81.56 | 88.70 | 73.20 | 74.50 | `PAPER_SPECIFIED` |
| AG-UNet (ours) | ResNet-50 | 73.14 | 85.30 | 75.24 | 85.44 | 75.35 | 77.50 | `PAPER_SPECIFIED` |

Some COCO-TS rows have F1 below fgIoU despite the stated set formulas; this is `UNRESOLVED`.

### 15.2 Table 4: pipeline efficiency targets

| Pipeline variant | FLOPs (G) | Source |
|---|---:|---|
| PARSeq + ResNet-50 DeepSolo | 35 | `PAPER_SPECIFIED` |
| PARSeq + ViTAEv2-S DeepSolo | 50 | `PAPER_SPECIFIED` |

### 15.3 Table 5: recognition accuracy (%)

| Method | ICDAR13 FST | ICDAR15 IST | TotalText | Source |
|---|---:|---:|---:|---|
| MaskTextSpotter | 84.1 | 70.3 | 71.2 | `PAPER_SPECIFIED` |
| MANGO | 88.7 | 72.9 | 72.9 | `PAPER_SPECIFIED` |
| SPTS | 88.5 | 73.5 | 74.2 | `PAPER_SPECIFIED` |
| TESTR | 86.4 | 73.3 | 73.3 | `PAPER_SPECIFIED` |
| Ours, DeepSolo ResNet-50 | 88.8 | 77.3 | 78.4 | `PAPER_SPECIFIED` |
| Ours, DeepSolo ViTAEv2-S | 89.7 | 78.2 | 79.8 | `PAPER_SPECIFIED` |

Reported routing targets:

- ICDAR13: 73% or 168/229 images avoid DeepSolo (`PAPER_SPECIFIED`), but use of 229 conflicts with Table 1’s 233-image test set (`UNRESOLVED`).
- ICDAR15: 54% or 272/500 images are handled by context recognition (`PAPER_SPECIFIED`).
- TotalText: no exact routing count is stated (`UNRESOLVED`).

## 16. Strict reproduction acceptance boundary

A result may be labeled strict only when all parameters that materially affect it are either `PAPER_SPECIFIED` or transparently resolved with new authoritative evidence. `REPOSITORY_SPECIFIED` defaults and future `ASSUMPTION` values must remain visible in metadata and normally imply an approximate reproduction. Any target-hardware adaptation must additionally be labeled according to the `HARDWARE_DEVIATION` rules in `AGENTS.md`.
