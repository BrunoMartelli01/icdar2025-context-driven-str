# Scientific Reproduction Plan

## 1. Scope and execution rules

This plan divides the reproduction into small, reviewable, testable milestones. It is designed for Ubuntu, one NVIDIA RTX 3070, and 8 GB VRAM. It does not authorize implementation by itself; each milestone begins only when explicitly requested.

Rules for every future milestone:

- Re-read `AGENTS.md`, the relevant paper section, `docs/REPRODUCTION_SPEC.md`, and relevant ambiguity entries before editing.
- Do not combine major scientific components unless explicitly requested.
- Keep strict configurations under `configs/strict/`, evidence-limited branches under `configs/approximate/`, and demo/debug configurations under `configs/demo/`.
- Do not commit datasets, weights, caches, secrets, or large outputs.
- Fail explicitly on missing paper artifacts; never fall back to demo models or defaults.
- Record all `ASSUMPTION`, `UNRESOLVED`, and `HARDWARE_DEVIATION` decisions.
- Run the listed tests plus `git diff --check` and inspect the full diff before completion.
- Commands below describe intended interfaces. They are not executed during documentation milestone 1 and may be refined before their milestone without changing scientific meaning.

## 2. Milestones

### Milestone 1 — Paper specification, ambiguity ledger, and implementation plan

- **Objective:** Establish the evidence-backed specification, preserve contradictions without resolving them, and sequence future work.
- **Files expected to change:** `docs/REPRODUCTION_SPEC.md`, `docs/PAPER_AMBIGUITIES.md`, `docs/REPRODUCTION_PLAN.md` only.
- **Tests:** Markdown structure review; verify all required source classifications and known ambiguity topics are present; inspect git status/diff; run whitespace validation.
- **Command that should run:** `git diff --check -- docs/REPRODUCTION_SPEC.md docs/PAPER_AMBIGUITIES.md docs/REPRODUCTION_PLAN.md`
- **Expected artifact/output:** Three reviewed documentation files with no implementation changes.
- **Paper section/table:** Sections 3–4, Algorithms 1–2, Figures 2–4, Tables 1–7.
- **Dependencies:** Initial repository audit and `AGENTS.md`.

### Milestone 2 — Resolve or freeze critical external evidence

- **Objective:** Obtain authoritative evidence for strict blockers before choosing architectures, checkpoints, splits, or metrics. Contact authors or inspect released immutable artifacts as necessary.
- **Files expected to change:** `docs/PAPER_AMBIGUITIES.md`, `docs/REPRODUCTION_SPEC.md`, proposed `provenance/SOURCES.yaml`, proposed `provenance/ARTIFACTS.yaml`; no model code.
- **Tests:** Schema validation; every artifact has source URL/location, revision, license note, checksum field, and availability state; every decision links to evidence and classification.
- **Command that should eventually run:** `python -m pytest tests/provenance/test_manifests.py`
- **Expected artifact/output:** Machine-readable evidence/artifact manifests and an updated blocker ledger. Unavailable evidence remains explicitly `UNRESOLVED`.
- **Paper section/table:** Figure 3; Sections 3.4–3.5 and 4.1; Tables 1, 3, and 5.
- **Dependencies:** Milestone 1.

Exit criterion: do not begin a strict implementation of AG-UNet, PARSeq-S, BLIP-2, DeepSolo, COCO-TS, or recognition evaluation while its identity/protocol remains unknowable. An explicitly requested approximate branch may proceed only with documented `ASSUMPTION` classifications.

### Milestone 3 — Reproducible isolated environment definitions

- **Objective:** Define compatible, pinned environments without installing them yet during planning; when authorized, create and validate a modern main environment and a separate legacy DeepSolo environment.
- **Files expected to change:** Proposed `environments/main/`, `environments/deepsolo/`, environment lock files, `docs/ENVIRONMENTS.md`, and environment validation tests/scripts.
- **Tests:** Python/CUDA/PyTorch compatibility; exact package-version report; import smoke tests; CUDA visibility; lock reproducibility; no dependency leakage between environments.
- **Command that should eventually run:** `bash scripts/verify_environments.sh`
- **Expected artifact/output:** Two pinned environment reports containing Python, CUDA, driver, PyTorch, torchvision, Transformers, Detectron2, and relevant library versions.
- **Paper section/table:** Section 4.1 implementation details; enables all experiments.
- **Dependencies:** Milestone 2 artifact choices, especially DeepSolo and BLIP-2 revisions.

The main and DeepSolo environments must not be merged merely for convenience. No model downloads occur implicitly in environment tests.

### Milestone 4 — Package and configuration foundation

- **Objective:** Repair package initialization and configuration loading while enforcing strict separation between paper and demo paths.
- **Files expected to change:** Package `__init__.py` files as needed, configuration loader modules, `configs/strict/`, `configs/approximate/`, `configs/demo/`, CLI entry points, and focused tests. Do not alter scientific model behavior.
- **Tests:** Namespace/import tests; OmegaConf or chosen config type tests; missing strict key/checkpoint/dataset must fail; paper config cannot inherit demo recognizers; resolved config round-trip.
- **Command that should eventually run:** `python -m pytest tests/config tests/test_package_imports.py`
- **Expected artifact/output:** Validated resolved-config files and successful CLI `--help`/configuration dry run without model loading.
- **Paper section/table:** Infrastructure for all sections; no direct paper result.
- **Dependencies:** Milestones 1–3.

### Milestone 5 — Dataset provenance and immutable split manifests

- **Objective:** Register official archives and exact image/annotation splits for ICDAR13 FST, ICDAR15 IST, TotalText, and COCO-TS without committing data.
- **Files expected to change:** Dataset manifest schemas, checksum tools, `data/manifests/` lightweight ID/checksum files, dataset documentation, and manifest tests. `.gitignore` changes only if explicitly authorized.
- **Tests:** Archive checksums; duplicate/leakage detection; expected counts; deterministic ordering; missing/corrupt sample failures; cross-check train/test disjointness.
- **Command that should eventually run:** `python -m reproduction.data.verify --manifest data/manifests/all.yaml --data-root "${DATA_ROOT}"`
- **Expected artifact/output:** Machine-readable verification report with exact counts, IDs, checksums, annotation versions, and unresolved count mismatches.
- **Paper section/table:** Section 4.1 and Tables 1–2.
- **Dependencies:** Milestone 2, especially COCO-TS and ICDAR13 split resolution; Milestone 3 for executable tooling.

COCO-TS verification must fail rather than normalize away the 14,690-versus-53,686 ambiguity. ICDAR13 recognition must use an explicit manifest rather than an inferred split.

### Milestone 6 — Annotation conversion and segmentation data loader

- **Objective:** Convert official segmentation annotations into deterministic binary masks and load image/mask pairs according to resolved paper preprocessing.
- **Files expected to change:** Segmentation dataset modules, conversion utilities, transform modules, fixtures, and tests; no generated full dataset committed.
- **Tests:** Polygon/mask rasterization; care/ignore handling; image-mask alignment; nearest/selected mask interpolation; deterministic augmentations; RGB/order/range; `256×256` image and resolved mask shape.
- **Command that should eventually run:** `python -m pytest tests/data/test_segmentation_datasets.py tests/data/test_segmentation_transforms.py`
- **Expected artifact/output:** Tiny synthetic fixture masks and a machine-readable dataset-loader smoke report for each benchmark.
- **Paper section/table:** Section 4.1 preprocessing and Table 1.
- **Dependencies:** Milestones 4–5 and resolution/classification of A-005, A-007, T-002, T-003, and DS-003.

### Milestone 7 — Paper-faithful connected-component localization

- **Objective:** Implement Algorithm 1 exactly where specified: component extraction, component-area filtering, padding, descending component-area ordering, bounds clamping, and top 10.
- **Files expected to change:** `localize/mask_to_blocks.py` or a replacement strict module, paper/demo configs, localization tests, and ambiguity/spec updates if evidence changes.
- **Tests:** Irregular contours distinguish component area from box area; padding/bounds; deterministic tie policy; top-10 cap; empty mask; small-component rejection; crop coordinates at image borders.
- **Command that should eventually run:** `python -m pytest tests/localize/test_algorithm1.py`
- **Expected artifact/output:** Golden JSON/PNG fixture outputs listing component area, unpadded/padded boxes, order, and retained crops.
- **Paper section/table:** Section 3.3 and Algorithm 1.
- **Dependencies:** Milestones 2, 4, and 6; numeric `Amin` and `p` must be resolved or explicitly classified as assumptions.

**M7 APPROXIMATE status:** The implementation uses foreground-pixel component area for
filtering and descending ordering, explicit exclusive `xyxy` boxes, clamped padding,
and the paper-fixed top-10 cap. It requires callers to supply `Amin`, `p`, connectivity,
and the equal-area tie policy. The retained M4 assumptions are 8-connectivity and
top-to-bottom then left-to-right ties (with component label as a final total-order
fallback). Golden fixtures retain rejected-component reasons and final crops. L-001,
L-003, L-005, and upstream mask threshold/resolution remain STRICT blockers; this is
not a STRICT M7 completion.

**M7 physical-host validation (2026-08-19):** The focused Algorithm 1 suite passed
12 tests in 0.31 seconds; the full repository suite passed 129 tests in 18.76 seconds;
and `git diff --check` passed. These are engineering validation results for the
APPROXIMATE implementation and do not resolve any STRICT blocker.

### Milestone 8 — AG-UNet architecture conformance

- **Objective:** Implement the resolved Attention-Gated U-Net/ResNet-50 architecture without training it.
- **Files expected to change:** `models/unet_adv.py` or a new strict architecture module, paper config, architecture documentation, and model tests.
- **Tests:** Exact parameter/tensor shape table; every attention skip; odd/even input shapes if permitted; forward/backward pass; deterministic initialization; output resolution; parameter count; checkpoint-key schema; strict load failures.
- **Command that should eventually run:** `python -m pytest tests/models/test_ag_unet_architecture.py`
- **Expected artifact/output:** Machine-readable architecture summary with layer names, channels, spatial dimensions, parameter count, and peak memory for a batch-1 forward smoke test.
- **Paper section/table:** Section 3.2 and Figure 3.
- **Dependencies:** Milestones 2–4; A-001 through A-006 must be resolved or the implementation explicitly labeled approximate.

No conventional Attention U-Net design may be silently substituted for missing author evidence.

**M8 sandbox status (2026-08-20):** The existing figure-derived branch is frozen as
`approximate-ag-unet-figure3-v1` with a deterministic JSON contract, exact
48,454,733-parameter/456-state-tensor schema, all four gate/skip shape contracts,
supported-size validation, CPU forward/backward and seeded-initialization checks, and
identity/schema/key/shape/dtype-strict checkpoint loading. `AdvancedUNet` remains unchanged.
A-001 through A-006 remain blocking, so this is APPROXIMATE only. The repository profiler
does not train and keeps hardware measurements outside the static architecture artifact.

**M8 sandbox validation (2026-08-20):** The focused model/profiler suite passed 17 tests;
the directly affected training/smoke/config suites passed 19 tests. The full suite passed
141 of 142 tests; its sole failure is the existing physical-environment assertion that
`torch.cuda.is_available()` must be true, while CUDA is hidden in this sandbox. This is
not recorded as a physical-host result and does not supply CUDA memory measurements.

**M8 profiler correction (2026-08-20):** Physical-host diagnosis isolated the initial
`reset_peak_memory_stats(cuda:0)` failure to lazy CUDA context/allocator initialization,
before AG-UNet construction. The profiler now performs a one-element allocation on the
selected device, deletes it, synchronizes and empties the cache, then resets peaks before
model loading. A CPU-only mock regression freezes this ordering; the physical profile is
physically validated. This engineering fix does not change the architecture contract.

**M8 test-isolation correction (2026-08-20):** Physical-host sequencing confirmed that
the deterministic-initialization architecture test leaked process-global deterministic
algorithm state into the later CUDA matmul smoke. The test now narrowly saves/restores
Python, NumPy, CPU Torch, and conditionally available CUDA RNG states, deterministic and
warn-only modes, plus cuDNN deterministic/benchmark flags in `try/finally`. Its original
same-seed/full-state equality assertion is preserved, and an explicit before/after assertion
guards against regression. Production seeding semantics are unchanged.

**M8 physical-host validation (2026-08-20):** The corrected batch-1 profiler passed on
`cuda:0` / NVIDIA GeForce RTX 3070 in float32 with repository-selected
`IMAGENET1K_V2`: `[1,3,256,256]→[1,1,128,128]`, `success=true`, `oom=false`,
264,071,680 peak allocated bytes, and 289,406,976 peak reserved bytes. The full physical
suite passed 142/142 tests in 30.68 seconds and `git diff --check` passed. No training was
performed. These results qualify the APPROXIMATE implementation only and leave A-001
through A-006 blocking for STRICT execution.

### Milestone 9 — Segmentation training completion

- **Existing foundation:** Macro-Milestones 4 and 8 already provide the checksum-aware
  dataset boundary, paired transforms, BCE/Adam contracts, deterministic one-epoch engine,
  versioned checkpoint payload and SHA-256, strict state loading, synthetic smoke, and the
  frozen `approximate-ag-unet-figure3-v1` architecture contract. These are not new work.
- **Objective:** Complete real scientific training using official data and the existing
  foundation: real multi-epoch orchestration, validation hooks, exact resume equivalence,
  epoch logging, CLI/run manifests, and production of the scientific checkpoint.
- **Files expected to change:** Official dataset/annotation integration, training
  orchestration and CLI, validation/logging utilities, run-manifest support, mode-isolated
  configs, and focused completion tests. Do not replace the qualified foundation.
- **Tests:** Official-manifest loading and annotation policy; deterministic multi-epoch
  behavior; validation isolation; uninterrupted-versus-resumed state/metric equality;
  checkpoint/config/environment capture; missing/corrupt artifacts fail; no validation
  leakage. Existing optimizer, BCE, transform, and checkpoint tests remain regressions.
- **STRICT command (expected to fail on recorded blockers):** `python -m segmentation.train --config configs/strict/segmentation_icdar13.yaml --data-root "${DATA_ROOT}" --output-root "${RUN_ROOT}" --device cuda`
- **APPROXIMATE real-data command:** `python -m segmentation.train --config configs/approximate/segmentation_icdar13.yaml --data-root "${DATA_ROOT}" --output-root "${RUN_ROOT}" --device cuda`
- **Expected artifact/output:** Scientific checkpoint plus SHA-256, fully resolved config,
  environment report, epoch/validation metrics, seed record, dataset/split manifest, and
  training manifest. Large checkpoints remain outside git.
- **Paper section/table:** Section 4.1 implementation details; prerequisite to Table 3.
- **Dependencies/blockers:** Existing M4 dataset/training/checkpoint foundation and M8
  architecture contract; official immutable datasets and strict annotation policies; A-001 through
  A-008, T-001 through T-004, DS-001, and DS-003 remain blocking for STRICT execution.

The APPROXIMATE full-model batch-size-8 CUDA training/VRAM feasibility check has already
passed on the physical RTX 3070 and must not be repeated as a new foundation task. The M9
scientific run must still record successful execution and peak allocated/reserved VRAM for
its actual model, weights, input, and data path. Preserve any OOM report. A lower batch,
gradient accumulation, AMP, or activation checkpointing requires a separate
`HARDWARE_DEVIATION` config and must not overwrite STRICT configuration.

**M9 implementation state:** The real-manifest CLI, deterministic multi-epoch loop,
validation isolation, exact resume contract, epoch logging, checkpoint/run/failure
manifests, and mode-isolated configs are implemented. The software path does not create or
infer paper-specified manifests. The short physical-host qualification below closed the M9
engineering gate, and the separately labeled full configured APPROXIMATE run is now
complete. The adopted split and training-mask policies remain assumptions. STRICT blockers
remain unchanged; M10 has not started.

**M9 ICDAR13 data state:** The 229-pair source was acquired from checksum-frozen public
mirror archives while RRC remains the separate official identity. The adopted APPROXIMATE
SHA-256 split assumption is frozen as 206 training/23 validation samples with relative paths,
per-file checksums, disjoint complete coverage, and deterministic manifests. M9 uses exact
RGB white as background and every other RGB value as foreground. Don't Care remains
foreground as an `ASSUMPTION`, TXT is audit-only, and BCE is unchanged. DS-003 remains a
STRICT blocker.

**M9 physical-host qualification (2026-08-20):** The real-data APPROXIMATE path passed on
an NVIDIA GeForce RTX 3070 8192 MiB (driver 595.84, torch 2.4.1+cu124) using CUDA float32,
48,454,733 parameters, batch 8, `[8,3,256,256]` images, and `[8,1,128,128]` masks.
`qualification-max-steps=1` executed one train step in each of the two configured epochs,
so the final `global_step` was 2. Train losses were `0.9355729818344116` and
`0.9342144727706909`; validation losses were `0.2795393764972687` and
`0.3452925980091095`. The run reported `success=true`, `oom=false`, 2,373,442,560 peak
allocated bytes, and 2,552,233,984 peak reserved bytes. The external final checkpoint has
SHA-256 `f06db536167ccd3d881adababaf00b2d6cefbb8cf6524bd22d303a908540e45e`.
The physical suite passed 171 tests in 38.56 seconds and `git diff --check` passed. This is a
short engineering qualification, not full scientific segmentation training, not a STRICT
claim, and not a Table 3 result. External outputs remain outside Git under
`$HOME/context-str-runs/m9-qualification-physical`.

**M9 full APPROXIMATE training (2026-08-21):** Distinct from the short qualification, the
complete configured run trained `approximate-ag-unet-figure3-v1` (48,454,733 parameters) on
206 samples and validated on 23 samples using an NVIDIA GeForce RTX 3070, CUDA float32,
batch 8, two epochs, and 26 steps per epoch (`global_step=52`). It retained Adam,
learning rate `1e-4`, weight decay `0.01`, no scheduler, and last-checkpoint selection.
Epoch 1 train/validation losses were `0.8646002301892031`/`0.8293485304583674`; epoch 2
losses were `0.7025007738650424`/`0.6749140412911124`. The run reported `success=true`,
`oom=false`, 2,373,967,360 peak allocated bytes, and 2,552,233,984 peak reserved bytes.
The external `checkpoints/epoch-0002.pt` SHA-256 is
`335a003ae00cf21d159151b9568e10eafde12d6ae524bf7505e1d62ae4c3bfe2`. It is the recorded
checkpoint input for a future APPROXIMATE M10 run, but no evaluation has started. This run
is not STRICT and is not a Table 3 result; all blockers and assumptions remain.

**Author-protocol update (2026-08-29):** A separate
`author-clarified-ag-unet-2048x1024-v1` APPROXIMATE branch now encodes the clarified
128x128 endpoint, scalar skip-preserving attention, frozen encoder/BatchNorm, all 229
training images with no validation holdout, 100 epochs, masked Don’t Care BCE, bilinear
target downsampling plus 0.5 re-binarization, and training-foreground-IoU checkpoint
selection. It does not alter or relabel the M8/M9 checkpoints or the prior 256/512 input
resolution ablations. The recalled `2048->1024` bottleneck, +/-15 degree rotation, and
remaining operation/metric choices are explicit assumptions. No 100-epoch run was
launched; physical CUDA/VRAM qualification is still required for this new identity.

### Milestone 10 — Segmentation evaluation + Table 3

- **Existing foundation:** Macro-Milestone 4 already provides inclusive thresholding,
  explicit empty-mask policies, valid-mask support, macro/micro foreground IoU and F1, and
  hand-calculated synthetic tests. These mechanics remain regression-tested, not new work.
- **Objective:** Run official complete-split segmentation inference and evaluation under a
  resolved, fully recorded protocol, then produce Table 3.
- **Files expected to change:** Scientific segmentation inference wrapper, evaluator and
  CLI, official benchmark configs, per-image/result schemas, and completion tests.
- **Tests:** Exact official manifest counts; threshold and logit/mask resolution checks;
  resolved care/ignore and aggregation protocol; no omitted failures; deterministic rerun;
  hand-calculated metric regressions; per-image-to-aggregate traceability.
- **Command that should eventually run:** `python -m reproduction.segmentation.evaluate --config configs/strict/segmentation_table3.yaml --data-root "${DATA_ROOT}" --checkpoint-root "${CHECKPOINT_ROOT}" --output-root "${RUN_ROOT}"`
- **Expected artifact/output:** Per-image JSONL/CSV, aggregate Table 3 JSON/CSV, masks
  outside git, failed-sample records and counts, resolved configs, dataset/checkpoint hashes,
  and provenance.
- **Paper section/table:** Section 4.2 and Table 3.
- **Dependencies/blockers:** Successful M9 scientific checkpoint plus official complete
  manifests and annotation policies. A-005/A-008, DS-001/DS-003, E-002, and E-003 remain
  explicit; unresolved metric aggregation or split semantics require the
  `approximate reproduction` label and prohibit a STRICT Table 3 claim.
- **Implemented boundary:** The M10 APPROXIMATE protocol is frozen in
  `configs/approximate/segmentation_table3_icdar13.yaml`; the 233-sample checksum manifest,
  batch-1 evaluator, strict checkpoint identity/hash checks, retained failures, and
  deterministic per-image/Table-3 JSON/CSV outputs are ready for physical CUDA validation.
  No STRICT or paper-exact protocol claim is made.
- **Author native-128 diagnostic:** The separate author evaluation config now compares
  native `128x128` predictions against bilinear-then-0.5 GT at the same resolution and
  records macro plus micro metrics. Training-style validity masking is implemented, but
  cannot be enabled for the frozen test artifact because its 233 samples have no paired
  TXT Don't Care annotations. No color-based ignore inference is permitted.

### Milestone 11 — Recognition/scoring completion + real E2E + Table 5

Sub-gates execute sequentially as A, B, then C and remain independently testable. Failure
in A blocks B and C; failure in B blocks C. Passing a later gate cannot waive an earlier
failure or promote APPROXIMATE prerequisites to STRICT.

#### Existing APPROXIMATE prerequisites — status only

- **PARSeq-S:** Macro-Milestone 5 already qualified the official v1.0.0 checkpoint/runtime
  candidate (`parseq-bb5792a6.pt`, SHA-256 `bb5792a...cd00`, runtime commit
  `1902db043c...`), strict state-dict loading, official transform, real sequence decoding,
  token metadata, and behavioral samples. TrOCR remains demo/debug only. R-001, R-003, and
  L-004 keep the paper-exact identity and full-image/crop semantics blocked for STRICT work.
- **BLIP-2:** Macro-Milestone 5 already qualified the pinned repository candidate and its
  prompt/generation metadata at revision `0eb0d3b...`. Physical RTX 3070 profiling recorded
  the FP16 single-device OOM at 7,171,370,496/7,656,701,952 allocated/reserved bytes and the
  successful unquantized FP16 CPU-offload path at
  6,938,498,560/7,220,494,336 allocated/reserved bytes and 10,865,696,768 bytes peak CPU
  RSS. C-001 through C-005 remain STRICT blockers; offload is an APPROXIMATE
  `HARDWARE_DEVIATION`, not paper evidence.
  On 2026-09-03 the RTX-3090 host additionally passed unquantized FP16 single-device
  inference at 9,035,619,840/9,726,590,976 peak allocated/reserved GPU bytes. This
  enables the selected host smoke lifecycle but remains an APPROXIMATE hardware deviation.
- **DeepSolo:** Macro-Milestone 6 already qualified the Docker-isolated ResNet-50 Total-Text
  candidate `ART-DEEPSOLO-R50-TOTALTEXT-CANDIDATE`, versioned JSON boundary,
  source/config/checkpoint hashes, raw multi-output retention, explicit reduction assumptions,
  failure propagation, and real physical-host CUDA inference. D-001/D-002, ViTAEv2-S and
  benchmark-specific selections, and evaluator semantics remain STRICT blockers. Additional
  qualification is evidence-triggered corrective work, not a duplicated future milestone.

**M6 status (2026-08-19):** The Docker-isolated ResNet-50 Total-Text APPROXIMATE
candidate passed source/config/checkpoint load and real physical-host inference. The
versioned JSON boundary passed through the staged repository worker, and explicit
reductions are implemented. ViTAEv2-S, other
dataset candidates, paper-exact selection, and evaluator semantics remain unresolved;
this is not a STRICT DeepSolo completion.

Missing or mismatched artifacts, revisions, hashes, runtime identities, model loads, or
subprocess results must fail closed. Neither the qualified candidates nor sequential loading,
CPU offload, or process isolation may enter STRICT mode without authoritative support. On the
8 GB host, simultaneous residency must never be assumed; output equivalence and timing claims
remain separate.

#### Sub-gate A — MPNet/scoring qualification closure

- **Existing foundation:** Macro-Milestone 5 already provides the pinned MPNet adapter,
  cosine comparison, named fuzzy modes, all three explicit scoring interpretations,
  empty/tie behavior, exact below/equal/above-`tau` tests, and the literal whole-string
  `tau=0.8` feasibility diagnostic. Do not reimplement or silently tune them.
- **Objective:** Qualify the real pinned MPNet runtime/artifact and emit auditable golden
  scorer outputs while preserving every unresolved interpretation as a named boundary.
- **Files expected to change:** Qualification runner, golden artifact schema/fixtures,
  provenance-aware runtime validation, and focused qualification tests; scorer mechanics
  change only if separately justified by new evidence.
- **Tests:** Exact model revision and weight identity; preprocessing, pooling, truncation,
  normalization, precision, and device metadata; deterministic batch-versus-single outputs;
  golden cosine/fuzzy/decision records; existing strategy, threshold, tie, empty-input, and
  feasibility tests remain regressions.
- **Expected artifact/output:** Golden JSON containing input and artifact checksums, raw and
  normalized strings, embeddings/checksums where practical, `S1/S3/L1/L3`, all confidence
  components and named strategies, selected output, and fallback flag.
- **Paper section/table:** Figure 2, Section 3.5, Algorithm 2, and Table 6.
- **Dependencies/blockers:** Existing M5 scoring foundation and pinned candidate records.
  S-001 through S-005 remain STRICT blockers; an evidence-limited qualification must remain
  `approximate reproduction`. Never alter hidden preprocessing to force routing counts.

#### Sub-gate B — Complete real scientific E2E sample

- **Existing foundation:** Macro-Milestones 5–7 already provide the trace schema,
  segmentation/localization/recognition/caption/scoring/fallback orchestration, top-10
  localization, no-fallback and constructed-fallback coverage, explicit error propagation,
  and the one-shot lifecycle that releases segmentation, PARSeq, BLIP-2, and MPNet before
  the isolated DeepSolo process starts. These are not new work.

**Updated downstream status (2026-09-03):** No-fallback and constructed-fallback fixture paths pass,
and the one-shot lifecycle releases segmentation, PARSeq, BLIP-2, and MPNet before
the isolated DeepSolo process starts. The real Docker fallback boundary passed physical-host CUDA
validation on the earlier host. The frozen author-protocol epoch-98 checkpoint now has a
strict-loading downstream adapter and a real-data smoke CLI; the complete smoke still needs
the external PARSeq/BLIP-2/MPNet artifacts. Docker is administratively inaccessible on
host `paper`; the separate non-Docker Python 3.8 subprocess passed environment, strict-load,
staged-boundary, real RTX-3090 CUDA inference, and memory qualification on 2026-09-02. It is
a qualified APPROXIMATE `HARDWARE_DEVIATION`, not a paper-selected runtime. The complete
real-data path subsequently passed on checksum-verified ICDAR13 `img_1.jpg`: all stages
ran, ten crops were retained, low confidence invoked the isolated fallback, and all
component outputs, timings, memory/release events, and raw DeepSolo detections were
recorded. The result JSON SHA256 is
`72099c04b754ac4dc3dc6084dd974c9201365a9bf5b1980e6c9306589906b4b7`. This closes the
engineering smoke execution only; it is explicitly not a Table 5 result.
- **Objective:** Run the frozen segmentation checkpoint and one complete real sample
  through every scientifically required component.
- **Files expected to change:** Trained-segmenter integration, scientific orchestration CLI
  and mode-isolated configs, real-sample manifest, and focused integration tests.
- **Tests:** Real checkpoint inference; no components/no crops/empty strings; exact top-10
  processing; all `T1/T2/T3` and score records; conditional fallback and DeepSolo failure
  propagation; deterministic trace; rejection of TrOCR/dummy/disabled fallback; provenance
  completeness; 8 GB lifecycle and peak-memory/OOM reporting.
- **Implemented smoke entry point:** `python -m scripts.run_table5_smoke --help` lists the
  required config, image, immutable artifact paths, and isolated DeepSolo transport.
- **Expected artifact/output:** One scientifically labeled per-image ledger containing mask
  references, components/crops, all recognizer and caption outputs, scores, routing/fallback
  decisions, final outputs, timings, memory, failures, and complete artifact provenance.
- **Paper section/table:** Figure 2, Algorithms 1–2, Sections 3.2–3.5.
- **Dependencies/blockers:** Successful M9 checkpoint and M11.A; M7 localization contract;
  the three qualified APPROXIMATE prerequisites above. All unresolved architecture,
  localization, PARSeq, caption, scoring, DeepSolo, and multiple-crop/output semantics remain
  STRICT blockers. Sequential loading/offload/process isolation must be visible and classified.

#### Sub-gate C — Recognition evaluator + Table 5

**Trace-collection status (2026-09-03):** A resumable, range-addressable ICDAR13 host
runner now verifies the frozen 233-image/1,095-word manifest, runs every sample in a
fresh main-environment subprocess, preserves masks/crops/logs/failure records, and writes
an atomically updated index. Its schema fixes `accuracy: null`, `table_5: false`, and the
blocking ambiguity IDs. This is evidence collection for evaluator development, not an
evaluator and not a Table 5 run. The first physical batch probe (IDs 3--5) completed all
three samples without runtime failures and verified the recorded hashes of three masks
and nineteen crops. All three samples nevertheless fell below `tau=0.8` and invoked
DeepSolo; the configured singular `highest_score` reduction discarded additional detected
strings in every sample. The IDs 6--10 extension also completed without failure and
verified another five masks and 43 crops; all five again invoked fallback. Across the
uniform batch records 3--10, the literal case-insensitive final-string-membership
diagnostic is 8/8 while case-insensitive one-to-one raw-DeepSolo/GT multiset overlap is
32/53 GT words. These intentionally different observations demonstrate the unresolved
denominator/output-mapping effect; neither is accuracy. Together with the independent
image-1/image-2 smokes, zero of the first ten exercised samples avoided fallback. A
fail-closed diagnostic CLI now verifies index, contract, result, and intermediate hashes,
keeps `accuracy: null`, and suppresses fractions if any recorded sample failed. Full-dataset
execution remains scientifically premature on this branch. Local author history identifies
a higher-fidelity `REPOSITORY_SPECIFIED` alternative: scorer commit `87b90db` plus runner
commit `2e8b544` score all crops, select the first maximum `C3`, threshold `max(C1,C3)`,
and then return the semantic winner. This is implemented in a separate APPROXIMATE config
with resolved SHA256
`3965fe5d17116ab4d6eded0b2b543661929ffeef13cb884bb70f7a8db721202d`. Its separate
physical IDs 1--10 qualification passed all 86 intermediate-file checks. It changed T3
on 7/10 samples and confidence on 5/10, but confidence remained at most
`0.26995353443669656`; fallback stayed 10/10 and all final outputs were identical to the
rank-1 branch. All recorded semantic scores are below `2/3`, so no bounded lexical
similarity could reach `tau=0.8` for these samples. Therefore full-population execution
remains premature until authoritative evidence resolves the score/caption/protocol
discrepancy. A separate all-pairs PARSeq/DeepSolo alignment report at SHA256
`93195baebca1106a6eb8ca78dce7cbd007e7471e58da95c4b89805dfa3bfa935`
finds that 66/76 crops overlap a DeepSolo envelope but only 31/54 DeepSolo detections
overlap any crop; positive crop coverage has median `0.9864153811332046`, while positive
detection coverage has median `0.2872761360324503`. Only 2/31 spatially best positive
pairs are case-fold exact. This `ASSUMPTION`-classified envelope diagnostic indicates
crop fragments and missing crop coverage rather than word-instance alignment. Before a
full run, a first-ten ground-truth-bridged crop sensitivity gate now associates frozen
ICDAR13 word boxes with every generated segmentation/PARSeq crop by geometry before text
inspection. Its 24 predeclared points keep paper-specified localization mechanics fixed
and vary only repository/assumption values. The baseline reproduced all 76 prior CUDA
crop boxes, pixels, and texts. Only 7/66 GT words reached IoU `0.5` at `150/6/8`, and no
tested variant improved that count. Padding 24 increased weak overlap/fuzzy counts but
reduced exact recognition within strong-IoU pairs, while changing `Amin` did not improve
GT-linked text. No point is selected. The next gate is therefore a joint GT bridge for
the frozen mask components and DeepSolo detections, followed by a representative-sample
confirmation; it must retain all pairings and keep accuracy null until E-001 is resolved.
The branch remains non-STRICT and cannot resolve A-008, L-004, R-003, S-001, S-003,
D-001, D-002, or E-001.

- **Objective:** Implement the resolved recognition matching, normalization, denominator,
  care/ignore, multiple-output, and failure protocol, then run complete official ICDAR13,
  ICDAR15, and TotalText manifests for both DeepSolo backbones.
- **Files expected to change:** Recognition evaluator, matching/normalization code,
  table-generation code, benchmark configs, result schemas, and focused tests.
- **Tests:** Hand-scored fixtures; care/don’t-care and multiple-output cases; exact
  denominators; failed samples retained; split-manifest checks; deterministic aggregation;
  baseline/pipeline schema compatibility; complete-versus-partial claim validation.
- **Command that should eventually run:** `python -m reproduction.recognition.evaluate --config configs/strict/recognition_table5.yaml --data-root "${DATA_ROOT}" --output-root "${RUN_ROOT}"`
- **Expected artifact/output:** Per-sample predictions/matches/decisions, aggregate Table 5
  CSV/JSON, fallback counts, failure report, and both backbone variants with full provenance.
- **Paper section/table:** Section 4.2 and Table 5.
- **Dependencies/blockers:** Successful M11.B and official immutable evaluation manifests;
  E-001, DS-002, paper-exact model selection, multiple-output mapping, and recognition
  accuracy semantics remain explicit. Partial datasets or assumption-based protocols must
  remain labeled partial and/or `approximate reproduction`; they cannot support a STRICT claim.

### Milestone 12 — Context ablations

Sub-gates execute sequentially as A then B and remain independently testable. Failure in A
blocks B, and neither gate may change cached inputs or upstream artifacts to improve results.

#### Sub-gate A — Table 6 scoring-parameter ablation

- **Objective:** Run the five printed `(alpha,beta,tau)` settings over the exact 500-image
  ICDAR15 protocol while holding all other artifacts fixed.
- **Files expected to change:** Ablation runner/configs, table generator, tests, and result schema;
  scorer code must not change merely to improve results.
- **Tests:** Parameter grid exactly matches the paper; common immutable cached
  inputs/checkpoints across settings; 500 unique manifest items; accuracy, false-positive,
  fallback, and failure accounting; deterministic rerun.
- **Command that should eventually run:** `python -m reproduction.ablations.context_score --config configs/strict/ablation_table6.yaml --data-root "${DATA_ROOT}" --output-root "${RUN_ROOT}"`
- **Expected artifact/output:** Per-image/per-setting JSONL, Table 6 CSV/JSON, deltas from
  paper, failure counts, resolved config, input/artifact hashes, and provenance.
- **Paper section/table:** Section 4.3 and Table 6.
- **Dependencies/blockers:** Successful M11.A–C and immutable evaluator-compatible inputs;
  E-005 and inherited scoring/evaluation blockers must be resolved for STRICT results.

#### Sub-gate B — Table 7 context-description ablation

- **Objective:** Compare correct, no-description, and wrong-description conditions using an
  immutable image-description mapping and otherwise identical artifacts.
- **Files expected to change:** Context-ablation runner/configs, deterministic
  wrong-description manifest generator, tests, and table generator.
- **Tests:** Fixed derangement/no self-match if evidence requires; seed and mapping checksum;
  identical recognizer/fallback artifacts across conditions; explicit no-description
  representation; exactly 500 unique items; accuracy, false-positive, fallback, and failure
  accounting; deterministic rerun.
- **Command that should eventually run:** `python -m reproduction.ablations.context_description --config configs/strict/ablation_table7.yaml --data-root "${DATA_ROOT}" --output-root "${RUN_ROOT}"`
- **Expected artifact/output:** Lightweight immutable mapping manifest if licensing permits,
  per-image condition results, Table 7 CSV/JSON, resolved config, hashes, and provenance.
- **Paper section/table:** Section 4.3 and Table 7.
- **Dependencies/blockers:** Successful M12.A and the same frozen M11 inputs/evaluator;
  AB-001, AB-002, E-005, and inherited caption/evaluation blockers must be resolved or
  explicitly assumed with `approximate reproduction` labeling.

### Milestone 13 — Integrated profiling + final reproduction synthesis

Sub-gates execute sequentially as A then B and remain independently testable. Failure in A
blocks B; component qualification results are inputs, not new profiling implementations.

#### Sub-gate A — Integrated conditional-pipeline FLOPs/latency/VRAM profiling

- **Existing foundation:** AG-UNet, PARSeq, BLIP-2, and DeepSolo component profilers and
  physical-host measurements already exist. Reuse them without presenting measurements from
  different model sets, precisions, placements, or workloads as interchangeable.
- **Objective:** Implement only the missing declared integrated protocol and conditional
  expected/worst-case pipeline accounting, compare with Table 4, and separate STRICT from
  RTX 3070-adapted results.
- **Files expected to change:** Integrated profiling/aggregation tools, workload manifests,
  benchmark configs, tests, and a machine-readable hardware report.
- **Tests:** CUDA synchronization and warmup; repeat count; FLOP convention; input/crop
  counts; conditional branch accounting; component-record compatibility; peak allocated and
  reserved VRAM; explicit OOM capture; strict/adapted config separation.
- **Command that should eventually run:** `python -m reproduction.profile.pipeline --config configs/strict/profile_table4.yaml --dataset-manifest data/manifests/profile.yaml --output-root "${RUN_ROOT}"`
- **Expected artifact/output:** Reused component measurements plus integrated component and
  conditional expected/worst-case FLOPs, latency distributions, peak VRAM, OOM events,
  workload/config/artifact hashes, and explicit `HARDWARE_DEVIATION` metadata.
- **Paper section/table:** Section 4.2 and Table 4; reported BLIP-2 latency.
- **Dependencies/blockers:** Completed M11–M12 run artifacts and qualified component
  prerequisites; E-004 and the model/workload/FLOP protocol remain STRICT blockers.
  Physical-host GPU execution is required for CUDA latency/VRAM claims.

#### Sub-gate B — Final tables/report/claim audit

- **Objective:** Generate final tables and a claim-by-claim report without upgrading smoke,
  partial, approximate, or hardware-adapted evidence into STRICT claims.
- **Files expected to change:** Final reporting code/templates, result index, and documentation;
  no scientific model code unless separately authorized corrective work is required.
- **Tests:** Every aggregate traces to retained per-sample records; config, checkpoint,
  dataset, mapping, and workload hashes are present; complete-manifest checks; claim-label
  validation; all deviations, failed samples, and denominators are included.
- **Command that should eventually run:** `python -m reproduction.report.build --results-root "${RUN_ROOT}" --output-root "${REPORT_ROOT}"`
- **Expected artifact/output:** Machine-readable result index and human-readable report for
  Tables 3, 5, 6, and 7 plus Table 4 resource analysis, preserving strict/approximate/partial
  and smoke-test distinctions.
- **Paper section/table:** Entire paper, especially Tables 3–7.
- **Dependencies/blockers:** Successful M13.A and every relevant prior gate. Missing or
  incomplete manifests, provenance, per-sample artifacts, failures, or authoritative evidence
  must fail the corresponding claim audit rather than be omitted or downgraded silently.

## 3. Future-plan migration

| Old compressed gate | Reconciled gate/status |
| --- | --- |
| M9 — Segmentation training | M9 — Segmentation training completion; existing foundation retained |
| M10 — Segmentation evaluation + Table 3 | M10 — same target; existing metric foundation retained |
| M11.A — PARSeq-S qualification | Existing APPROXIMATE prerequisite; paper-exact identity remains STRICT-blocked |
| M11.B — BLIP-2 qualification | Existing APPROXIMATE prerequisite; paper-exact identity/protocol remains STRICT-blocked |
| M11.C — MPNet/fuzzy/scoring | M11.A — MPNet/scoring qualification closure |
| DeepSolo qualification/corrective gate | Existing APPROXIMATE prerequisite; evidence-triggered corrective task only |
| M12.A — Complete real scientific E2E sample | M11.B — Complete real scientific E2E sample |
| M12.B — Recognition evaluation / Table 5 | M11.C — Recognition evaluator + Table 5 |
| M13.A — Table 6 scoring-parameter ablation | M12.A — Table 6 |
| M13.B — Table 7 context-description ablation | M12.B — Table 7 |
| M14.A — FLOPs/latency/VRAM/hardware deviations | M13.A — Integrated conditional-pipeline profiling |
| M14.B — Final tables/report/claim audit | M13.B — Final tables/report/claim audit |

## 4. Recommended immediate next milestone

Proceed with **Milestone 9: segmentation training completion**, building on the existing
M4/M8 foundation. Do not recreate the optimizer/loss, basic loader/transforms, checkpoint
schema, synthetic smoke, or initial batch-size-8 feasibility work. STRICT execution must
fail until the applicable architecture, preprocessing, training-protocol, official-data,
and checkpoint blockers are resolved. An evidence-limited run must use isolated APPROXIMATE
configuration and the `approximate reproduction` label.
