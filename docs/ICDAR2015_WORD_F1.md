# ICDAR2015 full training and symbol-free word F1

This is **approximate reproduction**, a standalone U-Net -> PARSeq diagnostic
on the official ICDAR2015 TEST population, not a paper Table 5 reproduction.
The user explicitly requested full training of the transferred U-Net followed
by word F1 without symbols. The segmentation architecture/targets remain those
qualified in `docs/ICDAR2015_TOTALTEXT_TRANSFER.md`; no TEST-driven training,
threshold sweep, crop selection or checkpoint selection is introduced.

## Frozen training and evaluation protocol

Train `configs/approximate/icdar2015_totaltext_transfer_v1.json` for 100 complete
epochs on the frozen 900/100 official TRAIN split. The selected checkpoint
maximizes validation native word-proposal F1 with the already declared tie
breaks. Preserve the original 1000-image target qualification and previous
smoke report. Training output is external:

`/extra1/bmartelli/context-str-runs/icdar2015-totaltext-transfer-v1-train/`.

After training completes, evaluate every image in the official 500-image TEST
manifest, checksum
`2d975369e44688e635b5fe4e79a595d662538e7607ad1ec9d95bad2b5c71fcf9`.
There are 2077 care words in the previously audited prepared population. The
evaluator fails if the complete care/image denominator differs. It never
silently drops a failed or empty sample.

Evaluation configuration:
`configs/approximate/icdar2015_totaltext_transfer_word_f1_v1.json`.
Model/evaluator:
`scripts/train_icdar2015_totaltext_transfer.py`,
`scripts/evaluate_icdar2015_totaltext_transfer_word_f1.py`.
Metric: `recognition/icdar2015_word_f1.py`.

- Segmentation: FP32, batch 1, RGB512 / native256 paired letterbox, threshold
  fixed at 0.5 from TRAIN. Clip only the RGB-derived valid canvas.
- Localization: all 8-connected components, sorted by component pixel area;
  original-RGB boxes via inverse letterbox; pad0; no area or top-ten filtering.
  These practical choices are explicit deviations from paper Algorithm 1.
- Recognition: the existing checksum-locked approximate official PARSeq
  checkpoint/runtime, strict state loading, its qualified autoregressive and
  refinement defaults, OCR batch32. This is not an identification of the exact
  paper PARSeq-S artifact. Segmentation unloads before OCR.
- Ground truth: original ICDAR2015 care quadrilaterals converted to clamped
  axis-aligned raster envelopes with exclusive upper edges (`max+1`). Exclude
  official ignored GT (`###`); retain every other care word. All emitted
  predictions count, including ignored-region predictions.
- Primary text rule: Unicode casefold, then retain `isalnum()` characters.
  Thus punctuation, symbols and whitespace do not affect equality; letters,
  numbers and precomposed accents remain. No NFKC, accent removal, dictionary,
  word splitting or fuzzy acceptance is added. Each crop remains one output.
- A true positive requires both bbox IoU >= 0.5 and exact normalized text,
  using maximum-cardinality one-to-one matching. Duplicate detections can only
  match distinct GT instances. Empty normalized readings never match. Symbol-
  only care GT and empty/symbol-only predictions remain in denominators.
- Micro counts across all images: `FP = predictions - TP`, `FN = care GT - TP`,
  `P = TP / predictions`, `R = TP / care GT`,
  `F1 = 2 TP / (predictions + care GT)`. Empty denominators are `null`, except
  F1 is zero when one nonempty side has no matches. Empty images stay counted.
- Report geometric detection F1 and punctuation-preserving exact casefold F1
  as controls; label the alphanumeric casefold result as the primary requested
  F1. This is a bbox end-to-end word proxy, not official polygon Hmean.

GT transcriptions enter scoring only after every prediction box, RGB crop and
PARSeq reading is frozen. Masks/probabilities, component areas, boxes, crop
hashes, full OCR diagnostics and per-image counts are preserved. Interrupted
inference resumes only with the identical protocol and verified file hashes.
An orphaned mask is reused only after exact deterministic array comparison.

## Evidence classification

Paper sections 3.2/3.3 and 4.1 establish segmentation/localization/recognition,
but not this custom normalized end-to-end F1. The existing metric, adapter,
letterbox and connected-component implementations were inspected before the
new evaluator. See `ICDAR2015-WORD-F1-001` in `docs/PAPER_AMBIGUITIES.md`.

- `PAPER_SPECIFIED`: ResNet-50 attention U-Net, connected region localization,
  crop recognition and the paper optimizer/loss families.
- `REPOSITORY_SPECIFIED`: qualified approximate PARSeq source/archive/weights
  and existing Unicode alphanumeric normalization/matching primitives.
- `ASSUMPTION`: transferred approximate segmentation, standalone U-Net/PARSeq
  protocol, 900/100 validation selection, native-to-original mapping, bbox IoU,
  ignore policy, exact symbol-free equality and complete micro denominators.
- `HARDWARE_DEVIATION`: actual PAPER RTX3090/24 GiB, FP32 with TF32 disabled,
  segmentation batch1, OCR batch32 and sequential model unloading. Save peak
  allocated/reserved memory separately for training, segmentation and OCR.
- `UNRESOLVED`: exact strict paper checkpoint/protocol identities; no full
  contextual routing/BLIP-2/MPNet/DeepSolo or official polygon result is claimed.

## Execution

Use the existing `.venvs/context-str-main-py311/bin/python`. No installation or
download is needed. Training command is in the transfer guide. After full
training, run from the repository with explicit paths:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
.venvs/context-str-main-py311/bin/python -u -m scripts.evaluate_icdar2015_totaltext_transfer_word_f1 \
  --config configs/approximate/icdar2015_totaltext_transfer_word_f1_v1.json \
  --training-config configs/approximate/icdar2015_totaltext_transfer_v1.json \
  --training-run /extra1/bmartelli/context-str-runs/icdar2015-totaltext-transfer-v1-train \
  --test-manifest /extra1/bmartelli/datasets/context-str/scene-segmentation-v1/manifests/icdar2015_test.yaml \
  --data-root /extra1/bmartelli/datasets/context-str/scene-segmentation-v1 \
  --parseq-source /extra1/bmartelli/context-str-artifacts/parseq/parseq-1902db043c029a7e03a3818c616c06600af574be \
  --parseq-source-archive /extra1/bmartelli/context-str-artifacts/parseq/parseq-1902db043c029a7e03a3818c616c06600af574be.tar.gz \
  --parseq-checkpoint /extra1/bmartelli/context-str-artifacts/parseq/parseq-bb5792a6.pt \
  --output-dir /extra1/bmartelli/context-str-runs/icdar2015-totaltext-transfer-word-f1-test-v2 \
  --device cuda:0
```

Focused checks are in `tests/recognition/test_icdar2015_word_f1.py`,
`tests/scripts/test_evaluate_icdar2015_transfer_word_f1.py` and
`tests/scripts/test_audit_icdar2015_word_f1.py`.

## Completed results

All 100 epochs completed successfully: 90000 training image passes,
11300 optimizer steps, the same 900/100 split throughout. Best checkpoint:
epoch **38**, SHA256
`823c4a416cdfa0af460d0f73bdb6c598aac6c8540e2c6acdfaa9c392dd55081c`.
Its repeated validation has 193 matches / 518 proposals / 430 care words,
proposal F1 **0.40717299578059074**, macro foreground IoU
**0.5448843381331329**. The final strictly loaded checkpoint replay reproduces
the selected epoch's metrics exactly. Training took 2730.83 seconds for this
invocation; no isolated whole-pipeline latency claim follows.

Every official TEST image was evaluated: **500 images, 2077 care words,
3153 ignored GT instances, 2507 emitted crops/readings, zero failed samples**.
All emitted predictions, including ignored-region outputs, count. There are
164 empty normalized predictions and zero symbol-only care GT words.

| Metric | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Correct words, Unicode alphanumeric casefold (primary)** | **639** | **1868** | **1438** | **25.4886%** | **30.7655%** | **27.8796%** |
| Correct words, exact casefold with punctuation retained | 602 | 1905 | 1475 | 24.0128% | 28.9841% | 26.2653% |
| Geometry only, bbox IoU >= 0.5 | 880 | 1627 | 1197 | 35.1017% | 42.3688% | 38.3944% |

Primary F1 is exactly `2*639/(2507+2077)`. It combines detection and correct
recognition; the higher geometry-only F1 is not substituted for it. These are
approximate bbox results, not official polygon Hmean or full contextual paper
recognition results. No matched original-model baseline was run, so this does
not establish a causal gain from the transferred changes.

TEST output:
`/extra1/bmartelli/context-str-runs/icdar2015-totaltext-transfer-word-f1-test-v2/results.json`,
SHA256 `0053bb7c3a5dbe23c5e5e92eac294681e896787ff3c2c74730194583f96cf5cd`.
Full masks/probabilities and sample/OCR caches accompany the report.

The separate `scripts/audit_icdar2015_word_f1.py` passes all 500 images:
independent CSV GT/ignore parsing, probability threshold and canvas replay,
SciPy 8-connected component/pixel-area reconstruction, inverse-letterbox box
mapping, all 2507 RGB crop/OCR hashes, independent SciPy bipartite matching
and exact aggregate-count/rate agreement. Audit file: `independent_audit.json`
in the TEST directory. All **30 focused tests** pass, including 768 independent
SciPy matching cases and a full synthetic artifact audit/corruption rejection.

Measured allocated/reserved GPU peaks (bytes), RTX3090/FP32, no OOM:

| Phase | Batch | Allocated | Reserved |
| --- | ---: | ---: | ---: |
| Training, RGB512/native256 | 8 | 4222142976 | 4647288832 |
| TEST segmentation, RGB512/native256 | 1 | 497434112 | 564133888 |
| TEST PARSeq crop recognition | 32 | 205955584 | 245366784 |

Model phases run sequentially. Final TEST inference/scoring took 34.78 seconds for
this invocation, with artifact checks and file I/O; this is not a whole
BLIP-2/context/DeepSolo benchmark. The original smoke qualification remains
preserved separately. Portable completion evidence:
`provenance/ICDAR2015_WORD_F1_COMPLETION_V1.json`; pre-TEST frozen protocol:
`provenance/ICDAR2015_WORD_F1_PROTOCOL_V1.json`.

### Completed-cache serialization correction

The first full result in the external `...word-f1-test-v1/` directory already
has the same scores and passes independent audit. A completed-cache reopening
then failed because Python GT polygon tuples compared unequal to their lossless
JSON list representation. The result file was not changed. Its original source
bytes and explicit failure record remain in `source_snapshot/` and
`completed_cache_failure.json` there.

The correction stores polygon coordinates as JSON-native lists. It changes no
model, threshold, coordinate values, crop, decoding, text normalization or
matching. Because source hashes are strict, the corrected complete evaluation
uses the separate `...word-f1-test-v2/` directory. Its completed-cache reopening
passes with no new inference. Both full runs have exactly equal probability
and binary-mask arrays for all 500 images, GT/proposals/counts, and all 2507
OCR diagnostic records (excluding the versioned protocol hash). This check is
saved as `serialization_equivalence_audit.json` in the final directory. No
scientific setting was selected after reading TEST results.
