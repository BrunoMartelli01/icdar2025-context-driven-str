# ICDAR2015: existing SynthText data for broader text pretraining

This is an **approximate reproduction / research extension**, not paper-specified
training. Paper3.2/4.1, the current U-Net/optimizer and the existing qualified
SynthText preparation were inspected. Motivation is the measured same-model
FIT900/VAL100 localization gap:88.6632% versus57.1429%, not correct-word F1.
Additional fitting epochs and encoder-decay changes did not improve validation.

## Data preparation

Reuse the exact already-extracted50,000-image cohort from the preserved manifest
backup, SHA256 `59e244b23db1409cecec0e861d23ea18f5fb1d004663d276c4188cc6037eb002`.
Verify original MAT SHA256
`5b44cba773f51dfa2d4a1b324c35cbc6905e4be8e9df867ac65b57c1bf8f150c`, every RGB
checksum and all original word polygons/transcriptions. No data download or
source modification occurs. Reuse `preserve_visible_words_v1`, retaining all
422,616 annotation records:422,610 visible words,6 wholly off-image annotations
explicitly recorded,20 restored contact words. Do not move boxes onto borders,
remove symbols from training labels or replace cohort members.

The separate `icdar2015_synthtext_data_v1.json` profile materializes lossless
uint8 masks/validity/separators and uint16 owners, with per-image audits and
checksums. Four spawned CPU workers each use one computation thread. Resume
requires the exact source/configuration contract; corrupted published cache
entries fail. Exclude exact RGB-byte overlap with allIC TRAIN1000, Total-Text
TRAIN1255 andIC TEST500. The TEST manifest supplies only image hashes for
leakage protection; its annotations/scores are not opened. Near-duplicate
scenes remain unresolved. All dataset/cache locations are explicit arguments.

Four cache tests passed, including bitwise equality with the existing uncached
paired RGB/target/validity/separator tensors at two augmentation epochs, matching
native-word-loss counts, immutable raw labels, overlap rejection and corruption
failure. The initial two fixture failures omitted the required probability
threshold; the fixture was corrected and the original failing XML retained.
This qualification is data preparation, not evidence of model performance.

## Predeclared training stages and evidence boundary

Warm the unchanged single-output58,155,081-parameter attention U-Net from c467
on these50k images for5 epochs at RGB512/native256, then fine-tune only original
FIT900 for50 epochs at RGB736x1280/native368x640. Keep all encoder BN affine
parameters/statistics frozen; enable layer3/layer4 convolutions. Use fresh AdamW
per stage, decoder decay0.01 and encoder decay0; initial decoder/encoder LR
1e-4/1e-5 during synthetic pretraining,1e-5/1e-6 during ICDAR fine-tuning,
cosine to0.1x, separator-weighted BCE and clipping1. This combined data/optimizer
experiment cannot isolate a causal data effect. AdamW avoids the previously
measured coupled-L2 encoder shrinkage; it is an ASSUMPTION outside strict Adam
reproduction. Source weights and all452 keys load strictly without migration.

Synthetic pretraining selects its predeclared last epoch. Subsequent target
checkpoint/policy choices use the unchanged VAL100/430care only, with raw word
F1 first and all original care words/emitted predictions retained. Original
sources remain eligible in the eventual global comparison. Batch8 during
synthetic pretraining and physical4/accumulation2 during native fine-tuning,
FP32 and actual RTX3090 memory/OOM records require full backward qualification
before each complete stage. No training success is claimed by this proposal or
by a prepared data cache. No new TEST evaluation has run;70% remains unachieved.

Data/optimizer/schedule choices are ASSUMPTION; unchanged source cohort/target
policy are REPOSITORY_SPECIFIED; physical accumulation and actual-hardware
execution are HARDWARE_DEVIATION. See ambiguity `ICDAR2015-SYNTHTEXT-001`.

## Actual readiness and start

All50,000 cached images completed preparation with the exact expected counts,
zero exact-byte exposure overlap and no sample replacements. Manifest SHA256:
`f911724549be7b6cd53d4490fdddab345a2561ca58458d9b141f703cf85b54e2`.
The cache occupies687MiB without RGB copies. All26 previously problematic
source images reproduce the uncached paired tensors and native-word-loss counts
bitwise. Eighteen stage/cache/encoder/accumulation tests passed. Stage metadata
records the actual AdamW optimizer, last-checkpoint synthetic selection and
target word-based selection; both stages initialize fresh optimizer moments.

Both full-size backward smokes passed,16 fitting samples each, without OOM.
Synthetic FP32 physical8/accumulation1 at512/native256 peaked at5,220,027,392
allocated /5,964,300,288 reserved bytes; native fine-tuning physical4/accumulation2
peaked at9,994,451,456 allocated /11,800,674,304 reserved bytes on RTX3090. The
small smoke OCR scores are not quality evidence. Full initial target validation
exactly replays the qualified c467 source on VAL100/430care.

The detached runner has started full synthetic pretraining and queues complete
fine-tuning only after the5-epoch source succeeds. The unchanged20-case target
operating grid is queued after complete fine-tuning. An external archiver keeps
actual word-validated checkpoint copies; it changes neither model updates nor
declared stage selection. Pretraining word metrics are measured each epoch for
audit, while source selection remains its predeclared last epoch. Native word
loss counts include augmentation/resampling losses and never remove original
annotations or evaluation care words. No full training success or new TEST
result is claimed here. See
`provenance/ICDAR2015_SYNTHTEXT_READINESS_V1.json`.

## Completed result

Both stages and the20-case target operating grid completed successfully.
Synthetic pretraining used all50,000 images in every one of5 epochs, with
422,610 visible annotations and6 off-image annotations recorded per epoch.
The declared last checkpoint is epoch5 (`51266c4b3b991a5ec725e4ceeb60ef6fea3ebe6683b3d403d2c26f4724306e8a`):
target validation raw word F1 28.9548%, emitted50.1385%. The subsequent50
complete FIT900 epochs select epoch42
(`99ffbeacc869576c986e7a5f2fcf58182c98b7c842db5aeddf138669edec8a43`):
raw33.9552%, emitted49.8551%. The best operating-grid case is threshold0.35,
oriented2 crops, mean character confidence>=0.9 and normalized length>=3:
203 raw correct words/775 proposals;198 emitted TP,114FP,232FN,312 predictions,
F1 53.3693%. All430 care words remain; these are VAL100 results, not TEST.

This combined data/AdamW recipe does not improve the prior61.7608% native
operating policy or63.2944% confidence-partition follow-up. It does not isolate
the causal contribution of either data or optimizer. No inferior checkpoint is
selected for TEST. All23 actual word-validated archived states, source data and
full per-image outputs are preserved. The target remains unachieved. See
`provenance/ICDAR2015_SYNTHTEXT_COMPLETION_V1.json`.
