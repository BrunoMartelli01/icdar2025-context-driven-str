# Literature-motivated CLIP router experiments (v3)

This is **approximate reproduction**, a user-requested extension of the frozen
Text+Kernel/PARSeq router. Local paper section 3.5 and Algorithm 2 specify
MPNet/fuzzy context scoring and whole-image DeepSolo fallback; they do not specify
CLIP. V3 leaves the strict path, frozen v2 source files, segmentation checkpoint,
PARSeq outputs, image/crop geometry and qualified DeepSolo outputs unchanged.
Scientific choices are recorded as `CLIP-ROUTER-LITERATURE-003` in
[PAPER_AMBIGUITIES.md](PAPER_AMBIGUITIES.md).

## Models and the evidence for testing them

| Experiment name | Published checkpoint | Why test it | Input |
|---|---|---|---|
| `datacomp_l14` | LAION CLIP ViT-L/14, DataComp XL | CLIP4STR Table III uses DataComp for its L backbone | 224×224 |
| `dfn_h14` | Apple DFN5B CLIP ViT-H/14 | CLIP4STR Table III uses DFN5B for its H backbone | 224×224 |
| `siglip2_l16_384` | Google SigLIP 2 ViT-L/16 FixRes | 2025 training improves multilingual, spatial and dense features; test its larger encoder and resolution | 384×384 |

[CLIP4STR](https://arxiv.org/html/2305.14014v4) trains its recognition model and
decoders. Its reported recognition scores **do not qualify a frozen, training-free
router**. DataComp/DFN are candidates motivated by those experiments, not OCR
specialists whose published STR accuracy can be copied to this task.
[SigLIP 2](https://arxiv.org/html/2502.14786v1) distinguishes FixRes from NaFlex;
NaFlex's aspect-ratio handling is particularly relevant to document/OCR retrieval.
This experiment uses the named **FixRes** checkpoint supported by the existing
Transformers runtime; it makes no NaFlex performance claim.

[CVPR 2025 scene-text retrieval](https://openaccess.thecvf.com/content/CVPR2025/html/Qin_CLIP_is_Almost_All_You_Need_Towards_Parameter-Efficient_Scene_Text_CVPR_2025_paper.html)
also motivates RN50x16 and learned retrieval adaptations. RN50x16 needs a native
CLIP ResNet runtime, which is absent here. Loading it into HF's ViT CLIPModel would
be an architecture substitution. RN50x16, NaFlex and MetaCLIP 2 remain separate
future experiments; no dependency installation or shared-environment upgrade is
performed by v3.

## Checkpoints and native processing

The complete specs live in
[clip_router_literature_v3.json](../configs/approximate/clip_router_literature_v3.json).
Weights are outside Git, downloaded at full immutable revisions; only the selected
weight file is downloaded, not both HF and OpenCLIP weight copies.

| Model | Revision | Selected weights SHA256 |
|---|---|---|
| DataComp L/14 | `84c9828e63dc9a9351d1fe637c346d4c1c4db341` | `cbae707ef72c3d2748f3c9482e9ca8645d8bbd915d59a86e4f11107c9b341068` |
| DFN H/14 | `11738501a1db6d5e0a3451a71ba100be02e577e6` | Authoritative pinned-revision Hub LFS metadata, resolved before download/execution and sealed in `pinned_lfs_receipt.json` |
| SigLIP 2 L/16 | `1b426889ea62b5a72bf9839009a1b184bfc9c178` | `11bdb3cf00c4dc4f87e437c9c75087ca16fa21aea437b3576c103670ee3c73be` |

For DFN the digest is for **`pytorch_model.bin`**, the published HF conversion;
the checksum of `open_clip_pytorch_model.bin` is a different artifact and cannot
be substituted. Missing revision metadata, checksum receipts, mismatched bytes
or checkpoint keys cause explicit errors. The resolved model identity records
the actual digest and all tokenizer/configuration file hashes.

DataComp uses its published HF processor. DFN lacks an HF processor file:
its [published OpenCLIP config](https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14/blob/main/open_clip_config.json)
explicitly specifies square resize, bicubic interpolation and RGB mean/std.
V3 reconstructs this exact supported profile and rejects other resize profiles.
Both conversions are checked for dimensions, layers, attention heads and
GELU/QuickGELU. DataComp's absent `head_width` uses the explicit 64 recorded in
v3, supported by the
[OpenCLIP v2.30.0 configuration](https://raw.githubusercontent.com/mlfoundations/open_clip/v2.30.0/src/open_clip/model.py).

DFN's published HF config has 49,409 embedding rows, whereas its native config
and published tokenizer have 49,408 tokens. V3 permits only this declared extra
unused row, retains all weights and verifies BOS=49,406, EOS=49,407, EOS padding,
attention masks and actual sequence IDs. DataComp's legacy `eos_token_id=2`
selects the first maximum token ID; DFN explicitly selects actual EOS.
No model config or embedding row is rewritten, and no non-strict loading is used.
Numeric equivalence to the original native checkpoints remains **UNRESOLVED**.

SigLIP 2 paper section 2.1 lowercases text before Gemma tokenization. V3 applies
this to both context-length checks and actual embedding inputs, preserves the
original PARSeq text in artifacts, and verifies the published right padding,
EOS insertion and absent BOS. The model uses fixed-length text padding without
an attention mask. Fast tokenizer JSON avoids a SentencePiece installation.

## Protocol and metrics

The checksum-qualified source consists of 125 previously analysed held-out TRAIN
images (983 care words) and 300 official TEST images (2,204 care words). TRAIN125
is not the full training dataset. Existing Text+Kernel/PARSeq predictions are
reused, with all predicted words, including evaluation-ignored outputs, entering
the gate. A rejected image replaces all its predictions with whole-image DeepSolo.

The frozen v2 scoring code is reused through process-local callbacks; it is not
edited. V3 keeps its 13 selector families: native confidence, crop similarity,
spelling controls, complete-transcript scene prompts and crop/scene combinations.
Prompts include `all the text present in this image is: {text}` and its Italian
version. Images/crops retain full fields of view on the existing square canvas.
All transcript characters enter context chunks instead of silent truncation.
Percentile references and the nearest 50% coverage cut use TRAIN80; the family
and cross-model selection objective is retained-word precision on TRAIN45.
The unchanged TRAIN cut transfers to TEST.

An additional GT-free score/ID ranking compares exactly 62 TRAIN and 149 TEST
accepted images, matching the original gate's budget. This is descriptive;
a transferred threshold need not accept exactly half of TEST. Previous TEST
results were already inspected, so these are exploratory comparisons, not a
new blind benchmark. No model/threshold is chosen from TEST correctness.

Outputs state casefold normalization, exact word+bbox matching at IoU≥0.5,
maximum one-to-one matching and the inherited care/ignore protocol through the
frozen source provenance. Word aggregation is micro over each split:

- Precision on detected words: `TP / (TP + FP)`.
- Requested accuracy with TN=0: `TP / (TP + FP + FN)`.
- Recall/F1, retained PARSeq words and precision, final care words, safe images
  (nonempty care predictions with no wrong care words), complete images
  (also no missed care words), and
  missing fallback IDs are reported separately.

TEST DeepSolo is available for all 300 images. TRAIN DeepSolo currently exists
for only 63 images: a newly rejected TRAIN image without a saved fallback leaves
the **entire combined TRAIN metric unavailable**, not a smaller denominator.
There is no implicit DeepSolo rerun or substitution in this experiment.

The fixed threshold grid 0–1, step 0.05, also includes each TRAIN cut. The complete
curves include observed score breakpoints. Scores are percentile combinations,
not calibrated probabilities of correctness. Curve reporting never selects a
TEST threshold.

The frozen old B/16 gate is the comparison reference:

| Split | Accepted PARSeq images | Final care words | Precision % | TN=0 accuracy % |
|---|---:|---:|---:|---:|
| TRAIN125 | 62 | 876 | 75.7991 | 55.5649 |
| TEST300 | 149 | 2069 | 70.5172 | 51.8479 |

DeepSolo alone on TEST300 gives 2,008 care words and TN=0 accuracy 60.5183%.
These are frozen prior results, **not new-model results**. See
[the completed v2 comparisons](CLIP_ROUTER_PROMPT_ABLATION.md).

## Run on the network/GPU host

From the repository root, this command downloads and evaluates the three new
models sequentially, with no package installation:

```bash
.venvs/context-str-main-py311/bin/python -m scripts.run_clip_router_literature_v3 \
  --mode approximate --device cuda:0 --download
```

Checks without downloading or inference:

```bash
.venvs/context-str-main-py311/bin/python -m scripts.run_clip_router_literature_v3 \
  --mode approximate --check
```

An initial real-checkpoint smoke test on the host can use `--limit 2`; it saves
only smoke results and deliberately creates no scientific full-split summary.
Optional smaller SigLIP 2 checkpoints and the cached old controls are listed in
the same config and selectable with `--models`. Each model processes TRAIN and
TEST separately before unloading. FP32/batch2, sequential loading and explicit
CPU execution are documented `HARDWARE_DEVIATION` choices. Device/model, image
shape, batch, software, seeds, TF32/deterministic settings and CUDA peak allocated
and reserved memory are recorded. OOM reports are saved and propagated; the
runner does not change precision or models after a failure.

Paths are configurable in
[clip_router_literature_local_paths_v3.json](../configs/approximate/clip_router_literature_local_paths_v3.json),
without cross-namespace fallback. Default results are outside Git:

- `/tmp/clip-router-literature-v3/<model>/<plan-hash>/`: resolved configuration,
  per-image features, TRAIN calibration/selection, result and failure artifacts.
- `/tmp/clip-router-literature-v3/summary/summary.{json,csv,md}`: all model/family
  comparisons, selected only with TRAIN evidence.
- `/tmp/clip-router-literature-v3/summary/original_router.json`: frozen reference.
- `/tmp/clip-router-literature-v3/summary/thresholds.{json,csv}`: separate TRAIN/TEST
  curves for each TRAIN-selected family, word counts, engine counts and metrics.

## Local qualification and remaining work

The initial local qualification had no new checkpoints or CUDA. Host runs on
2026-10-13 subsequently completed DataComp L/14 and DFN H/14 on all125 TRAIN
and300 TEST images. SigLIP2 L/16 downloaded and executed58 TRAIN images before
a prompt-chunk validation error; its full recognition/routing result is unavailable.
The 2026-10-14 audit and corrected-context qualification are recorded below.
Native-versus-HF numeric conversion equivalence remains unresolved.

67 focused tests pass, covering native resize/normalization, model dimension and
activation mismatches, declared extra vocabulary row, first-EOS pooling and pad
invariance with actual HF classes, Gemma fast JSON plus SigLIP inference,
lowercase/context preservation, published-checksum receipts, callback restoration,
threshold denominators, missing fallback and TRAIN-only model selection. Tiny
random models and toy vocabularies establish structural behavior only.
The initial test run had two incomplete synthetic-fixture failures (missing
image size); both were corrected and the relevant suite passes.

Real cached B/16 and L/14 FP32/batch2 inference also passes on two TRAIN and two
TEST images per model, labeled **smoke-test result**. This qualifies the isolated
runner on existing weights, not the three pending checkpoints or full accuracy.
The final qualification is recorded in
`/tmp/clip-router-literature-v3-qualification/qualification.json`: current code
hashes match both saved plans, and B/16 features agree with the frozen v2 run
on the four smoke images within 1.49×10⁻⁷. CPU CUDA-memory fields are null,
not GPU memory measurements.

## Completed host alternatives and independent replay (2026-10-14)

The sealed DataComp and DFN runs verify against their pinned checkpoint/config,
source files,425 per-image feature files per model, TRAIN selections and result
digests. Together with the completed old B16/B32 controls, the audit checks1700
feature payloads and replays208 model/family/split/policy comparisons. It preserves
the original source bytes under the audit artifact's `source_snapshot/` directory.

The main comparison below replays the already selected gates with the newer
paper-router **word metric**: one-to-one exact casefold word matching within
each image, punctuation retained, without bbox IoU. This same metric is applied
to native PARSeq and DeepSolo. No model, family or threshold is refitted. The
original geometric metric remains separate in the CSV; neither is established
as the context-driven paper's undefined recognition accuracy.

Keeping the original crop-ensemble selector fixed isolates the encoder change.
At62 TRAIN /149 TEST accepted images, B16/B32/DataComp/DFN give TRAIN retained
precision67.35/64.76/62.54/64.41%, TEST retained precision
63.29/63.95/64.25/63.53%, and final TEST TN052.66/52.37/52.39/52.31%.
Changing the encoder alone does not improve full-model TEST TN0 at this budget.
The next table also permits each encoder's separately TRAIN-selected score family.

At the descriptive matched budget of62 TRAIN /149 TEST accepted images:

| Router / TRAIN-selected family | TRAIN accepted words | TRAIN accepted precision % | TEST accepted words | TEST accepted precision % | TEST final words | TEST final precision % | TEST final TN0 % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original B16 crop ensemble |343|67.35|978|63.29|2069|71.24|52.66|
| B16 / crop_reads_en |349|61.03|937|61.69|2078|70.60|52.11|
| B32 / scene_plain |409|64.79|1058|63.99|2112|70.27|52.40|
| DataComp L14 / margin_scene_en |395|65.82|966|65.32|2116|70.94|53.25|
| DFN H14 / scene_plain |406|65.76|1007|63.56|2114|70.48|52.69|

Final TEST includes DeepSolo on every rejected image. DataComp raises retained
precision by2.03pp and final TN0 by0.58pp, while final precision falls0.31pp.
There are44 safe/36 complete accepted images, versus42/35 for the original gate.
A paired TEST-image bootstrap (5000 resamples, seed20261014) gives a descriptive
95% interval of[-1.71,+3.02]pp for the final TN0 difference. This small exploratory
gain does not establish a reliable selector improvement or near-perfect acceptance.

The unchanged DataComp TRAIN cut0.1850250903 accepts161/300 TEST images and gives
51.89% final TN0; the DFN TRAIN cut0.2043347516 accepts172/300 and gives50.64%.
Neither improves over the original gate at these transferred cuts. The62 TRAIN
budget requires13 unavailable fallback images for DataComp and8 for DFN, so
their combined TRAIN metric stays unavailable. Missing images are not omitted.
The complete descriptive threshold grid, word counts and per-split metrics are
saved in `thresholds.{json,csv}`.

Artifacts are outside Git at
`/tmp/clip-router-alternatives-audit-20261014/`: `comparison.{json,csv,md}`,
thresholds, source snapshots and qualification receipts. Host DataComp FP32/batch2
peak allocated/reserved CUDA memory is1,772,411,904/1,818,230,784 bytes; DFN is
4,091,493,376/4,183,818,240 bytes, on RTX3090. These are hardware deviations from
the primary RTX3070 target, not strict reproduction or target-GPU qualification.

## SigLIP2 context-boundary correction

The failed real run reaches sample`total_text:817`. The historical chunker first
finds a fitting character prefix, then selects a whitespace boundary without
checking that this boundary fits. The resulting all_en/all_it/reads_en prompts
can exceed SigLIP2's64-token context. Nonmonotonic token lengths and a space
immediately beyond the fitting prefix are both covered by regression tests.

The isolated v3 runner now uses
`recognition/clip_router_context_chunks.py` with the explicitly configured
`validated_whitespace_boundaries_v1` policy. It selects only token-valid
whitespace boundaries, preserves every raw transcript character and retains
the frozen prefix/chunk aggregation. The process-local callback is restored
on success/failure. Historical v2 scientific source files are untouched; changed
v3 code/config hashes create new run identities and preserve all old artifacts.

48 focused tests pass. All425 inputs for each of the three new models pass real
tokenizer validation:1700 scene prompts and39719 unique word/mutation prompts
per model. DataComp/DFN have no old chunk failures and all old successful prompts
are identical under the new policy; all3400 saved scene prompt lists match.
SigLIP2's old chunker fails20 prompts; all corrected prompts fit and all232 saved
scene prompt lists from its58 completed images remain identical.

Real SigLIP2 CPU FP32/batch2 inference passes2 TRAIN+2 TEST images, labelled
smoke-test result; it also embeds all six corrected chunks of the three failed
sample817 prompts. Checkpoint loading is strict, all embeddings are finite and
normalized, and no OOM occurs. CPU peaks for that failed-sample embedding smoke
are recorded (4,335,398,912 bytes RSS); CUDA peaks are null because this sandbox
does not expose CUDA. No full SigLIP2 routing accuracy is inferred from the smoke.

The remaining full run uses already downloaded pinned weights, without package
installation or downloading:

```bash
.venvs/context-str-main-py311/bin/python -m scripts.run_clip_router_literature_v3 \
  --mode approximate --models siglip2_l16_384 --device cuda:0
```

To obtain a fresh joint summary under the corrected code/config for all three
models, select `--models datacomp_l14 dfn_h14 siglip2_l16_384`. This also reruns
the two already completed alternatives under their new provenance identity.
