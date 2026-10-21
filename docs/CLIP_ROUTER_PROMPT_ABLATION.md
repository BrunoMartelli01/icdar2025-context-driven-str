# Updated CLIP-family encoders and transcript prompts

This is an **approximate reproduction / experimental extension** of the paper,
not a strict reproduction or Table5 claim. The relevant source is local paper
section3.5/Algorithm2: MPNet/fuzzy scoring followed by whole-image DeepSolo.
CLIP-based rejection gates are a separate experiment requested by the user.

The experiment preserves epoch61 Text+Kernel proposals, original stored RGB,
raw PARSeq readings, character confidence and all predicted words per image.
It does not re-detect, correct or remove words. Ignored words still enter the
router; ignore/care information is used exclusively when counting performance.
Previously audited branch metrics give full TEST DeepSolo outcomes for300 images
and TRAIN DeepSolo outcomes for63 of125 images. Unavailable complete TRAIN
metrics remain null, without dropping images or substituting recognizers.

## What is tested

- Existing CLIP B/16 and B/32, with L/14 available as another control.
- SigLIP2 FixRes B/16 at224 and384 pixels, with separate published checkpoints,
  immutable revisions and published weight SHA256. These are2025 encoders;
  successful execution with their real weights must be established on the host.
- Five crop prompt groups: the old three-template ensemble, plain text,
  `all the text present in this image is: {text}`, the Italian equivalent,
  and `the text in this image reads: {text}`.
- Four full-scene transcript prompts, using every raw predicted word in bbox
  top/left order, including an explicit all-text prefix in both languages.
- A spelling discrimination control: compare the predicted word to deterministic
  middle-character replacement, reversal and deletion on the **same crop**.
  Controls do not use GT or an annotation vocabulary. A mutation can coincidentally
  be a valid word; this is a discriminative probe, not a labelled negative.
- Native-confidence control, baseline crop/global combination and spelling/global
  combination. Every word contributes to the minimum; easy words cannot hide an
  unreliable word. There are13 image-selector families in total.

These prefixes are contrastive text embeddings, not instructions to a generative
model. A global similarity cannot establish that all true words were detected.
An object-related word can be semantically plausible and still misspell the sign.
For this reason compare spelling margins and native-only controls, and report
missing-word counts as well as retained-word precision.

## Calibration and reporting

The deterministic original TRAIN80/45 partition is reused. Percentile references
for words/scenes and the inclusive threshold nearest40/80 accepted images are
fitted using TRAIN80 without correctness labels. Each model's family is chosen
by retained care-word precision on TRAIN45, with retained-word count and family
name tie breaks. A selection artifact is sealed before TEST feature inference.
Model choice in the summarizer uses that same TRAIN45 evidence exclusively.

TRAIN125 is the already analysed held-out subset, not the full training dataset.
TEST300 has been inspected in earlier router experiments. Thus these comparisons
remain exploratory: selecting on TRAIN does not make previously inspected TEST
an untouched external benchmark. Multiple prompt/model comparisons increase
selection uncertainty; do not claim a zero-error guarantee.

Report both the TRAIN-fitted threshold transferred unchanged and a **descriptive**
equal budget comparison:62 TRAIN and149 TEST images, chosen only by score and
deterministic sample-ID tie breaks. This keeps fallback invocation counts equal
to the old roughly-half router. Exact-budget tie breaking is a budget policy,
not the inclusive scalar-threshold policy. Neither policy uses TEST correctness
to choose the threshold/model. Metrics are micro aggregates across each separate
split: exact Unicode casefold, punctuation retained, exclusive bbox IoU>=.5,
one-to-one maximum matching, care/ignore rules from the frozen audited protocol.
Precision on detected words is TP/(TP+FP); user-requested TN0 accuracy is
TP/(TP+FP+FN); recall is TP/(TP+FN). Raw/ignored counts remain saved.

Original images and crops are median-RGB padded to square so the visual processor
preserves their full field of view. CLIP has77 text positions and these SigLIP
checkpoints64; overlength transcripts are split into character-preserving chunks,
prefer whitespace, repeat the prefix and use the weakest scene-chunk cosine.
Overlength crop text uses a normalized mean across its prompt chunks. No characters
are silently truncated. This aggregation is an explicit assumption, not a learned
long-context encoder. Native/contrastive ranks are not probabilities of correctness.

## Commands

Existing encoders, CPU (executed here):

```bash
.venvs/context-str-main-py311/bin/python -m scripts.run_clip_router_prompt_ablation \
  --mode approximate --models vit_b16 vit_b32 --device cpu
```

Network/GPU host: first real smoke for both new checkpoints, then full experiment.
Downloads are about1.5GB per checkpoint plus tokenizer files; no packages are
installed. Immutable revisions/checksums are enforced before inference. Models
load sequentially in the current main environment; DeepSolo is reused from disk.

```bash
.venvs/context-str-main-py311/bin/python -m scripts.run_clip_router_prompt_ablation \
  --mode approximate --models siglip2_b16_224 siglip2_b16_384 --device cuda:0 \
  --download --limit 2 --output-root /tmp/clip-router-modern-smoke-v2
.venvs/context-str-main-py311/bin/python -m scripts.run_clip_router_prompt_ablation \
  --mode approximate --models siglip2_b16_224 siglip2_b16_384 --device cuda:0 --download
.venvs/context-str-main-py311/bin/python -m scripts.summarize_clip_router_prompt_ablation \
  --models vit_b16 vit_b32 siglip2_b16_224 siglip2_b16_384
```

The first command fails explicitly on checkpoint/runtime incompatibility or OOM;
there is no non-strict loading or silent alternate-model fallback. A completed
smoke remains labelled smoke and cannot enter the summarizer. Source/model hashes,
software, device, shape, FP32/batch8, deterministic settings, peak allocated and
reserved CUDA memory, per-image timings/errors, prompts/chunks, spelling controls,
per-word features, rank references and routes/accepted IDs remain auditable in
isolated `/tmp` outputs. CPU runs have null CUDA peaks and do not qualify GPU memory.
No new segmentation, PARSeq, captioning or DeepSolo inference is part of this run.

To complete all TRAIN fallback possibilities separately, the previously prepared
qualified command is `python -m scripts.sweep_clip_router_e2e --mode approximate
--execute-missing-train` with the main venv interpreter. The prompt experiment
currently pins its original branch-metrics snapshot; a completed additional-TRAIN
snapshot must be selected explicitly with an updated **approximate** config/path
profile and its SHA256, never silently inherited.

## Completed local result: CLIP B/16

The full CPU run has125 TRAIN and300 TEST images, thirteen families, and52
reported split/policy points. All52 focused tests pass. An independent scalar,
rank/calibration/chunk/count audit passes for all425 images and52 points; the
recomputed baseline contrast differs from the frozen previous B/16 contrast by
at most8.20e-8 over3865 words. This qualifies the old model inference and
recombination, not the unexecuted SigLIP2 checkpoints or CUDA memory.

At **62/125 TRAIN and149/300 TEST accepted images**, all families preserve the
same image budget. Retained-word precision and full TEST accuracy are:

| B/16 selector | TRAIN PARSeq precision % | TEST PARSeq words | TEST PARSeq precision % | TEST final words | TEST final precision % | TEST final TN0 % |
|---|---:|---:|---:|---:|---:|---:|
| Previous crop ensemble |66.47|978|62.27|2069|70.52|51.85|
| Crop: all the text present… |57.49|952|61.45|2100|69.71|51.55|
| Crop: tutto il testo presente… |63.22|1012|59.98|2068|68.62|49.74|
| Crop: the text…reads |60.17|937|60.51|2078|69.83|51.25|
| Scene: all the text present… |63.38|1079|63.02|2097|69.43|51.18|
| Crop ensemble + scene |65.17|1022|63.31|2083|69.66|51.16|
| Spelling margin |52.37|988|60.32|2106|68.23|50.02|
| Native only |65.63|996|62.15|2062|69.98|51.12|

The TRAIN45 threshold-based choice is `crop_reads_en` (62.60% retained-word
precision on131 words/27 images versus58.82% on187 words/32 images for the old
ensemble). Its apparent advantage does not survive equal-budget comparison on
the full TRAIN subset or TEST. Do not deploy that winner solely on that small
validation observation. Global context can raise retained-word precision a little
while worsening the complete model: which images lose DeepSolo matters, as do
missing words. These results support retaining the previous router while the
new encoder experiments await execution; none establishes a prompt improvement.

Artifacts: `/tmp/clip-router-prompt-ablation-v2/vit_b16/6742465b0595/`, result
SHA256 `05b8486e7f87afab1d20493222a949a593c089ae221a01fb84d7150c498278e0`,
independent audit and sealed TRAIN selection. Full combined TRAIN metrics are
unavailable for selectors requiring uncached fallback images; retained-word
TRAIN precision above uses all125 source images under the stated acceptance gate.

## Completed local result: CLIP B/32 and combined comparison

B/32 also completes all425 images and passes its independent52-point audit.
The baseline TRAIN contrast is checked separately against the previous frozen
B/32 run. At the same62 TRAIN/149 TEST budget:

| B/32 selector | TRAIN PARSeq precision % | TEST PARSeq words | TEST PARSeq precision % | TEST final words | TEST final precision % | TEST final TN0 % |
|---|---:|---:|---:|---:|---:|---:|
| Previous crop ensemble |64.18|971|62.82|2070|70.19|51.51|
| Crop: all the text present… |58.49|1014|60.06|2097|68.29|49.91|
| Crop: tutto il testo presente… |63.34|1006|60.14|2081|68.86|50.25|
| Scene: all the text present… |61.97|1099|63.33|2106|68.90|50.75|
| Scene: the text…reads |64.88|1017|63.82|2106|69.75|51.71|
| Scene: plain transcript |63.81|1058|62.38|2112|69.32|51.33|
| Spelling margin |58.40|1026|54.19|2121|66.62|48.52|

The predeclared TRAIN45 model/family choice is B/32 + plain scene transcript:
99/158 retained care words (62.6582%) versus B/16's82/131 (62.5954%). This tiny
exploratory validation difference is not evidence of improvement; its full TEST
TN0 at the matched budget is51.3324%, below the original B/16's51.8479%.
All26 model/family combinations fail to exceed the original B/16 full-model TEST
TN0 at the same image budget. Some raise PARSeq precision but sacrifice the
benefit DeepSolo would have given on the swapped images.

Real CPU FP32 batch8 runs take932.62s for B/16 and501.14s for B/32 (sum of recorded
per-image inference timings, not complete deployment latency). No CPU inference
failure occurs. GPU memory/OOM qualification for the new checkpoints remains
pending. Source schemas, checkpoints, frozen prior code and strict configuration
are unchanged. Results remain **approximate reproduction** and exploratory.

B/32 artifacts: `/tmp/clip-router-prompt-ablation-v2/vit_b32/5add22ad09b1/`, result
SHA256 `eb8d760861967d02122c5f691a9c756723891c2441f87e5c2b51563773700133`.
Full TRAIN/TEST, threshold-transfer and equal-budget results for both models:
`/tmp/clip-router-prompt-ablation-v2/summary/summary.md`, `summary.csv` and
`summary.json`. These tables contain retained/final word counts and precision,
TN0 accuracy, recall/F1, safe/complete image counts, and explicit missing fallbacks.

## Primary sources and further experiments

[SigLIP2 paper](https://arxiv.org/abs/2502.14786) and
[official model card](https://huggingface.co/google/siglip2-base-patch16-224)
support testing multilingual/denser image-text representations. They do not prove
that these prompts verify spelling. The
[Transformers SigLIP2 documentation](https://huggingface.co/docs/transformers/v4.50.0/en/model_doc/siglip2)
distinguishes FixRes (compatible with SigLIP1 model classes) from NaFlex. The
[SigLIP documentation](https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/siglip)
requires fixed-length text padding. The adapter uses the native image processor,
fast tokenizer JSON and fixed padding64 without an attention mask for SigLIP;
it does not replace a NaFlex checkpoint with FixRes. New real-weight compatibility
is still pending; mocked API tests are not sufficient qualification.

[CLIPTER](https://arxiv.org/abs/2301.07464) explores global scene context in text
recognition with trained fusion. [CLIP is Almost All You Need (CVPR2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Qin_CLIP_is_Almost_All_You_Need_Towards_Parameter-Efficient_Scene_Text_CVPR_2025_paper.html)
explores scene-text retrieval with parameter-efficient adaptation. These motivate
crop/global and spelling-sensitive controls, but their gains cannot be attributed
to these training-free prompts without this experiment.

Possible later controls: same-scene prompts with one predicted word changed at a
time; compare original-scene versus text-masked-scene embeddings to isolate the
signal contributed by lettering; put multiple word crops into a high-resolution
contact sheet to avoid shrinking tiny scene text; add caption context while
holding the crop/word fixed to measure whether semantics help or conceal spelling
errors. These are proposed, not implemented/tested here. MetaCLIP2 is another2025
candidate, but requires its own model class/newer runtime and is not silently
loaded with legacy CLIP. This change deliberately avoids dependency upgrades.
