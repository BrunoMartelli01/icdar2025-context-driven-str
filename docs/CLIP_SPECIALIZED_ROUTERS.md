# Native Long-CLIP / LLM2CLIP OCR router experiment

This is an **approximate reproduction / research extension**. The local paper
§3.5 uses MPNet/fuzzy context scoring, not CLIP. No Table5 or strict claim.
Scientific decisions: `CLIP-SPECIALIZED-ROUTERS-001` in
[PAPER_AMBIGUITIES.md](PAPER_AMBIGUITIES.md).

The question is whether the released specialized encoders can accept complete
PARSeq image predictions more reliably than ordinary CLIP, routing other images
to the unchanged DeepSolo fallback. A stronger text encoder is a hypothesis,
not evidence that a scene description proves exact OCR spelling.

## Models and evidence

| Encoder | Pinned weight revision | Native text context | Input |
| --- | --- | ---: | ---: |
| Long-CLIP-B | `107b6f178477e1bf8af8bfe565c13e691e2c356f` |248|224|
| Long-CLIP-L | `f7ead1bafa68ee36e3469b8bdcdba404c09952ab` |248|224|
| LLM2CLIP OpenAI-L14/336 | `92512331f393a003c3d98404677f991c188162c9` |512|336|

LLM2CLIP's paired fine-tuned Llama3-8B revision is
`97f7d164a1eb89a79e344b21c7a1be7eee265d2e`. All checkpoint files are verified
against published LFS SHA256. Downloaded weights/source/papers live outside Git
under `/extra1/bmartelli/context-str-artifacts/clip-specialized-audit-20261014`.
The weights total approximately19.65GB decimal, plus small metadata and sources.

Primary sources: [LLM2CLIP model and inference example](https://huggingface.co/microsoft/LLM2CLIP-Openai-L-14-336),
[LLM2CLIP paper](https://arxiv.org/abs/2411.04997),
[Long-CLIP implementation](https://github.com/beichenzbc/Long-CLIP),
[Long-CLIP paper](https://arxiv.org/abs/2403.15378).
The Microsoft release explicitly warns its HF models may perform differently
from the paper's native PyTorch weights.

## Frozen comparison

Keep the existing Text+Kernel/PARSeq outputs on TRAIN holdout125 images/983 care
words and TEST300 images/2204 care words. Reuse the thirteen prior crop,scene,
margin and calibrated score families, including raw text, “all the text present
in this image is:”, its Italian variant and “the text in this image reads:”.
Calibrate on TRAIN80 and select the family on TRAIN45 legacy geometric retained-
word precision; TEST labels never select a family or threshold.

Report both the frozen TRAIN threshold and descriptive equal accepted-image
budgets62 TRAIN/149 TEST. The latter uses TEST scores, not labels, to match
coverage; it is not a fixed-threshold deployment claim. Primary metrics are
the current per-image word multiset protocol, casefold exact with punctuation,
no bbox IoU. Report geometric metrics separately. Precision=TP/(TP+FP);
TN0 accuracy requested by the user=TP/(TP+FP+FN). Empty outputs retain FN.
If a rejected TRAIN image lacks cached DeepSolo output, whole-model TRAIN
metrics are unavailable; report the missing count.

## Execution and qualification

Use existing environments, no package installation. Long-CLIP runs natively in
the LAVIS environment; the main environment evaluates the frozen population.
The schema1 stdio/NPY interface preserves Unicode/order and hashes pixel and
embedding payloads. Model errors, OOM and nonzero subprocess exits propagate.

Both trained encoders are loaded strictly. Native Long-CLIP source is unchanged
on disk; only its removed packaging re-export is replaced in memory with the
same standalone packaging API. LLM2CLIP uses its shipped bidirectional Llama
class and pinned mask helper. Its thin inference wrapper matches published
LLM2Vec framing, IDs, masks and pooling on eight strings; the qualification
receipt is `llm2vec_wrapper_qualification.json` in the external audit directory.

Hardware deviations: actual RTX3090/24GB, sequential model processes,batch2,
4 CPU threads,deterministic,TF32off. Long-CLIP FP32; LLM2CLIP BF16 weights,eager
attention,no cache,FP32 pooled adapter input,BF16 autocast and FP32 final cosine
features. This differs from the published example's default FP16 autocast and
does not qualify the primary8GB target. Each output saves allocated/reserved
peaks, shapes, batch, dtype, source hashes and software metadata.

Source-only preflight (does not run or qualify model weights):

```bash
.venvs/context-str-main-py311/bin/python -m scripts.run_clip_specialized_routers \
  --mode approximate --check
```

Smoke tests (2 TRAIN +2 TEST images per model):

```bash
.venvs/context-str-main-py311/bin/python -u -m scripts.run_clip_specialized_routers \
  --mode approximate --models longclip_b longclip_l llm2clip_l14_336 \
  --device cuda:0 --limit 2 \
  --output-root /extra1/bmartelli/context-str-artifacts/clip-specialized-smoke-v1
```

Full comparison:

```bash
.venvs/context-str-main-py311/bin/python -u -m scripts.run_clip_specialized_routers \
  --mode approximate --models longclip_b longclip_l llm2clip_l14_336 --device cuda:0
```

Fully resolved configuration and per-image features/scores are sealed under
`/extra1/bmartelli/context-str-artifacts/clip-specialized-routers-v1/<model>/<plan-hash>`.
An exact-plan rerun validates and reuses its own features. No metric result is
inferred merely from a successful launch.

Initial failures were legacy CUDA initialization before allocator reset and
missing `pkg_resources`; both are preserved in smoke failure artifacts and
corrected with focused regression tests. A subsequent LLM2CLIP loading failure
identified distinct Python class identities from importing a dynamic config
twice; using the native model's own `config_class` resolved it without changing
model parameters. The original failure receipt is preserved.

All three encoders passed real-weight four-image smoke tests without
missing/unexpected/mismatched keys. Final preflight code also saves its source
snapshots and requires a successful worker exit before writing SUCCESS.
Relevant unit tests:59 passed. Preliminary allocated/reserved peaks were:

| Model | Allocated bytes | Reserved bytes |
| --- | ---: | ---: |
| Long-CLIP-B |629,357,568|679,477,248|
| Long-CLIP-L |1,752,279,552|1,826,619,392|
| LLM2CLIP-L14/336 + Llama3-8B |16,330,224,128|16,473,128,960|

Full-run measurements supersede smoke measurements. LLM2CLIP's unquantized
weights alone exceed the primary8GB target. Unrelated GPU training was active
during qualification; feature times are logged for auditing and are not a
controlled throughput benchmark or the cost of only the selected score family.
Allocated/reserved peaks are per worker.

The first full LLM2CLIP attempt failed at TEST sample1543 after287 completed
TEST images: its final image batch mixed a3×3 and2×2 RGB crop. Transformers4.47
guessed channels-first for the ambiguous3×3 image and rejected the2×2 image.
No sample/crop was dropped. The RGB bridge's HWC contract is now passed explicitly
as `input_data_format='channels_last'`. A real processor test covers1/2/3/5-pixel
RGB crops, exact channel colors and mixed-batch/single-image equivalence. The
actual failing sample's five normal batches produce identical pixel tensors;
its sixth batch now has shape2×3×336×336. Receipt:
`tiny_rgb_preprocessor_qualification.json` in the external audit directory.
The old partial run is preserved as failed evidence. The corrected LLM2CLIP
experiment reruns all425 images in a fresh plan and reuses no partial features.
After the RGB fix, the relevant suite passes102 tests. Comparing both complete
TRAIN feature sets found124 exact matches and one changed image,1189, consistent
with a silently ambiguous tiny crop in the old preprocessor. The corrected
TRAIN-selected family remains `crop_all_it`, threshold0.11091222039324732;
its selection receipt was captured with zero corrected TEST features produced.

The complete experiments and independent replay audit passed. Final source:
`/extra1/bmartelli/context-str-artifacts/clip-specialized-comparison-v1/final/`.
The directory contains `comparison.json`, `comparison.csv`, `thresholds.csv`,
`comparison.md`, and standalone PNG/PDF coverage curves.

## Completed results

Each model's family is chosen on TRAIN only. The equal-budget comparison below
accepts62/125 TRAIN and149/300 TEST images, with no TEST-label fitting. TRAIN
figures use the separately completed DeepSolo branches. Word precision is for
accepted PARSeq words; TN0 is for the complete PARSeq/DeepSolo system.

| Router | TRAIN PARSeq words | TRAIN word precision | TRAIN TN0 | TEST PARSeq words | TEST word precision | TEST TN0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original CLIP B16 |343|67.35%|56.22%|978|63.29%|52.66%|
| Local MPNet/caption paper-style |477|55.14%|49.38%|1169|57.57%|50.14%|
| Prior DataComp-L14, TRAIN-selected family |395|65.82%|53.71%|966|65.32%|53.25%|
| Long-CLIP B / `margin_scene_en` |386|60.36%|51.04%|1006|64.02%|52.08%|
| Long-CLIP L / `scene_reads_en` |396|67.93%|54.18%|1064|64.10%|52.47%|
| LLM2CLIP-L14/336 / `crop_all_it` |385|65.71%|54.03%|992|62.10%|51.86%|

No specialized encoder improves the original frozen router at this budget.
TEST conditional paired-bootstrap95% intervals for the TN0 difference in
percentage points are Long-B[-2.627,+1.459], Long-L[-1.849,+1.325],
LLM2CLIP[-2.503,+0.722]. These exploratory comparisons do not establish a
statistically reliable difference. The controlled `crop_baseline` family also
fails to improve whole-system TEST TN0: B16=52.66%, Long-B=52.19%,
Long-L=52.19%, LLM2CLIP=51.96%.

At TEST budget30/300, word precision improves from original84.62% to
Long-B90.91% /Long-L90.43%, but their complete-image counts are18/30 and19/30.
LLM2CLIP at15/300 has46 correct of49 accepted words (93.88%) and11/15 complete
images. Thus word precision does not certify complete scene recognition.
At149 accepted images, complete-image counts are35 original,32 Long-B,
35 Long-L and36 LLM2CLIP. DeepSolo-only TN0 is65.30% TRAIN /60.95% TEST.

Transferring the actual frozen TRAIN cuts gives TEST acceptance176 images
(Long-B,0.17562254,TN050.16%),180 (Long-L,0.19755537,TN049.42%),
170 (LLM2CLIP,0.11091222,TN049.70%). This differs from the equal-budget
description. Full grids and every exact acceptance budget are preserved.

Final LLM2CLIP allocated/reserved peaks are16,330,224,128/16,479,420,416 bytes.
All1275 final per-image feature payloads and312 original policy points replayed
exactly. All62 supplementary DeepSolo loads were strict; peaks were
869,240,320/1,270,874,112 allocated/reserved bytes. Native OpenCV/Detectron2
preprocessing was reapplied to all62 exact inputs to record decoded and model
input dimensions, with the qualified shortest-edge1000 /max1892 policy; the
shape probe is explicitly reconstruction, not instrumented model inference.
The final report also replays the previous models and local MPNet baseline on
completed TRAIN data, without changing their gates or original result files.

## Completing the TRAIN fallback comparison

The frozen source contains DeepSolo for all300 TEST images and63 TRAIN images.
To compare all frozen gates on TRAIN125, execute the missing62 TRAIN images
through the same qualified isolated runtime. This is an `ASSUMPTION` for complete
counterfactual branch evaluation, not a new recognizer or a refit. The new
executor validates the canonical M6/host records and the old E2E provenance,
then preserves all raw detections with the existing score0.4, EXIF and native
polygon-ignore rules. Empty/invalid predictions retain their evaluation costs.
Existing branches, GT, PARSeq and all TEST outputs remain byte-for-byte inputs.

```bash
.venvs/context-str-main-py311/bin/python -u -m scripts.complete_clip_router_train_fallback \
  --mode approximate \
  --output-root /extra1/bmartelli/context-str-artifacts/clip-specialized-train-deepsolo-v1
```

Run after native CLIP workers exit. Outputs are sealed separately and store
their full resolved configuration/source snapshots, raw responses, completed
word/geometric branches and per-sample evaluation. Completion must pass before
reporting new whole-model TRAIN figures. Tests reject GT/PARSeq changes,
replacement of any existing DeepSolo output, missing samples and duplicated IDs.

## Independent result audit

```bash
.venvs/context-str-main-py311/bin/python -m scripts.audit_clip_specialized_routers \
  --mode approximate \
  --output-root /extra1/bmartelli/context-str-artifacts/clip-specialized-comparison-v1/final \
  --train-fallback-supplement /extra1/bmartelli/context-str-artifacts/clip-specialized-train-deepsolo-v1 \
  --previous-audit /tmp/clip-router-alternatives-audit-20261014 \
  --paper-baseline /tmp/paper-image-router-baseline-v1/a622b7199f00
```

The audit verifies features/source snapshots, recomputes TRAIN references/family
selection and replays all104 word/geometric policy points per model. It checks
the PARSeq-only confidence control agrees across encoders, evaluates real score
threshold grids and exhaustive exact acceptance budgets, and retains split
denominators. At a tied score an exact budget uses deterministic sample IDs;
`rank_cut_score` and `inclusive_scalar_accepts` show why its synthetic binary
threshold1 is not necessarily a deployable scalar cutoff. The equal-budget
comparison is descriptive. A5000-draw paired image bootstrap uses frozen gates,
never selecting policies from TEST labels. Supplemental TRAIN metrics are stored
under `completed_train_comparisons`; original result files remain unchanged.

451 critical frozen JSON input/provenance files (18.2MB) formerly under `/tmp`
also have SHA-verified persistent copies and an original-path mapping in
`/extra1/bmartelli/context-str-artifacts/clip-specialized-input-snapshots-v1/manifest.json`.
Dataset images and model weights retain their separate pinned locations.

The frozen paper-style MPNet/caption baseline is also replayed with its original
T1/T3 decisions and gates. The original tau0.8 point and the equal149-image TEST
budget must remain separate: the former may defer every image to DeepSolo.
Its captions/results have a separate persistent mapping,
`paper_baseline_manifest.json` (1283 JSON files,47.0MB), under the same snapshots
directory. TRAIN125 remains a diagnostic holdout of the local segmentation
experiment; upstream DeepSolo training includes Total-Text training data, so
these TRAIN numbers do not establish unseen-data generalization.

## Literature interpretation and further hypotheses

[CLIPTER](https://openaccess.thecvf.com/content/ICCV2023/html/Aberdam_CLIPTER_Looking_at_the_Bigger_Picture_in_Scene_Text_Recognition_ICCV_2023_paper.html)
injects whole-image visual context into crop recognition through trained gated
cross-attention. [CLIP4STR](https://arxiv.org/abs/2305.14014) adapts a pretrained
vision-language model for character recognition; the inspected
[VamosC repository](https://github.com/VamosC/CLIP4STR) explicitly labels itself
a third-party implementation. [oCLIP](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1958_ECCV_2022_paper.php)
uses character-aware instance-level alignment rather than assuming generic
caption retrieval verifies spelling. These are distinct trained OCR systems,
not names for the generic DataComp/DFN encoders or the routers measured here.

`ASSUMPTION` hypotheses, not tested claims: compare `caption + transcription`
against the *same caption* plus a near-spelling alternative, so a good generic
scene description cannot independently certify a wrong word. Combine a local
spelling check with whole-image coverage evidence to detect omitted text; a
minimum over predicted crops alone cannot identify words never proposed.
The present thirteen-family experiment tests word/full-transcript templates and
spelling controls; it does not append BLIP2 captions to CLIP encoder prompts.
