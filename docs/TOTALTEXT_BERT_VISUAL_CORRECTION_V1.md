# Total-Text BERT visual correction — TRAIN development

APPROXIMATE / approximate reproduction. User-requested BERT extension, not the
strict paper pipeline, TEST, independent validation or Table5. The original latest
letterbox mask -> contours -> PARSeq source is unchanged:125 images,949 immutable
originalRGB crops,983 care GT,503 geometrically localizable words. The denominator
of the requested80–90% target remains unresolved, so report all-GT accuracy,
localized-word accuracy and baseline error recovery separately.

## Evidence and protocol

Config: `configs/approximate/totaltext_bert_visual_correction_v1.json`.
Scientific choices: `TT-BERT-VISUAL-V1` in `docs/PAPER_AMBIGUITIES.md`.
Paper sections3.3–3.5 motivate localized recognition and context; their strict
components remain BLIP-2/MPNet/lexical scoring/DeepSolo. This alternative stays
isolated in APPROXIMATE and imports the old sealed correction code unchanged.

[PARSeq](https://arxiv.org/abs/2207.06966) explicitly discusses language-only
rectification of correct predictions when image evidence is absent.
[LevOCR](https://arxiv.org/abs/2209.03594) combines textual hypotheses and visual
features for iterative refinement; [ABINet](https://arxiv.org/abs/2103.06495)
models character probabilities for iterative correction. These are inspirations;
we do not train or reproduce their alternative architectures.

Original source/checkpoint/split/caption hashes are inherited from the complete
`/tmp/totaltext-bert-methods-full-v1` run, with resultsSHA256
`1c8539835200573ef50ccc2bbb5147f3b249ec54e6f8e73b5a9d319bf5733620`.
The125 genuine Florence-2-large-ft captions remain image/SHA-bound; no captions
from another dataset are used. Model/revision/file and RGB manifests are pinned.
No Total-Text-derived dictionary, ground-truth prompts or annotation-driven
candidate generation is allowed. No TEST is read and no fine-tuning occurs.

## Predeclared correction battery

Prepare validates the full upstream run, verifies every RGB hash and creates
observable inputs plus a separate evaluation-label file. Inference cannot read
that label file. Use original OCR neighbours for every context, never recursively
corrected text. Unknown/unavailable reference scores KEEP; loading/runtime/
nonfinite errors propagate, never become alternate-model predictions.

The shared candidate pool (`ASSUMPTION`) includes exact OCR KEEP, the previously
measured insertion anchor, six frozen NeuSpell top10 lists, Florence caption
words,32 neighbours from the authenticated external pretrained vocabulary and
PARSeq causal AR beam10. Newly generated lexical alternatives allow1 edit for
source length<6 and3 otherwise, relative edit<=0.5. Preserve the insertion anchor
regardless of that new edit bound. Beam hypotheses may exceed lexical bounds.
Sources must be alphabetic single words2–25 characters; mixed-case names,
numbers/punctuation/multiword strings are protected. Pool maximum96. External
lexicon filtering retains74716 distinct alphabetic labels in source order; this
order is not a calibrated frequency prior.

With actual trained PARSeq-S, encode each identical32x128 crop once and score
complete candidates using causal teacher forcing, includingEOS. This AR sum
is not the native post-refinement confidence. Verify every beam's scored path
against a separate complete teacher-forced pass. Retain original/insertion
references plus strongest AR alternatives in a shared24-candidate BERT shortlist.
The shortlist can exclude a correct weak-visual hypothesis; audit its oracle
coverage separately from the full generated pool.

Six fixed controls: previous insertion KEEP; visual-only with original anchor;
and visual+BERT with original/insertion anchors, each under local OCR and local
OCR+Florence. All share candidate availability, including caption-derived words
in the local-score control. Thus the scene comparison isolates scoring context,
not removal of caption information from candidate generation.

For the fusion controls:

```
dv = AR_sum(candidate) - AR_sum(anchor)
dc = BERT_mean_PLL(candidate) - BERT_mean_PLL(anchor)
require dv >= -1.0
accept greatest dv + 0.5*dc only if strictly >0.1
```

Ties KEEP. Visual-only requires a strict AR gain>0.1. Fusion combines different
length/tokenization-biased scales (`ASSUMPTION`), not a calibrated posterior or
paper alpha/beta/tau. BERT is genuine frozen bert-base-uncased, pinned revision
`86b5e0934494bd15c9632b12f734a8a67f723594`, using target-only meanPLL-word-l2r
and a separate caption segment. No thresholds are swept after these new results.

## Results

The six controls completed on all125 images and949 crops, with no failed or excluded samples.

| Control | Exact /983 | Exact /503 localized | Recovered | Damaged | Net |
| --- | ---: | ---: | ---: | ---: | ---: |
| PARSeq baseline |384|76.3%|—|—|0|
|insertion_control|396|78.7%|14|2|+12|
|insertion_visual_local_ocr|377|75.0%|2|9|-7|
|insertion_visual_local_ocr_caption|378|75.1%|1|7|-6|
|original_visual_local_ocr|377|75.0%|2|9|-7|
|original_visual_local_ocr_caption|378|75.1%|1|7|-6|
|visual_only|381|75.7%|0|3|-3|

The new fusion does **not** improve the measured insertion control. It tends to undo
its missing-letter completions. The original-crop AR/LM evidence is insufficient
for this fixed rule; no score weights are tuned to these results.

GT-dependent candidate ceilings: initial pool432, expanded pool442, visual
shortlist442. The shortlist loses no aggregate oracle coverage; even perfect
selection would be87.9% of503, or45.0% of983. These are upper bounds, not achieved
accuracy. The preserved insertion control is396/983=40.3%,396/503=78.7%, with
14/119=11.8% baseline localized errors recovered and2 correct words damaged.

Original-crop results: `/tmp/totaltext-bert-visual-v2/results.json`, SHA256
`888a2098ec3f5222515eaf53a06f5cde8472b019f5d0706fdda5241a12394372`.


Exact Unicode casefold retains punctuation, uses deterministic maximum one-to-one
bboxIoU>=0.5 matching and all983 care GT. Unlocalized words stay in the all-GT
denominator; no sample is excluded. Also save case-sensitive/fuzzy metrics and
per-image predictions, recovered/damaged GT indices and candidate sources.
GT oracle coverage is a diagnostic upper bound, never achieved prediction accuracy
and never an input to model inference or per-sample selection.


## Separate padding experiment

The negative fusion motivates a clipping hypothesis, not a proved failure cause.
Local paper section3.3 explicitly pads boxes; the chosen amount is an ASSUMPTION:
p=max(1,ceil(0.25*original_bbox_height)), all sides clamped to the original image.
Config: `configs/approximate/totaltext_bert_padded_evidence_v1.json`;
scientific record `TT-BERT-PADDED-V1`. This changes recognition pixels and must
not be described as an identical-crop comparison. All original949 localization
boxes and503-match evaluation geometry stay fixed. Save both original/padded RGB
hashes, padding pixels, padded bbox and actual native predictions per block.

Five predeclared controls: insertion KEEP, padded native, padded visual-only,
and padded visual+BERT local/Florence with the same original-reference rule.
Reuse the previous shared24-candidate pool and append genuine padded-native
words when absent. Reuse BERT likelihood only for exact candidate/context/model
identity; new candidates (and any unscored original reference) require actual
fresh BERT scores. No GT enters inference, and candidate availability remains
shared across the context controls. Padding can expose adjacent text and damage
otherwise correct transcriptions; there is no threshold sweep or result promise.

All125 images/949 localizations completed, no failures or exclusions.

| Control | Exact /983 | Exact /503 localized | Recovered | Damaged | Net |
| --- | ---: | ---: | ---: | ---: | ---: |
|insertion_control|396|78.7%|14|2|+12|
|padded_native|403|80.1%|34|15|+19|
|padded_original_visual_local_ocr|401|79.7%|30|13|+17|
|padded_original_visual_local_ocr_caption|406|80.7%|35|13|+22|
|padded_visual_only|403|80.1%|34|15|+19|

The best predeclared padded/Florence arm reaches **406/503=80.7% conditional
localized-word accuracy**,406/983=41.3% all-GT accuracy. It recovers35 of119
baseline localized errors (29.4%) and damages13 baseline-correct words, net+22.
This is not80–90% error recovery. Relative to padded local BERT it gains5 exact
matches; relative to padded native PARSeq it gains3. Thus most gain here comes
from changed crop pixels, with a smaller measured contribution from scene
contextual selection. Reused TRAIN development cannot establish independent
transfer, strict reproduction or TEST accuracy.

The shared padded pool's GT-dependent oracle ceiling is444/503=88.3%;115 new
candidate strings receive fresh BERT likelihoods under both contexts. The other
exact-string scores retain their sealed cache values. No padding amount,
threshold or individual sample is changed after evaluating these results.
All-GT80–90% remains impossible with the unchanged503/983 localization ceiling.

Padded result: `/tmp/totaltext-bert-padded-v1/results.json`, SHA256
`d8043bb487c0e9915bd0fdc8bc23af5454e3f73c051413beaab68753d1672a75`.
ProtocolSHA256: `9bdd7c65a5d762552e849d54b66574d0e4138ced6be76aa44ff3e2338ee12c4c`.


## Runtime and verification

Separate CPU FP32 processes, fourthreads, deterministic seed20261006, TF32off,
scorebatch<=16; cached word heads and Florence. This is `HARDWARE_DEVIATION`.
PARSeq-S has23832671 parameters, input[1,3,32,128]. No model/checkpoint/source
substitution or non-strict learned-weight load occurs. Reuse existing isolated
einops0.8.1 after hash/version checks; no packages or environments are installed.
GPU allocated/reserved peaks are null because this is CPU execution. These
operational correction timings are not a paper end-to-end benchmark or GPU
qualification; caption/head generation costs are additional.

129 focused CPU tests pass, including26 new beam/causal mask/EOS probability,
future-token and padding independence, candidate/reference bounds, ties,
nonfinite/unknown reference, namespace isolation and corruption/one-to-one-oracle
checks. Actual PARSeq visual smoke passes, and actual BERT smoke succeeds without
any evaluation-label or baseline file in its output directory. All125 visual
image shards pass, including6490 complete beam/teacher-forcing comparisons.

Initial development failures are preserved: a missing insertion reference during
prepare (`SOCIE -> SOCIETY`) exposed an incorrect application of the new short-word
bound to an immutable control. It was fixed with a regression test before the
sealed v2 inference run. A smoke harness initially copied the wrong shard directory;
corrected copying of the `inference/visual` layout passed. An initial read-only
PARSeq probe used an unsupported source-type string and was corrected to the
qualified `github_commit_archive`. No failed experiment is counted as a result.
The old BERT/NeuSpell implementations/configs/reports remain unchanged.

An independent audit validates500 image inference shards across the original and
padded experiments, all949 crop identities per run,6490 complete beam paths,
31266 scored AR candidate paths, all contextual scores, original-neighbour/caption
identity, shared candidate pools, selection formulas, one-to-one metrics and
all recovered/damaged counts. The115 padded-native candidate strings are genuinely
scored, and all reused BERT values retain their exact sealed values. Full padded
BERT inference runs successfully with evaluation-label/baseline files withheld;
their original hashes are verified before restoring them for evaluation.

Audit: `/tmp/totaltext-bert-final-audit-v1.json`. Operational CPU totals excluding
loads: original AR47.26s/BERT740.63s; padded AR/native35.51s/fresh BERT12.86s.
The padded correction also reuses the earlier740.63s BERT work and845.29s caption
work; these additional costs are not hidden as an end-to-end speed measurement.

The selected TRAIN-development artifact is
`/tmp/totaltext-bert-padded-v1/corrected-ocr-padded-florence.json`:125 image records,
949 original/corrected strings, source/localization and padded-image hashes,
decisions and captions, with no GT fields. It is explicitly labelled TRAIN-selected,
not automatically promoted to a validated or strict configuration.


## Run again using existing models

The wrapper installs nothing. All dataset/cache/checkpoint paths are explicit.
Use a new output directory if code/config/device changes; conflicting sealed files
fail rather than overwrite. Full evaluation refuses absent partial-inference shards.

```bash
bash scripts/run_totaltext_bert_visual_correction_host.sh \
  --upstream-dir /tmp/totaltext-bert-methods-full-v1 \
  --asset-dir /extra1/bmartelli/context-str-artifacts/neuspell-word-head-v1 \
  --rgb-manifest /extra1/bmartelli/datasets/context-str/scene-segmentation-v1/manifests/total_text_train.yaml \
  --data-root /extra1/bmartelli/datasets/context-str/scene-segmentation-v1 \
  --parseq-source /extra1/bmartelli/context-str-artifacts/parseq/parseq-1902db043c029a7e03a3818c616c06600af574be \
  --parseq-checkpoint /extra1/bmartelli/context-str-artifacts/parseq/parseq-bb5792a6.pt \
  --parseq-archive /extra1/bmartelli/context-str-artifacts/parseq/parseq-1902db043c029a7e03a3818c616c06600af574be.tar.gz \
  --runtime-deps artifacts/diagnostics/lavis-captioner-candidates-20260914/runtime-deps \
  --hf-cache /extra1/bmartelli/context-str-artifacts/huggingface \
  --output-dir /tmp/totaltext-bert-visual-rerun-v1 \
  --device cpu
```

Stages: prepare -> visual -> bert -> evaluate. Visual and BERT support explicit
`--max-images1` smoke runs; partial coverage cannot produce full results.


The padding stages reuse the sealed original-crop visual run. Example preparation:

```bash
.venvs/context-str-main-py311/bin/python -m scripts.evaluate_totaltext_bert_padded_evidence prepare \
  --config configs/approximate/totaltext_bert_padded_evidence_v1.json \
  --upstream-dir /tmp/totaltext-bert-visual-v2 \
  --output-dir /tmp/totaltext-bert-padded-rerun-v1
```

Then run `visual` with the same explicit PARSeq source/checkpoint/archive/existing
einops dependency path and device as above, `bert` with output-dir/hf-cache/device,
and `evaluate` with output-dir. The complete CLI is available through `--help`.
Padded outputs require a separate directory; they never overwrite the original
identical-crop experiment or its model/caption outputs.


Complete padded wrapper command with the existing sealed upstream run:

```bash
bash scripts/run_totaltext_bert_padded_evidence_host.sh \
  --upstream-dir /tmp/totaltext-bert-visual-v2 \
  --output-dir /tmp/totaltext-bert-padded-rerun-v1 \
  --parseq-source /extra1/bmartelli/context-str-artifacts/parseq/parseq-1902db043c029a7e03a3818c616c06600af574be \
  --parseq-checkpoint /extra1/bmartelli/context-str-artifacts/parseq/parseq-bb5792a6.pt \
  --parseq-archive /extra1/bmartelli/context-str-artifacts/parseq/parseq-1902db043c029a7e03a3818c616c06600af574be.tar.gz \
  --runtime-deps artifacts/diagnostics/lavis-captioner-candidates-20260914/runtime-deps \
  --hf-cache /extra1/bmartelli/context-str-artifacts/huggingface \
  --device cpu
```
