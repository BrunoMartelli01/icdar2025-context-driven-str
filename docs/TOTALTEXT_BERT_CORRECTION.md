# BERT after PARSeq: Total-Text letterbox

**APPROXIMATE / approximate reproduction**, TRAIN-holdout diagnostic.
Not strict reproduction, Table5 measurement, or TEST result.

## Actual project path

`models/unet_adv.py` segments images. v4 targets come from individual-word
regions; letterboxing preserves aspect ratio with 512×512 inputs/native 256×256
masks. OpenCV external contours map rectangles through inverse letterbox to
original RGB. PARSeq retains crop text, token probabilities, checksums. The new
component operates **after** these outputs without changing masks, contours,
crops, or PARSeq.

After the user's “latest letterbox variant” instruction, the chosen reference
is single-head `word_f1_control`, selected by TRAIN-validation localization F1,
epoch 68. This identifies a specific variant, not best observed OCR accuracy.
The auxiliary-separator-head branch is another model, excluded here.

Source:`context-str-runs/totaltext-letterbox-word-f1-separator-parseq-val-v1/`.
Original evaluator's declared OCR reuse:
`context-str-runs/totaltext-v4-image-frequency-ablation-v1/validation-pad0-parseq-v1/ocr/`.
Preparation requires both directories explicitly. Reader checks report, protocol,
masks, checkpoint, bboxes, RGB hashes, text, OCR records. Missing/conflicting caches
block execution.

Existing protocol uses mask threshold 0,5, zero padding, contours without area
filter/top10 cap: practical choices distinct from Algorithm1. Word-F1 selection
and letterbox are also approximations. Local PDF sections3.3–3.5 specify
localization, BLIP-2, MPNet/Levenshtein scoring, DeepSolo fallback, not a BERT corrector.
Strict, MPNet, BLIP-2, DeepSolo paths stay unchanged.

## Model and research

Initial model:**google-bert/bert-base-uncased**, approximately110M parameters,
revision `86b5e0934494bd15c9632b12f734a8a67f723594`, FP32, frozen. Revision checked
against [official model API](https://huggingface.co/api/models/google-bert/bert-base-uncased).
Download only that revision's safetensors/config/tokenizer; record each SHA256.
No local training, quantization, automatic precision changes, or package installation.

Sources/decisions:

| Primary source | Idea used or limitation |
| --- | --- |
| [Salazar et al., Masked Language Model Scoring, ACL 2020](https://aclanthology.org/2020.acl-main.240/) | Score alternatives with frozen-MLM pseudo-log-likelihood. Paper concerns ASR/NMT; OCR transfer is a hypothesis. |
| [Kauf and Ivanova, A Better Way to Do Masked Language Model Scoring, 2023](https://arxiv.org/abs/2305.10588) | Mask current/all later subtokens in the same word, preventing copying future candidate subtokens. |
| [Fang et al., ABINet, CVPR 2021](https://arxiv.org/abs/2103.06495) | Separate visual/language evidence with bidirectional context. This corrector neither implements nor reproduces ABINet. |
| [Jayanthi et al., NeuSpell, EMNLP 2020](https://aclanthology.org/2020.emnlp-demos.21/) | BERT adapted to English spelling correction is a future alternative; its validation does not establish Total-Text efficacy. |

Also inspected:[`murali1996/bert-base-cased-spell-correction`](https://huggingface.co/murali1996/bert-base-cased-spell-correction),
visible revision `d2a5bbccc41a0f4ff5e7c16e1c3b8d96ba8883b2`.
[Original config](https://huggingface.co/murali1996/bert-base-cased-spell-correction/blob/main/config.json)
has100.002 classification labels.
[Author snippet](https://github.com/neuspell/neuspell/blob/master/scripts/huggingface/huggingface-snippet-for-neuspell.py)
averages subtoken representations/applies a word-level head with external
`vocab.pkl` mapping. Loading as masked LM would initialize a different head,
not reproduce that corrector. Original BERT-base MLM was chosen first because
its decoder is directly verifiable. **It is not OCR-fine-tuned BERT.** NeuSpell
head/mapping need a separate adapter/qualification, not silent substitution.

## Frozen experiment

Config:`configs/approximate/totaltext_v4_bert_correction_v1.json`.
Compare `word_only`/`local_ocr`; add `local_ocr_caption` if every image has a
caption. All conditions score identical candidates to isolate context effects.

- KEEP is original PARSeq text with original case.
- Correct only ASCII words of at least three characters. Numbers, punctuation,
  empty strings, multiword crops, internally capitalized names remain unchanged.
  Unchanged does not mean correctly recognized.
- Native minimum PARSeq probability ≥0,95: KEEP. Heuristic protection, not a
  calibrated correctness probability.
- Alternatives: whole-word BERT vocabulary and words observed in neighbors/
  captions; Levenshtein ≤1 for short words,≤2 from six characters, relative
  distance ≤0,3; maximum 24 candidates including KEEP. Order: distance, context
  presence, lexicographic. Cap may exclude correct words; no GT/Total-Text dictionary.
- Neighbors: vertical overlap ≥0,5, horizontal gap ≤8 heights, at most three words
  per side. Frozen original text; no iterative corrected-word propagation.
- Captions: first80 whitespace terms, declared budget; scene in BERT segment one,
  local writing in segment two. Above256 tokens or unknown/special targets
  causes explicit abstention, without further truncation/output filling.

Candidate score: mean subtoken log-scores under PLL-word-l2r masking. Sum,
individual log-scores, token count also saved. Kauf/Ivanova use sentence-level
sum; target-only mean is our **ASSUMPTION**, not calibrated OCR posterior.

```
gain = mean_BERT(candidate) - mean_BERT(PARSeq) - 1.0 * edit_distance
correct only if gain > 1.5; KEEP on ties
```

All config parameters are fixed, not optimized on BERT results. No TEST search/
automatic promotion. Edit limits and PARSeq protection retain visual evidence;
scoring has no PARSeq likelihood for new strings. Residual risks include proper
names, non-English languages, common-word bias, merged crops, incorrect geometric
context, and text invented by captions.

## PAPER command

From repository root, using the existing main environment:

```bash
bash scripts/run_totaltext_bert_correction_host.sh \
  --source-dir /extra1/bmartelli/context-str-runs/totaltext-letterbox-word-f1-separator-parseq-val-v1 \
  --reuse-ocr /extra1/bmartelli/context-str-runs/totaltext-v4-image-frequency-ablation-v1/validation-pad0-parseq-v1/ocr \
  --hf-cache /extra1/bmartelli/context-str-artifacts/huggingface \
  --output-dir /extra1/bmartelli/context-str-runs/totaltext-letterbox-bert-correction-v1 \
  --device cpu
```

GPU requires explicit `--device cuda:0`; unavailability/OOM aborts. Only BERT
loads during correction. Record model, dtype, mask batch, input limit, device,
software, seed, determinism, load/correction times, CUDA allocated/reserved peaks.
CPU VRAM is `null`, not measured zero. Timing excludes U-Net, contours, PARSeq,
caption, DeepSolo. Loading permits only config-listed inactive pooler/NSP keys
and records every key; MLM/backbone missing/mismatched keys are fatal.

Stages:`prepare`,`fetch-model`,`infer`,`evaluate`. Preparation/evaluation can
repeat; inference resume checks each shard's provenance/checksum. Code/config/
device/caption changes require a new directory/version. Launcher downloads
weights,**does not install/upgrade dependencies**.

`infer` reads `inputs.json`, never `evaluation_labels.json`. Sample IDs/hashes
link artifacts only, never enter BERT tokenizer. Each crop preserves bbox, order,
original text, alternatives, scores, choice. `results.json` requires every image/
condition; errors under `errors/` propagate. Failed samples stay in denominators.

Optional captions via `--captions FILE`, schema:

```json
{
  "schema_version": 1,
  "model_id": "model-used-to-describe-Total-Text",
  "revision": "40 hexadecimal characters",
  "model_files_sha256": {"weights-file": "64 hexadecimal characters"},
  "generation": {"all_decoding_parameters": "resolved values"},
  "samples": [
    {"sample_id": "total_text:14", "image_sha256": "64 hexadecimal characters", "text": "Scene description"}
  ]
}
```

Exact125-image coverage/matching checksums required. Existing ICDAR13 captions
cannot be reused. No new Total-Text captions were generated in this change.

## Metrics and verified state

Micro aggregation over **983 care words**, including words without crops.
Don't Care already excluded in original qualified shards. Maximum one-to-one
bipartite bbox IoU≥0,5 plus exact Unicode-casefold matching, punctuation retained.
Also report case-sensitive exact and RapidFuzz casefold ratio≥0,8. Empty outputs
do not match nonempty GT. Existing geometry/evaluation reused; not official
Total-Text polygon spotting protocol.

Gains/harms are differences between before/after correct-GT sets; their difference
must equal exact-total improvement. GT, prediction, per-image indices stay in
evaluation only.

Real `word_f1_control` preflight:**125 images,949 crops,983 words**,503 geometric
IoU≥0,5 matches,**384 exact casefold(39,06%)**,372 case-sensitive exact,475 fuzzy.
555 outputs have eligible form; native protection leaves137 possible targets
before BERT candidate availability. Original control, not a BERT result.

46 CPU tests passed, including a tiny random BERT as engineering fixture only,
subtoken masking, GT isolation, config namespaces, thresholds, checksums, historical
cache, abstention, matching, denominators, PARSeq/metric regressions. First preflight
exposed upstream-hash serialization; second exposed external OCR cache. Both
were fixed/tested while respecting the original contract.

Preflight:`/tmp/totaltext-letterbox-bert-preflight-v3/`. BERT weights were absent
then; user later completed launcher on networked host. Real CPU inference/
verification are complete. CUDA, GPU VRAM, caption effects, TEST remain unmeasured.
Repeatedly explored TRAIN holdout is not independent validation.

## Completed real results —6 October2026

Original complete report:
`/extra1/bmartelli/context-str-runs/totaltext-letterbox-bert-correction-v1/results.json`,
SHA256 `a7ae0a603323c4d2d6696b7383dbc6c6dd2646432f8200f4e52e8f674ee788ef`.
250 shards(125 per condition), hashes, crops, gates, recomputed metrics on original
denominator verified. No runtime errors/token-limit/UNK abstention; no omitted
sample. Backbone/MLM have no missing/mismatched keys; only four expected inactive
pooler/NSP keys excluded. Safetensors SHA256:
`68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3`.

| Rule | Exact / 983 | Gained | Harmed | Changed crops | CPU correction seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen PARSeq | 384 | 0 | 0 | 0 | — |
| BERT `word_only`, protection ≥0,95 | 381 | 0 | 3 | 10 | 4,41 |
| BERT `local_ocr`, protection ≥0,95 | 383 | 0 | 1 | 8 | 4,88 |
| BERT `word_only`, protection ≥1,0 | 341 | 1 | 44 | 62 | 13,31 |
| BERT `local_ocr`, protection ≥1,0 | 363 | 3 | 24 | 42 | 16,38 |

Without context, BERT harms `Kong → King`,`Attire → Lattice`,`MOTEL → MODEL`;
local context preserves the first two but changes `MOTEL → HOTEL`. Other changed
crops lack IoU≥0,5 GT matches and are not exact gains/harms under this metric.
That does not establish semantic correctness or harmlessness of every change.

### Protection diagnosis and ablation

GT-assisted audit finds theoretical420 exact words by one-to-one-per-crop
selection among saved candidates. Additional36 words form an **oracle**, not a
usable selector. In29/36 cases, native-confidence protection blocks correction,
including `MERICAN → AMERICAN`,`WORL → WORLD`; in the other seven, BERT keeps
original text. High produced-character probability does not check missing letters.

To isolate this, a **post-hoc TRAIN ablation** used
`configs/approximate/totaltext_v4_bert_confidence_ablation_v1.json`. Only
`protect_min_token_probability` changes0,95→1,0: exact saturation stays protected;
all other rules identical. Not predeclared before the first run or independent
validation; no tuning, grid, TEST selection. Original run preserved.

Complete ablation report:
`/tmp/totaltext-letterbox-bert-confidence-ablation-v1/results.json`, SHA256
`6919040e863c2b18f0727a0458957ed7e9e0bf4c244ba5545dc3c9f3534de7d5`.
With context, it recovers `EVER → EVERY`,`Oper → Open`,`OPICAL → TROPICAL`, but
harms24 formerly correct words. Net−21 shows weaker protection does not solve
selection in this configuration. Language plausibility does not sufficiently
establish that the image contains the candidate.

Both runs use FP32 BERT, four CPU threads, batch 16; timings exclude loading/all
upstream models. VRAM peaks are `null`, not GPU measurements. Geometry unchanged:
503 IoU≥0,5 matches in every condition.46 CPU/regression tests passed again.
No correction rule is promoted from these negative results. Spelling-fine-tuned
correctors or genuinely Total-Text captions remain to be qualified, along with
future independent validation.

Machine-readable audits:
`/tmp/totaltext-letterbox-bert-completed-audit-v1.json` and
`/tmp/totaltext-letterbox-bert-confidence-ablation-audit-v1.json`.
