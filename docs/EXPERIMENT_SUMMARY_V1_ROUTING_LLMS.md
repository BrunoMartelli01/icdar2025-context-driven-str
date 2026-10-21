# Experiment summary: V1, contextual routing and LLM correction

**Updated:** 2026-09-21. **Scope:** the main completed experiment families relevant to the current OCR baseline, plus the newly prepared battery.

All practical extensions below are `PRACTICAL_APPROXIMATION`, not strict paper reproduction. Training-free describes model use: selecting parameters on TRAIN is still calibration. Repeated TRAIN exploration is not independent validation. No inference was run to write this summary.

## 1. How V1 works

### 1.1 Frozen image-to-word pipeline

Here, **V1 means `practical-remap-v1` localization followed by PARSeq**, not the paper-faithful Algorithm1 path and not an LLM version.

```text
Original RGB image
  → frozen author-protocol AG-U-Net / ResNet-50
  → native 128×128 probability map
  → binary mask at P ≥ 0.5
  → nearest-neighbor binary backprojection to original image size
  → raw 8-connected components
  → geometric graph grouping
  → original-component union boxes, padding, area sorting, top 10
  → crops from original RGB
  → one PARSeq transcription per crop
```

- Segmentation input: 256×256, qualified ImageNet normalization; native logits: 128×128.
- Frozen checkpoint: epoch0098, selected in the earlier author-protocol training run. SHA256: `b450f19e79f649f8873a8b8db0d9a25ab0a7f144ac5f8676a2a80d5c1d8fd2f2`.
- In the audited V1 baseline, component extraction/grouping operates on the **backprojected binary mask**, not directly on native128. Native-resolution and bilinear alternatives were separate diagnostics.
- PARSeq is a word recognizer. Feeding it a full image does not make it a multiword detector or a sentence transcriber.

### 1.2 Grouping rule

Each raw component is a graph node. An edge requires all four frozen geometric gates:

| Feature | V1 gate |
|---|---:|
| Horizontal gap / mean pair height | ≤0.35 |
| Vertical overlap / smaller height | ≥0.70 |
| Component centroid-y distance / mean pair height | ≤0.50 |
| Larger/smaller component height | ≤2.50 |

Graph connected components become groups, including valid singletons. Their boxes are unions of original component boxes. Add **6 original-image pixels** of padding, clip to image borders, sort by summed real foreground pixel area, retain at most10, and crop RGB. V1 groups raw components; it does not insert Algorithm1's pre-group `Amin=150` filter.

The gates were selected using TRAIN geometry, without OCR. V1 is needed because ICDAR13 Task2.2 supervision is **glyph/pixel foreground**, not filled word rectangles: one connected component is not necessarily one word. Transitive graph connections can still undermerge or overmerge words.

Implementation: [grouped_word_blocks.py](../localize/grouped_word_blocks.py). Frozen parameters: [V1 config](../configs/approximate/pipeline_icdar13_practical_remap_v1.yaml).

### 1.3 Baseline and localization evidence

| TRAIN229 measurement | Result |
|---|---:|
| Care GT words | 849 |
| Retained V1 crops | 1,203 |
| GT rows with an associated retained crop | 745 |
| Fixed-spatial PARSeq exact, casefold | **502/849 = 59.13%** |
| Fixed-spatial RapidFuzz ≥0.8 | 578/849 = 68.08% |
| Image-level lexical matches | **503/849 = 59.25%** |
| Lexical precision / F1 | 41.81% / 49.03% |
| GT words with proposal IoU≥0.5 | 568/849 = 66.90% |
| Proposal precision / F1 at IoU0.5 | 47.22% / 55.36% |
| Crop wordness: good / partial / multiword / unmatched | 552 / 130 / 37 / 484 |
| PARSeq on GT word boxes, diagnostic only | **836/849 = 98.47% exact** |

The crop gap is substantial. On80 selected V1 failures, V1 was0/80 exact, recovered diagnostic crops78/80, and GT boxes78/80. This is a **selected, GT-informed diagnostic**, not an inference-safe improvement or a full-TRAIN rate.

Earlier localization work covered native/original coordinate frames, area/padding, interpolation/thresholds, graph/rescue/line methods, classical closing/RLSA, persistence and candidate proposals. These did not change the frozen crop population used in the LLM comparisons below. Better diagnostic crop coverage does not automatically provide a safe candidate selector.

### 1.4 Metric distinctions used throughout

| Metric | What counts as correct? |
|---|---|
| Fixed-spatial exact | The transcription of the historically assigned maximum-IoU intersecting V1 crop equals that GT word, casefolded. This metric does **not** itself require IoU≥0.5. |
| Geometric coverage | A retained proposal reaches the stated IoU threshold, regardless of OCR. |
| Lexical word matches | Per-image casefolded prediction/GT multiset intersection; duplicates capped, no spatial or ordering constraint. |
| Image membership | One selected string occurs among an image's GT strings. It says nothing about recovering every word. |
| Complete-image accuracy | The predicted and reference word multisets match exactly. |
| Oracle | GT selects a crop/candidate/operation after generation; a ceiling, not a deployable selector. |

TRAIN has229 images/849 GT words; TEST has233 images/1095 words. Beam studies use745 eligible GT rows attached to700 unique crops. **502/745 and502/849 are different rates.** Missing predictions remain failures in full-population metrics.

## 2. Contextual routing experiments

### 2.1 Intended decision

- `T1`: PARSeq on the whole image, generally a single word-like reading.
- `T3`: either a particular crop reading, or the reading-order concatenation of crop readings in the joined-text experiment.
- `T2`: the **whole** scene caption in the whole-caption protocol.

```text
S1 = cosine(MPNet(T1), MPNet(T2))
S3 = cosine(MPNet(T3), MPNet(T2))
k  = argmax(S1, S3)                     # T1 wins exact ties
Lk = RapidFuzz.ratio(Tk, T2) / 100
C  = 0.6 * Sk + 0.4 * Lk
accept Tk if C ≥ tau; otherwise use DeepSolo
```

This is the **selected-candidate paired** formulation: choose by semantic score, then use the lexical score of that same candidate. It does not independently maximize `L1` and `L3`. Earlier repository/independent-max variants were diagnostic alternatives, not equivalent formulas. The equality/tie handling is an explicit implementation convention.

Per-block acceptance, joined-text acceptance and image-level DeepSolo invocation are different units. Accepting some blocks does not necessarily avoid running DeepSolo on their image.

### 2.2 What was tested

| Experiment family | Result / interpretation |
|---|---|
| Original BLIP2-Flan-T5-XL captions | Repetition, short answers and prompt echo motivated caption diagnostics. |
| Anti-repetition decoding | Improved caption behavior, but did not solve word correctness or routing calibration by itself. |
| Captioner comparison | BLIP-base-COCO, BLIP2-ViT-L, InstructBLIP, Florence2-base/large, plus original and anti-repetition Flan profiles: seven profiles in the final TRAIN report. |
| Whole-caption versus local caption windows | Local windows accepted more strings at tau0.8, but change the comparison unit and can reward isolated lexical overlap. |
| Per-block versus joined T1/T3 routing | Both evaluated; neither caption quality nor a higher score guarantees complete text recovery. |
| Confidence-distribution audit | Correct and incorrect predictions overlap strongly in C. A threshold that rejects every observed TRAIN error retains very few correct units. |
| TRAIN-calibrated fallback budget | InstructBLIP paired routing calibrated to approximately30% fallback, then evaluated on TEST with tau frozen. |

At **tau0.8 with whole captions**, all seven profiles accepted **zero** of1203 TRAIN blocks and zero of229 joined-image outputs. InstructBLIP local-window routing accepted274 blocks or54 joined-image outputs, but these are **pre-fallback acceptance diagnostics**, not final accuracy.

For whole-caption per-block scores, a threshold with zero observed TRAIN text errors retained only0–10 blocks across the seven captioners; InstructBLIP retained9. This is not a useful general correctness separator.

### 2.3 The historical “88%” and the actual word results

The persistent TEST routing suite reproduced:

| TEST configuration | Fallback images | Selected-string image membership | Lexical word matches |
|---|---:|---:|---:|
| Algorithm1 + original routing | 233/233 | 207/233 = 88.84% | 861/1095 = 78.63% |
| V1 + original routing | 233/233 | 207/233 = 88.84% | 861/1095 = 78.63% |
| Balanced mean + original routing | 233/233 | 207/233 = 88.84% | 861/1095 = 78.63% |
| V1 + local anti-repetition context | 142/233 | 201/233 = 86.27% | 515/1095 = 47.03% |
| Balanced + local anti-repetition context | 140/233 | 204/233 = 87.55% | 519/1095 = 47.40% |

The first three final outputs were entirely DeepSolo-driven. In the local-context suite, acceptance returned one PARSeq string while fallback supplied all DeepSolo strings; this explains why word coverage could fall sharply despite good membership precision. It is not the same protocol as the subsequent joined-text experiment.

The paper's88.8% cannot be equated to local88.84% merely because they round similarly. Recognition unit, matching, normalization, output reduction and the229/233 split discrepancy remain unresolved.

### 2.4 Frozen InstructBLIP fallback-budget experiment

`tau = 0.23055392728092478`, selected on TRAIN and unchanged on TEST. Word counts below use per-image casefold multisets, all raw DeepSolo strings on fallback.

| Measurement | TRAIN229 /849 words | TEST233 /1095 words |
|---|---:|---:|
| Selected PARSeq before fallback | 480 = 56.54% | 392 = 35.80% |
| DeepSolo standalone | **715 = 84.22%** | **861 = 78.63%** |
| Routed system | 621 = 73.14% | 707 = 64.57% |
| Fallback images | 68/229 = 29.69% | 116/233 = 49.79% |

**Conclusion:** the desired approximately80% word recovery at30% fallback was not achieved. DeepSolo did not suddenly drop from88% to78%; those numbers describe different metrics. The weak PARSeq branch and imperfect acceptance decisions limit the combined system.

Sources: [routing audit](ICDAR13_ROUTING_AUDIT.md), [persistent routing suite](../../../context-str-runs/icdar13-routing-suite-v1/results.json), [caption comparison](../../../context-str-runs/caption-routing-five-models-train-v1/results.json), [budgeted TRAIN](../../../context-str-runs/instructblip-budgeted-paired-e2e-v1/train-results.json), [budgeted TEST](../../../context-str-runs/instructblip-budgeted-paired-e2e-v1/test-results.json).

## 3. LLM experiments in more detail

### 3.1 Common objective and controls

Use cached PARSeq readings as primary evidence; ask a lightweight pretrained language model to repair or select text without retraining. Unless explicitly marked oracle, GT is excluded from inference and used only afterwards for evaluation. Caption context is fallible, not a transcription authority.

Models tested included FLAN-T5-base, Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct. Most later work used Qwen1.5B. Original crops, model revisions and intermediate outputs were frozen for controlled comparisons.

### 3.2 Whole-image JSON, plaintext and few-shot correction

The first whole-image JSON experiment frequently failed the requested output schema. FLAN had0/229 valid OCR-correction images; Qwen1.5B had88/229 OCR-only and83/229 OCR+caption. Fallback often preserved V1, so502 exact was **not evidence of successful correction**.

Plaintext made more answers usable, but exposed destructive edits and verbosity:

| Full-TRAIN experiment | Exact/849 | Recovered | Damaged |
|---|---:|---:|---:|
| Plaintext FLAN-base, OCR only | 142 | 0 | 360 |
| Plaintext FLAN-base, OCR+caption | 233 | 1 | 270 |
| Plaintext Qwen0.5B, OCR only | 468 | 1 | 35 |
| Plaintext Qwen0.5B, OCR+caption | 467 | 4 | 39 |
| Plaintext Qwen1.5B, OCR only | 443 | 5 | 64 |
| Plaintext Qwen1.5B, OCR+caption | 340 | 17 | 179 |
| Few-shot Qwen1.5B, OCR only | 432 | 2 | 72 |
| Few-shot Qwen1.5B, OCR+caption | 447 | 16 | 71 |

Few-shot demonstrations helped some caption-conditioned cases relative to their plaintext counterpart, but did not beat V1. Higher lexical recall sometimes came from producing many extra words, not better crop transcription.

### 3.3 Multi-fragment and structural reconstruction

We tested all-word reconstruction, one-word reconstruction with **GT-box-associated fragments**, and inference-safe geometry/context variants. These are not interchangeable protocols.

| Diagnostic | Main result | Interpretation |
|---|---|---|
| Initial ALL_WORDS | 80/826 exact matches;93 invalid outputs | Severe output/completeness failure; not full849 evaluation. |
| Initial oracle-associated ONE_WORD | 423/826 versus historical489;27 recovered/93 damaged | Even GT-assisted grouping did not make reconstruction reliable. |
| Later ONE_WORD context variant | 412/831 versus491;33 recovered/112 damaged | Still negative; denominator differs because of demonstration exclusions. |
| GT-selected feasible structural operations | Oracle627 lexical matches versus503 | Potential exists, but GT chooses operations. |
| Qwen structural selector | 450 lexical matches versus503 | Failed to select useful merge/split actions safely. |
| Separate split/merge caption ablations | No improvement over503 | Context did not fix structural selection. |

The later geometry/context reconstruction removed GT grouping and compared bbox-only versus explicit geometry with no caption, InstructBLIP, Florence-base and BLIP2-ViT-L. Best lexical count was506 with bbox-only+Florence, but predictions grew to2781 tokens and precision fell to18.19%. Explicit geometry+Florence produced490 matches and3119 tokens. **Narration can contain correct words without being correct OCR.**

Task2.2 care/ignore annotations can also omit visible characters; examples such as the visible “Ouch!” require annotation-aware interpretation. They do not justify silently changing the frozen GT or reporting relaxed punctuation scores as strict exact.

### 3.4 Conservative per-crop correction and prompt comparisons

Each crop was corrected independently, preserving V1 on invalid answers:

| Qwen1.5B condition | Exact/849 | Recovered | Damaged | Lexical matches |
|---|---:|---:|---:|---:|
| OCR only | 490 | 2 | 14 | 497 before later single-word filtering |
| OCR+caption, unfiltered | 418 | 12 | 96 | 544, but2032 tokens |
| OCR+caption, reject multiword answers | 486 | 12 | 28 | 493, with1203 tokens |
| Above + edit distance≤1 | 494 | 6 | 14 | 495 |
| Above + edit distance≤2 | 493 | 7 | 16 | 494 |

The next prompt comparison tried short instructions, reconstruction instructions, character audits and explicit evidence priority, each with/without caption. OCR-only exact ranged465–481 for the new prompts; caption-conditioned exact430–449. None beat502. Best new OCR-only prompt recovered5 but damaged26; more explicit instructions alone were insufficient.

### 3.5 PARSeq alternatives: useful candidates, difficult selection

True autoregressive beam candidates were more useful than alternatives derived from final refined logits:

| Candidate source; oracle coverage on745 eligible GT rows | Top1 | Top3 | Top5 | Top10 | Top20 |
|---|---:|---:|---:|---:|---:|
| PARSeq AR beam | 502 | 517 | 523 | **533** | 538 |
| Final-logit alternatives | 502 | 508 | 509 | 509 | 509 |

AR top10 contains correct readings for31 additional V1 errors. A feasible **one-decision-per-crop** oracle is532 rather than533 because GT rows can share a crop. Candidate coverage is not achieved recognition accuracy.

**Qwen likelihood reranking:** score existing strings instead of generating new ones. Using rank-based visual penalties still failed: best local-context setting333/745, with6 recovered/175 damaged; local+Florence383/745, with8 recovered/127 damaged.

**Actual visual-margin gate:** reuse cached AR log probabilities and Qwen likelihoods:

```text
dv = logP_PARSeq(candidate) − logP_PARSeq(actual V1 reading)
dc = meanLogP_Qwen(candidate) − meanLogP_Qwen(V1)
consider only dv ≥ −m
change only if dv + beta × dc > tau
```

V1 is post-refinement and is not necessarily AR rank1; absent exact reference scores cause KEEP. Best TRAIN local setting reached504, and local+Florence **505/849** (also505/745 eligible), with **3 recovered, 0 damaged**. This was a small positive signal, but failed the predeclared minimum5-recovery criterion and was not promoted.

### 3.6 Completed40-condition evidence-generation battery

All40 conditions used Qwen1.5B and the same1203 crops. Tested caption budgets0/16/32/64/full, caption source, neighborhood size, AR3/10, whole-image AR1/3/5/10/20, sequence/character confidence, rich prompts, few-shot examples, structured responses and negative controls. More beams meant alternative readings of the **same crop**, not new fragments or boxes.

| Representative condition | Exact/849 | Recovered | Damaged |
|---|---:|---:|---:|
| No caption, neighboring OCR present | 210 | 10 | 302 |
| Caption32 | 200 | 7 | 309 |
| Full caption | 226 | 14 | 290 |
| No neighboring crops | 316 | 10 | 196 |
| Target AR10 | 350 | 14 | 166 |
| Target AR10 with scores | 431 | 8 | 79 |
| Rich evidence, no caption | 434 | 5 | 73 |
| Rich evidence, full caption | 437 | 9 | 74 |
| Rich JSON | **489** | 3 | 16 |
| Rich closed choice | **499** | 0 | 3 |
| Target character evidence only | 479 | 8 | 31 |
| Whole-image AR20 with scores | 404 | 9 | 107 |

**Deep-dive findings:**

- All41,960 unique cached generations passed integrity checks;48,120 logical requests had correct crop/prompt identities.
- In the no-caption condition,219/302 damaged GT rows matched another OCR shown in the prompt: strong evidence of target confusion, not a mismatched evaluation join. Coincidence counts are not proof of causation.
- Rich JSON had16 damaged rows,12 matching another shown crop. Structured output helps, but is not a correctness guarantee.
- Closed choice had no recovery; its near-baseline score mainly reflects keeping V1.
- Union of recovered GT rows across all40 conditions was53. This is an oracle across conditions, not an available routing rule.
- CPU-only edit≤1/≤2 filtering of every old condition found **no result above502**. Best was rich JSON at502, with2 recovered/2 damaged.

The full40-row table and per-family interpretation are in [the battery deep dive](OCR_TARGET_BOUND_BATTERY_TRAIN_V2.md).

### 3.7 Prepared next:80 target-bound conditions — not yet evaluated

As of this summary, there is **no `results.json`** for the new battery. Preparation, baseline verification and76 focused tests passed; GPU inference has not been run by Codex.

- **8 evidence packages:** target only; character probabilities; two neighbors; Florence32; InstructBLIP32; targetAR10; localAR10; image-wideAR10.
- **10 mechanisms:** target-last plain output; exact-ID JSON; KEEP/REPLACE; indexed character patches; copy-original-then-repair; uncertain-character-only patches; edit≤1 gate; AR choice; context/no-context agreement; proposal followed by verification.
-96,240 logical evaluations;64,184 unique primary prompts, plus judges for valid changed proposals. Shared prompts are reused; the80 conditions are not statistically independent.
-Keep V1 on malformed, multiword, wrong-ID or truncated responses. No new crops, captions, PARSeq, segmentation, training or TEST.
-Predeclared useful TRAIN signal: **≥5 recovered, ≤2 damaged, exact>502, no token inflation**. No automatic pipeline promotion.

Manual PAPER command, from the repository:

```bash
bash diagnostics/run_ocr_target_bound_battery_host.sh /extra1/bmartelli
```

## 4. Overall findings

1. **Localization is a major bottleneck:** GT-box PARSeq836 versus frozen V1 crop502. Text-only correction cannot reliably restore visual evidence missing from a crop.
2. **Routing is not correctness verification:** a plausible or caption-related word can score well while being wrong; tau alone has not delivered the desired accuracy/fallback tradeoff.
3. **Unconstrained LLM generation usually causes more damage than recovery.** Extra context and longer candidate lists do not improve results monotonically.
4. **AR candidates and conservative visual evidence are the clearest limited positive signal:** +3 correct with no damage, not a large practical gain.
5. **The new battery tests target binding and verifiable edits, not simply longer prompts.** If it also fails, change selector/model or improve visual evidence rather than treating further prompt variations as demonstrated progress.

## 5. Main evidence files

Paths below are relative to `artifacts/diagnostics/` unless stated otherwise. These links point to frozen reports; this document does not replace their per-sample records.

| Experiment | Report |
|---|---|
| V1 failure audit | [train-v1-ocr-failure-audit-v1](../artifacts/diagnostics/train-v1-ocr-failure-audit-v1/results.json) |
| Recovered diagnostic crops | [train-recovered-crops-parseq-v1](../artifacts/diagnostics/train-recovered-crops-parseq-v1/results.json) |
| Initial structured correction | [contextual-ocr-correction-train-v1](../artifacts/diagnostics/contextual-ocr-correction-train-v1/results.json) |
| Plaintext / few-shot | [plaintext-v2](../artifacts/diagnostics/contextual-ocr-plaintext-train-v2/results.json), [fewshot-v1](../artifacts/diagnostics/contextual-ocr-fewshot-train-v1/results.json) |
| Fragment reconstruction | [initial](../artifacts/diagnostics/fragment-reconstruction-train-v1/results.json), [v4-context](../artifacts/diagnostics/fragment-reconstruction-train-v4-context/results.json) |
| Structural selection | [oracle](../artifacts/diagnostics/geometry-structural-oracle-train-v1/results.json), [Qwen selector](../artifacts/diagnostics/qwen-structural-selector-train-v1/results.json), [caption ablation](../artifacts/diagnostics/structural-caption-ablation-train-v1/results.json) |
| Conservative correction / gates | [per-crop](../artifacts/diagnostics/contextual-ocr-conservative-crop-train-v1/results.json), [CPU gates](../artifacts/diagnostics/contextual-ocr-conservative-gates-train-v1/results.json) |
| Prompt / geometry-context comparisons | [prompts](../artifacts/diagnostics/ocr-prompt-comparison-train-v1/results.json), [geometry/context](../artifacts/diagnostics/geometry-context-reconstruction-train-v1/results.json) |
| AR coverage / context likelihood / visual gate | [coverage](../artifacts/diagnostics/parseq-beam-candidate-coverage-train-v1/results.json), [likelihood](../artifacts/diagnostics/parseq-context-reranker-train-v1/results.json), [visual gate](../artifacts/diagnostics/parseq-visual-margin-context-gate-train-v1/results.json) |
| Completed40-condition battery | [results](../artifacts/diagnostics/ocr-evidence-generation-battery-train-v1/results.json) |
| New80-condition protocol and old-response audit | [protocol](../artifacts/diagnostics/ocr-target-bound-battery-train-v2/protocol.json), [deep dive](../artifacts/diagnostics/ocr-target-bound-battery-train-v2/deep-dive.json), [offline gates](../artifacts/diagnostics/ocr-target-bound-battery-train-v2/offline-gate-audit.json) |

Selected provenance anchors:

- Budgeted TEST report: `130280f0b656334b9fc9186be141f078903261c127a37bd1fd28c7e747376d2f`.
- Completed40-condition report: `baa40faedc36a5627b5f0e74fd9db93de7685fbd3891f2ee7b3d9d2167207416`.
- New80-condition protocol: `55f4651e5c01a84019c030190e71d095ab845d71dca6d6478d0766ba2b697095`.
