# OCR evidence generation battery — frozen TRAIN exploration

`PRACTICAL_APPROXIMATION / approximate reproduction`, not paper reproduction,
not Table5 accuracy. Paper3.5 uses MPNet/fuzzy paired routing, not Qwen restoration.
No segmentation, localization, crop, PARSeq, caption, DeepSolo or production change.
CPU prepare/evaluate; GPU Qwen generation is run manually by the user on PAPER.

## Why this experiment

Longer context can supply missing clues but also irrelevant language, repetition
and competing instructions. Previous caption comparisons changed content/model
as well as length, so they did not isolate a length effect. This battery compares
explicit token budgets using the same caption. Prefix truncation still changes
content; do not interpret it as a pure intervention on length. Duplicating the
same caption is a stress control, not realistic extra evidence. All40 conditions
are declared in the approximate JSON BEFORE any generation, without TRAIN-result
prompt selection during the run. No new model training or installation.

## Forty conditions

| Family | Conditions | Fixed comparison |
|---|---|---|
| Caption length | CAP_NONE, CAP16, CAP32, CAP64, CAP_FULL | Same Florence caption, same target and two neighbor crops |
| Caption source | INSTRUCT32 | Same token budget as CAP32 |
| Other crops | NEIGHBOR0, NEIGHBOR4, NEIGHBOR_ALL | CAP32 uses two neighbors total |
| Target beam | AR3, AR10, AR3_SCORE, AR10_SCORE | Same target crop, readings with/without AR scores |
| Refined confidence | SEQUENCE_CONF, CHAR_CONF | Existing sequence/character probabilities, not model certainty labels |
| Rich combination | RICH, RICH_NO_CAP, RICH_FULL_CAP, RICH_INSTRUCT32 | All V1 neighbors, targetAR3+scores and character probabilities |
| Prompt/examples | RICH_SYNTHETIC_FEWSHOT | Four fixed invented examples, no dataset GT |
| Caption processing | RICH_QUOTED_TEXT, RICH_RELEVANT_SENTENCES | Extract quoted text or top-two OCR token-overlap sentences, GT-free |
| Position | RICH_CAPTION_FIRST | Same rich payload, caption before observations |
| Response schema | RICH_JSON, RICH_COPY_EDIT, RICH_CLOSED_CHOICE | Text value, explicit KEEP/REPLACE, or candidate index |
| Uncertainty hint | RICH_MASK_HINT | Auxiliary ? at character probability<.9; original text retained |
| Minimal evidence | TARGET_CHAR_ONLY | No caption, neighbors or beam |
| Stress controls | RICH_WRONG_CAPTION_CONTROL, RICH_DUPLICATED_CAPTION_CONTROL, RICH_REVERSED_AR_CONTROL | Next TRAIN image caption, repeated caption, reversed alternatives with true ranks/scores preserved |
| Neighbor selection | SPATIAL_NEIGHBORS2 | Center-distance vs frozen reading-order proximity |
| Beam per image crop | IMAGE_BEAM1/3/5/10/20 | All existing V1 crops, each with up toK AR readings |
| Beam per image crop scored | IMAGE_BEAM3/10/20_SCORE | Same conditions with summed AR logprob and delta to AR best |

User clarification: **more fragment evidence means more BEAM readings of the SAME
crop**, not more boxes or oracle fragments. The IMAGE_BEAM family implements this
for every available crop in the image. Neighbor count is a separate context test.
AR readings are hypotheses, never additional words to concatenate. One generation
per target crop; IMAGE_BEAM does NOT change the task to whole-image transcription.
Free-generation arms can output a spelling absent from the beam. CLOSED_CHOICE
is an explicit control that cannot invent a new string.

## Inputs, scoring and limitations

- All229 TRAIN images,1203 frozen V1 crops; all available probabilities aligned
  character-by-character to the original reading and source hashes verified.
- Only700 crops have AR cache from the prior GT-associated population. Do not
  generate beams for missing503 crops; keep their V1 observation. Every variant
  still processes all1203 crops. Report metrics stratified by AR availability.
  This inherited selection prevents interpreting that subgroup as an unbiased
  sample of all crop types. Keep empty hypotheses out without refilling ranks.
- Sequence confidence is the saved greedy product (EOS included where present);
  character probabilities exclude EOS. They are not calibrated correctness
  probabilities. AR sum likelihood and refined confidence are different evidence.
- Neighbor selection uses only frozen reading order or geometry. No word GT
  grouping, semantic GT selection or primary-reading designation by correctness.
- Full GT goes only into a separate evaluation file and post-generation log.
  Plain answers must be one token, or <KEEP>. JSON/edit/index schemas are checked
  strictly. Invalid, empty or truncated responses preserve exact original V1.
  No stripping punctuation or extracting favorable words from verbose responses.
- Report fixed-spatial exact on all849 words (baseline502) and eligible745,
  recovered/damaged, fuzzy>=.8, changed crops, lexical image-level multiset matches
  (baseline503), precision/recall/F1 and predicted tokens. The104 unavailable
  spatial predictions remain failures. GT derives from Task2.2 care glyph groups;
  some complete visible words may not match those fragments. Never silently
  reannotate or change the denominator to improve results.
- Identical full prompts share one generation across variants. They are not
  independent samples; logical per-variant times reuse shared latency. Actual
  unique GPU time is reported separately. Timing does not include caption/OCR
  creation, since both are cached. Negative controls are not deployment candidates.

## Execution, resources and audit

```bash
bash diagnostics/run_ocr_evidence_battery_host.sh /extra1/bmartelli
```

This runs CPU preparation and baseline verification, foreground manual GPU Qwen
generation with progress/output/associated GT, then CPU evaluation. No background
jobs, installation, downloads, TEST, new OCR/captions, or weight training. Same
pinned Qwen2.5-1.5B-Instruct BF16 batch1; longest input budget16384 tokens, output48.
No silent input truncation beyond the explicitly varied caption budgets. New long
GPU inputs remain unqualified until the user run; OOM is recorded and propagated.
Completed individual generations survive failures and are validated on resume.

Persistent root: artifacts/diagnostics/ocr-evidence-generation-battery-train-v1/.
protocol.json records resolved conditions/source/code hashes and prompt lengths;
inputs.json contains frozen OCR/probabilities, prompts.json stores unique inputs,
requests.json maps all48120 logical requests to those inputs. prompt-examples.json
shows all40 complete examples. No new images, crop copies or weight copies.
results.json contains all comparisons and per-word details. No automatic pipeline
promotion; this large, repeatedly TRAIN-informed search requires later independent
validation before claiming improvement.

