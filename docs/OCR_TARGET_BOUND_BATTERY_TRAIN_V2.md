# OCR battery: damage audit and 80 target-bound diagnostics

## Scope and evidence

`APPROXIMATE` / `PRACTICAL_APPROXIMATION`, exploratory TRAIN-only. No pipeline
promotion, training, TEST, new OCR, caption generation, crops or segmentation.
Paper §3.5 (local `papers/2503.15639v2.pdf`) selects OCR by MPNet/fuzzy scores and
routes to DeepSolo; it does not specify an LLM repair module. This experiment is
an explicit practical extension, not a reproduction of Table 5.

Frozen source: `artifacts/diagnostics/ocr-evidence-generation-battery-train-v1/`.
Results SHA256 `baa40faedc36a5627b5f0e74fd9db93de7685fbd3891f2ee7b3d9d2167207416`.
All **41,960** unique generation shards passed checksum/contract/parser checks.
40 conditions ×1203 crops =48,120 logical evaluations. Original inference took
4551.85 seconds; sharing prompts is accounted for only once. No old inference
was repeated for this audit.

Frozen spatial baseline: **502/849**, or502/745 among geometrically eligible GT.
Lexical multiset baseline: **503/849**. There are1203 predicted crop tokens.
Multiple GT rows can refer to one crop; recovered/damaged count GT rows, whereas
changed counts crops. These quantities are not interchangeable. Case-only changes
also count as changed crops but do not change casefold exactness.

## What failed, and why this is not a caption-length problem alone

- None of40 conditions exceeds V1. The union of all recovered GT rows is53,
  an **oracle across variants**, not an available selection rule.
- `CAP_NONE`:219 of302 damaged GT rows exactly match another shown OCR crop.
  With `CAP32`,126 of309 match another shown crop and160 match a caption token.
  These are nonexclusive coincidence proxies, not proof of copying/causation.
- Removing neighbors (`NEIGHBOR0`) improves200→316 relative toCAP32, still poor.
  Adding all neighbors gives179. More context is not consistently better.
- AR10 with scores improves350→431 versus unscoredAR10. But AR3 with scores
  worsens349→337; imageAR20 with scores is404 versus imageAR10 with scores411.
  More beams or numeric evidence does not monotonically improve correction.
- `RICH_JSON` reaches489 with3 recovered/16 damaged.12 of16 damaged rows match
  another shown crop.1114/1203 responses are syntactically valid; its improvement
  is not explained solely by parser fallback. Validity is not correctness.
- `RICH_CLOSED_CHOICE` is499 with0 recovered/3 damaged,1203 valid responses and
  only17 changed crops. It almost preserves V1, but does not solve recovery.
- `TARGET_CHAR_ONLY`:479,8 recovered/31 damaged, including6 punctuation-only
  damage rows. Character evidence is worth constraining, not blindly trusting.
- Caption-first, generic synthetic fewshot, reversed beams and wrong captions
  are not promising deployment directions. Wrong-caption damage is consistent
  with semantic distraction, but these controls do not isolate every cause.
- Punctuation does not explain the massive losses: e.g.CAP_NONE has only3
  punctuation-only damaged rows; RICH_JSON has1. GT care/ignore conventions may
  still affect individual examples; no labels were silently rewritten.

## Every old condition: raw results and offline edit-distance audit

The extra audit accepts a saved response only when casefold Levenshtein distance
from V1 is≤1 (also≤2 inJSON), otherwise keeps V1. No model inference. These
post-hoc TRAIN checks test whether an obvious safety gate already solves the
problem: **it does not**. Neither limit produces a condition above502.

| Old condition | Exact/849 | Recovered/damaged | Edit≤1 exact (rec/dam) |
|---|---:|---:|---:|
| CAP_NONE |210|10/302|501 (3/4)|
| CAP16 |188|6/320|500 (2/4)|
| CAP32 |200|7/309|499 (2/5)|
| CAP64 |222|15/295|500 (2/4)|
| CAP_FULL |226|14/290|500 (2/4)|
| INSTRUCT32 |241|14/275|499 (2/5)|
| NEIGHBOR0 |316|10/196|498 (3/7)|
| NEIGHBOR4 |194|8/316|498 (1/5)|
| NEIGHBOR_ALL |179|7/330|497 (1/6)|
| AR3 |349|17/170|499 (8/11)|
| AR10 |350|14/166|496 (6/12)|
| AR3_SCORE |337|11/176|499 (5/8)|
| AR10_SCORE |431|8/79|499 (6/9)|
| SEQUENCE_CONF |187|8/323|499 (2/5)|
| CHAR_CONF |214|8/296|498 (2/6)|
| RICH |379|11/134|499 (5/8)|
| RICH_NO_CAP |434|5/73|496 (2/8)|
| RICH_FULL_CAP |437|9/74|501 (4/5)|
| RICH_INSTRUCT32 |339|9/172|499 (2/5)|
| RICH_SYNTHETIC_FEWSHOT |216|5/291|497 (1/6)|
| RICH_QUOTED_TEXT |428|5/79|496 (2/8)|
| RICH_RELEVANT_SENTENCES |423|9/88|497 (3/8)|
| RICH_CAPTION_FIRST |204|10/308|498 (2/6)|
| RICH_JSON |489|3/16|502 (2/2)|
| RICH_MASK_HINT |335|7/174|501 (4/5)|
| RICH_COPY_EDIT |335|12/179|491 (4/15)|
| RICH_CLOSED_CHOICE |499|0/3|499 (0/3)|
| TARGET_CHAR_ONLY |479|8/31|489 (4/17)|
| RICH_WRONG_CAPTION_CONTROL |322|0/180|496 (0/6)|
| RICH_DUPLICATED_CAPTION_CONTROL |424|7/85|500 (5/7)|
| RICH_REVERSED_AR_CONTROL |368|8/142|481 (5/26)|
| SPATIAL_NEIGHBORS2 |200|7/309|499 (2/5)|
| IMAGE_BEAM1 |306|6/202|500 (1/3)|
| IMAGE_BEAM3 |345|13/170|498 (8/12)|
| IMAGE_BEAM5 |340|15/177|497 (6/11)|
| IMAGE_BEAM10 |362|12/152|496 (6/12)|
| IMAGE_BEAM20 |388|11/125|497 (6/11)|
| IMAGE_BEAM3_SCORE |359|8/151|500 (4/6)|
| IMAGE_BEAM10_SCORE |411|10/101|494 (4/12)|
| IMAGE_BEAM20_SCORE |404|9/107|491 (2/13)|

### Which nearby tests are justified?

| Old family | Decision and new mechanism |
|---|---|
| CAP/INSTRUCT/NEIGHBOR/SPATIAL | Do not extend length sweeps. Test target-last, exact target ID, copied-original check and context/no-context agreement. |
| AR/IMAGE_BEAM/confidence | Retain genuine AR10 scores, change output constraints and verification. Do not request new beams or OCR. |
| JSON | Most useful formatting lead: exact ID plus COPY_REPAIR, edit gate, agreement and independent judge. |
| TARGET_CHAR/CHAR_CONF/MASK_HINT | Test indexed character patches, then substitutions restricted to uncertain original positions. |
| COPY_EDIT | Old generic KEEP/REPLACE still copied context; new identity-bound operation and original-copy requirement are distinct tests. |
| CLOSED_CHOICE | Recheck target-bound AR10 choice as a conservative control. Its old near-baseline score is not evidence of useful correction. |
| RICH_NO_CAP/FULL_CAP/QUOTED/RELEVANT | Use smaller fixed evidence packages with constrained output; do not optimize caption views repeatedly on these results. |
| Synthetic fewshot/caption-first | Stop these prompt extensions; test verifiable target binding/patches instead. |
| Wrong/duplicated caption/reversed beam controls | Preserve as stress evidence, not candidate deployment policies. |

`deep-dive.json` includes a recommendation, five damaged examples and coincidence
counts for **each** old condition. A nearby test is a falsifiable hypothesis, not
a claim that it will succeed.

## New predeclared 80 conditions

Configuration: `configs/approximate/ocr_target_bound_battery_train_v2.json`.
8 evidence packages ×10 mechanisms, no search/tuning after results.

Evidence packages:

1. TARGET_ONLY: original OCR only.
2. TARGET_CHAR: original and cached character probabilities.
3. LOCAL_TWO: character evidence plus two nearest reading-order crops, total two.
4. FLORENCE32: character evidence plus first32 tokenizer tokens of cached caption.
5. INSTRUCT32: same budget with cached InstructBLIP caption.
6. TARGET_BEAM10: character evidence and target AR10 with original likelihoods.
7. LOCAL_BEAM10: same plus two neighboring crop readings.
8. IMAGE_BEAM10: all crop AR10 hypotheses/scores and up to32 tokens of quoted
   Florence text. These are multiple readings of the same crops, not new crops.

Mechanisms:

1. PLAIN_TARGET_LAST: short instruction, context first, one explicit target last.
2. BOUND_JSON: exact `target_id` plus one output word/code.
3. KEEP_REPLACE: identity-bound operation rather than unconstrained response.
4. CHAR_PATCH: at most two indexed edits on original Unicode character positions,
   nonoverlapping, simultaneous, total edit distance≤2.
5. COPY_REPAIR: first reproduce original OCR exactly in `observed`, then repair it;
   incorrect original copy causes rejection.
6. UNCERTAIN_PATCH: at most two one-character substitutions only at original
   PARSeq character probability<.9; no insertions/deletions. This is a heuristic,
   not a correctness guarantee or calibrated uncertainty threshold.
7. EDIT1_GATE: reuse BOUND_JSON output; accept only casefold edit distance≤1.
8. AR_CHOICE: select target reading from KEEP+cachedAR10, exactID and index only.
9. CONTEXT_AGREEMENT: BOUND_JSON with context and without context independently;
   change only when both valid outputs agree case-insensitively.
10. VERIFY_PROPOSAL: generate BOUND_JSON, then a separate fresh prompt checks a
    valid changed proposal against the same evidence; ACCEPT or KEEP only.

These are80 evaluated conditions, **not80 independent generation runs**. Prompt
identity sharing avoids repeated work. Agreement with a package already lacking
context is deliberately an identity control. AR_CHOICE explicitly adds target
AR10 where absent; UNCERTAIN_PATCH exposes indexed uncertainty positions even
forTARGET_ONLY. Therefore this is a structured battery, not a fully orthogonal
factorial causal study.

GT is used only for scoring/printing, never in prompts or gates. Caption content
is fallible. The104 unavailable spatial GT rows stay in the849 denominator.
Only700/1203 crops possess historical AR cache (selected by prior GT association);
report AR-available/unavailable strata and do not extrapolate this population bias.

## Reproducibility, cost and decision

Pinned Qwen2.5-1.5B-Instruct revision and verified weights are inherited from the
source protocol. BF16, batch1, greedy generation, seed0. Response cap128 instead
of48 enables indexed patch schemas; this is an explicit protocol change, so new
vs old scores are not a pure prompt-only ablation. Log truncation separately.
No model installation, updates, new captions or PARSeq inference.

CPU prepare records80 conditions, compact reconstructible prompt descriptors,
examples, code/input hashes and token counts. It references frozen source data
instead of duplicating crops/images/weights. GPU generation writes every result
atomically with checksum. Resume verifies existing shards and runtime metadata;
changed protocols fail closed. `progress-current.json` and Python faulthandler
help identify a native crash; there is no automatic restart loop.

The second GPU stage runs only on valid changed BOUND_JSON proposals. Identical
requests are reused. The final report separates pre-gate and accepted exact,
recovered/damaged, crop changes, lexical counts, invalid-schema/rejection reasons,
beam-availability strata, actual unique inference seconds and peak VRAM. Invalid,
multiword, wrong-ID or truncated outputs always preserve the exact original.

Predeclared **GO_TRAIN_DIAGNOSTIC_ONLY**: recovered≥5, damaged≤2, exact>502 and no
token inflation above1203. Even a GO is not automatic pipeline promotion or proof
of generalization;80 exploratory comparisons are TRAIN-informed and correlated.
If none qualifies, stop this class of Qwen1.5B prompt/evidence repair experiments
and investigate a different selector/model or visual evidence. No method is
declared effective before running the new battery.

## Launch and outputs

From the repository on PAPER, foreground with visible progress:

```bash
bash diagnostics/run_ocr_target_bound_battery_host.sh /extra1/bmartelli
```

Persistent directory:
`artifacts/diagnostics/ocr-target-bound-battery-train-v2/`

- `deep-dive.json`: all40 previous-condition audit and recommendations.
- `offline-gate-audit.json`: edit≤1/≤2 audit of cached old responses.
- `protocol.json`, `jobs.json`, `requests.json`, `prompt-examples.json`: frozen80.
- `baseline.json`: CPU fixed baseline check.
- `verifiers.json`: deterministic second-stage descriptors and parent hashes.
- `generations/`: atomic resumable inference shards, original raw output included.
- `results.json`: all80 evaluations, per-word/crop records, hashes and decisions.

No GPU inference was run by Codex while preparing this battery.

## CPU qualification completed before handoff

-76 focused unit tests passed: new battery, frozen previous battery, shared OCR
  correction/scoring tests. `py_compile`, shell syntax and `git diff --check` pass.
-All48,120 old logical request→prompt→crop identities agree with frozen OCR/sample
  IDs. The observed target confusion is not an incorrectly joined evaluation row.
-All96,240 new requests cover exactly1203 crops in each of80 conditions, and their
  primary/secondary job identities point to the correct crop in the same image.
-64,184 unique primary prompts; maximum5223 input tokens. Secondary verification
  count is determined by valid changed proposals, not GT. This is more work than
  the old41,960-job run; no latency prediction or GPU completion claim is made.
-CPU baseline reproduces502 spatial exact and503 lexical matches. Preparation
  occupies approximately36MiB and duplicates no images, crops or model weights.
-Protocol SHA256:
  `55f4651e5c01a84019c030190e71d095ab845d71dca6d6478d0766ba2b697095`.
-Deep-dive SHA256:
  `6760f354dd5f0fc301fe9c6b16121e89379fbcc1eaffde69769a9d8db46aeea3`.
-Offline-gate audit SHA256:
  `eb4d290a38408389e133a4c34da9566f96e66dc1749a7f3b4328c7904626b925`.

Source code/config/results of the previous battery remain unchanged. New GPU
conditions and memory behavior still require the manual PAPER execution above.
