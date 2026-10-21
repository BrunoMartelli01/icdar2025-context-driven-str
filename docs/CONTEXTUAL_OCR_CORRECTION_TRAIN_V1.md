# Contextual OCR correction: frozen TRAIN comparison v1

## Scope and boundaries

PRACTICAL_APPROXIMATION / APPROXIMATE mode. New diagnostic; it does not replace
Algorithm1, practical-remap-v1/v2, MPNet/scoring, routing, or DeepSolo. Not Table5.
No new segmentation, PARSeq, captions, training, or TEST.

V1 stays fixed as an audited control (502/849 exact,578/849 fuzzy), not a claim
that it is best for every metric. Changing localizer and corrector together
would prevent isolating the language-model effect. The GT-dependent78/80
recovery oracle never enters generation or prompts.

## Models and inputs frozen before outcomes

| Model | Revision | Precision |
|---|---|---|
|google/flan-t5-base|7bcac572ce56db69c1ea7c8af255c5d7c9672fc2|FP32|
|Qwen/Qwen2.5-0.5B-Instruct|7ae557604adf67be50417f59c2c2f167def9a775|BF16|
|Qwen/Qwen2.5-1.5B-Instruct|989aa7980e4cf806f80c7fef2b1adb7bc71aa306|BF16|

Official sources: [FLAN-T5](https://huggingface.co/google/flan-t5-base),
[Qwen0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct),
[Qwen1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/commit/989aa7980e4cf806f80c7fef2b1adb7bc71aa306).
Revisions checked against official repositories/configs, not mutable main at runtime.

Three conditions per model, one call per image:

1. `ocr_only`: all crop texts/bboxes in geometric order.
2. `ocr_caption`: identical inputs plus complete cached InstructBLIP caption.
3. `caption_only`: extract text claimed visible from caption alone; no OCR or
   spatial correspondence.

Maximum2061 generations (3 models x3 conditions x229 images). Models load one
at a time in the existing main environment. No installation/upgrade. Download
only necessary snapshots to persistent HF cache: safetensors/tokenizer/config,
not PyTorch/TF/Flax copies. Approximately5GB new weights if none are cached;
no duplicate image/crop storage.

Config: `configs/approximate/contextual_ocr_correction_train_v1.json`.
Greedy, beam1, max768 new tokens, max2048 input tokens, batch 1, seed 0. No outcome-based
tuning. Over-limit inputs are not silently truncated: explicit abstention.
FP32/BF16 differences are HARDWARE_DEVIATION.

## Prompts, outputs, and safeguards

- Complete prompts are in config and saved with each generation.
- Block text/captions are data, never instructions to execute.
- Never provide GT, sample IDs, paths, oracle results, or prior correctness.
- Do not invent confidences: these frozen inputs expose boxes/text, not
  calibrated confidences; no artificial estimate is added.
- JSON output must contain exactly original IDs and keep/correct/uncertain actions.
- Keep/uncertain preserve text exactly. No new/merged IDs, filling originally
  empty blocks, or deletion through correct.
- Malformed output is retained with error/explicit abstention: preserve all
  original OCR for that image; caption_only emits empty output. It stays in
  denominators and is not retried with different prompts.
- Runtime/OOM/checkpoint mismatch aborts with error, without model/precision
  substitution. Schema abstention is declared protocol, not ignored runtime error.
- Prompts do NOT guarantee hallucination-free output; valid JSON need not be
  correct text. Harm/surplus are explicitly measured.

## Provenance, cache, and resume

Inputs: `context-str-runs/caption-routing-five-models-train-v1`, report SHA:
`389219133531817df915413a14ef01f727f0e6d710dc1d1eb177ad843e43b46b`.
Protocol seal, input hashes,229 captions, caption model, associated images verified.
Preparation emits text/bboxes only, without GT or image reads.

Evaluation-only audit:
`artifacts/diagnostics/train-v1-ocr-failure-audit-v1/results.json`, SHA
`82acfe8d860aee1a0f7bf70b9308a83aefb4abc17561105c74324d8b990fc56a`.

Protocol/generations are atomic and sealed. Restart verifies prompt, code, model/
runtime identity and reuses every valid result, including previously malformed
outputs. Completed models are not reloaded. Weight/config/tokenizer hashes are
saved on first load. Different protocols require a new output directory/version;
never overwrite earlier reports.

## Metrics and interpretation

- Exact casefold and fuzzy>=.8 over 849 GT with fixed historical crop association:
  baseline502/578 reproduced before GPU work. New text/GT never selects a
  different crop. Words without crops remain in the denominator.
- Per-word errors corrected and correct words harmed, before/after.
- Separate lexical counts: per-image whitespace-token casefold multiset,
  duplicates capped at GT multiplicity. Recall, precision, surplus, missing.
- Caption-only has lexical metrics only: do NOT treat them as crop exactness.
  Surplus is diagnostic, not hallucination proof, also because of Don't Care
  and tokenization.
- Invalid-output/token-limit counts; correction time/VRAM per image;
  loading/parameter counts in model metadata.
- Measured latency excludes segmentation/PARSeq/caption/fallback. No
  end-to-end savings conclusion before measuring the whole system.
- No automatic promotion: discuss net benefit, harm, surplus, cost. Repeatedly
  explored TRAIN needs independent validation after freezing to establish
  generalization, not optimization on individual known errors.

## PAPER command

```bash
cd /extra1/bmartelli/projects/icdar2025-context-driven-str
bash diagnostics/run_contextual_ocr_correction_host.sh /extra1/bmartelli
```

Foreground with visible progress; uses existing `.venvs/context-str-main-py311`.
Only the user runs GPU. Codex may run preparation, evaluation, CPU unit tests.

Persistent outputs: `artifacts/diagnostics/contextual-ocr-correction-train-v1/`:
`protocol.json`, `inputs.json`, `baseline.json`, `models/`, `generations/`,
`results.json`, `results.md`; errors under `errors/`. Final report requires all
requested generations to exist and pass verification.

## Scientific classification

`ASSUMPTION`: generative correction, prompts/schema/abstention, diagnostic metrics,
V1 control choice. `HARDWARE_DEVIATION`: sequential loading, batch 1, FLAN FP32/
Qwen BF16. `UNRESOLVED`: comparable Table5 results and actual fallback savings.
Paper sections3.3/3.4 describe localization/context, not this correction;
no original-component changes or new fidelity claim.

## Checks completed before GPU run

- 36 CPU tests passed: prompt isolation, schema/IDs, abstention, cache corruption,
  recovery/harm/duplicate metrics, caption-routing/joined-T3 regressions.
- Real Transformers4.47.1, Qwen2 AutoConfig, GenerationConfig imports passed;
  no model loaded or CUDA call performed.
- py_compile, bash -n, git diff--check passed.
- Preparation/baseline evaluation executed twice: idempotent resume verified,
  229 images/1203 crops/229 captions,502 exact/578 fuzzy.
- Protocol SHA:`15d4eef2a0227969fd64944936b98f5e64cb898317f6b6432974cdffaf11c80a`.
- Inputs SHA:`a1b7320caceabd56f3816fcfbb4a84bd2e2bf4a885c65b97d35a7eb947fc33a4`.
- Baseline SHA:`2b700b21e3fe25e7568426869faa6f29ce26e0cad77acc1517a2d91f1a1b3c6c`.

The three corrector results remain PENDING_GPU, unmeasured rather than zero.
