# Frozen prompt comparison, TRAIN229

PRACTICAL_APPROXIMATION. Prompt-study extension explicitly requested after negative
conservative per-crop results. Those results reject tested configurations, not all
LLMs or reconstruction methods. Paper3.5 uses semantic/lexical routing, not this
LLM post-correction. No strict pipeline change.

Five prompt families, two conditions each (OCR-only / OCR+full cached caption):

1. CONTROL: prior conservative prompt, reuse its actual cached generations.
2. P1_SHORT: concise restoration instruction.
3. P2_RECONSTRUCT: explicitly reconstruct an imperfect observed transcription.
4. P3_CHARACTER_AUDIT: inspect character-level errors, preserve orthography/codes.
5. P4_EVIDENCE_PRIORITY: prioritize crop OCR against conflicting scene descriptions.

The four new instructions are fully specified in
`configs/approximate/ocr_prompt_comparison_train_v1.json` before inference.
Qwen2.5-1.5B, exact revision/weights/BF16, generation32 tokens greedy, seed,
input limit2048, demo examples/folds, OCR, captions and field ordering are unchanged.
Only instruction text changes. Inputs remain ONE independent crop, not multiple
fragments of a word or full-image reconstruction.

All variants, including cached CONTROL, receive the same parser: reject any
internal Unicode whitespace, empty/truncated/multiline output to exact original
PARSeq. Outer whitespace stripped as before; no punctuation cleanup, edit-distance
gate or semantic rescue. Raw outputs retained and abstentions reported. Cached
unfiltered source metrics remain in report separately for auditability.

All229 TRAIN images /849 GT words /1203 crops. Fivefold GT demonstration separation
is inherited unchanged. GT labels printed after each generation are stored in a
separate sealed log file and never included in the target prompt. Logs distinguish
frozen scoring assignment from best-overlap geometric GT. Every word stays in the
evaluation denominator. Exact uses frozen max-IoU audit association; lexical scores
use image-level multisets. No multi-fragment oracle grouping.

Cached baseline/control read on CPU; 4 x2 x1203 =9624 new GPU requests, one model
load. Foreground progress, atomic per-target outputs and checksum-verified resume.
No segmentation, OCR, caption inference, downloads or package installation.

```bash
bash diagnostics/run_ocr_prompt_comparison_train_host.sh /extra1/bmartelli
```

Outputs: `artifacts/diagnostics/ocr-prompt-comparison-train-v1/`.
`prompt-examples.json` holds complete representative prompts for each new condition.
`baseline.json` is CPU-only; `results.json` follows manual PAPER GPU inference.

GO is only a TRAIN diagnostic flag: recovered>damaged, exact>502, lexical
matches>=503, predicted tokens<=1203. No automatic promotion. This is a finite
predeclared study, not exhaustive proof; repeated TRAIN-based prompt selection
can overfit and needs a separately frozen evaluation later.

CPU preparation/verification: 38 targeted and regression tests passed; py_compile,
shell syntax and diff whitespace checks passed. Cached controls reproduced:
V1 exact502/lexical503; single-word CONTROL OCR-only exact490/lexical491 and
caption exact486/lexical493. Tokenizer-only preflight of all9624 new prompts:
maximum646 input tokens, zero over2048. No model inference/CUDA executed by Codex.
