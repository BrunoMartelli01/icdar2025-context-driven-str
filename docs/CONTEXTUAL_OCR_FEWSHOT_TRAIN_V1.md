# Image-held-out TRAIN few-shot OCR correction

User requested demonstrations built from actual TRAIN OCR and GT. This is a
PRACTICAL_APPROXIMATION with **supervised in-context examples**, not annotation-free
paper reproduction. No gradient updates, no fine-tuning, no TEST. Paper sections
3.4/3.5 use caption similarity/routing, not this correction mechanism.

## Frozen design

- All229 TRAIN images /849 GT words, same1203 retained V1 crops and cached full
  InstructBLIP captions. No new image inference or localization changes.
- Qwen2.5-1.5B only, same qualified revision/weights/BF16 and greedy64-token output
  as the zero-shot plaintext experiment. No model download or package installation.
- Five folds by image: sort unique source-image SHA256 values by a seeded digest
  and assign round-robin; identical image contents stay in the same fold. This
  protects exact duplicates but does not establish absence of near-duplicate scenes.
- For each held-out fold, choose four fixed demonstrations from the other folds,
  alternating KEEP/CORRECT/KEEP/CORRECT, from four distinct source images. Selection
  is by seeded candidate-ID hash, not manually chosen after inference outcomes.
- Demonstrations require nonempty original OCR, frozen best IoU>=.5, exactly one
  significantly intersected GT word, and an unambiguous historical crop association.
  Correct examples use that word's GT; keep examples preserve original OCR when
  casefold-exact. Full other OCR readings and optional caption are retained.
- Every target of a fold receives the same demonstrations. Target GT is never
  used to select examples, select requests, or form its prompt. Other-fold GT is
  deliberately included in demonstrations, with complete provenance in protocol.
- Same instruction as zero-shot; plain target text only, with/without caption.
  Input limit2048 (previous1024) accommodates demonstrations; token-limit and
  malformed-output abstentions are explicit and preserve only the affected target.
- All849 GT words remain in CPU evaluation with the original geometric assignment.
  Report original PARSeq, cached zero-shot, few-shot, recoveries/damage, per-fold
  counts, latency, VRAM, and abstentions. No automatic promotion.

Prior experiments have already informed design using TRAIN. Therefore these are
image-held-out demonstration diagnostics, **not** an untouched generalization
estimate. Future TEST must use a separately frozen final prompt/example bank,
not GT-dependent fold selection. No final TEST bank is selected here.

## Manual PAPER run

```bash
bash diagnostics/run_contextual_ocr_fewshot_train_host.sh /extra1/bmartelli
```

Foreground/unbuffered, single GPU model; per-target atomic caches and checksum
validation permit restart. Preparation and evaluation are CPU. GPU inference is
not run by Codex. Persistent directory:
`artifacts/diagnostics/contextual-ocr-fewshot-train-v1/`.

The zero-shot reference SHA256 is
`c34c9f758b2dc2499bd4b463c2d6f237ca2eb4d57f369b7c5c1ba6e6d45afe70`.
Existing runner source/artifacts remain unchanged. The standalone runner reuses
the previous inference/evaluation logic in a separate module to preserve its
code-hash-qualified historical protocol.

## CPU verification

36 targeted/new and regression unit tests passed. Preparation and baseline-only
evaluation reproduce502 exact /578 fuzzy out of849. Tokenizer-only preflight on
all2406 prompts found a maximum763 input tokens, none over the2048 limit (also
below the old1024 limit). No model inference or CUDA was executed by Codex.
