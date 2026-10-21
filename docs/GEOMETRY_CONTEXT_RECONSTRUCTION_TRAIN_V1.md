# Non-oracle geometry/context OCR reconstruction

Status: prepared diagnostic, **PRACTICAL_APPROXIMATION / approximate reproduction**.
Not paper Table5, not a change to localization, OCR or routing. Paper section3.5
compares recognizer output to a caption and routes to fallback; it does not specify
LLM reconstruction. No test split, training, new segmentation, OCR, or captioning.

## Frozen comparison

229 TRAIN images /849 care words /1203 retained V1 crops. Eight conditions:

| Representation | Caption conditions |
|---|---|
| BBOX_ONLY | none, InstructBLIP, Florence-2-base-ft, BLIP2 pretrain_vitL |
| EXPLICIT_GEOMETRY | same four |

Qwen2.5-1.5B-Instruct, qualified pinned checkpoint, BF16, greedy batch1,
max256 output tokens; one request per whole image/condition (1832 requests).
Zero-shot fixed instruction; no GT examples. Every condition uses identical crops
and frozen reading order. No primary reading is designated by GT, confidence,
size or correctness. All pairs receive observable geometry in the explicit arm:
containment, overlap, horizontal gap, vertical position, height compatibility,
and the existing fallible line membership. These are features, not new groups.

Full captions are reused from the verified five-captioner TRAIN run. No OCR-task
Florence output is appended. LLM responses, not captioner speed, are measured here.
Past captioner timing is separate from the new per-request Qwen latency.

The prompt and configuration are recorded verbatim in protocol.json and example
prompts in prompt-examples.json. Inputs are whitelisted separately from GT labels;
the latter appear only in post-generation terminal logging and evaluation.

## Parser and evaluation

ALL_WORDS accepts spaces AND newlines. No single-word filter. Preserve punctuation,
codes and genuine repetition. Empty, abstaining, EOS-truncated or detected
narration/structured responses fall back to the unchanged joined V1 transcription.
Narration detection is a heuristic, frozen before results; legitimate sign text
can resemble narration. Raw responses and raw lexical counts are saved separately.
No edits to older experiments or their parsers/results.

Per-image one-to-one casefold whitespace multiset matching; fuzzy uses maximum
cardinality matching at RapidFuzz ratio >=.8. Baseline lexical matches503/849,
1203 predicted tokens. This differs from the old spatial exact502: NEW outputs
have no boxes, so no spatial OCR accuracy is claimed. Recovered/lost are lexical
occurrences, not fixed GT-to-crop assignments. Precision, F1 and token counts
expose verbose outputs that merely include more GT vocabulary. All849 care words
remain in the denominator. No automatic promotion or threshold selection.

## Manual run and persistence

```bash
bash diagnostics/run_geometry_context_train_host.sh /extra1/bmartelli
```

Foreground, offline local models only, no dependency installation. CPU prepares
and verifies sources/tokenizer lengths/baseline. Only the explicit infer stage
uses CUDA on PAPER. CPU final evaluation follows. The wrapper reuses valid
per-request results after interruption, fails closed on drift, and never replaces
sealed artifacts. Output: artifacts/diagnostics/geometry-context-reconstruction-train-v1/.
No crops, weights or image copies are produced: compact text/geometry and JSON.
Model/runtime, hashes, input/output token lengths and peak GPU memory are recorded.

## Limitations and provenance

Retained V1 crops are not all raw glyph fragments: missing/filtered observations
remain missing. Caption disagreement may cause hallucination. TRAIN has informed
prior experiments, so this is not independent validation or a universal LLM test.
Scientific classifications are in the config and ambiguity CONTEXTUAL-OCR-GEOMETRY-006.

