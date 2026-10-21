# Full TRAIN plain-text OCR correction diagnostic

User-authorized PRACTICAL_APPROXIMATION / approximate reproduction, not Table5.
The unfinished TRAIN20 pilot was expanded to all 229 TRAIN images before execution.
No experiment or tests were launched by Codex after the user requested manual execution.

Run on PAPER from the repository root:

```bash
bash diagnostics/run_contextual_ocr_plaintext_train_host.sh /extra1/bmartelli
```

Uses the immutable previous correction report
`artifacts/diagnostics/contextual-ocr-correction-train-v1/results.json`, SHA256
`4c220819335738c66a2c1c9cf3e6f62643eb62b2bee69508f16f1c8ddd969b48`.
Reuses its V1 crop transcriptions, full InstructBLIP captions, qualified model
revisions, weight hashes and dtypes. No download, segmentation, PARSeq, caption
generation, training, TEST, or DeepSolo execution.

Compares FLAN-T5-base, Qwen2.5-0.5B and Qwen2.5-1.5B, each with other OCR readings
alone or other readings plus the full caption. One target crop per request,
plain-text output rather than model-generated JSON/IDs/actions. Target association
is performed by code. Every crop remains available as context regardless of GT.

Malformed, empty or token-limited output explicitly preserves only the affected
target. One-line narration and incorrect corrections are evaluated as generated,
not repaired. Empty OCR targets are preserved. GT is accessed only during CPU
evaluation with the prior frozen geometric association, all 849 words retained.
Report exact/fuzzy, recovered and damaged words, lexical surplus, abstentions,
total per-image correction latency and GPU peaks. No automatic promotion.

Generation runs sequentially on cuda:0, batch1, with foreground unbuffered progress.
Each target is atomically saved and sealed; rerunning verifies and reuses completed
targets. Results are separate under
`artifacts/diagnostics/contextual-ocr-plaintext-train-v2/`.
Previous results are neither overwritten nor removed.

CPU protocol tests were added at
`tests/recognition/test_contextual_ocr_plaintext_train.py` but not executed,
respecting the instruction not to run this task. No new experimental results yet.
