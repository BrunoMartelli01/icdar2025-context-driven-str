# Three context routers with full-TRAIN benefit supervision

This is **approximate reproduction**, extending the context/fallback motivation
in paper Section 3.5, Figure 4 and Algorithm 2. It is not the paper's MPNet/fuzzy
router, strict reproduction, or a Table 5 recognition-accuracy result.
Scientific decisions are recorded under `TOKEN-CONTEXT-ROUTER-001` in
[PAPER_AMBIGUITIES.md](PAPER_AMBIGUITIES.md).

## Implemented proposals

1. **Token context:** retain global and 2x2 tiled CLIP patch tokens, 32 BLIP2
   QFormer query tokens, every selected PARSeq word embedding, original word-crop
   embedding, geometry and confidence. A small attention network predicts paired
   light-minus-DeepSolo differences in true positives and output-word count.
   The frozen foundation models receive no optimizer updates. An auxiliary word
   head learns weak TRAIN annotation compatibility; it is not used as a claim of
   independent per-word correctness.
2. **Transcript likelihood:** score the actual fixed OCR transcript with frozen
   BLIP2/FLAN-T5 teacher forcing, retaining all target tokens and EOS. Compare
   global original/gray-erased text views and per-word expanded original/erased
   local contexts. Preserve sums, token means, EOS log probabilities and every
   token score. Whole-word chunks avoid silently truncating transcripts. This
   replaces Yes/No verification for this arm; probabilities are uncalibrated.
3. **Dense support:** train a small patch text-support head and paired visual/text
   adaptor only on the current fitting fold. Unbalanced optimal transport between
   visual support and word embeddings permits unmatched visual/word mass. This
   tests missing-text and unsupported-word evidence. Bbox-center support is a
   weak proxy, especially for curved text; CLIP intermediate patches are not
   inherently calibrated text detectors.

The suite includes matched 18-scalar baselines trained on the legacy 80 images
and the extended TRAIN, pooled context, each proposal separately, and their
combination. The scalar controls use the new matched feature schema; they are
not exact retrainings of the earlier 46-feature tabular router.

## Objective and evaluation

The primary goal is final **micro word F1** relative to all DeepSolo and the
same-budget GT oracle, without an image-F1>=80% target. The main bypass budget
is 50%; 20% and 30% are controls. The rank score is
`2*predicted_delta_TP - fitting_DeepSolo_F1*predicted_delta_word_count`.
Training uses paired count regression plus cost-weighted ranking. Mean/scale,
reference F1, auxiliary labels and all learned weights are fit-only.

Load the complete official Total-Text TRAIN 1,255 and TEST 300 manifests; exclude
TRAIN images pixel-identical to TEST and group any remaining duplicate RGBs in
cross-validation. Five grouped folds cover every usable TRAIN sample. Select
epochs and the overall arm with OOF final F1 at 50% bypass, then refit on all usable
TRAIN before extracting new TEST context. The predeclared primary arm remains
`token_context`; selecting another arm never uses TEST. The legacy 80 control
has its own OOF population and cannot win the overall full-TRAIN arm selection.

The expert checkpoints have seen Total-Text TRAIN and the repository TEST has
historically been explored. Full-TRAIN router fitting does not remove these
limitations. Training/refit scores are descriptive; OOF estimates and final TEST
measurements are separate. No universal or paper-superiority claim follows from
the engineering smoke test.

The light branch remains YOLO640 plus the qualified PARSeq selector; the fallback
remains qualified ViTAEv2-S/TextOCR DeepSolo. Reuse immutable original 125 TRAIN and
300 TEST branch outputs and require exact paired-count replay. Generate the
remaining TRAIN branch outputs under the same protocols. Do not edit words.

The metric is exact Unicode-casefold word multiset matching, retaining
punctuation and duplicates, without localization IoU. Care GT follows clipped
Total-Text annotations; native 256 validity handles ignored predictions exactly
as the frozen source. Empty predictions and images remain in the denominator;
TN=0 accuracy is `TP/(TP+FP+FN)`. Official end-to-end polygon F1 and the paper's
unresolved accuracy definition remain separate.

## Execution and provenance

Pass all local paths explicitly using
[token_context_router_local_paths_v1.json](../configs/approximate/token_context_router_local_paths_v1.json)
and scientific parameters using
[token_context_router_v1.json](../configs/approximate/token_context_router_v1.json).
The launcher hashes images, validity masks, annotations, manifests, pinned model
files, source caches, and a Python source snapshot. Every run executes from its
immutable snapshot so later workspace changes do not alter a running experiment.
No network downloads or dependency installation occur.

```bash
.venvs/context-str-main-py311/bin/python -u -m scripts.run_token_context_router \
  --mode approximate \
  --config configs/approximate/token_context_router_v1.json \
  --paths configs/approximate/token_context_router_local_paths_v1.json \
  --smoke
```

Remove `--smoke` for full TRAIN/TEST. Add `--preflight` to validate all inputs
without inference. Resume an existing sealed run with
`--run-root /absolute/path/to/run`; changed cache bytes fail explicitly.

PAPER qualification uses RTX 3090/24 GB, FP32, sequential frozen foundation models,
CLIP image batch 8, BLIP vision batch 1/language batch 2, router batch 8, CPU PARSeq
batch 32, and the unchanged isolated DeepSolo worker. TF32 is disabled; deterministic
math attention avoids nondeterministic attention backward. This is a documented
**HARDWARE_DEVIATION**, not a qualification for the historical 8 GB device.
Runtime receipts record native-forward equivalence, unchanged foundation-model
state hashes, peak allocated/reserved VRAM, tensor shapes, precision and batch
sizes. Inference and training errors propagate; OOM receives a failure record.

Save per-sample raw OCR, fallback responses, transcript token scores, dense and
query tensors, per-fold fitting IDs, learned checkpoints, OOF predictions, frozen
choices, and per-image routing decisions. Completion requires the independent
[audit](../scripts/audit_token_context_router.py), which checks feature provenance,
fitting scope, grouped leakage, OOF selection, native-equivalence receipts and
replays final counts with independent rational arithmetic. It does not claim an
independent neural refit.

## Qualification and launch status (2026-10-24)

- Focused tests: 48 passed. The extended relevant suite had 99 passes and one
  unchanged preexisting DeepSolo error-message assertion failure:
  `test_host_source_revision_verification_rejects_wrong_revision` expects
  `source revision mismatch`; the implementation raises
  `Declared DeepSolo source revision does not match Git HEAD`. The invalid
  source is still rejected. Neither this implementation nor that test was edited.
- An initial process-start smoke failed because CUDA memory counters were reset
  before child-process CUDA initialization. Explicit initialization fixed this;
  the failed run remains archived rather than overwritten.
- Completed engineering smoke: 17 TRAIN plus 5 TEST images, including 2 newly
  inferred TRAIN branch pairs, two-fold OOF training, seven router/control arms,
  strict learned checkpoint loads, final per-image predictions and independent
  audit **PASS**. All original cached branch counts replayed exactly.
- CLIP and BLIP2 staged/native forward maximum difference: 0. Foundation state
  hashes remained unchanged. CLIP peak allocated/reserved bytes:
  714321408/790626304; BLIP2: 15950744576/16064184320, FP32 on RTX 3090.
  The audit checked 133 original-minus-erased transcript contrasts, all fitting
  scopes and independent rational-count routing totals.
- Smoke source/result identity: external run
  `token-context-router-v1/smoke/df9fbf276f0b`, results SHA256
  `a8920aa00ca3148db087228ab206fadefb2d4d22ec421363a93c165b7624d6ca`.
  This tiny engineering subset establishes execution, not model quality.
- Full run launched with the same router/OCR runtime source hashes and scientific config,
  1,253 usable TRAIN and 300 TEST images. Exact RGB duplicate exclusions are
  TRAIN 81/TEST 656 and TRAIN 212/TEST 578. The launch receipt is
  `/extra1/bmartelli/context-str-artifacts/token-context-router-v1/current_full_launch.json`;
  run `token-context-router-v1/full/e8302d7d6054`, initial PID 1827387, log
  `/extra1/bmartelli/context-str-artifacts/token-context-router-v1/launches/full_20261024T183318Z.log`.

The full experiment completed successfully with independent audit PASS. Results
SHA256 is `c7868186ea6d1e7af8022839921aae6133986a2a8d711e65d112f7b2422ed548`.
At 50% bypass, TEST micro F1 is 77.87900% for the CV-selected scalar arm,
77.93776% for the predeclared token arm and 78.44408% for the best observed
combined arm, versus 80.35714% all DeepSolo and 82.32044% same-budget oracle.
These are approximate lexical metrics on historically explored TEST. Controlled
failure-cause tests are documented in [ROUTER_CAUSE_TESTS.md](ROUTER_CAUSE_TESTS.md).

An independent GPU smoke job was present during full-run startup; recorded stage
durations are not an isolated throughput benchmark. Resource observations are
saved with the launch receipts.
