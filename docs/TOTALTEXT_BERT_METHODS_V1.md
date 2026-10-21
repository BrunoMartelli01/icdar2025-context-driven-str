# Total-Text BERT methods with local OCR and Florence scene context

All results are **approximate reproduction**, post-hoc development on the same
125-image TRAIN holdout already used for earlier experiments. Independent TEST
transfer has not been measured. The mask letterbox -> contours -> PARSeq source,
949 crops,983 care words and503 geometric matches are unchanged. Baseline exact
casefold recognition is384/983. No Table5 or strict reproduction is claimed.

The measured best arm is selective NeuSpell missing-letter completion:
396/983 (14 recovered,2 damaged,+12). With local OCR and Florence it obtains
392/983 (10/2,+8). The separate generic BERT reranker benefits from scene context:
382/983 with local OCR alone,388/983 with local OCR plus Florence (5/1,+4 versus
baseline). Context is useful for that reranker; the best observed completion arm
remains the isolated-word control. These are multiple development comparisons,
not a validated default for independent images.

## Sources and classifications

The local paper `papers/2503.15639v2.pdf`, sections3.3–3.5, provides the frozen
pipeline/context motivation. Its strict path specifies MPNet, lexical matching,
BLIP-2 and DeepSolo; this user-requested BERT extension stays in APPROXIMATE.
Scientific decision record: `TT-BERT-METHODS-V1` in `docs/PAPER_AMBIGUITIES.md`.

Primary inspiration includes
[PIE: Parallel Iterative Edit Models](https://aclanthology.org/D19-1435/),
[Levenshtein OCR](https://arxiv.org/abs/2209.03594) and
[Misspelling Correction with Pre-trained Contextual Language Model](https://arxiv.org/abs/2101.03204).
They motivate bounded edits and contextual candidate selection. Our fixed
insertion gate is an engineering hypothesis (`ASSUMPTION`); it does not reproduce
their trained edit architectures or claimed results. NeuSpell's genuine frozen
word head and the published pooling/vocabulary decoding remain as documented in
`docs/TOTALTEXT_NEUSPELL_CORRECTION.md` (`REPOSITORY_SPECIFIED`).

## Predeclared method battery

Config: `configs/approximate/totaltext_bert_methods_v1.json`. Its thresholds were
fixed before these additional model outputs/metrics, after observing the negative
v2 TRAIN results. No additional threshold sweep, training or TEST access occurred.
The old sealed implementations/configs/results were not modified.

Two NeuSpell tokenization controls use identical learned weights:

- `published_lowercase`: default loading of the pinned HF export, lowercase=True.
- `training_cased`: explicit lowercase=False, following the original toolkit's
  named cased encoder. The name denotes a hypothesis, not proof of intended
  training preprocessing. Its casing/linked default accent stripping differs
  from the exported loading behavior. The ambiguity remains `UNRESOLVED`.

Three selection methods (`ASSUMPTION`):

1. `conservative`: the unchanged v2 bounded edit/probability/KEEP rule.
2. `oov_insertions`: source length>=4; original must be absent from the pretrained
   NeuSpell word-head vocabulary, not from a Total-Text-derived lexicon. Predicted
   head probability>=0.8; preserve every source character in order and insert
   at most2 characters. No substitutions/deletions, recursive corrections,
   numeric/punctuation/internal mixed-case rewriting or invented OOV KEEP score.
   Preserve original output casing. The probability is not calibrated accuracy.
3. `bert_pll_rerank`: original exact string is candidate0/immutable KEEP. Add up
   to5 unique, shape/edit-bounded candidates from the NeuSpell word_only top10
   labels, with head probability>=0.01. Consider only OOV sources of length>=4.
   Source lengths<6 allow1 edit, otherwise2; relative edit distance<=0.3.
   The candidate pool is frozen across all context conditions within a casing
   control. Score with the existing frozen `google-bert/bert-base-uncased`
   target-only mean PLL-word-l2r: target subwords at/to the right of the scored
   piece are masked, left pieces remain visible. Accept only
   `mean_PLL(candidate) - mean_PLL(KEEP) - 0.5*edits > 1.5`, KEEP on ties.
   Unknown target/token overflow yields an explicit abstention; runtime/loading/
   nonfinite errors are fatal. This mean/target-only adaptation retains known
   tokenization and length bias, rather than qualifying published sentence sums.

Each has word_only, local_ocr and local_ocr_caption controls. Local OCR is the
unchanged spatial horizontal-neighbour rule, using original PARSeq strings,
not corrected strings. NeuSpell receives the caption as a prefix; the MLM
scorer uses a separate caption segment and local word segment. The same bounded
80-word caption budget is recorded. No GT enters head inference or reranking.

## Actual Florence context

Generated captions on all125 matching original RGB images, not ICDAR13 or other
cached scenes. Model `microsoft/Florence-2-large-ft`, revision
`4a12a2b54b7016a48a22037fbd62da90cd566f2a`; weightSHA256
`8b4e610c952eef90a836c56cda0f398a672a3a6ca7b4d96b0e09a86dee42e2c3`.
Task `<MORE_DETAILED_CAPTION>`, greedy beam1/no sampling/max64 new tokens.
The model's pinned generation_config also specifies early_stopping=True, which
is ignored for beam1 and emits a library warning; that is not an inference
failure. All model/config/tokenizer/Python source file hashes are in the resolved
config. Loading audit has no missing/unexpected/mismatched keys.

This Florence substitution and caption coupling are `ASSUMPTION`, isolated from
strict-paper BLIP-2. CPU FP32, batch1, four threads, shorter generation, separate
model processes and cached stages are `HARDWARE_DEVIATION`. Existing isolated
einops0.8.1 is reused with source/version verification; no package was installed,
upgraded or environment created. HF dynamic-module copies use an explicit
writable module cache. Florence has770173952 parameters; input pixel shape
[1,3,768,768]; actual generated/input shapes and timings are saved per image.
Caption generation totals845.29s, excluding model loading. GPU peak fields are
null: no GPU execution or VRAM qualification was performed.

RGB lookup uses the current TRAIN prepared manifestSHA256
`05c41d87cf83e6d3d25cfc465eea9345ebd19b7cb0034030b87438486ff9be9b`.
That differs from the historical source TRAIN manifest; it serves only as image
lookup (`ASSUMPTION`). Every selected ID and JPEG hash must equal the frozen OCR
input, and no mask/annotation is opened by caption generation. Paths escaping
the declared data root fail. Partial caption smoke outputs cannot be attached
as complete125-image context. Complete captions are bound to model/revision,
decoding settings, file hashes, sample IDs and RGB hashes.

Qualitative checks of images66 and321 confirm the relevant names Hong Kong
Police and Selamat Datang/Gunung Ledang in the captions. Image66 also exposes
an incorrect flower count/color description: these two spot checks do not
qualify caption accuracy, and captions are treated as fallible observations.

## Full results

All18 arms use983 care GT in the denominator, including unlocalized words.
Exact Unicode casefold retains punctuation and uses deterministic maximum
one-to-one bboxIoU>=0.5 matching. Also save case-sensitive/fuzzy metrics,
per-image predictions, recovered/damaged GT sets and unchanged geometry.
No failed samples or denominator exclusions occurred. This inherited metric is
an `ASSUMPTION`, not official polygon spotting.

| Casing | Context | Method | Exact /983 | Recovered | Damaged | Net |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| — | — | PARSeq baseline | 384 | — | — | 0 |
| lowercase | word | conservative | 379 | 18 | 23 | -5 |
| lowercase | word | insertions | **396** | 14 | 2 | **+12** |
| lowercase | word | PLL | 383 | 0 | 1 | -1 |
| lowercase | local | conservative | 380 | 19 | 23 | -4 |
| lowercase | local | insertions | 392 | 10 | 2 | +8 |
| lowercase | local | PLL | 382 | 0 | 2 | -2 |
| lowercase | local+Florence | conservative | 377 | 17 | 24 | -7 |
| lowercase | local+Florence | insertions | 392 | 10 | 2 | +8 |
| lowercase | local+Florence | PLL | **388** | **5** | **1** | **+4** |
| cased hypothesis | word | conservative | 381 | 3 | 6 | -3 |
| cased hypothesis | word | insertions | 386 | 2 | 0 | +2 |
| cased hypothesis | word | PLL | 383 | 0 | 1 | -1 |
| cased hypothesis | local | conservative | 383 | 3 | 4 | -1 |
| cased hypothesis | local | insertions | 386 | 2 | 0 | +2 |
| cased hypothesis | local | PLL | 385 | 1 | 0 | +1 |
| cased hypothesis | local+Florence | conservative | 384 | 6 | 6 | 0 |
| cased hypothesis | local+Florence | insertions | 386 | 2 | 0 | +2 |
| cased hypothesis | local+Florence | PLL | 386 | 2 | 0 | +2 |

Selective word-only examples include `SLAND -> ISLAND`, `WORL -> WORLD`,
`TUDIO -> STUDIO`, `BRIDG -> BRIDGE`, `GIRAFF -> GIRAFFE`. Its two damages are
`RESIDENTAL -> RESIDENTIAL` and `INTEGRIT -> INTEGRITY`. Visual inspection of
image802 confirms the literal sign misspelling RESIDENTAL: grammatical repair
is incorrect OCR. Image1274 shows a pole across the end of the word region;
intended complete spelling/occlusion handling remains `UNRESOLVED`, and the
supplied literal GT INTEGRIT is preserved. No annotation was relabeled and no
sample-specific exception introduced.

The lowercase scene-PLL arm recovers `raditional -> traditional`,
`MILI -> MILL`, `ALARI -> ALARM`, `GIRAFF -> GIRAFFE`, `ULLPEN -> BULLPEN`;
its damage is `UAWEI -> HUAWEI` relative to unchanged GT. These are post-hoc
auditable examples, never candidate/threshold supervision. Captions improve
that fixed-pool reranker by6 exact matches relative to local OCR alone, but
worsen the broad conservative head. They do not automatically improve every
method. Independent transfer and protection of literal misspellings/occluded
characters remain unresolved; no strict/TEST/generalization claim follows.

Standalone cached correction runtime (`HARDWARE_DEVIATION`): CPU FP32, four
threads, seeded deterministic settings, head batch<=8/MLM mask-task batch<=16,
max256 input tokens. Lowercase head totals4.60s word/5.31s local/15.51s scene;
PLL totals2.10s/2.33s/8.84s. Source candidate heads and captions have separate
costs; timings are not paper end-to-end measurements. Models are unloaded by
separate process completion, with no co-resident GPU assumption.

## Tests and audit

103 focused CPU tests pass, including20 new selective-edit/KEEP/reranking,
real tiny-head+MLM integration/resume, absent GT files during both inference
stages, caption/model/coverage/corruption and RGB-boundary cases. Tiny weights
are engineering fixtures only. Actual pretrained caption smoke and full125
caption generation pass; all full-data head/PLL arms pass without runtime,
loading, nonfinite or OOM failures. No failures are ignored or replaced.

An independent audit checks750 head shards,750 PLL shards and125 caption
shards, reconstructs all18 metrics/deltas/crop identities, verifies frozen
candidate pools and confirms fast/slow tokenization parity on3330 scored head
strings. The12 no-caption controls reproduce exactly between the text-only
and complete scene runs, including per-image predictions and GT matches.
Existing sealed v2/MLM implementations, checkpoints, source data and configs
remain byte-for-byte unchanged. No git commit or large output/weight addition
was performed.

Artifacts:

- Full result: `/tmp/totaltext-bert-methods-full-v1/results.json`, SHA256
  `1c8539835200573ef50ccc2bbb5147f3b249ec54e6f8e73b5a9d319bf5733620`.
- Full protocolSHA256:
  `f504ff281158c965371320d50d5e5bca4b2572926e85dc51825cebd2e0897847`.
- Independent audit: `/tmp/totaltext-bert-methods-full-v1-audit.json`.
- Captions: `/tmp/totaltext-bert-florence-context-v1/captions.json`, SHA256
  `06b50d6c3a6b48a160cd208472397307620f4081ae1f952e287776cdb571ffa2`.
- Text-only controls: `/tmp/totaltext-bert-methods-text-controls-v1/results.json`,
  SHA256 `a3c52d9cd3d40d497661461a7f3eb7666d7f3311f3f2f2c5d56bb17a4c075ce5`.

## Re-run with existing assets

All paths are explicit; the wrapper installs nothing. It generates missing
captions or verifies existing caption metadata/coverage during preparation,
then runs the frozen heads, generic MLM reranker and evaluation.

```bash
bash scripts/run_totaltext_bert_methods_host.sh \
  --source-dir /extra1/bmartelli/context-str-runs/totaltext-letterbox-word-f1-separator-parseq-val-v1 \
  --reuse-ocr /extra1/bmartelli/context-str-runs/totaltext-v4-image-frequency-ablation-v1/validation-pad0-parseq-v1/ocr \
  --hf-cache /extra1/bmartelli/context-str-artifacts/huggingface \
  --asset-dir /extra1/bmartelli/context-str-artifacts/neuspell-word-head-v1 \
  --rgb-manifest /extra1/bmartelli/datasets/context-str/scene-segmentation-v1/manifests/total_text_train.yaml \
  --data-root /extra1/bmartelli/datasets/context-str/scene-segmentation-v1 \
  --runtime-deps artifacts/diagnostics/lavis-captioner-candidates-20260914/runtime-deps \
  --module-cache /tmp/totaltext-bert-methods-hf-modules \
  --caption-dir /tmp/totaltext-bert-florence-context-v1 \
  --output-dir /extra1/bmartelli/context-str-runs/totaltext-bert-methods-v1 \
  --device cpu
```

Stages are also exposed by `scripts/evaluate_totaltext_bert_methods.py`:
prepare -> head -> pll -> evaluate. `--conditions word_only local_ocr` explicitly
selects text-only controls; the full default requires complete captions. Use a
new output directory when configuration/code/device changes. GPU execution
requires separate qualification; there is no automatic device substitution.
