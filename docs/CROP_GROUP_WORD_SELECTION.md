# Selecting one reading from a crop group

## Scope and command

**Approximate reproduction; TRAIN diagnostic. No Table 5 or strict reproduction claim.**

From the repository root:

```bash
bash scripts/test_crop_group_selectors.sh
```

This runs focused tests, then all **14 selectors** on the previously completed
125-image crop/group fixture, with the original baseline as a control. The local
path profile already points to the installed PARSeq runtime and CLIP snapshot.
No packages or weights are downloaded. CUDA is selected when available; otherwise
CPU is used. Models run sequentially in float32, with batch size 16.

### Complete official TEST evaluation

```bash
bash scripts/test_totaltext_test.sh
```

This dedicated launcher processes **all 300 official TEST images / 2204 care
words**, starting from original RGB. It computes whole-image and overlapping
window probabilities with the same qualified letterbox checkpoint, generates
only original baseline/threshold/window crops, runs PARSeq, builds the fixed
`complete_link_iou0.55` groups, then runs all 14 selectors including cross-PARSeq
and all four CLIP arms. It does not discover or reuse a TRAIN crop fixture.
The TRAIN manifest is used only to validate the selected model's training-run
contract; TRAIN annotations/recognition scores are not selection inputs.

Rules, threshold grids, window geometry and selector budgets/weights remain
those fixed on TRAIN. No alternative grouping cutoff is searched on TEST.
TEST annotations enter only after source crop/OCR artifacts exist; each selector's
outputs are frozen across all images before its own evaluation. Normalization,
matching and unique GT-instance counting are identical to the TRAIN experiment.
Full denominator changes cause failure rather than silently omitting samples.

The paths are in
`configs/approximate/totaltext_crop_group_test_local_paths_v1.json`.
Generator/group protocol files are
`totaltext_crop_group_test_suite_v1.json` and
`totaltext_crop_word_groups_test_v1.json`, in the same approximate namespace.
Selector rules use the unchanged generic selector configuration.
Outputs are under `/tmp/totaltext-crop-group-test-suite/<hash>/selectors/`:
`results.json` and `summary.md` include each selector's correct words, predictions,
baseline and oracle. Model/source/data/code hashes, backend/software/device and
CUDA allocated/reserved peaks are preserved. Interruptions reuse hash-checked
completed shards; changed contracts create a different run identity.

For an explicit smoke test only:

```bash
bash scripts/test_totaltext_test.sh --limit 1 --device cpu
```

The real-model one-image TEST smoke completed every stage (92 original crops,
207 cross pairs, four CLIP arms); 65 focused/regression tests passed after the
channel correction. Its oracle was 0/1, so it supplies execution evidence only.
The first attempt exposed a
summary-formatting bug when mean groups per recoverable word is undefined;
the summary now emits `n/a` without hiding the underlying zero recovery. No
full TEST recognition result or GPU qualification is claimed before the complete
command runs. Approximate diagnostic only; no strict reproduction/Table 5 claim.

The full TEST attempt later failed in CLIP at image `total_text:999` on a 3×3
group crop. Its original whole/window probabilities, crops, OCR, groups and
70,657 cross-PARSeq text/crop scores had already completed for all 300 images.
The fixed launcher qualifies that complete TEST source using the manifest,
split, image IDs, model inputs, device, original runner revision, source-stage
code hashes, shard counts and full denominator before reuse. It copies the
cross-PARSeq scores only after checking their contracts and the exact old/new
runtime hashes. **All CLIP scores are recomputed**, including earlier completed
images: ambiguous tiny crops could previously receive wrong embeddings without
an exception. If the qualified local source is absent, the command generates a
new TEST source. An existing source with mismatched provenance fails explicitly.

The formerly failing TEST image was scored with the pinned CLIP model on CPU:
all 95 groups and 129 candidate readings received finite scores. The complete
corrected GPU TEST evaluation still needs to be run. Its output is an approximate
diagnostic, not a strict reproduction or paper Table 5 result.

Optional commands:

```bash
bash scripts/test_crop_group_selectors.sh --limit 1
bash scripts/test_crop_group_selectors.sh --device cpu
bash scripts/test_crop_group_selectors.sh --cached-only
```

`--limit` labels the output SMOKE_TEST. `--cached-only` runs the nine selectors
using existing OCR caches and explicitly lists cross-PARSeq and CLIP as skipped.
The complete experiment writes `results.json`, `summary.md`, per-image evidence,
selection/evaluation shards, model scores, resolved configuration, and stage
execution/memory records under `/tmp/crop-group-word-selection/<plan-hash>`.
Identical reruns reuse hash-checked model-score shards; changed code, configuration,
device, model files or source contracts create a different experiment identity.

Paths belong to
`configs/approximate/crop_group_word_selection_local_paths_v1.json`.
Rules belong to `configs/approximate/crop_group_word_selection_v1.json`.
The CLI also accepts `--source`, `--grouping`, `--grouping-variant`,
`--image-manifest`, `--clip-model-dir` and `--output`.
The default adapter discovers the latest completed TRAIN-125 fixture, not the
fixture with the highest recognition score. It loads the upstream manifest,
image hashes, generator provenance, probability maps and original per-crop OCR
cache paths. It does not regenerate crops or add final image variations.

## General rules and their limits

`recognition/group_word_selection.py` receives crop geometry, group memberships,
generator provenance and OCR token probabilities. It receives no annotations,
dataset name, lexicon, word frequencies or externally fitted weights. The runtime
module scores supplied images/texts without dataset-specific recognition rules.
The launcher currently adapts the existing crop-suite artifact schema; using
another dataset requires compatible upstream artifacts and a matching manifest.

All composite scores use equal geometric means of dimensionless values in [0,1].
There is no search for confidence cutoffs, preferred lengths, prompt weights,
temperature or dataset-specific rules. The five-word/three-crop cross-scoring
budgets are fixed resource limits. No scoring parameters were changed after
viewing this experiment's evaluation.

The existing `complete_link_iou0.55` groups are retained. Their geometry parameters
were compared previously on this TRAIN fixture; they have **not** been independently
validated. Rules without dataset-dependent code do not establish generalization.
Group references and soft masks can still be wrong. Nearby windows are correlated
even though the consensus cap removes repeated votes within a single window.

## Nine selectors using existing evidence

Let `p_i` be an original PARSeq character probability. For a nonempty raw reading:

| Method | Score |
|---|---|
| `confidence_mean` | exp(mean(log p_i)); length-normalized, excludes EOS |
| `confidence_min` | minimum p_i |
| `confidence_balanced` | sqrt(confidence_mean * confidence_min) |
| `consensus` | supporting originating views / all views represented in the group |
| `consensus_weighted` | sum of maximum balanced confidence per supporting view / all group views |
| `stability` | longest consecutive threshold-support interval, averaged across views |
| `consensus_stability_confidence` | geometric mean of balanced confidence, weighted consensus and stability |
| `mask_completeness_confidence` | geometric mean of balanced confidence and soft-mask completeness |
| `combined` | geometric mean of balanced confidence, weighted consensus, stability and completeness |

Readings are pooled after normalization, but the original raw spelling and crop
are preserved. Each originating whole-image/window view contributes at most once
to consensus, regardless of how many thresholds reproduce its reading.

Stability uses the observed threshold grid of each group/view, rather than
hardcoded probability thresholds. Thresholds receive Voronoi interval widths
over that grid's observed span; only the longest consecutive supported interval
counts. This reduces dependence on nonuniform threshold sampling. A singleton
grid gives support 1 if the reading occurs there; a view with no threshold grid
provides neutral stability for its observed reading. Absent views contribute zero.

For candidate box C, fixed group reference R and probability mass M:

```text
coverage = M(C intersect R) / M(R)
purity = M(C intersect R) / M(C)
completeness = sqrt(coverage * purity)
```

This penalizes truncation and foreign foreground. Background padding alone does
not lower soft-mass purity. Whole/tile probability maps are reprojected through
the original letterbox geometry and averaged only where a view is available.
No new mask cutoff is used. If foreground mass is zero, the documented geometric
intersection/area counterpart is used. This is an ASSUMPTION, not a learned
completeness predictor or guaranteed word boundary.

## Cross-PARSeq selector

Shortlist up to five different normalized readings per group using `combined`.
Retain their observed raw spellings. Choose up to three representative original
crops, favoring distinct originating views and then geometric diversity; duplicate
pixel hashes do not provide repeated evidence.

For every shortlisted raw reading on every representative crop, compute its
**causal teacher-forced** likelihood with frozen qualified PARSeq weights:

```text
L(text, crop) = mean log P(character/EOS | crop, actual preceding candidate characters)
score(text, group) = exp(mean_crop L(text, crop))
```

EOS is included, PAD excluded; the raw character count plus EOS normalizes
length. This decodes the actual candidate prefix, rather than reusing logits
from another greedy history. The recognizer is not retrained or substituted.
Each unique crop is encoded once within a paired batch.

PARSeq can emit a maximum-length no-EOS greedy reading longer than its capacity
to score that same full string plus EOS. Such pairs receive an explicit
`UNSUPPORTED_LENGTH` record and zero cross score. They are never truncated and
remain available to all other selectors. The count is reported in `results.json`;
the full raw strings/reasons remain in cross-score shards. Other inference,
checkpoint, pixel-contract or OOM errors propagate and fail the experiment.

The post-hoc `cross_shortlist_oracle_correct` measures the available oracle words
after the word shortlist, before architectural unscorable-length rejection. It is
a diagnostic, never an input to shortlist construction.

## Four CLIP selectors

Pinned model: `openai/clip-vit-base-patch32`, commit
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`.
The local snapshot files are hashed; incomplete checkpoint loading causes failure.

Each normalized candidate reading is compared with two fixed images:

1. `full`: the entire original RGB scene, embedded once per image.
2. `group`: the fixed geometric group-reference crop, embedded once per group.

Every competing text in a group sees the **same image embedding**. Scores are
not comparisons between different crop-image/text pairs, which would confound
reading choice with unrelated image quality/context.

Both image views are placed on a median-RGB square canvas before the official
CLIP processor. This preserves the entire scene/crop through square preprocessing;
it does not change the original PARSeq crops. Two templates are tested separately:

The CLIP image processor receives `input_data_format="channels_last"` for each
PIL RGB image. In particular, a 3×3 crop must not be inferred as channels-first.
This fixes image decoding without changing the candidate text or score rule.

```text
raw:     {text}
written: a photo of the written word "{text}"
```

Methods: `clip_full_raw`, `clip_full_written`, `clip_group_raw`,
`clip_group_written`. L2-normalized embedding cosine is saved unchanged, then
mapped to `(cosine+1)/2` for the common assignment score. No prompt fitting,
lexicon, pretrained-weight change or annotation-based candidate filtering occurs.
Prompts exceeding the model's context fail explicitly rather than being truncated.

Full-scene similarity may prefer a semantically plausible word unrelated to a
particular group; group similarity may still confuse visually similar spellings.
These are experimental selectors, not established character verifiers.

## Shared crops and evaluation

A maximum-total-score assignment imposes at most one output per group and one
use per original crop ID. A shared multiword parent remains in all its groups,
but its single observed reading cannot count as multiple recovered words.
Separate crops with the same spelling can still represent different word instances.
Each group has a zero-score abstention option. There is no fitted rejection
threshold. Exact-score ties follow deterministic sorted group/candidate ordering.
Shortlist ranking additionally breaks ties by completeness and balanced confidence.

All selector artifacts for all images are frozen **before** the launcher opens
evaluation shards. Evaluation then uses exact normalized text plus original-RGB
bbox IoU >=0.5 and maximum one-to-one prediction/GT matching. Normalization uses
Unicode casefold and removes punctuation/symbol categories P/S; whitespace,
digits and accented letters remain. No correct witness or GT box enters selection.

The fixture denominator is 983 care words (982 normalize to nonempty text).
Empty-normalized GT remains in the denominator and cannot become an exact hit.
Results report correct words, predictions, unmatched predictions, exact precision,
recovered/lost GT instances relative to baseline, and the frozen group oracle.
Group counts include background fragments and duplicate hypotheses; they are not
the number of true words. High word recall alone is not end-to-end accuracy.

## Scientific evidence and verification

The local reproduction paper, `papers/2503.15639v2.pdf`, sections 3.3/3.5 and
Algorithm 2, specifies PARSeq localized recognition and a different context
pipeline. These selectors and CLIP are ASSUMPTION experiments. No strict
configuration, caption/MPNet scoring or DeepSolo behavior is changed.

[PARSeq](https://arxiv.org/abs/2207.06966) describes the autoregressive/non-AR
recognizer used here. The teacher-forcing implementation was inspected against
the [pinned official source](https://github.com/baudm/parseq/tree/1902db043c029a7e03a3818c616c06600af574be).
[STR confidence calibration research](https://arxiv.org/abs/2012.12643) reports
overconfidence in recognition models, motivating caution when interpreting raw
confidence as correctness. We do not fit its calibration procedures.
[CLIP](https://arxiv.org/abs/2103.00020) supplies the image/text representation;
it does not establish the efficacy of the crop-group selectors proposed here.

Focused tests cover token alignment/normalization, correlated-vote caps, threshold
intervals, soft-mask mass/coordinate mapping, shared-crop assignment, repeated
word instances, GT-independent inputs, actual candidate-prefix/EOS teacher forcing,
unscorable length handling, fixed CLIP competitor images, configuration isolation
and synthetic OOM propagation. CPU execution with real pinned models is verified.
The runner records device, precision, input shape, batch size, model artifacts,
software, seed/backend flags and CUDA peak allocated/reserved bytes. CUDA peaks
are null on CPU; actual RTX 3070 execution/OOM qualification remains UNRESOLVED.
Sequential loading, batch size and CPU execution are HARDWARE_DEVIATION choices.

Initial full-run failure: an overlength no-EOS PARSeq reading hit the old strict
length guard. The failure artifact was retained, and the architectural exception
is now represented explicitly without text truncation or hiding other errors.
An earlier cache inspection assumed a crop-cache directory instead of following
its recorded source path; the adapter now follows and verifies that path. A
software probe queried an absent headless OpenCV package name; metadata now uses
the installed `cv2.__version__`. Neither required installation or substituted data.

See `CROP-GROUP-WORD-SELECTION-V1` in `docs/PAPER_AMBIGUITIES.md` for classifications
and unresolved transfer/noise/calibration questions. The complete run's
`summary.md` and `results.json` are the authoritative measured selector counts.

## Measured comparison on the frozen TRAIN-125 fixture

Completed with actual pinned models on CPU, float32, four CPU threads, batch16:
`/tmp/crop-group-word-selection/0d7113261cc3/results.json`.
All 14 selectors executed; no methods were skipped. 59 focused/regression tests
passed. Input preparation took 22.0s, cross-PARSeq 273.7s, CLIP 282.7s, selection
3.7s and evaluation 5.0s. These are local CPU timings, not GPU performance claims.
There were 27,773 unique cross-text/crop pairs; 8 had UNSUPPORTED_LENGTH status.
The five-reading shortlist retained a post-hoc oracle of 778 versus the complete
group oracle of 781. The original unrestricted crop oracle was 782.

| Method | Correct / 983 | Predictions | Recovered vs baseline | Lost vs baseline |
|---|---:|---:|---:|---:|
| Baseline original | 400 | 1173 | 0 | 0 |
| Confidence mean | 707 | 7522 | 324 | 17 |
| Confidence minimum | 705 | 7522 | 326 | 21 |
| Confidence balanced | 707 | 7522 | 326 | 19 |
| Consensus | 668 | 7522 | 282 | 14 |
| Weighted consensus | 706 | 7522 | 316 | 10 |
| Threshold stability | 645 | 7445 | 259 | 14 |
| Consensus + stability + confidence | 666 | 7447 | 276 | 10 |
| Soft-mask completeness + confidence | **732** | 7522 | 339 | 7 |
| Combined | 680 | 7447 | 289 | 9 |
| Cross-PARSeq | 691 | 7519 | 301 | 10 |
| CLIP full, raw | 596 | 7522 | 234 | 38 |
| CLIP full, written | 640 | 7522 | 262 | 22 |
| CLIP group, raw | 622 | 7522 | 259 | 37 |
| CLIP group, written | 670 | 7522 | 289 | 19 |

**Historical CLIP counts:** those four rows preceded the RGB channel-order fix.
Twelve TRAIN group reference canvases have a 3-pixel side, so the old CLIP
counts require a corrected rerun before exact comparison. Non-CLIP scores and
oracles are unaffected. The prior machine-readable result remains preserved.

The highest measured correct-word count is 732 (74.47% of care words, 93.73% of
the group oracle). Historical CLIP group/written reported 670; cross-PARSeq
reaches 691. The non-CLIP combined arm does not improve the strongest simpler
selector. No selector parameters were changed after evaluation.

There are many noisy/duplicate group hypotheses. Even the 732-correct selector
emits 7522 predictions, with 6790 unmatched predictions and **9.73% exact
precision**. This experiment demonstrates recovered-word availability/selection,
not a finished detector or reliable output rejection. Independent data are still
needed to evaluate transfer and any subsequent selection of a preferred method.
