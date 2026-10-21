# Paper image router baseline — approximate reproduction

This user-authorized baseline compares the section 3.5 MPNet/fuzzy router with
the existing CLIP routers on the **same frozen U-Net Text+Kernel epoch61 crops
and PARSeq outputs**. It does not change segmentation, crop recognition, STRICT
configuration, the old CLIP experiments, or DeepSolo inference. The inspected
authority is `papers/2503.15639v2.pdf`, sections 3.3–3.5, Figure 4 and Algorithm 2.
Scientific decisions are recorded under `PAPER-IMAGE-ROUTER-001` in
[PAPER_AMBIGUITIES.md](PAPER_AMBIGUITIES.md).

## Router and evaluation boundary

- **T1:** real PARSeq applied to the original RGB image using its native
  preprocessing and sequence decoding. All 425 T1 outputs have been generated
  on CPU FP32 and sealed. This remains the repository's qualified approximate
  PARSeq checkpoint; the paper's exact PARSeq-S checkpoint is unresolved.
- **T3:** every frozen crop transcription, including repeated words,
  low-confidence readings, and outputs subsequently ignored by evaluation.
  Sort by bbox top, left, and instance ID; join with spaces, exactly as the CLIP
  full-transcript control. Empty readings remain in the per-word records and
  make the image ineligible for acceptance. There is no confidence filter,
  deduplication, word correction, new crop, or GT-driven word selection.
- **T2:** a caption generated from the RGB image alone. Neither PARSeq outputs
  nor annotations enter the caption prompt.

The literal paper interpretation computes MPNet cosine S1/S3 against the whole
caption, selects T1 if S1 > S3 and T3 otherwise, then computes the selected
candidate's confidence `C = 0.6 S + 0.4 L`, where `L` is the repository's raw,
case-sensitive `rapidfuzz.fuzz.ratio / 100`. `C >= 0.8` accepts the selected text.
The semantic winner, both lexical scores, both paired confidences, selected text
and literal fallback decision are preserved per image.

Multi-crop aggregation is not defined by the paper; joining **all** words is the
user's requested interpretation. The paper's prose specifies the selected
candidate's confidence, while its diagram and underspecified Algorithm 2 allow
other interpretations. This baseline explicitly follows the prose and the
algorithm's strict semantic comparison, including its T3 tie behavior.

The user's clarification requires accepting a winning T1 when **its own**
confidence passes the same threshold. T1 then emits one raw PARSeq sequence;
T3 emits all original crop readings. Two gates are reported:

| Gate | Whole-image acceptance rule |
|---|---|
| `paper_selected` (primary) | If T1 wins, accept its one prediction only when C1 passes; if T3 wins, accept all crop outputs only when C3 passes. Otherwise request full-image DeepSolo. |
| `joined_t3` | Keep all original crop outputs when C3 passes the threshold, independently of the T1 winner. This isolates the user's joined-T3 routing proposal. |

The previous T1-winner veto is superseded by the user's clarification. Winning
the semantic comparison alone does not bypass the threshold. A lower-confidence
winner requests DeepSolo even if the other candidate has a higher paired C.
No whitespace splitting turns T1 into multiple predictions, and no bbox is
invented for it. An empty selected output/caption or any input exceeding MPNet's
384-token context forces fallback. Empty crop readings make T3 ineligible but
do not veto a nonempty, confident T1 winner.

**Semantic coherence cannot establish that no word was missed.** The report
therefore audits accepted images against annotations: `safe_images` means at
least one care prediction and FP=0 under the declared word metric;
`complete_images` additionally requires FN=0. With T1 accepted, the other GT
words remain FN. These counts do not establish geometric localization and
never enter the router or act as model probabilities.

## Caption investigation

Figure 4 shows a coherent 72-word paragraph describing the setting, text-bearing
objects, colors, spatial relationships and surroundings. The displayed prompt
is truncated. Section 3.4's 40/80 generation lengths and Figure 4's 80-word limit
do not uniquely specify tokenizer lengths or decoding. The profiles restate the
visible instructions and request 40–80 **words** as an explicit approximation;
raw outputs are preserved even when outside that range.

Real CPU FP32/batch-one probes used TRAIN `total_text:14`, inspected visually
without annotations: an outdoor restaurant facade, a red/white sign, illustrated
food, fencing, and a taller building behind it. There are no visible parked cars
or parking lot. These are **single-image smoke tests**, not a caption benchmark
or evidence of improved recognition accuracy.

| Profile | Model | Words | Observation |
|---|---|---:|---|
| `figure4_single` (reference) | BLIP-2 Flan-T5-XL | 7 | Native answer: “a building with a sign on it”; too generic for the paper's illustrated style. |
| `figure4_structured` | BLIP-2 Flan-T5-XL | 34 | Four raw answers joined with punctuation; repetition and an unhelpful “typefaces” answer. |
| `figure4_single_long` | BLIP-2 Flan-T5-XL | 47 | Forcing at least 60 new tokens adds a parking lot and cars absent from the image; rejected as the default. |
| `figure4_grounded` | BLIP-2 Flan-T5-XL | 26 | Six simple questions still misclassify the outdoor scene as “indoors.” Its profile name is not a grounding guarantee. |
| `instructblip_scene` | InstructBLIP Flan-T5-XL | 51 | Three coherent sentences with environment, placement and fence details; closer in style, but wrongly reads the sign as “Toyotu” and speculates about a busy street. Separate model ablation. |

The InstructBLIP output is:

> The image features a white building with a sign that reads "Toyotu" in a Japanese language. The building is surrounded by a fence, suggesting that it is located in an urban area. The sign is centered on the building, which may indicate that the restaurant is located near a busy street.

No authored facts, PARSeq text, repetitive filler, or GT are appended. Structured
profiles preserve every native answer and add punctuation only. InstructBLIP
uses its own native model class, processor, revision, weights, and family
metadata. It does not replace BLIP-2 in the reference baseline.

In the final integrated smoke test, T1 was `1` and T3 was
`TOKU TOKU 38h PERANIA`:

| Caption | S1 | S3 | C3 | Semantic winner | Literal paper decision at 0.8 |
|---|---:|---:|---:|---|---|
| BLIP-2 reference | 0.133943 | 0.125569 | 0.140648 | T1 | DeepSolo (selected C1=0.080366) |
| InstructBLIP ablation | 0.050869 | 0.254959 | 0.168611 | T3 | DeepSolo |

The more detailed caption increases S3 on this image, but its raw lexical score
is only 0.039088. It does not pass the paper threshold or prove better selection.
For `C=0.6S+0.4L`, even S=1 requires L>=0.5 to reach 0.8; the whole-caption
lexical/threshold ambiguity remains unresolved. The paper threshold is retained,
with exploratory threshold curves reported separately.

## Frozen comparison protocol

TRAIN is the existing held-out subset of 125 images / 983 care words, not the
entire 1,255-image Total-Text training split. TEST is all 300 images / 2,204 care
words. The TRAIN80 calibration partition uses the previous seed and a
GT-independent cut nearest 50% image acceptance. TEST labels never choose a
threshold or caption profile. Exact descriptive budgets request the same
62 TRAIN / 149 TEST images as the frozen CLIP reference; if fewer images are
eligible, actual acceptance is lower and is reported.

Each caption/gate gets the paper threshold, the TRAIN-calibrated cut, and fixed
threshold/breakpoint curves. For each split, report separate accepted T1/T3
image counts, the chosen source per accepted image, retained PARSeq words,
correct words, precision, accepted/safe/complete images, missing fallback IDs,
final PARSeq+DeepSolo words and final metrics. The full run also adds the frozen
original B16 router and each B16/B32 prompt ablation's TRAIN-selected family to
the same CSV/Markdown comparison. Frozen CLIP decisions are replayed without
refitting; model result, source plan, branch and split identities are checked.
The CLIP family remains the one previously selected on TRAIN using its original
geometric metric, not a new family chosen to maximize this word metric.

T1 supplies no localization, so the **main comparison** uses one-to-one,
casefold-exact word matching within each image **without bbox IoU**, consistently
for T1, T3, DeepSolo and CLIP. Repeated predictions match at most the same number
of repeated GT words; punctuation remains significant. Frozen T3/DeepSolo ignore
flags and invalid-envelope false-positive handling remain; unlocalized T1 is a
single care prediction. This evaluation choice is an explicit `ASSUMPTION`, not
the established paper/Table5 protocol. Counts are summed over all split images:

- Accuracy on detected care words / precision: `TP / (TP + FP)`.
- Requested accuracy with TN=0: `TP / (TP + FP + FN)`.
- An empty accepted prediction set has undefined precision, not perfect precision.
- Missing DeepSolo output makes the **full** combined metric unavailable; no
  missing image is dropped or replaced with PARSeq.

Original care annotations, per-image predictions and the independent TEST
DeepSolo audit are SHA-pinned and read only after routing scores are saved.
Their geometric counts are independently replayed against the frozen branch
metrics before using their words. `word_branch_metrics.json` preserves per-image
T1/T3/DeepSolo metrics, care GT and source hashes. The old bbox-IoU>=0.5
results and joined-T3 curves remain under `geometry_diagnostics`, separately
from the main word comparison. They are not mixed with unlocalized T1 counts.

DeepSolo outputs already cover TEST300, but only 63/125 TRAIN images. The
remaining 62 TRAIN images may be needed by the new router. Consequently,
combined TEST metrics can be complete after caption generation, whereas combined
TRAIN metrics depend on which missing images are rejected. The runner lists
their IDs. It does not run DeepSolo or change its qualified runtime.

For orientation, the **already completed** frozen original CLIP B16 router has
TEST149/300 PARSeq acceptances, 2,069 final care predictions, 70.5172% precision
and 51.8479% TN0 accuracy under the **original geometric criterion**. This is an
existing CLIP result, not the new word metric or a new paper baseline result.
The new baseline's full accuracy is pending full caption generation.

## Execution and artifacts

The runner is `scripts/run_paper_image_router_baseline.py`. Its protocol and
machine paths are separately configurable in
`configs/approximate/paper_image_router_baseline_v1.json` and
`configs/approximate/paper_image_router_baseline_local_paths_v1.json`. It uses
already installed packages and locally cached pinned models, with no downloads
or package upgrades. Configurations never fall back to STRICT/demo namespaces.

Complete comparison on the GPU host, including the BLIP-2 reference and separate
InstructBLIP ablation:

```bash
.venvs/context-str-main-py311/bin/python -m scripts.run_paper_image_router_baseline \
  --mode approximate --stage all --device cuda:0 \
  --caption-placement cpu_offload
```

Use `--profiles figure4_single` for the BLIP-2 reference alone. Other configured
caption profiles can be requested explicitly. A small real CPU run is:

```bash
.venvs/context-str-main-py311/bin/python -m scripts.run_paper_image_router_baseline \
  --mode approximate --stage all --split train --limit 1 --device cpu \
  --output-root /tmp/paper-image-router-baseline-v1-final-smoke
```

`--check` verifies source/config/model assets and writes the resolved plan.
`--stage t1` verifies the sealed 425-image T1 manifest and original shards.
`--stage captions` generates missing captions; `--stage score` reuses them.
Scoring must use the same device, placement, profiles, config and output root as
caption generation because these are cache identities; MPNet itself runs on CPU.
Cache contracts also include code/software, checkpoint, prompt/config and image
hashes. Existing raw outputs cannot be edited or relabeled as another run.

The default output root is `/tmp/paper-image-router-baseline-v1`; preserve its
T1 manifest and source shards, plus the earlier frozen CLIP/branch artifacts.
Every run writes `resolved_configuration.json`, caption shards, raw answers,
`per_image_scores.json`, `word_branch_metrics.json`, `results.json`,
`comparison.csv` and `comparison.md`.
The JSON retains all threshold curves and per-image decisions. `latest.json`
points to the completed report. Outputs and weights are outside Git.

Hardware choices are **HARDWARE_DEVIATION**: sequential model loading,
batch one, CPU FP32 T1/MPNet; CPU FP32 or explicit CUDA FP16 captioning;
optional Accelerate CPU offload with a configurable 6GiB GPU weight budget and
48GiB CPU budget. Offload's weight budget is not a peak-VRAM measurement.
CUDA allocated/reserved peaks and device placement are recorded per caption;
OOM and other failures propagate with a failure receipt. The native image
processor configuration/hashes and original source shape are preserved.

The initial integrated CPU smoke completed successfully with its recorded
scientific code/config hashes. Source shape was 1280x960x3, native processor size224,
batch one; process peak RSS was 16,575,545,344 bytes for BLIP-2 and
16,886,480,896 bytes after InstructBLIP. RSS is a process lifetime high-water
mark, not isolated model memory. CUDA fields are null because CUDA was
unavailable in this execution environment; CUDA FP16/offload and 8GB-VRAM
qualification remain pending the host run. CPU timings are not paper latency
claims.

Initial smoke artifacts, predating the user's T1 acceptance clarification, are under
`/tmp/paper-image-router-baseline-v1-final-smoke/c2bc04c68bed/`.
Its `qualification.json` independently verifies that earlier scientific code/config
hashes, all425 original T1 shard hashes/payloads, both native caption shards and
the two complete frozen CLIP comparison plans/results.
Earlier caption probes remain under the `-cpu-smoke`, `-long-probe`, and
`-grounded-probe` output roots; the initial instruction probe is also preserved
separately and has been incorporated into the main runner/profile.

The initial71 focused tests covered all-word ordering/preservation, lexical and semantic
selection, inclusive threshold/ties, invalid/empty inputs, long-transcript
fallback without truncation, image-only captioning and native instruction
backend, sealed-cache/T1 reuse, matched CLIP comparison, denominator preservation
and unavailable fallback accounting. No strict/Table5 reproduction or full
new-router accuracy is claimed from that smoke test.

## Verified T1 acceptance update

The clarified implementation passes83 focused tests, including T1 above/below
threshold, confident T1 with no crop output, exactly one T1 prediction, FN for
other GT words, repeated-word matching, empty/invalid inputs, sealed annotation
sources and CLIP replay on the same metric. The real two-profile CPU smoke also
passes under `/tmp/paper-image-router-baseline-v2-t1-smoke/d2c52b5444b3/`.
Its `qualification.json` verifies current code/config/plan hashes, both native
caption shards and word-source hashes, plus all425 frozen T1 identities checked
by the runner. Word-branch replay covers all125 TRAIN and300 TEST images with
unchanged983/2204 care denominators, while caption/router smoke covers one image.

For that image, BLIP2 selects T1 with C1=0.080366 and InstructBLIP selects T3
with C3=0.168611. Both fall back at0.8. Under the exploratory single-image
calibration cut, accepted BLIP2 T1 emits **one** prediction (`1`), rather than
the four crop readings. It is FP=1 with both care GT words still FN. This checks
the requested output/counting behavior, not a population threshold or accuracy
claim. The complete GPU command above remains unchanged.
