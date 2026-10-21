# Total-Text crop variation and CLIP selector battery

**Scope:** 125 held-out Total-Text TRAIN images, 983 care words. This is an
`APPROXIMATE_TRAIN_DIAGNOSTIC`, not a strict paper/Table 5 result or an official
TEST score. The runner never loads Total-Text TEST annotations.

## Why this experiment

The paper's section 3.3 and Algorithm 1 specify connected components, component
area filtering, padding, area ordering, at most ten original-image crops and
PARSeq. They do not specify a probability threshold, the amount of padding, or
CLIP reranking. The existing Total-Text letterbox model uses 512×512 input,
native 256×256 probabilities, threshold 0.5, external contours, zero padding,
inverse-letterbox RGB crops, and no area filter/top-ten cap. These are frozen
practical baseline choices. This suite changes only the crop geometry and the
selection experiment; it does not claim to reproduce Algorithm 1.

The existing Total-Text threshold sweep already ran PARSeq on pad-0 crops at
seven thresholds. Its 0.5 baseline is 368/983 exact casefold words with
one-to-one box IoU at least 0.5. It also showed that different thresholds
produce different correct word sets. We use its frozen probabilities and OCR
caches, plus the earlier pad-0 OCR cache, without rerunning segmentation.

Earlier [LLM routing](EXPERIMENT_SUMMARY_V1_ROUTING_LLMS.md) and
[OCR candidate](OCR_EVIDENCE_GENERATION_BATTERY_TRAIN_V1.md) audits in this
repository studied selection failures on ICDAR13. They motivate separating
candidate availability, GT-blind selection and an oracle in this protocol;
their numerical outcomes do not enter the Total-Text comparison.

CLIP4STR ([paper](https://arxiv.org/abs/2305.14014)) trains CLIP-based text
recognition branches; CLIPTER ([paper](https://arxiv.org/abs/2301.07464))
fine-tunes scene-level CLIP fusion. Their results motivate testing visual/text
agreement but do not show that untrained generic CLIP can distinguish similar
spellings. The frozen [OpenAI ViT-B/32 model card](https://huggingface.co/openai/clip-vit-base-patch32)
defines the exploratory model; its exact local commit and file hashes are
recorded by the runner. The shuffled-text control probes whether the score is
actually sensitive to the proposed word.

## Fixed experiment

For each frozen float32 native probability map, binarize at
`0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65`. Use the repository's external
contours and native mask padding `0, 2, 4, 8`; map each padded box back through
the frozen inverse-letterbox geometry and crop the original RGB. The
threshold-0.5/pad-0 proposal is first for every anchor. Match contours at
other thresholds to it one-to-one by tight *native* box IoU at least 0.3.
Unmatched contours are counted separately; this battery tests recovery of
existing baseline anchors, not newly detected words. Identical crop boxes
within an anchor are encoded once, with all origins retained.

PARSeq decodes each unique crop. Existing pad-0 results are verified and reused
by crop hash; the full run also decoded the 14,514 padded variants. Frozen
CLIP ViT-B/32 sees each original RGB crop through its standard processor and
the *exact* PARSeq output as either raw text or
`a close-up photo of the written word "<PARSeq output>"`. The score is the
cosine similarity of normalized image and text features. Empty PARSeq strings
cannot win. For each baseline anchor, compare scores separately within
padding-only, threshold-only and combined pools. A tie keeps the baseline;
a second gate requires a cosine improvement above 0.02. Compare both prompts,
PARSeq mean token log probability and cyclically shuffled within-anchor text.
Selection reads no GT, caption, LLM or DeepSolo output.

The frozen reference report is read during preparation to check baseline crop
identity; its GT fields are not used for candidate generation or selection.
GT annotation files are opened after model scoring. One-to-one matching counts exact casefold
word/box pairs at IoU at least 0.5 over **all care GT words**, including
unmatched words. Reports include per-image GT and selected candidate IDs,
matched GT indices, recoveries, damages, changed anchors and complete candidate
box/OCR/CLIP traces. A labeled GT-assisted one-crop-per-anchor maximum matching
is a ceiling on available candidate quality, never a deployable selector.
Selecting the best arm on this TRAIN population would be exploratory; evaluate
one prechosen rule later on official Total-Text TEST300 for a transfer check.

## Completed Total-Text checks

All 125 TRAIN images were prepared from the frozen probability cache. Every
baseline crop box and RGB hash matches the existing threshold-0.5 report.
The preparation has 1,173 baseline anchors and 19,392 distinct crop candidates.
All 4,878 candidates that have a pad-0 origin have already cached PARSeq OCR.
The remaining 14,514 padded candidates were decoded for the completed full
PARSeq/CLIP evaluation.

A CPU `SMOKE_TEST` on Total-Text image `total_text:14` decoded all 84 crop
candidates with qualified PARSeq; 63 OCR readings were new and 21 came from
the verified caches. This verifies crop extraction and PARSeq batching on the
correct dataset, not a recognition improvement or a GPU memory result.

The cached threshold-only audit ran on the same 125 images and 983 care words:

| One-crop-per-anchor rule | Exact / 983 | Recovered | Damaged | Changed anchors |
| --- | ---: | ---: | ---: | ---: |
| Frozen threshold-0.5/pad-0 baseline | 368 | 0 | 0 | 0 |
| PARSeq confidence over threshold variants at pad 0 | 403 | 47 | 12 | 843 |
| GT-assisted threshold-only oracle | 440 | 74 | 2 | 74 |

The oracle shows a threshold-only candidate ceiling of 72 net words over the
baseline under the one-anchor constraint. PARSeq confidence captures 35 net
words, while changing many crops. These figures do **not** assess padding or
CLIP. The existing global threshold-0.65 result (437/983) is a separate rule
and cannot be interpreted as CLIP performance. Machine-readable per-image
audit: `/tmp/totaltext-crop-clip-local-suite-b79b311ba43a/cached-threshold-audit/cached_threshold_audit.json`
on the present host; rerun the command below to regenerate it elsewhere.

### Full frozen-CLIP TRAIN run

The pinned CLIP ViT-B/32 snapshot at commit
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` was evaluated on all 125
TRAIN images and 983 care words. The complete per-image and per-candidate
report is `/tmp/totaltext-crop-clip-local-suite-b79b311ba43a/full-clip-train/results.json`
(SHA256 `101baf43e5758aa309af6059b10b6c5c792128eb7f9219fc2a2eb4b3d5ba6bc4`).

| One-crop-per-anchor rule | Exact / 983 | Recovered | Damaged | Changed anchors |
| --- | ---: | ---: | ---: | ---: |
| Frozen threshold-0.5/pad-0 baseline | 368 | 0 | 0 | 0 |
| PARSeq confidence, threshold variants | 403 | 47 | 12 | 843 |
| PARSeq confidence, all variants | 392 | 54 | 30 | 1,039 |
| CLIP quoted-word prompt, thresholds only, margin 0.02 | 374 | 20 | 14 | 254 |
| CLIP quoted-word prompt, all variants, margin 0.02 | 200 | 19 | 187 | 867 |
| CLIP shuffled-text control, all variants | 171 | 22 | 219 | 1,118 |
| GT-assisted oracle, thresholds only | 440 | 74 | 2 | 74 |
| GT-assisted oracle, all variants | 462 | 95 | 1 | 95 |

The best predeclared CLIP arm gains 6 words over the baseline but is 29 words
below PARSeq confidence on threshold variants. Padding increases the oracle
ceiling by 22 words over threshold variants alone, yet every padding-inclusive
CLIP arm damages many previously correct words. The unrestricted shuffled-text
control is similarly poor. These observations suggest that frozen, generic
CLIP crop/text cosine is not a reliable selector for these Total-Text crops.
This is a TRAIN comparison across multiple arms; the +6 result is exploratory
and cannot support a generalization claim without a frozen TEST rule.

The run used one RTX 3090, float32 inference, batch size 16 and sequential
PARSeq/CLIP loading. CLIP peak allocated/reserved VRAM was 724.6/792.0 MiB;
PARSeq peak was 135.9/164.0 MiB. This qualifies the observed execution on
that GPU, not the target RTX 3070.

## One-command execution

On the current host, run the 26 focused tests, full 125-image cached threshold
audit and one-image Total-Text PARSeq smoke with one short command:

```bash
bash scripts/run_totaltext_crop_clip_tests.sh
```

It writes outside the repository under `/tmp/totaltext-crop-clip-local-suite-<code-hash>`
by default, so an updated protocol gets a new output directory while rerunning
the same code resumes its existing shards. Set `TOTALTEXT_SUITE_OUTPUT` to
change the location. It uses the
qualified local artifacts and still verifies their hashes. To include the full
padding/PARSeq/CLIP stage once a pinned CLIP snapshot and working GPU are
available, use `CLIP_MODEL_DIR=/path/to/snapshots/FULL_COMMIT_SHA bash
scripts/run_totaltext_crop_clip_tests.sh`. The snapshot directory name must be
its full 40-character commit SHA. Without it, the shortcut reports the CLIP
stage as pending rather than suggesting that CLIP was evaluated.

For a first full run on a machine with internet and a working NVIDIA driver,
download the pinned official
[OpenAI CLIP ViT-B/32 revision](https://huggingface.co/openai/clip-vit-base-patch32/commit/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268)
into the external artifact directory and launch the suite:

```bash
CLIP_REV=3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268
CLIP_MODEL_DIR="/extra1/bmartelli/context-str-artifacts/clip/$CLIP_REV"
.venvs/context-str-main-py311/bin/huggingface-cli download openai/clip-vit-base-patch32 \
  --revision "$CLIP_REV" \
  --include config.json preprocessor_config.json tokenizer.json tokenizer_config.json \
            special_tokens_map.json vocab.json merges.txt pytorch_model.bin \
  --local-dir "$CLIP_MODEL_DIR" && \
CLIP_MODEL_DIR="$CLIP_MODEL_DIR" bash scripts/run_totaltext_crop_clip_tests.sh
```

The runner hashes the downloaded model files and excludes mutable Hugging Face
download metadata from the model identity. The full report above was produced
after the pinned snapshot became available on the host.

The host script first runs focused tests, then the chosen battery stage. It
does not download weights or install packages. Supply explicit input paths and
a new output directory outside this repository. For the completed CPU cached
audit:

```bash
bash scripts/run_totaltext_crop_clip_battery_host.sh \
  --data-root /path/to/totaltext-curved-word-boxes-v4 \
  --probability-cache /path/to/totaltext-letterbox-threshold-sweep-train-v1 \
  --baseline-ocr-cache /path/to/validation-pad0-parseq-v1/ocr \
  --output-dir /tmp/totaltext-crop-clip-audit \
  --device cpu --cached-threshold-audit
```

For the full PARSeq, CLIP and comparison battery, supply a locally pinned
`openai/clip-vit-base-patch32` snapshot. Its directory basename and revision
must be the same full 40-character commit SHA. The script verifies local
weights, config, processor and tokenizer before model inference.

```bash
bash scripts/run_totaltext_crop_clip_battery_host.sh \
  --data-root /path/to/totaltext-curved-word-boxes-v4 \
  --probability-cache /path/to/totaltext-letterbox-threshold-sweep-train-v1 \
  --baseline-ocr-cache /path/to/validation-pad0-parseq-v1/ocr \
  --parseq-source /path/to/parseq-source-at-pinned-revision \
  --parseq-source-archive /path/to/pinned-parseq-source.tar.gz \
  --parseq-checkpoint /path/to/qualified-parseq-checkpoint.pt \
  --output-dir /path/outside/repository/totaltext-crop-clip-train \
  --clip-model-dir /path/to/clip/snapshots/FULL_COMMIT_SHA \
  --clip-revision FULL_COMMIT_SHA \
  --device cuda:0 --batch-size 16
```

For a geometry-only smoke, add `--prepare-only --limit 1 --device cpu` and omit
the model arguments. For a one-image PARSeq smoke, use `--parseq-only --limit 1
--device cpu` with the PARSeq arguments and omit CLIP arguments. `--limit N`
labels outputs as `SMOKE_TEST`; the full 125
TRAIN samples are required for the TRAIN diagnostic. Outputs are restartable
only when the saved plan, source hashes and model contracts match. PARSeq and
CLIP load sequentially in float32; CUDA peak allocated/reserved memory and OOM
details are saved when GPU execution is available.

The completed full run records the RTX 3090 memory peaks. RTX 3070 memory and
independent TEST transfer remain unmeasured.
