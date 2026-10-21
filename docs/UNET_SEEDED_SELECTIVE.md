# Nucleus-guided links and selective crop refinement

**Approximate reproduction / experimental extension**, decision USS-001 in
`PAPER_AMBIGUITIES.md`. The local paper sections3.2–3.3 and Algorithm1 specify
attention/ResNet50 segmentation and component localization. They do not specify
pixel links, kernels, a second pass, or the new selection rules. STRICT paths,
original datasets, checkpoints and previous experiment outputs remain unchanged.

This experiment replays the **completed** `unet-crop-links-v1` predictions.
There is **no new training or U-Net inference**. Both hypotheses can first be
tested using the same frozen logits, RGB context windows and held-out TRAIN
population; PARSeq is rerun only for RGB crops not already in a compatible cache.

## Six arms and matched comparisons

| Arm | Frozen maps | Extraction and selection |
| --- | --- | --- |
| `control` | Original control epoch2 | Original first-pass exported supports |
| `crop_word_all` | Crop word epoch16 | Original unconditional second-pass exports |
| `crop_links_raw` | Crop links epoch4 | Original reciprocal graph exports |
| `crop_links_seeded_all` | Same crop links epoch4 | New nucleus-guided soft link growth, all crops |
| `crop_word_selective` | Same crop word epoch16 | GT-free selection and conservative replacement |
| `crop_links_seeded_selective` | Same crop links epoch4 | Both new procedures |

Selected checkpoint and validation hashes are pinned in
`configs/approximate/unet_seeded_selective_v1.json`. Source experiment completion,
configuration, helper hashes, checkpoint bytes, prepared-crop manifest receipt,
first-pass maps, per-crop evidence, exported supports and RGB checksums are checked.
Changing a dependency requires a new output root, rather than cache mixing.

Compare seeded links against `crop_links_raw` to isolate the extraction change;
compare each selective arm against its corresponding unconditional arm. Word and
link checkpoints differ, so their direct difference is not a controlled estimate
of adding links. The summary includes per-word gains/losses for matched ablations.

## Soft growth from nuclei

Existing thresholds remain text.65 and kernel.5. Eight-connected kernel
components are immutable markers. A priority flood expands them only through
predicted foreground. For an adjacent pixel pair, the edge barrier is the mean of:

1. `1 - min(link_forward, link_reverse)`;
2. `1 - min(confidence_at_source, confidence_at_destination)`, using the existing
   normalized word/kernel watershed landscape.

The path barrier is the maximum edge barrier encountered. Queue ties use path
length, marker ID and coordinates. A low link increases the traversal cost; it
does not remove a pixel or create another instance. An entirely unseeded,
disconnected foreground component survives once as a fallback. Foreground and
marker identities are asserted unchanged. No component-area filter is added.

This prevents link-induced singleton proliferation. It cannot split an already
fused kernel or associate words across disconnected foreground islands. Float64
flood arithmetic is an explicit ASSUMPTION, not a change to the frozen float32
network probabilities.

## Selective replacement

Only first-pass probability/kernel/instance maps choose a seed. A crop is flagged
when any of these is true:

- Mean word probability over its support is below.90.
- The support contains zero or multiple kernel components.
- Kernel area is below5% of the support.
- Foreground occupies below35% of its tight native box.

These are fixed heuristic proxies, not measured guarantees that a word is wrong.
The defaults were declared before the new evaluation. No GT, transcript, PARSeq
confidence or held-out correctness label influences selection. A lazy evidence
loader is called only for flagged seeds in a selective arm.

Children must touch the original seed. Replacement requires at least85% coverage
of its original support, union area growth at most1.75, and at most3 children.
Every proposed RGB box may cover at most10% of another first-pass seed's
foreground. This detects contamination even when predicted masks do not overlap.
Rejected replacements keep the original exact support and RGB box.

Selective arms skip cross-seed duplicate merging, which could change protected
seeds. Unconditional seeded growth retains the previous support-overlap merge. A
whole-image fallback is explicitly selected and is not represented as a known
good seed. Selection, acceptance/rejection reasons, metrics and decoder counts
are saved per seed in the JSON reports. Conservative guards can reject useful
changes; counts and paired word identities make this visible.

## Evaluation and run

All six geometries are frozen before scoring consults annotations. Original
exclusive RGB bbox maximum one-to-one IoU.5 gives precision/recall/F1; .25/.75
are secondary. Merge/split counts use any shared valid original owner pixel.
Wholly ignored supports are excluded; partially ignored crops remain intact.
All983 care words on the125-image reused TRAIN holdout remain in denominators.
There is no padding, area filtering or top10 cap in this isolated diagnostic.
The three reference arms must exactly replay every per-image geometry count.

PARSeq uses the existing hash-qualified checkpoint/runtime, FP32, AR=True,
refine_iters1 and batch8. Exact casefold plus IoU.5 is primary; punctuation/symbol
stripping is secondary. Backend-specific OCR cache reuse is read-only for the
source run. CPU/GPU or batch mismatch fails explicitly. Segmentation replay and
PARSeq are separate processes; the latter records allocated/reserved VRAM and
propagates OOM. CUDA/RTX3070 qualification requires host execution.

From the repository root:

```bash
bash scripts/run_unet_seeded_selective_host.sh \
  artifacts/totaltext-curved-word-boxes-v4 \
  /extra1/bmartelli/context-str-runs/unet-crop-links-v1 \
  /extra1/bmartelli/context-str-runs/unet-seeded-selective-v1 \
  /extra1/bmartelli/context-str-artifacts/parseq
```

The command performs CPU replay, real GPU PARSeq and writes
`comparison.md/json`, `geometry_comparison.json`, per-image supports, seed
decisions and `ocr_comparison.json`. Reissuing it verifies the same contract,
preserves compatible geometry and resumes recognition from its immutable cache.
No installation or downloads occur. A separate `--limit-images N` replay is
labeled `smoke-test result`; final OCR rejects it unless `--smoke` is explicitly
selected. Smoke reports never become final scientific results.

Single seed, repeatedly reused TRAIN validation and heuristic thresholds limit
generalization. Official TEST is untouched. No result is automatically promoted
to the main pipeline, and the user's93% GT-crop OCR is not remeasured.
