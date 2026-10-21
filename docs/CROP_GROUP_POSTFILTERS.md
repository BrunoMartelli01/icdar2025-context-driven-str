# Frozen crop-group output postfilters

This is an **approximate reproduction diagnostic**, separate from the paper's
Algorithm 1 and Table 5. It consumes the previously frozen Total-Text TEST
selection shards. No segmentation, PARSeq or CLIP model is run again.

Run both postfilters and their combination with:

```bash
bash scripts/test_crop_group_postfilters.sh
```

The launcher discovers the newest complete 300-image TEST selector artifact
under the configured output root. To use a specific completed artifact, pass
`--selector-dir PATH`. The resolved plan records that path, source hashes,
configuration and code hashes. Results are written outside the repository.

## Inference rules

Input: the frozen `mask_completeness_confidence` selection, one crop at most per
geometric group. The original selected words and crop boxes are never changed.

1. **Duplicate suppression.** Sort selected outputs by their frozen selection
   score. An output loses to an already retained output only if its original
   boxes overlap by IoU at least 0.5 **and** its boxes' shared soft foreground
   probability mass has IoU at least 0.5. The exact whole/tile-aligned soft
   probability canvas from the source run is reconstructed and hash-checked.
   If two groups share a parent crop but predict different readings, both are
   retained so that a multiword parent cannot erase distinct child words.
   Equal-score ties use stable group/candidate IDs. Zero soft support uses the
   geometric overlap as a documented fallback.
2. **Evidence abstention.** Keep a group if one of its *core* proposals was an
   original baseline crop, or if two distinct core RGB crop hashes both produce
   the selected normalized word. Shared parent crops do not count as core
   corroboration. This rule uses no dictionary, minimum word length, learned
   confidence cutoff, annotations or dataset-dependent output budget.
3. **Combined arm.** Apply evidence abstention, then duplicate suppression.

The 0.5 overlap values are fixed generic geometric assumptions declared before
the TEST postfilter run. They have **not** been optimized on Total-Text metrics.
Earlier TEST diagnostics showing many unsupported groups motivated this study;
this result is exploratory and cannot serve as an unbiased transfer result.
The abstention rule is deliberately structural: PARSeq character probabilities
are not calibrated as probabilities that a group contains real text. Repeated
readings from nearby crops can still be correlated; baseline proposals can
still be false positives. Duplicate suppression can prefer an incorrect word
when its frozen score is higher. These limitations remain unresolved.

All postfiltered per-image selections, including rejected group IDs and reasons,
are frozen before ground-truth evaluation. The evaluation then uses the existing
local one-to-one normalized-text plus axis-aligned-bbox-IoU>=0.5 matcher. Its
Hmean is a **local proxy**, not the official polygon Total-Text spotting metric.

## Frozen 300-image TEST result

Artifact: `/tmp/totaltext-crop-group-test-suite/30643625ace4/postfilters-d7fee644accb/results.json`.
There are 2,204 care words in the local evaluation denominator.

| Arm | Final predictions | Correct | Unmatched | Precision proxy | Recall proxy | Hmean proxy |
|---|---:|---:|---:|---:|---:|---:|
| Frozen selector input | 18,052 | 1,722 | 16,330 | 9.54% | 78.13% | 17.00% |
| Duplicate suppression | 15,462 | 1,637 | 13,825 | 10.59% | 74.27% | 18.53% |
| Evidence abstention | 6,509 | 1,600 | 4,909 | 24.58% | 72.60% | 36.73% |
| Abstention then suppression | 5,667 | 1,548 | 4,119 | 27.32% | 70.24% | 39.33% |

The rules were frozen before this postfilter run. Transfer to a different
dataset and comparison with official polygon evaluation remain unqualified.
