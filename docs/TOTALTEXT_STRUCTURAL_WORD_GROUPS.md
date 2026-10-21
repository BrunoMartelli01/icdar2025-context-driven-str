# Structural grouping of frozen Total-Text crop proposals

This is an **approximate TEST diagnostic**, separate from paper Algorithm 1.
It does not train, rerun segmentation or PARSeq, or change any original crop.
The suite regenerates proposal masks and checks every candidate ID, box, method
and origin against the frozen 47,538-crop source before grouping. The proposal
trace is used only for grouping. A crop covering multiple word hypotheses may
belong to more than one group; its original OCR output and capacity remain one.

Run all four ablations and the baseline with:

```bash
bash scripts/test_totaltext_structural_groups.sh
```

`--limit N` runs a smoke test; omit it for the complete official TEST-300.
The output location is printed at the end and contains resolved configuration,
source/code hashes, per-image group memberships, proposal-trace event counts and
post-hoc evaluation, plus `results.json` and `summary.md`.

## Ablations

1. `lineage`: link compatible whole-image component and seed-piece crop groups
   whose foreground footprints are nearly contained and of comparable area.
   Complete-link merging prevents a chain of overlapping words from fusing.
2. `lineage_footprint`: also match whole-image and tiled proposals on a common
   original-image grid. A tile component cut by its window edge can join a
   larger matching whole-image component when its support is contained.
3. `lineage_footprint_boundaries`: split low-threshold components into local
   seed runs using distance along their principal axis and transverse overlap.
   Disjoint pieces are joined only if an existing two-piece crop has a PARSeq
   reading equal to the concatenation of the piece readings. A redundant
   parent-only group is removed only when all its crop IDs survive in other
   groups. Multiword parent crops remain shared among child groups.
4. `lineage_footprint_boundaries_admission`: retain structurally confirmed
   groups or an isolated group with a nonempty OCR reading and foreground mass
   above the median of groups in its own image. This is an image-relative
   abstention experiment, not a calibrated textness probability.

The 256-cell projection grid and dimensionless comparison bounds are declared
in the resolved experiment configuration. They are `ASSUMPTION`, not values
specified by the paper. The third method uses frozen PARSeq text during
grouping, while the first two use mask geometry only. None uses GT during group
construction. The small TEST smoke test exposed that gap proximity alone
can merge different words; the concatenation guard was added before the full
run. Consequently the full TEST comparison is exploratory rather than an
independent validation of that rule.

## Metrics and interpretation

All 2,204 care GT words remain in the full TEST denominator. A candidate is a
correct witness when casefolding and removal of Unicode punctuation/symbols
make its PARSeq reading equal to a GT word and the original-coordinate
axis-aligned boxes have IoU >= 0.5. The source pool's one-to-one oracle is
1,823 words. For each grouping, the suite reports its one-group/one-original-
crop oracle, group count, number of recoverable words appearing in multiple
groups, mixed core groups, and the unchanged mask-completeness/confidence
selector's outputs and correct matches. Mixed-core and unlabeled-group counts
are geometric audits under this local IoU convention, not official polygon
spotting metrics. Reducing the number of groups is useful only if recoverable
words survive and selection improves.

## Complete TEST-300 result

Report: `/tmp/totaltext-crop-group-test-suite/ba82c525eadb/source/structural_groups/443e094771fb/results.json`.

| Stage | Groups | One-per-group oracle | Correct selected | Outputs | Mixed core groups |
|---|---:|---:|---:|---:|---:|
| Original grouping | 21,284 | 1,823 | 1,722 | 18,052 | 16 |
| Lineage | 18,360 | 1,822 | 1,546 | 15,243 | 23 |
| + footprint across windows | 17,343 | 1,820 | 1,507 | 14,298 | 24 |
| + boundary/parent handling | 16,596 | 1,751 | 1,396 | 13,572 | 238 |
| + image-relative admission | 9,629 | 1,732 | 1,379 | 9,533 | 236 |

The first two structural stages preserve nearly all 1,823 recoverable words:
only 1 and 3 are lost in the one-per-group oracle. The unchanged selector loses
176 and 215 correct outputs respectively, because combining alternatives
forces a single choice in groups previously allowed to emit several guesses.
All 199 baseline-selected words lost under lineage remain reachable by a crop
in its new group; this is a selection failure, not missing OCR coverage. The
third stage merges genuinely distinct words much more often (238 mixed core
groups versus 24) and loses 72 oracle words. The fourth stage loses another
19 oracle words while reducing outputs; it is a precision/recall tradeoff, not
a recovery improvement. None of the four stages approaches the 2,204 annotated-word
group count without substantial loss or extra outputs. These observations are
local axis-aligned IoU diagnostics, not official Total-Text polygon hmean.
