# Grouping threshold and window crops

This is **approximate reproduction**, a diagnostic on the frozen Total-Text
held-out TRAIN-125 population. It adds an experimental grouping stage alongside
the crop recovery suite, leaving the paper Algorithm 1 path unchanged. Paper
sections 3.2/3.3 establish masks and connected localization; they do not specify
grouping of alternatives from different thresholds/windows. All new grouping
parameters are `ASSUMPTION`, recorded as `TT-CROP-WORD-GROUPS-V1`.

## Run

```bash
bash scripts/test_totaltext_groups.sh
```

The command runs the focused tests, finds the latest completed TRAIN-125 crop
suite under the existing local paths profile, and analyzes its frozen outputs.
It runs on CPU and performs no model inference or download. To select a source
explicitly:

```bash
bash scripts/test_totaltext_groups.sh --source /path/to/completed/crop-suite
```

Use `--limit 1` for a partial smoke run or `--output /path/to/new/directory`.
The default output is `SOURCE/word_groups/RESOLVED_PLAN_HASH/`. Config lookup is
limited to `configs/approximate/`; incompatible strict configurations fail.

## Method

1. Keep all unmodified recipes belonging to baseline, threshold tree, or tiled
   segmentation. Recipe deduplication is inherited from the source suite.
   Reconstruction/adjacent-union hypotheses of the threshold tree are included.
   Ignore membership/provenance from other generators for grouping decisions.
2. All boxes are already inverse-letterboxed into original RGB coordinates.
   A window's local coordinates are never compared directly to another view.
3. Geometric compatibility requires bbox IoU above the configured value and
   per-axis center displacement divided by the smaller respective box dimension
   at most the configured center limit. No horizontal-text assumption is used.
4. Build deterministic core groups. The `complete_link` strategy attaches a
   crop only if it is compatible with **every** existing core member. Order
   prefers support from distinct windows and threshold/view pairs, then smaller
   boxes and candidate-ID ties. The `connected` negative control takes connected
   components of the pairwise-compatibility graph, permitting transitive chains.
5. Find crops covering two disjoint smaller group reference boxes. A reference
   is an existing box nearest the median coordinates, never a newly rendered
   crop or an OCR-selected representative. Record competing group links and
   add the **same candidate ID** to both/all implicated groups. Its original
   core membership is retained. `parent_only` marks groups whose core members
   are all detected composite hypotheses; these remain explicit alternatives.

Shared crop membership does not trigger extra PARSeq inference. Per-crop text
and confidence never enter the grouping API. Only geometry and generator
provenance are copied across the inference boundary. Every grouping file is
saved before the corresponding annotation/evaluation shard is opened.

The sharing detector is an approximate containment test. Curved/overlapping
word boxes, fragmented cores, crops containing very many tiny words, and a word
with two separated character clusters can violate its assumptions. A shared
parent denotes competing spatial hypotheses, not proof that it contains two
complete words. Exact foreground support could improve these cases, but the
current prototype does **not** reconstruct pixel masks from bounding boxes.

## Evaluation

- Reuse the upstream exact Unicode casefold + P/S removal and source-box
  IoU >= .5 witness edges, retaining all 983 care words in the denominator
  (982 have nonempty normalized strings).
- Unrestricted reference: maximum distinct-crop/GT bipartite matching.
- Group oracle: maximum flow `source -> group -> crop-in -> crop-out -> GT ->
  sink`, with unit capacities on groups, original crops and GT words. This
  asks how many words remain recoverable if a future selector emits at most one
  reading per group. Sharing one crop cannot duplicate a single reading for two
  instances of the same spelling. No actual selector has been implemented.
- Core pair precision: among core-member pairs uniquely assignable to a GT by
  maximum bbox IoU >= .5, the fraction assigned to the same GT instance.
  Shared parents and unlabelled candidates are excluded. This is a geometric
  diagnostic, not a semantic proof or all-candidate grouping accuracy.
- Fragmentation: number of groups containing a correct witness for each
  recoverable word. Smaller values indicate more consolidated alternatives.

Results preserve per-image grouping, candidate IDs, shared-parent links,
source/plan/code hashes, geometry diagnostics and matched GT indices. Source
prepared/OCR shard hashes must agree with the frozen evaluation contract.
Parameter comparisons use the same TRAIN data; independent transfer and the
eventual prediction-selection method remain `UNRESOLVED`.

## First full cached experiment

Source: `d957d64fb251`, 125 images, **18,912** threshold/window/baseline crops,
unrestricted oracle **782/983**. All input recipes remain available.
Center limit .75; exact parameters are in
`configs/approximate/totaltext_crop_word_groups_train_v1.json`.

| Strategy / IoU | Groups | Shared parent crops | Group oracle correct | Lost count | Core pair precision | Mean groups per recoverable word |
|---|---:|---:|---:|---:|---:|---:|
| Connected / .45 | 5,757 | 382 | 698 | 84 | .7744 | 1.06 |
| Connected / .55 | 6,851 | 976 | 759 | 23 | .9322 | 1.20 |
| Connected / .65 | 8,119 | 1,943 | 778 | 4 | .9913 | 1.52 |
| Connected / .75 | 9,651 | 2,782 | 781 | 1 | .9955 | 2.06 |
| Complete link / .45 | 7,711 | 1,900 | 780 | 2 | .9962 | 1.76 |
| Complete link / .55 | 8,683 | 2,426 | 781 | 1 | .9977 | 2.04 |
| Complete link / .65 | 9,858 | 2,853 | 781 | 1 | .9984 | 2.45 |
| Complete link / .75 | 11,389 | 3,077 | 781 | 1 | .9992 | 3.14 |

Only 6,523/18,912 crops receive unique audit geometry labels. The group totals
include background/fragment candidates and explicit parent alternatives; they
are **not** estimates of the number of true scene words. The .55 complete-link
setting is a conservative starting point; .45 consolidates more alternatives
at the cost of one additional oracle word. This is an experimental comparison,
not an independently validated parameter choice.

The residual .55 failure is TRAIN image `total_text:45`: overlapping TOMATO and
KETCHUP regions generate nearly identical boxes with different correct PARSeq
readings, which end up in one core group. Geometry alone does not resolve this
case. The audit identifies the issue without modifying grouping from its GT.

Verification: 11 focused unit tests and 23 existing crop-suite tests passed
(34 total), real cached smoke image `total_text:14` preserved 2/2 words, and
all 125 images completed with the final configuration and code. The first
environment probe using bare `python` failed because that executable is absent;
the launcher uses the existing qualified Python environment. No package was
installed. No new GPU model run or VRAM measurement is applicable to this CPU
geometry/cache analysis; upstream model execution metadata remains in the source.
