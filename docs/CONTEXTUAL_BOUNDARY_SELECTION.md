# Contextual selection of OCR word boundaries

**Approximate reproduction / exploratory transcription experiment.** This study
keeps U-Net, original RGB boxes, PARSeq weights and OCR observations fixed. Its
output is transcription, without a spatial matching claim. It tests whether
context and lexical/visual evidence can distinguish useful word divisions from
damaging divisions of complete words, names and compounds.

Before implementation, local paper `papers/2503.15639v2.pdf` sections3.3–3.5,
the frozen OCR reconstruction code and its failure analysis, and the negative
copied-character gap-head experiment were inspected. These new decisions are
absent from the paper; the STRICT/context/DeepSolo path is unchanged.
Decision: `CONTEXTUAL-BOUNDARY-SELECTION-001` in `docs/PAPER_AMBIGUITIES.md`.

Implementation: `recognition/contextual_boundary_selection.py`,
`scripts/run_contextual_boundary_selection.py`,
`scripts/audit_contextual_boundary_selection.py`,
`scripts/analyze_contextual_boundary_selection.py`.
Configuration: `configs/approximate/contextual_boundary_selection_v1.json`.
Tests: `tests/recognition/test_contextual_boundary_selection.py`.

## Frozen inputs and folds

Reuse1030FIT/100calibration/125evaluation/300TEST images and original M/K blocks
from `text-reconstruction-tests-20261023-v1/full`. Hash-check every input and
pretrained asset. Image-separated3-fold ByT5-small training produces FIT
proposals without observing that image's annotations. Fold-specific word priors
and multi-block training operations are rebuilt from the other folds only.
The OOF generator retains the historical corruption recipe and one-epoch
schedule, including documented potential geometric/synthetic label noise; this
study changes selection rather than claiming to repair those generator issues.

The historical selected epoch1 ByT5 checkpoint remains the final generator on
calibration/evaluation/TEST. OOF models see fewer FIT images; this distribution
difference is explicit. Models and checkpoints are external artifacts. The same
OOF generators are reused across selector seeds, not independent full retrains.

## Candidate pool and controlled comparisons

For each eligible alphabetic original string, retain KEEP plus strict
copied-character boundary alternatives from greedy/beam4 ByT5 and at most6
lexical alternatives. At most3words, length4–32Unicode characters. Restore
original casing by copying original characters. Nonalphabetic, empty and other
ineligible originals remain predictions. Candidate generation never reads GT.

Use exactly the same candidates for four matched multiclass HGB selectors:

1. Lexical counts/trigrams, external vocabulary, casing and original OCR evidence.
2. The same plus frozen BERT-base target-only PLL without neighbouring context.
3. The same plus target-only PLL with GT-blind local original OCR neighbours.
4. The same plus RGB contrast/edge projection features from the identical
   original crop. Candidate cut positions assume proportional character widths:
   this is a weak alignment proxy, not true character alignment or a general
   learned multimodal verifier. No new boxes or OCR reads are introduced.

FIT labels distinguish actual recovered lexical occurrences, damaged occurrences
and neutral edits. Utility is P(recovery without damage)−P(damage). No deletion
or false-positive-only cleanup is a positive reconstruction label. Each
observation chooses exactly one candidate; KEEP is always available.

Train120iterations,7leaves, minimum30examples/leaf, learning rate.06,L2=1,
three seeds. Choose thresholds only on100calibration images, separately for
damage budgets0and2. Maximize recovered correct count, then F1, then conservative
threshold. Also retain raw zero-utility controls; these are not promoted.
Write all selections before parsing evaluation/TEST transcripts for scoring.

An additional sensitivity was declared before primary selections/results existed:
`scripts/calibrate_contextual_boundary_sensitivity.py` with
`configs/approximate/contextual_boundary_threshold_sensitivity_v1.json`.
On the fixed primary-seed scores, compare every calibration decision breakpoint
against the original grid, with both correct-count-first and F1-first objectives
and the same damage budgets. No new model or candidate is fitted. Save all
supplementary choices before evaluating their evaluation/TEST outputs. More
calibration search can overfit the small calibration cohort; this is a reported
sensitivity, not independent validation or TEST tuning.

## Evaluation and limitations

Primary metric: micro per-image Unicode-casefold whitespace-token multiset;
punctuation retained; all care GT, duplicates and empty observations retained;
no IoU, ignore-prediction neutralization or case-sensitive claim. Report
correct/output/FP/FN counts, recovered and damaged words, precision/recall/F1,
image-paired bootstrap intervals and seed sensitivity. GT-assisted candidate
oracles are diagnostics, not deployable results. Report local-edit labels
separately from joint per-image gains, since lexical multiplicities couple edits.

Historical cohorts were repeatedly explored, the source segmenter saw
calibration images, and TEST578/TRAIN212 plus TEST656/TRAIN81 have identical
RGB. Primary results keep all2204TEST care words; separately assess298images.
No independent-generalization,80%F1, official spotting or strict-paper claim.

No packages or weights are installed. Qualify actual GPU before each stage;
FP32, no AMP/TF32, sequential ByT5/BERT residency, deterministic settings,
allocated/reserved peaks and OOM propagation are recorded. RTX3090 does not
qualify RTX3070/8GiB. All numerical choices are isolated APPROXIMATE assumptions.

## Execution

External root:
`/extra1/bmartelli/context-str-runs/contextual-boundary-selection-20261024-v1`.
Run from an immutable source snapshot with explicit paths:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 /explicit/python \
  -m scripts.run_contextual_boundary_selection --mode approximate \
  --config configs/approximate/contextual_boundary_selection_v1.json \
  --output /explicit/new/output --stage all
```

Add `--smoke` for24FIT and4images per other role. Full-run completion requires
the results, independent audit and qualification receipt; implementation or
smoke success alone does not establish efficacy.

## Completed results and interpretation

All 1030 FIT, 100 calibration, 125 evaluation and 300 TEST images completed.
Three image-separated generators each trained for 1125 updates on 18,000
records; the final historical generator weights were unchanged. All four
selectors use the identical candidate pool. No additional PARSeq reading,
localization change, dependency installation or model download occurred.

Primary TEST: 2204 care words, casefold lexical matching, punctuation retained.
The table uses the predeclared grid, primary selector seed and calibration
damage budget 2. That budget constrains calibration, not TEST damage counts.

| Arm | Correct | Outputs | FP | Recovered | Damaged | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original OCR | 1322 | 2824 | 1502 | 0 | 0 | 52.5855% |
| Historical ByT5, boundary-only control | 1366 | 2948 | 1582 | 73 | 29 | 53.0280% |
| Lexical selector | 1331 | 2839 | 1508 | 11 | 2 | 52.7860% |
| Lexical + word-only BERT | 1342 | 2848 | 1506 | 22 | 2 | 53.1275% |
| Add neighbouring OCR context | 1332 | 2834 | 1502 | 10 | 0 | 52.8781% |
| Add same-crop RGB projection | 1343 | 2860 | 1517 | 24 | 3 | 53.0411% |

Word-only BERT improves baseline F1 by 0.5420 percentage points, conditional
95% paired-image interval [0.0730, 1.0550]. Its gain over the historical
boundary-only control is just 0.0995 points, interval [-0.7778, 0.9403]: the
experiment does not establish a robust F1 improvement over that control.
It changes the recovery/damage tradeoff substantially, accepting fewer edits.
The historical generator and new candidate pool differ, so this historical
comparison is a whole-method comparison, not an isolated selector ablation.

The controlled word-only BERT versus lexical comparison is +0.3414 points,
interval [0.0498, 0.7159]. Adding neighbouring OCR is -0.2493 points relative
to word-only BERT, interval [-0.7356, 0.2086]. Thus this TEST experiment does
not support an additional context benefit. The internal evaluation reverses
their ordering: 51.3208% word-only versus 51.6769% contextual. These cohorts
remain historically explored, not independent validation.

With calibration damage budget 0, the grid lexical/word-only policies choose
KEEP. Context+RGB selects a limited policy with TEST 8 recovered/0 damaged,
52.8197% F1; evaluation has 5 recovered/1 damaged. Zero calibration damage is
not a guarantee of zero damage on new images. All three HGB seeds produce
identical results, which measures algorithmic stability on fixed proposals,
not independent generator training variability.

### Threshold and objective sensitivity

The supplementary search independently verified all 1942 CAL breakpoints and
16 policies. Under correct-count-first selection and damage budget 2, TEST
results are 52.8571% lexical, 53.1411% word-only, 52.8175% context and 53.1565%
context+RGB. The last policy recovers 23/damages 2; word-only recovers 26/damages
3. F1-first calibration selects more conservative context/RGB policies, with
52.7215%/52.6525% TEST F1. Finer calibration therefore does not produce a large
gain; TEST values remain descriptive and do not select a promoted winner.

### Why selection still fails

- FIT has 461 beneficial, 8902 damaging and 2431 neutral alternatives. TEST has
  138 beneficial, 2358 damaging and 781 neutral alternatives. These count
  alternatives, not distinct words: a word can have several wrong divisions.
  Only about 4.2% of TEST alternatives recover lexical matches without damage.
- Of the 138 beneficial TEST alternatives, 80 have no selected neighbouring
  OCR context. Only 1012 of 1561 neighbour occurrences exactly match a care GT
  spelling; other neighbours include partial/fused text and errors, rather than
  necessarily background. This is an exact-match proxy, not proof that every
  other neighbour is meaningless or a causal isolation of context noise.
- Word-only BERT correctly accepts `SANFRANCISCO -> SAN FRANCISCO`,
  `TOURISTVEHICLE -> TOURIST VEHICLE` and
  `UNIVERSITISAINSMALAYSIA -> UNIVERSITI SAINS MALAYSIA`. It protects
  `AHEAD` and `VITABELLA`, but also rejects the correct `Krispy Kreme`
  proposal. It selects `BananaLeaf -> Ban ana Leaf`, a damaging overdivision.
- A linguistically plausible division can disagree with annotation granularity:
  TEST639 explicitly annotates `HolidayInn` as one unit, whereas TEST93 has
  separate `Krispy` and `Kreme` annotations. Manual inspection of the original
  TEST639 RGB shows the Holiday Inn sign. This is a two-sample convention
  diagnostic, not a dataset-wide annotation-error claim. GT is unchanged.
- Some edits recover only part of a string: for example a recovered word in
  `AUTHENTIC RAKERY` does not establish correction of `RAKERY`. A neutral split
  can increase output count and FP without improving any lexical match.
- The current pool supports at most 1445 correct lexical occurrences under
  GT-optimal selection, versus 1322 originally; neural-only supports 1441,
  lexical-only 1416. The word-only calibrated selector reaches 1342, leaving
  substantial selection regret. These are feasible correct-count diagnostics,
  not achieved inference results or F1-optimal policies.
- Since divisions cannot reduce the original 2824 outputs, this exact pool
  has a loose optimistic F1 upper bound of 57.4781%. It is not a ceiling for
  future post-OCR character correction, new candidate generators or filtering.
  This copied-character study cannot repair missing/substituted characters.

The reading-order character diagnostic is 44.8106% original versus 44.8463%
word-only BERT; it includes order, spaces and missing/extra text, not pure
recognizer CER. More lexical matches do not establish uniformly better complete
transcription. Excluding the two known RGB duplicates retains the main pattern:
word-only 22 recovered/2 damaged, 52.3428% to 52.8896% F1 on 298 images.

### Qualification and artifacts

75 focused/regression tests pass. Real-model smoke completed 36 images and
independent audit. Full audit passes 15,750 image/arm counts, 12,600 frozen
policy replays, 7076 original RGB crop hashes, 17,732 candidate character-copy
checks, fold-only priors and 36 exhaustive oracle cases. The supplementary
audit passes 1942 breakpoints and 8400 image/policy checks.

The first smoke stopped at preflight because another GPU process occupied the
required memory; that process was preserved and the failed receipt is retained.
The next smoke and full run completed without actual OOM. The unit OOM is an
intentional failure-propagation check, not a physical OOM.

Actual GPU: RTX3090/24GiB, FP32 without AMP/TF32. OOF training peaks at
6,404,422,144 allocated / 7,197,425,664 reserved bytes; frozen ByT5 inference
3,760,555,008 / 4,305,453,056; BERT 527,370,240 / 3,288,334,336. Read model-specific
receipts for batches and shapes. Stage timings include training/loading/offline
scoring; they are not an end-to-end latency benchmark or 8GiB qualification.

Under the external root preserve `full/results.json`, `full/diagnostics.json`,
`full/analysis.md`, `full/independent_audit.json`, `full/predictions.json`, all
OOF generators/training pairs, `threshold_sensitivity/results.json` and its
independent audit, and both immutable source snapshots. No policy was enabled
in defaults or STRICT and no Git commit was created.

Primary result SHA256:
`6179386a9974300818cb74924fe6c8b013d8416c51b493b6b5399afac2a3b764`.
Supplementary result SHA256:
`f7d0834e04d1896d9b6177cfce886d38e17ee0e465facc413d6273e222c06539`.
