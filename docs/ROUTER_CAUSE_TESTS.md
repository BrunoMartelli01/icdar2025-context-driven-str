# Controlled tests of context-router failure causes

This **approximate reproduction** battery tests the proposed causes of the
completed token-context run. It preserves original expert predictions and strict
configuration. Paper Section 3.5/Figure 4/Algorithm 2 supplies the context/fallback
motivation; these tests are extensions. Decisions and limitations are recorded in
`ROUTER-CAUSE-TESTS-001` in [PAPER_AMBIGUITIES.md](PAPER_AMBIGUITIES.md).

The original run is the completed, independently audited external
`token-context-router-v1/full/e8302d7d6054`. Its TEST F1 is 80.35714% for all
DeepSolo, 78.44408% for the best observed new combination at 50% bypass and
82.32044% for the same-budget GT oracle. The new tests investigate the gap;
starting a test does not confirm its hypothesis.

Those historical values retain punctuation. A completed
[improved-DeepSolo replay](ROUTER_IMPROVED_DEEPSOLO_REPLAY.md) now evaluates the
same frozen routing decisions with both punctuation-retaining and symbol-free
metrics. With the separately selected native-only fallback, symbol-free TEST F1
is82.6540% all DeepSolo (300/300,100% sent to DeepSolo),80.8702% for the historical
combined router at50% bypass (150/300,50% sent to DeepSolo), and84.2983% for its
same-budget GT oracle (150/300,50%). These are logical cached-output decisions.
Preserve the original cause tests;
metric and fallback transfer is not router retraining.

## Predeclared tests

| Hypothesis | Separate tests | What can be concluded |
| --- | --- | --- |
| H1: router overfitting limits generalization | Original/smaller capacity x original/stronger weight decay, three matched seeds; 25/50/100% learning curves; shuffled-label and shuffled-context controls | A reproducible refit/OOF gap establishes poor generalization. Improved OOF/TEST performance under controlled regularization supports a causal contribution; a gap alone does not identify the fix. |
| H2: center cropping loses useful word evidence | Boundary markers through the actual processor; native/pad/stretch/complete windows; correctness AUC on wide/compact crops; prefix/middle/suffix character foils; matched padded/stretched router refits | Geometry proves lost pixels. Only paired held-out changes establish an effect on verification or final routing F1. Padding can also lower character resolution, so multiple interventions matter. |
| H3: missing-word evidence is insufficient | Remove one or half of safely matched correct words, image unchanged; BLIP fixed/recomputed erasure masks; unchanged-word-bag order control; privileged true-deltaTP versus true-deltaPred gates | Known extra FN must increase predicted risk to pass the directional test. Word order has no effect on the primary multiset metric. GT count gates are diagnostic bounds, not usable models. |
| H4: TRAIN/TEST benefit distributions differ | Common-reference prevalence controls; pixel-group bootstrap; count/confidence strata; monotone affine calibration controls | Distribution differences can be verified. Matching can test observed difficulty explanations. Expert exposure remains unresolved without an expert-training counterfactual. One-dimensional monotone recalibration cannot improve fixed-quota ranking. |
| H5: fixed 50% bypass amplifies ranking errors | Curves from 0–50%; exact same-budget GT oracle; zero-utility gate; TRAIN-OOF-selected fraction transferred to TEST | Separates budget pressure from the quality of ranking. TEST-optimal fractions are descriptive; only TRAIN chooses a transferable fraction. |
| H6: weak auxiliary labels contribute | Replay primary and spatial labels; remove auxiliary loss; replace it with lexical care/ignore labels; audit patch-center coverage | The label-alignment and auxiliary-objective effects are tested separately. A small number of contradictory labels does not establish their effect on F1. |

There are 12 configured variants and 20 seed/variant trials. The four capacity /
weight-decay combinations each use three seeds. Other interventions use the
matched original seed. Every full trial uses five pixel-group-disjoint folds and
TRAIN OOF selection among epochs 5/10/20 at 50% bypass. Main final models refit on
all 1,253 usable TRAIN images; learning-curve controls use their declared fraction.
The TEST contains all 300 original images. The two known TRAIN/TEST pixel copies
remain excluded from fitting as in the source run.

The identical first-seed baseline can reuse audited source CV checkpoints after
checking fitting IDs, seed, source hashes and exact neural score replay. Other
models are fitted anew. Context donors come only from the current fitting fold;
held images and identical RGBs cannot donate. Label permutation is fit-only and
paired with an auxiliary-loss-off control.

## Input and diagnostic boundaries

Source NPZ/JSON bytes, RGB crops, validity masks and annotations are pinned.
Crop embedding extraction uses no GT: interventions transform the same original
word pixels. Foils mutate only a predeclared character position. Correctness
labels are attached later for evaluation; literal-foil tests require a locally
compatible correct reading of at least three characters.

Fresh TEST crop arrays are created only after every trial's epoch/model choice
is frozen. Previously available TEST outcomes enter predeclared descriptive
diagnostics, never fitting or epoch selection. This is historically explored
TEST, not an independent confirmatory dataset.

BLIP probes are a **privileged partial-dataset diagnostic**, up to 64 TRAIN and
64 TEST images with at least two safely removable correct care predictions.
Removing a prediction that has a surplus duplicate can leave TP unchanged;
those words are excluded from the omission intervention. Every probe checks the
actual FN increment. All original RGB pixels and expert files remain unchanged.
This eligible subset does not represent images with zero or one correct reading.

The main metric remains exact Unicode-casefold word-multiset micro F1,
punctuation/duplicates retained, native256 care/ignore handling, empty images and
predictions retained. TN=0 accuracy is TP/(TP+FP+FN). It is distinct from official
polygon spotting and the paper's unresolved accuracy definition.

## Execution

Use the explicit approximate scientific and host-path profiles:

```bash
.venvs/context-str-main-py311/bin/python -u -m scripts.run_router_cause_tests \
  --mode approximate \
  --config configs/approximate/router_cause_tests_v1.json \
  --paths configs/approximate/router_cause_tests_local_paths_v1.json \
  --smoke
```

Remove `--smoke` for full execution, or add `--preflight` for input validation.
Resume a sealed run with `--run-root /absolute/run/path`. Every run executes its
own Python source snapshot. Missing inputs, changed hashes, incompatibilities and
inference errors fail explicitly. No environment installation or downloads occur.

Actual hardware is RTX 3090/24 GB. Models use FP32, deterministic math attention,
TF32 disabled, sequential CLIP/BLIP loading, native image/router batch 8 and BLIP
vision batch 1/language batch 2. Record original-forward equivalence, unchanged
foundation state hashes, allocated/reserved VRAM and source-embedding replay.
This is a documented HARDWARE_DEVIATION; no 8 GB qualification is implied.

## Qualification and current status

- 49 focused unit tests passed, including paired-statistics and JSON feature-order
  guards plus actual-processor boundary-marker
  removal/preservation, complete window coverage, duplicate-safe FN changes,
  care/ignore labels, donor/fitting groups and monotone ranking invariance.
- Completed smoke: 20 TRAIN / 6 TEST, all 12 variants, two-fold fitting,
  crop/foil extraction and 2 TRAIN / 2 TEST BLIP omission probes. Audit PASS;
  native CLIP word-embedding replay maximum difference 0.
- Expanded audit checks actual checkpoint fitting IDs/seeds, OOF-selected epoch,
  donor scope and freeze-before-TEST binding. It also passed on the smoke.
- Smoke source: external `router-cause-tests-v1/smoke/cef367054974`.
- Full run: external `router-cause-tests-v1/full/1115c7bd7812`, initially launched
  as PID 1898883. The operational pointer is
  `/extra1/bmartelli/context-str-artifacts/router-cause-tests-v1/current_full_launch.json`.

The full run completed successfully, with all 20 trials, complete TRAIN/TEST crop
extraction, 986 TRAIN / 236 TEST eligible token-omission images, and 64 TRAIN /
64 TEST BLIP probe images. `completion.json` records SUCCESS and the independent
audit records PASS. The audit checks frozen bytes, fitting and donor
scopes, OOF population and independent rational-count replay; it does not perform
an independent neural refit. Other GPU jobs may coexist; recorded times are not
an isolated throughput benchmark.

Portable numerical evidence is saved in
[the completion summary](../provenance/router_cause_tests_20261026_completion.json).
Original external evidence remains under the full run: `results.json`,
`independent_audit.json`, `paired_analysis/analysis.json` and `.md`,
`constant_step_analysis/analysis.json`, `findings.json`, and
`figures/cause_tests.png` / `.pdf`. Source result SHA256:
`c83e2bc6ce9ae428efc16bd66a3257b7cc9f64c3816e85156999c581757fecf3`.
The paired-analysis SHA256 is
`bc3c6bd0bd1dc4b807eba7f4f9dce5e483998a40ed4e115cc238677ddd496839`.

## Automatic paired analysis

After successful experiment completion and independent audit, a separate pinned
analysis snapshot runs on CPU. It reports paired pixel-group and matched-seed
bootstrap intervals for final micro F1 and restores likelihood feature order
explicitly after JSON serialization. The frozen likelihood head is replayed on
CPU against its original GPU scores within an explicit tolerance. Holding the
18 scalar features or the 14 likelihood features fixed separates their response
to diagnostic omissions; nonlinear interactions are reported separately. These
are sensitivity controls, not additive fractions of physical causation.

The analysis uses `configs/approximate/router_cause_analysis_v1.json` and
`scripts/analyze_router_cause_tests.py`. Its snapshot and launch receipt are
saved alongside the run. The analysis smoke also passed; no TEST outcomes select
a model, epoch or threshold.

Learning-curve comparisons hold epochs rather than optimizer updates constant.
The supplementary `scripts/analyze_router_learning_steps.py`, with isolated
`configs/approximate/router_learning_steps_v1.json`, also compares existing OOF
checkpoints at equal or nearly equal optimizer-update budgets. It uses no TEST
outcomes or new fitting. Its smoke and full analyses both completed.
The random-label control uses seeded random permutations in OOF fits and a cyclic
permutation in its final refit; interpret its random-label evidence primarily
on OOF. Both limitations remain explicit in the generated paired report.

Fit-versus-OOF gaps for partial learning-curve models also compare different
image populations. Interpret those models as learning curves; use full-population
variants for direct generalization-gap comparisons.

## Completed findings

All F1 values below concern the complete combined system at **50% image bypass**
unless a different quota is stated. OOF means TRAIN predictions from models that
did not fit those RGB groups. Three-seed rows are means, not a selected best seed;
one-seed interventions are compared only to the matching first-seed baseline.
The historical combined router (78.4441% TEST F1) is distinct from this battery's
token-head baseline.

| Variant | Seeds | TRAIN refit F1 (%) | TRAIN OOF F1 (%) | TEST F1 (%) |
| --- | ---: | ---: | ---: | ---: |
| Original token head | 3 | 80.6468 | 77.9580 | 78.2508 |
| Smaller head | 3 | 80.0821 | 77.7720 | 78.1459 |
| Stronger weight decay | 3 | 80.6503 | 77.9550 | 78.0998 |
| Smaller head + stronger decay | 3 | 80.0732 | 77.7804 | 78.1244 |
| Original token head, matched first seed | 1 | 81.6081 | 77.7735 | 77.9378 |
| Square-padded word crops | 1 | 81.6936 | 77.6285 | 77.3699 |
| Full-word stretched crops | 1 | 81.0949 | 77.8739 | 78.2629 |
| Lexical auxiliary labels | 1 | 81.6263 | 77.7950 | 77.8321 |
| Auxiliary loss disabled | 1 | 81.5742 | 77.4408 | 77.4925 |
| Shuffled scene context | 1 | 80.9256 | 77.2188 | 77.5595 |

**H1 — a generalization gap exists, but the proposed regularization fixes did
not improve held-out F1.** The three-seed mean refit–OOF gap is 2.6888 points.
Smaller heads reduce it to 2.3101 points; smaller heads with stronger decay reduce
it to 2.2928. Their paired TEST F1 differences are respectively −0.1050 points
(95% interval [−0.6941, +0.5544]) and −0.1264 [−0.7453, +0.5352]. Stronger decay
alone gives −0.1510 [−0.7646, +0.1178]. None establishes a positive remedy.
Random-label OOF performance is 3.0158 points below the matched auxiliary-off
control [−4.0669, −1.9760], showing that genuine supervised signal exists.
Shuffled-context OOF loss is 0.5546 points, but its interval crosses zero
[−1.3281, +0.1389]; this does not establish either context necessity or uselessness.

The 25%-TRAIN curve loses 1.1503 OOF points versus full TRAIN at their selected
epochs [−1.9691, −0.4623]; 50% versus full is inconclusive. Under exactly matched
630 updates per fold, full TRAIN exceeds half TRAIN by only 0.0694 OOF points
[−0.6322, +0.7034]. At 1,260 updates the difference is +0.2703
[−0.3572, +0.9460]. Full versus quarter TRAIN at 630 versus 640 updates gives
+0.6642 [−0.0771, +1.4293]. Therefore these tests do not establish that merely
adding more examples of the same distribution resolves the routing problem.

**H2 — cropping damages local evidence; preserving it improves a word-level
signal, without a supported final routing gain.** Actual-processor marker tests
remove both ends under native center cropping and retain both with all three
complete-coverage remedies. On 2,143 care TEST predictions, correctness AUC is
0.6202 native, 0.6495 padded, 0.6804 stretched and 0.6260 pooled windows. Stretched
minus native is +0.0602 [0.0355, 0.0868]; the wide-crop AUC improves from 0.6272
to 0.7162. This held-out word-verification improvement is supported under the
declared labels. Padding and stretching also change character scale/distortion,
so the test does not isolate lost pixels as the only mechanism.

On 1,369 locally compatible correct TEST words of at least three characters,
native CLIP prefers the correct word over a first/last-character alteration in
55.15% / 50.44% of pairs. Padding raises these to 97.15% / 92.51%; stretching to
94.23% / 88.13%. These are synthetic foil preferences, **not OCR accuracy**;
word priors and tokenization also affect them. The stretched router's matched
TEST improvement is only +0.3251 F1 points [−0.5293, +1.1660], while padding gives
−0.5679 [−1.6439, +0.3481]. Better local evidence has not yet produced a reliable
relative-expert ranking improvement.

**H3 — sensitivity to missing correct words is weak, and some responses are
misdirected.** Removing one or half of the safely matched correct predictions,
while preserving every original image pixel, must reduce true utility. In the
236 eligible TEST images, the original token router decreases its score in only
54.66% / 55.93% of cases. Padded crops give 55.93% / 52.12%; smaller heads with
stronger decay give 40.68% / 27.12%. The known true utility drop per omitted
correct word is `2 − reference_F1`, about 1.176, whereas the original token head's
mean predicted drops are only 0.0201 for one word and 0.0016 for half the words.
Synthetic omissions can be out of distribution; these are directional sensitivity
measurements, not natural-error classification accuracy.

The frozen likelihood router reacts correctly in 26.56% of the 64 eligible TEST
images with a fixed erasure mask, or 28.13% when recomputing that mask, for either
omission size. For half-word removal, its mean full score drop is −0.4384
[−0.6459, −0.2157]: utility rises despite known extra FN. Holding scalar features
fixed leaves a likelihood-only drop of +0.0069 [−0.1584, +0.1696]; holding
likelihood fixed gives a scalar-only drop of −0.5398 [−0.6635, −0.3965]. Thus the
head's count/confidence/geometric scalar response is misleading in this probe,
and its BLIP likelihood evidence supplies no clear correction. Reordering the
unchanged word bag changes mean TEST score by +0.1032 although the lexical metric
is invariant. These results concern the tested likelihood features and trained
head, rather than proving a general inability of BLIP to perceive missing text.

**H4 — the relative-expert target distribution shifts; its cause remains
unresolved.** Among eligible images, positive light-versus-DeepSolo utility occurs
in 122/1,204 TRAIN images (10.13%) versus 65/291 TEST images (22.34%). The
difference is +12.2039 points [7.0604, 17.4908]. Matching predicted count gives a
reweighted TRAIN rate of 10.10%; matching count and confidence gives 9.95%, both
versus 22.34% TEST with complete stratum support. Those observables do not explain
the shift. Expert exposure, other difficulty factors and annotation differences
have not been causally separated. Strictly positive affine score transformations
leave fixed-quota rankings and F1 unchanged across all tested arms/splits.

**H5 — the 50% quota magnifies ranking errors but is theoretically feasible on
TEST.** Historical combined-router TEST F1 is 79.6954% / 79.1107% / 78.4441%
at 20% / 30% / 50% bypass; all-DeepSolo gives 80.3571%. The exact same-budget
TEST oracle gives 82.3204% at 50%. On TRAIN the 50% oracle gives 81.7926%, below
all-DeepSolo's 82.4164%. Quotas from 0–50% in five-point steps select zero bypass
on TRAIN OOF for every original arm; this policy transfers to all-DeepSolo on
TEST and saves no fallback calls. The uncalibrated positive-utility gate accepts
51/300 TEST images with the combined router and yields 79.9540%; relaxing the
budget alone does not demonstrate a gain.

Privileged replacement of predicted deltaTP with its true value raises combined
TEST F1 from 78.4441% to 81.4680% at the same 50% quota. Replacing only deltaPred
gives 77.7935%; both true values give the 82.3204% oracle. This points toward
estimating the relative number of correct words as a useful diagnostic priority.
It neither separates recognition from detection errors nor provides a deployable
method or an additive partition of the performance gap.

**H6 — auxiliary-objective mismatch is real but its effect on final F1 is
unproven.** Under the declared word assignment, 79/6,021 TRAIN TPs (1.31%) and
19/1,496 TEST TPs (1.27%) have negative spatial auxiliary labels. These assignment
counts differ from the historical minimum-count diagnostic and should not be
conflated. Lexical auxiliary labels give −0.1057 matched TEST points
[−0.7825, +0.5406]; disabling auxiliary loss gives −0.4453
[−1.5785, +0.5724]. Neither shows a supported improvement. Also, 678/9,268 TRAIN
and 151/2,204 TEST GT boxes contain no valid patch center. This establishes a
weakness in center-based spatial supervision, not absence of text pixels from
CLIP's receptive fields.

## Priorities supported by these tests

1. Target missing-word coverage and relative correct-word gain explicitly. A
   possible next experiment is supervised omission consistency with matched
   removals of incorrect words, keeping evaluation on natural unmodified images.
   This is a proposal, not an improvement established by the current battery.
2. Preserve the complete word crop when measuring literal evidence. Stretched
   crops currently improve verification AUC; retain a paired routing control
   because the end-to-end benefit remains uncertain.
3. Validate relative-expert targets on genuinely expert-unseen data before
   attributing the TRAIN/TEST shift to expert exposure or collecting more similar
   TRAIN data. Keep compute-saving quotas explicit in the optimization objective.

Larger regularization sweeps, simple score offsets and auxiliary-label repair
have less evidence of benefit than these priorities. No new tested router
surpasses the fixed historical combination or all-DeepSolo under this protocol.
