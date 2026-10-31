# Image-wide OCR correction: completed results

**Approximate reproduction / exploratory transcription research. Completed and audited on 2026-10-27.**

The new image-wide selector improves TEST F1 from 52.5855% to 53.3861%, but does not demonstrate that the relevant image context caused the gain. A matched wrong-image control is at least as effective. The largest measured improvement comes from deleting low-quality outputs: 60.5834% with 24 correct words lost, or 60.3539% with 9 lost. Expanded correction plus filtering reaches 61.2986% with 24 lost, or 61.0589% with 9 lost. These are frozen CAL-selected profiles, not independent validation or an 80% result.

Every reported profile uses saved PARSeq readings. DeepSolo receives 0/300 TEST images (0.00%); no new crops, recognition calls or online latency are measured. U-Net, original boxes and recognizer weights remain frozen.

## Scope and protocol

The battery contains 21 primary settings (15 learned selectors, 3 matched generator recipes, 3 filters), 8 joint configurations and 2 legacy-feature controls: 123 saved policies per evaluated role, including references and declared objective/damage-cap variants. They are correlated comparisons, not 123 independent experiments. ByT5 uses three image-separated OOF models plus one final model per recipe: 12 generator stages. All use 6000 training records and 750 updates. Four unchanged historical controls are reused after exact data/AST/runtime/hash checks; eight corrected-recipe stages are trained anew.

The Total-Text populations are 1030 FIT, 100 CAL, 125 EVAL and 300 TEST images, with 2204 TEST care words. The inherited source is the experimental Attention-Gated ResNet-50 Text+Kernel M/K U-Net control, not a newly trained segmentation model. Candidate generation, FIT supervision and CAL selection are separated from evaluation labels in this run. Thresholds are frozen before EVAL/TEST label evaluation. Nevertheless, these cohorts were explored historically, the source U-Net had calibration exposure, and two TEST images duplicate TRAIN RGB. No independent-generalization or strict paper-reproduction claim is supported.

Primary metric: micro per-image casefold whitespace-token multiset. Punctuation, duplicates and empty observations remain; all care GT words count. Matching has no IoU requirement and ignored regions do not neutralize predictions. A missing output contributes a false negative; an empty original observation remains in the prediction denominator until explicitly filtered. This lexical protocol is not directly comparable with spatial spotting, symbol-free F1 or paper recognition accuracy.

## Frozen correction profiles

The primary predeclared comparison is global-context boundary selection against the same-pool word-only selector. Character and fragment arms retain boundary alternatives and add their respective new proposal types.

| Profile | EVAL F1 | TEST F1 | TEST recovered / lost | TEST FP | DeepSolo EVAL; TEST |
| --- | ---: | ---: | ---: | ---: | --- |
| Original OCR | 50.9990% | 52.5855% | 0 / 0 | 1502 | 0/125 (0.00%); 0/300 (0.00%) |
| Historical ByT5 boundary | 51.7082% | 53.0280% | 73 / 29 | 1582 | 0/125 (0.00%); 0/300 (0.00%) |
| Previous word-LM reference | 51.3208% | 53.1275% | 22 / 2 | 1506 | 0/125 (0.00%); 0/300 (0.00%) |
| New same-pool word-only | 51.0156% | 53.0345% | 15 / 0 | 1501 | 0/125 (0.00%); 0/300 (0.00%) |
| Local OCR context | 51.7857% | 53.4966% | 34 / 2 | 1504 | 0/125 (0.00%); 0/300 (0.00%) |
| Whole-image ordered OCR | 51.4205% | 53.0556% | 15 / 0 | 1499 | 0/125 (0.00%); 0/300 (0.00%) |
| Whole-image inventory | 51.5123% | 53.1460% | 23 / 2 | 1507 | 0/125 (0.00%); 0/300 (0.00%) |
| Combined image context: primary | 51.6281% | 53.3861% | 26 / 0 | 1498 | 0/125 (0.00%); 0/300 (0.00%) |
| Wrong-image inventory control | 51.9981% | 53.4439% | 35 / 3 | 1509 | 0/125 (0.00%); 0/300 (0.00%) |
| Class-balanced global selector | 51.9321% | 53.5312% | 32 / 1 | 1498 | 0/125 (0.00%); 0/300 (0.00%) |
| Conservative aligned labels | 52.3135% | 53.4971% | 29 / 1 | 1493 | 0/125 (0.00%); 0/300 (0.00%) |
| Listwise selector with KEEP | 51.8448% | 53.3281% | 24 / 0 | 1498 | 0/125 (0.00%); 0/300 (0.00%) |
| Character proposals, word-only | 51.5551% | 53.1089% | 20 / 1 | 1505 | 0/125 (0.00%); 0/300 (0.00%) |
| Character proposals, global | 52.6119% | 53.3700% | 40 / 4 | 1527 | 0/125 (0.00%); 0/300 (0.00%) |
| Fragment proposals, word-only | 51.4313% | 53.7077% | 36 / 0 | 1495 | 0/125 (0.00%); 0/300 (0.00%) |
| Fragment proposals, global | 52.3899% | 53.9933% | 47 / 0 | 1498 | 0/125 (0.00%); 0/300 (0.00%) |
| All proposal types, global | 52.3521% | 53.8522% | 49 / 1 | 1514 | 0/125 (0.00%); 0/300 (0.00%) |
| All proposals, aligned labels | 51.9811% | 53.6266% | 32 / 1 | 1489 | 0/125 (0.00%); 0/300 (0.00%) |

Combined global minus word-only: +0.3516 percentage points, paired conditional 95% interval [-0.0120, +0.7546]. Local minus word-only: +0.4621 points, [+0.0652, +0.9289]. Matched inventory minus wrong-image inventory: -0.2978 points, [-0.7817, +0.1044] on TEST and -0.4858 points on EVAL. The latter comparison has identical feature dimensions, labels, pool and HGB capacity. These results provide no positive evidence for image-specific global information in this implementation. They do not prove that all context-based correction is ineffective.

A separate predeclared control adds context to the exact earlier word-LM features, preserving internal cut-casing cues. It reproduces all 525 earlier cap2 image predictions exactly. Context raises EVAL F1 from 51.3208% to 52.2556% and TEST from 53.1275% to 53.4544% (+0.3269 points; conditional 95% interval [+0.0162, +0.6850]). DeepSolo: 0/125 EVAL (0.00%), 0/300 TEST (0.00%) for each profile. This is a small positive feature-addition signal; without a legacy matched wrong-context control it cannot isolate semantic relevance. None of these intervals adjusts for the many explored comparisons, historical reuse or training-seed variation.

## Why the correction gains remain small

1. **Insufficient candidate coverage.** The exact GT-assisted pool oracle reaches 1445 correct words with boundary alternatives and 1474 with all proposals, versus 1322 originally. Therefore 730 of 2204 care words remain unrecovered even with ideal selection in the tested pool. The new character and fragment proposals together add only 29 correct-word slots beyond the boundary pool. This is a limitation of this bounded generator/pool, not a universal upper bound for text-only correction.

2. **Strong rejection of useful proposals.** In 79 TEST parent groups containing a beneficial boundary proposal, the combined global model ranks a beneficial option first in 68, yet accepts only 15 and rejects 53. In the expanded pool, 118 groups contain a useful proposal; 99 rank one first, 30 accept it and 69 reject it. The 19 remaining expanded groups have a ranking error. Expanded overlap conflicts also contribute to rejection. Accepting every positive-utility proposal is not a remedy: the fixed threshold 0 profile recovers 70 words but loses 5 and creates 1571 FP, yielding 53.7389% F1, below the selected expanded 53.8522%.

3. **Low positive prevalence and imperfect supervision.** Only 479/13178 FIT boundary proposals and 655/25540 expanded proposals are beneficial. TEST contains 138 beneficial boundary, 20 character-only and 24 two-parent proposals among 7019 alternatives. The metric labels partial successes as beneficial: `AUTHENTICRAKERY → AUTHENTIC RAKERY` recovers `AUTHENTIC` while leaving the incorrect `RAKERY`. Class balancing and clean aligned labels make small changes, without a supported improvement over the main global model. Listwise AP rises from 0.370 to 0.518 but F1 falls slightly to 53.3281%; better ranking of all candidates does not guarantee a useful calibrated acceptance decision.

4. **Context availability is not context reliability.** Among 79 useful boundary groups, 33 have horizontal local context and 78 have other OCR. Across 988 other-observation occurrences, only 422 exactly match care GT strings. These are repeated group-conditioned occurrences, not 988 unique words or a semantic-quality score. Saved examples contain fragments, symbols, several languages and words from unrelated signs. The current stable-order windows pool BERT likelihoods; they do not learn which sign/line or relation supplies evidence for a particular boundary. The wrong-image control is consistent with lexical-prior effects or poor use of relevant context; it does not uniquely identify the cause.

5. **Annotation conventions are not purely linguistic.** `KrispyKreme` remains unchanged even though a useful split exists. `HolidayInn` is retained by the combined model, but the inventory model splits it and loses a GT match. `SANFRANCISCO → SAN FRANCISCO` succeeds, while `TOURISTVEHICLE` is corrected by word-only/expanded selection and rejected by combined boundary context. `MULTIGRAINFLAKES` becomes `MULTI GRAIN FLAKES` in the primary model and `MULTI GRAINFLAKES` in the expanded model; the former recovers only `FLAKES` under the original annotations. Semantic plausibility does not establish the annotation unit, exact letters or all correct boundaries. These named examples are post-hoc illustrations, not population estimates.

## Matched generator recipes

| Raw single-block generator | Recovered | Lost | TEST F1 | DeepSolo TEST |
| --- | ---: | ---: | ---: | --- |
| historical | 46 | 15 | 53.2677% | 0/300 (0.00%) |
| random | 27 | 12 | 52.8354% | 0/300 (0.00%) |
| empirical | 10 | 7 | 52.6107% | 0/300 (0.00%) |

Neither cleaner alignment plus random noise nor empirical character confusions improves on the historical recipe. In fold 0 the aligned recipes retain 2870 real pairs, of which 2604 are identities, 205 character errors and 61 boundary errors. The historical recipe has 672 real identities, 841 character errors and 15 real boundary errors, plus its different synthetic mixture. Cleaner recipes consequently contain far more identity examples and fewer difficult real corrections. This is a measured data-composition difference and a plausible explanation for conservatism, not a controlled causal proof: alignment, mixture, targets and noise change together. Lower training loss would not establish better correction. OOF models also see fewer images than final models despite the same record/update budget.

## False positives and joint composition

| Frozen profile | Correct / outputs | FP | Recovered / lost | TEST F1 | DeepSolo TEST |
| --- | ---: | ---: | ---: | ---: | --- |
| Exact-read filter | 1298 / 2081 | 783 | 0 / 24 | 60.5834% | 0/300 (0.00%) |
| Care-coverage-aware filter | 1313 / 2147 | 834 | 0 / 9 | 60.3539% | 0/300 (0.00%) |
| Overlap-duplicate control | 1322 / 2824 | 1502 | 0 / 0 | 52.5855% | 0/300 (0.00%) |
| Primary joint supplement | 1315 / 2148 | 833 | 2 / 9 | 60.4320% | 0/300 (0.00%) |
| Expanded + care-coverage filter | 1332 / 2159 | 827 | 19 / 9 | 61.0589% | 0/300 (0.00%) |
| Expanded + exact-read filter | 1317 / 2093 | 776 | 19 / 24 | 61.2986% | 0/300 (0.00%) |

The exact-read filter removes 743 outputs: 719 FP and 24 correct words. The care-coverage-aware filter removes 677: 668 FP and 9 correct words. Annotation proxies attribute 585/743 exact-filter removals and 652/677 coverage-aware removals to insufficient care-word coverage. Other deletions include incorrect readings of real text, ignored/unannotated text and partial detections. These counts do not prove that a removed output was background. The overlap-duplicate control selects KEEP on CAL and makes no TEST change.

The calibration cap2 is a constraint on CAL, not a TEST damage guarantee: both standalone filters lose only 1 correct CAL word, but 24 and 9 on TEST. Baseline CAL precision is 63.05%, versus 46.81% on TEST; different error prevalence/composition makes the same acceptance policy less safe. This observed shift does not isolate a single distribution-shift cause.

Joint selection is performed again on CAL, rather than adding isolated gains. It selects correction threshold 0.8 for the primary boundary supplement and 0.65 for expanded policies, stricter than isolated correction thresholds 0.5 and 0.2. Expanded joint policies recover 19 words versus 49 for isolated expanded correction. The two composition orders coincide on CAL and TEST. On EVAL they differ for one image (`total_text:802`) in the expanded/exact/F1-cap2 profile: correct-first gives 59.5186% F1 versus 59.3647% for filter-first. Thus the orders are not generally equivalent. The safer expanded profile has 1332 correct words, 10 above the original baseline; the higher-F1 exact profile has 1317, 5 below baseline. Its F1 improvement is mostly removal of incorrect outputs.

Post-hoc `expanded_global__exact__correct_first__f1_cap2_versus_filter_alone`: F1 change +0.7151 points, conditional 95% interval [+0.2639, +1.2491]. DeepSolo for both profiles: 0/300 TEST (0.00%).
Post-hoc `expanded_global__presence__correct_first__f1_cap2_versus_filter_alone`: F1 change +0.7050 points, conditional 95% interval [+0.2603, +1.2262]. DeepSolo for both profiles: 0/300 TEST (0.00%).

The transcript-level character diagnostic confirms the tradeoff. Reading-order CER is 44.8106% originally, 44.8320% after primary boundary correction and 45.0465% after expanded correction. The coverage-aware filter gives 34.4246%, versus 34.4818% when expanded correction is added; the exact filter gives 36.8835%, versus 36.9335% with correction. DeepSolo for each: 0/300 TEST (0.00%). This diagnostic includes spaces, missing/extra text and reading order; it is not isolated recognizer CER. Word F1 gains therefore do not establish improved whole-transcript character fidelity.

## What the 80% target requires

With G=2204 care words, `F1 = 2*TP/(TP+FP+G)`. The exact boundary-pool maximum TP 1445 implies an optimistic perfect-precision F1 ceiling 79.1998%, even with free deletion of every unmatched word. The expanded maximum TP 1474 raises this loose ceiling to 80.1523%. Achieving 80% inside that pool would require at least 1470 correct words; even at the maximum 1474, at most 7 FP could remain. The current expanded policy has 1370 TP and 1514 FP. Thus threshold tuning alone is an implausible route to 80%: almost the entire oracle gain and virtually perfect FP deletion would be needed. This is a bound on these proposals only; stronger text reconstruction could expand the pool without changing U-Net.

The most justified next experiments, without changing image crops, are:

- **Expand useful textual proposals first.** Train on actual high-coverage OCR errors with controlled identity/error proportions; compare noise recipes with identical source/target records and matched updates. Preserve separate boundary and letter-edit channels, and evaluate candidate recall before spending time on a more complex selector. The present empirical-noise recipe supplies too few useful new character proposals to support a large gain.
- **Learn contextual relations and acceptance separately.** Use original line/sign groups and a reliability-weighted selection of other words, keeping original OCR as the context. Train correct candidate versus KEEP using hard false splits, proper names and alternative annotation units. Compare against the same-dimension wrong-image and shuffled-relation controls. The current listwise test shows that ranking AP alone is insufficient; explicitly measure accepted useful groups, false splits and rejection curves.
- **Separate exact-read quality from care-coverage rejection.** Use the coverage-aware filter as the measured lower-damage reference. Calibrate expected correct-word loss on image-separated TRAIN OOF data, with explicit uncertainty/reliability bins, before freezing one operating point. This is a proposed risk-control experiment, not a guarantee that CAL cap2 transfers to TEST.
- **Validate on expert-unseen images and transcription conventions.** Collect or reserve independent images with consistent word-unit annotations; retain originals and report punctuation/ignore/order rules. Repeat a small number of predeclared contrasts across seeds. Current reused cohorts and conditional intervals cannot establish an 80% deployable system.

## Qualification and preserved failures

All 105 focused/regression tests pass. The primary independent audit checks 72000 training records, 12 generator stages, 38007 context groups, 39959 windows, 38040 candidate operations, 45675 image/arm counts, 39375 selector replays, 3150 filter replays and 1575 generator replays. Joint count replay passes 16800 image/policy cases; the legacy control reproduces 525 earlier predictions exactly. The portable-summary audit independently recounts all 2100 legacy image/policy cases. A separate final audit validates all three immutable source manifests (1762, 1762, 1764 files), 32 pinned input references, every newly trained record digest and all 106 primary/joint/legacy artifact files.

The first two smokes stopped at occupied-memory preflight. Smoke 3 completes 42 images and passes 1044 image/arm count checks. The original full run was stopped before efficacy evaluation after detecting 5987/5991 nonempty records instead of 6000. It is invalid for matched-recipe comparison and remains preserved. Empty records are now rejected before counting and replenished; impossible exact budgets fail explicitly. An initial shortfall regression failed because of error-order wording, then passed after correction. Four unaffected historical controls were reused only after full regeneration/checksum qualification. Actual execution always used the qualified Python 3.11 environment; the earlier system-Python metadata probe is retained separately as superseded. The first external summary-script invocation needed an explicit PYTHONPATH. Its next invocation rejected an overbroad assertion that the composition orders always coincide; exact comparison found the single EVAL difference reported above, so the summary now stores per-role differences. Both reporting failures are preserved in `report_generation_failures.json`; no model, threshold or prediction changed.

Actual hardware: RTX 3090/24 GiB, CUDA 12.4, FP32, no AMP/TF32, deterministic settings, sequential model loading. Across generators, peak allocated 5034582528 bytes and reserved 6268387328 bytes; BERT peak allocated 1134137856 bytes and reserved 4286578688 bytes, batch 48 and maximum observed 76 input tokens. ByT5 batch 8, with per-stage shapes in receipts. No actual model OOM occurred; injected OOM tests confirm propagation. The four unused BERT pooler/NSP keys are explicitly allowlisted; no missing/mismatched learned tensors are tolerated. Strict checkpoint reload qualification verifies 172 ByT5 tensors bit-exactly. Concurrent GPU jobs make timing unsuitable for isolated throughput claims. These runs do not qualify RTX 3070/8 GiB.

Removing the two known duplicate TEST images gives 298 images: original F1=52.3428%, primary global=53.1499%, expanded=53.6209%, exact filter=60.3951%, coverage-aware filter=60.1344%. DeepSolo for each: 0/298 (0.00%). The directions persist, but historical exploration and source calibration exposure remain.

No STRICT/default configuration or scientific source expert is changed. No tested winner is promoted, no package/model installation occurs, and no commit is created.

## Evidence and rerun entry points

- Complete experiment root: `/extra1/bmartelli/context-str-runs/image-context-correction-battery-20261025-v1/run_v2`.
- `full/results.json`, `full/predictions.json`, `full/selection_frozen.json`, `full/diagnostics.json` and `full/failure_diagnosis.json`: primary outputs, thresholds, failures and named cases.
- `joint/` and `legacy_features/`: separately frozen supplementary results and audits.
- `full/character_diagnostic.json`, `supplementary_contrasts.json`, `final_audit.json`, `portable_audit.json` and `artifact_manifest.json`: supplemental metrics and hash-bound qualification.
- Repository portable evidence: `provenance/IMAGE_CONTEXT_CORRECTION_V1.json`; guide: `docs/IMAGE_CONTEXT_CORRECTION.md`.
- Run implementation and analysis from the archived source snapshots, with their explicit configurations and the recorded existing environment. External datasets/checkpoints are not part of the source repository. No source-only check substitutes for verifying those original bytes.
