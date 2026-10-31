# Exact budget feasibility of existing ICDAR experts

**Approximate reproduction / privileged GT diagnostic.** User targets: IC13 F1>=90% with at most61/233 DeepSolo images (26.18%); IC15 F1>=79% with at most228/500 (45.60%). Metric, full care populations and every expert output remain unchanged. These values are upper bounds using TEST GT, not trained or deployable gates.

The current nonempty-light guard is reported separately from unrestricted choice between the same whole-image expert outputs. Removing this guard does not add detections or change the metric. Exact requested counts, best at-most-budget values and minimum target-feasible counts are independently certified with rational arithmetic.

| Dataset | Expert | Eligible choices | Ideal F1 at requested count % | DeepSolo images/total (%) | Minimum DeepSolo for target |
| --- | --- | --- | ---: | --- | --- |
| icdar13 | unet | current_guard | 87.781955 | 61/233 (26.18%) | 82/233 (35.19%) |
| icdar13 | unet | unrestricted | 88.521493 | 61/233 (26.18%) | 77/233 (33.05%) |
| icdar13 | yolo | current_guard | 90.004649 | 61/233 (26.18%) | 61/233 (26.18%) |
| icdar13 | yolo | unrestricted | 90.358640 | 61/233 (26.18%) | 55/233 (23.61%) |
| icdar2015 | unet | current_guard | 81.512170 | 228/500 (45.60%) | 181/500 (36.20%) |
| icdar2015 | unet | unrestricted | 81.968912 | 228/500 (45.60%) | 167/500 (33.40%) |
| icdar2015 | yolo | current_guard | 81.665015 | 228/500 (45.60%) | 167/500 (33.40%) |
| icdar2015 | yolo | unrestricted | 81.908549 | 228/500 (45.60%) | 157/500 (31.40%) |
| icdar13 | unet_native_matched | current_guard | 87.145558 | 61/233 (26.18%) | 86/233 (36.91%) |
| icdar13 | unet_native_matched | unrestricted | 88.263359 | 61/233 (26.18%) | 79/233 (33.91%) |

IC13: current U-Nets cannot meet90% at61 calls even with privileged unrestricted image decisions. Current YOLO reaches90.004649% under its existing guard, leaving virtually no ranking/error margin. IC15: both long768 and YOLO can exceed79% at228 calls under their existing guards; the current learned ranking is the limiting component.

Source classifications: PAPER_SPECIFIED Algorithm2 fallback motivation only; REPOSITORY_SPECIFIED existing fixed experts, lexical counts and oracle primitives; ASSUMPTION user budgets/targets and unrestricted guard control; HARDWARE_DEVIATION CPU cached diagnostic; UNRESOLVED learnability, generalization and official paper metric equivalence. No new neural inference or TEST-based operational threshold selection.

Original per-image evidence, selected IDs and exact target-residual curves are preserved under `/extra1/bmartelli/context-str-runs/budget-target-router-20261029-v1`. Test: `tests/recognition/test_budget_target_oracle.py` compares exact and at-most optima plus minimum target counts against exhaustive small-population enumeration. Independently certify every returned optimum at its own rational F1 and verify the maximum residual over all allowed cardinalities is zero.

## Trained extensions at the requested budgets

**Approximate reproduction, complete official TEST populations, lexical multiset micro-F1.** Casefold each string and retain Unicode alphanumeric characters. Count every care GT word, every care prediction and every image, including empty and punctuation-only cases. Native256 polygon ignore with care override is unchanged. The router acts on complete images. The DeepSolo column is a *logical cached-output choice*, not a measured online invocation or latency. Exact-budget ranking uses all test-image scores together to send precisely the requested number; it is a transductive budget policy. Selection, model choice and corrections were frozen using TRAIN selection/CAL or TRAIN OOF before each variant's TEST replay. This sequence was exploratory after prior TEST exposure, so it is not an untouched holdout or a strict reproduction.

The original direct light predictions and lexicon-corrected light predictions are different experts. The lexical extension deliberately applies Generic/Weak correction to PARSeq words as well as using the already qualified Generic/Weak DeepSolo fallback. In ICDAR2015, the Weak light-training vocabulary is derived from TRAIN care words, whereas the released Weak TEST dictionary is fixed; the shift is explicit.

| Dataset and light expert | Frozen method | F1 % | DeepSolo images/total (%) | Ideal F1 % at same count |
| --- | --- | ---: | --- | ---: |
| IC13 TT U-Net/PARSeq | Lexical correction and OCR/lexical gate | 82.3416 | 61/233 (26.18%) | 88.8469 |
| IC13 YOLO640 | Lexical correction and OCR/lexical gate | 85.4177 | 61/233 (26.18%) | 90.3586 |
| IC13 YOLO640 | TRAIN-only Generic teacher | 86.5019 | 61/233 (26.18%) | 91.1903 |
| IC13 YOLO640 | Recovered low-floor PARSeq slots, selected admission .6 | 86.6381 | 61/233 (26.18%) | 91.2150 |
| IC13 YOLO640 | Five-fold OOF word-risk descriptor | 86.9814 | 61/233 (26.18%) | 91.2150 |
| IC13 YOLO at 640/960/1280 | Validation-selected canonical-word scale fusion | 86.4787 | 61/233 (26.18%) | 90.1924 |
| **IC13 YOLO640 + TT U-Net complement** | **Validation-selected .99 confidence, IoU < .1; OOF word-risk gate** | **87.7275** | **61/233 (26.18%)** | **91.8744** |
| IC13 YOLO640 + TT U-Net complement | Five-fold RGB-group OOF model choice; final fit on all 229 TRAIN | 87.5408 | 61/233 (26.18%) | 91.8744 |
| IC13 YOLO640 + TT U-Net complement | Same full-TRAIN fit without upstream word-risk features | 87.2439 | 61/233 (26.18%) | 91.8744 |
| IC15 long768 U-Net/PARSeq | Lexical correction and OCR/lexical gate | 78.9333 | 228/500 (45.60%) | 84.5533 |
| **IC15 YOLO/PARSeq** | **Lexical correction and OCR/lexical gate** | **79.9490** | **228/500 (45.60%)** | **84.7364** |

The IC15 goal of at least 79% at 228/500 images (45.60%) is met by the fixed YOLO router: TP1567, FP276, FN510 on all 2077 care words. The corrected light branch alone is 72.9945% with DeepSolo 0/500 (0.00%); the fixed DeepSolo branch alone is 82.7905% with DeepSolo 500/500 (100.00%). The nominal paper number is not a verified same-metric, same-protocol comparison. The IC15 U-Net router misses the 79% target by 0.0667 points.

The best trained IC13 gate reaches TP940, FP108, FN155 on all 1095 care words. Its corrected mixed light branch alone is 84.6797% with DeepSolo 0/233 (0.00%); the fixed DeepSolo branch alone is 90.1707% with DeepSolo 233/233 (100.00%). The **90% at 61/233 images (26.18%) target is not achieved**. Its privileged same-output oracle reaches 91.8744% with DeepSolo 61/233 (26.18%), so feasibility does not imply a learned selector can locate those images. No one of the 28 heads trained on the original FIT183 attained 90% on TEST with DeepSolo 61/233 (26.18%); the best *TEST-privileged* choice among them was 88.2765% with DeepSolo 61/233 (26.18%), and is diagnostic only. The five-fold head-selection score on all TRAIN229 was 92.4081% with logical DeepSolo 60/229 (26.20%), but its final 229-image fit scored 87.5408% on TEST with DeepSolo 61/233 (26.18%). That internal OOF score is **conditional and optimistic**: upstream word-risk features can carry labels across head folds, the shared TRAIN vocabulary contains fold-heldout words, and the light complement was selected on selection23. Removing word-risk features gives a distinct conditional OOF score of 92.5728% with DeepSolo 60/229 (26.20%), then 87.2439% on TEST with DeepSolo 61/233 (26.18%). Neither internal OOF score is an unbiased end-to-end generalization estimate.

The failure has a measurable train/test component: the corrected mixed light branch scores 88.8415% on FIT183 with DeepSolo 0/183 (0.00%), 90.2256% on selection23 with DeepSolo 0/23 (0.00%), 90.9091% on CAL23 with DeepSolo 0/23 (0.00%), and only 84.6797% on TEST233 with DeepSolo 0/233 (0.00%). The fixed teacher scores 90.7502%, 90.2256%, 92.5532%, and 90.1707% on those same populations respectively, with DeepSolo 183/183 (100.00%), 23/23 (100.00%), 23/23 (100.00%), and 233/233 (100.00%). This is evidence of a light-expert distribution shift, not proof of a single causal mechanism. A CAL-selected scalar cut for the best mixed gate scored 89.7388% while sending DeepSolo 147/233 images (63.09%); it does not satisfy the requested budget.

## Cost, provenance and limits

The minimal IC15 target-meeting gate reads already available YOLO/PARSeq outputs and dictionary evidence; it adds no CLIP or BLIP encoder. The IC13 word-risk version adds a small CPU classifier and twelve aggregate scalars computed from existing low-floor PARSeq hypotheses. The mixed IC13 branch also requires TT U-Net inference, so its light cost is greater than YOLO alone; its better F1 is not a free router improvement. The three-resolution control requires three YOLO/PARSeq views and performs worse.

All fallback neural outputs in this study were cached and authenticated from earlier experiments; **zero new DeepSolo neural inferences** were run here. The recovered low-floor study generated genuine PARSeq outputs on 229 TRAIN images and reused 233 TEST outputs. The resolution study generated 1,386 image-resolution YOLO detector outputs and 1,386 PARSeq reader outputs over 462 images and three views. These are offline research costs, not online timing measurements. On the observed RTX3090, FP32/batch1 detector and FP32/batch32 PARSeq were run sequentially; the maximum per-image PyTorch allocated/reserved peaks were 230,706,176/327,155,712 bytes for a 1280 detector view and 206,086,656/285,212,672 bytes for PARSeq. The saved forward shapes reach [1,3,1280,1280] for YOLO and [32,3,32,128] for PARSeq. These peaks exclude unrelated processes and do not qualify an 8 GB joint deployment. No OOM or inference failure occurred in the recorded runs.

Source classifications: `PAPER_SPECIFIED` only for the general DeepSolo fallback rationale in Section 3.5/Algorithm 2. `REPOSITORY_SPECIFIED` for the source OCR rows, qualified lexicon matcher, native256 ignore, exact GT-free feature extraction and previously frozen fallback profiles. `ASSUMPTION` for applying the lexicon to light predictions, TRAIN-only Generic construction, recovered-slot admission, word-risk set-membership labels, scale consensus, the U-Net novelty rule, supervised utility heads, conditional head-OOF splits and user budgets. `HARDWARE_DEVIATION` for sequential FP32 cached experiments on the observed RTX3090. `UNRESOLVED` for paper-equivalent metric/checkpoints, prior TEST exploration, the small validation sizes, upstream feature/vocabulary dependencies across the head folds and whether another light expert can generalize sufficiently to meet the IC13 target.

All raw per-image counts, frozen choices, TEST scores, model files, OOF evidence, source hashes and independent audit receipts are under `/extra1/bmartelli/context-str-runs/budget-target-router-20261029-v1`. New source/config entrypoints are `scripts/run_lexical_budget_router.py` with `configs/approximate/lexical_budget_target_router_v1.json`, `scripts/prepare_ic13_word_risk.py` with `configs/approximate/ic13_word_risk_budget_v1.json`, `scripts/prepare_ic13_complement.py` with `configs/approximate/ic13_complement_v1.json`, and `scripts/train_ic13_fulltrain_oof.py` with `configs/approximate/ic13_complement_fulltrain_oof_v1.json` or the no-word-risk control `configs/approximate/ic13_complement_fulltrain_no_wordrisk_v1.json`. The independent checks cover every corrected observation, all 28 model trials per case, the exact TEST route, 229 authenticated TRAIN teacher envelopes, five disjoint word-risk folds, 17 complement validation policies and 462 staged images, and two five-fold conditional all-TRAIN head studies.
