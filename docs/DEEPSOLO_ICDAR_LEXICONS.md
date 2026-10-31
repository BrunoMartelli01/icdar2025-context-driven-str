# DeepSolo with released ICDAR lexicons

Date: 2026-10-28. **Approximate reproduction.** User requests vocabulary-assisted ICDAR2015 DeepSolo and at least 90% on ICDAR2013 using only DeepSolo. The full, unchanged symbol-free lexical micro F1 is used throughout. No U-Net, YOLO, PARSeq, new neural weights, GT-dependent correction or new GPU inference is used.

## Main outcomes

**ICDAR2013 reaches 90.1707% F1 on all 233 images / 1095 care words with DeepSolo plus the dataset-specific released Generic vocabulary.** This is the TRAIN-calibration-selected policy: the existing Total-Text ViTAEv2-S checkpoint, detector threshold0.40, no character-confidence rejection, and keeping unmatched words. TP977,FP95,FN118,precision91.1381%,recall89.2237%; DeepSolo233/233 images (100.00%). Compared with the prior best native standalone88.4956%, DeepSolo233/233 (100.00%), the gain is1.6752 points, with conditional paired95% interval[0.4912,2.8562] points.

For ICDAR2015, applying the original source lexical rule to the frozen75.2753% expert raises F1 to79.6764%, both DeepSolo500/500 (100.00%). The separate calibration-selected keep-unmatched policy reaches78.9790%, DeepSolo500/500 (100.00%). Its lexical confidence cut remains the preceding calibration-selected0.35/0.85. The source and calibration choices are separate presets, rather than selecting a winner using TEST.

## ICDAR2015: all fixed settings

| Neural filtering preset | Vocabulary / unmatched policy | F1 % | TP | FP | FN | DeepSolo images/total (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
|prior_calibrated|None|75.2753|1504|415|573|500/500 (100.00%)|
|prior_calibrated|Generic_source_reject|79.6764|1576|303|501|500/500 (100.00%)|
|prior_calibrated|Generic_keep_unmatched|78.9790|1578|341|499|500/500 (100.00%)|
|prior_calibrated|Weak_source_reject|82.8968|1614|203|463|500/500 (100.00%)|
|prior_calibrated|Strong_source_reject|87.1156|1643|52|434|500/500 (100.00%)|
|upstream_source|None|71.9286|1572|722|505|500/500 (100.00%)|
|upstream_source|Generic_source_reject|77.6658|1657|533|420|500/500 (100.00%)|
|upstream_source|Generic_keep_unmatched|75.9094|1659|635|418|500/500 (100.00%)|
|upstream_source|Weak_source_reject|82.7720|1708|342|369|500/500 (100.00%)|
|upstream_source|Strong_source_reject|88.9908|1746|101|331|500/500 (100.00%)|

`prior_calibrated` is score0.35 / character confidence0.85; `upstream_source` is score0.30 / no character-confidence pruning. Both use the strict-loaded official IC15 ViTAEv2-S+TextOCR checkpoint c0aac6fddc297ca1a7e3c2d0bbd5f7f8b2f55214795c7fa32f13fb51c9db95a8, native shortest1440/max4000, and the same cached raw detections as the preceding study. With the same prior-calibrated filter, the original lexical rule gives Generic79.6764%,Weak82.8968%,Strong87.1156%, each DeepSolo500/500 (100.00%). The fixed upstream neural filter plus Strong gives88.9908%, DeepSolo500/500 (100.00%). These are predeclared settings, not TEST-tuned thresholds.

Generic uses the released broad dictionary; Weak contains dataset-specific word candidates and Strong contains image-specific candidates. Weak/Strong results are benchmark-assisted settings, unsuitable as an assumption that an arbitrary new image supplies its candidate words. Generic itself must also be identified by the exact dataset release and file hashes, rather than treated as an interchangeable English dictionary.

## ICDAR2013: correct dataset resources

| Neural filtering preset | Vocabulary / unmatched policy | F1 % | TP | FP | FN | DeepSolo images/total (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
|prior_native|None|88.4956|950|102|145|233/233 (100.00%)|
|prior_native|Generic_source_reject|90.1027|965|82|130|233/233 (100.00%)|
|prior_native|Generic_keep_unmatched|89.8929|965|87|130|233/233 (100.00%)|
|prior_native|Weak_source_reject|91.4393|972|59|123|233/233 (100.00%)|
|prior_native|Strong_source_reject|85.4675|841|32|254|233/233 (100.00%)|
|ordinary_native|None|88.0480|954|118|141|233/233 (100.00%)|
|ordinary_native|Generic_source_reject|90.5468|977|86|118|233/233 (100.00%)|
|ordinary_native|Generic_keep_unmatched|90.1707|977|95|118|233/233 (100.00%)|
|ordinary_native|Weak_source_reject|92.5597|989|53|106|233/233 (100.00%)|
|ordinary_native|Strong_source_reject|86.7337|863|32|232|233/233 (100.00%)|

`prior_native` is score0.35 / character confidence0.925; `ordinary_native` is score0.40 / no character-confidence pruning. The same original Total-Text ViTAEv2-S+TextOCR checkpoint54d3a454ecc704ef3168ebf9f23fc8390a3b2637d0ec83d5650e8df45e2aca4a and native1024/max1892 preprocessing are retained. No IC13 neural fine-tuning is performed. The90.1707% primary is chosen on the original23-image TRAIN calibration cohort before this corrected-resource TEST run. The fixed original-source reject rule at the same ordinary filter reaches90.5468%, DeepSolo233/233 (100.00%); it is exported separately, without claiming it won calibration.

The source resources are decisive: IC15 Generic has88078 ordered entries, while the actual released IC13 Generic has87778 and different canonical pairs. A deliberately approximate transfer of the IC15 dictionary produced76.2925%/77.0651% with keep-unmatched and the prior/ordinary neural filters respectively, both DeepSolo233/233 (100.00%). Those negative controls remain in the original report; they are not overwritten. Correct IC13 resources include short words such as IN/A/KM absent from the IC15 prepared dictionary. The dictionaries are not constructed or augmented from our evaluation annotations.

Under the unchanged metric, IC13 Strong need not outperform Generic. The supplied Strong lists cover only6/185 short GT instances, versus171/185 for Generic; they cover909/910 long instances. Thus their short-word rejection substantially lowers full-denominator recall. This is a diagnostic of supplied-resource coverage, not a rule for excluding GT or adding test words to the vocabulary. All1095 care words remain in every row, including empty-normalized targets.

## Sources and exact correction

The pinned [official DeepSolo source](https://github.com/ViTAE-Transformer/DeepSolo/blob/dbadae995035246bad3376c7a44c015c69e9b313/DeepSolo/adet/evaluation/text_evaluation_all.py) uppercases the raw CTC string, selects minimum ordinary Levenshtein distance, keeps the first file entry on ties, maps to its canonical pair and retains predictions at distance strictly below1.5. Since edit distances are integers, a distance<=1 RapidFuzz query gives exactly those accepted matches. A separate keep-unmatched policy corrects eligible words while retaining the others. This extension is explicitly an assumption. Invalid-envelope and ignore flags do not affect which word is selected.

IC15 resources are downloaded from the frozen official DeepSolo README annotation link. IC13 resources are supplied by the [SPTS v2 author repository](https://github.com/bytedance/SPTSv2/blob/7cec76a44be5af90a4b17fc21a4ab53edb2966ba/README.md), whose separate lexical archive includes both dataset-specific dictionaries; the DeepSolo author refers IC13 resources to SPTS. Pinned revision7cec76a44be5af90a4b17fc21a4ab53edb2966ba, archive hashes, ordered files and every pair-list hash are preserved. The IC13 archive is downloaded from the linked public file1JxmuDsOZ-x_WO5lck2ZQZHRcjoUtUiLo. No dependency is installed.

The broader official IC13 annotation archive is now available too:229 images and757 prepared training word instances. This resolves availability of one historical missing training artifact, without a claim that the author-specific IC13 adaptation has been trained or reproduced.

## Calibration, metric and uncertainty

Each dataset compares the two frozen neural filters and the two unmatched-word policies on its existing TRAIN calibration set: IC15 100 images/430 words and IC13 23 images/93 words. Select exact rational micro F1, then more TP, prefer keep-unmatched, then the first declared confidence candidate. Choices and code/config hashes are sealed before TEST correction. Upstream training may include these calibration images, and historical TEST exposure is retained. The dataset-specific IC13 run follows the observed negative cross-dataset dictionary control; it is exploratory research rather than a new blind benchmark. Weak/Strong do not choose Generic thresholds.

Normalize each prediction and care word with Unicode casefold+isalnum. Keep every care denominator, match per-image word multisets without spatial IoU, and aggregate TP/FP/FN. Empty/symbol-only care words and outputs never match but remain in denominators; invalid envelopes remain unmatchable FP. Polygon ignore remains the prior native256 centered-letterbox rule. No three-character GT exclusion, accent removal, fuzzy acceptance or TEST label lookup is added. This differs from official polygon/lexicon benchmark evaluation, so numerical proximity to paper F1 is not a strict reproduction claim. IC13 TEST233 versus the context paper229 remains unresolved.

Conditional paired image bootstrap uses 2000 resamples and seed20261028. IC15 source Generic gains4.4012 points versus the same frozen75.2753% expert, with95% gain interval[3.4838,5.4056] points; IC15 calibrated Generic gains3.7037 points, interval[2.8391,4.6476], all DeepSolo500/500 images (100.00%). Intervals omit model/policy selection and historical-search uncertainty.

An additional, separately predeclared guarded correction protects numbers, short words, high-confidence readings and ambiguous neighbors. It reaches75.8759%,DeepSolo500/500 (100.00%) on IC15 and88.0480%,DeepSolo233/233 (100.00%) on IC13 using the initial transferred dictionary. It is a preserved negative/control follow-up and is not promoted as the selected vocabulary expert. Its focused boundary tests pass; the complete independent primary lookup audit covers the original-source/keep variants.

## Prediction-only profiles and outputs

`recognition/deepsolo_lexicon_deployment.py` exposes `DeepSoloVocabularyPostprocessor(profile).apply(response)`. It receives a successful native DeepSolo response with authenticated checkpoint metadata, confidence scores and CTC probabilities; it receives no GT, care masks or other expert predictions. It verifies the checkpoint and dictionary file hashes, applies the fixed confidence filter and emits corrected detections, strings and per-word decisions while retaining original geometry/raw CTC.

Profiles: `configs/approximate/deepsolo_icdar2015_generic_calibrated_v1.json`, `configs/approximate/deepsolo_icdar2015_generic_source_v1.json`, `configs/approximate/deepsolo_icdar13_generic_calibrated_v1.json`, `configs/approximate/deepsolo_icdar13_generic_source_v1.json`. The original fallback profiles and router results remain unchanged. New dictionaries/checkpoints do not enter STRICT defaults.

External root: `/extra1/bmartelli/context-str-runs/deepsolo-icdar2015-lexicon-20261028-v1`. `prediction_only/{dataset}/{calibrated,source}` contains all1466 audited image/profile outputs without GT, with native shard/profile hashes and lexical decisions. Original raw CTC probability arrays remain in the authenticated source shards. No new neural inference, model training, or online-latency benchmark occurs; logical DeepSolo image coverage remains100% on both datasets.

## Verification

**53 focused tests pass.** The final independent auditor uses a separate character-by-character edit-distance test, length bounds, source-order tie handling and independent pair-list parsing, without RapidFuzz. It passes8728 TEST image/count replays,26 complete metric aggregates,30380 dictionary decisions,584 calibration-image replays,12 finite-grid points and3 frozen optima. A second audit passes1466 prediction-only image/profile replays with exact admitted IDs, texts and unchanged geometry/CTC arrays, and exports each verified result. Full per-image records and unchanged-baseline replays are retained.

No runtime failures occurred. Scientific failures remain visible: the IC15-to-IC13 dictionary transfer and the guarded follow-up underperform; calibration-selected keep-unmatched does not always beat the fixed source rule on TEST; IC13 Strong performs poorly under the full short-word denominator. None is hidden by changing normalization, GT eligibility, cutoffs or the reported setting.

Configurations: `configs/approximate/deepsolo_icdar_lexicon_v1.json`, `configs/approximate/deepsolo_ic13_lexicons_v1.json`. Implementation: `recognition/deepsolo_lexicon.py`, `recognition/deepsolo_lexicon_guard.py`, `recognition/deepsolo_lexicon_deployment.py`, `scripts/evaluate_deepsolo_icdar_lexicon.py`, `scripts/evaluate_deepsolo_ic13_lexicons.py`, `scripts/evaluate_deepsolo_lexicon_guard.py`, `scripts/audit_deepsolo_lexicons.py`, `scripts/audit_deepsolo_lexicon_deployment.py`. Tests: `tests/recognition/test_deepsolo_lexicon.py`, `tests/recognition/test_deepsolo_lexicon_guard.py`, `tests/recognition/test_deepsolo_lexicon_deployment.py` and existing metric/worker boundaries.
