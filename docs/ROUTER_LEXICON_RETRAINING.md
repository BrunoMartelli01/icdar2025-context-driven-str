# ICDAR router retraining with corrected DeepSolo teachers

**Approximate reproduction / research extension.** User-authorized retraining of
both ICDAR2013 gates against the calibrated Generic teacher, and both ICDAR2015
gates against the ICDAR2015-finetuned Weak teacher after selection of the best
eligible completed U-Net. Local paper Section3.5/Algorithm2 and the existing
teacher/postprocessor/native budget gate/validation implementations were inspected.

Keep the original RGB-disjoint router partitions: IC13 FIT183/selection23/CAL23,
TEST233/1095 care words; IC15 FIT900/selection50/CAL50, TEST500/2077 care words.
IC13 retains its selected corrected TotalText U-Net; YOLO retains direct640 and
genuine PARSeq. Use the same finite five-head recipes and OCR/token/genuine-map
descriptors. Fit new paired-utility targets, choose heads on selection only,
calibrate the three original quota targets on separate CAL and never refit.
IC13 primary DeepSolo budget70/233 (30.04%); IC15 primary250/500 (50.00%).
Scalar cuts need not transfer those percentages. Report every actual count.

IC15 candidate comparison is finite and frozen before retraining: TotalText raw
and corrected Text+Kernel experts, qualified source768/long768/adapt1280, completed
tt263/tt_partial23 native1280 transfer branches, two encoder-decay controls and
two completed OHEM branches. Existing upstream operating-policy selection used
VAL100 historically; the new U-Net comparison uses only the original50 selection
images. Ongoing SynthText training is excluded rather than reading partial,
mutable weights. Test the corrected TotalText expert on all500 TEST images too.

IC13 teacher uses the already qualified Generic dictionary/profile. IC15 Weak
teacher uses exactly the new checkpoint and score0.35/character0.85/source-reject
policy. For TRAIN fitting/selection/calibration, construct the ordered uppercase
Weak dictionary from all official TRAIN1000 care transcriptions only, retaining
the first canonical spelling. This models dataset-level Weak side information
without importing TEST words into gate fitting. TEST uses the previously
qualified released Weak dictionary and frozen predictions. No lexicon is applied
to the light experts. A TRAIN1000/TEST500 dictionary-size/domain shift remains.

The metric remains per-image multiset microF1 after Unicode casefold/isalnum,
all care GT and empty/symbol-only denominators retained, no localization match,
native256 polygon ignore with the original router care override and invalid
envelopes unmatchable. The IC15 endpoint is therefore82.7905%, DeepSolo500/500
(100%), rather than the standalone prepared-validity score82.8968%, DeepSolo500/500
(100%). All GT/image counts are identical; preserve both results/protocols.

Classifications: PAPER_SPECIFIED broad fallback motivation only;
REPOSITORY_SPECIFIED qualified experts/partitions/head recipes/metric procedures;
ASSUMPTION TRAIN-only Weak dictionary and new dataset-specific teacher targets;
HARDWARE_DEVIATION shared RTX3090 FP32 batch1 legacy export and CPU fitting;
UNRESOLVED finite validation, upstream selection/expert exposure, historical TEST
exposure, pretraining overlap, Weak vocabulary transfer and paper metric/latency
equivalence. DeepSolo's IC15 author checkpoint has seen official TRAIN; the
router validation is held out from router fitting, not from upstream DeepSolo.
No strict or newly blind result is claimed.

External run: `/extra1/bmartelli/context-str-runs/router-lexicon-retraining-20261028-v1`.
Explicit config: `configs/approximate/router_lexicon_retraining_v1.json`.
Missing new-checkpoint TRAIN inference:900 IC15 images. Existing100 IC15,229 IC13
TRAIN, and733 TEST fallback predictions are reused; actual physical calls are
reported separately from logical final routing counts. Preserve other authorized
GPU training and report allocator peaks without an isolated runtime claim.

Completion results and independent audits will be appended after full evaluation.

## TRAIN Weak serialization repair

All900 new neural exports completed successfully. The first TRAIN dictionary
construction stopped before router fitting because33 of2379 raw vocabulary
entries contain embedded spaces. The released whitespace-delimited pair parser
cannot represent those literal keys. Preserve the failed source, traceback and
partial vocabulary under `failures/train_weak_pair_format_v1/`. The corrected
TRAIN-only artifact stores the same complete uppercase-to-canonical mapping in
JSON and instantiates the existing correction class directly. Word order,
distance threshold, canonical spellings, GT population and metric are unchanged;
no neural inference is repeated. A focused regression verifies space-containing
and punctuation-containing entries survive serialization and correction. The
supervised v2 workflow preserves the failed v1 status instead of overwriting it.

## Completed results

**Approximate reproduction / research extension.** Complete official TEST and unchanged word lexical F1. The previous controls below already use the new Generic/Weak DeepSolo; differences isolate new router fitting rather than the prior fallback upgrade.

| Dataset | Expert | Light-only F1 % | DeepSolo light-only/total (%) | Previous router F1 % | Retrained F1 % | DeepSolo router/total (%) |
| --- | --- | ---: | --- | ---: | ---: | --- |
| icdar13 | unet (tt_corrected) | 73.3237 | 0/233 (0.00%) | 84.0038 | 81.2772 | 70/233 (30.04%) |
| icdar13 | yolo (yolo) | 82.3254 | 0/233 (0.00%) | 87.3120 | 86.9321 | 70/233 (30.04%) |
| icdar2015 | unet (long768) | 59.7031 | 0/500 (0.00%) | 71.4805 | 76.6527 | 250/500 (50.00%) |
| icdar2015 | yolo (yolo) | 65.8134 | 0/500 (0.00%) | 75.8759 | 77.7581 | 250/500 (50.00%) |

### Separate validation scalar cuts

Cutoffs are selected again from the separate CAL scores after retraining, with no subsequent refit. Batch budgets and fixed scalar cuts are different policies. The scalar before/after routes can send different numbers of images to DeepSolo; their F1 difference is not a matched-cost improvement.

| Dataset | Expert | New cutoff | Previous F1 % | Previous DeepSolo/total (%) | Retrained F1 % | New DeepSolo/total (%) |
| --- | --- | ---: | ---: | --- | ---: | --- |
| icdar13 | unet | -0.1130834379 | 86.3398 | 109/233 (46.78%) | 85.2899 | 109/233 (46.78%) |
| icdar13 | yolo | -0.0290923625 | 88.4201 | 103/233 (44.21%) | 87.0787 | 73/233 (31.33%) |
| icdar2015 | unet | -0.5500571779 | 71.7295 | 263/500 (52.60%) | 75.7225 | 234/500 (46.80%) |
| icdar2015 | yolo | -1.7407489754 | 75.9709 | 254/500 (50.80%) | 75.8897 | 196/500 (39.20%) |

### ICDAR2015 U-Net comparison

All eleven candidates use only the original50 selection images /215 care words. This finite comparison includes six additionally qualified completed branches; it is not an exhaustive or unbiased comparison after historical upstream VAL100 policy exposure.

| Candidate | Selection F1 % | DeepSolo images/total (%) |
| --- | ---: | --- |
| long768 | 62.5323 | 0/50 (0.00%) |
| source768 | 60.9626 | 0/50 (0.00%) |
| adapt1280 | 60.5405 | 0/50 (0.00%) |
| icdar2015-encoder-decay-v1__operating-policy-validation-decay_control-v1 | 60.5405 | 0/50 (0.00%) |
| icdar2015-encoder-decay-v1__operating-policy-validation-no_encoder_decay-v1 | 60.5405 | 0/50 (0.00%) |
| icdar2015-target70-ohem-v1__operating-policy-validation-ohem_full-v1 | 60.5405 | 0/50 (0.00%) |
| icdar2015-target70-ohem-v1__operating-policy-validation-ohem_partial-v1 | 60.5405 | 0/50 (0.00%) |
| icdar2015-totaltext-weight-transfer-v1__operating-policy-validation-tt_partial23-v1 | 55.2486 | 0/50 (0.00%) |
| tt_corrected | 47.3520 | 0/50 (0.00%) |
| tt_raw | 45.6647 | 0/50 (0.00%) |
| icdar2015-totaltext-weight-transfer-v1__operating-policy-validation-tt263-v1 | 34.3949 | 0/50 (0.00%) |

The pure TotalText corrected expert also runs on complete IC15 TEST: F146.0818%, DeepSolo0/500 (0.00%). The selected long768 remains59.7031%, DeepSolo0/500 (0.00%). Additional candidate checkpoints, completed training manifests and matching FIT900/VAL100 IDs are verified in candidate_qualification.json;550 selection word recounts and500 TotalText TEST recounts pass candidate_selection_audit.json.

### Interpretation, gate size and qualifications

Retraining is not automatically an improvement. Report negative outcomes, keep both old and new heads and do not promote a winner using TEST. The new fixed-budget differences and their conditional uncertainty are stored in comparison.json. Tiny selection populations, imperfect observable routing signals and the changed teacher utilities remain limitations. TRAIN-only Weak versus released TEST Weak is an explicit vocabulary distribution shift; the neural IC15 teacher itself has seen official TRAIN upstream.

| Dataset | Expert | New groups | Feature dimension | Serialized deployment bytes | Additional encoder families |
| --- | --- | --- | ---: | ---: | ---: |
| icdar13 | unet | ocr | 18 | 6531 | 0 |
| icdar13 | yolo | ocr, tokens | 26 | 6779 | 0 |
| icdar2015 | unet | ocr, tokens | 26 | 22841 | 0 |
| icdar2015 | yolo | ocr | 18 | 22566 | 0 |

Single-image deployment replay validates every gate score and all three CAL policies for every TEST image. The core independent audit verifies weighted Ridge normal equations, boosted-tree refits, all candidate selection routes, disjoint RGB groups, calibration cuts and exact conditional-oracle certificates. Teacher audits recount all1962 TRAIN/TEST word records and verify the Weak dictionary contains only TRAIN words. Raw CTC/geometry outputs, checkpoint/source hashes and per-image GPU/input/precision/memory records are preserved. No sample is omitted.

Physical new inference is900 IC15 TRAIN calls; reuse229 IC13 TRAIN,100 IC15 TRAIN and733 TEST exports (1062 reused). All final routing is logical on cached outputs; no online timing/speedup was measured. New heads have no extra image/text encoder. Deployment files and bootstrap intervals are in each dataset/full/report/. The combined comparison.json/.csv bind the matched previous controls. Both old and newly trained models remain available.

No strict or paper-equivalent result is claimed. All source, tests, errors and comparison artifacts must be sealed in the final completion receipt.
