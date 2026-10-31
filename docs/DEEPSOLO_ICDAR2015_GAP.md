# DeepSolo ICDAR2015: checkpoint and threshold gap analysis

Date: 2026-10-28. **Approximate reproduction.** The user explicitly requires the unchanged word F1 with punctuation removed. All TEST comparisons use the complete 500 images and 2077 care words; standalone DeepSolo receives500/500 images (100.00%). No router is evaluated in this study.

## Result

Using official IC15-trained ViTAEv2-S+TextOCR weights, native IC15 resize and calibration-selected filtering improves lexical micro F1 from **70.5669% to 75.2753%**, a gain of **4.7084 percentage points**. Conditional paired 2000-draw image-bootstrap95% gain interval is **[3.1279, 6.2830] points**. The nominal82% paper reference has **not** been reached under this fixed local metric.

| Configuration | F1 % | Precision % | Recall % | TP | FP | FN | DeepSolo images/total (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|Original Total-Text weights,1024, original filter|70.5669|76.3158|65.6235|1363|423|714|500/500 (100.00%)|
|Total-Text weights,1024, IC15 source filter|67.2996|62.1533|73.3751|1524|928|553|500/500 (100.00%)|
|Total-Text weights,1024, calibrated filter|70.3775|70.7543|70.0048|1454|601|623|500/500 (100.00%)|
|Total-Text weights,1440, original filter|67.1788|74.6761|61.0496|1268|430|809|500/500 (100.00%)|
|Total-Text weights,1440, IC15 source filter|61.2137|52.9123|72.6047|1508|1342|569|500/500 (100.00%)|
|Total-Text weights,1440, calibrated filter|69.3582|74.9860|64.5161|1340|447|737|500/500 (100.00%)|
|IC15 weights,1440, original filter|74.8686|82.4551|68.5604|1424|303|653|500/500 (100.00%)|
|IC15 weights,1440, IC15 source filter|71.9286|68.5266|75.6861|1572|722|505|500/500 (100.00%)|
|IC15 weights,1440, calibrated filter|75.2753|78.3742|72.4121|1504|415|573|500/500 (100.00%)|

The original filter is instance score 0.35 / character confidence 0.925. The fixed official IC15 source filter is 0.30 / no character-confidence rejection. Calibrated choices are 0.35/0.80 for Total-Text1024,0.40/0.85 for Total-Text1440, and0.35/0.85 for IC15 weights1440. No TEST-best threshold is searched or substituted.

## What caused the gap

1. **The old result was a dataset transfer.** The original checkpoint was fine-tuned on Total-Text, rather than being the official IC15-fine-tuned model. The complete public IC15+TextOCR checkpoint downloaded from the pinned upstream README has 135,515,957 bytes and SHA256 `c0aac6fddc297ca1a7e3c2d0bbd5f7f8b2f55214795c7fa32f13fb51c9db95a8`. Complete ZIP CRC and strict GPU loading pass with zero missing/unexpected/incorrect-shape keys and 33,748,854 parameters. Authors publish no reference checksum; source URL, complete bytes and architectural compatibility establish this approximate pairing.

2. **More pixels alone do not solve it.** With the same original filter, changing only Total-Text weights' resize from 1024/max1892 to 1440/max4000 reduces F1 from 70.5669% to 67.1788%, with DeepSolo500/500 (100.00%) in each case. At the same1440 resize and original filter, switching to IC15 weights reaches 74.8686%, DeepSolo500/500 (100.00%). The experiment separates those changes; it does not establish a complete checkpoint-by-resolution interaction study because IC15 weights at 1024 are not tested.

3. **Filtering is dataset-dependent.** The official source filter, the old Total-Text filter and the calibration-selected filter have different precision/recall balances. For the IC15 weights, calibration raises the fixed-source-filter F1 from 71.9286% to 75.2753%, both DeepSolo500/500 (100.00%). The 100 calibration images come from official TRAIN and can occur in upstream pretraining/fine-tuning; their 82.7423% calibration F1 is not a TEST result or an unseen-expert validation claim.

4. **Paper82 is not a defined equivalent F1.** The authoritative local paper, Section4.3 after Table5, cites standalone DeepSolo82%; Table5 itself gives the context pipeline77.3/78.2. Its recognition-accuracy normalization, matching, denominator, checkpoint and lexicon settings remain unspecified. The [pinned official DeepSolo model zoo](https://github.com/ViTAE-Transformer/DeepSolo/blob/dbadae995035246bad3376c7a44c015c69e9b313/DeepSolo/README.md) reports distinct IC15 Strong/Weak/Generic-lexicon E2E scores:88.1/83.9/79.5 for ViTAEv2-S+TextOCR. Those are not this lexicon-free lexical F1. The exact provenance of the context paper's82 cannot be inferred from a rounded nearby zoo number.

## Remaining errors under the unchanged metric

The new model adds 141 net true positives, removes eight net false positives and reduces false negatives from 714 to 573. Geometry below is a separate error diagnostic, never a replacement F1 or a GT-dependent inference policy. Using bbox IoU 0.5, matched care GT increases from 1476/2077 to 1684/2077; correctly transcribed spatial matches increase from 1359 to 1493. The new geometry matching leaves 393 GT unmatched; its cardinality exceeds exact normalized spatial matching by 191. These are independently maximized matchings, so their difference is not a unique per-word attribution of recognition failures. Lexical TP is 1504 because its declared matching has no spatial constraint.

| Original GT word height | Care GT | Old localized | New localized | Old correctly transcribed at IoU0.5 | New correctly transcribed at IoU0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
|At most12 pixels|22|7|14|6|12|
|Over12 through24 pixels|536|333|429|305|375|
|Over24 pixels|1519|1136|1241|1048|1106|

The word-length error audit further shows the old filtered model has 193 false positives of length1/2 versus 15 correct length2 predictions. The new model reduces those short false positives to 25, with 7 correct length2 predictions, but still has335 false positives of length at least4. Simply deleting short predictions is therefore insufficient to resolve the remaining gap; no such TEST-derived length policy is deployed.

## Fixed evaluation and calibration protocol

Normalize every prediction and care transcription with Unicode casefold followed by isalnum, removing punctuation/spacing without NFKC, lexicons, edit-distance acceptance or accent removal. Match normalized words as per-image multisets; aggregate TP/FP/FN over all 500 images. Empty/symbol-only outputs and targets remain in denominators and never match; invalid envelopes remain unmatchable FP candidates. Ignore only predicted nonempty polygons wholly outside the existing care-validity mask after center-square letterbox and nearest resize256. No failed/empty image is removed. Ground-truth polygons/text and validity enter evaluation only. Raw native greedy CTC, RGB conversion, padding and inverse-coordinate mapping are verified against the pinned source. Prepared JPEG bytes are preserved by dataset preparation.

The frozen nine-by-six score/character-confidence grid is selected by exact rational calibration micro F1, then higher recall, lower score threshold and lower character-confidence threshold. Both the per-model choices and winning model are saved before new TEST inference. Calibration is100 official TRAIN images, separate from all 500 TEST images, with the upstream-exposure caveat above. The direct source filter and original filter are predefined controls rather than TEST-tuned alternatives. Paired2000-draw intervals, seed20261028, condition on those fixed choices and omit their selection uncertainty.

## Artifacts, hardware and verification

External study: `/extra1/bmartelli/context-str-runs/deepsolo-icdar2015-gap-20261028-v1`. Main configuration: `configs/approximate/deepsolo_icdar2015_gap_v1.json`. Exported selected native profile: `configs/approximate/deepsolo_icdar2015_selected_native_word_v1.json`; callers must supply its explicit external runtime/checkpoint locations. The prior router fallback and historical reports remain unchanged.

Two new models/configurations each run 100 calibration and 500 TEST images: 1200 physical DeepSolo inferences total, including the two successful smoke images reused by the full export. The old 600 calibration/TEST predictions are reused. Logical standalone cost for every TEST row remains500/500 images (100.00%). This is offline full-expert evaluation, without a router or complete online-latency benchmark.

Actual hardware is RTX3090 24GiB, FP32 batch1, CHW3x1440x2560; one native DeepSolo model per legacy process. Both variants peak at6,948,111,872 allocated and12,215,910,400 reserved bytes. The reservation includes allocator history; no8GiB complete-pipeline qualification is claimed. No OOM occurs. Inference seed20261025 seeds Python, NumPy, PyTorch and CUDA. cuDNN is enabled, benchmarkFalse and deterministicFalse; existing legacy matmul/cudnn TF32 permissions are both enabled and preserved. FP32 describes model/tensor storage, without a claim that TF32 arithmetic permissions were disabled. Legacy Python3.8/torch1.9.0+cu111/detectron2 0.6+cu111 and main Python3.11/torch2.4.1+cu124 remain isolated and unchanged; no packages are installed.

**49 focused tests pass.** Independent audit passes162 calibration-grid optima,4500 full TEST image/count replays,9 complete metric aggregates,19826 independent polygon-ignore decisions and1200 authenticated new inference shards. It independently reconstructs normalized lexical matching, official CSV care transcriptions and centered polygon ignore, verifies exact threshold/model selection, and checks strict loading, shapes, precision, sample/profile/plan bindings and payload hashes. Historical70.5669% baseline replays exactly.

Preserved failures: the direct official download succeeds; an alternative public OneDrive API endpoint returns401 and is not used. The first TEST process stops before inference because its generated plan accidentally points to the validation plan instead of the profile. The original invalid plans, failure traceback and corrected-pointer receipt remain; only the file pointer is repaired, with identical model parameters, images and frozen calibration choices. Successful runs omit no images.

Implementation: `recognition/deepsolo_icdar2015_gap.py`, `scripts/analyze_deepsolo_icdar2015_gap.py`, `scripts/audit_deepsolo_icdar2015_gap.py`; inference reuses unchanged `scripts/export_essential_transfer_deepsolo.py` and the qualified `deepsolo/worker.py`. Tests: `tests/recognition/test_deepsolo_icdar2015_gap.py` plus existing decoding, word metric, transfer, worker and process-boundary suites. Scientific classifications: `DEEPSOLO-IC15-GAP-20261028` in `docs/PAPER_AMBIGUITIES.md`.
