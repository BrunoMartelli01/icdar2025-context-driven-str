# ICDAR2015 U-Net: the 70% correct-word F1 objective

This is **approximate reproduction**, not a strict paper result. The user target
is at least 70% symbol-free spatial correct-word F1 on all official TEST500
images, with 80% aspirational. It has not yet been achieved. The preserved
baseline is 40.75558099599313%, with 1068 true positives, 2096 false positives,
1009 false negatives, 3164 predictions and 2077 care words. See
[the preceding completed experiment](ICDAR2015_UNET_IMPROVEMENT.md).

## Evidence and fixed evaluation

The local authoritative paper, section 3.2/Figure 3 and section 4.1, specifies
an attention-gated U-Net with pretrained ResNet-50, BCE and Adam. Decoder
details and the choices below remain evidence-limited; see ambiguity
`ICDAR2015-TARGET70-001`. Existing source implementations and qualified reports
are preserved. The new configuration is
`configs/approximate/icdar2015_target70_v1.json`; STRICT is untouched.

Use the existing 900/100 split of official TRAIN, assigned by the recorded
SHA256 ordering with seed 202503. All training, epoch, threshold, output policy
and branch choices use those 100 validation images only. The final TEST weights
and output policy must be frozen before evaluation. No TEST-driven tuning is
permitted. Every original care word stays in the denominator, including short
words and words with empty normalized text under the established metric rules.
Normalization is Unicode casefold followed by alphanumeric characters only;
matching is one-to-one original-image bounding-box IoU >=0.5 and exact normalized
text. Every emitted prediction counts, including predictions in ignored regions.
No GT-assisted ignore censoring or altered denominator is allowed.

The mask audit already established disjoint word owners and no fused target
components in either dataset. Increasing resolution reduces lost tiny words;
separation alone did not solve false positive detections or recognition errors.

## Predeclared first training comparison

Both branches start from the qualified 768/BCE checkpoint at continuation epoch
18 (source effective epoch 56), SHA256
`79efc8f7f44c719be92c6d8b2a69ce9f204030ff77e2fa05194ac191fa10fe91`.
Restore its complete decoder Adam state and RNG; newly enabled encoder
parameters receive explicitly fresh optimizer moments. Use 100 additional
epochs, FP32, deterministic backends and cosine learning rates. The source
state remains eligible as epoch zero.

| Branch | RGB / native mask | Encoder | Physical batch / accumulation | Supervision |
| --- | --- | --- | --- | --- |
| long768 | 768x768 / 384x384 | frozen | 8 / 1 | unchanged separator-weighted masked BCE |
| adapt1280 | 736x1280 / 368x640 | layer3/layer4 convolutions trainable | 4 / 2 | ignored TRAIN text receives background weight 3 |

Decoder LR decreases from 1e-5 to 1e-6. The adaptation encoder LR decreases from
1e-6 to 1e-7, with all encoder BN affine tensors and statistics frozen. Source
1280x720 images receive 8 pixels vertical padding per side in adapt1280; padding
and rotation fill have no loss. Care masks, owner IDs and separator files are
unchanged. Gradient clipping at norm 1 is used only in adapt1280. This is a
combined adaptation experiment and cannot isolate each change's causal effect.

The existing parent's entire encoder forward is under `no_grad`; enabling
`requires_grad` alone cannot fine-tune it. The new model explicitly overrides
that forward. Tests must establish actual gradients in layer3/layer4, frozen
earlier stages/encoder BN, and identical frozen-source predictions.

Actual host hardware must be recorded for each run. Physical batch 4 and
accumulation 2 are a HARDWARE_DEVIATION; decoder BN sees batch 4, so this is not
claimed numerically equivalent to physical batch 8. Measure full backward peak
allocated/reserved VRAM and preserve OOM failures; do not silently reduce batch,
switch precision or substitute hardware profiles.

## Word selection and output calibration

Run the qualified frozen PARSeq on CPU at epoch zero, every fifth epoch, every
new geometry best, and the final epoch. Retain crop RGB hashes, OCR token outputs,
all raw proposals and per-image metrics. Before final reporting, compare CPU
and GPU recognition on the selected validation crops.

The separate `scripts/qualify_icdar2015_target70_ocr_device.py` command rebuilds
every selected validation crop from the saved probability maps and original RGB,
verifies the CPU cache hashes, then recognizes every unique crop on CUDA in FP32
with the same frozen PARSeq weights, decoder and batch size. It requires exact
raw text, token IDs, output-policy decisions and raw/emitted aggregate metrics;
floating confidence differences are measured. Empty outputs remain included.
This device comparison is a HARDWARE_DEVIATION qualification, not a new model
selection step. It must pass on the final selected validation policy before a
GPU TEST result is reported. Full per-crop GPU outputs and allocated/reserved
memory peaks are saved; failures and OOM propagate without a device substitution.

The source operating policy passed this comparison on all 100 validation images
and all 722 unique raw crops: zero text, token-ID or output-membership differences;
raw/emitted aggregate metrics are identical. Six focused equivalence tests also
passed. RTX3090 FP32 PARSeq batch 32 peaked at 205,955,584 allocated and
249,561,088 reserved bytes in its own process; the U-Net control was training
in a separate process on the same GPU. Free/total GPU memory before OCR is
recorded separately. See
`provenance/ICDAR2015_TARGET70_OCR_DEVICE_V1.json`. This qualifies the source
operating policy only; a different final selected model still requires its own
full validation-crop check.

On 2026-10-27 the original training process was confirmed absent after epoch 51;
its complete last checkpoint, optimizer/RNG state, 51-epoch history and unchanged
source contract were verified. No failure file was present. Resume starts at
epoch 52 under the identical scientific configuration. An explicitly recorded
detached runner preserves training across app-turn interruptions and queues
adapt1280 after long768 completes. This operational restart does not change
weights, thresholds, splits or planned epoch counts. At epoch 50 the control
reached validation emitted F1 60.3306%; this is not a TEST result or goal completion.

The profile predeclares confidence thresholds and normalized length gates.
Calibrate them exclusively on fixed TRAIN validation. Output rejection uses
only predicted OCR text/confidence and cannot inspect GT. Every care GT remains
in the denominator, so rejecting a real short word causes a false negative.
Report raw U-Net/PARSeq F1 separately from F1 after the selected output policy.
Select checkpoint/policy by emitted word F1, then raw word F1, geometry F1 and
earliest epoch. Policy ties prefer recall, shorter minimum length, smaller
threshold, then confidence-field name. Save the whole calibration table.

A preliminary validation-only replay of the previous source suggested roughly
58% F1 after confidence/length rejection versus 41% raw F1. This is a diagnostic,
not a TEST result or evidence that the U-Net has reached the target. New runs
must persist their own auditable calibration outputs.

## Perfect-localization diagnostic

The separate `icdar2015_target70_oracle_v1.json` profile predeclares GT axis boxes
with 0/2/4 original RGB pixels of padding and official quadrilateral rectification
with no padding. Bilinear perspective sampling uses the official ordered four
corners and replicated borders; invalid quadrilaterals fail explicitly. Run only
on the same 100 TRAIN validation images and all 430 care words using frozen CPU
PARSeq. Preserve every OCR output and crop hash. This ASSUMPTION measures
recognition under perfect localization; its accuracy is a GT-assisted diagnostic
and cannot be reported as achieved detector F1 or used for TEST decisions.

## Execution and completion evidence

Final model selection uses the isolated
`configs/approximate/icdar2015_target70_selection_v1.json` profile and
`scripts/select_icdar2015_target70.py`. Both full 100-epoch branches, their
contiguous complete epoch histories and complete 20-case validation operating
grids are required; smoke or partial training is rejected. The qualified source
remains eligible. Replay each candidate's raw and emitted metrics against every
original validation care word, verify all source/checkpoint/report hashes, then
freeze the winner without reading TEST. Use the predeclared operating-policy
ordering, then fewer RGB pixels, earlier effective epoch, source preference and
name for exact ties. This cross-model rule is an ASSUMPTION. Load the eventual
winner with its original architecture identity, fully strict checkpoint loading
and matching saved training/run contracts; no non-strict architecture conversion
is permitted. A frozen selection is not proof that TEST F1 reaches 70%.

`scripts/evaluate_icdar2015_target70.py` uses the separate target70 word-F1
profile, a frozen selection and the exact selected-policy CPU/GPU qualification.
The complete original TEST500 manifest and counts (2077 care words, 3153 ignored
annotations) are required. Segmentation, crop generation and OCR never read TEST
annotations; annotations enter only after all predictions are frozen. Preserve
probability masks, every raw component, crop hashes and OCR token outputs. The
primary result counts every output emitted by the validation-selected policy,
including ignored-region predictions; raw F1 and rejected-output counts are
reported separately. Keep all original care words in both metrics. Exact-protocol
cache reopening verifies every raw crop and rejects omitted singleton components.
Run U-Net and PARSeq sequentially in FP32 and measure separate allocated/reserved
peaks. This is an approximate standalone U-Net/PARSeq result, not paper Table 5.
The complete selected-model evaluation and audit below qualify this entry point;
unit checks alone would not establish a complete TEST result.

`scripts/audit_icdar2015_target70.py` independently parses the original CSV GT,
reconstructs native connected components with SciPy, inverts coordinates, replays
output rejection, and recomputes one-to-one matching and micro F1 for both raw
and emitted predictions. Every official image, care word, ignored annotation,
singleton component and raw OCR crop must remain auditable. Crop byte replay
explicitly shares the frozen production crop renderer; it is not an independent
orientation implementation. This audit does not recompute neural inference and
does not by itself qualify a new GPU/checkpoint. The complete selected-model
audit below has now passed.

The audit entry point passed a compatibility smoke replay of the immutable
historical baseline on all 500 images: 2077 care words, 3153 ignored annotations,
3164 raw/emitted predictions, 1068 correct normalized spatial matches and
F1 40.75558099599313%. Only schema/protocol metadata and original cached OCR
confidence fields were adapted in a separate qualification directory; original
boxes, transcripts, masks, GT and metrics were preserved. This reruns no neural
inference, selects no TEST parameters and is not a new target70-model TEST
result. See `provenance/ICDAR2015_TARGET70_LEGACY_AUDIT_SMOKE_V1.json`.

An additional predeclared operating grid uses saved validation probability maps
at thresholds 0.2/0.35/0.5/0.65/0.8, with axis-aligned or predicted-component
minimum-area-rectangle OCR crops, each with 0/2 original RGB pixels padding.
See `icdar2015_target70_operating_policy_v1.json`. Every connected component is
retained, including single pixels. Matching always uses its original component
bounding box; padding/rotation affects only the OCR crop. The default 0.5/axis0
must exactly replay its qualified raw report. This ASSUMPTION calibrates detector
and crop operating choices solely on TRAIN validation. Report all candidates,
raw/emitted populations and selected confidence policy. Predicted orientation
uses no GT. Preserve the source and compare its baseline policy first; repeat
the same grid on completed selected training branches before freezing a final
model/policy for TEST. These extensions do not establish a strict paper result.

The source-only operating-grid replay has completed all 20 candidates on VAL100,
with zero omitted samples and an exact replay of the qualified 0.5/axis0 raw
baseline. Its best validation policy remains threshold 0.5, with predicted
oriented crops and 2 RGB pixels padding. Mean character confidence >=0.95 and
minimum normalized length 3 retain 347 predictions: 230 TP, 117 FP, 200 FN,
F1 59.2021%. Raw F1 for those crops is 42.0139% (242/722 predictions correct).
The unchanged source axis0 raw F1 is 41.1458%; confidence filtering alone gives
58.1867%. These are validation results, not new TEST results or proof of 70%.
The orientation convention places the longer rectangle axis horizontally with
angle in [-90,90) degrees; all geometry comes from predicted component pixels.

The saved source diagnostic has 147 unmatched care words under geometric
maximum matching; 125 of those are correctly read from perfect GT axis crops.
Another 46 geometrically matched words fail the word match. Geometry and word
maximum matchings are separate; these counts diagnose the saved matching sets,
and are not alternative deployable F1 metrics. The main unresolved work is
improving the learned detector and crop quality, not changing denominators.

Qualification: 52 focused training/regression tests, 2 oracle-geometry tests,
and 6 operating-crop tests passed. Initial test collection failed because two
test modules shared a basename; one later assertion referenced the wrong
aggregate field. Both test issues were corrected before the passing run.
Both full-size FP32 backward smokes passed on RTX3090: long768 peak allocated
8,853,711,872 / reserved 12,136,218,624 bytes; adapt1280 allocated 9,992,550,912 /
reserved 13,035,896,832 bytes. No OOM occurred in qualification. See
`tests/segmentation/test_icdar2015_target70.py`,
`tests/models/test_icdar2015_target70_encoder.py`,
`tests/scripts/test_icdar2015_target70_oracle_geometry.py`,
`tests/segmentation/test_icdar2015_operating_crops.py`, and
`provenance/ICDAR2015_TARGET70_QUALIFICATION_V1.json`.

The perfect-localization diagnostic completed on all 430 validation care words:
axis0 392/430 (91.16%), axis2 393/430 (91.40%), axis4 390/430 (90.70%), and
quadrilateral rectification 401/430 (93.26%). This is GT-assisted accuracy,
not deployable word F1. It supports investigating detection/crop quality.

## Completed first comparison and TEST500 result

Both branches completed all 100 additional epochs. Both retained epoch 50:
long768 checkpoint `66f27858d6e91d418305aefdd95a3f12207709494821de7059ee3ef5e2c5b29a`,
adapt1280 checkpoint `c4676914dbcbf559a26fb98e51f5ae137b2a3ac66930d0191fca5b214fce5eb6`.
All three eligible models completed the 20-case VAL100 operating grid. The
frozen selection chose adapt1280, threshold 0.5, axis-aligned crops with two RGB
pixels padding, mean character confidence >=0.95 and normalized length >=3.
Its validation emitted F1 was 61.7608%; raw F1 was 49.8480%.

The selected policy passed CPU/CUDA FP32 PARSeq equivalence on all 557 unique
validation crops, with no text, token-ID or output-membership differences.
Twelve focused final-selection/evaluation/audit tests passed. Complete TEST500
inference, independent audit and exact-protocol cache reopening then succeeded:

| TEST500 output | Correct words | Predictions | FP | FN | Word F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Preserved original, raw | 1068 | 3164 | 2096 | 1009 | 40.7556% |
| adapt1280, raw | 1007 | 2544 | 1537 | 1070 | 43.5836% |
| adapt1280, validation-selected rejection | 963 | 1439 | 476 | 1114 | 54.7782% |

Every row includes all 2077 care words. All 500 images and 3153 ignored
annotations were accounted for, with zero failed samples. The independent audit
verified all 2544 raw crop hashes, connected components, output membership and
both micro metrics. It shares the crop renderer and does not repeat neural
inference. The saved results SHA256 is
`f20ecf59be21832315651d8338b6160e8e6d40c22ef22508d192501959d97b48`.

Applying the same already-frozen confidence/length gate to immutable historical
baseline predictions gives 54.0819% (997 TP, 613 FP, 1080 FN). This is a post-hoc
fixed-gate diagnostic, with no TEST parameter selection or new inference; crops
and detector policies differ. The new neural/raw gain is 2.8281 percentage
points, and the comparison with this common gate is only 0.6962 points. Correct
raw word recall fell from 1068/2077 to 1007/2077. The main remaining problem is
missed words; the 70% objective remains unachieved. See
`provenance/ICDAR2015_TARGET70_TEST500_COMPLETION_V1.json` for artifact hashes
and measured memory. Historical readiness receipts retain their original scope.

The entry point is `python -m scripts.train_icdar2015_target70`; dataset,
checkpoint, source reports, PARSeq source/archive/weights, output root, branch,
device and validation batch are required arguments. No assets are downloaded or
packages installed. Fully resolved configuration and source hashes are saved.
Smoke results, full training results and final TEST results must remain separate.

Focused tests and full-size backward smoke qualification precede full training.
Do not claim completion from pixel F1, validation F1, a partial TEST subset,
or a filtered metric that silently discards GT or ignored-region predictions.
The goal remains active until the unchanged complete TEST metric verifies 70%,
or a documented external blocker actually prevents further meaningful work.
