# Controlled ICDAR2015 U-Net improvement

This completed experiment is an **approximate reproduction**. It extends the
qualified [Total-Text transfer](ICDAR2015_TOTALTEXT_TRANSFER.md) after its complete
100-epoch training and [TEST500 word evaluation](ICDAR2015_WORD_F1.md). The local
paper `papers/2503.15639v2.pdf`, sections 3.2/Figure 3 and 4.1, was inspected
before implementation. Attention-gated U-Net/ResNet50 and BCE/Adam families are
paper specified. Larger inputs, per-word balancing, GroupNorm and this
continuation/selection procedure are assumptions, not strict paper settings.
See ambiguity `ICDAR2015-UNET-IMPROVEMENT-001`.

## Frozen protocol

`configs/approximate/icdar2015_unet_improvement_v1.json` predeclares the variants.
All use the existing 900 FIT / 100 validation assignment from official TRAIN.
Only those 100 validation images select resolution, epoch and final branch.
The historical epoch-38 best checkpoint initializes every branch; its decoder
has never trained on validation. Fresh Adam, FP32, batch 8, learning rate 1e-5,
weight decay 0.01 and 20 complete continuation epochs are matched. Augmentation
continues from source epoch 38 with identical random seeds and sample order.
The starting state is an eligible epoch-zero candidate.

First audit all 1000 ICDAR and 1255 Total-Text TRAIN target maps at source size,
then at RGB sizes 512x512, 768x768, 1024x1024 and 576x1024 (native masks half
each dimension). Count every original care word, including words whose native
pixels disappear. Record losses, fragmentation and any 8-connected component
touching multiple word owners before and after the qualified native contact
guard. These are target diagnostics, not detection results.

Screen frozen source weights at all four shapes. Select the larger shape by
original-coordinate validation geometric F1, recall, lower RGB pixel count,
then name. Compare four matched continuations: 512/BCE/BatchNorm control,
larger/BCE/BatchNorm, larger/word-balanced loss/BatchNorm, and
larger/word-balanced loss/GroupNorm. The latter explicitly migrates the 21
decoder/bottleneck normalization layers, preserving affine parameters and
convolution weights; encoder BatchNorm remains frozen. Complete source states
load strictly before migration; no keys are silently discarded.

Word-balanced loss gives each surviving care word equal positive BCE weight,
balances aggregate foreground and separator-weighted background costs, and
adds local per-word Dice with weight 0.5. Dice excludes other owners in the
local expanded box. Ignore and padding contribute no loss. These GT-derived
maps are training supervision and metric references only; inference never
uses them.

Compare resolutions against the same original-coordinate care boxes, using
maximum-cardinality one-to-one box IoU >= 0.5. This differs from the historical
native-target-envelope selector; rerun the source baseline under this common
metric. Keep probability cutoff 0.5, all 8-connected proposals, pixel-area
ordering, padding zero, no area cutoff or proposal cap. Each epoch maximizes
geometric F1, then fewer fused words, recall, pixel IoU and earliest epoch.
The final branch maximizes validation spatial word F1 with Unicode casefold
and alphanumeric normalization, then geometry, lower pixel count and name.
Only the selected branch receives the final complete TEST500 evaluation under
the existing word protocol. All predictions, including blanks and predictions
in ignored GT regions, count in precision. Previous TEST inspection means this
is exploratory improvement, not an untouched confirmatory test or paper claim.

## Data integrity qualification

Initial full checksum validation found three RGB mismatches in historical
`artifacts/totaltext-curved-word-boxes-v4` (TRAIN IDs 40, 149 and 336). All mask,
validity, separator and owner files matched the pinned manifest. The original
scene-segmentation dataset contains byte-identical RGB originals. The isolated
external view `totaltext-v4-verified-improvement-view-v1` uses these three
verified originals and contains copies of the other qualified files. Its manifest bytes
are unchanged; `integrity_qualification.json` records observed/expected hashes
and recovery paths. The historical artifact is preserved. The initial failure
is not resolved by disabling checksum validation. An initial linked view failed
the dataset's root-containment guard; links were materialized before successful
auditing. All 1255 Total-Text annotation files also match the original dataset
byte for byte. The historical RGB drift is detected, not explained by this audit.

## Completed target and resolution checks

The complete target audit verifies 1000 ICDAR2015 / 4448 care words and 1255
Total-Text / 9285 care words. There are zero fused word components after the
native guard in every tested resolution for both datasets. Thus absence of
word separation in the qualified targets does not explain the current network's
fused predictions. Without the native guard, 2146 ICDAR words and 3082 Total-Text
words touch fused components at native256; maintaining the guard is necessary.

At native256 the foreground occupies 496,913 / 36,483,278 valid pixels
(1.3620%) in ICDAR2015, versus 5,421,263 / 59,135,295 (9.1676%) in Total-Text.
This is a descriptive TRAIN population comparison, with different datasets and
target semantics; it does not establish that balanced loss will improve F1.

| Dataset / target size | Lost words | Fragmented words | Fused components |
| --- | ---: | ---: | ---: |
| ICDAR2015 source | 0 | 87 | 0 |
| ICDAR2015 native256 | 4 | 7 | 0 |
| ICDAR2015 native384 | 2 | 6 | 0 |
| ICDAR2015 native512 | 2 | 6 | 0 |
| Total-Text source | 0 | 18 | 0 |
| Total-Text native256 | 1 | 6 | 0 |
| Total-Text native384 | 0 | 6 | 0 |
| Total-Text native512 | 0 | 6 | 0 |

Source fragmentation includes small pieces of an owner after spatial separation;
these counts are not removed from denominators. The inspected RGB/owner preview
includes ICDAR TRAIN197/213 and Total-Text TRAIN40/149. It visibly confirms
separate word regions and much smaller ICDAR text. The preview uses PIL drawing
because Matplotlib is absent from the qualified environment; no package was
installed and no target values were changed.

Frozen source weights, common original-coordinate validation references:

| RGB / native shape | Matched / care words | Proposals | Geometry F1 | Macro pixel IoU |
| --- | ---: | ---: | ---: | ---: |
| 512 square / 256 square | 173 / 430 | 518 | 0.364979 | 0.544884 |
| 768 square / 384 square | 248 / 430 | 776 | 0.411277 | 0.589507 |
| 1024 square / 512 square | 265 / 430 | 887 | 0.402430 | 0.586630 |
| 576x1024 / 288x512 | 264 / 430 | 914 | 0.392857 | 0.583991 |

The predeclared selector chooses **768 square**. Higher recall at 1024 is
offset by more false proposals. The historical native-envelope validation F1
0.407173 is a different reference geometry; use the reevaluated 0.364979 for
these scale comparisons. These validation scores are geometry only, before OCR.

Two actual backward/optimizer steps at RGB768, batch8, FP32 passed for both
standard BCE/BatchNorm and word loss/GroupNorm. Peaks were 8,864,066,048 and
8,864,012,288 allocated bytes respectively, with 11,884,560,384 reserved bytes
in each smoke. No OOM occurred. Both smokes include validation, selected-state
checkpoint reload and deterministic validation replay; they are partial runs.

`scripts/audit_icdar2015_improvement_target_geometry.py` also replays perfect
native targets against original-coordinate validation boxes: matched care words
are 411/ 430 at 512 RGB, 429/ 430 at 768 and 430/ 430 at both 1024 shapes. Resulting
geometric F1 is 0.952491, 0.996516 and 1.0 respectively. This is a GT-assisted
target diagnostic, not network inference, a deployable score or an upper bound
on all possible network outputs. It does not change the frozen resolution or
branch selection. Separate words are learnable from these qualified targets;
the network remains far from perfect-target replay.

## OCR runtime qualification

A complete validation100 / 576-crop CPU PARSeq qualification on the completed
512 control passes the independent CSV/SciPy audit: 170 correct alphanumeric
spatial word matches, word F1 0.337972, geometry F1 0.433400. This auxiliary CPU
execution does not select the final branch. All selection runs will use the
same GPU device, and device differences will be checked separately.

| CPU qualification, fixed validation100 | Correct words / 430 | Predictions | Symbol-free word F1 |
| --- | ---: | ---: | ---: |
| Source epoch38, RGB512 | 131 | 518 | 0.276371 |
| Source epoch38, RGB768 | 192 | 776 | 0.318408 |
| 512 continuation, selected epoch12 | 170 | 576 | 0.337972 |
| 768 BCE continuation, selected epoch 18 | 237 | 722 | 0.411458 |

All four CPU results pass independent CSV/SciPy auditing. These are validation
results, not TEST500 gains. The first two completed continuations also have
identical sample order in all 20 epochs, unchanged encoder tensors/buffers and
checkpoint optimizer learning rate 1e-5. Fifty focused tests pass, including
the count-replay regression and explicit OOM propagation tests. Historical
47-test XML and failed-reader source bytes remain preserved separately.

The first OCR reader failed before recognition because it expected stored
proposal boxes where frozen geometry rows contain proposal counts. The fixed
reader reconstructs boxes from checksum-pinned native maps and verifies all
stored proposal/GT/match counts. The failed output is preserved; corrected CPU
qualification is in `qualification-validation-ocr-cpu-v2`. Training source
contracts, masks, weights and geometric scores are unchanged. A regression test
covers the count-based replay; separate simulated OOM tests verify failure
records and error propagation without fallback.

The older post-hoc `icdar2015-transfer-diagnosis-v1/diagnostic_summary.json`
had overwritten flat matched/recognized fields in its size summary. Its new
`resolution_groups_corrected.json` preserves nested groups regenerated from
the original saved polygons/matches: below four native pixels, 627 GT words,
173 geometry matches and 103 correct words; at least six pixels, 796/440/329.
Original results and the historical diagnostic are preserved.

## Execution and evidence

All data, source checkpoints and result roots are explicit CLI arguments.
`scripts/audit_icdar2015_word_masks.py` audits targets;
`scripts/screen_icdar2015_unet_resolutions.py` screens resolution on validation.
`scripts/train_icdar2015_unet_improvement.py` trains each predeclared branch.
`scripts/run_icdar2015_improvement_evaluation.py` can wait for the verified
training supervisor process to exit, then requires all four completed runs.
It evaluates both source baselines and four branches on the same GPU, audits
each, checks CPU/GPU recognition differences, freezes validation selection,
runs the selected TEST500 evaluation and independent audit, and reopens the
completed cache. No TEST result participates in the branch selection.
Resolved profiles, source hashes, per-sample masks/geometry and GPU allocated
and reserved peaks are saved outside Git. Actual PAPER hardware is checked
before each run. No automatic batch reduction, mixed precision or model
substitution is permitted after an error. Historical code and strict profiles
remain separate. Training, selection, the complete TEST500 evaluation and
independent audits have completed successfully; the results follow.

## Completed matched training and GPU validation

All four branches completed 20 continuation epochs on FIT900: 18,000 image
passes / 2,260 optimizer steps per branch, 72,000 image passes in total. The
saved integrity audit verifies identical sample order in every matched epoch,
identical scientific source hashes across branches, unchanged encoder tensors
and buffers, Adam learning rate 1e-5, and the explicitly enumerated 63 buffer
removals in the 21-layer GroupNorm migration. No OOM occurred. Training peaks
were 4,220,307,968 allocated bytes for 512 and approximately 8,864,066,048 for 768.

| Branch | Selected continuation epoch | Geometry F1 | Symbol-free word F1 |
| --- | ---: | ---: | ---: |
| 512 / separator BCE / BatchNorm | 12 | 0.433400 | 0.337972 |
| **768 / separator BCE / BatchNorm** | **18** | **0.491319** | **0.411458** |
| 768 / word-balanced BCE + local Dice / BatchNorm | 0 | 0.411277 | 0.318408 |
| 768 / word-balanced BCE + local Dice / GroupNorm | 20 | 0.168872 | 0.126024 |

All entries use the same 100 validation images / 430 care words and GPU PARSeq.
All six baseline/branch validation reports pass independent CSV/SciPy audits.
For the four auxiliary CPU qualifications there are zero changed OCR texts and
identical aggregate metrics on GPU. Epoch 0 in the word-balanced branch means
that none of its 20 trained states improved its initial geometric selector;
all 100 retained probability and prediction arrays exactly match source 768.
The final trained state is preserved separately. GroupNorm's selected epoch 20
wins the declared geometry tie breaks over epoch 15.

The selected model is **768 RGB / 384 native, separator-weighted BCE,
BatchNorm, continuation epoch 18 (source-equivalent epoch 56)**. Selection was
frozen at `2026-10-26T12:20:34.931516+00:00` in `selection.json`, before TEST
inference. Its checkpoint is external:

`/extra1/bmartelli/context-str-runs/icdar2015-unet-improvement-v1/larger_bce/best.pt`

SHA256: `79efc8f7f44c719be92c6d8b2a69ce9f204030ff77e2fa05194ac191fa10fe91`.

Balanced loss increases coverage but also word fusion and false proposals under
the frozen 0.5 cutoff. For example, its epoch 16 touches 234 GT words with fused
components, versus 55 for selected BCE 768. These trials do not establish global
inferiority of every balanced loss or GroupNorm configuration; they establish
that the tested continuations do not beat the larger-input BCE candidate.

Portable stage receipts are
`provenance/ICDAR2015_IMPROVEMENT_PROTOCOL_V1.json` and
`provenance/ICDAR2015_IMPROVEMENT_QUALIFICATION_V2.json`. The former describes
initial qualification before the OCR reader correction. Final evidence is in
`provenance/ICDAR2015_IMPROVEMENT_COMPLETION_V1.json` and the independently
audited TEST500 report.

## Complete TEST500 result

The validation-selected model was evaluated on all 500 official TEST images,
2077 care words and 3153 ignored GT instances. All predictions count, including
blank normalized strings and predictions inside ignored GT regions. The
original normalization, matching, thresholds and PARSeq artifacts are unchanged.
This is standalone U-Net/PARSeq **approximate reproduction**, not official
polygon Hmean or a strict paper Table 3/5 result. TEST had previously been inspected;
the new result is exploratory improvement under the preserved protocol.

| Model | Correct words | Predictions | Word precision | Word recall | Symbol-free word F1 | Geometry F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Preserved 512 baseline | 639 | 2507 | 0.254886 | 0.307655 | 0.278796 | 0.383944 |
| **Selected 768 BCE** | **1068** | **3164** | **0.337547** | **0.514203** | **0.407556** | **0.507155** |

The word F1 gain is **12.876 percentage points**, with **429 additional correct
words**. All500 samples remain in the denominator; failed samples=0. There are
189 empty normalized predictions in the new output, all retained in precision.
The punctuation-preserving casefold control has 1016 TP and F1 0.387712.

The independent audit reconstructs CSV care GT, native threshold/canvas,
8-connected components, inverse coordinates, all 3164 original-RGB crop hashes,
cached OCR texts and maximum bipartite matches using SciPy. It passes for all 500
images. Reopening the completed evaluation also passes and preserves every
scientific output. TEST peak allocated memory is 650,522,112 bytes for U-Net
(FP32, batch 1, RGB 768/native 384) and 205,955,584 bytes for sequential PARSeq
(FP32, batch 32); reserved peaks are 731,906,048 and 245,366,784 bytes. No OOM occurs.

External final evidence:

- `selected-test500/results.json`, SHA256
  `b40cbab72d3f5a9a4c57a05faae7278e59bcae4a3dee693361597e51e8e73061`.
- `selected-test500/independent_audit.json`, all 500 images / 3164 crops verified.
- `training_integrity_audit.json`, all four 20-epoch continuations verified.
- `comparison.json`, separate validation and TEST projections with original report hashes.
- `validation-prediction-comparison.png`, two largest validation gains and the
  largest remaining loss in correct-word TP; inspected alongside RGB care GT.

These paths are relative to the explicit run root
`/extra1/bmartelli/context-str-runs/icdar2015-unet-improvement-v1/`.
The selected checkpoint remains at the
path above. The historical 100-epoch training, TEST report, masks, checkpoints
and failed-reader artifact are preserved. Scientific source hashes from the
historical training also remain unchanged. The improvement comes from the
tested larger-input BCE continuation; the tested balanced-loss and GroupNorm
branches are retained as negative results.

Implementation entry points are `models/icdar2015_improvements.py`,
`segmentation/icdar2015_improvements.py` (paired geometry, ownership and loss),
and `segmentation/icdar2015_improvement_runtime.py` (qualification, training
and metric replay). Evaluation uses
`scripts/evaluate_icdar2015_improvement_validation.py`,
`scripts/select_icdar2015_improvement.py`,
`scripts/evaluate_icdar2015_improvement_test.py` and
`scripts/audit_icdar2015_improvement_word_f1.py`. The inspected prediction
figure is rendered by `scripts/preview_icdar2015_improvement_predictions.py`.
Focused new tests are `tests/segmentation/test_icdar2015_improvement_targets.py`,
`tests/models/test_icdar2015_improvements.py`,
`tests/scripts/test_icdar2015_improvement_selection.py`,
`tests/scripts/test_icdar2015_improvement_evaluation.py` and
`tests/scripts/test_icdar2015_improvement_failures.py`; the 50-test run also
covers the preserved transfer, word metric, independent matcher and PARSeq adapter.
