# Matched training of CLIP, BLIP and combined image routers

Date: 2026-10-27. Approximate reproduction; a trained-router extension rather
than the paper's training-free caption/MPNet/lexical scoring path. STRICT
configuration and scientific components are unchanged.

## Predeclared comparison

The frozen corrected U-Net/PARSeq/GNN/verifier/byte-editor expert and native
DeepSolo profile are identical across arms. All comparisons use symbol-free
lexical micro F1 with the original native 256 care/ignore handling, as specified
in [the corrected-token source](UNET_CORRECTED_TOKEN_ROUTER.md). No GT or actual
DeepSolo output enters any inference observation.

Four matched heads use 30 cheap OCR/mask scalars and every predicted word as
UTF8 bytes plus 12 geometry/confidence descriptors. A learned byte embedding
and convolution produce word tokens. Two blocks apply context cross-attention,
word self-attention and feed-forward updates. A decision token, pooled words
and the 30 scalars feed a 256-wide two-layer MLP predicting paired differences
in TP and emitted care-word counts. The training loss combines SmoothL 1 count
regression with cost-weighted pair ranking. Shared parameters initialize
identically in all arms; optimizer, sample order and update counts are matched.

| Arm | Additional frozen evidence | Incremental encoder tensor bytes |
| --- | --- | ---: |
|none|No foundation encoder; learned OCR bytes and predicted geometry/maps|0|
|clip|196 global CLIP patches plus current OCR-word text/crop embeddings|598485140|
|blip|32 raw 768-dimensional BLIP2 visual QFormer queries|4364457984|
|both|The same CLIP and BLIP evidence together|4962943124|

CLIP uses one original global view, without the historical four tile forwards.
BLIP uses original RGB visual queries; captions, language decoding, transcript
likelihoods and erased views are excluded. The BLIP visual-only implementation
explicitly constructs only vision, QFormer and query tokens. Its native subset
is loaded with strict key matching and no precision conversion. This qualifies
a distinct, documented router implementation, not a change to the strict paper
captioning component. Source checkpoints still reside in their complete shards;
omitting unused tensors from a load is distinct from deleting those source files.

The 1128 U-Net-gradient TRAIN images are used for head fitting. U-Net-gradient
holdout 125 is split deterministically into 62 model-selection and 63 calibration
images, identical to [the budget study](ESSENTIAL_ROUTER_BUDGET50.md). This group
was already used for upstream checkpoint selection; it is not an untouched
validation experiment for the complete correction chain. Every
usable TRAIN image has an assigned role; fitting all 1253 would eliminate the
held-out data needed for the user's validation-based cutoff. Thirty epochs,
batch 16, AdamW learning rate 0.0003, decay 0.01, gradient clip 5 and fixed seed
20261027 are recorded in the resolved configuration. Feature normalization and
paired-target normalization are fitted on 1128 only. No byte or word truncation
is allowed. Native confidence remains inherited from the original PARSeq reading
after a corrected word edit; it is not claimed to be recalibrated confidence.

For each arm, choose the earliest best checkpoint among epochs 10/20/30 using
selection 62 micro F1 at 50% light acceptance. Among arms within 0.25 F1 points of
the best, prefer fewer additional frozen-encoder plus head tensor bytes. The
63 CAL images set 45/50/55% light-acceptance score quantiles for those exact
checkpoints, with no refit. Checkpoints and all cuts are frozen to JSON before
loading TEST features or outcomes. TEST exact 50% ranking is a descriptive
budget control, never an encoder-selection criterion.

## Evidence and implementation

External root: `/extra1/bmartelli/context-str-runs/encoder-family-router-20261027-v1`.
The predeclared configuration, training-source snapshot and all original smoke
artifacts remain preserved. Full results and qualification are reported below
with their independent audits.

Implementation: `recognition/encoder_family_router.py`,
`recognition/encoder_family_inference.py`, `recognition/blip_visual_only.py`,
`scripts/run_encoder_family_router.py`, `scripts/audit_encoder_family_router.py`,
`scripts/analyze_encoder_family_signal.py`, `scripts/qualify_blip_visual_only.py`.
Configuration: `configs/approximate/encoder_family_router_v1.json`.
Tests: `tests/recognition/test_encoder_family_router.py`,
`tests/recognition/test_encoder_family_inference.py`,
`tests/recognition/test_blip_visual_only.py`.

Paper 79.8 remains a nominal reference: recognition accuracy is not operationally
defined as our lexical F1. The paper's TotalText fallback count is unavailable.
All router results must include logical DeepSolo images/total and percent. No
online DeepSolo inference or end-to-end speedup is implied by cached routing.
Single-seed variance, historical TEST/model-selection exposure and the small
validation population limit claims of a generally sufficient encoder signal.

## Completed full experiment

All four arms fitted 1128 TRAIN images for 30 epochs, with checkpoints 10/20/30, selection 62 and CAL 63. All 300 TEST images were retained. The primary encoder arm selected before TEST was **CLIP**, because its selection F1 exceeded the others and combined context gave no useful validation gain at its higher cost. The predeclared primary acceptance remains 50%; failed transfer results are preserved.

| Arm | Selected epoch | Selection F1 % | DeepSolo on selection, images/total (%) | Head parameters | Head tensor MB | Encoder tensor GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
|none|10|79.8736|31/62 (50.00%)|570654|2.282616|0.000000|
|clip|10|80.6723|31/62 (50.00%)|768030|3.072120|0.598485|
|blip|30|80.0420|31/62 (50.00%)|673182|2.692728|4.364458|
|both|10|80.6584|31/62 (50.00%)|870558|3.482232|4.962943|

Every primary CAL cut sends 31/63 images to DeepSolo (49.21%). The cuts are tied to the exact frozen weights.

| Arm | Fixed primary score cut | TEST F1 % | DeepSolo on TEST, images/total (%) |
| --- | ---: | ---: | ---: |
|none|-0.2400110311|79.7791|175/300 (58.33%)|
|clip|-1.3266125035|77.7061|131/300 (43.67%)|
|blip|-1.0229436107|77.0893|140/300 (46.67%)|
|both|-1.6433919721|76.8717|121/300 (40.33%)|

The selected CLIP primary misses both the nominal 79.8 target and the 45–55% fallback range. BLIP and combined context also miss the accuracy target. The acceptance-quantile cut does not enforce an identical TEST budget. There is no TEST retuning or promotion of another control.

## Equal-budget signal control

| Gate | TEST F1 % | DeepSolo images/total (%) | Additional router weights |
| --- | ---: | ---: | ---: |
|Frozen essential OCR18+mask12 Ridge|79.9904|150/300 (50.00%)|728 numerical bytes;28146-byte deployment|
|none|78.7777|150/300 (50.00%)|0.002283096 GB, including 480 normalization bytes|
|clip|78.6055|150/300 (50.00%)|0.601557740 GB, including 480 normalization bytes|
|blip|77.7299|150/300 (50.00%)|4.367151192 GB, including 480 normalization bytes|
|both|78.0812|150/300 (50.00%)|4.966425836 GB, including 480 normalization bytes|

The practical essential baseline has a different, already frozen 1028-image Ridge fit. It had selection F1 81.3347%, DeepSolo 31/62 (50.00%), before this encoder experiment; it is not a newly TEST-selected head. The four neural arms isolate encoder evidence under a matched protocol, while their comparison with Ridge is a practical comparison rather than a causal isolation of head capacity.

The earlier essential primary remains F1 80.0383%, DeepSolo 151/300 (50.33%), with a cutoff fixed on CAL 63. It reaches the nominal 79.8 target near half fallback. It uses no CLIP or BLIP and is preserved rather than replaced by the worse neural variants.

At exactly half fallback, combined minus CLIP F1 is−0.5243 points, paired 95% interval[−1.8665,+0.8831], both DeepSolo 150/300 (50.00%). On selection the difference is−0.0138 points, interval[−3.3980,+3.1522], both DeepSolo 31/62 (50.00%). These conditional exploratory intervals cross zero; there is no demonstrated gain from adding BLIP. Single-seed fixed-head intervals do not prove that context is universally useless.

The fixed-cut transfer and ranking problems are separate: the CLIP cut sends only 131/300 images to DeepSolo (43.67%) and scores 77.7061%, but even enforcing 150/300 (50.00%) yields only 78.6055%. A larger head therefore does not solve this study by calibration alone. CLIP and combined training losses keep falling while selection F1 worsens after epoch 10; this is consistent with overfitting, not evidence that more epochs will recover the target.

## All predeclared fixed-cut controls

| Arm | CAL light acceptance target | Fixed cut | TEST F1 % | DeepSolo images/total (%) |
| --- | ---: | ---: | ---: | ---: |
|none|45%|-0.1144111609|80.0672|193/300 (64.33%)|
|none|50%|-0.2400110311|79.7791|175/300 (58.33%)|
|none|55%|-0.5220576651|78.5439|148/300 (49.33%)|
|clip|45%|-1.0156348131|78.6072|153/300 (51.00%)|
|clip|50%|-1.3266125035|77.7061|131/300 (43.67%)|
|clip|55%|-1.6689430225|77.5647|112/300 (37.33%)|
|blip|45%|-0.8381655292|77.4580|149/300 (49.67%)|
|blip|50%|-1.0229436107|77.0893|140/300 (46.67%)|
|blip|55%|-1.3608521251|76.8161|126/300 (42.00%)|
|both|45%|-1.0343350234|78.3373|157/300 (52.33%)|
|both|50%|-1.6433919721|76.8717|121/300 (40.33%)|
|both|55%|-1.7508191859|76.6556|115/300 (38.33%)|

## Visual-only BLIP qualification and failures

The strict native visual subset contains 4364457984 FP32 tensor bytes, omitting T 5 and the unused language projection. All three authenticated TRAIN images produce query arrays **bit-identical** to the qualified GPU caches (maximum absolute error 0), without constructing any language model. No precision, tolerance, checkpoint or preprocessing parameter was changed to obtain this result.

GPU: NVIDIA GeForce RTX 3090, batch 1, original 224×224 RGB processor inputs,32×768 queries. Peak allocated 4515573760 bytes and reserved 4603248640 bytes. No OOM occurred. These measurements qualify the visual module alone, not simultaneous U-Net/PARSeq/DeepSolo execution or an 8 GB full pipeline. Complete source checkpoint shards remain on disk; the 4.36 GB figure is the loaded tensor subset rather than a claim that the original files shrink.

Two earlier failures are preserved. The first GPU attempt stopped before loading because the predeclared 6 GiB free-memory guard was not met by the shared GPU. Explicit CPU FP32 qualification then failed the original 1e-5 absolute tolerance: maximum discrepancy 0.0009799004. CPU equivalence is **not qualified**; the threshold was not relaxed. The later GPU qualification passed at the same tolerance with exact equality.

Training the cached-feature heads used only the trainable heads on GPU: peak allocated 348256768–362611712 bytes, reserved 694157312–744488960 bytes, FP32 batch 16; all arms completed without OOM. This excludes frozen encoder inference, shared experts, framework overhead and other users of the GPU.

## Independent verification and reproducibility

Independent audit **PASS**:3124 authenticated source files, disjoint 1553 sample roles, FIT-only statistics,4 earliest-best checkpoint selections,4 selection-score/rank replays,12 calibration cut/count replays,12 TEST fixed-cut count replays and 52 quota points. CPU head replay checked 48 TEST observations and 144 decisions against the GPU run; maximum score discrepancy 3.33964081e-06. This small CPU head replay is distinct from the failed CPU BLIP-encoder equivalence test.

Fifty focused tests passed, covering strict BLIP subset loading and missing-key rejection, matched shared initialization, inactive encoder boundaries, empty words/padding, complete UTF8 bytes, finite gradients, single-image feature interfaces, deterministic cutoff ties, coverage and source metrics. Full results SHA256: `f121b8ae792922dcb69acfc8594722f1245e3331be7a38a40da868c671917bbb`.

The run started from the live working tree; its scientific execution files were preserved in `source_snapshot_v1` before execution and verified unchanged afterward. Later support/audit interfaces are preserved separately. The initial parameter-only CLIP cost in the prototype declaration was corrected before the full run to include buffers; the full resolved configuration records this accounting. No training observation or model output changed for that correction.

Files: `full_v1/results.json`, `frozen_choices.json`, `test_scores.json`, `training_history.json`, `training_memory.json`, `independent_audit.json`, `signal_comparison.json`; root-level native GPU/CPU qualification receipts and failed initial guard logs remain available. New DeepSolo inference images 0; all recognition routing is logical cached-output selection.
