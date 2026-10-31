# ICDAR2015 target70: hard-negative loss and complete encoder adaptation

This is **approximate reproduction**, not strict paper reproduction. The user
target remains at least70% correct-word F1 on complete TEST500 under the unchanged
spatial casefold/alphanumeric metric. The preceding target70 comparison reached
54.7782% emitted / 43.5836% raw TEST500 F1; see
[the completed result](ICDAR2015_TARGET70.md). No OHEM model has been selected
using TEST or evaluated on TEST.

## Evidence for the next comparison

Both first-stage100-epoch branches completed successfully on the unchanged
FIT900/VAL100 split. At the selected standard-policy checkpoints, long768 has
248/430 correctly localized/read words,743 predictions, raw F1 42.2847%, emitted
F1 60.3306%. adapt1280 has245/430,557 predictions, raw F1 49.6454%, emitted
F1 61.3333%. After the identical20-case validation operating grid, emitted F1 is
60.3535% and61.7608%, respectively. Resolution/adaptation mainly reduced false
positives; it did not increase raw correct-word recall. The native checkpoint
still misses148/430 care words geometrically. Geometry has 37 more matches than
correct-word matching; these are independent maximum matchings, so this is not
an instance-level error partition. No denominator was removed.

The authoritative local paper sections3.2/Figure3 and4.1 and current training,
mask, encoder-gradient and loss implementations were inspected. The paper uses
BCE/Adam and does not specify hard-negative mining. Existing local
`segmentation/totaltext_word_ablation.py` already offers a per-word balanced loss
with hard negatives; its equal positive/negative terms differ from this new loss.
The official [DBNet loss source](https://raw.githubusercontent.com/MhLiao/DB/65ca77a0bcfbd7114b916cf8a1e9ca85114286ce/decoders/balance_cross_entropy_loss.py)
uses a3:1 hard-negative ratio. This is motivation only: no external DBNet model,
weights or loss implementation enters the runtime. Our exact code/config hashes
are recorded; per-image normalization, source pixel weights and the negative-only
rule are explicit assumptions, not claims of reproducing DBNet.

## Predeclared experiment

Profile: `configs/approximate/icdar2015_target70_ohem_v1.json`.
Both arms start from completed adapt1280's best epoch50, source effective
epoch106, checkpoint SHA256
`c4676914dbcbf559a26fb98e51f5ae137b2a3ac66930d0191fca5b214fce5eb6`.
Original source weights/reports and the40.7556% TEST baseline stay preserved.

| Arm | Encoder feature convolutions | Loss | RGB/native | Physical/accumulation |
| --- | --- | --- | --- | --- |
| ohem_partial | layer3/layer4 | weighted hard-negative BCE |736x1280 /368x640 |4/2 |
| ohem_full | all feature stages | identical loss |736x1280 /368x640 |4/2 |

Keep all encoder BN affine tensors/statistics and unused ResNet classification
FC frozen. Decoder BN still sees physical batch4. Strictly copy every source
state tensor without adding/removing keys. The new architecture identity records
the training/gradient regime; the single-output forward architecture is retained.
Confirm identical source predictions and actual gradients in every enabled stage.

Restore both inherited Adam groups, every moment and RNG state. Explicitly
restart the100-epoch cosine schedule: decoder1e-5 to1e-6, inherited encoder1e-6
to1e-7. Newly enabled early encoder convolutions receive a fresh Adam group at
1e-6 to1e-7. Weight decay.01, betas(.9,.999), epsilon1e-8 and gradient clip norm1
remain fixed. This LR restart and100 additional epochs are ASSUMPTION choices.
Use FP32; no automatic batch/precision substitution after OOM. Qualify actual
full-size backward allocated/reserved peaks before either full arm starts.

## Exact supervision and checkpoint selection

For each image, include every positive valid pixel. Rank valid negative pixels
by separator/ignored-region-weighted BCE and select at most floor(3*Npositive)
hardest negatives. Keep source separator and ignored-negative weights3. Divide
the selected weighted loss by selected pixel weights, then average images.
All-negative images use every valid negative; images without valid pixels fail.
Padding/rotation fill has zero gradient. Nonfinite logits/loss/gradients fail.
Log positive, available-negative and selected-negative pixel counts.

Only unused training owner-map replay is omitted; the base RGB/mask/validity/
separator tensors and exact paired augmentation remain identical. Check used
tensors and RNG states against the existing owner wrapper. Validation retains
all owner-map topology diagnostics. No target file, care GT or ignored-region
evaluation rule changes.

The new checkpoint priority is raw symbol-free spatial correct-word F1, then
emitted F1, geometry F1 and earliest epoch. This explicitly focuses on learned
U-Net output quality. Initial source epoch zero remains eligible. Evaluate fixed
CPU PARSeq at epoch zero, every5 epochs, new geometry bests and the final epoch;
retain all raw outputs, per-image counts and the same predeclared confidence
grid as separately reported calibration. Every epoch, threshold and branch
decision uses the original VAL100 only. Freeze eventual weights/output policy
before TEST; all2077 original care words and every emitted prediction must count.

See `ICDAR2015-TARGET70-OHEM-001` in the ambiguity register. These extensions
must not mutate STRICT settings or be presented as an achieved70% result.

## Qualification and validation evidence

Eighteen focused tests passed, including actual enabled-stage gradients, frozen
encoder BN, inherited Adam moments, identical used augmentation tensors/RNG and
hard-negative selection. Both FP32 full-size backward smokes passed on the
RTX3090: partial encoder allocated 9,992,354,304 / reserved 13,180,600,320 bytes;
full encoder allocated 14,253,846,016 / reserved 17,498,636,288 bytes. Inputs were
4x3x736x1280, outputs 4x1x368x640, accumulation 2, with no OOM. Source epoch-zero
full VAL100 raw metrics exactly matched the qualified source. See
`provenance/ICDAR2015_TARGET70_OHEM_QUALIFICATION_V1.json`.

Both arms completed all 100 epochs and retained epoch zero: raw word F1
49.6454%, emitted F1 61.3333%. No learned improvement was obtained. Every epoch
contains all900 fitting images. Both selected state dictionaries are bitwise
equal to source `c4676914dbcbf559a26fb98e51f5ae137b2a3ac66930d0191fca5b214fce5eb6`;
all100 selected validation probability maps and binary masks also match that
source exactly. Therefore no duplicate TEST inference is warranted. The full
arm peaked at14,256,860,672 allocated /17,498,636,288 reserved bytes, without OOM.
See `provenance/ICDAR2015_TARGET70_OHEM_COMPLETION_V1.json`. The full-arm
operating grid was still running when that receipt was created; its completion
must be checked separately before claiming a complete operating comparison.

The separate `icdar2015_ohem_threshold_screen_v1.json` is an ASSUMPTION diagnostic
on saved complete VAL100 maps: partial epochs 0/20/50/100 and full epochs
0/20/50/65, thresholds 0.5/0.65/0.8/0.9/0.95/0.98. Every component and all 430 care
words remain. Higher thresholds do not recover the initial geometry F1 of
57.1429%. Geometry is not correct-word F1; no TEST data or new training setting
entered this screen.
