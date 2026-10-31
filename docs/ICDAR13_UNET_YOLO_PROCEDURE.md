# ICDAR2013 U-Net: matched YOLO selection procedure

**Approximate reproduction / active experiment.** The user requests retraining
needed to apply the same procedure used for the ICDAR2013 YOLO result to a
U-Net. Preserve the previously qualified YOLO, ICDAR2015 and Total-Text results;
use full ICDAR2013 TEST with exactly 61/233 (26.18%) DeepSolo routing decisions.
The metric is per-image Unicode-casefolded/alphanumeric lexical multiset micro
F1, native256 care/ignore, including all images and care words; it is not paper
accuracy or official polygon Hmean.

Use the independently qualified 30-epoch text-encoder U-Net and its existing
frozen word-localization policy as the full-FIT deployment expert. For the
router's training observations, train five independent decoder fits from the
same original initialization on 1,130 Total-Text images plus the corresponding
146/147 ICDAR2013 FIT images. The same YOLO RGB-group folds hold out 36/37 IC13
FIT images per model. Freeze the epoch at 30, inherited from the already selected
full-FIT U-Net; the fold's heldout images never enter decoder fitting or
checkpoint monitoring. Existing 46 validation images monitor the fold, but do
not choose its checkpoint. Then apply the existing finite emission grid,
word-risk gates, optional conditional fusion and no-extra-neural selection rule.

**Material boundary:** the text-pretrained encoder has known prior ICDAR2013
TRAIN exposure, including the heldout folds and validation. Thus these are
conditional decoder-heldout predictions, not independently pretrained expert
OOF outputs. The original full model's checkpoint and localization policy were
already selected in a historically adaptive research sequence. Benchmark
vocabulary assistance, repeated TEST exploration and the conditional/non-nested
router remain; no pristine holdout or exact paper reproduction claim follows.

The initial five-fold planning and hardware qualification are in progress. No
new TEST score is claimed until the complete frozen router and full TEST recount
and independent audit are finished.

The real decoder-backward smoke and its independent training audit pass. The
full-FIT source export also passes independent replay of 229 images, 950 real
PARSeq crop hashes and 229 predicted-map summaries. Five full decoder folds
are prepared; the sequential fail-fast supervisor is running the first fold
and will audit/export each fold before proceeding. No new TEST score is
available yet. External root:
`/extra1/bmartelli/context-str-runs/icdar13-unet-yolo-procedure-20261030-v1`.

Implementation: `scripts/prepare_ic13_unet_oof.py`,
`scripts/supervise_ic13_unet_oof.py`, `scripts/export_ic13_unet_expert.py`,
`scripts/audit_ic13_unet_expert.py`, `recognition/unet_expert_records.py`.
Configurations: `configs/approximate/icdar13_unet_yolo_oof_v1.json` and
`configs/approximate/icdar13_unet_yolo_supervisor_v1.json`.
