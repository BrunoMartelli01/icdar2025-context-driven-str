# ICDAR2015: confidence seeds for a frozen single-output U-Net

This is **approximate reproduction**, not a strict paper result. Paper3.3 and
existing localization/word-instance code were inspected before implementation.
The qualified c467 model stays unchanged. Both support and seeds come from its
same foreground probability; there is no new trained kernel head. Reuse the
existing GT-blind minimax partition, retain unseeded components, and remove
nonseed contacts to preserve eight-connected gaps. Keep every resulting
component, including singletons/fragments, and every original care word.

The geometry-only profile predeclares supports0.2/0.35/0.5 and seeds0.5/0.65/0.8,
all9 cases on fixed VAL100/430care. Equal0.5 thresholds exactly replay the
source binary masks and geometry. The best geometry F1 remains57.1429%, so
the original0.62 resource gate was not met and its immutable report keeps
`run_full_ocr=false`. However geometric matches increase from282 to307, with
extra noise. A separately configured, explicitly exploratory follow-up therefore
evaluated frozen PARSeq on ALL9 cases, because confidence rejection can improve
the actual word metric even when geometry F1 declines. No TEST data was read.

| Support / seed | Raw word F1 | Emitted word F1 |
| --- | ---: | ---: |
|0.2 /0.5 |44.7602% |62.9921% |
|0.2 /0.65 |44.9477% |63.2944% |
|0.2 /0.8 |43.9541% |60.9023% |
|0.35 /0.5 |46.2107% |61.4796% |
|0.35 /0.65 |45.1383% |60.9787% |
|0.35 /0.8 |44.3447% |60.0000% |
|0.5 /0.5 |49.6454% |61.3333% |
|0.5 /0.65 |46.5909% |60.6142% |
|0.5 /0.8 |45.3686% |58.8079% |

The selected validation case uses support0.2/seed0.65, axis0 crops, sequence
confidence>=0.7 and normalized length>=3. Raw:258TP,460FP,172FN,718 predictions.
Emitted:244TP,97FP,186FN,341 predictions, precision71.5543%, recall56.7442%.
All430 care words remain. The previously selected native operating policy had
61.7608% emitted F1 (235TP,96FP,195FN); crop/confidence policies differ. The
new gain is from mask postprocessing/output selection, not newly trained neural
weights. Its raw F1 is lower despite13 more correct raw words. It has not been
evaluated on TEST and does not establish the70% objective.

Seven focused separation tests passed, covering exact baseline replay, fused
support, protected seeds, diagonal connectivity, fallback singletons, empty
predictions and canvas clipping. Independent SciPy matching and independent
confidence-membership replay verified raw/emitted counts and all three matching
modes for every case. This replay consumes saved GT/OCR; it does not independently
parse CSV annotations or repeat inference. Source RGB, mask, crop and OCR hashes,
all per-image outputs and all calibration policies remain saved externally.

See `configs/approximate/icdar2015_seed_partition_probe_v1.json`, the distinct
`icdar2015_seed_partition_word_probe_v1.json`, ambiguity
`ICDAR2015-SEED-PARTITION-001`, and
`provenance/ICDAR2015_SEED_PARTITION_WORD_PROBE_V1.json`. Threshold partition,
boundary handling, resource gate and adaptive word follow-up are ASSUMPTION
choices; fixed source/split/metric are REPOSITORY_SPECIFIED. Repeated validation
exploration is a selection-variance limitation. STRICT is unchanged.
