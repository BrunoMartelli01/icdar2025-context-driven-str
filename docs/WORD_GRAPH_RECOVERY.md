# Word hypotheses, complete crops, geometric decoders and regional distillation

**Approximate reproduction / user-authorized research extensions.**
These are four isolated experiments, not an exact reproduction of Das or a
replacement for the STRICT paper configuration. No model is promoted by launch,
successful execution, decreasing training loss, or a favorable smoke metric.

## Evidence and changes relative to earlier negative tests

Read the authoritative `papers/2503.15639v2.pdf`, sections3.2–3.3, and Das et al.,
*Soft set-based MSER end-to-end system for occluded scene text detection,
recognition and prediction*, KBS305(2024)112593, sections3.2 and3.5. The original
paper specifies attention U-Net and connected-component localization. Das
describes classification/distance/direction priors, iterative contour GCNs and
teacher training. Its complete geometry and teacher recipe remains unresolved.
Related primary sources: [TextBPN](https://arxiv.org/abs/2107.12664),
[Mean Teacher](https://arxiv.org/abs/1703.01780),
[SemiETS](https://arxiv.org/abs/2504.09966).

Earlier frozen-teacher GT/pixel/profile ablations, EMA consistency, set-loss
selection, geometric heads, small crop-completeness MLPs, and auxiliary CTC
training are existing controls, not relabeled new methods. The latest failure
diagnosis established score-band transfer failure, destructive transitive merges,
and better held-out point loss with worse crop/OCR outcomes. The new factors are
locked-output incremental selection, full-owner extent supervision for the GCN,
a genuinely U-Net-free inference decoder, and verified regional Dice/extrema
distillation. The current task authorizes implementation of all four.

## 1. Locked-output word hypothesis graph

Freeze source U-Net, original candidate pool, pre-OCR ranking, PARSeq and every
accepted baseline observation. Admit at most20 unique crop RGB arrays before
recognition. Reuse those same readings; no new crops or OCR calls are introduced
by graph admission. Its outputs append compatible candidates to an unchanged
baseline prefix. Never rerun global NMS or remove a low-scoring baseline output.

Nodes include qualified geometry/map statistics, frozen64-channel CNN crop
mean/std, an8x4 gray crop bitmap, OCR summaries, hashed Unicode1/2/3-grams and
context against the accepted baseline. Hashing uses SHA256, no held-out lexicon.
Edges encode spatial/overlap/family/text-related neighbors; they exchange
evidence and never unite crops. Equal-tensor self/GNN models share initialization,
FIT normalization, training order, updates, loss and policy candidates.

FIT GT labels the incremental maximum exact matching gain from adding each
eligible candidate to the fixed baseline. Ordinary unweighted BCE preserves the
observed candidate prevalence. These conditional labels are not a globally
optimal set target; competing positives can remain. Inference uses the qualified
sibling/family/IoU compatibility rule. Calibration includes KEEP. Selected and
raw final-threshold endpoints are both reported. Preserved observations protect
maximum correct-match count, not F1: wrong additions still increase its denominator.

## 2. Full-word contour objective and uniform-padding control

Replay hash-qualified frozen contour fields/proposals from the preceding run.
Both equal-capacity contour GCNs start at identity and receive the same original
cyclic32-point loss. The new arm additionally supervises all four exclusive bbox
extrema derived from the FULL FIT owner mask, penalizes inward clipping more
strongly, and penalizes soft-rectangle coverage of other annotated words. These
targets use the approximate v4 filled-word owners, not hidden-glyph/amodal GT.

The original crop is an identity checkpoint option for the whole branch; it is
not a per-sample GT-assisted routing choice. A separate uniform-padding grid
uses calibration word F1, the same pre-OCR budget and frozen post-filter. It is
not silently pooled with the learned crops. Changed boxes refresh exact RGB
hashes before trace export and recognition. Selected and final endpoints expose
regression even when identity wins calibration.

## 3. True U-Net-free geometry proposal/deformation decoder

Construct a torchvision ResNet50 without downloads and load EVERY qualified
encoder tensor strictly. Freeze its parameters and BatchNorm buffers. The new
decoder directly fuses five frozen encoder scales through learned lateral
projections/dilated convolutions and GroupNorm. There is no U-Net decoder or
attention skip path. It predicts text/kernel logits, normalized interior distance,
and nearest-exterior unit direction. Background direction is unsupervised.
Nearest-exterior distance is normalized independently for each FIT owner; these
targets are explicit adaptations, not an assertion of exact TextBPN targets.

Compare fresh U-Net+geometry, new proposal-only, proposal+self-only contour, and
proposal+GCN contour. All share the frozen encoder, dense objectives, FIT images,
optimizer update budget and calibration schedule. The three new proposal models
start tensor-identically; self/GCN have equal parameter counts. Graph branches
add contour supervision, so proposal-versus-graph is a full component treatment;
self-versus-GCN isolates neighbor exchange. The original warm U-Net is an
operational reference, not an equal-pretraining control.

Training contour windows are generated from current predictions using only the
RGB letterbox canvas. Owners/validity label windows afterward, never inference.
Dense losses bootstrap proposals; per-epoch valid predicted-contour counts reveal
whether the graph actually receives training supervision. No missing proposals
are secretly replaced with GT or source U-Net boxes. At inference both boxes and
refinement come from the new model; PARSeq and the same frozen budget/filter
follow. Architecture checkpoint selection never substitutes the U-Net reference.
All four selected architectures and their final endpoints are reported separately.
Full training fails explicitly if a graph never receives any valid predicted
FIT contour. A one-epoch smoke can have no bootstrapped proposals; its audit
labels this NONE_SMOKE, never a trained-graph efficacy result. A separate real-RGB
CUDA gradient probe uses archived first-FIT boxes as an explicit interface
fixture to verify CNN-to-GCN coordinate/feature gradients after two updates.
That fixture is never substituted into experimental proposal generation.

## 4. Verified regional teacher/student

Warm-start matched GT-only, frozen-teacher and EMA-teacher U-Net arms from the
qualified original checkpoint. Same labels, clean inputs, random order, RGB strip
occlusions, photometric strength, optimizer and update count. No horizontal flip:
teacher OCR verifies ordinary left-to-right FIT transcripts. Encoder weights and
running statistics remain frozen; decoder BatchNorm updates are explicit.

At declared refresh epochs, teacher predictions generate candidates, fixed
pre-OCR ranking admits at most20 unique clean crop arrays, and frozen PARSeq reads
them. FIT GT verifies exact text/IoU plus native owner purity/completeness. One
region per owner survives in proposal order. This is supervised regularization
with GT-verified teacher regions, not an unlabeled-data or amodal reconstruction
claim. Unknown/unaccepted regions add no distillation background loss.

The new objective combines instance-window soft Dice with differentiable soft
extrema, rather than matching row/column profiles or adding another CTC reader.
Teacher targets detach; clean teacher probabilities update between refreshes in
EMA, while accepted windows retain their last verification. This staleness is
recorded by refresh epoch and is a limitation. EMA versus frozen with the same
regional objective tests teacher updating. All arms retain epoch0 on ties and
report actual final models as well as selected models. Teacher/encoder audits
must pass; a complete branch with zero accepted regions fails qualification.

## Protocol, limitations and execution

Explicit config: `configs/approximate/word_graph_recovery_v1.json`.
Scientific choices: `WORD-GRAPH-RECOVERY-001` in `PAPER_AMBIGUITIES.md`.
Implementation: `localize/word_graph_recovery.py`, `models/word_graph_decoder.py`,
`scripts/run_word_graph_recovery.py`, `scripts/word_graph_geometry_stages.py`,
`scripts/word_graph_regional_stages.py`.

Reuse1030 FIT/100 calibration/125 internal evaluation/300 historically explored
TEST images, with known duplicate RGBs retained as disclosed by the source.
Source segmentation already saw calibration; local head fitting does not. Fresh
geometry decoders use no calibration gradient. No historical cohort becomes an
independent validation set by reshuffling it. Config explicitly declares this
limitation; requesting independent validation fails before model loading.
No positive generalization claim is justified without a separately qualified new
cohort or baseline training that excluded its validation population.

Metric: micro maximum one-to-one original unpadded bbox IoU>=0.5 AND exact Unicode
casefold transcription, punctuation retained. All care GT, duplicate/empty
outputs count; no ignore-overlap prediction neutralization. TEST denominator2204,
internal denominator983. Calibration selects checkpoint/policy before evaluation
GT is parsed. Report delta F1, recovered/damaged GT IDs, FP, per-image changes,
matched controls and conditional paired-image bootstrap intervals. Independent
SciPy assignment rechecks every reported endpoint. Intervals omit training
variance and historical model search; the initial experiment has one seed.

Use the existing main environment; install nothing. Actual GPU must be checked
before each launch. FP32/no AMP/no TF32, explicit batch and deterministic settings.
CPU differentiable interpolation/grid sampling is HARDWARE_DEVIATION, not a
detached gradient workaround. Record allocated/reserved peaks, model coexistence,
input shapes, training/recognition batches and explicit OOM failures. RTX3090
execution does not qualify RTX3070 hardware. Packages/weights/caches stay external.

Run from an immutable source snapshot with explicit paths:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 /explicit/main/python -u \
  -m scripts.run_word_graph_recovery --mode approximate \
  --config /snapshot/configs/approximate/word_graph_recovery_v1.json \
  --output /explicit/new/run --stage all --smoke
```

Omit `--smoke` for the full experiment. Stage choices: additions, contours,
geometry, regional, report, all. Completed stages may be resumed only under an
unchanged protocol/code/config identity; incomplete stages require investigation
and a fresh output root. Latest checkpoints retain optimizer/generator state;
automatic interrupted-epoch replay is not implemented. Primary input artifacts
are hash-verified, source runs are immutable, errors/OOM propagate with receipts.

External run root:
`/extra1/bmartelli/context-str-runs/word-graph-recovery-20261024-v1`.
Read stage receipts and `status.json`; implementation and smoke success never
establish an F1 gain. No complete-run performance result is claimed in this guide.

Initial qualification:65 focused tests passed. Smoke1 stopped on a missing
separation-mask field in the new training batches; it is preserved with its
failure receipt. The corrected shared batch builder is covered by an actual
separation-weighted loss/gradient test. Smoke2 completed all four families and
216 independently checked image/arm outcomes. Its one-epoch fresh geometry
models had zero valid predicted training contours, so further gradient
qualification and an explicit full-run no-supervision guard were added. Read
the latest external qualification receipts for the final code/smoke identity.

Final code qualification:smoke3 PASS,216 independent image/arm checks and470
hashed artifacts. All11 memory records report no OOM; peak allocated4.015GiB,
reserved5.197GiB on RTX3090. The separate real-FIT RGB gradient probe passes:
after its second update, both refiner offsets and the proposal CNN receive
nonzero finite contour gradients; frozen encoder parameters receive none.
This is a component-interface check, not a successful learned proposal result.

Before the production experiment, inactive descriptions inherited from the
parent config were moved under `parent_recipe_provenance` with an explicit
NOT-executed scope. Active numeric/runtime fields and all Python bytes match
the passing smoke recipe; `qualification/config_metadata_correction.json`
records the equivalence. The first full launch was deliberately stopped and
preserved under `full_attempt1_legacy_metadata`, with a termination receipt.
The corrected configuration names WORD-GRAPH-RECOVERY-001. Current launch state
is external `launch.json`/`full/status.json`; no completed F1 gain is claimed.
