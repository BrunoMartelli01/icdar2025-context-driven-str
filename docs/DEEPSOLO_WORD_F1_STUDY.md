# DeepSolo symbol-free word F1 study — 2026-10-25

**Approximate reproduction.** This user-authorized study measures ordinary
DeepSolo first, diagnoses its errors, then selects confidence, resolution and
cross-scale fusion policies using TRAIN125 only. TEST300 has already been
explored in earlier work: paired TEST results are descriptive, not a new blind
holdout or proof of a global maximum. No dependency, checkpoint or STRICT
configuration is changed.

## Evidence and comparison boundaries

Inspected the authoritative `papers/2503.15639v2.pdf`, sections 3.5/4.1/4.2,
Algorithm 2 and Table 5; the qualified M6 and PAPER host records in
`docs/DEEPSOLO_RUNTIME.md`; the DeepSolo entries in
`docs/STRICT_REPRODUCTION_STATUS.md`; the frozen upstream inference/CTC loss,
decoding and evaluation implementations at commit
`dbadae995035246bad3376c7a44c015c69e9b313`.

The [official model zoo](https://github.com/ViTAE-Transformer/DeepSolo/blob/dbadae995035246bad3376c7a44c015c69e9b313/DeepSolo/README.md)
reports Total-Text E2E None F1 79.7% for R50 without TextOCR, 82.5% for R50
with TextOCR and 83.6% for ViTAEv2-S with TextOCR. These use the official
polygon/word-spotting protocol. Our paper's DeepSolo checkpoint and recognition
accuracy definition remain unresolved; see `docs/PAPER_TABLE5_SOURCE_AUDIT.md`.
Numerical proximity under a different metric does not reproduce these scores.
The incomplete R50+TextOCR checkpoint is excluded.

Two explicit local metrics are reported separately. Both normalize OCR and GT
using Unicode casefold followed by `isalnum`, preserving accents and numbers,
with no NFKC, accent removal, lexicon or spelling oracle. Each detection remains
one word; repeated words count separately. All original care denominators remain
983 TRAIN and 2204 TEST. Empty normalized strings never match; symbol-only
annotations and outputs remain errors. Ignore masks and invalid-envelope FP
treatment are preserved.

- **Word F1:** maximum exact normalized word-multiset matching within each image,
  without position constraints.
- **Localized word F1:** normalized exact text plus original-frame bbox
  IoU>=0.5, maximum one-to-one matching. This is still distinct from official
  polygon IoU>0.5 and word-spotting care rules.

Counts are summed over every image: `F1=2TP/(2TP+FP+FN)`, with precision
`TP/(TP+FP)` and recall `TP/(TP+FN)`. Empty images remain included; undefined
ratios stay null. GT, validity and transcripts enter evaluation only. The fusion
decision uses instance/character confidence and predicted geometry, never GT
or ignore flags.

## Ordinary baseline and first diagnostic

The complete October 19 ViTAE report is SUCCESS, not merely a launched job.
Its native FP32 outputs use shortest1024/max1892 and divisor32 padding,
threshold0.4, batch1 and no lexicon. The original R50 native baseline uses
shortest1000/max1892. Existing punctuation-retaining word F1 is 75.736% R50
and 80.357% ViTAE; those historical results are preserved.

| TEST300 endpoint | TP | FP | FN | Symbol-free word F1 % |
| --- | ---: | ---: | ---: | ---: |
| Ordinary R50 | 1622 | 386 | 582 | 77.018 |
| Ordinary ViTAE+TextOCR | 1785 | 379 | 419 | 81.731 |
| ViTAE, TRAIN-selected character confidence>=0.9 | 1732 | 268 | 472 | 82.398 |

Ordinary ViTAE localized word F1 is 81.364% (1777/387/427). Geometry-only
maximum matching covers1949 of2204 care words. Its255-word localization gap
cannot be repaired by changing the transcript alone; another172 matches separate
this geometric upper bound from correct localized transcripts. Descriptive error
examples include `nauGHTY -> noughty`, `1996 -> 1995`, merged `THIRSTY ->
thirstybear`, and missing first letters. An arbitrary geometry-only assignment
can pair repeated words differently, so its list of wrong transcripts is not
itself an exact FN partition.

The first TRAIN-only grid compared greedy and lexicon-free CTC prefix beam10,
instance thresholds, emitted-character geometric-mean confidence and same-text
spatial deduplication. It selected greedy, native threshold0.4 and character
confidence>=0.9, with no deduplication. Beam decoding was not selected. It
removes111 FP but loses53 TP on TEST: its modest F1 gain is precision-driven.

The initial diagnostic excluded one symbol-only TRAIN target (982 words). Its
sources and results remain archived under `cached-analysis-v1`; it is superseded
by `cached-analysis-final`, which preserves all983 and selects the same policy.
TEST has no symbol-only target, and its scores are unchanged. The final
`configs/approximate/deepsolo_word_f1_v2.json` records the corrected boundary;
the exploratory v1 configuration is retained for provenance.

## Additional inference and final selection

The corrected resolution study exports all425 images at both native and1.5x
resize (shortest1536/max2838), with detector floor0.05. It strictly validates
the existing ViTAE checkpoint with SHA256
`54d3a454ecc704ef3168ebf9f23fc8390a3b2637d0ec83d5650e8df45e2aca4a`.
One checkpoint stays resident in the existing legacy Python3.8 process; no model
training or package installation is performed.

`scripts/evaluate_deepsolo_word_f1_fusion.py` fits independent policies for the
two metrics on TRAIN. Candidates include the two individual resolutions and
confidence-ranked fusion. Fusion suppresses overlapping detections only across
views, preserving spatially repeated words. The complete finite grid, fixed
tie-breaking order, raw responses, image/checkpoint/source hashes, resolved
configuration and per-image errors are retained. The selections are sealed
before any corresponding TEST metric is computed. Native floor0.05 outputs
filtered back to0.4 must exactly replay the old baseline under both metrics.

Final results and uncertainty are recorded in the completion section below.

## Corrected results and deployable profiles

The corrected462-policy TRAIN grid and complete TEST300 evaluation passed exact
native-baseline replay. All983 TRAIN and2204 TEST care words remain counted.

| Profile | TRAIN word F1 % | TEST word F1 % | TEST localized word F1 % | TEST word TP/FP/FN |
| --- | ---: | ---: | ---: | --- |
| Ordinary ViTAE native | 78.624 | 81.731 | 81.364 | 1785/379/419 |
| Best TRAIN native-only word profile | 80.196 | 82.654 | 82.133 | 1744/272/460 |
| Best TRAIN word profile across all candidates | 80.297 | 82.452 | 81.333 | 1769/318/435 |

The global TRAIN word winner is **fusion**, instance threshold0.35, character
confidence>=0.95, cross-view bbox suppression IoU>0.3 and ranking by instance
confidence. Its TEST word gain is0.721 percentage points, conditional paired95%
interval[-0.170,+1.634] points, including zero. Its localized TEST change is
-0.031 points. It must remain the reported primary global TRAIN selection even
though native happens to score higher on TEST.

The separate native-only TRAIN winner uses instance threshold0.35 and character
confidence>=0.925, greedy decoding, ordinary1024/1892 resize and no suppression.
It is also the global TRAIN winner for the localized objective. TEST word F1
improves0.923 points, with precision86.508% and recall79.129%; the conditional
paired95% interval is[+0.126,+1.715] points. It removes107 net FP while losing41
net TP. Localized F1 improves0.768 points; its interval[-0.00046,+1.575] includes
zero. These fixed-policy intervals do not remove historical TEST exploration.
The native profile is provided as an explicit native-only operating choice;
it is not substituted into the global-word selection using TEST results.

The native profile recovers33 previously unmatched correct words (including
`coffee` and `kuala`) and loses74; fusion recovers61 and loses77. Thus these
gains mainly improve precision, with a real recall cost. The diagnostic preserves
all word repetitions and reports changes per image in `error_change_analysis.json`.
All normalized TEST targets fit the36 printing-character alphabet; one target,
`cnaguangdongpilotfreetradezone`, exceeds the25 CTC point positions and cannot
be emitted as one word by this head. No GT splitting, label correction or
character substitution is applied to hide that failure.

Both profiles are usable on new images through
`scripts/infer_deepsolo_word_f1.py`, without GT or validity annotations:

```bash
.venvs/context-str-deepsolo-host/bin/python -u -m scripts.infer_deepsolo_word_f1 \
  --mode approximate \
  --config configs/approximate/deepsolo_word_f1_selected_native_word_v1.json \
  --runtime RUNTIME --source-code-root UPSTREAM \
  --image IMAGE --output-root FRESH_INFERENCE_ROOT
```

For the global TRAIN word winner, use
`configs/approximate/deepsolo_word_f1_selected_global_word_v1.json` instead.
Both configurations pin the checkpoint, upstream YAML/base-YAML hashes, source
evaluation and selector hashes. The main pipeline and STRICT defaults remain
unchanged. The global profile requires both resolutions and the larger memory
budget measured below.

`scripts/report_deepsolo_word_f1_profiles.py` exports these profiles from TRAIN
scores only. `scripts/audit_deepsolo_word_f1.py` passed850 independent Hungarian
image/metric checks and verified1704 source hashes.
`scripts/audit_deepsolo_word_f1_profiles.py` passed1700 further independent
profile checks and six real GT-free GPU image/profile replays. Text, integer
boxes, order and selections match exactly; maximum confidence difference is0.
Both GPU profiles strictly loaded the same checkpoint without missing,
unexpected or incorrectly shaped keys.

All73 focused unit/regression tests passed (`tests-final.txt` outside Git),
including OOM/failure propagation, normalization denominators, threshold edges,
CTC paths, scale switching and legacy Python3.8 syntax. Task source/evidence
hashes and whitespace checks passed. Experiment-library and source-only audit
navigation were rebuilt and verified.

Completed outputs: `fusion-final-v2/results.json`, `selection.json`,
`processed_predictions.json`, `independent_audit.json`, `deployment_profiles.json`,
`deployment_independent_audit.json`, and both `smoke-*-final/` directories under
the external study root. Portable completion evidence is
`provenance/deepsolo_word_f1_20261025_completion.json`.

## Reproduce and inspect

Code: `recognition/deepsolo_word_f1.py`,
`scripts/analyze_deepsolo_word_f1.py`, `scripts/run_deepsolo_word_f1_study.py`,
`scripts/evaluate_deepsolo_word_f1_fusion.py`.
Configurations: `configs/approximate/deepsolo_word_f1_v2.json`,
`configs/approximate/deepsolo_word_f1_resolution_v2.json`,
`configs/approximate/deepsolo_word_f1_fusion_v2.json`.
Tests: `tests/recognition/test_deepsolo_word_f1.py`,
`tests/scripts/test_deepsolo_word_f1_study.py`,
`tests/scripts/test_deepsolo_word_f1_fusion.py`,
`tests/scripts/test_infer_deepsolo_word_f1.py`,
`tests/scripts/test_audit_deepsolo_word_f1.py`.

Outputs are outside Git at
`/extra1/bmartelli/context-str-artifacts/deepsolo-word-f1-20261025/`.
Use the existing legacy Python for export and the existing main Python for CPU
evaluation, passing all paths explicitly:

```bash
.venvs/context-str-deepsolo-host/bin/python -u -m scripts.run_deepsolo_word_f1_study \
  --mode approximate --stage export \
  --config configs/approximate/deepsolo_word_f1_resolution_v2.json \
  --source-plan SOURCE_PLAN --runtime RUNTIME --source-code-root UPSTREAM \
  --data-root DATA_ROOT --output-root EXPORT_ROOT

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venvs/context-str-main-py311/bin/python -u \
  -m scripts.evaluate_deepsolo_word_f1_fusion --mode approximate \
  --config configs/approximate/deepsolo_word_f1_fusion_v2.json \
  --source-root EXPORT_ROOT --source-plan SOURCE_PLAN \
  --data-root DATA_ROOT --output-root FRESH_RESULT_ROOT
```

The frozen source plan is the October19 full run's
`resolved_configuration.json`, SHA256
`d21b32113271a9de2a9ef2d40c2d2b8b9a3e8f6a20331056d2c942522c244abe`.
See `docs/HOST_OPERATIONS.md` for host paths. This guide's operational paths
do not become model defaults. Evaluator/config/source changes require a new
output directory; changed evidence is never overwritten.

## Rejected first resolution experiment

The complete `resolution-v1` export and `fusion-final-v1` analysis exposed an
engineering error: changing `INPUT.MIN_SIZE_TEST/MAX_SIZE_TEST` in the wrapper
does not update the ViTAE detector's cached `min_size_test/max_size_test`.
The pinned `detector_postprocess` reads those cached values to undo resizing.
Consequently the first large view's coordinates were approximately1.5x too
large; ignore filtering, localized scores and cross-view fusion were invalid.
The zero localized TRAIN F1 in that view triggered inspection before completion.
The native view's exact baseline replay passed and remains valid.

The repair synchronizes both cached inverse-resize attributes with the actual
augmentation for each view, records the pair in every shard and repeats all850
inferences in `resolution-v2`. Evaluation rejects shards without the matching
postprocess pair. The regression explicitly checks that a point restores to600
original pixels rather than900 when switching to the larger input, then checks
switching back to native. The original artifacts, source snapshot and
`invalid_resolution_v1.json` remain outside Git; they are superseded evidence,
not a negative scientific result about larger inputs.

Current configurations are
`configs/approximate/deepsolo_word_f1_resolution_v2.json` and
`configs/approximate/deepsolo_word_f1_fusion_v2.json`. The v1 configurations and
invalid artifacts are retained for provenance. Final results below refer only
to the corrected run.

## Completed GPU export and memory

All850 corrected inferences completed (125 TRAIN+300 TEST at each resolution),
with strict checkpoint compatibility and no OOM or alternate-model/precision
fallback. Actual hardware was RTX3090, FP32 batch1, one ViTAE model. The qualified
cuDNN flags were retained and all four RNG families explicitly seeded.

| View | Images | Peak allocated bytes | Peak reserved bytes | Allocated-peak input CHW |
| --- | ---: | ---: | ---: | --- |
| Native | 425 | 2117437952 | 4099932160 | 3x1024x1920 |
| 1.5x | 425 | 9690508288 | 16651386880 | 3x1536x2848 |

Native and larger allocated peaks are about1.97GiB and9.03GiB. Reserved peaks
are about3.82GiB and15.51GiB. These are measured allocator peaks, not predictions
of RTX3070 behavior. Each raw shard retains its actual input and both peaks;
`resolution-v2/memory_summary.json` summarizes all850. Independent CUDA jobs
were observed after the export and preserved; the resource observation is saved.
Timing is not presented as an isolated-GPU benchmark.

## Scientific classifications and limitations

- `PAPER_SPECIFIED`: Algorithm2 calls DeepSolo on the full image.
- `REPOSITORY_SPECIFIED`: frozen native model/config/CTC interpretation,
  checkpoint compatibility checks and qualified preprocessing; existing local
  geometry/care handling and all-denominator alphanumeric comparison.
- `ASSUMPTION`: downloaded-candidate identity without an author-published hash;
  symbol-free local metrics, threshold grid, CTC beam marginalization,1.5x
  resolution, cross-scale suppression and TRAIN-only finite policy selection.
- `HARDWARE_DEVIATION`: existing isolated host process on the actual RTX3090,
  FP32 batch1, one resident model across samples, explicit seeds and qualified
  cuDNN enabled/benchmarkFalse/deterministicFalse. The large input exceeds8GiB
  VRAM on some images; it is not an RTX3070 qualification. Timing is not a
  paper latency/FLOP benchmark.
- `UNRESOLVED`: strict checkpoint/metric/lexicon equivalence, complete
  R50+TextOCR bytes, official polygon evaluation here, independent validation
  after historical TEST exploration, known data duplicates and global optimality.

The paired image-bootstrap interval conditions on fitted policies and these
fixed images. It does not measure retraining variance or remove selection and
historical exploration bias. No strict or Table5 reproduction claim is made.
See `DEEPSOLO-WORD-F1-20261025` in `docs/PAPER_AMBIGUITIES.md`.
