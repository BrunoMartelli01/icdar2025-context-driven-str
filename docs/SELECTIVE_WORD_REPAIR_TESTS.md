# Selective word repair and false-positive control

**Approximate reproduction / exploratory extension.** The user authorized four
controlled tests with a frozen U-Net and PARSeq: incremental admission of already
read candidates, deferred rejection of uncertain text, boundary-only language
correction, and selective local RGB rereading. This is not the strict paper path,
official polygon spotting, or Table 5. Decision: SELECTIVE-WORD-REPAIR-TESTS-001.

Implementation: `recognition/selective_word_repair.py`,
`scripts/run_selective_word_repair_tests.py`, and
`scripts/audit_selective_word_repair_tests.py`. Post-hoc analysis:
`scripts/analyze_selective_word_repair_tests.py`. Focused tests:
`tests/recognition/test_selective_word_repair.py`.

Before implementation, the authoritative local paper `papers/2503.15639v2.pdf`,
sections 3.3–3.5, existing localization/recognition code, trained pointwise filter,
ByT5 reconstruction, and completed graph/addition experiments were inspected.
The paper does not specify these interventions. Its component-area localization,
cap10 and full context/DeepSolo pipeline are not replaced by this budget20 study.

## Controlled experiments

1. Freeze the qualified budget20 pipeline and its accepted prefix. Train an
   unweighted HGB on real FIT candidates and incremental exact-word matching
   gains. Conditional features describe overlap and duplicated readings relative
   to accepted outputs. A positive label must recover a word without damage;
   F1-only deletion is not a positive. Greedy additions preserve every accepted
   output. Replay the separately completed equal-capacity MLP/GNN additions as
   explicitly identified existing-run comparisons, never as new training here.
2. Fit separate care-annotation-overlap and exact-read predictors. The first
   is an annotation proxy, not physical text-presence supervision. Correct reads
   are kept, uncertain text is deferred, and low-evidence candidates are rejected.
   Calibration chooses the deferred policy with at most two damaged correct
   words; identity is eligible. A matched hard-rejection control is reported.
   Three-fold FIT out-of-fold scores, using image-level folds, route at most two
   uncertain parents per image for training the rereading stage.
3. Extract the genuine pinned pretrained ByT5 encoder (revision
   `68377bdc18a2ffec8a0533fef03b1c513a4dd49d`) after a checked complete model
   load. Freeze every encoder tensor and train a gap MLP. UTF-8 byte positions
   are mapped to Unicode character boundaries; decoding copies every original
   character and may only change whitespace. This is a new frozen-encoder head,
   not full decoder fine-tuning. Compare the original-block baseline, historical
   full-fine-tuned ByT5, its boundary-only control, and raw/calibrated gap heads.
   Supervision includes identities, false spaces, actual OCR with exactly equal
   compact character sequences, and joined GT words that are spatial neighbors.
   No fallback to arbitrary positive-IoU target words or global sorted neighbors.
   The gap head can split any copied Unicode sequence, including punctuation
   and numbers; the historical ByT5 policy has an alphabetic eligibility guard.
   This is a whole-method comparison, not an isolated encoder-freezing ablation.
4. Reread at most two uncertain parents using original RGB expansion, a nearby
   unselected union, and a bounded predicted-probability valley split. No learned
   RGB warping. The unchanged qualified PARSeq reads every required alternative.
   FIT utility/harm predictors distinguish actual recovery from damaging changes;
   calibration retains KEEP and a two-word damage budget. Compare selected/raw
   learned rereading with confidence-only replacement and the unchanged triage
   outputs. Report all generation/OCR costs, including rejected alternatives.

All new numeric choices and source classifications are explicit in
`configs/approximate/selective_word_repair_tests_v1.json`. No packages or weights
are installed. Existing input files are checksum verified, including exact-byte
restorations for three previously changed TRAIN images; user data is unchanged.

## Data, metrics and qualification

Reuse 1030 FIT / 100 calibration / 125 internal evaluation / 300 TEST images.
Selection uses calibration only. Inherited segmentation calibration exposure,
historical repeated TEST exploration, one seed and two known TRAIN/TEST identical
RGBs remain limitations. Primary TEST retains all 2204 care words; report a
separate 298-image sensitivity without the two known duplicates.

The budget20 track uses maximum one-to-one original bbox IoU>=0.5 AND exact
Unicode casefold, punctuation retained. Lexical multiset results are additional.
All predictions, duplicates and empty readings count; no ignore-based prediction
neutralization. The original-block boundary track uses per-image casefold
whitespace-token multiset matching without IoU; source block populations differ,
so gains must not be added across tracks. Boundary repair alone does not produce
new word boxes and is not claimed as spatial recovery.

Actual GPU execution must be qualified per launch. CUDA FP32, no AMP/TF32,
deterministic algorithms, explicit batches and sequential ByT5/PARSeq loading are
recorded hardware choices. Preserve allocated/reserved peaks and OOM failures.
RTX3090 qualification does not qualify 8GiB hardware. Encoder states are cached
on CPU; HGB models fit on CPU. Cold-start OCR cost includes cache hits outside
the already admitted source pool, not only actual GPU cache misses.

Run from an immutable source snapshot:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 /explicit/main/python -u \
  -m scripts.run_selective_word_repair_tests --mode approximate \
  --config /snapshot/configs/approximate/selective_word_repair_tests_v1.json \
  --output-root /explicit/new/output --stage all --smoke
```

Omit `--smoke` for the full experiment. Stages are prepare, selectors, boundaries,
reread and report. Resume completed prerequisites only with unchanged config and
critical code. Failed/incomplete stages require investigation and a fresh output
root. Do not infer efficacy from a smoke, implementation, or training loss.

External root: `/extra1/bmartelli/context-str-runs/selective-word-repair-tests-20261024-v1`.
Completion, per-image outputs, conditional bootstrap intervals, failure attribution,
memory and real OCR costs must be read from the completed machine-readable report.

## Completed full results, 2026-10-24

All four tests completed on 1030 FIT, 100 calibration, 125 evaluation and 300 TEST
images. The immutable code/config snapshot is under the external root. No U-Net,
PARSeq or pretrained ByT5 encoder tensor updates occurred; the trained gap head
has 94529 parameters and 828 optimizer updates. Full encoder parameter digests
before/after are identical. Complete ByT5 loading reports zero missing/unexpected/
mismatched keys; PARSeq uses its unchanged strict-load qualified adapter.

Budget20 spatial track, all 300 TEST /2204 care words:

| Arm | Correct | Outputs | FP | FN | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original budget20 selection |1255|2465|1210|949|53.7588%|
| Previously qualified pointwise filter |1187|1707|520|1017|60.7006%|
| New conditional HGB additions |1226|1810|584|978|61.0862%|
| Deferred rejection |1250|2001|751|954|59.4530%|
| Hard-rejection control |1160|1513|353|1044|62.4159%|
| Deferred rejection + selected rereading |1258|2006|748|946|59.7625%|
| Deferred rejection + confidence-only rereading |1251|2028|777|953|59.1210%|
| Prior MLP additions, existing-run replay |1209|1793|584|995|60.4954%|
| Prior GNN additions, existing-run replay |1201|1733|532|1003|61.0109%|

The new addition model recovers39 relative to the filtered baseline, damaging0.
Of these,34 restore unfiltered successes and only5 are new relative to the original
unfiltered pipeline. Its103 extra outputs include64 false positives. F1 delta
is+0.386points, conditional paired95% interval[-0.098,+0.918]; the observed gain
does not establish a reliable F1 improvement. Internal125 gives+19correct,
0damaged and+0.593points, interval[-0.354,+1.622]. HGB versus prior GNN gives
only+0.075points, interval[-0.313,+0.500], despite more recovered words.
The HGB and prior GNN also differ in inputs/training; this descriptive comparison
does not isolate classifier architecture under identical features.

Deferred rejection removes459FP (37.93%) while losing5correct (99.60% retention
of original successes). Compared with the older filter it retains63more correct
words but231moreFP, so F1 is lower. The hard control has the highest displayed
F1 but loses95correct TEST words and32on calibration: it fails the declared
two-damage policy criterion and is not promoted as a safe policy.

Rereading recovers8 and damages0 relative to deferred rejection; F1 delta
+0.309points, conditional interval[+0.053,+0.616]. Internal125 recovers1,
damages0, delta+0.079points[-0.072,+0.332]. TEST accepts17variants (11expansions,
5splits,1union); six improve exact matching. Examples include `#SHOES -> SHOES`,
`House- -> House`, `OUL -> Out`, and `EXIT11 -> EXIT / 11`. No long missing-word
completion success is established by these examples. Confidence-only replacement
is negative on TEST despite a slightly positive internal endpoint.

Text-boundary track, original M/K observations, lexical matching only:

| Arm | Correct | Outputs | Recovered / damaged | F1 |
| --- | ---: | ---: | ---: | ---: |
| Original readings |1322|2824|0/0|52.5855%|
| Previous fully fine-tuned ByT5 |1365|2948|78/35|52.9891%|
| Previous ByT5 boundary-only control |1366|2948|73/29|53.0279%|
| New frozen-encoder gap head, raw0.5 |1334|3103|76/64|50.2732%|
| New calibrated gap policy, KEEP |1322|2824|0/0|52.5855%|

The gap head uses12000clean pairs and70260gap positions, with2422positive gaps
and54conflicting source segmentations. FIT loss drops0.2367to0.0523, but raw TEST
F1 decreases2.312points, interval[-3.344,-1.190]. Character copying does not
protect names/compounds against wrong splits (`FORMAGGI -> FOR MAGGI`,
`nauGHTY -> nau GHTY`). Calibration correctly selects KEEP. This is negative
evidence for the frozen-encoder/head implementation, not all boundary models
or a fine-tuned multimodal corrector. The historical ByT5 is not overwritten.

## Failure limits, costs and qualification

The original admitted candidate readings support a loose1304exact spatial
matches; adding every valid generated reread supports1328. Even hypothetical
perfect FP removal bounds their spatial F1 at74.34% and75.20%, respectively.
These GT-assisted bounds ignore compatibility and apply to saved readings,
not every future recognizer on these boxes. Independent SciPy matching also
finds1587/1605geometry matches, so the geometry-to-reading deficit includes
incomplete/contaminated crops and is not a pure recognizer-error decomposition.
The present saved candidate set cannot support80% spatial F1 by selection alone.

All-stage selective rereading makes1467new GPU crop inferences and reuses source
readings. Deployment cost is larger than cache misses: TEST requires685additional
cold-start crop reads on top of3418admitted source crops, a1.2004x OCR count.
Those costs include evaluated alternatives that are later rejected. No complete
pipeline speedup is claimed. Measured actual RTX3090 peaks in sequential FP32:
encoder1020611584/1142947840 allocated/reserved bytes; gap training491131392/
507510784; PARSeq177873408/188743680. No actual OOM or silent model fallback.

60focused/regression tests pass. Initial unit provenance/dedup fixture failure
and smoke1's incorrect nested-manifest expectation are preserved and corrected;
smoke2/3 both pass. Final smoke3 independent audit passes172image/arm cases.
Full audit passes7675image/arm cases,4073final RGB crop hashes,1555routing budgets,
9580Unicode-copy checks and1591input hashes. The controlled synthetic OOM test
verifies error propagation, not an observed runtime OOM.

Excluding the two known identical TRAIN/TEST RGBs preserves+39additions and+8
reread successes; selected F1 values are60.8543% and59.5307%, with their respective
filtered/deferred controls60.4591% and59.2184%. These are sensitivity checks on
historically explored data, not independent validation. No strict/default path,
dependency environment, source dataset, historical checkpoint or result is changed.

Final result SHA256:
`662d2d0ed816b0f4a96a3d81cfa415deb7bf1b0fdadf830c7931498e071a4432`.
Read `full/comparison.md`, `full/results.json`, `full/independent_audit.json` and
`full/posthoc_v3/analysis.md` / `diagnostics.json` under the external root.
