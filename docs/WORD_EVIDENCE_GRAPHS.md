# Word evidence graphs: four controlled factors

**Approximate reproduction / exploratory extension.** Decision:
WORD-EVIDENCE-GRAPHS-001. Frozen source U-Net, native PARSeq, STRICT, historical
results and dataset bytes are unchanged. No novelty priority claim is made.

Evidence inspected: authoritative `papers/2503.15639v2.pdf` sections3.2–3.5;
Das et al., KBS305(2024)112593 sections3.2–3.5; completed word graph recovery,
selective word repair, beam/LLM correction and OCR-guided crop experiments.
Neither paper establishes these adaptations. Negative contour/decoder/EMA
variants are not continued. The prior graph's60.7006 ->61.0109% TEST word F1
is the positive reference. Previous rereading's8 recovered/0 damaged was against
its deferred control; its59.7625% endpoint did not beat the prior graph.

## Factors and matched controls

1. Typed relations separate support from sibling/IoU conflict. Compare identical
   model tensors receiving collapsed adjacency in both channels. Empty region
   inputs isolate relation typing. No crops are merged or modified.
2. Visual ownership uses predicted text tiles and their foreground-pixel
   incidence with hypotheses. Nodes include text/kernel statistics and soft
   accepted-set coverage, never GT ownership. Compare identical region features
   and model tensors with all-to-all incidence. Tiles are not GT characters.
3. Spatial consensus uses frozen PARSeq last-decoder cross-attention on existing
   native strings under causal teacher forcing. Read-only hooks change no model
   outputs. ORIGINAL native character probabilities stay separate from alignment.
   Attention centroids are an imperfect spatial proxy. Overlapping readings
   donate more confident monotonically aligned characters, including from partial
   donor crops. The target template length stays fixed: substitutions only, no
   hidden glyphs, character GT, dictionary or language completion. Compare the
   strongest whole reading in the same neighborhood. Equal-capacity selectors
   learn FIT incremental exact-word gains. Extra decoder computation is counted.
4. A utility graph acquires at most two directional/neighbor-isolated RGB views.
   No hypothetical action OCR is an inference feature. FIT genuinely recognizes
   counterfactual actions; oracle choices construct TRAIN states only. Compare
   equal-capacity self-only utility and fixed queries with the SAME trained GNN
   admission. Graph states update with acquired readings. Localization boxes
   remain original; recognition pixels change. Rejected reads count. The two-read
   limit is a cap; realized counts can differ when acceptance prevents a later
   compatible query. Report counts, not an assumed equal realized cost.

The typed/ownership arms preserve every filtered-source output. Consensus/active
arms preserve every prior-GNN output. Such preservation protects maximum TP,
not F1. No global NMS/deletion is rerun. Raw final models and selected KEEP
endpoints are separate; all arms compare against BOTH references.

## Protocol and execution

Explicit parameters/classifications: `configs/approximate/word_evidence_graphs_v1.json`.
All new factors and hyperparameters are ASSUMPTIONs; actual RTX3090 single-GPU
FP32, no AMP/TF32, deterministic execution, batch/caching/sequential loading are
HARDWARE_DEVIATIONs. No installations/downloads. Record allocated/reserved VRAM
and propagate failures/OOM, with no substitute outputs.

Reuse1030 FIT/100 calibration/125 evaluation/300 historically explored TEST images.
Source calibration exposure, one seed and the two disclosed duplicate TEST RGBs
remain limitations. Primary TEST2204 care words; evaluation983. No independent
validation or strict reproduction claim. Micro maximum one-to-one original bbox
IoU>=.5 AND exact Unicode casefold retains punctuation; every care GT and output
counts, without ignore-overlap neutralization. Conditional image bootstrap omits
training/selection variability. No TEST-based policy selection.

Run an immutable source snapshot using the existing main environment, explicit
config/output and `CUBLAS_WORKSPACE_CONFIG=:4096:8`, offline model flags. Entry:
`python -m scripts.run_word_evidence_graphs --mode approximate --config PATH --output PATH`.
Add `--smoke` for qualification. Stages: prepare, graphs, alignment, consensus,
active, report. Preserve failures, per-image outputs, hashes, histories,
checkpoints, character donors and query traces. Independent SciPy assignment
audits all reported image/arm counts. No automatic promotion or further training.

## Completed full results

The full immutable run completed on the declared1030/100/125/300 populations.
External root: `/extra1/bmartelli/context-str-runs/word-evidence-graphs-20261024-v1`.
Authoritative code: `source_snapshot_v2`; results: `full/results.json` and
`full/comparison.md`; detailed mechanisms and real RGB examples:
`analysis_v1/report.md`, `analysis_v1/visual_cases.png`. Qualification:
`provenance/word_evidence_graphs_v1_qualification.json`.

| Selected endpoint | TEST word F1 % | Delta vs prior GNN, points | Recovered/lost vs prior GNN |
| --- | ---: | ---: | ---: |
| Previous qualified GNN |61.0109|0|0/0|
| Typed support/conflict |60.8541|-0.1568|2/6|
| Visual ownership |60.8318|-0.1791|1/10|
| Whole-reading control, KEEP |61.0109|0|0/0|
| Character consensus, KEEP |61.0109|0|0/0|
| Active observations, self-only |60.9371|-0.0738|15/0|
| Active observations, GNN |60.8283|-0.1826|18/0|
| Fixed queries, KEEP |61.0109|0|0/0|

Typed versus collapsed is only+0.0464points; ownership versus global incidence
is+0.0866. Both conditional intervals include zero. No new selected method beats
the previous GNN. Character consensus creates85 novel TEST strings, none yielding
a correct word absent from the original pool. Its raw61.0462% endpoint equals
the whole-reading control: both re-admit existingJack, with no donated characters.

Active GNN adds18TP/53FP, below the marginal useful share needed to increase F1.
Of its53FP,38 lack valid care-word geometry and15 have valid geometry but wrong
text. Its queries contain44 distinct potential additional matches versus6 for
fixed queries at448/446 queries. This is an upper diagnostic over acquired
readings, ignoring admission/compatibility; it is not achieved F1. Seven actual
recoveries need new readings. Retain this acquisition evidence, without promoting
the negative admission pipeline or extending negative variants.

61 focused tests pass; final smoke378 and full8925 independent checks pass.
Fresh SHA256 integrity covers178 primary artifacts. Actual RTX3090 peaks across
all stages:0.3130GiB allocated/0.4473GiB reserved, no actual OOM. Frozen PARSeq
state checks pass. Preserved qualification failures: smoke4 missing configuration
field, corrected; initial helper used system Python without rapidfuzz, rerun in
the existing main environment. Earlier launch intentionally stopped for explicit
parameter binding;2536 real RGB action comparisons establish identical pixels.
The original61.0109% method remains the recommendation; no model is promoted.
