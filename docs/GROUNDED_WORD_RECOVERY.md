# Grounded verification, new localization queries and visual byte edits

**Approximate reproduction / exploratory extension.** Decision:
GROUNDED-WORD-RECOVERY-001. User-authorized implementation and GPU evaluation of
three proposed remedies for the completed word-evidence failures. Neither the
Chakraborty paper nor Das specifies these adaptations; no novelty-priority claim.

## Evidence and controlled factors

Inspected the authoritative local `papers/2503.15639v2.pdf` sections3.2–3.5,
the attached Das KBS305(2024)112593 sections3.2–3.3 and teacher training,
existing PARSeq, protected-output selectors, attention consensus, action utility,
and frozen U-Net localization code. LevOCR (ECCV2022, ECVA paper136880319) supplies
precedent for visual insertion/deletion; this implementation is a different small
UTF8 action transducer and is not a LevOCR reproduction.

The previous GNN gives61.010922% TEST word F1. Its1,003 misses include616 without
valid boxes in the20-crop pool,243 only behind protected/conflicting hypotheses,
and144 with query-eligible geometry. Prior consensus cannot change length and
its262 TEST parents include225 without care geometry and18 with incompatible
length. Prior active admission accepts18TP/53FP, including38 with invalid care
geometry. Those completed outcomes motivate these isolated interventions.

1. A small RGB/context/text verifier predicts care geometry, spelling and JOINT
   exact-word validity. Inputs use frozen native PARSeq encoder columns, keeping
   all384channels and16columns after vertical mean pooling; this loses vertical
   detail and is an explicit assumption. Current-crop and expanded-context RGB
   are actually encoded. FIT negatives include actual errors and truncated
   correct strings; positive contrasts include FIT target strings on valid
   predicted boxes. No GT strings become held-out candidate predictions.
   Policies compare KEEP, additions and single-conflict replacements. Multiple
   destructive conflicts abstain. Compare original-confidence replacement and
   grounded addition-only. Identity is eligible on calibration.
2. A utility GNN selects at most two actions: a previously unread original
   proposal, or a60%-side corner tile. The existing frozen U-Net actually runs
   at512input/256output on a chosen tile; boxes are mapped to original RGB.
   The tile's novel child maximizes mean text probability times square-root
   component area times(1-max source-pool IoU), with explicit novelty IoU cutoff.
   An action with no novel child is recorded and spends a query. Each action
   returns at most one new PARSeq reading. FIT counterfactuals genuinely execute
   all actions; held-out OCR/localization occurs only after an action is chosen.
   Earlier-run cache hits count as cold-start reads when outside the20-crop pool.
   Pre-query features use RGB encoding and geometry, never hypothetical OCR.
   Expected-gain labels require extra matches above the entire already-read
   pool. All single-observation histories enter FIT, with32-state block-diagonal minibatches averaging per-state losses and prefixes updated by
   the actual frozen verifier, rather than an oracle first positive. Final
   verification is stateless, removing the former unseen two-observation input.
   Compare identical-capacity self-only utility and fixed ordered queries with
   the same GNN checkpoint and admission threshold.
3. Equal-capacity visual/text-only transducers predict explicit KEEP, DELETE,
   INSERT-byte and STOP with a source cursor. Substitution is deletion+insertion;
   output length varies. FIT minimum insertion/deletion paths use actual native
   OCR and care-word targets at best bboxIoU>=0.5, stable annotation-order ties.
   Invalid geometry is excluded from edit training; every held-out care word
   remains in evaluation. The64-byte limit is explicit; unsupported originals
   remain unchanged. Invalid generated UTF8 is a logged model abstention, never
   dummy OCR. Cursors and output lengths are constrained; unexpected inference
   errors propagate. All native outputs remain available through KEEP. Editors
   operate on the frozen verified output, with the same verifier selecting edits.

## Protocol and assumptions

Config: `configs/approximate/grounded_word_recovery_v1.json`. All new numeric
settings, bounds, losses and training budgets are ASSUMPTIONs. Frozen original
model states and scientific inputs are SHA256-qualified; new heads train on GPU.
STRICT configuration, source datasets, weights and prior results remain unchanged.

Reuse1030FIT/100calibration/125evaluation/300historically explored TEST. Primary
matching is micro maximum one-to-one ORIGINAL bboxIoU>=0.5 AND exact Unicode
casefold, punctuation retained. Every care GT/output, empty reading and duplicate
counts; no ignore-overlap neutralization. Native spelling labels are annotation
proxies, not physical-text or hidden-character GT. FIT edits and query labels
use care annotations; inference never uses those labels. Full outputs, scripts,
query outcomes, local probability maps and matching details support audits.

All selection uses calibration, with KEEP eligible; selected and trained-final
endpoints remain separate. Compare every endpoint against both previous GNN and
native verified output. Effects from separate factors are not added. Conditional
image bootstrap omits training/search variation. Reused validation/calibration,
one seed and two known identical TRAIN/TEST RGBs remain limitations. No independent
generalization or exact reproduction claim is supported.

## Runtime and qualification

Verified PAPER GPU: RTX3090/24GiB, single GPU FP32, no AMP/TF32, deterministic
algorithms, declared batch sizes and CUDA workspace settings. These are recorded
HARDWARE_DEVIATIONs. Frozen U-Net and PARSeq may coexist during selected queries;
record model sets and allocated/reserved VRAM at every stage. OOM/errors propagate
without alternate models. No packages or model weights are installed.

Implementation: `recognition/grounded_word_recovery.py`,
`models/grounded_word_recovery.py`, `scripts/run_grounded_word_recovery.py`,
`scripts/grounded_word_search.py`, `scripts/grounded_word_editor.py`,
`scripts/analyze_grounded_word_recovery.py`. Focused tests:
`tests/recognition/test_grounded_word_recovery.py`.

Run from an immutable source snapshot with explicit main Python, approximate
configuration and fresh output. Entry `python -m scripts.run_grounded_word_recovery
--mode approximate --config PATH --output PATH`; use `--smoke` for qualification.
The stages are prepare,extract,verifier,search,editor,report. Independent SciPy
matching checks every image/arm, query limits and original denominators. Record
encoder passes, native reads including rejected observations, tile segmentation
calls and all-policy preparation separately. These are not total latency claims.

External run root:
`/extra1/bmartelli/context-str-runs/grounded-word-recovery-20261025-v1`.
Completed run: `full/status.json` is SUCCESS/COMPLETE; final CPU mechanism analysis
is `analysis_v1/report.md` under that external root. Primary outputs remain sealed.

## Completed results and decision

All rows below are approximate reproduction, one seed, historically explored
TEST300 with2204 care words. Micro maximum one-to-one matching requires original
bboxIoU>=0.5 and exact Unicode casefold; punctuation and all output denominators
are retained. Recovered/lost words are relative to the previous calibrated GNN.

| Selected endpoint | TEST word F1 % | Delta vs GNN, points | Recovered / lost | FP |
| --- | ---: | ---: | ---: | ---: |
| Previous GNN |61.0109|0|0 /0|532|
| Grounded verification, addition-only |61.3333|+0.3224|18 /0|552|
| Grounded verification, replacements |61.5887|+0.5778|26 /2|549|
| New localization queries, self-only utility |61.8741|+0.8631|49 /2|582|
| New localization queries, graph utility |61.8056|+0.7946|47 /2|582|
| Text edit generator + RGB verifier |62.8959|+1.8850|59 /9|523|
| Visual edit generator + RGB verifier |62.8959|+1.8850|66 /16|523|

Queries and editors independently extend the same61.5887% verified prefix;
their gains are not additive. Self-only queries add only+0.2853 points
(conditional95%[-0.1028,+0.6925]),23 recovered/0 lost and33 extra FP over that
prefix. Graph queries add+0.2168 points,21 recovered/0 lost and33FP. Internal
incremental F1 is-0.0702/-0.3022 points. There is no measured graph advantage:
graph versus self-only is-0.0685 points[-0.2802,+0.1440]; matched graph versus fixed
queries is+0.0399[-0.4055,+0.4802]. Do not promote this graph policy.

The verifier improves selection of existing native readings; all26 recoveries
already existed in the20-read pool. JUNCTION->JUNCTIO and FRANCAIS->FRANC are its
two losses: the shortened boxes still pass the0.5 care-geometry target and receive
higher joint scores. That supervision does not guarantee full-word coverage.
The calibrated0.9 gate and raw0.5 gate use the same epoch10 checkpoint; raw F1
falls to60.9607% because extra false positives outweigh its additional matches.
This is a policy failure, not evidence of overtraining. Verifier delta versus
GNN has interval[+0.1477,+1.0296]; versus the historical strong HGB control its
gain is+0.5025[-0.0169,+1.0326], which does not establish superiority.

Editors keep1774 boxes/outputs and raise TP1225->1251. Over verification, text
edits recover33/lose7 words; visual edits recover40/lose14. Both add+1.3072 F1
points; conditional intervals are[+0.6189,+2.0726] and[+0.4812,+2.1607]. Internal
F1 reaches61.6921%/61.9271%. Visual versus text TEST F1 is exactly0 difference,
interval[-0.4860,+0.4690]. The text control removes images only from the edit
generator; both pipelines retain the same image-based verifier.

The mechanism limits the scientific claim: all33 accepted text repairs and38
of40 visual repairs only remove punctuation. The remaining two visual repairs,
EXIT11->EXIT and CANNERYROW->CANNERY, also delete material. The single accepted
insertion n->nO gives no match. Missing-letter reconstruction has not succeeded
on TEST. Apostrophe deletion improves Sams/Prydes annotations but damages
McCOY'S/Holly's annotations. Literal raw labels are preserved. Among7480 FIT
pairs,5372 are identities,171 differ only in case,413 only in punctuation and1524
in alphanumeric content; the raw `real_OCR_error` counter includes case-only
differences. More epochs of this deletion-dominated strategy are not supported
as a remedy. The verifier rejects13 useful visual edits while accepting14
damaging ones; ungated outputs are diagnostic only, not new TEST-selected policy.

Of953 words still missed by either editor,776 lack valid geometry among final
outputs and177 have geometry but wrong text. This is distinct from the earlier
20-read-pool count616. A separate full-native-pool diagnosis finds99 of those616
elsewhere in the full pool and517 absent there too. Future scientific work should
target full-word coverage and balanced character-level correction, with retained
KEEP controls. These results do not support replacing U-Net with this GNN.

Qualification:59 focused/regression unit tests pass. Final smoke passes270
image/arm counts and108 query traces; full independent SciPy audit passes6375
image/arm counts and2550 query traces. All7247 primary artifact hashes verify;
all saved query policies replay exactly. Peak task allocated/reserved CUDA memory
is818767872/893386752 bytes; no actual OOM. Per-stage model sets, batches, shapes,
precision, software and frozen-weight digests are saved. The search battery's
5555 segmentations/4198 newly executed OCR reads/16484 encoder passes include FIT
counterfactuals and all policies; selected TEST self-search alone uses600 actions,
564 segmentations and479 attempted native reads, including rejected observations.
The shared native-feature stage adds31284 encoder-only passes. No latency or
8GiB-hardware qualification is inferred.

Known duplicate removal leaves298 TEST images: baseline60.7738%, verifier61.3590%,
self-query61.6654%, graph-query61.5962%, both editors62.6775%. Historical exposure,
one-seed uncertainty and strict ambiguities remain. Earlier smoke1/2/3/4 failures
(missing import, namespace launch, adjacency field, free-memory preflight) and a
read-only filename lookup/optional plotting import failure are preserved; final
smoke/full success does not erase them. No packages were installed. Machine-readable
qualification: `provenance/grounded_word_recovery_v1_qualification.json`.
